"""Compare hard negative voting followed by conditional Gain against direct Gain on Madelon."""

import argparse
import hashlib
import multiprocessing
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaincc, betaln
from xgboost import DMatrix

from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test.replay_online_relative_voting import AUC_TIE, PATHS, RISK, _auc_columns, _calibrate
from experiments.test.thesis_madelon_alpha_data import load_arm, load_heldout
from experiments.thesis_sampling_worker import prediction_scores, subset_validation_scores, validation_halves
from minppi_mofs.experiment_protocol import AuditLog, JobSpec, atomic_write_json, canonical_hash
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.utility import compute_utility


LIMIT_S = 900
SEED = 42
MARGIN = .005
METHODS = ("hard_negative_then_gain", "direct_gain")


def permutations(seed, width):
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(width) for _ in range(PATHS)])


def negative_vote(model, valid, labels):
    parts = model.get_booster().predict(DMatrix(valid, nthread=1), pred_contribs=True,
                                       approx_contribs=False, strict_shape=True)
    phi = parts[:, 0, :-1].astype(float)
    positive = np.asarray(labels, dtype=bool)
    width = phi.shape[1]
    positions = np.arange(width)
    tiers = np.where(positions < width // 3, 0, np.where(positions < 2 * width // 3, 1, 2))
    reference, cal_min, eligible = _calibrate(phi, positive, permutations(SEED + 100003, width), tiers)
    full_auc = _auc_columns(phi.sum(axis=1, keepdims=True), positive)[0]
    neg, pos = np.indices((PATHS + 1, PATHS + 1))
    evidence = (np.exp((neg + pos + 1) * np.log(2.) + betaln(neg + 1, pos + 1))
                * betaincc(neg + 1, pos + 1, .5))
    up = np.zeros((3, width), dtype=int)
    down = np.zeros_like(up)
    lock = np.zeros(width, dtype=int)
    counted_votes = ranked_prefixes = 0
    for step, order in enumerate(permutations(SEED, width), 1):
        active = np.flatnonzero(eligible[order] & (lock[order] == 0))
        if not len(active):
            break
        edges = np.union1d(active, active + 1)
        interior = edges[(edges > 0) & (edges < width)]
        aucs = np.empty(width + 1)
        aucs[0], aucs[-1] = .5, full_auc
        if len(interior):
            # Keep absorbed features as coalition context for the remaining features.
            margins = np.cumsum(phi[:, order], axis=1)
            aucs[interior] = _auc_columns(margins[:, interior - 1], positive)
        residual = aucs[active + 1] - aucs[active] - reference[active]
        up[tiers[active], order[active]] += residual > AUC_TIE
        down[tiers[active], order[active]] += residual < -AUC_TIE
        counted_votes += len(active)
        ranked_prefixes += len(interior)
        lock[eligible & (lock == 0) & (evidence[down, up].max(axis=0) >= 3 / RISK - 1e-10)] = step
    return dict(lock_round=lock.tolist(), calibration_min=cal_min.tolist(), eligible=eligible.tolist(),
                calibration_votes=width * PATHS, primary_votes=counted_votes, primary_ranked_prefixes=ranked_prefixes)


def fit_rank_model(train, yt, val, yv, halves, policy):
    safe = [f"f{i}" for i in range(train.shape[1])]
    valid = val.set_axis(safe, axis=1)
    model = make_model(**policy)
    model.fit(train.set_axis(safe, axis=1), yt)
    return model, valid, prediction_scores(yv, model.predict_proba(valid), halves, "roc_auc")


def candidate_sizes(d, negatives):
    return list(dict.fromkeys(k for k in (0, 1, (d+3)//4, (d+1)//2, (3*d+3)//4, 7*d//8, negatives)
                              if negatives <= k < d))


def run_arm(width, method, out):
    started = stage_start = time.monotonic()
    stages = {}
    fits = Counter(full_model=0, vote=0, ranking=0, subset_validation=0, final_refits=0)

    def progress(phase, **details):
        atomic_write_json(out / "progress.json", dict(phase=phase, elapsed_s=time.monotonic()-started,
                          fits_completed=sum(fits.values()), **details))

    def end_stage(phase):
        nonlocal stage_start
        now = time.monotonic()
        stages[phase] = now - stage_start
        stage_start = now

    progress("loading")
    # alpha is unused by these games. load_heldout checks this metadata field against load_arm.
    train, yt, val, yv, top100, audit = load_arm(width, alpha=None)
    policy = {key: audit["policy"][key] for key in ("task", "seed", "backend", "model_params")}
    if (policy["task"] != "binary" or policy["seed"] != SEED or policy["backend"] != "cuda"
            or policy["model_params"]["n_estimators"] != 100 or audit["policy"]["utility"] != "roc_auc"
            or audit["classification_margin"] != MARGIN):
        raise ValueError("Archived AUC model or noninferiority settings differ from the comparison.")
    halves, halves_identity = validation_halves(yv, "binary", seed=SEED)
    features = list(train.columns)
    full = (1 << width) - 1

    def names(mask):
        return [name for i, name in enumerate(features) if mask & (1 << i)]

    end_stage("loading")
    progress("full_model")
    model, valid, base = fit_rank_model(train, yt, val, yv, halves, policy)
    fits["full_model"] += 1
    end_stage("full_model")
    vote = None
    negative = []
    if method == "hard_negative_then_gain":
        progress("vote")
        vote = negative_vote(model, valid, yv)
        negative = [i for i, step in enumerate(vote["lock_round"]) if step]
    elif method != "direct_gain":
        raise ValueError(f"Unknown method {method!r}")
    end_stage("vote")
    hard_mask = full ^ sum(1 << i for i in negative)
    remaining = [i for i in range(width) if hard_mask & (1 << i)]
    progress("ranking", negatives=len(negative))
    hard_base = base
    if negative:
        columns = names(hard_mask)
        model, _, hard_base = fit_rank_model(train[columns], yt, val[columns], yv, halves, policy)
        fits["ranking"] += 1
    gains = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(range(len(remaining)), key=lambda i: (gains.get(f"f{i}", 0.), i))
    order = [remaining[j] for j in ranked]
    end_stage("ranking")
    sizes = candidate_sizes(width, len(negative))
    trials = {}
    for i, size in enumerate(sizes):
        progress("subset_validation", candidate=i+1, total_candidates=len(sizes))
        mask = hard_mask ^ sum(1 << j for j in order[:size-len(negative)])
        if mask == hard_mask:
            scores = hard_base
        else:
            scores = subset_validation_scores(train, yt, val, yv, names(mask), halves, utility="roc_auc", **policy)
            fits["subset_validation"] += 1
        deltas = [v - b for v, b in zip(scores, base, strict=True)]
        worst = min(deltas[1:])
        trials[mask] = dict(mask=hex(mask), retained=mask.bit_count(), scores=list(scores), deltas=deltas,
                            passes=min(deltas) >= -MARGIN - 1e-12, worst_slack=worst + MARGIN,
                            score=size / width - max(0., -worst) / MARGIN)
    allowed = [m for m, v in trials.items() if v["passes"]]
    chosen = max(allowed, key=lambda m: (trials[m]["score"], trials[m]["worst_slack"])) if allowed else None
    end_stage("subset_validation")
    selected_names = names(chosen) if chosen is not None else None
    frozen = dict(dataset="madelon", seed=SEED, method=method, width=width, data=audit, top100=top100,
                  half_identity=halves_identity, features=features, full_validation=base, vote=vote,
                  negative=[features[i] for i in negative], remaining_gain_order=[features[i] for i in order],
                  sorted_gains=[gains.get(f"f{j}", 0.) for j in ranked], hard_mask=hex(hard_mask),
                  margin=MARGIN, candidate_remove_counts=sizes, trials=list(trials.values()),
                  selected=hex(chosen) if chosen is not None else None, selected_features=selected_names)
    atomic_write_json(out / "validation.json", frozen)
    identity = canonical_hash(frozen)
    end_stage("freeze")
    selected_auc = full_auc = None
    if chosen is not None:
        progress("test_access")
        job = JobSpec(identity, audit["protocol_hash"], "removal", "madelon", SEED, method,
                      dict(mask=hex(chosen)), (), "result.json")
        access = AuditLog.from_manifest(out / "test_access.json", [job])
        event = access.append_test_access(job_id=identity, stage="removal", dataset_id="madelon", seed=SEED,
                                          reason="final_removal_evaluation", decision="allowed")
        if event["decision"] != "allowed":
            raise ValueError("Held-out audit did not authorize the frozen arm.")
        test, ye = load_heldout(width, audit)
        end_stage("test_access")
        progress("final_refits")
        both = pd.concat([train, val], ignore_index=True)
        labels = np.concatenate([yt, yv])
        predictions = {}
        for mask in dict.fromkeys((full, chosen)):
            columns = names(mask)
            predictions[mask] = fit_predict(both[columns], labels, test[columns], **policy)
            fits["final_refits"] += 1
        np.savez_compressed(out / "predictions.npz", y=ye, full=predictions[full], selected=predictions[chosen])
        full_auc, selected_auc = (float(compute_utility(ye, predictions[m], metric="roc_auc")) for m in (full, chosen))
        end_stage("final_refits")
    result = dict(feasible=bool(allowed), retained=chosen.bit_count() if chosen is not None else None,
                  outside_top100=len(set(selected_names)-set(top100)) if selected_names is not None else None,
                  negative=len(negative) if vote is not None else None, full_auc=full_auc, auc=selected_auc,
                  conditional_validation_auc=hard_base[0], full_validation_auc=base[0],
                  selected_validation_auc=trials[chosen]["scores"][0] if chosen is not None else None,
                  total_fits=sum(fits.values()), fits=dict(fits), stages_s=stages,
                  voting_s=stages["vote"] if vote else 0., frozen_hash=identity, worker_s=time.monotonic()-started)
    atomic_write_json(out / "result.json", result)
    progress("complete")


def worker(width, method, out):
    try:
        run_arm(width, method, out)
    except Exception:
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = ROOT / "results/test/thesis_sampling" / datetime.now(MSK).strftime("madelon-s42-binary-widths-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "protocol.json", dict(dataset="madelon", seed=SEED, widths=[100, 500], methods=METHODS,
                      independent_limit_s=LIMIT_S, paths_each=PATHS, calibration_seed=SEED+100003,
                      tie=AUC_TIE, evidence_threshold=3/RISK, gate="strict majority ranked above calibration Min",
                      validation_margin=MARGIN, direct_removal_grid={w:candidate_sizes(w, 0) for w in (100, 500)},
                      negative_grid="original-width removal grid >= negatives, plus all negatives",
                      no_admissible_candidate="no selection, no held-out evaluation",
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows = []
    with (root / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        def live(message):
            print(message, file=log, flush=True)
            print(message, flush=True)

        live(f"Output: {root}")
        for width in (100, 500):
            for method in METHODS:
                out = root / f"{width}-{method}"
                out.mkdir()
                process = multiprocessing.get_context("spawn").Process(target=worker, args=(width, method, out))
                live(f"{out.name} started with a {LIMIT_S}s limit")
                try:
                    outcome = supervise(process, out, LIMIT_S, live, heartbeat_s=3.,
                        heartbeat_formatter=lambda p, t, left: f"{out.name} phase={p.get('phase', 'starting')} "
                            f"elapsed={t:.1f}s fits={p.get('fits_completed', 0)} deadline_in={left:.1f}s")
                finally:
                    process.close()
                result = outcome["result"]
                row = dict(width=width, method=method, status=outcome["status"], wall_s=outcome["elapsed_s"])
                for key in ("feasible", "negative", "retained", "outside_top100", "full_auc", "auc",
                            "full_validation_auc", "selected_validation_auc", "total_fits", "voting_s", "worker_s"):
                    row[key] = result.get(key)
                rows.append(row)
                write_summary(root, rows)
                live(str(row))
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
