"""Independent, auditable stages of the raw-width paired PIv3 experiment."""

import json
import time
from pathlib import Path

import numpy as np

from experiments.test.piv3_raw_splits import load_split
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import make_model
from minppi_mofs.utility import compute_utility


def _read(path):
    return json.loads(path.read_bytes())


def _scores(labels, predictions, utility):
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite model predictions")
    other = (np.mean(np.abs(labels - predictions)) if utility == "r2"
             else np.mean(labels == predictions.argmax(axis=1)))
    return [float(compute_utility(labels, predictions, metric=utility)), float(other)]


def _predict(model, matrix, task):
    values = model.predict(matrix) if task == "regression" else model.predict_proba(matrix)
    return np.asarray(values, dtype=float)


def _risk(scores, reference, tolerance, regression):
    losses = np.subtract(reference, scores)
    if regression:
        losses[1] *= -1
    budget = np.asarray(reference) * tolerance
    ratios = np.where(losses > 0, 2., 0.)
    np.divide(losses, budget, out=ratios, where=budget > 0)
    return float(ratios.max())


def run_stage(root: Path, data: Path, dataset: str, stage: str, method: str, arm: str, pct: int, out: Path):
    started = time.monotonic()
    meta_path = Path(data)/dataset/"metadata.json"
    meta = _read(meta_path)
    columns = meta["features"]
    offsets = {name: index for index, name in enumerate(columns)}
    policy = {k: v for k, v in meta["policy"].items() if k != "utility"}
    task, utility = policy["task"], meta["utility"]
    base = Path(root)/"stages"/dataset
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reference_dir = base/"reference"
    front_dir = base/"front"/f"tol-{pct:02}"
    backend_dir = base/"backend"/arm/method/f"tol-{pct:02}"
    fits, last_tick, last_phase = 0, 0., None

    def progress(phase, **counts):
        nonlocal last_tick, last_phase
        now = time.monotonic()
        if now - last_tick >= 1.0 or phase != last_phase:
            atomic_write_json(out/"progress.json", dict(phase=phase, fits=fits, **counts))
            last_tick, last_phase = now, phase

    def matrix(frame, names):
        if names == columns:
            return frame.to_numpy(copy=False)
        return frame.iloc[:, [offsets[c] for c in names]].to_numpy(copy=False)

    def finish(names, scores, **details):
        check_names(names)
        record = dict(dataset=dataset, stage=stage, method=method, arm=arm, pct=pct,
                      names=names, k=len(names), scores=scores, fits=fits, elapsed_s=time.monotonic()-started, **details)
        atomic_write_json(out/"stage.json", record)
        progress("completed", checked=record.get("checked", 1))
        return record

    def check_names(names):
        members = set(names)
        if not names or names != [c for c in columns if c in members]:
            raise ValueError("Expected nonempty ordered subset")

    progress("loading", checked=0)
    train, yt = load_split(data, dataset, "train")
    val, yv = load_split(data, dataset, "val")

    def fit_val(names):
        nonlocal fits
        model = make_model(**policy)
        model.fit(matrix(train, names), yt)
        fits += 1
        prediction = _predict(model, matrix(val, names), task)
        return model, prediction, _scores(yv, prediction, utility)

    if stage == "reference":
        progress("full_validation", checked=0)
        model, predictions, scores = fit_val(columns)
        model.save_model(out/"model.ubj")
        np.savez(out/"predictions.npz", y=yv, predictions=predictions)
        return finish(columns, scores)

    if stage in ("test_full", "test_selected"):
        frozen_path = (backend_dir/"stage.json") if stage == "test_selected" else meta_path
        frozen = _read(frozen_path)
        if stage == "test_selected":
            if (frozen["dataset"], frozen["stage"], frozen["arm"], frozen["method"], frozen["pct"]) != (
                    dataset, "backend", arm, method, pct):
                raise ValueError("Mismatched frozen selection")
            names = frozen["names"]
        else:
            names = frozen["features"]
        check_names(names)
        digest = canonical_hash(frozen)
        progress("train_val_refit", checked=0)
        model = make_model(**policy)
        model.fit(np.concatenate((matrix(train, names), matrix(val, names))), np.concatenate((yt, yv)))
        fits += 1
        atomic_write_json(out/"test_access.json", dict(dataset=dataset, selection_hash=digest,
                          frozen_source=str(frozen_path), test_x_sha256=meta["sha256"]["test.npy"],
                          test_y_sha256=meta["sha256"]["test_y.npy"]))
        test, ytest = load_split(data, dataset, "test")
        progress("held_out_prediction", checked=0)
        predictions = _predict(model, matrix(test, names), task)
        scores = _scores(ytest, predictions, utility)
        np.savez(out/"predictions.npz", y=ytest, predictions=predictions)
        return finish(names, scores, selection_hash=digest)

    ref = _read(reference_dir/"stage.json")
    if ref["dataset"] != dataset or ref["stage"] != "reference" or ref["names"] != columns:
        raise ValueError("Mismatched full validation reference")
    reference = ref["scores"]
    regression = task == "regression"
    tolerance = pct / 100
    if pct not in (3, 5, 8):
        raise ValueError("Unexpected tolerance")

    if stage == "front":
        from minppi_mofs.piv3 import search

        budget = 12
        trials, predictions = [], {}
        winner, winner_model = (len(columns), 0., columns), None

        def evaluate(names):
            nonlocal winner, winner_model
            progress("voting_search", checked=len(trials), candidates=budget, candidate_k=len(names))
            model, pred, scores = fit_val(names)
            ratio = _risk(scores, reference, tolerance, regression)
            trials.append(dict(names=names, scores=scores))
            predictions[f"trial_{len(trials)}"] = pred
            key = (len(names), ratio, names)
            if ratio <= 1 + 1e-12 and key < winner:
                winner, winner_model = key, model
            return scores

        progress("voting", checked=0, candidates=budget)
        names, diag = search(train, yt, search_seed=42, tolerance=tolerance, reference_scores=reference,
                             evaluate=evaluate, budget=budget, task=task)
        if names != winner[2] or fits != len(trials) or len(trials) != len(diag["steps"]):
            raise ValueError("Front fit ledger or winning model is inconsistent")
        scores = reference
        if winner_model is not None:
            winner_model.save_model(out/"model.ubj")
            scores = trials[diag["winner_id"] - 1]["scores"]
        np.savez(out/"validation_trials.npz", y=yv, **predictions)
        atomic_write_json(out/"voting.json", diag)
        return finish(names, scores, checked=len(trials), trials=trials, search_seed=42, budget=budget)

    if stage != "backend" or arm not in ("piv3", "plain"):
        raise ValueError(f"Unknown stage or arm {stage!r}, {arm!r}")
    from experiments.test.piv3_raw_ranker import rank

    support, model_dir = columns, reference_dir
    cached = {tuple(columns): (reference, 0)}
    if arm == "piv3":
        front = _read(front_dir/"stage.json")
        if (front["stage"], front["dataset"], front["pct"]) != ("front", dataset, pct):
            raise ValueError("Mismatched PI validation selection")
        support = front["names"]
        check_names(support)
        if support != columns:
            model_dir = front_dir
        for index, trial in enumerate(front["trials"], 1):
            cached[tuple(trial["names"])] = trial["scores"], index
        if cached[tuple(support)][0] != front["scores"]:
            raise ValueError("PI support missing from fit ledger")

    def input_model():
        model = make_model(**policy)
        model.load_model(model_dir/"model.ubj")
        return model

    progress("ranking", checked=0, candidates=len(support))
    ordering, aux, _ = rank(method, train, yt, val, yv, policy=policy, metric=utility, support=support,
                            input_model=input_model, progress=progress)
    fits += aux
    checks, ordered, seen = [], [], set()
    allowed = set(support)
    thresholds = {p for p in (3, 5, 8) if p >= pct} if arm == "plain" else {pct}
    for name in ordering:
        if name in seen or name not in allowed:
            raise ValueError("Ranker returned repeated or ineligible feature")
        seen.add(name)
        ordered.append(name)
        names = [c for c in support if c in seen]
        progress("prefix_validation", checked=len(checks), candidates=len(support), candidate_k=len(names))
        cached_trial = cached.get(tuple(names))
        if cached_trial is None:
            _, predictions, scores = fit_val(names)
        else:
            scores, index = cached_trial
        crossed = sorted(p for p in thresholds if _risk(scores, reference, p/100, regression) <= 1 + 1e-12)
        accepted = pct in crossed
        if crossed:
            if cached_trial is not None:
                archive = (front_dir/"validation_trials.npz") if index else (reference_dir/"predictions.npz")
                with np.load(archive, allow_pickle=False) as saved:
                    predictions = saved[f"trial_{index}" if index else "predictions"]
            np.savez(out/f"validation-k{len(names)}.npz", y=yv, predictions=predictions)
            if accepted:
                np.savez(out/"predictions.npz", y=yv, predictions=predictions)
        checkpoint = dict(k=len(names), scores=scores, fits=fits, elapsed_s=time.monotonic()-started,
                          accepted=accepted, reused=cached_trial is not None)
        checks.append(checkpoint)
        with (out/"prefix_trace.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(dict(added=name, **checkpoint), allow_nan=False) + "\n")
        for p in crossed:
            atomic_write_json(out/f"pass-tol-{p:02}.json", dict(
                pct=p, k=len(names), names=names, scores=scores, fits=fits, elapsed_s=checkpoint["elapsed_s"],
                predictions=f"validation-k{len(names)}.npz", selector_aux_fits=aux))
        thresholds.difference_update(crossed)
        if accepted:
            return finish(names, scores, checked=len(checks), checks=checks, prefixes=checks, ranking=ordered, ranking_prefix=ordered,
                          selector_aux_fits=aux, support_k=len(support))
    raise ValueError("Ranker exhausted its certified support without a passing prefix")
