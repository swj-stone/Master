"""Independently replay the fixed-panel votes and verify predictions, timing and paired effects."""

from functools import lru_cache
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "results/thesis_benchmark"
BASE = RUNS / "baselines-eight-preprocessed-five-seeds"
MAIN = RUNS / "context-vote-20261115"
CONTROL = RUNS / "context-vote-controls-20261115-final"
DATASETS = ("abalone", "spambase", "madelon", "credit_card_fraud", "gisette", "isolet")
SEEDS = (42, 123, 456, 789, 1024)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def signed(x):
    return hashlib.sha256(
        json.dumps(
            x, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(str(message))


@lru_cache(maxsize=30)
def source(dataset, seed, folder=None):
    folder = folder or BASE / "inputs" / f"seed-{seed}" / dataset
    m = read(folder / "metadata.json")
    for n, digest in m["sha256"].items():
        check(sha(folder / n) == digest, folder / n)
    yv, ye = [np.load(folder / f"{part}_y.npy") for part in ("val", "test")]
    indices = [
        set(np.load(folder / f"{part}_indices.npy").tolist())
        for part in ("train", "val", "test")
    ]
    check(all(a.isdisjoint(b) for a, b in combinations(indices, 2)), folder)
    rank, guard = train_test_split(
        np.arange(len(yv)),
        test_size=0.5,
        random_state=seed,
        stratify=None if dataset == "abalone" else yv,
    )
    return m, ye, yv, np.sort(rank), np.sort(guard)


def metrics(y, pred, dataset):
    if dataset == "abalone":
        return dict(mae=mean_absolute_error(y, pred), r2=r2_score(y, pred))
    classes = pred.argmax(axis=1)
    auc = (
        roc_auc_score(y, pred, multi_class="ovr", average="macro")
        if pred.shape[1] > 2
        else roc_auc_score(y, pred[:, 1])
    )
    result = dict(
        auc=auc,
        accuracy=accuracy_score(y, classes),
        precision=precision_score(
            y,
            classes,
            average="macro" if pred.shape[1] > 2 else "binary",
            zero_division=0,
        ),
    )
    if dataset == "credit_card_fraud":
        result["ap"] = average_precision_score(y, pred[:, 1])
    return result


def validate_metrics(observed, stored):
    for k, v in stored.items():
        actual = max(0.0, observed[k]) if k == "r2" else observed[k]
        np.testing.assert_allclose(actual, v, rtol=1e-8, atol=1e-10)


def ballot(candidate, reference):
    check(
        candidate.keys() == reference.keys()
        and all(np.isfinite(v) for v in (*candidate.values(), *reference.values())),
        "inconsistent panel metric schema",
    )
    return all(
        (a - reference[k] if k == "mae" else reference[k] - a)
        <= abs(reference[k]) * (0.05 if k == "ap" else 0.03) + 1e-12
        for k, a in candidate.items()
    )


def replay(est, arm, final, rank_len):
    order = est["anchor_order"]
    anchor = frozenset(order)
    live = anchor
    tails = list(map(frozenset, est["context_extensions"]))
    observed = {frozenset(s["mask"]): s["scores"] for s in est["mask_scores"]}
    rows = est["witness_rows"]
    weights = np.array(est["panel_weights"])
    check(
        len(set(rows)) == len(rows) <= 256 and min(rows) >= 0 and max(rows) < rank_len,
        "panel rows",
    )
    check(
        weights.shape == (4, len(rows))
        and (weights >= 0).all()
        and (weights.sum(axis=1) > 0).all(),
        "panel weights",
    )
    check(
        len(observed) == est["witness_predictions"]
        and all(len(s) == 4 for s in observed.values()),
        "mask cache",
    )
    check(
        tails and not tails[0] and all(anchor.isdisjoint(t) for t in tails), "contexts"
    )
    check(len(tails) == len(set(tails)), "duplicate contexts")
    path = []
    for step in est["trace"]:
        block = frozenset(step["block"])
        check(block <= live and step["before"] == len(live), "proposal path")
        new = live - block
        votes = []
        for tail in tails:
            votes.append(
                [
                    ballot(a, b)
                    for a, b in zip(
                        observed[new | tail], observed[anchor | tail], strict=True
                    )
                ]
            )
            if arm == "min" and sum(votes[-1]) <= 2:
                break
        if arm == "point":
            accept = votes[0][0]
        elif arm == "mean":
            accept = sum(map(sum, votes)) > 2 * len(votes)
        else:
            accept = all(sum(v) > 2 for v in votes)
        check(
            accept == step["accepted"] and votes == step["votes"],
            "stored ballots mismatch",
        )
        path.append((step["phase"], step["before"], tuple(step["block"]), accept))
        if accept:
            live = new
    check(
        live == final
        and live
        and (
            len(live) == 1
            or all(
                any(
                    not s["accepted"]
                    and s["before"] == len(live)
                    and s["block"] == [col]
                    and s["phase"] == est["singleton_sweeps"]
                    for s in est["trace"]
                )
                for col in live
            )
        ),
        "terminal certificate",
    )
    return path


def audit_arm(folder, row, is_primary, input_folder=None):
    d, s, a = row.dataset, int(row.seed), row.arm
    m, ye, yv, rank, guard = source(d, s, input_folder)
    result, fr, access, timer, est = [
        read(folder / n)
        for n in (
            "result.json",
            "frozen.json",
            "test_access.json",
            "watchdog.json",
            "estimator.json",
        )
    ]
    check(
        timer["status"] == "completed"
        and timer["result"] == result
        and 0 < timer["elapsed_s"] < timer["limit_s"] <= 3600,
        folder,
    )
    check(
        read(folder / "worker-completion.json")["exit_code"]
        == timer["completion_receipt_code"]
        == 0,
        folder,
    )
    check(
        fr["dataset"] == d
        and fr["seed"] == s
        and fr["mode"] == "shared"
        and not fr["enforce_guard"]
        and not fr["fallback"],
        folder,
    )
    check(
        fr["policy"] == {k: v for k, v in m["policy"].items() if k != "utility"}
        and fr["method"]
        == ("minpi" if is_primary else "gain" if a == "matched_gain" else a),
        folder,
    )
    check(signed(fr) == result["frozen_hash"] == access["frozen_hash"], folder)
    check(
        fr["guard_rows_sha256"] == signed(guard.tolist())
        and fr["rank_rows_sha256"] == signed(rank.tolist()),
        folder,
    )
    check(
        access["test_x_sha256"] == m["sha256"]["test.npy"]
        and access["test_y_sha256"] == m["sha256"]["test_y.npy"],
        folder,
    )
    support = frozenset(fr["selected"])
    check(
        len(support) == row.k == result["k"]
        and fr["selected"] == result["selected"] == fr["candidate"]
        and support <= set(m["features"]),
        folder,
    )
    fits = [
        json.loads(line) for line in (folder / "fits.jsonl").read_text().splitlines()
    ]
    started = [r["fit_id"] for r in fits if r["event"] == "started"]
    finished = [r["fit_id"] for r in fits if r["event"] == "completed"]
    check(
        started == finished and len(set(started)) == result["fits"] == row.fits, folder
    )
    for f in ("reference_s", "selector_s", "guard_s", "final_s"):
        if hasattr(row, f):
            np.testing.assert_allclose(result[f], getattr(row, f))
    np.testing.assert_allclose(timer["elapsed_s"], row.wall_s)
    check(
        sum(result[f] for f in ("reference_s", "selector_s", "guard_s", "final_s"))
        <= row.wall_s + 1.0e-6,
        folder,
    )
    with np.load(folder / "predictions.npz") as npz:
        for k, expected in [
            ("y_test", ye),
            ("rank_rows", rank),
            ("guard_rows", guard),
            ("y_guard", yv[guard]),
        ]:
            np.testing.assert_array_equal(npz[k], expected)
        test = metrics(ye, npz["test_pred"], d)
        validate_metrics(test, result["test_scores"])
        for key, pred in [
            ("full", "guard_full"),
            ("selected", "guard_selected"),
            ("candidate", "guard_candidate"),
        ]:
            validate_metrics(metrics(yv[guard], npz[pred], d), fr["validation_" + key])
        for k in result["test_scores"]:
            np.testing.assert_allclose(result["test_scores"][k], getattr(row, k))
        test_digest = hashlib.sha256(npz["test_pred"].tobytes()).hexdigest()
        guarded = metrics(yv[guard], npz["guard_selected"], d)
    trace = None
    if is_primary:
        if input_folder is None:
            check(read(folder / "arm.json") == dict(arm=a, dataset=d, seed=s), folder)
        check(est["arm"] == a, folder)
        trace = replay(est, a, support, len(rank))
    return dict(
        dataset=d,
        seed=s,
        arm=a,
        k=int(row.k),
        width=int(row.width),
        wall_s=row.wall_s,
        selector_s=row.selector_s,
        retained=row.k / row.width,
        fits=int(row.fits),
        witness_predictions=est["witness_predictions"] if is_primary else None,
        proposals=len(trace) if is_primary else None,
        anchor_k=est["anchor_k"] if is_primary else None,
        **test,
        **{"guard_" + k: v for k, v in guarded.items()},
    ), dict(
        support=support,
        trace=trace,
        test_digest=test_digest,
        estimator=est,
        frozen_hash=signed(fr),
    )


def open_campaign(root):
    protocol = read(root / "protocol.json")
    baseline_protocol = BASE / "protocol.json"
    baseline = read(baseline_protocol)
    key = "reference_protocol_sha256" if root == MAIN else "baseline_protocol_sha256"
    check(
        sha(baseline_protocol) == protocol[key]
        and protocol["inputs"] == baseline["inputs"],
        root,
    )
    if root == CONTROL:
        check(protocol["primary_protocol_sha256"] == sha(MAIN / "protocol.json"), root)
    for seed in SEEDS:
        for dataset in DATASETS:
            meta = BASE / "inputs" / f"seed-{seed}" / dataset / "metadata.json"
            check(
                sha(meta) == protocol["inputs"][str(seed)][dataset]["metadata_sha256"],
                meta,
            )
    sources = protocol["source"] if root == MAIN else protocol["sources"]
    check(
        all(sha(ROOT / path) == digest for path, digest in sources.items()),
        "running source changed: " + root.name,
    )
    p = read(root / "effective.json")
    check(p["protocol_sha256"] == sha(root / "protocol.json"), root)
    f = pd.read_csv(root / "per-seed-effective.csv")
    expected = {
        (d, s, a)
        for d in DATASETS
        for s in SEEDS
        for a in (
            (["min", "mean", "point"] + (["eager"] if s == 42 else []))
            if root == MAIN
            else ["gain", "minpi", "matched_gain"]
        )
    }
    keys = list(f[["dataset", "seed", "arm"]].itertuples(index=False, name=None))
    check(len(expected) == len(keys) and expected == set(keys), root)
    group_map = {(g["dataset"], g["seed"]): g for g in p["groups"]}
    check(
        set(group_map) == {(d, s) for d in DATASETS for s in SEEDS}
        and len(group_map) == len(p["groups"]),
        root,
    )
    invalid = []
    for (d, s), rows in f.groupby(["dataset", "seed"]):
        g = group_map[d, s]
        for attempt in g["invalid_attempts"]:
            paths = [
                root / attempt["folder"] / arm / "watchdog.json" for arm in rows.arm
            ]
            timers = [read(path) for path in paths if path.exists()]
            check(
                len(timers) < len(paths)
                or any(
                    t["status"] != "completed" or t["elapsed_s"] >= t["limit_s"]
                    for t in timers
                )
                or sum(t["elapsed_s"] for t in timers) >= 3600,
                attempt,
            )
            invalid.append(attempt)
        check(
            rows.wall_s.sum() < 3600
            and all(
                r.run_folder == g["folder"] + "/" + r.arm for r in rows.itertuples()
            ),
            g,
        )
    return f, invalid


def fraud_intervals():
    """Stratified paired row bootstrap conditional on each fitted pair and the two class counts."""
    paths = {}
    for root, arm in [(MAIN, "min"), (CONTROL, "gain")]:
        frame = pd.read_csv(root / "per-seed-effective.csv")
        for r in frame[
            (frame.dataset == "credit_card_fraud") & (frame.arm == arm)
        ].itertuples():
            paths[arm, r.seed] = root / r.run_folder / "predictions.npz"
    output = []
    for s in SEEDS:
        with np.load(paths["min", s]) as candidate, np.load(paths["gain", s]) as anchor:
            y = candidate["y_test"]
            np.testing.assert_array_equal(y, anchor["y_test"])
            pair = np.column_stack(
                [candidate["test_pred"][:, 1], anchor["test_pred"][:, 1]]
            )
        unique, groups = np.unique(pair, axis=0, return_inverse=True)
        counts = [np.bincount(groups[y == c], minlength=len(unique)) for c in (0, 1)]
        orderings = []
        for col in unique.T:
            order = np.argsort(-col, kind="stable")
            last = np.flatnonzero(np.r_[col[order][:-1] != col[order][1:], True])
            orderings.append((order, last))

        def ap(neg, pos, ordering):
            order, last = ordering
            tp = np.cumsum(pos[order])[last]
            total = np.cumsum((pos + neg)[order])[last]
            ppv = np.divide(
                tp, total, out=np.zeros_like(tp, dtype=float), where=total != 0
            )
            return np.dot(np.diff(tp, prepend=0), ppv) / pos.sum()

        observed = ap(*counts, orderings[0]) - ap(*counts, orderings[1])
        np.testing.assert_allclose(
            observed,
            average_precision_score(y, pair[:, 0])
            - average_precision_score(y, pair[:, 1]),
        )
        sample = np.empty(2000)
        rng = np.random.default_rng(20261115 + s)
        for i in range(len(sample)):
            neg, pos = [rng.multinomial(c.sum(), c / c.sum()) for c in counts]
            sample[i] = ap(neg, pos, orderings[0]) - ap(neg, pos, orderings[1])
        lo, hi = np.quantile(sample, [0.025, 0.975])
        output.append(
            dict(
                seed=s,
                positives=int(y.sum()),
                test_rows=len(y),
                ap_min_minus_gain=observed,
                row_bootstrap_q025=lo,
                row_bootstrap_q975=hi,
                replicates=len(sample),
            )
        )
    pd.DataFrame(output).to_csv(HERE / "fraud-uncertainty.csv", index=False)


def emit(main_only):
    rows = []
    evidence = {}
    attempts = {}
    audit_runs = {}
    roots = [MAIN] if main_only else [MAIN, CONTROL]
    for root in roots:
        frame, invalid = open_campaign(root)
        attempts[root.name] = invalid
        audit_runs[root.name] = dict(
            arms=len(frame),
            max_group_wall_s=frame.groupby(["dataset", "seed"]).wall_s.sum().max(),
            effective_sha256=sha(root / "effective.json"),
            protocol_sha256=sha(root / "protocol.json"),
            per_seed_sha256=sha(root / "per-seed-effective.csv"),
            invalid_attempts=invalid,
        )
        for row in frame.itertuples():
            out, info = audit_arm(root / row.run_folder, row, root == MAIN)
            key = (out["dataset"], out["seed"], out["arm"])
            evidence[key] = info
            rows.append(out)
    df = pd.DataFrame(rows)
    stability = []
    effects = []
    refit = []
    for d in DATASETS:
        g = df[df.dataset == d]
        score = "mae" if d == "abalone" else "ap" if d == "credit_card_fraud" else "auc"

        def arm(name):
            return g[g.arm == name].set_index("seed").sort_index()

        for a in (
            ["mean", "point"]
            if main_only
            else ["mean", "point", "gain", "matched_gain", "minpi"]
        ):
            delta = arm("min")[score] - arm(a)[score]
            effects.append(
                dict(
                    dataset=d,
                    contrast="min - " + a,
                    metric=score,
                    mean_delta=delta.mean(),
                    sd_delta=delta.std(),
                    min_delta=delta.min(),
                    max_delta=delta.max(),
                    min_seed=int(delta.idxmin()),
                    max_seed=int(delta.idxmax()),
                    mean_k_delta=(arm("min").k - arm(a).k).mean(),
                    mean_wall_delta_s=(arm("min").wall_s - arm(a).wall_s).mean(),
                )
            )
        for a in (
            ("min", "mean", "point")
            if main_only
            else ("min", "mean", "point", "gain", "matched_gain", "minpi")
        ):
            overlaps = []
            for u, v in combinations(SEEDS, 2):
                x = evidence[d, u, a]["support"]
                y = evidence[d, v, a]["support"]
                overlaps.append(len(x & y) / len(x | y))
            stability.append(
                dict(
                    dataset=d,
                    arm=a,
                    mean_jaccard=np.mean(overlaps),
                    min_jaccard=min(overlaps),
                    max_jaccard=max(overlaps),
                )
            )
        lazy, eager = evidence[d, 42, "min"], evidence[d, 42, "eager"]
        check(
            lazy["trace"] == eager["trace"]
            and lazy["support"] == eager["support"]
            and lazy["test_digest"] == eager["test_digest"],
            d,
        )
        for s in SEEDS:
            e = evidence[d, s, "min"]["estimator"]
            for a in ("mean", "point"):
                other = evidence[d, s, a]["estimator"]
                for name in ("anchor_order", "order", "witness_rows", "panel_weights"):
                    check(e[name] == other[name], (d, s, name))
            if not main_only:
                matched = evidence[d, s, "matched_gain"]
                check(
                    matched["estimator"]["donor_hash"]
                    == evidence[d, s, "min"]["frozen_hash"]
                    and matched["support"] == set(e["order"][: e["k"]])
                    and evidence[d, s, "gain"]["support"] == set(e["anchor_order"]),
                    (d, s),
                )
                minrow, anchor = arm("min").loc[s], arm("gain").loc[s]
                names = (
                    ["r2", "mae"]
                    if d == "abalone"
                    else ["auc", "accuracy", "ap"]
                    if d == "credit_card_fraud"
                    else ["auc", "accuracy"]
                )
                for sample, prefix in [("reserved validation", "guard_"), ("test", "")]:
                    m = {k: minrow[prefix + k] for k in names}
                    a = {k: anchor[prefix + k] for k in names}
                    refit.append(
                        dict(
                            dataset=d,
                            seed=s,
                            sample=sample,
                            metric=score,
                            min_value=m[score],
                            anchor_value=a[score],
                            min_minus_anchor=m[score] - a[score],
                            counts_against_mask_tolerance=not ballot(m, a),
                        )
                    )
    stem = "primary-" if main_only else ""
    for filename, frame in [
        ("metrics", df),
        ("paired-effects", pd.DataFrame(effects)),
        ("stability", pd.DataFrame(stability)),
    ]:
        frame.to_csv(HERE / (stem + filename + ".csv"), index=False)
    if refit:
        pd.DataFrame(refit).to_csv(HERE / "refit-vs-anchor.csv", index=False)
    print(
        df.groupby("arm")[
            ["retained", "width", "k", "selector_s", "wall_s", "auc", "mae", "ap"]
        ]
        .mean()
        .round(5)
    )
    print(
        "max group seconds by campaign",
        {k: v["max_group_wall_s"] for k, v in audit_runs.items()},
    )
    print("Invalid original groups", json.dumps(attempts))
    if not main_only:
        fraud_intervals()
        report = dict(
            campaigns=audit_runs,
            per_seed_limit_s=3600,
            distinct_effective_arms_checked=len(df),
            lazy_eager_equal_trace_support_prediction_pairs=len(DATASETS),
            matched_width_and_anchor_pairs=len(DATASETS) * len(SEEDS),
        )
        (HERE / "audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(pd.DataFrame(effects).to_string(index=False))


def width_check():
    root = RUNS / "context-vote-width-20261115"
    proto = read(root / "protocol.json")
    frame = pd.read_csv(root / "per-seed.csv")
    expected = {
        (d, a)
        for d in ("madelon", "gisette", "isolet")
        for a in ("min", "gain", "full")
    }
    check(
        len(frame) == 9
        and set(zip(frame.dataset, frame.arm)) == expected
        and (frame.seed == 42).all(),
        root,
    )
    check(all(sha(ROOT / k) == v for k, v in proto["source"].items()), root)
    check(proto["primary_protocol_sha256"] == sha(MAIN / "protocol.json"), root)
    rows = []
    for d, ds in frame.groupby("dataset"):
        raw = ROOT / proto["input_root"] / d
        prepared = BASE / "inputs/seed-42" / d
        meta = read(raw / "metadata.json")
        check(
            sha(raw / "metadata.json") == proto["inputs"][d]["metadata_sha256"]
            and ds.wall_s.sum() < 3600,
            raw,
        )
        check(
            meta["sha256"] == proto["inputs"][d]["arrays"]
            and meta["width"] == int(ds.width.iloc[0]),
            raw,
        )
        for part in ("train", "val", "test"):
            for suffix in ("_y.npy", "_indices.npy"):
                fname = part + suffix
                np.testing.assert_array_equal(
                    np.load(raw / fname), np.load(prepared / fname)
                )
        infos = {}
        for row in ds.itertuples():
            check(
                row.run_folder == f"runs/seed-42/{d}/{row.arm}"
                and row.status == "completed",
                row,
            )
            result, infos[row.arm] = audit_arm(
                root / row.run_folder, row, row.arm == "min", raw
            )
            rows.append(result)
        check(
            infos["gain"]["support"] == set(infos["min"]["estimator"]["anchor_order"])
            and infos["full"]["support"] == set(meta["features"]),
            d,
        )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "width.csv", index=False)
    print(
        out[["dataset", "arm", "width", "k", "auc", "accuracy", "wall_s"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-only", action="store_true")
    p.add_argument("--width", action="store_true")
    args = p.parse_args()
    if args.width:
        width_check()
    else:
        emit(args.primary_only)
