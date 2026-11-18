"""Measure training-selected pivotal exchanges against matched and exact-constant filters."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_benchmark as adaptive
from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments import moment_minpi_benchmark as previous
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, SPLIT_SEED, digest, file_hash, write_json
from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_minpi import _impute_validation


CANDIDATES = ("exchange_auto", "exchange_75")
METHODS = (*CANDIDATES, "random_exchange", "F_exchange", "random_75", "F_75", "constant", "full")
CASE = adaptive.CASE


def _varies(X):
    """Return a mask of columns containing at least two different finite values."""
    if sparse.issparse(X):
        columns = X.tocsc(copy=False)
        if not columns.has_canonical_format:
            columns = columns.copy()
            columns.sum_duplicates()
        counts = np.diff(columns.indptr)
        stored = counts > 0
        lo, hi = np.full(columns.shape[1], np.inf), np.full(columns.shape[1], -np.inf)
        if stored.any():
            starts = columns.indptr[:-1][stored]
            data = columns.data
            finite = np.isfinite(data)
            lo[stored] = np.minimum.reduceat(np.where(finite, data, np.inf), starts)
            hi[stored] = np.maximum.reduceat(np.where(finite, data, -np.inf), starts)
        implicit = counts < columns.shape[0]
        lo[implicit] = np.minimum(lo[implicit], 0)
        hi[implicit] = np.maximum(hi[implicit], 0)
        return lo < hi
    lo, hi = X.min(axis=0), X.max(axis=0)
    exceptional = ~(np.isfinite(lo) & np.isfinite(hi))
    varying = lo < hi
    if exceptional.any():
        x = X[:, exceptional]
        ok = np.isfinite(x)
        varying[exceptional] = (np.min(x, axis=0, where=ok, initial=np.inf) <
                                np.max(x, axis=0, where=ok, initial=-np.inf))
    return varying


def constant_columns(X):
    """Keep all varying columns. Drop constant and entirely missing columns.

    An initial bounded deterministic scan can certify variation immediately.
    Check every remaining candidate on all training rows. At least one column
    is returned so the common downstream estimator can also fit constant data.
    """
    n, p = X.shape
    if n <= 32:
        varying = _varies(X)
    else:
        probes = np.linspace(0, n - 1, 32, dtype=np.intp)
        varying = _varies(X[probes])
        remaining = np.flatnonzero(~varying)
        if len(remaining):
            varying[remaining] = _varies(X if len(remaining) == p else X[:, remaining])
    kept = np.flatnonzero(varying)
    return kept if len(kept) else np.array([0], dtype=np.intp)


def select(method, panel, widths):
    if method in ("full", "F_75", "random_75"):
        return previous.select(method, panel)
    if method == "constant":
        return constant_columns(panel.X), {}
    if method in CANDIDATES:
        count = None if method == "exchange_auto" else reduced_width(panel.X.shape[1], .75)
        return exchange_vote_top_from_training(panel.X, panel.y, count, task=panel.task,
                                               random_state=panel.seed, groups=panel.groups,
                                               ordered_groups=panel.name in ("gas", "secom"))
    name, suffix = method.split("_")
    if suffix != "exchange" or name not in ("random", "F"):
        raise ValueError(f"Unknown comparison: {method}")
    X = panel.X
    if name == "F" and panel.name in deployment.ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(name.lower(), X, panel.y, panel.task, panel.seed, widths["exchange"]), {}


def measure(panel, repeats, reference=None):
    """Measure one independent panel, return raw times, arrays, diagnostics, unit scores and replay."""
    key = dict(dataset=panel.name, fold=panel.fold, seed=panel.seed)
    p = panel.X.shape[1]
    saved, records, diagnostics, unit_scores, audited = {}, [], [], [], []
    scores, widths = {}, {}
    for method in METHODS:
        cols, info = select(method, panel, widths)
        five.check_columns(cols, p)
        if method == "exchange_auto":
            widths["exchange"] = len(cols)
        elif method.endswith("_exchange"):
            assert len(cols) == widths["exchange"]
        elif method.endswith("_75"):
            assert len(cols) == reduced_width(p, .75)
        pred, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
        score, units, ap = deployment.score_predictions(panel.name, panel.gold, pred, panel.units,
                                                        panel.task, decision)
        scores[method] = score, ap
        saved[f"{method}_cols"], saved[f"{method}_pred"] = cols, pred
        if panel.name == "secom":
            saved[f"{method}_decision"] = decision
        if method in CANDIDATES:
            folds = info.get("folds", [])
            if folds and all(isinstance(v, dict) and {"fit_rows", "held_rows"} <= v.keys() for v in folds):
                deployment.check_folds(info, panel.groups, panel.name in ("gas", "secom"), len(panel.y))
            saved[f"{method}_diagnostics"] = np.asarray(json.dumps(
                info, default=adaptive.json_value, sort_keys=True, separators=(",", ":")))
            scalars = {f"diag_{k}": v for k, v in info.items() if np.isscalar(v)}
            diagnostics.append(dict(**key, method=method, width=len(cols), p=p, **scalars))
        if method == "full" and panel.historical_full is not None:
            np.testing.assert_array_equal(pred, panel.historical_full)
        unit_scores.extend(dict(**key, method=method, unit=u, score=s) for u,s in units.items())
        if reference is not None:
            audited.append(dict(**key, method=method, width=len(cols), score=score,
                                support_sha256=digest(cols), prediction_sha256=digest(pred)))
    if reference is not None:
        assert set(saved) == set(reference.files)
        for field, array in saved.items():
            np.testing.assert_array_equal(array, reference[field], err_msg=field)
        return records, saved, diagnostics, unit_scores, audited

    order = np.random.default_rng([SPLIT_SEED, panel.fold, panel.seed]).permutation(METHODS).tolist()
    for repeat in range(repeats):
        shift = repeat * 3 % len(order)
        for pos, method in enumerate(order[shift:] + order[:shift]):
            start = perf_counter_ns()
            cols, _ = select(method, panel, widths)
            middle = perf_counter_ns()
            pred, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols,
                                                    panel.task, panel.seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, saved[f"{method}_cols"])
            np.testing.assert_array_equal(pred, saved[f"{method}_pred"])
            if panel.name == "secom":
                np.testing.assert_array_equal(decision, saved[f"{method}_decision"])
            score, ap = scores[method]
            records.append(dict(**key, method=method, repeat=repeat, position=pos, width=len(cols), p=p,
                                score=score, ap=ap, select_ms=(middle-start)/1e6,
                                refit_ms=(end-middle)/1e6, total_ms=(end-start)/1e6))
    return records, saved, diagnostics, unit_scores, audited


def summarize(output, records):
    cases = five.summarize_timings(output, records)
    assert len(cases) == len(METHODS)*69 and len(cases[CASE].drop_duplicates()) == 69
    old = pd.read_csv(adaptive.RUN/"cases.csv")
    common = old[old.method.isin(["full","F_75","random_75"])][CASE+["method", "width", "score", "joint"]]
    match = cases.merge(common, on=CASE+["method"], suffixes=("", "_old"), validate="one_to_one")
    assert len(match) == 3*69
    np.testing.assert_array_equal(match.width, match.width_old)
    np.testing.assert_allclose(match.score, match.score_old, rtol=0, atol=1e-14)
    match.to_csv(output/"archived-controls.csv", index=False)
    frame = pd.DataFrame(records)
    secom = frame[frame.dataset == "secom"].drop_duplicates(["seed","method"])
    secom[["dataset","seed","method","score","ap"]].to_csv(output/"secom-ap.csv", index=False)
    for label, subset in (("first", frame[frame.repeat < 3]),
                          ("last", frame[frame.repeat >= frame.repeat.max()-2])):
        times = subset.groupby(CASE+["method"], as_index=False).total_ms.median()
        full = times[times.method=="full"][CASE+["total_ms"]].rename(columns={"total_ms":"full_ms"})
        times = times.merge(full, on=CASE, validate="many_to_one")
        times[label] = times.total_ms < times.full_ms
        cases = cases.merge(times[CASE+["method",label]], on=CASE+["method"], validate="one_to_one")
    cases["joint_both_halves"] = cases.joint & cases["first"] & cases["last"]
    cases["joint_no_loss"] = cases.joint & (cases.gap >= 0)
    cases["keep_fraction"] = cases.width / cases.p
    stats = dict(cases=("fold","size"), joint=("joint","sum"), within=("within","sum"),
                 joint_5pct=("joint_5pct","sum"), joint_10pct=("joint_10pct","sum"),
                 joint_no_loss=("joint_no_loss","sum"), joint_both_halves=("joint_both_halves","sum"),
                 mean_keep=("keep_fraction","mean"), median_select_ms=("select_ms","median"),
                 median_ratio=("ratio","median"))
    cases.groupby("method").agg(**stats).to_csv(output/"sensitivity.csv")
    cases.groupby(["dataset","method"]).agg(**stats).to_csv(output/"source-sensitivity.csv")


def protocol(repeats):
    current = adaptive.protocol(repeats)
    for f in (Path(__file__), ROOT/"src/minppi_mofs/exchange_vote_minpi.py"):
        current["source_sha256"][f.relative_to(ROOT).as_posix()] = file_hash(f)
    current.update(methods=list(METHODS), primary="exchange_auto; no dataset names or test labels in selection",
                   sensitivity="exchange at reduced_width(p, .75), independently timed",
                   controls="Matched uniform random and task F supplied the true width; exact train-only "
                            "constant filter at its own width; all features directly",
                   timing="One untimed warm-up per method. Every round recomputes the complete selector and "
                          "fits and predicts via the common historical pipeline. Seeded initial permutation "
                          "rotates three of eight places per round. Exclude scoring, assertions and file I/O.")
    return current


def run(output, repeats, verify):
    current = protocol(repeats)
    manifest = output/"protocol.json"
    if verify:
        assert json.loads(manifest.read_text(encoding="utf-8")) == json.loads(json.dumps(current))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest, current)
    inputs, trials, metadata, units, audit = [], [], [], [], []
    supports = {}
    for panel in previous.panels():
        filename = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        inputs.append(adaptive.input_hashes(panel))
        if verify:
            with np.load(output/filename) as reference:
                *_, rows = measure(panel, repeats, reference)
            audit.extend(rows)
        else:
            times, values, info, scores, _ = measure(panel, repeats)
            with np.load(adaptive.RUN/filename) as old:
                for method in ("full", "F_75", "random_75"):
                    for suffix in ("cols", "pred", *(("decision",) if panel.name == "secom" else ())):
                        field = f"{method}_{suffix}"
                        np.testing.assert_array_equal(values[field], old[field], err_msg=field)
            np.savez_compressed(output/filename, **values)
            trials.extend(times)
            metadata.extend(info)
            units.extend(scores)
            for method in CANDIDATES:
                supports.setdefault((panel.name,panel.fold,method), []).append((panel.seed,values[f"{method}_cols"]))
        if panel.seed == five.SEEDS[-1] or panel.name not in previous.DATASETS:
            print(f"{'Verified' if verify else 'Measured'} {filename}", flush=True)
    assert current == protocol(repeats)
    frame = pd.DataFrame(inputs)
    assert len(frame) == 69 and not frame.duplicated(CASE).any()
    pd.testing.assert_frame_equal(frame, pd.read_csv(adaptive.RUN/"inputs.csv"), check_dtype=False)
    if verify:
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/"inputs.csv"), check_dtype=False)
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        frame.to_csv(output/"inputs.csv", index=False)
        pd.DataFrame(metadata).to_csv(output/"diagnostics.csv", index=False)
        pd.DataFrame(units).to_csv(output/"unit-scores.csv", index=False)
        pairs = [dict(dataset=name, fold=fold, method=method, a=a, b=b,
                      jaccard=len(np.intersect1d(x,y))/len(np.union1d(x,y)))
                 for (name,fold,method), rows in supports.items() for (a,x),(b,y) in combinations(rows,2)]
        pd.DataFrame(pairs).to_csv(output/"same-outer-seed-overlap.csv", index=False)
        summarize(output, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("At least seven repeats are required")
    with threadpool_limits(limits=1):
        run(args.output, args.repeats, args.verify)
