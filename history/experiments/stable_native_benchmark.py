"""Check guarded native minPI against NumPy and original fixed-width supports on all 69 cases."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_benchmark as adaptive
from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments import moment_minpi_benchmark as previous
from experiments import native_minpi_benchmark as unguarded
from experiments.stable_minpi_stress import ROOT, SPLIT_SEED, digest, file_hash, write_json
from minppi_mofs.native_pivotal_minpi import load_native
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_native_minpi import stable_native_top_from_training


CANDIDATES = ("stable_auto", "stable_75")
METHODS = (*CANDIDATES, "random_stable", "random_75", "moment_75", "full")


def select(method, panel, widths):
    if method in CANDIDATES:
        count = None if method == "stable_auto" else reduced_width(panel.X.shape[1], .75)
        return stable_native_top_from_training(panel.X, panel.y, count, task=panel.task,
                                               random_state=panel.seed, groups=panel.groups,
                                               ordered_groups=panel.name in ("gas", "secom"))
    if method == "random_stable":
        return unguarded.select("random_native", panel, widths)
    return previous.select(method, panel)


def run_panel(panel, repeats, reference=None):
    key = dict(dataset=panel.name, fold=panel.fold, seed=panel.seed)
    widths, records, scores, arrays, checks, audit, units = {}, [], {}, {}, [], [], []
    for method in METHODS:
        cols, info = select(method, panel, widths)
        five.check_columns(cols, panel.X.shape[1])
        if method == "stable_auto":
            widths["native"] = len(cols)
        elif method == "random_stable":
            assert len(cols) == widths["native"]
        elif method.endswith("_75"):
            assert len(cols) == reduced_width(panel.X.shape[1], .75)
        pred, dec = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
        score, by_unit, ap = deployment.score_predictions(panel.name, panel.gold, pred, panel.units, panel.task, dec)
        scores[method] = score, ap
        units.extend(dict(**key, method=method, unit=u, score=s) for u,s in by_unit.items())
        arrays[f"{method}_cols"], arrays[f"{method}_pred"] = cols, pred
        if panel.name == "secom":
            arrays[f"{method}_decision"] = dec
        if method == "full":
            np.testing.assert_array_equal(pred, panel.historical_full)
        if method in CANDIDATES:
            arrays[f"{method}_info"] = np.asarray(json.dumps(info, default=adaptive.json_value,
                                                              sort_keys=True, separators=(",", ":")))
            deployment.check_folds(info, panel.groups, panel.name in ("gas", "secom"), len(panel.y))
            # The reference evaluation, including ambiguity checks, is outside
            # every performance measurement. The selector's own fallback is timed.
            raw_method = method.replace("stable", "native")
            canonical, ref_info = unguarded.select(raw_method, panel, {}, backend="numpy")
            np.testing.assert_array_equal(cols, canonical, err_msg=f"{panel.name}-{panel.seed}-{method}")
            checks.append(dict(**key, method=method, width=len(cols), p=panel.X.shape[1],
                               reference_support_sha256=digest(canonical), support_sha256=digest(cols),
                               selection_backend=info["selection_backend"], guard_reason=info["guard_reason"],
                               max_score_difference=np.max(np.abs(ref_info["scores"] - info["scores"]))))
        if reference is not None:
            audit.append(dict(**key, method=method, width=len(cols), score=score,
                              support_sha256=digest(cols), prediction_sha256=digest(pred)))
    np.testing.assert_array_equal(arrays["stable_75_cols"], arrays["moment_75_cols"])
    np.testing.assert_array_equal(arrays["stable_75_pred"], arrays["moment_75_pred"])
    if reference is not None:
        assert set(arrays) == set(reference.files)
        for field, value in arrays.items():
            np.testing.assert_array_equal(value, reference[field], err_msg=field)
        return records, arrays, checks, audit, units
    order = np.random.default_rng([SPLIT_SEED, panel.fold, panel.seed]).permutation(METHODS).tolist()
    for repeat in range(repeats):
        shift = repeat % len(order)
        for pos, method in enumerate(order[shift:] + order[:shift]):
            start = perf_counter_ns()
            cols, _ = select(method, panel, widths)
            mid = perf_counter_ns()
            pred, dec = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[f"{method}_cols"])
            np.testing.assert_array_equal(pred, arrays[f"{method}_pred"])
            if panel.name == "secom":
                np.testing.assert_array_equal(dec, arrays[f"{method}_decision"])
            score, ap = scores[method]
            records.append(dict(**key, method=method, repeat=repeat, position=pos, width=len(cols), p=panel.X.shape[1],
                                score=score, ap=ap, select_ms=(mid-start)/1e6, refit_ms=(end-mid)/1e6,
                                total_ms=(end-start)/1e6))
    return records, arrays, checks, audit, units


def protocol():
    current = unguarded.protocol(7)
    for f in (Path(__file__), ROOT/"experiments/native_minpi_benchmark.py",
              ROOT/"src/minppi_mofs/stable_native_minpi.py"):
        current["source_sha256"][f.relative_to(ROOT).as_posix()] = file_hash(f)
    current.update(methods=list(METHODS), primary="stable_auto with both numerical guards and matched random",
                   sensitivity="stable_75 with the numerical ranking guard, matched random and NumPy moment_75",
                   timing="Seven full computations after warm-up. Shift six seeded arms by one place per round. "
                          "Width selection and boundary reference recomputation are within the timed selector. "
                          "Independent support checks and compilation/library loading are outside the clock.")
    return current


def run(output, verify):
    current = protocol()
    manifest = output/"protocol.json"
    if verify:
        assert json.loads(manifest.read_text(encoding="utf-8")) == json.loads(json.dumps(current))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest, current)
    start = perf_counter_ns()
    load_native()
    first_load = (perf_counter_ns() - start) / 1e6
    write_json(output/("verification-load.json" if verify else "initial-load.json"),
               dict(first_explicit_load_ms=first_load, scope="After importing the package"))
    inputs, records, guards, audit, per_unit, supports = [], [], [], [], [], {}
    for panel in previous.panels():
        filename = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        inputs.append(adaptive.input_hashes(panel))
        if verify:
            with np.load(output/filename) as stored:
                _, _, checks, verified, _ = run_panel(panel, 7, stored)
                audit.extend(verified)
        else:
            rows, arrays, checks, _, units = run_panel(panel, 7)
            with np.load(adaptive.RUN/filename) as old:
                for method in ("full","random_75","moment_75"):
                    for suffix in ("cols", "pred", *(("decision",) if panel.name == "secom" else ())):
                        k = f"{method}_{suffix}"
                        np.testing.assert_array_equal(arrays[k], old[k], err_msg=k)
            records.extend(rows)
            per_unit.extend(units)
            for method in CANDIDATES:
                supports.setdefault((panel.name,panel.fold,method), []).append((panel.seed, arrays[f"{method}_cols"]))
            np.savez_compressed(output/filename, **arrays)
        guards.extend(checks)
        if panel.seed == five.SEEDS[-1] or panel.name not in previous.DATASETS:
            print(f"{'Verified' if verify else 'Measured'} {filename}", flush=True)
    assert current == protocol()
    frame = pd.DataFrame(inputs)
    pd.testing.assert_frame_equal(frame, pd.read_csv(adaptive.RUN/"inputs.csv"), check_dtype=False)
    checks = pd.DataFrame(guards)
    assert len(checks) == 138 and checks.support_sha256.eq(checks.reference_support_sha256).all()
    if verify:
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/"inputs.csv"), check_dtype=False)
        saved = pd.read_csv(output/"boundary-audit.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(checks, saved, check_dtype=False)
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        frame.to_csv(output/"inputs.csv", index=False)
        checks.to_csv(output/"boundary-audit.csv", index=False)
        pd.DataFrame(per_unit).to_csv(output/"unit-scores.csv", index=False)
        pairs = [dict(dataset=name, fold=fold, method=m, a=a, b=b,
                      jaccard=len(np.intersect1d(x,y))/len(np.union1d(x,y)))
                 for (name,fold,m), bank in supports.items() for (a,x),(b,y) in combinations(bank,2)]
        pd.DataFrame(pairs).to_csv(output/"same-outer-seed-overlap.csv", index=False)
        cases = five.summarize_timings(output, records)
        assert len(cases) == 69 * len(METHODS)
        raw = pd.DataFrame(records)
        secom = raw[raw.dataset == "secom"].drop_duplicates(["seed","method"])
        secom[["dataset","seed","method","score","ap"]].to_csv(output/"secom-ap.csv", index=False)
        window = cases[adaptive.CASE+["method","joint"]].copy()
        for name, subset in (("first", raw[raw.repeat<3]), ("last", raw[raw.repeat>=4])):
            t = subset.groupby(adaptive.CASE+["method"], as_index=False).total_ms.median()
            f = t[t.method=="full"][adaptive.CASE+["total_ms"]].rename(columns={"total_ms":"baseline"})
            t = t.merge(f,on=adaptive.CASE,validate="many_to_one")
            t[name] = t.total_ms<t.baseline
            window = window.merge(t[adaptive.CASE+["method",name]],on=adaptive.CASE+["method"],validate="one_to_one")
        window["robust_joint"] = window.joint & window["first"] & window["last"]
        window.to_csv(output/"speed-windows.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
