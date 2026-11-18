"""Time compiled paired pivotal scoring, its NumPy predecessor and matched-width controls."""
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
from experiments import exchange_vote_benchmark as exchange
from experiments import fast_minpi_five_source as five
from experiments import moment_minpi_benchmark as previous
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, SPLIT_SEED, digest, file_hash, write_json
from minppi_mofs.native_pivotal_minpi import load_native, native_top_from_training
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_minpi import _impute_validation


CANDIDATES = ("native_auto", "native_75")
METHODS = (*CANDIDATES, "random_native", "F_native", "moment_75", "F_75", "random_75", "full")
ARCHIVED = ("full", "F_75", "random_75", "moment_75")
BUILD = ROOT/"tmp/pivotal_native"


def select(method, panel, widths, *, backend="native"):
    if method in ARCHIVED:
        return previous.select(method, panel)
    if method in CANDIDATES:
        width = reduced_width(panel.X.shape[1], .75) if method == "native_75" else None
        return native_top_from_training(panel.X, panel.y, width, task=panel.task, random_state=panel.seed,
                                        groups=panel.groups, ordered_groups=panel.name in ("gas", "secom"),
                                        backend=backend)
    if method not in ("random_native", "F_native"):
        raise ValueError(f"Unknown method {method}")
    X = panel.X
    if method == "F_native" and panel.name in deployment.ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(method.split("_")[0].lower(), X, panel.y, panel.task, panel.seed, widths["native"]), {}


def measure(panel, repeats, reference=None):
    key = dict(dataset=panel.name, fold=panel.fold, seed=panel.seed)
    saved, scores, widths, rows, contracts, checks, diagnosis, units = {}, {}, {}, [], [], [], [], []
    for method in METHODS:
        cols, info = select(method, panel, widths)
        five.check_columns(cols, panel.X.shape[1])
        if method == "native_auto":
            widths["native"] = len(cols)
        elif method.endswith("_native"):
            assert len(cols) == widths["native"]
        elif method.endswith("_75"):
            assert len(cols) == reduced_width(panel.X.shape[1], .75)
        prediction, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt,
                                                      cols, panel.task, panel.seed)
        score, unit_scores, ap = deployment.score_predictions(panel.name, panel.gold, prediction,
                                                              panel.units, panel.task, decision)
        scores[method] = score, ap
        saved[f"{method}_cols"], saved[f"{method}_pred"] = cols, prediction
        if panel.name == "secom":
            saved[f"{method}_decision"] = decision
        units.extend(dict(**key, method=method, unit=u, score=v) for u, v in unit_scores.items())
        if method in CANDIDATES:
            assert info["backend"] == "native"
            if info.get("folds"):
                deployment.check_folds(info, panel.groups, panel.name in ("gas","secom"), len(panel.y))
            saved[f"{method}_info"] = np.asarray(json.dumps(info, default=adaptive.json_value,
                                                            sort_keys=True, separators=(",", ":")))
            diagnosis.append(dict(**key, method=method, width=len(cols), p=panel.X.shape[1],
                                  **{f"diag_{k}": v for k,v in info.items() if np.isscalar(v)}))
            if reference is not None:
                other, detail = select(method, panel, {}, backend="numpy")
                assert detail["backend"] == "numpy"
                exact = np.array_equal(other, cols)
                fallback_pred = prediction if exact else deployment.fit_predict(
                    panel.name, panel.X, panel.y, panel.Xt, other, panel.task, panel.seed)[0]
                contracts.append(dict(**key, method=method, native_width=len(cols), numpy_width=len(other),
                                      same_support=exact, same_prediction=np.array_equal(prediction, fallback_pred),
                                      max_score_difference=np.max(np.abs(info["scores"] - detail["scores"])),
                                      numpy_support_sha256=digest(other), native_support_sha256=digest(cols)))
        if method == "full":
            np.testing.assert_array_equal(prediction, panel.historical_full)
        if reference is not None:
            checks.append(dict(**key, method=method, width=len(cols), score=score,
                               support_sha256=digest(cols), prediction_sha256=digest(prediction)))
    if reference is not None:
        assert set(reference.files) == set(saved)
        for name, value in saved.items():
            np.testing.assert_array_equal(value, reference[name], err_msg=name)
        return rows, saved, checks, contracts, diagnosis, units

    order = np.random.default_rng([SPLIT_SEED, panel.fold, panel.seed]).permutation(METHODS).tolist()
    for repeat in range(repeats):
        shift = 3 * repeat % len(order)
        for position, method in enumerate(order[shift:] + order[:shift]):
            start = perf_counter_ns()
            cols, _ = select(method, panel, widths)
            middle = perf_counter_ns()
            pred, dec = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, saved[f"{method}_cols"])
            np.testing.assert_array_equal(pred, saved[f"{method}_pred"])
            if panel.name == "secom":
                np.testing.assert_array_equal(dec, saved[f"{method}_decision"])
            score, ap = scores[method]
            rows.append(dict(**key, method=method, repeat=repeat, position=position, width=len(cols), p=panel.X.shape[1],
                             score=score, ap=ap, select_ms=(middle-start)/1e6, refit_ms=(end-middle)/1e6,
                             total_ms=(end-start)/1e6))
    return rows, saved, checks, contracts, diagnosis, units


def protocol(repeats):
    current = adaptive.protocol(repeats)
    for file in (Path(__file__), ROOT/"src/minppi_mofs/native_pivotal_minpi.py",
                 ROOT/"src/minppi_mofs/native_pivotal_kernel.c", ROOT/"experiments/build_native_pivotal.py",
                 ROOT/"experiments/exchange_vote_benchmark.py"):
        current["source_sha256"][file.relative_to(ROOT).as_posix()] = file_hash(file)
    current.update(methods=list(METHODS), primary="Native training-selected width",
                   sensitivity="Fixed native width and exact replay of prior NumPy moment_75 supports",
                   timing="7 fresh select/fit/predict calls for all 8 arms in 7 distinct positions after "
                          "one warm-up. Explicit native backend. DLL load and compilation measured separately.",
                   build=json.loads((BUILD/"build.json").read_text(encoding="utf-8")),
                   binary_sha256={f.name: file_hash(f) for f in BUILD.iterdir()
                                  if f.suffix.lower() in (".dll", ".so", ".dylib")})
    if not current["binary_sha256"]:
        raise FileNotFoundError("Compile the native kernel before running this experiment")
    return current


def run(output, verify):
    current = protocol(7)
    manifest = output/"protocol.json"
    if verify:
        assert json.loads(json.dumps(current)) == json.loads(manifest.read_text(encoding="utf-8"))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest, current)
    start = perf_counter_ns()
    library = load_native()
    first_load_ms = (perf_counter_ns() - start) / 1e6
    assert library is not None
    write_json(output/("verification-load.json" if verify else "initial-load.json"),
               dict(first_load_ms=first_load_ms, scope="First explicit DLL load after Python package imports; "
                                                      "excluded from warm selection and fitting timings"))
    rows, inputs, audited, contracts, parity, diagnostics, units, supports = [], [], [], [], [], [], [], {}
    for panel in previous.panels():
        label = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        inputs.append(adaptive.input_hashes(panel))
        if verify:
            with np.load(output/label) as archive:
                _, _, records, fallbacks, _, _ = measure(panel, 7, archive)
                audited.extend(records)
                contracts.extend(fallbacks)
        else:
            trials, data, _, _, logs, unit_records = measure(panel, 7)
            with np.load(adaptive.RUN/label) as old:
                for method in ARCHIVED:
                    for suffix in ("cols", "pred", *(("decision",) if panel.name == "secom" else ())):
                        np.testing.assert_array_equal(data[f"{method}_{suffix}"], old[f"{method}_{suffix}"])
                a, b = data["native_75_cols"], old["moment_75_cols"]
                one = dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, width=len(a),
                           same_support=np.array_equal(a, b), same_predictions=np.array_equal(
                               data["native_75_pred"], old["moment_75_pred"]),
                           jaccard=len(np.intersect1d(a,b))/len(np.union1d(a,b)))
                parity.append(one)
            rows.extend(trials)
            diagnostics.extend(logs)
            units.extend(unit_records)
            for method in CANDIDATES:
                supports.setdefault((panel.name,panel.fold,method), []).append((panel.seed, data[f"{method}_cols"]))
            np.savez_compressed(output/label, **data)
        if panel.seed == five.SEEDS[-1] or panel.name not in previous.DATASETS:
            print(f"{'Verified' if verify else 'Measured'} {label}", flush=True)
    assert current == protocol(7)
    frame = pd.DataFrame(inputs)
    pd.testing.assert_frame_equal(frame, pd.read_csv(adaptive.RUN/"inputs.csv"), check_dtype=False)
    if verify:
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/"inputs.csv"), check_dtype=False)
        pd.DataFrame(contracts).to_csv(output/"numpy-fallback.csv", index=False)
        pd.DataFrame(audited).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        frame.to_csv(output/"inputs.csv", index=False)
        pd.DataFrame(parity).to_csv(output/"fixed-width-parity.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(output/"diagnostics.csv", index=False)
        pd.DataFrame(units).to_csv(output/"unit-scores.csv", index=False)
        pairs = [dict(dataset=name, fold=fold, method=method, a=a, b=b,
                      jaccard=len(np.intersect1d(x,y))/len(np.union1d(x,y)))
                 for (name,fold,method), bank in supports.items() for (a,x),(b,y) in combinations(bank,2)]
        pd.DataFrame(pairs).to_csv(output/"same-outer-seed-overlap.csv", index=False)
        exchange.summarize(output, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
