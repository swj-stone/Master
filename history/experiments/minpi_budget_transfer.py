"""Moment minPI at both fixed widths on four panels outside the 69 development configurations."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.minpi_study_external import FILES, load_challenge
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction
from minppi_mofs.moment_minpi import moment_top_from_training
from minppi_mofs.sketch_minpi import reduced_width


SEEDS = (7883, 16091, 32029)
FRACTIONS = {"50": .5, "75": .75}
METHODS = tuple(f"{name}_{width}" for width in FRACTIONS for name in ("moment", "F", "random")) + ("full",)


def panels():
    # The official validation labels are used only after fitting.
    for name in FILES:
        (X, y), (Xt, truth) = load_challenge(name)
        order = np.random.default_rng(391).permutation(X.shape[1])
        yield (name, np.ascontiguousarray(X[:, order]), y,
               np.ascontiguousarray(Xt[:, order]), truth, "classification")
    for name, seed in (("redundant_regression", 810049), ("dense_regression", 968357)):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(2400, 128))
        if name == "redundant_regression":
            latent = X[:, :8].copy()
            X[:, 8:48] = np.tile(latent, 5) + .1 * X[:, 8:48]
            signal = latent.sum(axis=1) / np.sqrt(8)
        else:
            signal = X.sum(axis=1) / np.sqrt(128)
        y = signal + .3 * rng.normal(size=len(X))
        X = np.ascontiguousarray(X[:, rng.permutation(X.shape[1])], dtype=np.float32)
        yield name, X[:1600], y[:1600], X[1600:], y[1600:], "regression"


def select(method, X, y, task, seed):
    if method == "full":
        return np.arange(X.shape[1])
    name, suffix = method.split("_")
    width = reduced_width(X.shape[1], FRACTIONS[suffix])
    if name == "moment":
        return moment_top_from_training(X, y, width, task=task, random_state=seed)[0]
    return choose(name.lower(), X, y, task, seed, width)


def run(out, verify):
    files = (Path(__file__), ROOT/"src/minppi_mofs/moment_minpi.py",
             ROOT/"src/minppi_mofs/sketch_minpi.py", ROOT/"src/minppi_mofs/utility_minpi.py",
             ROOT/"experiments/minpi_study_external.py", ROOT/"experiments/minpi_study.py",
             ROOT/"experiments/compare_predictive_minpi.py", ROOT/"experiments/stable_minpi_stress.py",
             ROOT/"experiments/fast_minpi_five_source.py", ROOT/"configs/minpi-study.json")
    protocol = dict(seeds=SEEDS, real_source_sha256={name: spec[1] for name, spec in FILES.items()},
                    fractions=FRACTIONS, methods=METHODS, metric="balanced accuracy or R-squared",
                    primary="Half width; 75% included after the 69 runs and before the transfer evaluation",
                    repeats=5, threads=1, python=platform.python_version(), numpy=np.__version__,
                    sklearn=sklearn.__version__, scipy=scipy.__version__,
                    files={str(path.relative_to(ROOT)): file_hash(path) for path in files})
    manifest = out / "protocol.json"
    if verify:
        if json.loads(manifest.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("The transfer protocol changed before verification")
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    trials, audits = [], []
    for panel_id, (name, X, y, Xt, truth, task) in enumerate(panels()):
        trace = out / f"{name}.npz"
        arrays = {"truth": truth, "input_hashes": np.array([digest(a) for a in (X, y, Xt, truth)])}
        for seed in SEEDS:
            order = np.random.default_rng([seed, panel_id]).permutation(METHODS)
            scores = {}
            for r in range(-1, 0 if verify else protocol["repeats"]):
                for position, method in enumerate(np.roll(order, r)):
                    start = perf_counter_ns()
                    cols = select(method, X, y, task, seed)
                    selected = perf_counter_ns()
                    pred = prediction(X, y, Xt, cols, task, seed)
                    end = perf_counter_ns()
                    if r == -1:
                        check_columns(cols, X.shape[1])
                        assert len(cols) == (X.shape[1] if method == "full" else
                                             reduced_width(X.shape[1], FRACTIONS[method.split("_")[1]]))
                        arrays[f"{method}_{seed}_cols"] = cols
                        arrays[f"{method}_{seed}_pred"] = pred
                        scores[method] = metric(truth, pred, task)
                        if verify:
                            audits.append(dict(dataset=name, seed=seed, method=method,
                                               support_sha256=digest(cols), prediction_sha256=digest(pred)))
                    else:
                        np.testing.assert_array_equal(cols, arrays[f"{method}_{seed}_cols"])
                        np.testing.assert_array_equal(pred, arrays[f"{method}_{seed}_pred"])
                        trials.append(dict(dataset=name, fold=1, seed=seed, method=method, repeat=r,
                                           position=position, width=len(cols), p=X.shape[1], score=scores[method],
                                           select_ms=(selected-start)/1e6, refit_ms=(end-selected)/1e6,
                                           total_ms=(end-start)/1e6))
        if verify:
            with np.load(trace) as original:
                for key, array in arrays.items():
                    np.testing.assert_array_equal(array, original[key])
        else:
            np.savez_compressed(trace, **arrays)
        print(name, "verified" if verify else "measured", flush=True)
    if verify:
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv", index=False)
    else:
        summarize_timings(out, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
