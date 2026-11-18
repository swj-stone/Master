"""Only the designated ISOLET/Digits first-fold development configurations.

One informal timed pass includes all calibration work. This is deliberately
not the final two-process performance protocol; it cannot establish speed.
"""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments.compare_predictive_minpi import choose
from experiments.moment_minpi_benchmark import Panel
from experiments.stable_minpi_stress import ROOT, digest, file_hash
from minppi_mofs.grouped_minpi import grouped_top_from_training
from minppi_mofs.shared_grouped_minpi import shared_grouped_top_from_training


METHODS = ("shared", "representatives_auto", "F_auto", "full", "grouped50", "F_match", "random_match")
NEW = {"shared": "pivotal", "representatives_auto": "representatives", "F_auto": "F"}


def panels():
    # This calls the SAME loaders and split constructors as the 69-case runner,
    # without iterating other sources or looking at their test outcomes.
    data = next(deployment.configurations("isolet"))
    seed, X, y, groups, Xt, gold, units, task, extra = data
    assert seed == 1129 and groups is None
    full, _, _ = deployment.prior_arrays("isolet", seed, gold, units, extra)
    yield Panel("isolet", 1, seed, X, y, Xt, gold, task, groups, units, full)
    X, y, task, groups, source, duplicates = five.inputs("digits")
    protocol = json.loads((five.ARCHIVE / "protocol.json").read_text(encoding="utf8"))
    assert dict(source=source, X=digest(X), y=digest(y),
                groups=None if groups is None else digest(groups)) == protocol["input_hashes"]["digits"]
    tr, te = next(five.folds(X, y, task, groups))
    assert not np.intersect1d(duplicates[tr], duplicates[te]).size
    with np.load(five.ARCHIVE / "digits-1-split.npz") as old:
        np.testing.assert_array_equal(tr, old["train"])
        np.testing.assert_array_equal(te, old["test"])
    with np.load(five.CONSENSUS_ARCHIVE / "digits-1.npz") as old:
        full = old["full_s1129_pred"].copy()
    assert groups is None
    yield Panel("digits", 1, 1129, X[tr], y[tr], X[te], y[te], task, None, None, full)


def select(method, panel, matched_width):
    if method in NEW:
        return shared_grouped_top_from_training(panel.X, panel.y, random_state=panel.seed,
                                                ranking=NEW[method])
    p = panel.X.shape[1]
    if method == "full":
        return np.arange(p), {}
    if method == "grouped50":
        return grouped_top_from_training(panel.X, panel.y, ceil(p/2), random_state=panel.seed)
    kind = {"F_match": "f", "random_match": "random"}[method]
    return choose(kind, panel.X, panel.y, panel.task, panel.seed, matched_width), {}


def check_calibration(panel, info):
    fit, held = info["pilot_fit_rows"], info["calibration_rows"]
    assert not np.intersect1d(fit, held).size
    assert len(fit) <= 512 and len(held) <= 256
    order = info["ranking"]
    complete = np.r_[order, np.setdiff1d(np.arange(panel.X.shape[1]), order)]
    X, Xt = panel.X[fit].astype(float), panel.X[held].astype(float)
    errors = []
    for i, width in enumerate(info["evaluated_widths"]):
        cols = complete[:width]
        scaler = StandardScaler().fit(X[:, cols])
        model = RidgeClassifier(alpha=1., solver="cholesky").fit(scaler.transform(X[:, cols]), panel.y[fit])
        prediction = model.predict(scaler.transform(Xt[:, cols]))
        score = model.decision_function(scaler.transform(Xt[:, cols]))
        np.testing.assert_array_equal(prediction, info["calibration_predictions"][i])
        np.testing.assert_allclose(score, info["calibration_decisions"][i], rtol=1e-8, atol=1e-8)
        errors.append(float(np.max(np.abs(score-info["calibration_decisions"][i]))))
    return max(errors)


def run(output, verify=False):
    sources = [Path(__file__), ROOT/"src/minppi_mofs/shared_grouped_minpi.py",
               ROOT/"src/minppi_mofs/grouped_minpi.py", ROOT/"src/minppi_mofs/certified_minpi.py",
               ROOT/"src/minppi_mofs/exact_minpi.py", ROOT/"src/minppi_mofs/piv2.py",
               ROOT/"docs/superpowers/plans/2026-11-17-minpi-combination-search.md"]
    hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sources}
    if verify:
        saved_protocol = json.loads((output/"protocol.json").read_text(encoding="utf8"))
        assert saved_protocol["source_sha256"] == hashes
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"protocol.json").write_text(json.dumps(dict(
            stage="development_only", cases=["isolet-1-1129", "digits-1-1129"],
            methods=METHODS, source_sha256=hashes, threads=1, repeats=1,
            timing="INFORMAL one warmup then fresh selection/calibration, slice, original outer fit and prediction",
            controls="Matched F/random receive shared width for free; representative/F automatic controls independently pay all calibration costs",
            pilot="fit<=512, isolated validation<=256; witnesses<=256 within pilot fit; unit-ridge Cholesky pilot",
            outer="Unchanged archived standardization and lsqr RidgeClassifier from original69",
            prediction_check="Every actual pilot-prefix decision and class checked against a direct sklearn fit outside timer",
        ), indent=2), encoding="utf8")
    records, metadata, checks = [], [], []
    for panel in panels():
        cache, information, width = {}, {}, None
        for method in METHODS:
            cols, info = select(method, panel, width)
            if method == "shared":
                width = len(cols)
            pred, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt,
                                                     cols, panel.task, panel.seed)
            score, _, _ = deployment.score_predictions(panel.name, panel.gold, pred, panel.units,
                                                        panel.task, decision)
            cache[method+"_cols"], cache[method+"_pred"] = cols, pred
            if method == "full":
                np.testing.assert_array_equal(pred, panel.historical_full)
            if method in NEW:
                error = check_calibration(panel, info)
                checks.append(dict(dataset=panel.name, method=method,
                                   prefixes=len(info["evaluated_widths"]), max_direct_error=error))
                for field, value in info.items():
                    cache[f"{method}_info_{field}"] = np.asarray(value)
            information[method] = (info, score)
        record_path = output/f"{panel.name}-1-1129.npz"
        if verify:
            with np.load(record_path) as prior:
                assert set(prior.files) == set(cache)
                for key, value in cache.items():
                    np.testing.assert_array_equal(value, prior[key], err_msg=key)
            continue
        np.savez_compressed(record_path, **cache)
        order = np.random.default_rng(panel.seed).permutation(METHODS)
        for position, method in enumerate(order):
            start = perf_counter_ns()
            cols, info = select(method, panel, width)
            selected = perf_counter_ns()
            pred, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt,
                                                     cols, panel.task, panel.seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, cache[method+"_cols"])
            np.testing.assert_array_equal(pred, cache[method+"_pred"])
            records.append(dict(dataset=panel.name, fold=1, seed=panel.seed, method=method,
                                position=position, p=panel.X.shape[1], width=len(cols),
                                score=information[method][1], select_ms=(selected-start)/1e6,
                                refit_ms=(end-selected)/1e6, total_ms=(end-start)/1e6,
                                reason=info.get("reason", "fixed_budget")))
        metadata.append(dict(dataset=panel.name, n_train=len(panel.X), n_test=len(panel.Xt),
                             p=panel.X.shape[1], X=digest(panel.X), y=digest(panel.y),
                             Xt=digest(panel.Xt), gold=digest(panel.gold)))
        print(f"Finished designated development case: {panel.name}", flush=True)
    assert hashes == {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sources}
    pd.DataFrame(checks).to_csv(output/("replay-calibration-checks.csv" if verify else "calibration-checks.csv"), index=False)
    if verify:
        (output/"fresh-process-verification.json").write_text(json.dumps(dict(
            cases=2, method_outputs=14, exact_arrays=True, source_sha256=hashes), indent=2), encoding="utf8")
    else:
        pd.DataFrame(records).to_csv(output/"cases.csv", index=False)
        pd.DataFrame(metadata).to_csv(output/"inputs.csv", index=False)
        print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
