"""Compare median-three minPI and the archived one-sketch selector on fixed new seeds."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as original
from experiments.fast_document_vote_benchmark import select as baseline_select
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.seed_document_spam import SEEDS
from minppi_mofs.fast_document_vote_minpi import library_path as original_library_path
from minppi_mofs.repeated_document_minpi import repeated_document_top_from_training
from minppi_mofs.robust_document_minpi import robust_document_top_from_training, library_path


ARMS = ("robust_vote", "fast_vote", "full", "F_50", "chi_50", "random_50",
        "F_match", "pooled_match", "random_match")
CASE = original.CASE
ROOT = original.ROOT
PREVIOUS = ROOT / "results/minpi-study/seed-document-spam-20261117a"
INPUTS = ROOT / "results/minpi-study/document-vote-20261117b"


def select(method, X, y, seed, width):
    if method == "robust_vote":
        return robust_document_top_from_training(X, y, random_state=seed, backend="native")[0]
    return baseline_select(method, X, y, seed, width)


def measure(X, y, Xt, truth, meta, output, verify):
    key = {name: meta[name] for name in CASE}
    seed = key["seed"]
    filename = f"{key['dataset']}-{key['fold']}-{seed}.npz"
    candidate, native = robust_document_top_from_training(X, y, random_state=seed, backend="native")
    reference, plain = repeated_document_top_from_training(
        X, y, random_state=seed, repeats=3, aggregation="median", backend="numpy")
    np.testing.assert_array_equal(candidate, reference)
    for field in ("scores", "zero_peer_scores", "one_peer_scores", "replicate_endpoints"):
        np.testing.assert_array_equal(native[field], plain[field])
    width = len(candidate)
    arrays, audit, scores, trials = {}, [], {}, []
    with np.load(PREVIOUS / filename) as previous:
        for method in ARMS:
            cols = select(method, X, y, seed, width)
            check_columns(cols, X.shape[1])
            if method.endswith("match"):
                assert len(cols) == width
            if method.endswith("50"):
                assert len(cols) == (X.shape[1] + 1) // 2
            pred = original.fit_predict(X, y, Xt, cols, "classification", seed, False)
            arrays[method + "_cols"], arrays[method + "_pred"] = cols, pred
            if method in ("fast_vote", "full", "F_50", "chi_50", "random_50"):
                np.testing.assert_array_equal(cols, previous[method + "_cols"])
                np.testing.assert_array_equal(pred, previous[method + "_pred"])
            scores[method] = original.balanced_accuracy_score(truth, pred)
            audit.append(dict(**key, method=method, score=scores[method], width=len(cols),
                              support_sha256=original.digest(cols), prediction_sha256=original.digest(pred)))
    if verify:
        with np.load(output / filename) as saved:
            assert set(saved) == set(arrays)
            for name, value in arrays.items():
                np.testing.assert_array_equal(value, saved[name])
        return [], audit
    np.savez_compressed(output / filename, **arrays)
    order = np.random.default_rng([536019, key["fold"], seed]).permutation(ARMS)
    for repeat in range(len(ARMS)):
        for position, method in enumerate(np.roll(order, repeat)):
            start = perf_counter_ns()
            cols = select(method, X, y, seed, width)
            selected = perf_counter_ns()
            pred = original.fit_predict(X, y, Xt, cols, "classification", seed, False)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[method + "_cols"])
            np.testing.assert_array_equal(pred, arrays[method + "_pred"])
            trials.append(dict(**key, method=method, repeat=repeat, position=position, width=len(cols),
                               p=X.shape[1], score=scores[method], select_ms=(selected-start)/1e6,
                               refit_ms=(end-selected)/1e6, total_ms=(end-start)/1e6))
    return trials, audit


def summarize(output, trials):
    cases = summarize_timings(output, trials)
    vote = cases[cases.method == "robust_vote"].set_index(CASE).copy()
    for method in ARMS[1:]:
        baseline = cases[cases.method == method].set_index(CASE)
        vote[method + "_score_delta"] = vote.score - baseline.score
        vote[method + "_time_ratio"] = vote.total_ms / baseline.total_ms
        vote[method + "_width_ratio"] = vote.width / baseline.width
        vote[method + "_joint"] = (
            vote.joint & (vote.width <= baseline.width) &
            (vote[method + "_score_delta"] >= -.02) & (vote[method + "_time_ratio"] < 1)
        )
    previous = cases[cases.method == "fast_vote"].set_index(CASE)
    vote["old_score"] = previous.score
    vote["old_width"] = previous.width
    vote["old_joint"] = previous.joint
    vote["independent_baselines"] = vote.F_50_joint & vote.chi_50_joint & vote.random_50_joint
    vote["matched_baselines"] = vote.F_match_joint & vote.pooled_match_joint & vote.random_match_joint
    vote.to_csv(output / "comparisons.csv")
    aggregation = dict(
        cases=("score", "size"), min_width=("width", "min"), max_width=("width", "max"),
        min_score=("score", "min"), max_score=("score", "max"),
        old_min_score=("old_score", "min"), old_max_score=("old_score", "max"),
        quality=("within", "sum"), joint=("joint", "sum"), joint10=("joint_10pct", "sum"),
        old_joint=("old_joint", "sum"), relative_old=("fast_vote_joint", "sum"),
        independent=("independent_baselines", "sum"), matched=("matched_baselines", "sum"),
        F_half=("F_50_joint", "sum"), F_match=("F_match_joint", "sum"),
        random_match=("random_match_joint", "sum"), pooled_match=("pooled_match_joint", "sum"),
        median_old_score_delta=("fast_vote_score_delta", "median"),
        median_old_time_ratio=("fast_vote_time_ratio", "median"),
        median_old_width_ratio=("fast_vote_width_ratio", "median"),
        median_random_score_delta=("random_match_score_delta", "median"),
        median_random_time_ratio=("random_match_time_ratio", "median"),
    )
    for keys, filename in (
        (["dataset"], "by-source"), (["dataset", "fold"], "by-fold"), (["dataset", "seed"], "by-seed"),
    ):
        table = vote.groupby(keys).agg(**aggregation)
        table["score_range"] = table.max_score - table.min_score
        table["old_score_range"] = table.old_max_score - table.old_min_score
        table.to_csv(output / (filename + ".csv"))


def run(output, verify):
    files = [
        Path(__file__), Path(original.__file__), ROOT / "experiments/seed_document_spam.py",
        INPUTS / "panels.csv", PREVIOUS / "protocol.json", library_path(), original_library_path(),
        ROOT / "experiments/fast_document_vote_benchmark.py", ROOT / "experiments/minpi_constant_transfer.py",
        ROOT / "experiments/fast_minpi_five_source.py", ROOT / "experiments/minpi_study.py",
        ROOT / "src/minppi_mofs/robust_document_minpi.py", ROOT / "src/minppi_mofs/repeated_document_minpi.py",
        ROOT / "src/minppi_mofs/document_repeated_kernel.c", ROOT / "src/minppi_mofs/document_vote_kernel.c",
        ROOT / "src/minppi_mofs/fast_document_vote_minpi.py", ROOT / "src/minppi_mofs/document_vote_minpi.py",
        ROOT / "src/minppi_mofs/signal_vote_minpi.py", ROOT / "configs/minpi-study.json",
        ROOT / "experiments/build_native_repeated_document.py",
    ]
    protocol = dict(
        seeds=list(SEEDS), methods=list(ARMS), repeats=len(ARMS), sources=original.ASSETS,
        file_sha256={file.relative_to(ROOT).as_posix(): original.file_hash(file) for file in files},
        archived_outputs={file.name: original.file_hash(file) for file in sorted(PREVIOUS.glob("*.npz"))},
        test="All eight archived spam folds and all three previously frozen new selector seeds. "
             "Median-three candidate fixed before these scores, exact original selector replay. "
             "Same fold, same training vocabulary, same refit; nine balanced timing rotations. "
             "Genuine reduction, score tolerance 0.02, inclusive selection-to-prediction time. "
             "Compare independently to old minPI, F50, chi50, random50 and candidate-matched F, "
             "marginal ridge and random. Three seeds per fold are dependent configurations.",
    )
    if verify:
        assert json.loads((output / "protocol.json").read_text()) == protocol
    else:
        output.mkdir(parents=True, exist_ok=False)
        original.write_json(output / "protocol.json", protocol)
    frozen = pd.read_csv(INPUTS / "panels.csv").set_index(["dataset", "fold"])
    trials, audits = [], []
    for X, y, Xt, truth, meta in original.panels():
        if meta["task"] != "spam":
            continue
        old = frozen.loc[(meta["dataset"], meta["fold"])]
        for field, value in meta.items():
            if field.endswith("_sha256"):
                assert value == old[field]
        for seed in SEEDS:
            timed, checked = measure(X, y, Xt, truth, dict(meta, seed=seed), output, verify)
            trials.extend(timed)
            audits.extend(checked)
        print("Verified" if verify else "Measured", meta["dataset"], meta["fold"], flush=True)
    assert len(audits) == 24 * len(ARMS)
    if verify:
        pd.DataFrame(audits).to_csv(output / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == 24 * len(ARMS)**2
        summarize(output, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output.resolve(), args.verify)
