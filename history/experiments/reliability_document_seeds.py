"""Two-pass post-development seed replay for one-sketch directional minPI."""
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
from experiments.robust_document_seeds import CASE, INPUTS, PREVIOUS, ROOT, SEEDS
from minppi_mofs.fast_document_vote_minpi import library_path
from minppi_mofs.reliability_document_minpi import reliability_document_top_from_training


ARMS = ("reliability_vote", "fast_vote", "full", "F_50", "chi_50", "random_50",
        "F_match", "pooled_match", "random_match")
SCORE_EPS = 1e-12


def select(method, X, y, seed, width):
    if method == "reliability_vote":
        return reliability_document_top_from_training(X, y, random_state=seed, backend="native")[0]
    return baseline_select(method, X, y, seed, width)


def measure(X, y, Xt, truth, meta, output, verify, pass_id):
    key = {name: meta[name] for name in CASE}
    seed = key["seed"]
    filename = f"{key['dataset']}-{key['fold']}-{seed}.npz"
    candidate, native = reliability_document_top_from_training(X, y, random_state=seed, backend="native")
    reference, plain = reliability_document_top_from_training(X, y, random_state=seed, backend="numpy")
    np.testing.assert_array_equal(candidate, reference)
    for field in ("scores", "zero_peer_scores", "one_peer_scores", "directional_endpoints"):
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
    order = np.random.default_rng([237519, pass_id, key["fold"], seed]).permutation(ARMS)
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
    vote = cases[cases.method == "reliability_vote"].set_index(CASE).copy()
    vote["within"] = vote.gap >= -.02 - SCORE_EPS
    vote["joint"] = (vote.width < vote.p) & vote.within & (vote.ratio < 1)
    vote["joint_10pct"] = vote.joint & (vote.ratio <= .9)
    for method in ARMS[1:]:
        baseline = cases[cases.method == method].set_index(CASE)
        delta = vote.score - baseline.score
        vote[method + "_score_delta"] = delta
        vote[method + "_score_win"] = delta > SCORE_EPS
        vote[method + "_score_tie"] = np.abs(delta) <= SCORE_EPS
        vote[method + "_score_loss"] = delta < -SCORE_EPS
        vote[method + "_time_ratio"] = vote.total_ms / baseline.total_ms
        vote[method + "_width_ratio"] = vote.width / baseline.width
        vote[method + "_joint"] = (
            vote.joint & (vote.width <= baseline.width) &
            (delta >= -.02 - SCORE_EPS) & (vote[method + "_time_ratio"] < 1)
        )
    previous = cases[cases.method == "fast_vote"].set_index(CASE)
    vote["old_score"] = previous.score
    vote["old_width"] = previous.width
    vote["old_joint"] = (
        (previous.width < previous.p) & (previous.gap >= -.02 - SCORE_EPS) & (previous.ratio < 1)
    )
    vote["independent_baselines"] = vote.F_50_joint & vote.chi_50_joint & vote.random_50_joint
    vote["matched_supervised"] = vote.F_match_joint & vote.pooled_match_joint
    vote["matched_baselines"] = vote.matched_supervised & vote.random_match_joint
    vote.to_csv(output / "comparisons.csv")
    aggregation = dict(
        cases=("score", "size"), min_width=("width", "min"), max_width=("width", "max"),
        min_score=("score", "min"), max_score=("score", "max"),
        old_min_score=("old_score", "min"), old_max_score=("old_score", "max"),
        quality=("within", "sum"), joint=("joint", "sum"), joint10=("joint_10pct", "sum"),
        old_joint=("old_joint", "sum"), relative_old=("fast_vote_joint", "sum"),
        old_score_wins=("fast_vote_score_win", "sum"), old_score_ties=("fast_vote_score_tie", "sum"),
        old_score_losses=("fast_vote_score_loss", "sum"),
        independent=("independent_baselines", "sum"), matched=("matched_baselines", "sum"),
        matched_supervised=("matched_supervised", "sum"), F_half=("F_50_joint", "sum"),
        F_match=("F_match_joint", "sum"), random_match=("random_match_joint", "sum"),
        pooled_match=("pooled_match_joint", "sum"),
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


def run(output, verify, pass_id):
    files = [
        Path(__file__), Path(original.__file__), ROOT / "experiments/seed_document_spam.py",
        ROOT / "experiments/robust_document_seeds.py", INPUTS / "panels.csv",
        PREVIOUS / "protocol.json", library_path(),
        ROOT / "experiments/fast_document_vote_benchmark.py", ROOT / "experiments/minpi_constant_transfer.py",
        ROOT / "experiments/fast_minpi_five_source.py", ROOT / "experiments/minpi_study.py",
        ROOT / "src/minppi_mofs/reliability_document_minpi.py", ROOT / "src/minppi_mofs/document_vote_kernel.c",
        ROOT / "src/minppi_mofs/fast_document_vote_minpi.py", ROOT / "src/minppi_mofs/document_vote_minpi.py",
        ROOT / "src/minppi_mofs/signal_vote_minpi.py", ROOT / "configs/minpi-study.json",
    ]
    protocol = dict(
        seeds=list(SEEDS), methods=list(ARMS), repeats=len(ARMS), sources=original.ASSETS,
        pass_id=pass_id, score_tolerance=.02, score_epsilon=SCORE_EPS,
        file_sha256={file.relative_to(ROOT).as_posix(): original.file_hash(file) for file in files},
        archived_outputs={file.name: original.file_hash(file) for file in sorted(PREVIOUS.glob("*.npz"))},
        interpretation="Post-development replay of all eight archived spam folds and all three seeds. "
                       "These folds and seed outputs were seen in earlier median-three experiments. "
                       "This is a sensitivity replay, not new external evidence.",
        test="One frozen directional-minimum candidate; no parameter choices by this replay. "
             "Exact native/NumPy score and support equality; exact original selector output replay. "
             "Same fold, training vocabulary and refit; nine balanced timing rotations. "
             "Genuine reduction, BA tolerance 0.02, inclusive selection-to-prediction time. "
             "Report old, full, three half-width and three candidate-width comparisons separately. "
             "Two timing passes have distinct rotation seeds and require paired configuration intersection.",
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
            timed, checked = measure(X, y, Xt, truth, dict(meta, seed=seed), output, verify, pass_id)
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
    parser.add_argument("--pass-id", choices=(1, 2), type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output.resolve(), args.verify, args.pass_id)
