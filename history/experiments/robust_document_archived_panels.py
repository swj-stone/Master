"""Compare repeated-median minPI on the original CNAE and Dexter outer partitions."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from threadpoolctl import threadpool_limits

from experiments import document_vote_archived_panels as old
from experiments.fast_minpi_five_source import summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, write_json
from minppi_mofs.fast_document_vote_minpi import library_path as old_library_path
from minppi_mofs.robust_document_minpi import library_path, robust_document_top_from_training


ARMS = ("robust", "document_vote", "varying_only", "F_match", "random_match",
        "F_50", "random_50", "full")
KEY = ["dataset", "fold", "seed"]
REFERENCE = ROOT / "results/minpi-study/document-archived-20261117a"


def protocol():
    files = [Path(__file__), Path(old.__file__), library_path(), old_library_path()]
    files.extend(ROOT / "src/minppi_mofs" / name for name in (
        "robust_document_minpi.py", "repeated_document_minpi.py", "document_repeated_kernel.c",
        "document_vote_kernel.c", "fast_document_vote_minpi.py", "document_vote_minpi.py",
        "signal_vote_minpi.py", "sketch_minpi.py", "utility_minpi.py"))
    files.extend(ROOT / "experiments" / name for name in (
        "signal_vote_benchmark.py", "fast_minpi_five_source.py", "minpi_study.py",
        "budget_minpi_deployments.py", "moment_minpi_benchmark.py", "adaptive_minpi_transfer.py",
        "compare_predictive_minpi.py", "stable_minpi_stress.py"))
    files.append(ROOT / "configs/minpi-study.json")
    return dict(
        source_sha256={f.relative_to(ROOT).as_posix(): file_hash(f) for f in files},
        reference_sha256={f.name: file_hash(f) for f in sorted(REFERENCE.glob("*.npz"))},
        methods=list(ARMS), repeats=len(ARMS), expected_cases=6, threads=1,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        protocol="Keep original dense inputs and downstream models. All screened arms pay for scanning "
                 "varying terms, and each voting arm pays for its dense-to-CSR conversion. F and random "
                 "receive the new candidate width for free. Half width uses the varying-term count. "
                 "Warm all arms, then rotate all eight methods through eight positions.",
    )


def selected(panel, arm, width):
    if arm != "robust":
        return old.selected(panel, arm, width)
    seen = np.count_nonzero(panel.X, axis=0)
    varying = np.flatnonzero((seen > 0) & (seen < len(panel.y)))
    X = sparse.csr_matrix(panel.X[:, varying])
    cols, _ = robust_document_top_from_training(X, panel.y, random_state=panel.seed, groups=panel.groups)
    return varying[cols]


def paired_baselines(output, cases):
    vote = cases[cases.method == "robust"].set_index(KEY)
    comparisons = []
    for arm, frame in cases.groupby("method"):
        control = frame.set_index(KEY)
        pair = vote[["score", "width", "total_ms", "joint"]].copy()
        pair["baseline"] = arm
        pair["score_delta"] = vote.score - control.score
        pair["width_ratio"] = vote.width / control.width
        pair["time_ratio"] = vote.total_ms / control.total_ms
        quality = vote.joint & (pair.score_delta >= -.02) & (pair.width_ratio <= 1)
        pair["relative_joint"] = quality & (pair.time_ratio < 1)
        pair["relative_joint_10pct"] = vote.joint_10pct & quality & (pair.time_ratio <= .9)
        comparisons.append(pair)
    comparison = pd.concat(comparisons).reset_index()
    comparison.to_csv(output / "paired-baselines.csv", index=False)
    comparison.groupby(["dataset", "baseline"]).agg(
        configs=("seed", "size"), relative_joint=("relative_joint", "sum"),
        relative_joint_10pct=("relative_joint_10pct", "sum"),
        median_score_delta=("score_delta", "median"), median_time_ratio=("time_ratio", "median"),
        median_width_ratio=("width_ratio", "median"),
    ).to_csv(output / "baseline-summary.csv")


def run(output, verify):
    state = protocol()
    if verify:
        assert state == json.loads((output / "protocol.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", state)
    trials, checks, metadata = [], [], []
    for suite, panel in old.cases():
        key = old.prior.case_id(panel)
        filename = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        truth = panel.gold if suite == "original" else panel.truth
        arrays = {"input_hashes": np.array([digest(x) for x in (panel.X, panel.y, panel.Xt, truth)])}
        columns, predictions, scores = {}, {}, {}
        width = len(selected(panel, "robust", None))
        for arm in ARMS:
            columns[arm] = selected(panel, arm, width)
            pred, decision = old.prior.fit(panel, columns[arm], suite)
            predictions[arm] = pred
            scores[arm] = old.prior.score(panel, pred, decision, suite)[0]
            arrays[arm + "_cols"], arrays[arm + "_pred"] = columns[arm], pred
            checks.append(dict(**key, method=arm, width=len(columns[arm]), score=scores[arm],
                               support_sha256=digest(columns[arm]), prediction_sha256=digest(pred)))
        with np.load(old.prior.REFERENCE[suite] / filename) as reference:
            np.testing.assert_array_equal(predictions["full"], reference["full_pred"])
        with np.load(REFERENCE / filename) as reference:
            for arm in ("full", "document_vote", "varying_only", "F_50", "random_50"):
                for field in ("cols", "pred"):
                    np.testing.assert_array_equal(arrays[arm + "_" + field], reference[arm + "_" + field])
        metadata.append(dict(**key, suite=suite, p=panel.X.shape[1], n_train=len(panel.y),
                             n_test=len(truth), varying=len(columns["varying_only"]),
                             old_width=len(columns["document_vote"]), new_width=width,
                             input_sha256=digest(arrays["input_hashes"])))
        if verify:
            with np.load(output / filename) as reference:
                assert set(reference.files) == set(arrays)
                for field, values in arrays.items():
                    np.testing.assert_array_equal(values, reference[field])
        else:
            np.savez_compressed(output / filename, **arrays)
            order = np.random.default_rng([33839, panel.seed, panel.fold]).permutation(ARMS)
            for repeat in range(len(ARMS)):
                for position, arm in enumerate(np.roll(order, repeat)):
                    start = perf_counter_ns()
                    subset = selected(panel, arm, width)
                    chosen = perf_counter_ns()
                    pred, _ = old.prior.fit(panel, subset, suite)
                    end = perf_counter_ns()
                    np.testing.assert_array_equal(subset, columns[arm])
                    np.testing.assert_array_equal(pred, predictions[arm])
                    trials.append(dict(**key, method=arm, repeat=repeat, position=position,
                                       width=len(subset), p=panel.X.shape[1], score=scores[arm],
                                       select_ms=(chosen-start)/1e6, refit_ms=(end-chosen)/1e6,
                                       total_ms=(end-start)/1e6))
        print("Verified" if verify else "Measured", panel.name, panel.fold, panel.seed, flush=True)
    assert len(metadata) == 6 and len(checks) == 6 * len(ARMS)
    panels = pd.DataFrame(metadata)
    assert protocol() == state
    if verify:
        pd.testing.assert_frame_equal(panels, pd.read_csv(output / "panels.csv"))
        pd.DataFrame(checks).to_csv(output / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == 6 * len(ARMS)**2
        panels.to_csv(output / "panels.csv", index=False)
        cases = summarize_timings(output, trials)
        paired_baselines(output, cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
