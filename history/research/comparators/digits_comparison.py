"""Matched-width Digits benchmark and cross-process replay for the discrete MID baseline."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments import fast_minpi_five_source as five
from experiments.grouped_solver_cost_audit import fit, select
from research.algorithms.witness_exchange import exchange_top_from_training
from research.comparators.mrmr_mid import mrmr_mid_order


ARMS = ("full", "F50", "random50", "grouped50", "witness_exchange", "mrmr_mid")
HASH_FILES = (
    "research/comparators/digits_comparison.py", "research/comparators/mrmr_mid.py",
    "research/algorithms/witness_exchange.py", "experiments/grouped_solver_cost_audit.py",
    "experiments/compare_predictive_minpi.py", "src/minppi_mofs/grouped_minpi.py",
    "src/minppi_mofs/word_minpi.py", "src/minppi_mofs/exact_minpi.py",
    "src/minppi_mofs/certified_minpi.py", "src/minppi_mofs/integer_minpi.py",
    "src/minppi_mofs/piv2.py", "experiments/stable_minpi_stress.py",
    "experiments/minpi_study.py", "configs/minpi-study.json",
)


@dataclass
class DigitsFold:
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    seed: int = 1129
    task: str = "classification"


def pipeline(panel, method):
    k = panel.X.shape[1] // 2
    start = perf_counter_ns()
    extra = {}
    if method == "mrmr_mid":
        order, _ = mrmr_mid_order(panel.X, panel.y, k)
        cols = np.sort(order)
        extra["order"] = order
    elif method == "witness_exchange":
        cols, info = exchange_top_from_training(panel.X, panel.y, k, random_state=panel.seed)
        extra["initial_support"] = info["initial_support"]
    elif method in ("full", "F50", "random50", "grouped50"):
        cols = select(method, panel)
    else:
        raise ValueError(f"Unknown Digits comparison method: {method!r}. Expected one of {ARMS}.")
    selected = perf_counter_ns()
    pred, decision, parts = fit(panel, cols, "cholesky")
    end = perf_counter_ns()
    return {"support": cols, "pred": pred, "decision": decision, **extra}, {
        "select_ms": (selected-start)/1e6, "total_ms": (end-start)/1e6, **parts}


def run(output, pass_id):
    X, y, task, groups, source, duplicates = five.inputs("digits")
    assert np.array_equal(np.unique(X), np.arange(17)) and task == "classification"
    inputs = dict(X=five.digest(X), y=five.digest(y), source=source, groups=None)
    archive = json.loads((five.ARCHIVE/"protocol.json").read_text(encoding="utf8"))
    assert groups is None and archive["input_hashes"]["digits"] == inputs
    # A new directory prevents a rerun from overwriting measured times.
    output.mkdir(parents=True, exist_ok=False)
    state = dict(inputs=inputs, split_hashes=[], numpy=np.__version__, sklearn=sklearn.__version__,
                 seed=1129, methods=ARMS, paper="10.1109/TPAMI.2005.159 Eq. (7), discrete",
                 threads=1, repeats=len(ARMS), pass_id=pass_id, solver="cholesky", alpha=1.0,
                 pipeline="fresh train-only selection + slicing + scaling, Ridge fit + test prediction",
                 source_sha256={path: five.file_hash(five.ROOT/path) for path in HASH_FILES})
    costs = []
    for fold, (tr, te) in enumerate(five.folds(X, y, task, groups), 1):
        split_file = five.ARCHIVE/f"digits-{fold}-split.npz"
        with np.load(split_file) as archived:
            np.testing.assert_array_equal(archived["train"], tr)
            np.testing.assert_array_equal(archived["test"], te)
            np.testing.assert_array_equal(archived["gold"], y[te])
        assert not np.intersect1d(duplicates[tr], duplicates[te]).size
        state["split_hashes"].append(five.file_hash(split_file))
        panel = DigitsFold(X[tr], y[tr], X[te])
        truth = y[te]
        # One warm-up per arm. The same routine runs in every timed call.
        reference = {method: pipeline(panel, method)[0] for method in ARMS}
        np.testing.assert_array_equal(reference["witness_exchange"]["initial_support"],
                                      reference["grouped50"]["support"])
        rng = np.random.default_rng([1129, pass_id, fold])
        order = rng.permutation(ARMS)
        for repeat in range(len(ARMS)):
            for position, method in enumerate(np.roll(order, repeat)):
                arrays, times = pipeline(panel, method)
                for field, value in arrays.items():
                    np.testing.assert_array_equal(value, reference[method][field])
                assert len(arrays["support"]) == (64 if method == "full" else 32)
                costs.append(dict(fold=fold, method=method, repeat=repeat, position=position,
                                  k=len(arrays["support"]), p=64,
                                  ba=balanced_accuracy_score(truth, arrays["pred"]), **times))
        saved = {f"{method}_{field}": values for method, fields in reference.items()
                 for field, values in fields.items()}
        np.savez_compressed(output/f"fold-{fold}.npz", train=tr, test=te, truth=truth, **saved)
    (output/"protocol.json").write_text(json.dumps(state, indent=2), encoding="utf8")
    pd.DataFrame(costs).to_csv(output/"trials.csv", index=False)


def compare(first, second):
    protocol1, protocol2 = [json.loads((x/"protocol.json").read_text(encoding="utf8")) for x in (first, second)]
    assert protocol1.pop("pass_id") == 1 and protocol2.pop("pass_id") == 2
    assert protocol1 == protocol2
    stats = []
    for directory in (first, second):
        raw = pd.read_csv(directory/"trials.csv")
        assert len(raw) == 3*len(ARMS)**2
        groups = raw.groupby(["fold", "method"])
        assert (groups.position.nunique() == len(ARMS)).all() and (groups.ba.nunique() == 1).all()
        stats.append(groups[["k", "p", "ba", "select_ms", "total_ms"]].median())
    summary = stats[0][["k","p","ba"]].copy()
    np.testing.assert_allclose(summary, stats[1][summary.columns], rtol=0, atol=0)
    for metric in ("select_ms", "total_ms"):
        summary[metric] = np.maximum(stats[0][metric], stats[1][metric])
    ratios = [s.total_ms / s.total_ms.xs("full", level="method").reindex(
        s.index.get_level_values("fold")).to_numpy() for s in stats]
    summary["max_paired_ratio_to_full"] = np.maximum(*ratios)
    array_count = 0
    for fold in (1, 2, 3):
        with np.load(first/f"fold-{fold}.npz") as left, np.load(second/f"fold-{fold}.npz") as right:
            assert set(left.files) == set(right.files)
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
                array_count += 1
            old, new = left["grouped50_support"], left["witness_exchange_support"]
            print(f"Fold {fold} removed {np.setdiff1d(old,new).tolist()} and added {np.setdiff1d(new,old).tolist()}")
    print(f"{array_count} arrays agree exactly across fresh processes")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    measure = command.add_parser("run")
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--pass-id", type=int, choices=(1, 2), required=True)
    audit = command.add_parser("compare")
    audit.add_argument("first", type=Path)
    audit.add_argument("second", type=Path)
    audit.add_argument("--save", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "run":
            run(args.output, args.pass_id)
        else:
            result = compare(args.first, args.second)
            if args.save:
                result.to_csv(args.save, mode="x")
            print(result.to_string())
