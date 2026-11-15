"""Time three nomination rules on the frozen forest slates and verify their returns."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import matched_stopping as original
from experiments.context_review import matched_stopping_audit as audit
from experiments.context_review.timing_process import bind_logical_cpu
from minppi_mofs import minimum_vote


ROOT = original.bg.ROOT
DEST = ROOT / "thesis-paper/evidence/vote_certificates"
SLATE = ["panel", "dataset", "fold", "design", "draw"]
POLICIES = ("depth", "best", "point")


def sources():
    files = [
        Path(__file__), Path(minimum_vote.__file__), Path(original.__file__),
        Path(original.primary.__file__), Path(original.forest_predict.__file__),
        ROOT / "experiments/context_review/timing_process.py",
        ROOT / "experiments/test/minpi_convergence_metrics.py",
        ROOT / "src/minppi_mofs/piv3_compact_witness.py",
        ROOT / "src/minppi_mofs/context_design.py", original.DEST / "folds.csv",
    ]
    for release in (original.forest_predict, original.filter_neighborhood, original.neighborhood_replicates):
        files.append(release.DEST / "prediction-lock.json")
    return {p.relative_to(ROOT).as_posix(): audit.text_digest(p) for p in files}


def nomination(slate, model, xt, xp, labels, seed, rare, policy):
    point = policy == "point"
    metric = "point" if point else "vote"
    start = perf_counter()
    ballot = original.primary.build_ballot(model, xt, xp, labels, seed, slate.slates[0], rare, point=point)
    evaluate = slate.evaluate(metric, ballot.score)
    if policy == "best":
        winner = minimum_vote.minimum_vote_winner(
            slate.indices, slate.family, lambda candidate, context: evaluate(candidate, context)[0])
        if winner is None:
            winner = 0
    else:
        winner = original.choose(slate.indices, slate.path(metric), evaluate, metric, prune=not point)
    elapsed = perf_counter() - start
    original.check_locked(slate, metric, winner)
    return winner, len(ballot.scores), elapsed


def orders(fold_index, slate_index):
    generator = np.random.default_rng(np.random.SeedSequence([51873, fold_index, slate_index]))
    cycle = list(generator.permutation(POLICIES))
    return [cycle[r:] + cycle[:r] for r in range(3)]


def measure(directory):
    directory.mkdir(exist_ok=True, parents=True)
    if any((directory / name).exists() for name in ("timings.csv", "timing-receipt.json")):
        raise FileExistsError("Choose a fresh --output for new timing observations")
    source_hashes = sources()
    cpu = bind_logical_cpu()
    started = perf_counter()
    rows = []
    with threadpool_limits(limits=1):
        for fold_index, (key, src, rec, slates) in enumerate(original.folds()):
            assert original.bg.digest(src / "inputs.npz") == rec["inputs_sha"]
            with np.load(src / "inputs.npz") as data:
                x, y, train, prop = [data[k] for k in ("x", "y", "train", "proposal")]
            xt, xp = original.forest_predict.transform(x, train, x[prop])
            seed = 20421 + rec["fold"]
            model = original.forest_predict.new_forest(seed).fit(xt, y[train])
            labels = y[prop]
            for slate_index, (_, _, slate) in enumerate(slates):
                for repeat, cycle in enumerate(orders(fold_index, slate_index), 1):
                    for position, policy in enumerate(cycle, 1):
                        winner, masks, seconds = nomination(
                            slate, model, xt, xp, labels, seed, "ap" in rec["metrics"], policy)
                        rows.append(dict(panel=rec["panel"], dataset=rec["dataset"], fold=rec["fold"],
                                         design=slate.design, draw=slate.draw, policy=policy, repeat=repeat,
                                         position=position, winner=winner, masks=masks, nomination_s=seconds))
            elapsed = perf_counter() - started
            print(f"[{fold_index+1}/46] {key}, elapsed {elapsed:.1f}s, "
                  f"ETA {elapsed*(45-fold_index)/(fold_index+1):.1f}s", flush=True)
    if sources() != source_hashes:
        raise ValueError("The selector, timing source or input locks changed during the experiment")
    timings = directory / "timings.csv"
    pd.DataFrame(rows).to_csv(timings, index=False, lineterminator="\n")
    environment = audit.environment()
    environment["cpu_affinity"] = "not pinned" if cpu is None else f"logical CPU {cpu}"
    original.bg.store(directory / "timing-receipt.json", dict(
        protocol="51873, fold and slate index select the first order; rotate by one for each repetition",
        slates=184, folds=46, repetitions=3, live_nominations=len(rows), environment=environment,
        source_and_inputs_utf8_lf_sha256=source_hashes, timings_utf8_lf_sha256=audit.text_digest(timings)))


def verify(directory=DEST):
    receipt = original.bg.read(directory / "timing-receipt.json")
    assert receipt["source_and_inputs_utf8_lf_sha256"] == sources()
    assert receipt["timings_utf8_lf_sha256"] == audit.text_digest(directory / "timings.csv")
    live = pd.read_csv(directory / "timings.csv", float_precision="round_trip")
    assert len(live) == receipt["live_nominations"] == 184*3*3
    assert live.nomination_s.gt(0).all() and not live.duplicated(SLATE + ["policy", "repeat"]).any()
    order = original.bg.read(original.forest_predict.DEST / "prediction-lock.json")["folds"]
    indices = {key: i for i, key in enumerate(order)}
    for (panel, dataset, fold, design, draw, repeat), block in live.groupby(SLATE + ["repeat"]):
        index = indices[f"{panel}/{dataset}/fold-{fold}"]
        cycle = orders(index, 0 if design == "gain" else draw)[repeat-1]
        assert block.sort_values("position").policy.tolist() == cycle
        assert sorted(block.position) == [1, 2, 3]
    records = pd.read_csv(original.DEST / "folds.csv")
    old = records.query("(method == 'vote' and pruned) or method == 'point'").copy()
    old["policy"] = old.method.replace({"vote": "depth"})
    paired = pd.read_csv(DEST / "folds.csv")
    best = paired.rename(columns={"best_masks": "masks"}).assign(policy="best")
    expected = pd.concat([old, best])[SLATE + ["policy", "winner", "masks"]]
    observed = live.merge(expected, on=SLATE + ["policy"], validate="many_to_one", suffixes=("", "_expected"))
    assert len(observed) == len(live) and len(expected) == 3*184
    for field in ("winner", "masks"):
        np.testing.assert_array_equal(observed[field], observed[field + "_expected"])
    medians = live.groupby(SLATE + ["policy"])[["masks", "nomination_s"]].median()
    summary = medians.groupby(["panel", "design", "draw", "policy"]).sum().reset_index()
    summary["masks"] = summary.masks.astype(int)
    summary.to_csv(directory / "timing-summary.csv", index=False, lineterminator="\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("time", "verify"))
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    if args.action == "time":
        measure(args.output)
    else:
        verify(args.output)
