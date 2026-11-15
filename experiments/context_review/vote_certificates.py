"""Reconstruct minimum-vote decisions and the shortest fixed-prefix proof of each choice."""
from contextlib import ExitStack
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import matched_stopping as prior
from experiments.context_review.forest_replay import read_member
from minppi_mofs.minimum_vote import minimum_vote_winner


DEST = prior.DEST.parent / "vote_certificates"
KEY = ["panel", "dataset", "fold", "design", "draw"]


def prefix_minimum(counts, panels):
    """Shortest total prefix certifying the first true winner, given the full matrix.

    This hindsight cost counts candidate/context queries, not unique masks
    or seconds. The only free information about an unread count is 0..panels.
    """
    if not len(counts):
        return 0
    scores = counts.min(axis=1)
    winner = int(scores.argmax())
    score = scores[winner]
    # Prefix zero already bounds a later row by the number of panels.
    bounds = np.column_stack([np.full(len(counts), panels), np.minimum.accumulate(counts, axis=1)])
    cost = counts.shape[1]
    for i, row in enumerate(bounds):
        if i != winner:
            enough = row < score if i < winner else row <= score
            cost += int(np.flatnonzero(enough)[0])
    if winner == 0 and all(scores[1:] == 0):
        # Even an unread first row beats (or ties) all-zero rivals.
        no_winner_reads = sum(int(np.flatnonzero(row == 0)[0]) for row in bounds[1:])
        cost = min(cost, no_winner_reads)
    return cost


def evaluate_slate(slate, tensor):
    ids = list(slate.indices)
    restorations = slate.family
    columns = range(len(restorations))
    tolerances = np.array([.05 if name == "ap" else .03 for name in slate.metrics])
    if ids:
        reference = tensor[0]
        assert np.isfinite(tensor).all()
        approved = reference - tensor[ids] <= tolerances * abs(reference) + 1e-12
        votes = np.all(approved, axis=-1).sum(axis=-1)
    else:
        votes = np.empty((0, len(restorations)), dtype=int)
    positions = {candidate: i for i, candidate in enumerate(ids)}
    expected = ids[int(votes.min(axis=1).argmax())] if ids else 0
    assert expected == slate.winners["vote"]
    optimum = prefix_minimum(votes, 4)
    exceptional = bool(ids and expected == ids[0] and np.all(votes[1:].min(axis=1) == 0))
    result = dict(winner=expected, candidates=len(ids), contexts=len(columns),
                  prefix_minimum=optimum, first_and_zero_rivals=exceptional)
    for name in ("depth", "best"):
        read = set()
        masks = set()

        def count(candidate, context):
            key = (candidate, context)
            assert key not in read
            read.add(key)
            masks.update((slate.slates[candidate] | restorations[context], slate.slates[0] | restorations[context]))
            return int(votes[positions[candidate], context])

        if name == "depth":
            nominee = prior.choose(ids, columns, lambda i, e: (count(i, e),), "vote", True)
            old_reads = read
        else:
            nominee = minimum_vote_winner(ids, columns, count) if ids else None
            nominee = 0 if nominee is None else nominee
            assert read <= old_reads
        assert nominee == expected and len(read) >= optimum
        result[name + "_queries"] = len(read)
        result[name + "_masks"] = len(masks)
    if not exceptional:
        assert result["best_queries"] == optimum
    return result


def replay():
    rows = []
    with ExitStack() as stack:
        archives = {p: stack.enter_context(ZipFile(p / "predictions.zip"))
                    for p in (prior.forest_predict.DEST, prior.neighborhood_replicates.DEST)}
        for key, _, _, slates in prior.folds():
            panel, dataset, fold = key.split("/")
            for directory, meta, slate in slates:
                with read_member(archives[directory], key, meta["member_sha"]) as data:
                    result = evaluate_slate(slate, data["tensor"])
                rows.append(dict(panel=panel, dataset=dataset, fold=int(fold.split("-")[1]),
                                 design=slate.design, draw=slate.draw, **result))
    frame = pd.DataFrame(rows)
    archived = pd.read_csv(prior.DEST / "folds.csv").query("pruned and method == 'vote'")
    joined = frame.merge(archived, on=KEY, validate="one_to_one", suffixes=("", "_old"))
    assert len(frame) == len(archived) == len(joined) == 184
    assert (joined.winner == joined.winner_old).all() and (joined.depth_masks == joined.masks).all()
    summary = frame.groupby(["design", "panel"], as_index=False).agg(
        slates=("winner", "size"), depth_queries=("depth_queries", "sum"), best_queries=("best_queries", "sum"),
        depth_masks=("depth_masks", "sum"), best_masks=("best_masks", "sum"), prefix_minimum=("prefix_minimum", "sum"))
    DEST.mkdir(exist_ok=True, parents=True)
    frame.to_csv(DEST / "folds.csv", index=False)
    summary.to_csv(DEST / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"{sum(frame.best_queries == frame.prefix_minimum)} of {len(frame)} attain their prefix minimum.")


if __name__ == "__main__":
    replay()
