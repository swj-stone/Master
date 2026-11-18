"""Development-only replay of two early stopping rules for the ordinary PIv1 vote search."""

import argparse
import csv
import io
import json
from math import ceil, sqrt
from pathlib import Path

import numpy as np

from minppi_mofs.piv1_search import search_subsets


TOLERANCE = 1e-14


class _StopSearch(Exception):
    """Exit the parent proposal generator before it requests another real fit."""


class _PrefixMonitor:
    def __init__(self, votes, consistent, protect_refinement):
        votes = np.asarray(votes, dtype=float)
        consistent = np.asarray(consistent, dtype=bool)
        if (votes.ndim != 1 or not len(votes) or votes.shape != consistent.shape
                or not np.isfinite(votes).all() or (np.abs(votes) > 1.).any()):
            raise ValueError("Expected aligned, finite net votes and consistent-negative flags")
        width = len(votes)
        self.negative_count = int(((votes < 0.) & consistent).sum())
        self.budget = min(24, max(6, ceil(2.4 * sqrt(width))), 2**min(width, 5) - 1)
        self.earliest = self.budget - (self.budget * self.negative_count) // (2 * width)
        self.recipe = "vote_racing"
        if protect_refinement:
            # Preserve the width-refinement portion of the ordinary proposal budget.
            self.earliest = max(self.earliest, self.budget - max(1, round(self.budget / 4)))
            self.recipe = "vote_racing_refinement"
        self.patience = ceil(sqrt(self.budget))
        self.best = -np.inf
        self.misses = 0
        self.trials = []

    def observe(self, trial):
        value = trial["utility"]
        if not np.isfinite(value):
            raise ValueError("A PIv1 prefix needs finite pooled validation utility")
        self.misses = 0 if value > self.best + TOLERANCE else self.misses + 1
        self.best = max(self.best, value)
        self.trials.append(trial)
        return len(self.trials) >= self.earliest and self.misses >= self.patience

    def result(self):
        winner = min((t for t in self.trials if t["utility"] >= self.best - TOLERANCE),
                     key=lambda t: (len(t["indices"]), tuple(t["indices"])))
        return dict(recipe=self.recipe, selected=tuple(winner["indices"]), utility=winner["utility"],
                    retained=len(winner["indices"]), winner_source=winner["source"], trials=self.trials,
                    evaluated=len(self.trials), early_stopped=len(self.trials) < self.budget,
                    base_budget=self.budget, budget=self.budget, earliest_stop=self.earliest,
                    patience=self.patience, negative_count=self.negative_count)


def replay_vote_prefix(trials, vote_scores, *, consistent_negative, protect_refinement=False):
    """Replay solely from the validation-time trials of the ordinary vote recipe."""
    monitor = _PrefixMonitor(vote_scores, consistent_negative, protect_refinement)
    if len(trials) != monitor.budget:
        raise ValueError("Replay requires a complete, ordered primary vote trajectory")
    for trial in trials:
        if monitor.observe(trial):
            break
    return monitor.result()


def search_vote_prefix(vote_scores, shap_importance, gain_importance, evaluate, *,
                       consistent_negative, protect_refinement=False):
    """Follow exactly the ordinary vote proposal order until the stopping condition.

    A callback requesting an uncached mask performs one evaluation. Interrupting
    during that callback cannot return the parent generator's branch labels. For
    such a prefix, trials are explicitly identified as coming from the ordinary
    vote trajectory; no unevaluated suffix is simulated for bookkeeping.
    """
    monitor = _PrefixMonitor(vote_scores, consistent_negative, protect_refinement)

    def tracked(indices):
        observed = np.asarray(evaluate(indices), dtype=float).reshape(-1)
        if not len(observed):
            raise ValueError("The scoring callback must return a pooled utility")
        trial = dict(indices=indices, utility=float(observed[0]), source="vote_trajectory")
        if monitor.observe(trial) and len(monitor.trials) < monitor.budget:
            raise _StopSearch
        return observed

    try:
        full = search_subsets(vote_scores, shap_importance, gain_importance, tracked, recipe="vote",
                              consistent_negative=consistent_negative)
    except _StopSearch:
        return monitor.result()
    if full["budget"] != monitor.budget or len(full["trials"]) != monitor.budget:
        raise RuntimeError("Ordinary vote search did not exhaust its planned trajectory")
    monitor.trials = full["trials"]
    return monitor.result()


def replay_run(root, output):
    """Recompute the original validation-only CSV and verify any existing copy."""
    rows = []
    for dataset in ("abalone", "spambase", "madelon", "gisette", "credit_card_fraud", "isolet"):
        arm = root / "development" / "seed-42" / dataset / "shared"
        ranking = json.loads((arm / "ranking.json").read_bytes())
        trials = json.loads((arm / "trials-vote.json").read_bytes())
        votes = ranking["votes"]["min_vote"]
        consistent = ranking["votes"]["negative_consistent"]
        history = {tuple(t["indices"]): t["utility"] for t in trials}
        highest = max(history.values())
        winner = min((mask for mask, val in history.items() if val >= highest - TOLERANCE),
                     key=lambda mask: (len(mask), mask))
        for protected in (False, True):
            options = dict(consistent_negative=consistent, protect_refinement=protected)
            replay = replay_vote_prefix(trials, votes, **options)
            online = search_vote_prefix(votes, ranking["training_mean_abs_shap"], ranking["gain"],
                                        history.__getitem__, **options)
            if ([tuple(t["indices"]) for t in replay["trials"]] != [t["indices"] for t in online["trials"]]
                    or replay["selected"] != online["selected"] or replay["utility"] != online["utility"]):
                raise ValueError(f"Live stopping and replay disagree for {dataset}")
            rows.append(dict(dataset=dataset, seed=42, metric=ranking["metric"], variant=replay["recipe"],
                             budget=replay["budget"], consistent_negative=replay["negative_count"],
                             earliest_stop=replay["earliest_stop"], checked=replay["evaluated"],
                             saved=replay["budget"] - replay["evaluated"], full_retained=len(winner),
                             retained=replay["retained"], full_val=history[winner], val=replay["utility"],
                             delta=replay["utility"] - history[winner], same_mask=winner == replay["selected"],
                             source=replay["winner_source"], selected_indices=json.dumps(replay["selected"]),
                             online_and_replay_equal=True))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8-sig")
    if output.exists():
        if output.read_bytes() != raw:
            raise ValueError(f"Existing report differs from replay: {output}")
    else:
        with output.open("xb") as handle:
            handle.write(raw)
    print(f"Verified {len(rows)} validation replays: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed PIv1 development run directory")
    parser.add_argument("--output", type=Path, help="Write a separate copy; default is <run>/racing_replay.csv")
    args = parser.parse_args()
    replay_run(args.run, args.output or args.run / "racing_replay.csv")
