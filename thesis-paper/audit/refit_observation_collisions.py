"""Count refit decisions indistinguishable to specified summaries of the archived masks."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "results/context_review"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations():
    for path in sorted((RUNS / "rule-sensitivity").glob("*/fold-*/voting.json")):
        folder = path.parent
        dataset, fold = folder.parent.name, int(folder.name.split("-")[-1])
        donor = RUNS / "refit-target" / dataset / folder.name
        stored = read(path)
        completed = read(folder / "completed.json")
        assert completed["voting"] == digest(path)
        assert stored["votes"] == digest(folder / "votes.npz")
        assert stored["donor"] == digest(donor / "selection.json")
        design = read(donor / "selection.json")
        sep = [frozenset(e) for e in design["contexts"]["separating"]]
        all_contexts = sorted(
            {frozenset(e) for family in design["contexts"].values() for e in family},
            key=lambda e: (len(e), sorted(e)),
        )
        with np.load(folder / "votes.npz") as source:
            index = {frozenset(np.flatnonzero(mask)): i for i, mask in enumerate(source["xgb_masks"])}
            scores = source["xgb_scores"]
            names = source["metrics"]
        assert set(names) == {"auc", "accuracy"} and scores.shape[1:] == (12, 2)
        anchor = frozenset(design["anchor"])
        base = scores[[index[anchor | e] for e in sep], :4]
        outcomes = completed["outcomes"]
        for i, columns in enumerate(design["supports"]):
            support = frozenset(columns)
            reduced = scores[[index[support | e] for e in sep], :4]
            ballots = (base - reduced <= 0.03 * abs(base) + 1e-12).all(axis=2)
            assert np.array_equal(ballots, design["ballots"][i]["separating"])
            recorded = scores[[index[support | e] for e in all_contexts]]
            assert np.isfinite(recorded).all()
            # Within a fold the mask order, anchor tensor, panels and scoring conventions are fixed.
            signatures = {
                "width": (),
                "worst approvals": int(ballots.sum(axis=1).min()),
                "separating ballots": ballots.tobytes(),
                "separating metrics": reduced.tobytes(),
                "all recorded metrics": recorded.tobytes(),
            }
            cohorts = [k for k, members in design["cohorts"].items() if columns in members]
            assert cohorts
            for cohort in ["union", *cohorts]:
                for target, values in outcomes.items():
                    loss = np.array(values["loss"][i])
                    fail = bool(np.any(loss > 0.03 * abs(np.array(values["reference"])) + 1e-12))
                    assert fail == values["failure"][i]
                    for summary, signature in signatures.items():
                        yield (cohort, target, summary), (dataset, fold, len(columns), signature), fail


def summarize():
    records = defaultdict(lambda: defaultdict(list))
    for design, signature, fail in observations():
        records[design][signature].append(fail)
    rows = []
    for (cohort, target, summary), groups in records.items():
        mixed = [values for values in groups.values() if 0 < sum(values) < len(values)]
        cases = sum(map(len, groups.values()))
        expected = {"union": 704, "path": 400, "uniform_deletions": 379}
        assert cases == expected[cohort]
        rows.append(dict(
            cohort=cohort, target=target, summary=summary, proposals=cases, groups=len(groups),
            mixed_groups=len(mixed), mixed_proposals=sum(map(len, mixed)),
            mixed_tasks=len({key[0] for key, values in groups.items() if 0 < sum(values) < len(values)}),
            error_floor=sum(min(len(g)-sum(g), sum(g)) for g in mixed),
            cost_floor_miss2=sum(min(len(g)-sum(g), 2*sum(g)) for g in mixed),
        ))
    with (HERE / "refit-observation-collisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["cohort"] == "union":
            print(row["target"], row["summary"], row["groups"], row["mixed_groups"],
                  row["mixed_proposals"], row["error_floor"], row["cost_floor_miss2"])


if __name__ == "__main__":
    summarize()
