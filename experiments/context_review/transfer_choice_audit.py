"""Independently verify all learned choices and held-source net scores using fractions."""
from collections import defaultdict
import csv
from fractions import Fraction

from experiments.context_review.decision_opportunities import EVIDENCE


def read(name):
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify():
    outcomes = defaultdict(list)
    for row in read("within_tier/policies.csv"):
        method = "original" if row["method"] == "first" else row["method"]
        outcomes[row["pipeline"], row["panel"], row["dataset"], method].append(row)
    for row in read("forest_cross/policies.csv"):
        if row["voter"] == row["receiver"] == "forest" and row["method"] == "point":
            outcomes["gain", row["panel"], row["dataset"], "slate_point"].append(row)
    for row in read("neighborhood_replicates/policies.csv"):
        if row["delivery"] == "guarded" and row["method"] == "point":
            outcomes["anova", row["panel"], row["dataset"], "slate_point"].append(row)

    tolerance = Fraction(1, 10**12)
    tier = ["original", "uniform", "point", "full"]
    expanded = ["slate_point", *tier]
    source = {}
    for key, rows in outcomes.items():
        folds = {r["fold"] for r in rows}
        assert all(sum(r["fold"] == f for r in rows) == (3 if key[0] == "anova" else 1) for f in folds)
        source[key] = [sum(Fraction(r[m]) for r in rows) / len(rows) for m in ("u", "breach")]
    held = read("transfer_choice/held.csv")
    for row in held:
        pipeline, panel, task = row["pipeline"], row["panel"], row["dataset"]
        penalty = Fraction(row["penalty"])
        training = {name for design, group, name, _ in source if (design, group) == (pipeline, panel)} - {task}
        assert len(training) == int(row["training_sources"])

        def net(name, method):
            u, breach = source[pipeline, panel, name, method]
            return u - penalty * breach

        chosen = row["chosen"]
        if row["policy"].startswith("learn_"):
            options = tier if row["policy"] == "learn_tier" else expanded
            scores = {m: sum(net(t, m) for t in training) / len(training) for m in options}
            assert next(m for m in options if max(scores.values()) - scores[m] <= tolerance) == chosen
            assert abs(scores[chosen] - Fraction(row["training_net"])) < tolerance
        else:
            assert chosen == row["policy"]
        assert abs(net(task, chosen) - Fraction(row["net"])) < tolerance
        assert abs(net(task, chosen) - net(task, "slate_point") - Fraction(row["dnet_vs_point"])) < tolerance
    assert len(held) == 672
    print("Verified 672 held-source net scores and 192 learned choices with rational arithmetic.")


if __name__ == "__main__":
    verify()
