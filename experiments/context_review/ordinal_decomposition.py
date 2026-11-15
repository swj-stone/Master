"""Post-outcome diagnostic of changing only the ordering on already scored slates.

The intermediate policy uses the full margin only among maximum-vote candidates.
Original slate order resolves the final tie for each of the three policies. A return
of zero means the anchor. Draws are averaged within folds, then folds within sources.
Only sources are resampled for the descriptive intervals. This diagnostic was chosen
after reviewing outcomes and does not provide prospective evidence for a new policy.
"""
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import forest_predict as crossed
from experiments.context_review import neighborhood_replicates as repeats
from experiments.context_review.forest_replay import bootstrap, read_member, read_metrics
from experiments.context_review.refit_opportunity import outcome


DEST = bg.ROOT / "thesis-paper/evidence/ordinal_decomposition"
LABELS = bg.ROOT / "thesis-paper/evidence/receiver/outcomes.zip"
ARMS = ("vote", "vote_slack", "slack")
LEGS = (("within_vote", "vote", "vote_slack"), ("across_votes", "vote_slack", "slack"),
        ("total", "vote", "slack"))
COUNT = ("changed_nominee", "changed_return", "u_increased", "u_decreased",
         "breach_increased", "breach_decreased")


def choices(tensor, slate, names):
    """Return support IDs in the frozen union and their ordinal scores."""
    if len(slate) == 1:
        return dict.fromkeys(ARMS, 0), {0: 0}
    atol = abs(tensor[0]) * np.array([.05 if name == "ap" else .03 for name in names])
    delta = tensor[slate[1:]] - tensor[0]
    votes = np.all(-delta <= atol + 1e-12, axis=3).sum(axis=2).min(axis=1)
    margin = ((delta + atol + 1e-12) / np.maximum(atol, 1e-12)).min(axis=3)
    full = np.sort(margin, axis=2)[:, :, 1].min(axis=1)
    np.testing.assert_array_equal(votes >= 3, full >= 0)
    top = np.flatnonzero(votes == np.max(votes))
    ids = (top[0], top[np.argmax(full[top])], np.argmax(full))
    chosen = {arm: slate[i + 1] for arm, i in zip(ARMS, ids, strict=True)}
    return chosen, dict(zip(slate[1:], votes, strict=True))


def archived(directory):
    """Use the matching stored end-to-end fold metrics as independent checks."""
    data = pd.read_csv(directory / "policies.csv")
    if directory == crossed.DEST:
        data = data.query("voter=='forest' and receiver=='forest'").assign(draw=1, delivery="guarded")
    data = data[data.method.isin(ARMS)]
    data = data.set_index(["panel", "dataset", "fold", "draw", "delivery", "method"])
    assert data.index.is_unique
    return data


def records(directory, name):
    lock = bg.read(directory / "prediction-lock.json")
    assert bg.digest(directory / "predictions.zip") == lock["archive_sha"]
    reference = archived(directory)
    results = []
    with ZipFile(directory / "predictions.zip") as predictions, ZipFile(LABELS) as labels:
        for key, rec in lock["folds"].items():
            identity = {k: rec[k] for k in ("panel", "dataset", "fold")}
            supports = rec["supports"]
            with read_member(predictions, key, rec["member_sha"]) as p, read_member(labels, key, rec["targets_sha"]) as y:
                np.testing.assert_array_equal(p["test_row"], y["test_row"])
                metrics = read_metrics(p, p["guard_y"], y["y"], len(supports), "ap" in rec["metrics"])
                tensor = p["tensor"]
            guard = [outcome(g, metrics[0][0], .03) for g, o in metrics]
            outer = [outcome(o, metrics[0][1], .03) for g, o in metrics]
            savings = [1 - len(s) / len(supports[0]) for s in supports]
            slates = rec.get("slates", [list(range(len(supports)))])
            for j, slate in enumerate(slates):
                selected, votes = choices(tensor, slate, rec["metrics"])
                frozen = rec["winners"][j] if "slates" in rec else rec["winners"]
                assert all(selected[arm] == frozen[arm] for arm in ("vote", "slack"))
                assert votes[selected["vote"]] == votes[selected["vote_slack"]]
                assert (selected["slack"] == selected["vote_slack"] or
                        votes[selected["slack"]] < votes[selected["vote_slack"]])
                deliveries = ("guarded", "unchecked") if directory == repeats.DEST else ("guarded",)
                for delivery in deliveries:
                    scores = {}
                    for arm, i in selected.items():
                        returned = i if delivery == "unchecked" or guard[i] else 0
                        u, breach = savings[returned] * outer[returned], int(not outer[returned])
                        scores[arm] = (returned, u, breach)
                        idx = (*identity.values(), j + 1, delivery, arm)
                        if arm != "vote_slack" or directory == crossed.DEST:
                            row = reference.loc[idx]
                            np.testing.assert_allclose([u, breach, savings[returned]], row[["u", "breach", "saved"]].astype(float),
                                                       atol=1e-12, rtol=0)
                            if "nominees" in row:
                                assert int(row["nominees"]) == i
                    legs = []
                    for stage, a, b in LEGS:
                        old, new = selected[a], selected[b]
                        returned_a, u_a, risk_a = scores[a]
                        returned_b, u_b, risk_b = scores[b]
                        legs.append(dict(pipeline=name, **identity, draw=j + 1, delivery=delivery, stage=stage,
                                         from_method=a, to_method=b, nominee_a=old, nominee_b=new,
                                         vote_a=votes[old], vote_b=votes[new], returned_a=returned_a, returned_b=returned_b,
                                         u_a=u_a, u_b=u_b, breach_a=risk_a, breach_b=risk_b, du=u_b-u_a,
                                         dbreach=risk_b-risk_a, changed_nominee=int(old != new),
                                         changed_return=int(returned_a != returned_b), u_increased=int(u_b > u_a),
                                         u_decreased=int(u_b < u_a), breach_increased=int(risk_b > risk_a),
                                         breach_decreased=int(risk_b < risk_a)))
                    for metric in ("du", "dbreach"):
                        np.testing.assert_allclose(legs[0][metric] + legs[1][metric], legs[2][metric], atol=1e-12, rtol=0)
                    results.extend(legs)
    return pd.DataFrame(results)


def aggregate(rows):
    by = ["pipeline", "panel", "delivery", "stage", "dataset"]
    levels = []
    repeated = rows[rows.pipeline == "anova_exchange"]
    for label, data in [("all", rows), *[(str(k), v) for k, v in repeated.groupby("draw")]]:
        # Each dataset retains all of its original folds and all available draws.
        by_fold = data.groupby([*by, "fold"])[["du", "dbreach"]].mean()
        tasks = by_fold.groupby(by).mean()
        counts = data.groupby(by)[list(COUNT)].sum()
        counts["comparisons"] = data.groupby(by).size()
        levels.append(tasks.join(counts).reset_index().assign(draw=label))
    tasks = pd.concat(levels, ignore_index=True)
    summary = []
    for key, g in tasks.groupby(["pipeline", "panel", "delivery", "draw", "stage"]):
        row = dict(zip(["pipeline", "panel", "delivery", "draw", "stage"], key, strict=True))
        for metric in ("du", "dbreach"):
            row[metric], row[metric + "_low95"], row[metric + "_high95"] = bootstrap(g[metric].to_numpy())
        summary.append({**row, "sources": len(g), **g[[*COUNT, "comparisons"]].sum().to_dict()})
    summary = pd.DataFrame(summary)
    for frame, keys in [(tasks, ["pipeline", "panel", "dataset", "delivery", "draw"]),
                        (summary, ["pipeline", "panel", "delivery", "draw"])]:
        for metric in ("du", "dbreach"):
            pivot = frame.pivot(index=keys, columns="stage", values=metric)
            np.testing.assert_allclose(pivot.within_vote + pivot.across_votes, pivot.total, atol=1e-12, rtol=0)
    return tasks, summary


def replay():
    folds = pd.concat([records(repeats.DEST, "anova_exchange"), records(crossed.DEST, "gain_half_width")],
                      ignore_index=True)
    tasks, summary = aggregate(folds)
    DEST.mkdir(parents=True, exist_ok=True)
    for name, frame in (("folds", folds), ("task-contrasts", tasks), ("summary", summary)):
        frame.to_csv(DEST / (name + ".csv"), index=False)
    print(summary.query("draw=='all' and delivery=='guarded'").to_string(index=False))


if __name__ == "__main__":
    replay()
