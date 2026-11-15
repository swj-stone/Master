"""Matched forest tie policies on the archived support slates."""
import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import forest_predict, neighborhood_replicates, transport_predict
from experiments.context_review.forest_replay import bootstrap, read_member, read_metrics
from experiments.context_review.ordinal_decomposition import LABELS
from experiments.context_review.receiver_transport import SOURCE
from experiments.context_review.refit_opportunity import outcome


DEST = bg.ROOT / "thesis-paper/evidence/within_tier"
DESIGN = bg.ROOT / "configs/context_review/within_tier.json"
SLATE = ["pipeline", "panel", "dataset", "fold", "draw"]
METRICS = ["u", "breach", "auc", "accuracy"]


def recover_points():
    """Recover original-slate scores with the same single-panel code used during nomination."""
    origin = forest_predict.DEST / "prediction-lock.json"
    values = {}
    with threadpool_limits(limits=1):
        for key, rec in bg.read(origin)["folds"].items():
            if len(rec["supports"]) == 1:
                values[key] = []
                continue
            source = bg.ROOT / "results/context_review" / SOURCE[rec["panel"]] / key.split("/", 1)[1]
            assert bg.digest(source / "inputs.npz") == rec["inputs_sha"]
            with np.load(source / "inputs.npz") as inputs:
                x, y, train, prop = (inputs[k] for k in ("x", "y", "train", "proposal"))
            seed = 20421 + rec["fold"]
            xt, xp = forest_predict.transform(x, train, x[prop])
            model = forest_predict.new_forest(seed).fit(xt, y[train])
            ballot = transport_predict.build_ballot(model, xt, xp, y[prop], seed, rec["supports"][0],
                                                    "ap" in rec["metrics"], point=True)
            scores = [transport_predict.rating(ballot, s, [frozenset()], "point") for s in rec["supports"][1:]]
            assert 1 + int(np.argmax(scores)) == rec["winners"]["point"]
            values[key] = list(map(float, scores))
    DEST.mkdir(exist_ok=True, parents=True)
    bg.store(DEST / "gain-points.json", {"source_sha": bg.digest(origin), "scores": values})


def rankings(tensor, names):
    if len(tensor) == 1:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    allowance = np.abs(tensor[0]) * np.array([.05 if n == "ap" else .03 for n in names])
    delta = tensor[1:] - tensor[0]
    votes = np.all(-delta <= allowance + 1e-12, axis=3).sum(axis=2).min(axis=1)
    margin = ((delta + allowance + 1e-12) / np.maximum(allowance, 1e-12)).min(axis=3)
    full = np.sort(margin, axis=2)[:, :, 1].min(axis=1)
    np.testing.assert_array_equal(votes >= 3, full >= 0)
    return votes, full, margin[:, 0, 0]


def evaluate_slate(candidates, frozen, vote, full, point, returns, outcomes, widths, design):
    candidates = np.asarray(candidates, dtype=int)
    if candidates.size:
        offsets = candidates - 1
        tier = candidates[vote[offsets] == vote[offsets].max()]
        assert tier[0] == frozen["vote"]
        assert candidates[np.argmax(point[offsets])] == frozen["point"]
        assert candidates[np.argmax(full[offsets])] == frozen["slack"]
        point_id = tier[np.argmax(point[tier - 1])]
        full_id = tier[np.argmax(full[tier - 1])]
    else:
        tier = np.array([0], dtype=int)
        point_id = full_id = 0
    choices = {"first": tier[:1], "uniform": tier, "point": [point_id], "full": [full_id]}
    policies, laws = [], {}
    for method, ids in choices.items():
        law = np.bincount(returns[ids], minlength=len(widths)) / len(ids)
        laws[method] = law
        policies.append(dict(method=method, nominees=" ".join(str(i) for i in ids),
                             **dict(zip(METRICS, law @ outcomes, strict=True))))
    reachable = outcomes[returns[tier]]
    opportunity = dict(candidates=len(candidates), tier_size=len(tier),
                       distinct_returns=len(np.unique(returns[tier])),
                       mixed_u=int(np.ptp(reachable[:, 0]) > 1e-12),
                       mixed_breach=int(np.ptp(reachable[:, 1]) > 0))
    bounds = []
    for a, b in design["primary_comparisons"] + design["reference_comparisons"]:
        coefficients = (laws[a] - laws[b]) * widths
        low = np.minimum(coefficients, 0).sum()
        high = np.maximum(coefficients, 0).sum()
        delta = (laws[a] - laws[b]) @ outcomes[:, 0]
        assert low - 1e-12 <= delta <= high + 1e-12
        bounds.append(dict(method=a, reference=b, lower=low, upper=high, difference=delta,
                           differing_return_law=int(not np.array_equal(laws[a], laws[b]))))
    return policies, opportunity, bounds


def extract(directory, pipeline, gain_points, design):
    lock = bg.read(directory / "prediction-lock.json")
    assert lock["archive_sha"] == bg.digest(directory / "predictions.zip")
    policy_rows, opportunities, envelopes = [], [], []
    with ZipFile(directory / "predictions.zip") as predictions, ZipFile(LABELS) as labels:
        for key, rec in lock["folds"].items():
            identity = {k: rec[k] for k in ("panel", "dataset", "fold")}
            supports = rec["supports"]
            with read_member(predictions, key, rec["member_sha"]) as p, read_member(labels, key, rec["targets_sha"]) as y:
                np.testing.assert_array_equal(y["test_row"], p["test_row"])
                measures = read_metrics(p, p["guard_y"], y["y"], len(supports), "ap" in rec["metrics"])
                votes, full, batch_points = rankings(p["tensor"], rec["metrics"])
                point = p["points"] if pipeline == "anova" else np.array(gain_points[key])
                np.testing.assert_allclose(point, batch_points, rtol=0, atol=1e-10)
            guard = np.array([outcome(g, measures[0][0], .03) for g, o in measures])
            passed = np.array([outcome(o, measures[0][1], .03) for g, o in measures])
            assert guard[0] and passed[0]
            returns = np.where(guard, np.arange(len(supports)), 0)
            widths = np.array([1 - len(s) / len(supports[0]) for s in supports])
            outcomes = np.column_stack((widths * passed, ~passed,
                                        [o["auc"] for _, o in measures], [o["accuracy"] for _, o in measures]))
            slates = rec.get("slates", [list(range(len(supports)))])
            winners = rec["winners"] if "slates" in rec else [rec["winners"]]
            for draw, (slate, frozen) in enumerate(zip(slates, winners, strict=True), 1):
                assert slate[0] == 0
                index = dict(pipeline=pipeline, **identity, draw=draw)
                policies, counts, bounds = evaluate_slate(slate[1:], frozen, votes, full, point,
                                                           returns, outcomes, widths, design)
                policy_rows.extend({**index, **r} for r in policies)
                opportunities.append({**index, **counts})
                envelopes.extend({**index, **r} for r in bounds)
    return policy_rows, opportunities, envelopes


def source_means(frame, groups, columns):
    by = ["pipeline", "panel", "dataset", *groups]
    folds = frame.groupby([*by, "fold"], as_index=False)[columns].mean()
    return folds.groupby(by, as_index=False)[columns].mean()


def comparisons(tasks, design):
    rows = []
    for (pipeline, panel), group in tasks.groupby(["pipeline", "panel"]):
        for metric in METRICS:
            values = group.pivot(index="dataset", columns="method", values=metric)
            for a, b in design["primary_comparisons"] + design["reference_comparisons"]:
                paired = values[a] - values[b]
                est, low, high = bootstrap(paired.to_numpy())
                rows.append(dict(pipeline=pipeline, panel=panel, metric=metric, method=a, reference=b,
                                 difference=est, low95=low, high95=high, sources=len(paired),
                                 positive=int((paired > 1e-12).sum()), negative=int((paired < -1e-12).sum())))
    return pd.DataFrame(rows)


def replay():
    design = bg.read(DESIGN)
    gain = bg.read(DEST / "gain-points.json")
    assert gain["source_sha"] == bg.digest(forest_predict.DEST / "prediction-lock.json")
    chunks = [extract(forest_predict.DEST, "gain", gain["scores"], design),
              extract(neighborhood_replicates.DEST, "anova", {}, design)]
    policies, opportunities, bounds = [pd.DataFrame(a + b) for a, b in zip(*chunks, strict=True)]
    assert len(policies) == 184 * 4
    tasks = source_means(policies, ["method"], METRICS)
    means = tasks.groupby(["pipeline", "panel", "method"], as_index=False)[METRICS].mean()
    tied = opportunities.loc[opportunities.tier_size > 1, SLATE]
    tied_records = policies.merge(tied, on=SLATE, validate="many_to_one")
    tied_contrasts = comparisons(source_means(tied_records, ["method"], METRICS), design)
    tied_counts = tied.groupby(["pipeline", "panel"]).size().rename("eligible_slate_records").reset_index()
    tied_contrasts = tied_contrasts.merge(tied_counts, on=["pipeline", "panel"], validate="many_to_one")
    task_bounds = source_means(bounds, ["method", "reference"], ["lower", "upper", "difference"])
    envelope = task_bounds.groupby(["pipeline", "panel", "method", "reference"], as_index=False)[
        ["lower", "upper", "difference"]].mean()
    frames = {"policies": policies, "opportunities": opportunities, "fold-bounds": bounds, "tasks": tasks,
              "means": means, "contrasts": comparisons(tasks, design), "envelopes": envelope,
              "ties-only": tied_contrasts}
    for name, frame in frames.items():
        frame.to_csv(DEST / (name + ".csv"), index=False)
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "points"), nargs="?", default="replay")
    if parser.parse_args().action == "points":
        recover_points()
    else:
        replay()
