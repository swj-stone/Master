"""Choose a forest policy on other sources, then score it on the omitted source."""
import json

import numpy as np
import pandas as pd

from experiments.context_review.decision_opportunities import EVIDENCE
from experiments.context_review import budgeted_guard as bg


DESIGN = bg.ROOT / "configs/context_review/transfer_choice.json"
DEST = EVIDENCE / "transfer_choice"
SLATE_KEY = ["pipeline", "panel", "dataset", "fold", "draw"]
METRICS = ["u", "breach", "auc", "accuracy"]


def source_outcomes():
    """Align all four tie rules with whole-slate point selection before aggregation."""
    tier = pd.read_csv(EVIDENCE / "within_tier/policies.csv", float_precision="round_trip")
    tier["method"] = tier.method.replace({"first": "original"})
    original = pd.read_csv(EVIDENCE / "forest_cross/policies.csv", float_precision="round_trip")
    local = pd.read_csv(EVIDENCE / "neighborhood_replicates/policies.csv", float_precision="round_trip")
    forest = original.loc[(original.voter == "forest") & (original.receiver == "forest")]
    guarded = local.loc[local.delivery == "guarded"]
    baseline = pd.concat([forest.assign(pipeline="gain", draw=1), guarded.assign(pipeline="anova")])
    check = baseline.loc[baseline.method == "vote"].set_index(SLATE_KEY).sort_index()
    ref = tier.loc[tier.method == "original"].set_index(SLATE_KEY).sort_index()
    pd.testing.assert_index_equal(ref.index, check.index)
    np.testing.assert_allclose(ref[METRICS], check[METRICS], rtol=0, atol=1e-12)
    point = baseline.loc[baseline.method == "point"].assign(method="slate_point")
    frame = pd.concat([tier, point], ignore_index=True)[[*SLATE_KEY, "method", *METRICS]]
    assert len(frame) == 184 * 5 and not frame.duplicated([*SLATE_KEY, "method"]).any()
    assert (frame.groupby(SLATE_KEY).size() == 5).all()
    group = ["pipeline", "panel", "dataset", "method"]
    fold = frame.groupby([*group, "fold"])[METRICS].mean()
    return fold.groupby(group).mean().reset_index()


def select_rule(training, order, penalty, epsilon):
    """Compare equal-source training means. Stable order resolves tied averages."""
    metrics = training.groupby("method")[["u", "breach"]].mean().loc[order]
    net = metrics.u - penalty * metrics.breach
    for method in order:
        if net[method] >= net.max() - epsilon:
            return method, net[method]
    raise ValueError("Incomplete training data")


def replay():
    config = json.loads(DESIGN.read_text(encoding="utf-8"))
    tasks = source_outcomes()
    groups = ["pipeline", "panel"]
    out = []
    for (pipeline, panel), group in tasks.groupby(groups):
        for held, test in group.groupby("dataset"):
            train = group.loc[group.dataset != held]
            test = test.set_index("method")
            count = train.dataset.nunique()
            assert count == (9 if panel == "first" else 5)
            for penalty in config["breach_penalties"]:
                methods = {m: (m, np.nan) for m in config["expanded_preference"]}
                for label, pref in (("learn_tier", "within_tier_preference"), ("learn_all", "expanded_preference")):
                    methods[label] = select_rule(train, config[pref], penalty, config["tie_epsilon"])
                for label, (choice, training_net) in methods.items():
                    row = test.loc[choice]
                    base = test.loc["slate_point"]
                    net = row.u - penalty * row.breach
                    difference = net - (base.u - penalty * base.breach)
                    out.append(dict(pipeline=pipeline, panel=panel, dataset=held, penalty=penalty,
                                    policy=label, chosen=choice, training_sources=count, training_net=training_net,
                                    **row[METRICS].to_dict(), net=net, du_vs_point=row.u - base.u,
                                    dbreach_vs_point=row.breach - base.breach, dnet_vs_point=difference))
    held = pd.DataFrame(out)
    assert len(held) == 32 * len(config["breach_penalties"]) * 7
    summary = []
    for keys, rows in held.groupby([*groups, "penalty", "policy"]):
        metrics = [*METRICS, "net", "du_vs_point", "dbreach_vs_point", "dnet_vs_point"]
        pos = sum(rows.dnet_vs_point > config["tie_epsilon"])
        neg = sum(rows.dnet_vs_point < -config["tie_epsilon"])
        summary.append(dict(zip([*groups, "penalty", "policy"], keys), **rows[metrics].mean().to_dict(),
                            higher=int(pos), tied=len(rows) - pos - neg, lower=int(neg),
                            sources=len(rows), chose_slate_point=sum(rows.chosen == "slate_point")))
    tables = {"held": held, "summary": pd.DataFrame(summary)}
    DEST.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(DEST / (name + ".csv"), index=False)
    return tables


if __name__ == "__main__":
    replay()
