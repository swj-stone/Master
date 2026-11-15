"""Evaluate matched Gain shortlists from their precomputed nominees and the full refit slate."""
import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.decision_opportunities import (
    EVIDENCE, PHASES, RUNS, envelope, guarded_distribution,
)
from experiments.context_review.gain_increment import DESIGN, DEST, define_choices
from experiments.context_review.marginal_replay import logistic_observations, xgb_observations


METRICS = ["u", "breach", "retained", "auc", "accuracy"]
FACTORS = (("vote_uniform", "uniform"), ("vote_point", "point"))
GROUP = ["panel", "receiver", "shortlist"]


def completed_records():
    lock = bg.read(DEST / "nomination-lock.json")
    src = bg.read(EVIDENCE / "receiver/prediction-lock.json")["folds"]
    assert lock["design"] == bg.digest(DESIGN)
    assert len(src) == 46 and src.keys() == lock["folds"].keys()
    seals = {p: bg.read(EVIDENCE / fields[2]) for p, fields in PHASES.items()}
    logistic = (pd.read_csv(EVIDENCE / "receiver/candidates.csv", dtype={"columns": str})
                .rename(columns={"columns": "columns_text"})
                .set_index(["panel", "dataset", "fold", "index"]).sort_index())
    folds, paired, costs = [], [], []
    for full_key, spec in src.items():
        panel, key = full_key.split("/", 1)
        result = lock["folds"][full_key]
        supports = spec["supports"]
        source = RUNS / PHASES[panel][0] / key / "selection.json"
        assert result["selection_sha"] == bg.digest(source) == spec["selection_sha"]
        sets, groups = define_choices(bg.read(source), supports, result["gain"])
        assert sets == result["shortlists"] and groups == result["policies"]
        ident = {field: spec[field] for field in ("panel", "dataset", "fold")}
        for policy, cost in result["primary_cost"].items():
            costs.append(dict(**ident, policy=policy, candidates=len(supports)-1,
                              choices=len(sets["half"]) if len(supports)>1 else 0, **cost))
        obs = {"xgb": xgb_observations(panel, key, spec, seals[panel][key]),
               "logistic": logistic_observations(spec, logistic)}
        save = np.array([1-len(s)/len(supports[0]) for s in supports])
        for receiver, values in obs.items():
            guard, outer, auc, accuracy = zip(*values, strict=True)
            assert guard[0] and outer[0]
            metrics = np.column_stack((save*outer, np.logical_not(outer), 1-save, auc, accuracy))
            for restriction, policies in groups.items():
                common = dict(**ident, receiver=receiver, shortlist=restriction,
                              candidates=len(supports)-1, choices=len(sets[restriction]) if len(supports)>1 else 0)
                returns = {}
                for policy, indices in policies.items():
                    prob = guarded_distribution(indices, guard, len(supports))
                    returns[policy] = prob
                    scores = dict(zip(METRICS, prob @ metrics, strict=True))
                    folds.append(dict(**common, policy=policy, lottery_size=len(indices), **scores))
                for a, b in FACTORS:
                    lo, hi = envelope(returns[a], returns[b], save)
                    diff = (returns[a] - returns[b]) @ metrics
                    assert lo-1e-12 <= diff[0] <= hi+1e-12
                    paired.append(dict(**common, policy=a, reference=b,
                                       nominee_changed=set(policies[a]) != set(policies[b]),
                                       returned_changed=not np.allclose(returns[a], returns[b], atol=1e-12, rtol=0),
                                       delta_lower=lo, delta_upper=hi, **dict(zip(METRICS, diff, strict=True))))
    return pd.DataFrame(folds), pd.DataFrame(paired), pd.DataFrame(costs)


def validate_controls(folds):
    original = pd.read_csv(EVIDENCE / "opportunities/folds.csv")
    keys = ["panel", "dataset", "fold", "receiver", "policy"]
    for condition, policies in (
        ("all", {"uniform": "uniform", "top_uniform": "vote_uniform", "point": "point", "refined": "vote_point"}),
        ("strict_rank", {"gain": "point"}),
    ):
        expected = original.loc[original.method.isin(policies)].copy()
        expected["policy"] = expected.method.map(policies)
        observed = folds.loc[(folds.shortlist==condition) & folds.policy.isin(policies.values())]
        fields = ["u", "breach", "auc", "accuracy"]
        a = observed.set_index(keys)[fields].sort_index()
        b = expected.set_index(keys)[fields].sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, rtol=0)


def summarize(paired):
    group_cols = [*GROUP, "policy", "reference"]
    by_task = paired.groupby([*group_cols, "dataset"])[[*METRICS, "delta_lower", "delta_upper"]].mean()
    count_fields = ["nominee_changed", "returned_changed"]
    counts = paired.groupby(group_cols)[count_fields].sum()
    results = []
    for key, group in by_task.groupby(group_cols):
        sample = group[METRICS].to_numpy()
        n = len(group)
        draws = np.random.default_rng(33119).integers(n, size=(30000, n))
        lower, upper = np.percentile(sample[draws].mean(axis=1), [2.5, 97.5], axis=0)
        common = dict(zip(group_cols, key, strict=True))
        common.update(tasks=n, up=int((group.u>1e-12).sum()), down=int((group.u< -1e-12).sum()),
                      max_gain=group.delta_upper.mean(), max_loss=group.delta_lower.mean(),
                      **counts.loc[key].to_dict())
        for j, name in enumerate(METRICS):
            results.append(dict(**common, metric=name, delta=sample[:,j].mean(), low95=lower[j], high95=upper[j]))
    return by_task.reset_index(), pd.DataFrame(results)


def main():
    folds, pairs, costs = completed_records()
    validate_controls(folds)
    assert len(folds) == 46*2*6*4
    tasks = folds.groupby([*GROUP, "policy", "dataset"])[METRICS].mean()
    means = tasks.groupby([*GROUP, "policy"]).mean()
    task_diff, contrasts = summarize(pairs)
    for name, df in (("folds", folds), ("tasks", tasks.reset_index()), ("means", means.reset_index()),
                     ("paired-folds", pairs), ("paired-tasks", task_diff), ("contrasts", contrasts),
                     ("costs", costs)):
        df.to_csv(DEST / f"{name}.csv", index=False)
    print(means.xs("half", level="shortlist")[["u","breach"]].round(4))


if __name__ == "__main__":
    main()
