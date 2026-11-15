"""Verify the per-mask score archive and reduce the prespecified finite grid."""
from itertools import product
import json

import numpy as np
import pandas as pd

from experiments.context_review.rule_sensitivity import AXES, DESIGN, DONOR, LEARNERS, OUT, read, rule_grid
from experiments.context_study.data import config, digest


COUNTS = ["tp", "fp", "fn", "tn"]
BY = ["voter", "target", "cohort", "rule_id"]
BASE = ("separating", "minimum", 4, "strict_majority", 3)


def replay(votes, design, learner):
    masks = {tuple(np.flatnonzero(m)): i for i,m in enumerate(votes[learner+"_masks"])}
    scores = votes[learner+"_scores"]
    families = dict(design["contexts"])
    sep = families["separating"]
    families["endpoints"] = [sep[0], sep[-1]] if sep[0] != sep[-1] else [sep[0]]
    families["empty"] = [sep[0]]
    supports = np.array(design["supports"], dtype=object)
    labels = list(votes["metrics"])
    grids = rule_grid()
    decisions = np.zeros((len(supports), len(grids)), dtype=bool)
    for family, extensions in families.items():
        indexes = np.array([[masks[tuple(sorted(set(s) | set(e)))] for e in extensions]
                            for s in [design["anchor"], *supports]])
        original, reduced = scores[indexes[0]], scores[indexes[1:]]
        difference = original-reduced
        for tolerance in read(DESIGN)["vote_tolerance_pct"]:
            allowance = np.asarray([.05 if name == "ap" else tolerance/100 for name in labels])
            yes = ~(difference > (np.abs(original)*allowance+1e-12)).any(axis=-1)
            for index,(fam,agg,count,quorum,tol) in enumerate(grids):
                if fam != family or tol != tolerance:
                    continue
                p_yes = yes[:, :, :count].mean(axis=2).min(axis=1) if agg == "minimum" else yes[:, :, :count].mean(axis=(1,2))
                if quorum == "strict_majority":
                    decisions[:,index] = p_yes <= .5
                else:
                    decisions[:,index] = p_yes < {"half_or_more":.5, "three_quarters_or_more":.75, "unanimous":1}[quorum]
    np.testing.assert_array_equal(decisions, votes[learner+"_veto"])
    return decisions


def fold_rows(name, fold):
    folder = OUT/name/f"fold-{fold}"
    design = read(DONOR/name/f"fold-{fold}"/"selection.json")
    voting, done = read(folder/"voting.json"), read(folder/"completed.json")
    assert voting["protocol"] == digest(OUT/"protocol.json")
    assert voting["donor"] == digest(DONOR/name/f"fold-{fold}"/"selection.json")
    assert voting["votes"] == digest(folder/"votes.npz") and done["voting"] == digest(folder/"voting.json")
    assert done["logistic_predictions"] == digest(folder/"logistic-outer.npz")
    n = len(design["supports"])
    indices = {tuple(s): i for i,s in enumerate(design["supports"])}
    cohorts = {c:[indices[tuple(s)] for s in items] for c,items in design["cohorts"].items()}
    failures = {k:np.array(done["outcomes"][k]["failure"]) for k in LEARNERS}
    frames, concordance, interiors = [], [], []
    grid = rule_grid()
    interior_ids = [i for i, rule in enumerate(grid) if rule[0]=="separating"]
    endpoint_ids = [grid.index(("endpoints", *grid[i][1:])) for i in interior_ids]
    with np.load(folder/"votes.npz") as saved:
        for voter in LEARNERS:
            matrix = replay(saved, design, voter)
            for target, cohort in product(LEARNERS, cohorts):
                selected = cohorts[cohort]
                veto, fail = matrix[selected], failures[target][selected,None]
                cols = dict(zip(COUNTS, [v.sum(axis=0) for v in (veto & fail, veto & ~fail, ~veto & fail, ~veto & ~fail)], strict=True))
                frame = pd.DataFrame(cols).rename_axis("rule_id").reset_index()
                frames.append(frame.assign(dataset=name, fold=fold, voter=voter, target=target,
                                           cohort=cohort, n=len(selected)))
                added = veto[:,interior_ids] & ~veto[:,endpoint_ids]
                released = veto[:,endpoint_ids] & ~veto[:,interior_ids]
                assert not released[:, [grid[i][1]=="minimum" for i in interior_ids]].any()
                interior = pd.DataFrame(dict(rule_id=interior_ids, added_tp=(added&fail).sum(axis=0),
                    added_fp=(added&~fail).sum(axis=0), released_tp=(released&fail).sum(axis=0),
                    released_fp=(released&~fail).sum(axis=0)))
                interiors.append(interior.assign(dataset=name, fold=fold, voter=voter, target=target,
                                                 cohort=cohort, n=len(selected)))
    for cohort,ids in cohorts.items():
        xgb, log = (failures[k][ids] for k in LEARNERS)
        concordance.append(dict(dataset=name, fold=fold, cohort=cohort, n=len(ids),
                                both=int((xgb&log).sum()), xgb_only=int((xgb&~log).sum()),
                                logistic_only=int((~xgb&log).sum()), neither=int((~xgb&~log).sum())))
    return pd.concat(frames), pd.concat(interiors), concordance, dict(dataset=name, fold=fold, unique=n,
        overlap=sum(map(len, cohorts.values()))-n, **{k:len(v) for k,v in cohorts.items()},
        warnings=len(done["warnings"]), elapsed_seconds=done["elapsed_seconds"])


def assign_rates(frame, prefix=""):
    tp, fp, fn, tn = (frame[prefix+k] for k in COUNTS)
    for k in ("tp", "fp"):
        frame[prefix+k+"_per_proposal"] = frame[prefix+k]/(tp+fp+fn+tn)
    frame[prefix+"sensitivity"] = tp/(tp+fn)
    frame[prefix+"specificity"] = tn/(fp+tn)
    frame[prefix+"precision"] = tp/(tp+fp)
    frame[prefix+"veto_rate"] = (tp+fp)/(tp+fp+fn+tn)


def main():
    assert len(rule_grid()) == 360
    frames, interiors, overlap, exposures, missing = [], [], [], [], []
    for name, fold in product(config()["datasets"], range(3)):
        folder = OUT/name/f"fold-{fold}"
        if not (folder/"completed.json").exists():
            missing.append(dict(dataset=name, fold=fold, reason=read(folder/"failure.json") if
                                (folder/"failure.json").exists() else "worker not complete"))
            continue
        frame, interior, pairs, exposure = fold_rows(name, fold)
        frames.append(frame)
        interiors.append(interior)
        overlap.extend(pairs)
        exposures.append(exposure)
    (OUT/"missing.json").write_text(json.dumps(missing, indent=2))
    if not exposures:
        raise RuntimeError("No workers complete")
    folds = pd.concat(frames, ignore_index=True)
    assert (folds[COUNTS].sum(axis=1)==folds.n).all()
    folds.to_csv(OUT/"by-fold.csv.gz", index=False)
    exp = pd.DataFrame(exposures)
    exp.to_csv(OUT/"exposure.csv", index=False)
    if not missing:
        assert tuple(exp[["unique", "path", "uniform_deletions", "overlap"]].sum())==(704, 400, 379, 75)
    counts = folds.groupby(BY)[["n", *COUNTS]].sum()
    fractions = folds[COUNTS].div(folds.n, axis=0)
    task = folds.assign(**{k:fractions[k] for k in COUNTS}).groupby(BY+["dataset"])[COUNTS].mean()
    task.to_csv(OUT/"by-task.csv.gz")
    macro = task.groupby(BY)[COUNTS].mean().add_prefix("equal_task_")
    summary = counts.join(macro)
    assign_rates(summary)
    assign_rates(summary, "equal_task_")
    rules = pd.DataFrame(rule_grid(), columns=AXES).rename_axis("rule_id")
    summary = summary.reset_index().merge(rules.reset_index(), on="rule_id", validate="many_to_one")
    summary.to_csv(OUT/"summary.csv", index=False)
    base = rule_grid().index(BASE)
    displayed = summary.loc[[sum(a != b for a,b in zip(row, BASE)) <= 1 for row in
                             summary[list(AXES)].itertuples(index=False, name=None)]].copy()
    bcols = ["voter", "target", "cohort"]
    for k in ("tp", "fp", "equal_task_tp", "equal_task_fp"):
        by = summary.loc[summary.rule_id==base].set_index(bcols)[k]
        displayed["delta_vs_primary_"+k] = displayed[k]-displayed.set_index(bcols).index.map(by)
    displayed.to_csv(OUT/"primary-comparisons.csv", index=False)
    changes = ["added_tp", "added_fp", "released_tp", "released_fp"]
    interiors = pd.concat(interiors, ignore_index=True)
    task_change = interiors.assign(**{k:interiors[k]/interiors.n for k in changes}).groupby(BY+["dataset"])[changes].mean()
    task_change.to_csv(OUT/"interior-by-task.csv.gz")
    complete = interiors.groupby(BY)[["n", *changes]].sum().join(task_change.groupby(BY).mean().add_prefix("equal_task_"))
    complete.reset_index().merge(rules, on="rule_id", validate="many_to_one").to_csv(OUT/"interior-vs-endpoints.csv", index=False)
    cross = pd.DataFrame(overlap)
    cross.to_csv(OUT/"outcome-concordance-by-fold.csv", index=False)
    cross.groupby("cohort").sum(numeric_only=True).drop(columns="fold").to_csv(OUT/"outcome-concordance.csv")
    print(exp[["unique", "path", "uniform_deletions", "overlap", "warnings"]].sum().to_string())
    print(displayed.loc[displayed.rule_id == base, bcols+COUNTS+["equal_task_sensitivity",
          "equal_task_specificity"]].to_string(index=False))
    print(f"Verified {len(exp)} folds, {len(summary)} cohort/rule/voter/target rows, {len(missing)} missing")


if __name__ == "__main__":
    main()
