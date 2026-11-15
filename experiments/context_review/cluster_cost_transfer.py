"""Choose a veto policy on other datasets and measure paired held-out error costs."""
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs/context_review/cluster_cost_transfer.json"
OUT = ROOT / "results/context_review/cluster-cost/transfer"
CELLS = ["tp", "fp", "fn", "tn"]
FAMILIES = ["empty", "endpoints", "nested", "separating", "random"]
QUORUMS = ["strict_majority", "half_or_more", "three_quarters_or_more", "unanimous"]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def confuse(veto, event):
    return np.array([(veto & event).sum(), (veto & ~event).sum(),
                     (~veto & event).sum(), (~veto & ~event).sum()])


def prepare(spec):
    gridpath, rulepath, source = [ROOT / spec[k] for k in ("grid", "rules", "decisions")]
    assert sha256(gzip.decompress(gridpath.read_bytes())).hexdigest() == spec["grid_uncompressed_sha256"]
    assert digest(rulepath) == spec["rules_sha256"] and digest(source) == spec["decisions_sha256"]
    rules = pd.read_csv(rulepath)
    assert rules.rule_id.tolist() == list(range(360))
    table = pd.read_csv(gridpath)
    table = table[(table.target == spec["target"]) & (table.voter == spec["voter"])]
    decisions = pd.read_csv(source)
    tasks = sorted(decisions.dataset.unique())
    cohorts = spec["cohorts"]
    cutoffs = [Fraction(s) for s in spec["size_cutoffs"]]
    names = ([f"mask_{i}" for i in rules.rule_id] + ["no_veto", "veto_all"]
             + [f"drop_ge_{v}" for v in cutoffs])
    shape = (len(tasks), 3, len(cohorts), len(names))
    counts = np.zeros((*shape, 4), dtype=int)
    workload = np.zeros(shape, dtype=int)
    hashes = {}
    for task, name in enumerate(tasks):
        for fold in range(3):
            path = ROOT / spec["contexts"].format(dataset=name, fold=fold)
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
            design = json.loads(path.read_bytes())
            families = design["contexts"]
            endpoints = families["separating"]
            families.update(empty=[endpoints[0]], endpoints=[endpoints[0], endpoints[-1]])
            n_contexts = {k:len(set(map(tuple, extensions))) for k, extensions in families.items()}
            for c, cohort in enumerate(cohorts):
                take = table[(table.dataset == name) & (table.fold == fold) & (table.cohort == cohort)]
                take = take.sort_values("rule_id")
                assert take.rule_id.tolist() == rules.rule_id.tolist()
                block = decisions[(decisions.dataset == name) & (decisions.fold == fold) & (decisions.cohort == cohort)]
                assert len(block) > 0 and take.n.eq(len(block)).all()
                counts[task, fold, c, :360] = take[CELLS].to_numpy()
                workload[task, fold, c, :360] = (len(block)+1)*rules.family.map(n_contexts).to_numpy()
                event = block.fail.to_numpy()
                masks = [np.zeros_like(event), np.ones_like(event)]
                masks += [(v.denominator*(block.anchor_k-block.k) >= v.numerator*block.anchor_k).to_numpy()
                          for v in cutoffs]
                for i, veto in enumerate(masks, 360):
                    counts[task, fold, c, i] = confuse(veto, event)
                expected = counts[task, fold, c]
                assert np.all(expected[:, 0] + expected[:, 2] == event.sum())
                assert np.all(expected.sum(axis=1) == len(block))
                # Cross-check the held-out labels and original rules against the independent proposal table.
                for rule, family, aggregation, panels in (
                    ("point", "empty", "minimum", 1), ("empty", "empty", "minimum", 4),
                    ("endpoints", "endpoints", "minimum", 4), ("nested", "nested", "minimum", 4),
                    ("one_panel_min", "separating", "minimum", 1), ("pooled", "separating", "pooled", 4),
                    ("separating", "separating", "minimum", 4), ("random", "random", "minimum", 4)):
                    match = rules[(rules.family == family) & (rules.aggregation == aggregation)
                                  & (rules.panels == panels) & (rules.quorum == "strict_majority")
                                  & (rules.vote_tolerance_pct == 3)]
                    np.testing.assert_array_equal(expected[match.rule_id.item()], confuse(block[rule].to_numpy(), event))
    assert len(tasks) == 14 and len(table) == 14*3*2*360
    np.testing.assert_array_equal(counts[:, :, :, 0].sum(axis=(0, 1, 3)), [400, 379])
    # Tie keys follow training workload, so no test-fold context count affects the selected policy.
    priority = [(int(row.panels), FAMILIES.index(row.family), ["minimum", "pooled"].index(row.aggregation),
                 QUORUMS.index(row.quorum), [3, 1, 5].index(row.vote_tolerance_pct)) for row in rules.itertuples()]
    priority += [(0, 0), (0, 1)]
    priority += [(1, -v) for v in cutoffs]
    libraries = {"mask": np.arange(362), "size": np.arange(360, len(names))}
    return tasks, names, counts, workload, priority, libraries, hashes


def choose(candidates, costs, workload, priority):
    best = int(costs[candidates].min())
    ties = candidates[costs[candidates] == best]
    selected = min(ties, key=lambda i: (workload[i], priority[i]))
    return selected, ties, best


def hold_out(active, tasks, names, counts, workload, priority, libraries, spec):
    detail, selections = [], []
    for test in active:
        train = [task for task in active if task != test]
        train_counts = counts[train].sum(axis=(0, 1))
        train_masks = workload[train].sum(axis=(0, 1))
        for c, cohort in enumerate(spec["cohorts"]):
            for r in spec["cost_ratios"]:
                keys = dict(dataset=tasks[test], cohort=cohort, ratio=r)
                train_cost = train_counts[c, :, 1] + r*train_counts[c, :, 2]
                selected = {}
                for library, candidates in libraries.items():
                    i, ties, cost = choose(candidates, train_cost, train_masks[c], priority)
                    selected[library] = i
                    selections.append(dict(**keys, library=library, selected=names[i], training_tasks=len(train),
                                           training_n=train_counts[c, i].sum(), training_cost=cost,
                                           training_masks=train_masks[c, i], training_ties=len(ties),
                                           training_minimizers="|".join(names[j] for j in ties)))
                for fold in range(3):
                    row = dict(**keys, fold=fold)
                    for library, i in selected.items():
                        tp, fp, fn, tn = counts[test, fold, c, i]
                        row.update(n=tp+fp+fn+tn, event=tp+fn, nonevent=fp+tn)
                        row[library + "_policy"] = names[i]
                        row.update({library + "_" + k: v for k,v in zip(CELLS, (tp, fp, fn, tn))})
                        row[library + "_cost"] = fp + r*fn
                    row["difference"] = row["mask_cost"] - row["size_cost"]
                    detail.append(row)
    return pd.DataFrame(detail), pd.DataFrame(selections)


def summarize(detail):
    columns = ["n", "event", "nonevent", "mask_cost", "size_cost", "difference"]
    columns += [f"{lib}_{k}" for lib in ("mask", "size") for k in CELLS]
    tasks = detail.groupby(["dataset", "cohort", "ratio"], as_index=False)[columns].sum()
    tasks["difference_per_100"] = 100*tasks.difference/tasks.n
    rows = []
    for (cohort, ratio), block in tasks.groupby(["cohort", "ratio"]):
        totals = block[columns].sum().to_dict()
        d = block.difference
        rows.append(dict(cohort=cohort, ratio=ratio, **totals, tasks=len(block),
                         mask_lower=int((d < 0).sum()), tied=int((d == 0).sum()), size_lower=int((d > 0).sum()),
                         pooled_difference_per_100=100*d.sum()/block.n.sum(),
                         equal_task_difference_per_100=block.difference_per_100.mean()))
    return tasks, pd.DataFrame(rows)


def main():
    spec = json.loads(DESIGN.read_bytes())
    tasks, names, counts, workload, priority, libs, hashes = prepare(spec)
    fit_args = (tasks, names, counts, workload, priority, libs, spec)
    folds, choices = hold_out(range(len(tasks)), *fit_args)
    by_task, summary = summarize(folds)
    by_task = by_task.merge(choices.pivot(index=["dataset", "cohort", "ratio"], columns="library",
                                         values="selected").add_suffix("_policy").reset_index(),
                            on=["dataset", "cohort", "ratio"], validate="one_to_one")
    omitted = []
    for i, task in enumerate(tasks):
        reduced, _ = hold_out([j for j in range(len(tasks)) if j != i], *fit_args)
        omitted.append(summarize(reduced)[1].assign(omitted_task=task))
    omissions = pd.concat(omitted, ignore_index=True)
    bounds = omissions.groupby(["cohort", "ratio"]).pooled_difference_per_100.agg(["min", "max"])
    summary = summary.merge(bounds.add_prefix("omit_one_"), on=["cohort", "ratio"], validate="one_to_one")
    frequency = choices.groupby(["cohort", "ratio", "library", "selected"], as_index=False).size()
    tables = dict(summary=summary, selections=choices, frequencies=frequency, omissions=omissions,
                  **{"by-fold": folds, "by-task": by_task})
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for file, df in tables.items():
        path = OUT / f"{file}.csv"
        df.to_csv(path, index=False)
        receipts[path.name] = digest(path)
    record = dict(specification=digest(DESIGN), implementation=digest(Path(__file__)),
                  source_contexts=hashes, result_sha256=receipts)
    (OUT / "manifest.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(summary[["cohort", "ratio", "n", "mask_cost", "size_cost", "mask_lower", "tied", "size_lower",
                   "pooled_difference_per_100", "omit_one_min", "omit_one_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
