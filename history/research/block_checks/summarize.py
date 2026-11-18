"""Summarize paired changes without pooling scores across metrics or timings."""
import numpy as np
import pandas as pd

from research.block_checks.ablation import HERE, KEY


def run():
    results = pd.read_csv(HERE/"cases.csv", keep_default_na=False)
    indexed = results.set_index(KEY + ["arm"]).unstack("arm")
    steps = pd.read_csv(HERE/"residual-pairs.csv")
    ranking = pd.read_csv(HERE/"external-ranks.csv").query("arm == 'block'")
    comparisons = [
        ("block", "pair_cap"), ("pair_matched", "pair_cap"),
        ("block", "pair_matched"), ("block", "anchor_matched"),
        ("block", "vote_matched"), ("block", "singleton_stop"), ("block", "vote_cap"),
    ]
    split = indexed.index.get_level_values("cohort")
    rows = []
    for panel, mask in [
            ("all", np.ones(len(split), dtype=bool)),
            ("prior_68", split != "fresh"), ("fresh_6", split == "fresh"),
            ("external_12", split == "external"),
            ("archived_56", ~split.isin(["fresh", "external"]))]:
        side = indexed.loc[mask]
        for arm, baseline in comparisons:
            gap = side["score", arm] - side["score", baseline]
            delta_utility = side["utility",arm] - side["utility",baseline]
            rows.append(dict(
                panel=panel, arm=arm, compared_to=baseline, n=len(side),
                fewer=(side["k",arm] < side["k",baseline]).sum(),
                same_columns=(side["columns",arm] == side["columns",baseline]).sum(),
                score_higher=(gap>1e-12).sum(), score_tied=(np.abs(gap)<=1e-12).sum(),
                score_lower=(gap < -1e-12).sum(),
                train_gain_higher=(delta_utility>1e-9).sum(),
                train_gain_tied=(np.abs(delta_utility)<=1e-9).sum(),
                train_gain_lower=(delta_utility < -1e-9).sum()))
    table = pd.DataFrame(rows)
    table.to_csv(HERE/"comparison.csv", index=False)
    print(table[table.panel.isin(["prior_68", "fresh_6", "all"])].to_string(index=False))
    subset = steps.loc[~steps.same_set]
    print("Residual pairs:", len(steps), "different from two greedy steps:", len(subset),
          "with no eligible single:", (steps.best_single_score<=0).sum())
    print(subset[KEY + ["prefix", "chosen", "greedy", "best_single_score"]].to_string(index=False))
    print(ranking[KEY + ["p","k","score","random_median","total","rank_min","rank_max"]].to_string(index=False))


if __name__ == "__main__":
    run()
