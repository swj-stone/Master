"""Rebuild operating-rule, clustered interval, and refit-budget tables from portable CSVs."""
import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
RUN = PAPER.parent / "results/context_review"
DATA = PAPER / "evidence/operating"
INPUTS = {"rates": "cluster-cost/rates.csv", "paired-rates":"cluster-cost/paired-rates.csv",
          "thresholds":"cluster-cost/thresholds.csv","frontier":"cluster-cost/frontier.csv",
          "cost-transfer":"cluster-cost/transfer/summary.csv","vote-settings":"rule-sensitivity/primary-comparisons.csv",
          "interiors":"rule-sensitivity/interior-vs-endpoints.csv","prediction":"incremental-prediction/summary.csv",
          "budget":"budgeted-guard/summary.csv","budget-pairs":"budgeted-guard/paired-effects.csv",
          "budget-cases":"budgeted-guard/paired-decisions.csv","digits":"budgeted-guard/optdigits.csv",
          "learned":"learned-guard/by-source.csv", "learned-pairs":"learned-guard/paired-effects.csv",
          "clock-by-task":"guard-clock/by-task.csv", "heldout-probabilities":"incremental-prediction/predictions.csv"}
ARMS = {"point":"Empty, point", "empty":"Empty, four panels", "endpoints":"Both endpoints",
        "nested":"Nested minimum", "one_panel_min":"Separating, point", "pooled":"Separating, pooled",
        "separating":"Separating minimum", "random":"Random minimum"}


def table(key, caption, tabular, headings, body):
    directory = PAPER / "tables"
    directory.mkdir(exist_ok=True)
    lines = [r"\begin{table}[tb]",r"\centering",r"\caption{"+caption+"}",r"\label{tab:"+key+"}",
             r"\small\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{"+tabular+"}",r"\toprule",
             *headings,r"\midrule",*body,r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (directory/(key+".tex")).write_text("\n".join(lines)+"\n",encoding="utf8")


def ci(cell):
    return (r"\shortstack[r]{"+f"{100*cell.estimate:.1f}"+r"\\[-.5pt]{\scriptsize ["
            +f"{100*cell.lo95:.1f}, {100*cell.hi95:.1f}"+r"]}}")


def clustered():
    df = pd.read_csv(DATA/"rates.csv").query("weighting == 'pooled' and bootstrap == 'task_fold'")
    rates = df.set_index(["rule","cohort","metric"])
    assert rates.index.is_unique
    rows = []
    for arm,label in ARMS.items():
        cells = [ci(rates.loc[(arm,c,m)]) for c in ("path","uniform_deletions") for m in ("tpr","fpr")]
        rows.append(" & ".join([label,*cells])+r" \\[2pt]")
    table("operating-rates",r"True- and false-positive rates (\%) for a veto, with 95\% task--fold percentile intervals. "
          r"Search-path proposals include 189 observed failures and 211 passes; drawn deletions include 163 and 216. "
          r"Seventy-five fold--support pairs occur in both sources. Rates pool proposals within each source.",
          r"@{}lrr@{\quad}rr@{}",
          [r"& \multicolumn{2}{c}{Search path} & \multicolumn{2}{c}{Drawn deletions} \\",
           r"Rule & TPR & FPR & TPR & FPR \\"], rows)


def settings():
    stats = pd.read_csv(DATA/"vote-settings.csv")
    stats = stats[stats.voter == stats.target].set_index(
        ["family","aggregation","panels","quorum","vote_tolerance_pct","voter","cohort"])
    assert stats.index.is_unique
    cells = [("Both endpoints","endpoints","minimum",4,"strict_majority",3),
             ("Separating minimum","separating","minimum",4,"strict_majority",3),
             ("Pooled votes","separating","pooled",4,"strict_majority",3),
             ("One panel","separating","minimum",1,"strict_majority",3),
             ("Twelve panels","separating","minimum",12,"strict_majority",3),
             ("Ties accepted","separating","minimum",4,"half_or_more",3),
             (r"Vote tolerance 1\%","separating","minimum",4,"strict_majority",1),
             (r"Vote tolerance 5\%","separating","minimum",4,"strict_majority",5)]
    rows=[]
    for label,*config in cells:
        values=[label]
        for learner in ("xgb","logistic"):
            for group in ("path","uniform_deletions"):
                q=stats.loc[(*config,learner,group)]
                values.append(f"{q.tp}/{q.fp}")
        rows.append(" & ".join(values)+r" \\")
    table("operating-rules",r"True/false veto counts after changing one aspect of the four-panel separating "
          r"minimum. Each learner judges its own refits. The proposal-side tolerance varies only in the last two rows. "
          r"The outer tolerance stays fixed. Logistic refits exceeded it in 216 of 400 path and 187 of 379 drawn proposals.",
          r"@{}lrr@{\quad}rr@{}",
          [r"&\multicolumn{2}{c}{XGBoost}&\multicolumn{2}{c}{Logistic regression}\\",
           r"Rule & Path & Drawn & Path & Drawn \\"],rows)


def guards():
    df = pd.read_csv(DATA/"budget.csv").set_index(["budget","method"])
    rows = []
    for budget,name in (("one","One"),("three","Up to three")):
        for method,label in (("fraction_gain","Gain"),("vote_gain","Vote, then Gain"),("fraction_lottery","Lottery"),
                             ("vote_lottery","Vote, then lottery")):
            q=df.loc[(budget,method)]
            cols=[name if method=="fraction_gain" else "",label,f"{100*q.safe_saved:.2f}",
                  f"{int(q.outer_failures)}/41", f"{q.auc:.4f}"]
            rows.append(" & ".join(cols)+r" \\")
        if budget=="one":
            rows.append(r"\midrule")
    table("operating-guards",r"Comparing proposals at the same guard-refit budget. $U$ is the fraction of "
          r"anchor columns saved when the final refit passes the outer tolerance, and zero otherwise. Breaches count "
          r"folds; other columns weight tasks equally.",
          r"@{}llrrr@{}",[r"Budget & Within-width order & $100\bar U$ & Breaches & AUC \\"],rows)


def posterior_costs():
    predictions = pd.read_csv(DATA / "heldout-probabilities.csv")
    rows = []
    for cohort, part in predictions.groupby("cohort"):
        failure = part["fail"].astype(bool)
        for ratio in (1, 2, 5, 10):
            for predictor in ("p_size", "p_size_vote"):
                veto = part[predictor] >= 1 / (1 + ratio)
                fp = int((veto & ~failure).sum())
                fn = int((~veto & failure).sum())
                rows.append(dict(cohort=cohort, miss_cost=ratio, model=predictor, fp=fp, fn=fn,
                                 total=fp + ratio * fn))
    pd.DataFrame(rows).to_csv(DATA / "posterior-costs.csv", index=False)


def guard_net_benefit():
    """Post hoc task uncertainty for the benchmark cost trade-off, with compute priced at zero."""
    decisions = pd.read_csv(DATA / "budget-cases.csv")
    primary = decisions.query("dataset != 'optdigits' and budget == 'one' and tie == 'gain'").copy()
    primary["extra_breach"] = primary.vote_fail.astype(int) - primary.prefix_fail.astype(int)
    by_task = primary.groupby("dataset")[["gain_saved", "extra_breach"]].mean()
    assert len(primary) == 41 and len(by_task) == 14
    draws = np.random.default_rng(33119).integers(len(by_task), size=(30000, len(by_task)))
    rows = []
    for penalty in (0, 1/12, 1/6, 1/3, 1):
        paired = by_task.gain_saved - penalty * by_task.extra_breach
        lo, hi = np.quantile(paired.to_numpy()[draws].mean(axis=1), [.025, .975])
        leave_one_out = (paired.sum() - paired) / (len(by_task) - 1)
        rows.append(dict(breach_cost_per_unit_saved=penalty, mean=paired.mean(), lo95=lo, hi95=hi,
                         omit_one_min=leave_one_out.min(), omit_one_max=leave_one_out.max(),
                         omit_new_thyroid=leave_one_out["new_thyroid"]))
    pd.DataFrame(rows).to_csv(DATA / "guard-net-benefit.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        DATA.mkdir(exist_ok=True,parents=True)
        for name, source in INPUTS.items():
            copyfile(RUN/source, DATA/(name+".csv"))
        copyfile(RUN/"guard-clock/summary.json", DATA/"clock-summary.json")
    posterior_costs()
    guard_net_benefit()
    clustered()
    settings()
    guards()
