# Data for the proposal-cost and guarded-policy results

The [table generator](../../operating_results.py) reads the portable files here. Its source map identifies the original result folder for each input. Percentages in the manuscript generally use fractions stored in the CSVs. Named counts such as proposals, false positives, false negatives and breaches retain their original denominators. [Campaigns and reproduction commands](../../../experiments/context_review/README.md)

| Files | Unit and use |
|---|---|
| `rates.csv`, `paired-rates.csv` | Conditional veto rates and paired changes, including proposal-pooled and task/fold-weighted estimates under task/fold and whole-task resampling. Path and drawn cohorts overlap on 75 fold–support pairs. |
| `thresholds.csv`, `frontier.csv`, `cost-transfer.csv` | Pairwise finite-sample break-even costs, minimum-cost policies in the observed library, and selection on 13 tasks evaluated on the fourteenth. One cost unit prices a false alarm. |
| `vote-settings.csv`, `interiors.csv` | True and false vetoes for voting rule, learner and cohort, plus extra detections and false alarms from adding interior restorations to the two endpoints. |
| `heldout-probabilities.csv`, `prediction.csv`, `posterior-costs.csv` | Out-of-task risk predictions, task-averaged probability and ranking scores, and veto costs at the threshold `1/(1+r)` for miss cost `r`. The generator derives the costs from these held-out probabilities. |
| `budget.csv`, `budget-pairs.csv`, `budget-cases.csv`, `digits.csv` | Fixed-width guarded policies, task-level paired effects and fold decisions on 41 PMLB folds. Optdigits has its own table. In these files `total_s` adds measured components with shared fits. |
| `guard-net-benefit.csv` | Whole-task resampling of the additional half-width vote compression minus the chosen breach penalty, holding compute price at zero. |
| `learned.csv`, `learned-pairs.csv` | Cross-width nominations that combine the deletion fraction with a learned probability of staying within tolerance. Each policy has at most one candidate guard refit. |
| `clock-by-task.csv`, `clock-summary.json` | Stand-alone half-width prefix and vote times measured through final prediction. The summary aggregates 164 PMLB runs. Four additional Optdigits runs appear in the task-level data. |

From `MastersThesis`, use `.\thesis-python.ps1 thesis-paper/operating_results.py` to rebuild the tables from this directory. With the original local result folders present, add `--collect` to refresh the portable inputs first. The working records include the complete 360-setting grid, sampled proposals, policy nominations, test probabilities and the individual clocks. [Evidence ZIP](../../../packages/thesis-operating-evidence.zip) · [168 clock records](../../../results/context_review/guard-clock/runs.csv) · [Independent guarded-policy checks](../../audit/operating-guard-audit.md) · [Artifact hashes](../revision/artifacts.json)
