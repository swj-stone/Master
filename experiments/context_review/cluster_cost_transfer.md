# Choosing a veto policy on other tasks

At equal error costs, choosing a deletion-size cutoff on 13 tasks and applying it to the fourteenth produced 98 errors on the 379 drawn proposals. Choosing from the masking-rule library on the same training tasks produced 183 errors. The size policy had lower cost on 12 of the 14 held-out datasets. On the 400 search-path proposals the corresponding totals were 170 and 159. The direction of that smaller path difference changed when individual datasets were removed and the selection repeated.

## Design

This follow-up was specified after examining the cost frontier on these tasks. For each cohort and miss cost *r* in {1, 2, 5, 10}, all three folds of a task were set aside. Each policy library minimized total `FP + r × FN` on proposals from the other 13 tasks. Both selected policies then incurred that same loss on the held-out task's proposals. The split was repeated once per task. Each proposal contributes once per cohort and cost ratio. The path cohort had 189 failures and 211 passes, and the drawn cohort had 163 failures and 216 passes. The cohorts share 75 fold–support pairs. Both use the archived XGBoost ballots and XGBoost refit outcomes with the original 3% joint tolerance and the 5% rare-positive AP allowance. [Locked follow-up](../../configs/context_review/cluster_cost_transfer.json) · [Voting-rule definitions](../../configs/context_review/rule_sensitivity.json)

The masking library contained all 360 specified combinations of context family, aggregation, panel count, quorum and masking tolerance, plus the two constant decisions. The size library vetoed deletions at or above one of six fractions of the anchor width (1/12, 1/6, 1/4, 1/3, 1/2, 3/4), and included both constants. The training criterion used integer error counts. Among tied masking policies the specified order preferred constants, fewer nominal masks on the training folds, fewer panels, then a fixed ordering of the remaining settings. Among equally costly size policies it preferred constants, then larger cutoffs. All decisions about ties used training information. Error costs omit the time and data needed to obtain the masks.

## Held-out costs

The cost difference is masking minus size, expressed per 100 held-out proposals. The equal-task column averages the 14 held-out task-specific costs per 100. The three task counts give masking wins, ties and size wins. The last column removes each task from both **training and evaluation**, selects on 12 of the remaining tasks, evaluates on the thirteenth and repeats for all remaining tasks. It reports the minimum and maximum pooled difference over the 14 omissions. This is a finite sensitivity analysis.

| Cohort | Miss cost | Mask / size total cost | Pooled difference per 100 | Equal-task difference per 100 | Task wins / ties / losses for masking | Omit-one range |
|---|---:|---:|---:|---:|---:|---:|
| Path, 400 proposals | 1 | 159 / 170 | −2.75 | −0.86 | 8 / 1 / 5 | [−11.81, +10.44] |
| Path | 2 | 221 / 226 | −1.25 | +2.40 | 5 / 0 / 9 | [−5.49, +11.75] |
| Path | 5 | 244 / 261 | −4.25 | −1.38 | 6 / 1 / 7 | [−13.19, +16.67] |
| Path | 10 | 275 / 241 | +8.50 | +6.40 | 3 / 8 / 3 | [−30.22, +36.54] |
| Drawn, 379 proposals | 1 | 183 / 98 | +22.43 | +24.16 | 2 / 0 / 12 | [+14.58, +29.06] |
| Drawn | 2 | 170 / 154 | +4.22 | +6.97 | 3 / 1 / 10 | [+1.08, +11.37] |
| Drawn | 5 | 200 / 160 | +10.55 | +15.09 | 3 / 2 / 9 | [+0.28, +15.74] |
| Drawn | 10 | 229 / 221 | +2.11 | +15.81 | 7 / 0 / 7 | [−20.22, +17.56] |

On drawn proposals at *r* = 1, the size library selected the one-third cutoff in all 14 training sets. At *r* = 5 it selected one sixth in all 14. On path proposals at *r* = 1, 12 of 14 training sets selected pooled endpoint voting with one panel, a half-or-more quorum and 3% masking tolerance. The size policy selected one third in 10 training sets and one half in four. Some distinct voting settings had the same minimum training cost. Across all runs the number of minimizing mask settings ranged from one to five and the number of minimizing size policies from one to two. [Choices and training ties](../../results/context_review/cluster-cost/transfer/selections.csv) · [Selection frequencies](../../results/context_review/cluster-cost/transfer/frequencies.csv)

This procedure evaluates the complete rule-selection strategy across the 14 previously examined tasks. The complete omission range crossed zero at every path cost ratio. Under the drawn proposal design the size library retained a lower pooled cost across all task omissions at ratios 1, 2 and 5. At ratio 10 the ordering depended on the tasks included. Substantial overlap between training sets and the convenience selection of datasets limit population inference. Search-path and drawn proposals also represent different deployment choices. [Held-out task results](../../results/context_review/cluster-cost/transfer/by-task.csv) · [Refitted omission results](../../results/context_review/cluster-cost/transfer/omissions.csv)

From the thesis root, run `./thesis-python.ps1 -m experiments.context_review.cluster_cost_transfer` after generating the complete rule-sensitivity grid. The script checks the eight original rules against the independent proposal archive on all 42 folds before selecting policies. [Implementation](cluster_cost_transfer.py) · [Paired held-out folds](../../results/context_review/cluster-cost/transfer/by-fold.csv) · [Summary](../../results/context_review/cluster-cost/transfer/summary.csv) · [Run manifest](../../results/context_review/cluster-cost/transfer/manifest.json)
