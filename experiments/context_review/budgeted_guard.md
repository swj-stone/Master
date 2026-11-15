# Which deletion should receive a limited refit?

A deployment starts with a Compact Gain anchor. The policy nominates a smaller set, fits a fresh model on that set, and checks it against a fresh anchor model on guard rows held out from nomination. A failed check returns the anchor. The outcome is measured after fitting the selected support on all development rows and predicting on the outer fold. The earlier proposal-level study instead scored hypothetical vetoes on a fixed list. [Proposal costs](cluster_cost.md) · [Guard design](../../configs/context_review/budgeted_guard.json)

## A common candidate pool and refit budget

Each eligible width offered the Gain prefix and up to seven distinct random alternatives from the anchor. The primary budget allowed one nomination at half the anchor width, rounded up. The larger budget allowed one at each distinct, strictly smaller width among one quarter, one half and three quarters, rounded up, then returned the smallest guard-passing candidate. The reference nominated the Gain prefix. The vote nominated the candidate with the most approving panels in its worst restoration, resolving ties by Gain ranks. Even a minimum of one approval could win this relative comparison. A second pair of policies used a shared random tie order. All policies used the same pool and disjoint proposal and guard rows. [Implementation](budgeted_guard.py) · [Task and fold eligibility](../../results/context_review/refit-guard/eligibility.csv)

The 14 previously studied PMLB tasks supplied 41 eligible folds. One Wine Quality Red ranking fold had a singleton class and could not be split as specified. One SPECTF fold had a one-column anchor and stayed in the denominator with zero eligible deletions. The table gives each task equal weight after averaging its folds. Compression within tolerance credits the fraction of **anchor** columns saved only if the final outer refit passes the stated joint tolerance. A fallback and a breach both receive zero credit.

| Candidate refits per fold | Nomination | Mean compression within tolerance | Outer breaches | Mean outer AUC |
|---|---|---:|---:|---:|
| One | Gain | 9.33% | 1/41 | 0.93507 |
| One | Vote with Gain ties | 10.12% | 3/41 | 0.93413 |
| Up to three | Gain | 15.36% | 2/41 | 0.93054 |
| Up to three | Vote with Gain ties | 16.75% | 6/41 | 0.92793 |

At one refit, voting changed five nominations and four final supports. New Thyroid fold 1 accounted for the entire 0.79 percentage-point gain in compression within tolerance (paired 95% task-bootstrap interval 0.00 to 2.38). Ionosphere fold 0 and SPECTF fold 1 accounted for the extra breaches. The minimum was zero for every candidate in 32 of the 40 folds that offered a deletion. At up to three refits the paired gain was 1.39 points (−1.50 to 4.76). [All four methods, including lottery controls](../../results/context_review/budgeted-guard/summary.csv) · [Case-level independent reconstruction](../../thesis-paper/audit/operating-guard-audit.md)

At a price of one twelfth of a full unit of compression per outer breach and zero compute cost, the paired net benefit was +0.397 percentage points (task-resampling interval −0.794 to +2.183). Omitting New Thyroid made it −0.427 points. This sensitivity charges each breach equally. [Net-benefit calculations](../../thesis-paper/evidence/operating/guard-net-benefit.csv)

## Selecting across widths with learned risks

The follow-up ranked all candidates by the fraction removed times the estimated probability of staying within tolerance, then gave the top candidate one genuine guard refit. That score omits the probability of passing the guard. The size-only and augmented risk models learned from the other 13 tasks. On PMLB the primary drawn-deletion models changed 16 nominations and 15 final supports after adding dissent. Mean compression within tolerance rose from 8.94% to 11.92% (paired difference 2.98 points, 95% interval −1.47 to 7.46), breaches rose from one to four, and AUC fell from 0.93291 to 0.93042. Training on search paths gave a 2.64-point difference (−1.65 to 6.94) and one versus three breaches. This experiment was designed after the fixed-width results. Its component times assume the historical risk models have already been fitted. [Design](../../configs/context_review/learned_guard.json) · [Risk model evaluation](incremental_prediction.md) · [Paired effects](../../results/context_review/learned-guard/paired-effects.csv)

The first Optdigits comparison froze the fixed-width policy, its features and predictions locally before scoring the official 1,797 test examples. The official train and test files represent disjoint sets of writers. All 24 candidates obtained zero worst-context approvals. Gain and voting therefore gave identical predictions at each budget. The half-width candidate fell back to the 24-column anchor (94.66% accuracy). At the larger budget the 18-column support passed (92.65%). All four learned policies subsequently selected that same 18-column support on this already examined task. [Optdigits fixed-policy results](../../results/context_review/budgeted-guard/optdigits.csv) · [Learned policy decisions](../../results/context_review/learned-guard/per-fold.csv)

## Stand-alone wall time and checks

Two order-balanced pairs per split gave **168 completed runs** on one logical CPU, 164 on PMLB and four on Optdigits. Each arm used fresh models and an empty cache. The clock started with in-memory preprocessing and stopped after final prediction. Python startup, input/output and comparisons against the frozen reference were outside the interval. Gain constructed only its prefix. The vote stopped scoring a candidate once its running minimum could no longer beat the incumbent under the Gain tie order. The worker checked each nominee, final support and probability array against the full evaluation. [Timing design](../../configs/context_review/guard_clock.json) · [All runs](../../results/context_review/guard-clock/runs.csv)

Across the 14 PMLB task means, the prefix took 0.8423 s and voting 0.8658 s. The paired difference was +0.0235 s (whole-task 95% interval −0.0584 to +0.1046), and voting required 11.73 additional masked evaluations on average. Eight of 14 tasks took longer with voting. Optdigits is reported separately. The fixed-policy summary also gives component sums in a shared-fit implementation. Use the stand-alone clocks for the deployment time comparison. [Task means](../../results/context_review/guard-clock/by-task.csv) · [Clock summary](../../results/context_review/guard-clock/summary.json)

## Reproduce the campaigns

Run the following from `MastersThesis` after completing the [PMLB refit guard and incremental risk model](README.md). The fixed-width runner reads the frozen refit-guard eligibility and fold records. For Optdigits it obtains the digest-checked UCI archive at `data/context_study/uci-optdigits/uci80.zip`. Allow the timing runner exclusive CPU access. Each launcher can reuse its completed fold records.

```powershell
.\thesis-python.ps1 -m experiments.context_review.budgeted_guard
.\thesis-python.ps1 -m experiments.context_review.budgeted_guard_report
.\thesis-python.ps1 -m experiments.context_review.learned_guard
.\thesis-python.ps1 -m experiments.context_review.guard_clock
.\thesis-python.ps1 thesis-paper/audit/operating-guard-audit.py --learned
```

The independent audit reconstructed 774 distinct fold–candidate supports, 1,048 fixed-budget minimum scores, 261 guard fits and 101 final fits. Its learned-policy check reconstructed 3,096 scores from the saved coefficients, 104 guard fits and 73 final fits. Freshly computed probabilities matched the saved arrays. [Audit report and limitations](../../thesis-paper/audit/operating-guard-audit.md)
