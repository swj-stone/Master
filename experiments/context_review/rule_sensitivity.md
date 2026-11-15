# How the voting rule and refitted learner change deletion warnings

The original four-panel rule detected 184 of 189 observed XGBoost refit failures on 400 search-path proposals. It also vetoed 194 of the 211 proposals that passed after refitting. Replacing XGBoost with regularized logistic regression on the same supports gave 213 detections out of 216 failures and 176 false alarms among 184 passing refits. The refitted learner changed the outcome on 111 of the 400 proposals. Both evaluations used the original XGBoost-derived supports and context families.

## Inputs and operating rules

The locally fixed [design](../../configs/context_review/rule_sensitivity.json) covers all 14 PMLB tasks and 42 original outer folds. The two collections contain 400 path and 379 drawn deletion proposals, including 75 shared fold-support pairs, or 704 distinct pairs overall. All 42 folds completed. Each logistic fit used standardized features, L2 regularization with (C=1), the LBFGS solver and the original fitting or development partition as appropriate. All 746 logistic development refits converged without warnings. Each anchor and proposed support received a separate refit.

We used the empty context, both endpoints, nested contexts, separating contexts, and the archived random contexts. For each family we crossed minimum versus pooled aggregation, 1, 4 or 12 paired panels, four required yes fractions, and 1%, 3% or 5% vote-side loss allowances. The quorums were at least one half, strictly more than one half, at least three quarters, and unanimity. At four panels, a strict majority and a minimum of three quarters coincide within a context. The first four panels match the original panels exactly. The additional eight reweight the same sampled ranking observations. All 360 settings per voter are retained, including numerically equivalent settings. Both refit outcomes use the same fixed 3% relative AUC and accuracy allowance throughout. No fold met the rare-positive condition for including average precision.

The counts below refer to observed outer-fold refit outcomes. Each entry is true-positive / false-positive vetoes. The rows change one setting from the primary four-panel, separating-family minimum with a strict majority and a 3% voting allowance.

| Rule | XGBoost path | XGBoost drawn | Logistic path | Logistic drawn |
|---|---:|---:|---:|---:|
| Primary | 184 / 194 | 161 / 178 | 213 / 176 | 184 / 173 |
| Endpoints only | 173 / 174 | 161 / 169 | 211 / 159 | 182 / 144 |
| Matched random interiors | 182 / 196 | 161 / 183 | 213 / 179 | 184 / 173 |
| Pooled votes | 156 / 139 | 153 / 148 | 203 / 122 | 174 / 124 |
| One panel | 179 / 193 | 160 / 175 | 212 / 172 | 183 / 169 |
| Twelve panels | 184 / 191 | 161 / 177 | 213 / 178 | 183 / 171 |
| At least two of four | 178 / 186 | 161 / 171 | 212 / 168 | 183 / 162 |
| Four of four | 187 / 201 | 162 / 191 | 214 / 181 | 185 / 184 |
| 1% voting allowance | 184 / 204 | 162 / 201 | 216 / 182 | 187 / 189 |
| 5% voting allowance | 174 / 166 | 157 / 162 | 211 / 159 | 180 / 155 |
| Number failing / passing after refit | 189 / 211 | 163 / 216 | 216 / 184 | 187 / 192 |

Pooled aggregation admitted more passing proposals and missed more failures than the minimum in both cohorts and for both learners. Across equally weighted tasks, the minimum's sensitivity/specificity was 98.2%/9.3% for XGBoost on path proposals and 99.1%/19.4% on drawn proposals. For logistic regression these rates were 95.7%/4.8% and 95.3%/10.3%. With pooled votes they became 81.9%/34.7% and 89.7%/32.9% for XGBoost and 89.2%/35.8% and 88.4%/39.2% for logistic regression. These conditional rates divide equal-task fractions of TP by TP+FN, or TN by TN+FP. The underlying fractions first average proposal indicators within each fold and the three folds within each task.

## What interior contexts add at each voting allowance

Each entry counts extra observed failures detected / extra passing refits vetoed when separating interiors are added to the two endpoints. Both rules in each comparison use the same four panels, minimum, quorum and vote-side tolerance.

| Vote-side allowance | XGBoost path | XGBoost drawn | Logistic path | Logistic drawn |
|---|---:|---:|---:|---:|
| 1% | 3 / 11 | 1 / 8 | 3 / 10 | 1 / 20 |
| 3% | 11 / 20 | 0 / 9 | 2 / 17 | 2 / 29 |
| 5% | 21 / 21 | 6 / 11 | 6 / 29 | 8 / 32 |

At 5%, the 21 additional true and 21 additional false XGBoost vetoes on path proposals amount to 5.16 and 4.17 percentage points when tasks receive equal weight. On drawn proposals, the corresponding weighted increments are both 2.18 points. The zero additional XGBoost detections in the primary drawn comparison depends on the operating rule. It also occurred with 12 panels and the primary tolerance, while one panel added two detections and ten false alarms. Logistic interiors produced more additional false alarms than additional detections at each of the three tabulated tolerances.

Of the 400 path proposals, 147 failed with both refitted learners, 42 failed with XGBoost alone, 69 failed with logistic regression alone, and 142 passed with both. For the 379 drawn proposals, the counts were 134, 29, 53, and 163, respectively. The two learners therefore have different target labels on 27.8% of path proposals and 21.6% of drawn proposals. The crossed voter/target comparisons are recorded separately in the full grid. These results describe alternative operating rules on the original, previously inspected XGBoost proposals. They establish neither optimization on a fresh cohort nor the behavior of a feature search driven by logistic regression.

## Reproduction and evidence

Run the following from the thesis checkout using the existing environment.

```powershell
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity --jobs 4
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity_report
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity_audit
```

The [run folder](../../results/context_review/rule-sensitivity/) contains the hash lock, fold-level score matrices, masks, panel weights, vote arrays, and logistic outer probabilities. The XGBoost outer probabilities are linked by digest to the original refit-target archive. The [full grid](../../results/context_review/rule-sensitivity/summary.csv) contains 2,880 cohort/rule/voter/target rows, with per-task and per-fold counterparts. The [one-setting comparisons](../../results/context_review/rule-sensitivity/primary-comparisons.csv), [paired interior counts](../../results/context_review/rule-sensitivity/interior-vs-endpoints.csv) and [learner outcome agreement](../../results/context_review/rule-sensitivity/outcome-concordance.csv) support the tables above.

The reconstruction checks the complete archived four-panel XGBoost ballot arrays and eight original rule decisions for all 704 proposals. The report rebuilds all 360 settings per voter directly from saved per-mask metrics using fraction comparisons. The separate [raw-data audit](rule_sensitivity_audit.py) reproduced all 746 logistic outer probability arrays exactly. It reconstructed all 12 ranking-panel weights and used scikit-learn metric functions to check both learners on the anchor and first proposed support under a fixed context in each fold. The check used the first interior context when available and the final endpoint otherwise. Across 2,016 mask-panel evaluations, the largest absolute discrepancy in a ranking metric was 3.67e-15. See the [fold-level audit](../../results/context_review/rule-sensitivity/independent-audit.csv).
