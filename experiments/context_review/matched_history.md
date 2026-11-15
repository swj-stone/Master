# Historical information available to the baseline selectors

The initial Gas Drift comparison allowed native RFE and ANOVA to select on all seven historical batches. The voting reference was fitted on batches 1–5, nominated features on batch 6 and used batch 7 only to check the nominated subset. The HAR comparison allowed native RFE to select on all 21 development participants. The voting procedure fitted its reference on 15 participants and used six other participants for ranking and voting. These differences can affect comparisons at the same feature count.

The supplementary analysis repeats the original baselines at shorter selection histories. Both original external studies had already been scored when the [supplementary grid](../../configs/context_review/matched_history.json) was written. Every combination in that grid is reported. No width, score threshold or split was chosen from these supplementary estimates. The subsequent refit always uses all eligible development data. The test units and endpoints are identical to those of the [Gas Drift](drift.md) and [HAR](external_har.md) studies. Reusing these cohorts adds a sensitivity analysis, with the same external observations and the same limitations on population inference.

## Gas Drift

Each cell shows the equal-batch mean macro recall, followed in parentheses by seconds for selection and final refit/prediction combined. Batches 1–5 contain 3,633 measurements, 1–6 contain 5,933, and 1–7 contain 9,546. Every final model uses all 9,546. The 5-column maximin procedure and 6-column separating vote in the original assessment attained 0.3465 and 0.3581 respectively.

| Method | Select on batches 1–5 | Select on batches 1–6 | Select on batches 1–7 |
|---|---:|---:|---:|
| Tree RFE, 5 columns | .4067 (7.46 s) | .5788 (6.57 s) | .5741 (8.44 s) |
| Tree RFE, 6 columns | .4172 (4.99 s) | .5263 (6.65 s) | .5780 (8.51 s) |
| ANOVA, 13 columns | .4342 (.43 s) | .5876 (.42 s) | .5995 (.45 s) |
| ANOVA, 32 columns | .6099 (.78 s) | .6229 (.78 s) | .6264 (.81 s) |

RFE attained higher mean macro recall than the corresponding voting method at both restricted histories. The comparison using batches 1–6 gives both RFE and the vote access to the same historical labels through nomination. RFE uses them jointly, whereas the vote uses the first five batches for fitting and the sixth for the proposal. RFE ran more slowly. The two ANOVA controls were fast, and their relative recall depended on the feature budget and selection history. All four full-history controls recovered the original columns and every element of the original prediction arrays. [Per-batch results](../../results/context_review/matched-history/gas-by-unit.csv) · [Means and measured times](../../results/context_review/matched-history/gas-summary.csv)

## Human Activity Recognition

All methods here retained 22 features and their final predictors were fitted on all 7,352 windows from the 21 development participants. Each of the nine external participants counted equally in the primary accuracy measure.

| Method and selection history | Mean participant accuracy | Pooled accuracy | Pooled macro AUC | Seconds |
|---|---:|---:|---:|---:|
| Separating vote, original 15 fit and 6 proposal participants | .89404 | .89854 | .99132 | 10.85 |
| Tree RFE, 15 initial-fit participants (5,116 windows) | .89469 | .89820 | .99066 | 29.42 |
| Tree RFE, all 21 participants (7,352 windows) | .90678 | .91042 | .99355 | 34.81 |

Restricting the RFE selector to the original 15 fitting participants reduced its advantage over voting from 1.2743 to 0.0649 percentage points in mean participant accuracy. RFE scored higher than the vote on five people and lower on four in that restricted comparison. Its result shows sensitivity to information allocation on this cohort. Only the vote could use labels from the six proposal participants in the comparison with 15-person RFE. The full-history control reproduced the original 22 columns and every predicted probability. The original voting time is from its own run. [Per-participant results](../../results/context_review/matched-history/har-by-unit.csv) · [Means and measured times](../../results/context_review/matched-history/har-summary.csv)

## Reproduction

From the thesis checkout, run the primary experiments first to create their frozen supports, split rows and predictions. The command below then trains the supplementary grid, bounded to one hour per dataset. If the supplementary predictions are already present, it checks their bindings and recalculates the scores. The audit reads the labeled source data and checks unit-level and aggregate outcomes against the saved probabilities using tied ranks for AUC.

```powershell
.\thesis-python.ps1 -m experiments.context_review.matched_history
.\thesis-python.ps1 -m experiments.context_review.matched_history_audit
```

The times measure a single run at each stated history and include the baseline's internal model fits as well as its selected-feature final refit and prediction. They exclude reading the archive, scoring and obtaining the widths from the original voting runs. Gas selection includes fitting its median imputer separately on each permitted history. The input is finite. All original native RFE settings, including its 20% elimination step, remain fixed. The ANOVA controls retain their original filter and budgets. The independent scorer verified the endpoints at all 54 method-by-batch or method-by-participant cells. The largest absolute error for a unit or dataset metric was 3.3 × 10⁻¹⁶. [Audit receipt](../../results/context_review/matched-history/audit.csv)
