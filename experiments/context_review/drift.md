# Forward test on the UCI Gas Sensor Array Drift Dataset

On three successive gas-monitoring batches, the 5-column maximin subset had mean macro recall 0.3465. A 5-column tree RFE subset reached 0.5741. A separate refit on batch 7 rejected the maximin subset, the weakest-Gain deletion and the 6-column voting subset. Each guarded method consequently used the same 42 columns as Compact Gain. This cost more computation than simply fitting the anchor. All 42 columns together require readings from the full set of 16 physical sensors.

## Population and split

One laboratory sensor array measured six gases. Its 128 predictors represent eight response summaries per sensor. The archive contains 13,910 records in ten chronological batches. The split in [the study protocol](../../configs/context_review/drift.json) was fixed from repository metadata before the archive was acquired. The experiment used the corrected batch 10 currently distributed by UCI. UCI displays CC BY 4.0 and retains a research-use restriction in the donor description. The reproducer downloads the data from UCI. [Dataset and collection notes](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) · [Vergara et al., 2012](https://doi.org/10.1016/j.snb.2012.01.074)

| Batches | Purpose | Rows |
|---|---|---:|
| 1–5 | Fit the initial classifier and establish the Gain order | 3,633 |
| 6 | Choose the compact prefix and run the paired voting panels | 2,300 |
| 7 | Compare independently refitted candidates with the refitted prefix | 3,613 |
| 8 | Future evaluation, months 22–23 | 294 |
| 9 | Future evaluation, months 24 and 30 | 470 |
| 10 | Future evaluation, month 36 | 3,600 |

Toluene is absent in batches 3–5. The initial model pools batches 1–5, preserving 79 toluene examples. All subsequent proposal, guard and test batches include all six gases. The guard fits on batches 1–6. Every final predictor fits on batches 1–7. Forward test scores average the three batch means equally. Recall weights each gas equally within its batch. Accuracy gives each record equal weight within batch. AUC averages the six one-versus-rest AUCs within batch. The three test batches come from one array and do not provide three independent site replications.

## Feature selectors and costs

The published experiment code fixes the XGBoost predictor to 100 depth-4 trees with learning rate 0.1, CPU histograms, one thread and seed 20421. Original vote, random-family vote and both one-deletion challengers start from the same Gain anchor. Maximin chooses which of the six surviving columns to delete using the worst-context panel count. The weakest-Gain control spends the same one-deletion budget. Batch 7 accepts a refitted candidate if its AUC and accuracy each fall within the prespecified 3% relative loss bound against the anchor. The fallback is the original stopping subset if that passes, otherwise the anchor.

The conditional RFE and Gain-prefix controls receive their target widths from the unguarded vote or challenger. Their reported times omit discovering those widths. RFE uses native scikit-learn recursive elimination with the same XGBoost estimator and a 20% step. ANOVA uses native `SelectKBest(f_classif)` at two widths fixed before data acquisition. RFE and ANOVA select using all historical batches 1–7. This gives them more labeled selection data than the proposal stage. All arms get identical historical data for final fitting. The reported seconds sum the recorded stages needed for one path, including its final refit and prediction. They exclude reading the archive, verification, timing the brute-force oracle, and scoring. These are single-run observations, with shared stages measured once. The complete selection and prediction worker took about 34 seconds on this machine.

| Method | Columns | Sensors used | Macro recall | Accuracy | Macro AUC | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Full | 128 | 16 | .6802 | .6978 | .9162 | 2.81 |
| Compact Gain | 42 | 16 | .6175 | .6451 | .9155 | 2.74 |
| Vote | 6 | 4 | .3581 | .3778 | .7443 | 2.44 |
| Random-family vote | 5 | 3 | .3465 | .3831 | .7226 | 2.41 |
| Maximin | 5 | 3 | .3465 | .3831 | .7226 | 2.48 |
| Weakest-Gain deletion | 5 | 4 | .3652 | .3838 | .7413 | 2.43 |
| Vote with guard | 42 | 16 | .6175 | .6451 | .9155 | 4.11 |
| Maximin with guard | 42 | 16 | .6175 | .6451 | .9155 | 4.34 |
| Weakest-Gain deletion with guard | 42 | 16 | .6175 | .6451 | .9155 | 4.30 |
| Gain prefix at vote width | 6 | 4 | .4392 | .4296 | .8373 | 2.04 |
| Tree RFE at vote width | 6 | 6 | .5780 | .5939 | .8623 | 8.87 |
| Tree RFE at challenger width | 5 | 5 | .5741 | .5890 | .8583 | 8.84 |
| ANOVA, 10% | 13 | 7 | .5995 | .6428 | .8475 | .42 |
| ANOVA, 25% | 32 | 12 | .6264 | .6184 | .8931 | .80 |

The 42-column anchor retained almost all of the full predictor's AUC while losing 6.27 percentage points of equal-batch macro recall. The guard compared candidates with that anchor. It had no mechanism for detecting the anchor's own loss against the full model. RFE's 5 columns gave higher recall in each future batch than the 5 chosen by maximin. The lazy oracle made 26 context visits and returned the same nominee and minimum count as all 66 visits in the exhaustive scan. This saving in proposal computation did not improve the selected subset's forward scores. Physical sensor counts measure coverage of the selected columns. Acquisition energy and latency were not measured.

## Reproduction and audit

Run from the thesis worktree. The first command fetches the pinned UCI archive when absent. In a fresh checkout without frozen outputs it also trains all arms, hashes their supports and predictions, then evaluates the future batches. With frozen outputs present it verifies those hashes and re-scores the stored predictions. The final command independently refits every distinct final support.

```powershell
.\thesis-python.ps1 -m experiments.context_review.drift
.\thesis-python.ps1 -m experiments.context_review.drift_costs
.\thesis-python.ps1 -m experiments.context_review.drift_audit
```

The selection worker read the feature fields of future records and discarded each leading label field. A separate scoring stage accessed those labels after all 14 prediction arrays were saved and checksummed. This local binding records an evaluation conducted on publicly labeled data. A first worker stopped when scikit-learn's univariate selector lacked `support_`. Replacing that access with its documented `get_support(indices=True)` method allowed the unchanged frozen design to run. The failed worker stopped before producing or scoring any future predictions. Both source bindings are archived. The second worker froze its outputs at 09:42:27 UTC on 15 November 2026.

The [audit](drift_audit.py) reread the upstream archive with a separate LIBSVM parser, checked both ANOVA rankings, replayed RFE elimination, refitted the four guard models and all ten distinct final supports, and recomputed AUC via tied ranks. All saved probabilities matched exactly. Across 42 method-by-batch cells the largest absolute difference over the three metrics was 2.2 × 10⁻¹⁶. The full [per-batch scores](../../results/context_review/drift-locked/by-batch.csv), [stage times](../../results/context_review/drift-locked/costs.csv), [audit receipt](../../results/context_review/drift-locked/audit.csv), and [supports and guard scores](../../results/context_review/drift-locked/supports.json) are retained alongside the probabilities.
