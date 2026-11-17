# Transfer of guarded deletion to six new UCI sources

The guarded deletion rule was fixed after a twelve-source retrospective study. The [six-source protocol](../../../../experiments/minpi_compression/witness_transfer.md) fixes the full source panel, file hashes, partition groups and metrics. Predictions and opportunity counts for all five methods and all thirty source/split pairs were sealed before outer scoring. The two local protocol and bound commits are 54cb7c6 and c0b3667. The sources had 6 to 100 input columns; Musk used the first 100 descriptors in archive order. Real Estate uses the same chronological holdout for every seed, and all other sources use grouped repeated splits.

| Source | Eligible splits / 5 | Accepted witness / tail / random | Mean coefficient → witness width | Coefficient → witness outer scores |
|---|---:|---:|---:|---|
| MAGIC | 0 | 0 / 0 / 0 | 3.4 → 3.4 | AUC .8324, accuracy .7745, identical |
| Musk v1 | 5 | 2 / 2 / 3 | 67.6 → 66.0 | AUC .7767 → .7716, accuracy .7204 → .7026 |
| Wall-Following | 0 | 0 / 0 / 0 | 19.2 → 19.2 | AUC .8613, accuracy .6385, identical |
| Real Estate | 5 | 0 / 5 / 1 | 5.0 → 5.0 | R² .5439, MAE 6.6074, identical |
| Computer Hardware | 2 | 2 / 0 / 2 | 2.8 → 2.4 | R² .5339 → .5038, MAE 51.13 → 52.47 |
| Forest Fires | 0 | 0 / 0 / 0 | 3.0 → 3.0 | Log-scale R² −.0097, MAE 1.1125, identical |

Before scoring, the opportunity bound showed that accepting every eligible batch could reduce the geometric coefficient-prefix width by at most 8.44% for classification and 12.67% for regression. The realized reductions were 3.55% and 5.93%. Their three-source bootstrap intervals extended from no reduction to 8.14% and 13.37%, respectively. In both domains the source-average predictive differences were within the registered margins, and neither met the 10% width threshold. The interval and the bound condition on different source weights. The bound uses the fixed panel, whereas the bootstrap reweights its sources.

On the fifteen classification splits the witnessed arm took 31.0% more selection time than coefficient selection (paired geometric ratio). On the fifteen regression splits it took 64.1% more. Its geometric width was equal to the coefficient-tail control in classification and 1.34% greater in regression. Relative to uniformly removing the same count, it was 1.19% and 1.50% wider. The matched controls each computed the same votes to determine the removal count and each spent 1,259 ranking factorizations, 170 prefix-validation fits, 12 extra fits and 30 deployment fits. The pure coefficient arm used 30 ranking factorizations, 170 prefix-validation fits and 30 deployment fits.

Among all 117 certified feature/split instances, 84 lay inside the baseline prefix. On one Computer Hardware split the only selected feature was negative. The protocol skips a proposal that would empty the model. Independent SVD refits verified all 84 within-prefix downward edges, 36 optional guard decisions and 150 deployed forecasts. On Musk, aggregating conformation scores to held-out molecules gave coefficient/witness AUC .8378/.8311 and accuracy .7263/.7158. On Forest Fires the inverse-transformed hectare-scale R² was −.0460, with MAE 12.60 hectares for both methods.

[Pre-score bounds by split](opportunity.csv) · [Bounds by source and domain](opportunity-summary.csv) · [All source scores, native metrics and times](sources.csv) · [Split-level predictions summary and selected columns](scores.csv) · [Unconditional paired contrasts and whole-source intervals](paired-effects.csv) · [Negative counts](negative-counts.csv) · [Guard audit](guard-audit.csv) · [Forecast and receipt archive](forecasts.zip)

## Reproduce

From the thesis checkout, run the commands below. Source acquisition requires network access on the first run. GNU gzip decodes the original Unix-compressed Musk file. Selection uses the existing benchmark environment and the original witness selector. A complete timed panel took 14.1 seconds on the original machine, excluding source download, scoring and the separate audit.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.witness_transfer_sources
.\thesis-python.ps1 -m experiments.minpi_compression.witness_transfer select
.\thesis-python.ps1 -m experiments.minpi_compression.witness_transfer score
.\thesis-python.ps1 -m experiments.minpi_compression.witness_transfer_audit
```

To replay recorded times and predictions with the exact recorded files, extract forecasts.zip into `results/minpi_compression/witness-transfer/` in a checkout containing the pinned source archives and evidence directory. Then run the scoring and audit commands. The checked path is fully contained in the thesis worktree. Three sources per domain limit inference about new tasks.
