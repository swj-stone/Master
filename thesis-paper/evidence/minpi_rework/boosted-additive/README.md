# Quadratic marginal votes from a boosted teacher

The teacher was the archived CUDA XGBoost configuration, fit on the original fit rows. Its exact TreeSHAP contributions on the separate voting partition supplied one vector per feature, row and class. Each vector was computed with the entire row available. A support in this game sums selected attribution vectors; it does not mask the original row and query XGBoost again. The downstream XGBoost was actually refit on the proposed columns. Its validation procedure used the archived row partitions and accuracy/AUC or MAE/R² floors. Every recomputed Gain order, selected width and queried validation metric reproduced the archive. [XGBoost prediction interface](https://xgboost.readthedocs.io/en/stable/prediction.html)

In the primary game the target was the teacher's raw margin and the loss was squared error. Summing all contributions plus the bias had to reconstruct that margin. The exact minimum, pooled and full-context rules used the same additive quadratic game and backward deletion path. SHAP energy and mean absolute SHAP supplied static rankings. The supervised sensitivity used observed voting labels and a Brier loss for classification, or squared error in target units for regression. For classification the code allocates the change in each class exponential by its secant slope, then subtracts the normalization term. The resulting per-feature vectors sum exactly to the full probability change. This is a separate allocation in probability space; only the raw-margin vectors have the standard TreeSHAP interpretation.

## Original sources

| Source, seed 42 | Gain | Raw minimum | Pooled | Full context | Energy | Mean absolute SHAP |
|---|---:|---:|---:|---:|---:|---:|
| Abalone | 4 | 8 | 8 | 4 | 4 | 4 |
| Credit card fraud | 8 | 4 | 8 | 4 | 4 | 4 |
| Gisette | 25 | 13 | 25 | 25 | 25 | 25 |
| ISOLET | 50 | 75 | 50 | 50 | 50 | 50 |
| Madelon | 13 | 13 | 13 | 13 | 13 | 25 |
| Spambase | 14 | 28 | 28 | 28 | 14 | 28 |

The width reductions on Gisette and fraud motivated completing all five historical seeds for those two sources. On Gisette the minimum yielded 13, 25, 50, 75 and 50 columns. Direct Gain yielded 25, 25, 50, 50 and 25 in the same seed order. Allowing pairs of features to leave together gave the same minimum-rule widths as the singleton path on all six seed-42 splits. All fitted variants, their inner metrics, the number of guard fits and the component clocks are in the [development table](development.csv). The remaining four sources were stopped after one seed.

## Historical prediction on credit card fraud

The five accepted minimum-rule widths were 4, 4, 4, 8 and 8. The full-context rule selected those same widths. The two methods selected identical columns on four splits. On seed 789 they differed on two of the eight selected columns, accounting for their entire mean predictive difference. The split-specific orders were frozen before final training. All five sets of forecasts were sealed before scoring. Full and Gain forecasts reproduced the archived arrays exactly. These splits were studied previously, and the source was chosen for this follow-up after observing its width reductions. The following averages describe this source. [Per-seed measurements](fraud-historical.csv) · [Prediction digests](fraud-lock.json)

| Method | Mean width | AUC | Accuracy | Average precision |
|---|---:|---:|---:|---:|
| Full | 29.0 | 0.97340 | 0.99947 | 0.80832 |
| Direct Gain | 10.8 | 0.97108 | 0.99941 | 0.78075 |
| Raw minimum | 5.6 | 0.96296 | 0.99940 | 0.76907 |
| Pooled | 8.6 | 0.97252 | 0.99940 | 0.78227 |
| Full context | 5.6 | 0.96057 | 0.99936 | 0.75834 |
| Energy | 6.2 | 0.97310 | 0.99936 | 0.76076 |
| Mean absolute SHAP | 6.2 | 0.97380 | 0.99930 | 0.75162 |

The minimum reduced mean width from 10.8 to 5.6 and lost 0.00812 AUC against Gain. Against the full model it lost 0.01044 AUC and 0.03925 average precision. Accuracy barely changed in this heavily imbalanced task. On average the minimum required 3.4 strong width fits, compared with 4.4 for Gain and 3.4 for point, energy and absolute SHAP. Ranking cost including the teacher fit averaged 1.18 s for the minimum and 1.11 s for Gain. Measured components of the ascending selection path summed to 3.42 and 4.06 s, respectively. Identical fits were shared across methods during development and some GPU jobs overlapped. These clocks cannot establish a stand-alone runtime advantage.

## Reproduction

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive_check
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive --datasets gisette credit_card_fraud --seeds 123 456 789 1024
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive --pairs
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive --export
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive_fraud deploy
.\thesis-python.ps1 -m experiments.minpi_rework.boosted_additive_fraud score
```
