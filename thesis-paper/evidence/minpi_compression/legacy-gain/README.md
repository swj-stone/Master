# Ridge selection against the archived direct Gain ranker

This transfer uses the 30 original prepared splits for Abalone, Credit Card Fraud, Gisette, ISOLET, Madelon and Spambase. The original train-only variance filter, mutual-information cap and standardization were applied before the new experiment. The validation partition was divided between voting and evaluating prefix widths. The outer row split is historical. ISOLET uses random rows and Credit Card Fraud uses random transactions. Results from these folds describe their original row distributions.

The [protocol](../../../../experiments/minpi_compression/legacy_gain_protocol.md) specifies the direct Gain ranker, common downstream predictor and point success criteria. Its frozen XGBoost model uses CUDA histogram trees, 100 estimators, maximum depth four and learning rate 0.1, with descending feature index on equal mean split gain. Ranking costs include fitting the GPU model, reading its importances and sorting. The three-repeat times reflect a warm process on one host core (logical CPU 8 on Windows), with a local RTX 5070 Laptop GPU available to Gain. Earlier PIv1 and exhaustive Gain timings used XGBoost for the downstream scan and evaluated different candidates.

The ridge transfer protocol was registered in commit `24e3970`. Exact-input overlap counts preceded the novel-input sensitivity protocol in `eaad764`; the outer ridge scoring followed both. The nonlinear receiver and common XGBoost guard were added after examining the earlier ridge and historical XGBoost scores, and registered in `6cab95c` before calculating their own outcomes. Commit `90a26d0` pins the code that generated the nonlinear forecasts.

From the isolated thesis checkout, the following commands use the original prepared files and the archived Gain orders in `packages/thesis-piv1-evidence.zip`. From a fresh checkout with the package available, extract its `results/icomp2026_formal/` subtree in the same relative location first. The archive's SHA-256 is listed in [the PIv1 release](../../../../docs/thesis/archives/piv1/evidence.sha256).

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_integrity audit
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain select
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain score
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_integrity score
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_replay export
```

The selector prints elapsed time per repeat and elapsed and estimated remaining time per completed split. Gain and MinPI fit their ranking models on the same original training rows. MinPI also uses the reserved voting rows to label coalitions. Every width fit uses the pooled training and voting rows and the same width-validation rows. The prediction files are locked before scoring. Durations exclude data loading, writing traces and outer scoring. A 3,600-second watchdog limits each dataset/seed. Published prediction scores can also be recomputed from the recorded forecast package with `legacy_gain_replay verify` and the shared Python environment.

[Paired differences and ratios](contrasts.csv) · [Per-source means in original metric units](sources.csv) · [All 120 scores and supports](scores.csv) · [Overlap by dataset and seed](split-overlap.csv) · [Frozen forecasts scored on novel test inputs](novel-input-scores.csv) · [Forecasts, traces and test targets](forecasts.zip)

## Matched ranking comparison and historical deployment

On the five classification sources the direct Gain ranker reproduced its archived order in all 25 splits, as it did in the five regression splits. Using ridge for width selection and final prediction, MinPI took 0.272 times as long to select and kept 0.906 times as many features as Gain (paired geometric ratios). The source-resampling ranges were [0.129, 0.524] for time and [0.719, 1.201] for width. The AUC difference was +0.00001 and the accuracy difference was −0.00126. After removing test rows with exact development-vector matches the differences were +0.00002 and −0.00101. Each of 3,125 source draws resamples five source means with replacement. On Abalone alone the selection-time ratio was 0.097, the width ratio 0.977 and the R² difference +0.0074.

The coefficient control selected exactly the same support as MinPI in 25 of the 30 splits. Across the five classification sources MinPI took 1.096 times the coefficient control's selection time and kept 1.012 times as many columns. Against the previously deployed exhaustive-prefix Gain **with XGBoost prediction**, the ridge-MinPI pipeline had average AUC lower by 0.0583 and accuracy lower by 0.0744. The ridge speed comparison and the historical deployment comparison answer different practical questions. [Historical predictive scores](historical-scores.csv) · [Classification source resampling](classification-source-intervals.csv)

## Nonlinear receiver and width guard

The [frozen-support transfer](../../../../experiments/minpi_compression/legacy_gain_receiver.md) fits the earlier XGBoost learner on the 120 already selected sets. The [strong-guard protocol](../../../../experiments/minpi_compression/legacy_gain_strong_guard.md) reruns the four methods and uses the same XGBoost scoring rule and receiver for all widths. The earlier datasets and test outcomes were known before these extensions. Each strong-guard method has one fresh end-to-end timing for each of the 30 original splits. The saved traces include individual width scores and elapsed ranking, selection, and final deployment time. The self-contained prediction package references the test targets included in the parent ridge forecast archive.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_summary ridge
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_receiver select
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_receiver score
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_summary receiver
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_receiver_audit
.\thesis-python.ps1 -m experiments.minpi_compression.legacy_gain_binary
```

With XGBoost for both width validation and prediction, MinPI's five-source classification AUC and accuracy were 0.0036 and 0.0051 lower than matched Gain. The geometric mean width ratio was 1.467 [1.184, 1.770]. Selection time ratio was 0.896, and the deployment ratio including final fitting and prediction was 0.926. On Abalone the width ratio was 1.516 and selection time ratio 0.920, with R² higher by 0.0029 and MAE lower by 0.0065 rings. Both domains missed the predefined combination of quality, time and width conditions against Gain. Against full XGBoost the width ratios were 0.520 and 0.435, respectively. Strong MinPI's five-source classification differences from the earlier exhaustive-prefix XGBoost deployment shrank to −0.0088 AUC and −0.0089 accuracy. The corresponding source-resampling ranges were [−0.0247, −0.0002] and [−0.0200, −0.0014]. [Strong-guard size, cost and quality](xgb-receiver/contrasts.csv) · [Every source and receiver](xgb-receiver/sources.csv) · [Fold-level scores and clocks](xgb-receiver/scores.csv)

The second scorer reconstructed all selected supports from the stored ranks and inner curves, and independently scored 120 fresh pipelines, 120 fixed-support refits and 60 historical predictions. Summing binary-class score midranks required float64 on Credit Card Fraud. Native historical binary AUC uses the positive-class score alone; averaging both one-vs-rest AUCs can differ by up to 0.00000283 due to ties in float32 complements. This table uses the common averaging convention. In an exploratory precision-recall check across the four binary datasets, MinPI on Credit Card Fraud had average precision 0.761 and recall 0.435 with ridge, compared with 0.812 and 0.763 under the strong XGBoost guard. The positive prevalence averaged 0.00172. [Binary outcomes](xgb-receiver/binary-sources.csv)
