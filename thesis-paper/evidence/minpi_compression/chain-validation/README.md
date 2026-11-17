# Chain search on four new UCI datasets

Each source contributed five development splits. The Landsat test set is identical across its five repetitions. The [protocol](../../../../experiments/minpi_compression/chain_validation_protocol.md) records the source citations, composition of the test sets and decision criteria. Exact copies of feature vectors were grouped before splitting. Landsat, Biodegradation, Aquatic Toxicity and Power Plant have 6,435, 1,052, 537 and 9,527 distinct predictor vectors.

| Source | Input columns | Mean columns kept | Outer metric, chain / full | Second metric, chain / full | Mean selection time |
|---|---:|---:|---|---|---:|
| Landsat | 36 | 14.4 | AUC .9125 / .9145 | accuracy .7343 / .7445 | 66.3 ms |
| Biodegradation | 41 | 31.0 | AUC .9099 / .9149 | accuracy .8499 / .8565 | 24.0 ms |
| Aquatic Toxicity | 8 | 6.0 | R² .4920 / .5002 | MAE .9254 / .9164 | 9.1 ms |
| Power Plant | 4 | 2.0 | R² .9184 / .9306 | MAE 3.8773 / 3.6045 MW | 10.4 ms |

The table averages five split scores and five within-split median times arithmetically. The paired protocol uses geometric ratios for widths and times. Relative to full ridge, chain MinPI retained 52.4% of classification predictors and 59.8% of regression predictors, passing both domains' point criteria. Landsat alone exceeded its accuracy allowance by 0.02 percentage points.

Relative to the previous individual-addition sampler, chain search took 92.3% of the selection time and retained 97.2% as many columns on the two classification tasks. On the two regression tasks the corresponding figures were 77.3% and 102.9%. Neither domain met the joint 10% speed and no-larger-width criterion. Relative to univariate F, chain search retained 78.2% and 87.1% as many columns while taking 9.2% and 14.4% longer, respectively. Relative to the coefficient control, it changed four of 20 supports and took 10.4% and 17.3% longer. [All methods, scores and time estimates](effects.csv) · [Per-source means](sources.csv) · [Individual splits and selected columns](scores.csv)

A matched control received the same brackets and the exact count of additional distinct coalition queries as bisection. In 14 of the 20 splits that budget was zero. Across the 20 splits bisection detected nine negative feature indices, whereas uniform selection among interior prefixes detected six. Their supports agreed on 19 splits. The previous individual-addition method found 43 negatives. Independent exhaustive SVD refits on the two sources with eight or fewer columns found 12 true negatives, all on Aquatic Toxicity. Both bisection and random prefixes found five. The previous sampler found six. The exact reference covers 60 feature/split pairs. [Query-matched control](matched-control.csv) · [Exact indices](exact-minima.csv)

## Reproduce

Run from the thesis checkout using the shared environment. The first command downloads the four hash-pinned UCI archives into the local ignored data directory. Selection writes sealed receipts and predictions. The subsequent scoring verifies all 120 pipeline predictions and all 20 control predictions against separate SVD fits.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.chain_validation_sources
.\thesis-python.ps1 -m experiments.minpi_compression.chain_validation select
.\thesis-python.ps1 -m experiments.minpi_compression.chain_validation_control select
.\thesis-python.ps1 -m experiments.minpi_compression.chain_validation score
.\thesis-python.ps1 -m experiments.minpi_compression.chain_validation_control audit
```

The [primary archive](forecasts.zip) contains split indices, rankings, inner curves, clocks, all outer forecasts, SHA-256 digests and the selection receipts. The [random-control archive](matched-forecasts.zip) contains its rankings, curves, budgets, forecasts and source hashes. To rescore saved forecasts with source files matching the recorded run, extract these archives into `results/minpi_compression/chain-validation/` and its `matched-control/` subdirectory, respectively, and run the two scoring commands after acquiring the raw data. Fresh timings depend on the hardware and concurrent load. The illustrative speed ratios refer to the implementations in this checkout.
