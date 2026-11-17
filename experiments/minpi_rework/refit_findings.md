# Refit the receiver on the selected raw inputs

The spline ridge model fits on pooled fitting and voting rows. Column-wise basis functions use scaled linear terms and three hinges at the fitting quartiles. The full model nominates smaller supports by their squared-error fidelity to its intact predictions on the fitting pool. Integer-game MinPI screens precede exact enumeration. Keep-five and omit-five searches consider all columns; other widths optimize at most 24 columns around the cutoff in the energy ordering, with higher-ranked columns held fixed. Energy ranking, global improving one-for-one swaps, and GPU XGBoost Gain feed the same raw-column refitter and width guard. Gain uses the original XGBoost model parameters and receives all pooled fitting rows.

For each proposal, imputation, scaling, knots and ridge coefficients are fitted again using only its raw columns. The same fitted sparse receiver scores the width split and predicts the outer rows. Width options are one, five, all but five, the full width, and 12.5%, 25%, 50% and 75% of the full width rounded up, with endpoints clipped and duplicates removed. The smallest eligible tested width is chosen. All arms cache the full reference for validation. The selected model never receives values of omitted outer columns.

For classification, the guard requires the candidate's macro AUC to fall at most .01 below the full model. Let each width-split observation contribute the indicator that the full prediction was correct minus the indicator that the candidate was correct. Its mean plus one empirical standard error must be at most .01. For regression, paired absolute-loss and squared-loss increases are divided by the pooled fitting-response standard deviation and its square, respectively. Both means plus one empirical standard error must be at most .02. Each standard error is the sample SD of the differences divided by the square root of the number of labeled width rows. This is an empirical width-choice rule, with possible dependence within and between validation splits.

### Development outcomes

Each source contributes equally after averaging five seeds. Width and time comparisons use geometric means of paired ratios. The 18 UCI sources were previously used for development. The original six use their existing partitions and novel-input masks for outer scoring. Selection times measure complete adaptive pipelines on the development machine, with one BLAS thread per process, one observation per arm and split and overlapping development workloads.

| Panel | Width / full | Δ AUC | Δ accuracy | Δ R² | Δ scaled MAE |
|---|---:|---:|---:|---:|---:|
| Original five classification sources | .414 | +.0075 | +.0060 | | |
| Original Abalone regression | .615 | | | −.0023 | +.0057 |
| Nine previous UCI classification sources | .587 | −.0045 | −.0088 | | |
| Nine previous UCI regression sources | .506 | | | +.0033 | −.0071 |

Original classification sources averaged 1.097 times the width and .530 times the selection time of GPU Gain, with AUC +.0018 and accuracy −.0015. Against energy ranking they used 1.222 times the width and 1.539 times the time. Against greedy exchange the width ratio was .981 and the time ratio 1.381. For the nine other classification sources the respective width ratios versus energy and greedy were .926 and 1.012. Gains in compression or time depend on the control.

| Original dataset | Adaptive mean width | Full mean width | Sparse AUC / R² | Full AUC / R² | Sparse accuracy / MAE | Full accuracy / MAE |
|---|---:|---:|---:|---:|---:|---:|
| Madelon | 33.6 | 100 | .6836 | .6370 | .6262 | .5915 |
| Spambase | 33.6 | 55.8 | .9609 | .9616 | .9132 | .9158 |
| Gisette | 69.0 | 100 | .9653 | .9678 | .9223 | .9230 |
| Isolet | 79.0 | 100 | .9820 | .9821 | .7847 | .7858 |
| Credit Card Fraud | 10.0 | 29 | .9516 | .9573 | .9989 | .9990 |
| Abalone | 5.2 | 8 | .5604 | .5626 | 1.5345 | 1.5160 |

Forcing a five-column model on the original classification panel improved AUC by .0205 and accuracy by .0440 over five-column greedy exchange. The same comparison on the nine UCI classification sources yielded AUC −.0058 and accuracy −.0052. Against full width the fixed budget cost approximately ten accuracy percentage points on both panels. Sparse Madelon spline models also remain less accurate than the strong archived nonlinear receiver.

Removing the exchange screen while keeping the quadratic game, tie order, exact solver, proposal widths and all receiver fits fixed gave identical adaptive and fixed supports and bitwise-identical forecasts in 120 splits. Screened / unscreened adaptive selection time ratios were .708 on original classification, 1.019 on Abalone, .961 on the other classification panel and 1.031 on the other regression panel. Within original classification they were .286 for Madelon, .626 for Spambase, .979 for Credit Card Fraud, .998 for Isolet and 1.017 for Gisette. The source bootstrap interval for the .708 ratio was [.432, 1.002] across five source clusters. Each direct-comparison arm ran its own complete pipeline.

The compressed evidence holds all 1,500 rescored records and selected support identities. Fixed-policy standalone timings were not measured and are blank. The command below replays the model and guard invariants.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.refit_checks
.\thesis-python.ps1 -m experiments.minpi_rework.refit_delivery --run my-refit-check --names madelon spambase abalone gisette isolet credit_card_fraud --seeds 42 123 456 789 1024 --methods width greedy energy gain full
```
