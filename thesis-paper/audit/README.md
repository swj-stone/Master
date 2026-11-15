# Empirical audit

Run the following command from `MastersThesis` after both campaigns have published their effective indices.

```powershell
.\thesis-python.ps1 thesis-paper/audit/verify.py
.\thesis-python.ps1 thesis-paper/audit/verify.py --width
.\thesis-python.ps1 thesis-paper/audit/isolet_official.py --archive tmp/isolet-uci-official-20261115.zip
```

The verifier needs the local run archives under `results/thesis_benchmark`. It reads run receipts, input manifests, model settings, selection traces, frozen supports, test-access hashes and stored predictions. It recomputes the binary and multiclass AUCs, accuracy, precision, fraud AP, MAE and ordinary R² and replays each returned vote from its stored panel scores. This verifies the recorded panel decisions. It does not refit the tree for each mask. The independent score calculations use scikit-learn. Reproduction of the selection runs uses the experimental drivers.

| Output | Interpretation |
|---|---|
| `metrics.csv` | One row per effective arm with refitted guard and test scores. A cold time includes process startup, selection, training, scoring and teardown. |
| `paired-effects.csv` | Each contrast is MinPI minus its comparator on the same five dataset splits. The minimum, maximum and SD describe split variation. The five splits reuse observations. |
| `refit-vs-anchor.csv` | On each split, apply the masking vote's numerical tolerance to refitted scores for a diagnostic comparison with compact Gain. The vote itself uses masked scores of a fixed model. |
| `stability.csv` | Jaccard overlap of feature names across the ten pairs of splits of each dataset. The initial filtering and its feature universe may also vary across splits. |
| `fraud-uncertainty.csv` | A paired, class-stratified percentile bootstrap for Fraud AP differences. Within each split, resample the same positive and negative transactions for both fixed predictors, keeping the observed two class counts. Percentiles use 2,000 replicates per split; they describe transaction-level sampling uncertainty conditional on the fitted models and assume exchangeable rows within class. |
| `width.csv` | Three raw-width datasets, three arms, seed 42. The audit checks all array hashes and the equality of split indices and labels with the corresponding prepared-data runs. |

The equal-width Gain comparator receives the width of the already selected MinPI support. Its reported time excludes that donor selection and should not be interpreted as time to choose a width independently. Timing for conventional baselines from the earlier campaign is descriptive across runs. Dataset averages use a common metric; Abalone error and Fraud AP are presented separately.

## Findings on the repeated splits

All 96 primary and 90 contemporaneous control arms passed. Two original primary groups had late or missing completions and were repeated in full. The longest effective group took 69.73 s in the primary campaign and 45.09 s in the control campaign. All six eager and short-circuit pairs had the same sequence of proposals and decisions, final support and held-out prediction vector.

MinPI retained 29.01% of the supplied features, versus 34.54% for compact Gain and 32.90% for five-block MinPI. Their respective mean cold times were 6.56 s, 6.06 s and 6.35 s. At matched width, MinPI and Gain chose identical supports in 15 of 30 splits. On ISOLET, the new support had higher AUC in all five matched-width comparisons (mean difference +0.00077). On Fraud, MinPI lost 0.01925 AP on average at matched width and 0.03652 AP against the anchor's own width. The largest latter loss was 0.06146 on seed 42, with conditional paired bootstrap limits for the AP difference of −0.10149 and −0.02315.

Applying the proxy's numerical tolerance to *refitted* scores, 4 of 30 reserved validation comparisons and 5 of 30 test comparisons breached at least one limit against compact Gain. On Abalone and Madelon all ten MinPI selections equalled their original anchors. Cross-split mean Jaccard varied from 0.412 on Fraud to 0.895 on Abalone. These results delimit where compression, prediction quality and feature stability changed.

All nine original-width runs at seed 42 also passed the audit. On Madelon, MinPI and Gain both kept 8 of 500 columns and obtained AUC 0.89388. On Gisette, MinPI kept 10 of 5,000 columns, down from 33 with Gain, with an AUC loss of 0.01802. On ISOLET it kept 85 of 617, down from 154 with Gain, with an AUC loss of 0.00035. The MinPI cold time exceeded Gain on all three tasks.

## Secondary ISOLET speaker-group split

The standalone audit decodes both original UCI data members, compares all 4,810,749 feature values and the bijectively recoded labels with the local CSV, and verifies the row indices and training-only standardization. The official test file contains 1,559 utterances. The preceding 6,238 records are divided into 4,680 training rows (the first three speaker groups) and 1,558 validation rows (group four). Two records are missing from the first four groups combined. Both alternative cut positions would assign 61 examples of one letter to group four despite its maximum of 60.

All four arms passed. At equal width (96 features), MinPI correctly classified 43 utterances missed by the Gain prefix, and the prefix correctly classified 24 missed by MinPI. Against Gain at its own chosen width (158 features), these counts were 25 and 23. The full model correctly classified 14 more utterances than MinPI overall. Full per-arm timings and metrics are in `isolet-official.csv`. These data describe one speaker-group holdout separately from the repeated random splits. The archive is available from the [UCI repository](https://archive.ics.uci.edu/dataset/54/isolet).
