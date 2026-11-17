# Prospective test of analytic coalition-size integration

The seven-source protocol was committed as `b6126b2`. Collection sealed all 35 source/seed jobs as `f2376cb` after [registering the sources](../../../../experiments/minpi_rework/validation_predictive.md). Each job trained its forest on the fitting sample and measured normalized Brier or squared-loss differences over the voting sample. Test-input predictions and path increments were stored before the release of the outer outcomes. The first table uses the voting sample. The second uses held-out outcomes after release.

All methods used the .02 quota, the same tie policy and exactly the same fitted probability or regression forest. A sampled bank was held fixed throughout each greedy path; reversing each permutation formed its antithetic partner. Both orientations counted toward the budget. The faster sparse empirical implementation accumulated the earliest and latest activation rank in one pass over a pair of permutations. An independent native search worked directly on masked calibration-row predictions and avoided constructing a Gram matrix. All corresponding empirical paths agreed, including the intermediate deletions. All 35 analytical paths matched both the production solver and an independent exact recount.

| Comparator | Exact/comparator selector time | Fit-inclusive ratio | Equal final supports | Equal full paths | Exact-quota violations | Median fixed-support error across sizes |
|---|---:|---:|---:|---:|---:|---:|
| Native exact recount | .976 | .999 | 35/35 | 35/35 | 0 | 0 |
| Sparse empirical weights, 8 permutations | .941 | .996 | 30/35 | 12/35 | 1 | .00486 |
| Sparse empirical weights, 32 permutations | .889 | .990 | 33/35 | 21/35 | 1 | .00220 |
| Sparse empirical weights, 128 permutations | .770 | .974 | 34/35 | 27/35 | 0 | .00116 |
| Direct masked rows, 8 permutations | 1.677 | 1.053 | 7/7 | 3/7 | 0 | .00675 |
| Direct masked rows, 32 permutations | .443 | .950 | 7/7 | 4/7 | 0 | .00241 |
| Direct masked rows, 128 permutations | .111 | .685 | 7/7 | 7/7 | 0 | .00127 |

The main 128-permutation selector comparison gave .770 [.690, .852] against the sparse implementation and .677 [.573, .782] against dense histograms. The analytical selector took a median 7.13 ms across the 35 jobs. Its ratio against the exact recount was .976 [.960, .991]. The ratios use log times averaged across seeds within source, then equally across sources; the intervals resample the seven sources. Training-inclusive figures add shared fit and calibration path extraction, excluding file I/O, JIT compilation and scoring. Selection includes the implementation's own required dictionary, moment setup and actual search. The 128-permutation estimator selected the identical support in 34 of 35 jobs and produced no exact-quota violation on these data.

All seven seed-42 jobs passed the prespecified Gram-free memory/work caps at all permutation budgets. Direct 8-permutation search took less time than the analytical selector on Beijing, Churn, German Credit, Liver and Votes, choosing the very same final support on all seven seed-42 jobs. The analytical selector was faster at that budget on Seoul and SGEMM. At 128 permutations it was faster on all seven. The choice of comparator, sampling precision and size of the path dictionary substantially affect the time ratio. These are computational comparisons on small feature sets; they give no evidence of an accuracy improvement from the minimum itself.

The [selector jobs](jobs.csv), [three repeated timings](repeats.csv) and [source-equal contrasts](contrasts.csv) preserve each measured component. The [lock](lock.json) and [archive digest](archive-sha256.txt) identify the prediction package. Replay the collector using `.\thesis-python.ps1 -m experiments.minpi_rework.predictive_mc_confirmation`.

## Held-out consequences

After both the predictive portfolio and the computational comparison had sealed all 35 jobs, the coordinator released their outcomes. I verified the selection and archive member hashes against the locked receipts, reconstructed the clean predictions from saved group increments and checked every method's exact held-out risk curve using two independently coded formulas. Held outcomes were used only for scoring. The reported normalized worst excess is the maximum across positive availability sizes. All seven sources contribute equally; AUC/accuracy cover four classification tasks, whereas R²/scaled MAE cover three regression tasks.

| Selector | Average width | Worst excess | Held quota violations | AUC | Accuracy | R² | Scaled MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Analytic | 6.229 | .026223 | 13/35 | .820291 | .814733 | .280877 | .587657 |
| 8 permutations | 6.229 | .026069 | 13/35 | .820323 | .814733 | .281172 | .587904 |
| 32 permutations | 6.200 | .026252 | 13/35 | .820323 | .814733 | .280827 | .588255 |
| 128 permutations | 6.229 | .026221 | 13/35 | .820291 | .814733 | .280860 | .587808 |

The same 13 jobs exceeded the held .02 quota for all four methods. Seoul exceeded it in five of five seeds, Liver in three, German Credit and Votes in two each, and SGEMM in one. On held worst excess, 8 permutations improved four jobs relative to exact greedy choice, worsened one and tied in thirty. At 128 permutations, one improved and 34 tied. The source-average difference at 128 permutations was −.000002, with a descriptive source-bootstrap interval from −.000006 to zero. These results give no predictive reason to prefer exact search. The evidence for it is the removal of integration error in the specified calibration game and the observed selector time at higher sampling precision.

The [released scores](released-scores.csv) and [paired source contrasts](released-contrasts.csv) retain all methods and source exceptions. Reproduce this stage using `.\thesis-python.ps1 -m experiments.minpi_rework.predictive_mc_score` after the joint release.
