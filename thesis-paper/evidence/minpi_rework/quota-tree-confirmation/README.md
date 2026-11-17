# Prospective sparse deployment of a fixed boosted tree ensemble

The signed protocol preceded outcome scoring. All 65 splits of the 13 preselected sources were evaluated. The 585 forecast arrays, their supports, source splits and code digests were locked in commit `c54b096` before release. The shared-data package `packages/minpi-quota-tree-forecasts.zip` is identified by `forecast-sha256.txt`. Every forecast accepts only its reported retained columns.

The primary selector follows exact greedy deletion until the next singleton deletion would exceed 20% relative Euclidean distortion of the calibration margins. The native full teacher, the minimum feasible Gain prefix and the ascending-Gain sweep provide deployment comparisons. Three exact greedy solvers evaluated the identical removal order and produced identical forecast arrays. A secondary arm chooses the smallest visited, fidelity-feasible support meeting an additional supervised calibration guard. A forecast's calibration fidelity does not guarantee a small change of outer AUC, accuracy or regression metrics.

| Thirteen-source comparison | Estimate | Source-bootstrap 95% interval |
|---|---:|---:|
| Retained width, exact greedy / Gain sweep | .894 | .847–.939 |
| Retained width, exact greedy / Gain prefix | .768 | .668–.860 |
| Retained width, exact greedy / full teacher | .398 | .259–.563 |
| Outer AUC difference versus Gain sweep (7 classification sources) | −.0025 | −.0083–.0019 |
| Outer accuracy difference versus Gain sweep (7 sources) | −.0015 | −.0055–.0022 |
| Outer R² difference versus Gain sweep (6 regression sources) | .120 | −.005–.366 |
| Outer MAE / SD difference versus Gain sweep (6 sources) | −.0153 | −.0508–.0045 |
| Guarded width, exact greedy / guarded Gain sweep | .864 | .809–.922 |

Each paired log-width ratio or score difference is first averaged across seeds within source. The bootstrap resamples whole sources 20,000 times. Five seeds do not supply five independent sources. The geometric width ratio was .906 on the eight-source core and .875 on the five wider sources. The estimate achieved the registered 10% target across all sources; its interval includes reductions below 10%. Robotlp lost .0176 AUC and .0111 accuracy against Gain sweep. Appliances increased R² by .734 and drives the pooled regression mean. Its chronological split is shared across seeds. Relative to the full teacher, CT Location lost about .057 R² and Superconductor about .049 R². Overall loss guarantees are unsupported.

## Runtime controls

- The signed Python/Numba implementation takes .706 times the archived dense SciPy exact search time; its identical no-projection cache baseline gives .949 [.887, 1.006]. Neither separates predictor quality because all three choose the same path.
- Subsequent compute-only controls use Numba for both alternatives. Compiled Gain sweep reproduces every previously locked Gain sweep deletion order. Native XGBoost probabilities/margins were verified against the full forecasts on all test inputs without accessing test labels in the timing code.
- The compiled four-bin implementation uses 3.42 times the direct compiled Gain selector time. Its training-inclusive geometric runtime ratio is 1.010 [1.003,1.018]. Compared with native full XGBoost, the complete ratio is 1.101 [1.082,1.119].
- The direct compiled exact greedy control also returns every locked intermediate deletion and uses a contiguous vector update for overlapping paths. It checks all competing norms and may abort a distance calculation once a 32-coordinate partial sum exceeds the incumbent. The compiled four-bin exact kernel takes 3.66 [2.80,4.66] times this simple search time. Within the same compiled squared-norm-cache engine, four bins reduce search time to .932 [.870,.991] of the zero-direction control. Selection savings versus the best straightforward exact control are unsupported.
- A separate fused kernel projects the current residual and candidate on the expected tree margin for a quarter, half or full coalition. Each direction bounds their cross term by applying Cauchy–Schwarz in its orthogonal complement. An inexpensive triangle bound with no direction cuts checked distances to .436 of the direct partial search and takes .968 [.912,1.008] of its time. The three-direction minimum cuts checked distances to .383 but takes 1.730 [1.434,2.054] of the same time. The full-coalition direction alone achieves effectively the same check count. These are computational ablations on unchanged forecasts.

`scores.csv` preserves original arm measurements and predictive metrics. `source-contrasts.csv` and `source-means.csv` contain the two panels, whole-source resamples and per-source means. `native-costs.csv` compares the signed implementation to compiled Gain and native full prediction; the alternative fully compiled implementations are in `compiled-exact-partial.csv` and `compiled-exact-partial-contrasts.csv`. The custom kernels use three timed repetitions after warming. Complete runtime combines measured shared training, required preprocessing, selector and restricted prediction. These are component-matched timings on the study machine.

## Numerical audit

The signed Gain-prefix implementation compared squared residual norms to an absolute squared-space slack. On a two-row stump with leaf margins ±10⁻⁸ that rule admits the empty support. The norm-scale correction rejects the empty prefix. The outcome-blind replay in `fidelity-audit.csv` reconstructed all 65 teachers and every signed support. The corrected smallest prefix is identical on all 65, and all 585 signed supports satisfy their physical radius without numerical slack (smallest radius minus realized residual norm .0002541). Use `experiments.minpi_rework.quota_compress_tree_v2.extract` for future work.
