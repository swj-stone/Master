# Integer interval extrema on the MinPI voting path

The complete ranking step was faster on the three wide sources. The change leaves the scientific estimator and its predictive outputs invariant. On four sources, the production algorithm took 46.6% of the strongest previous ranker's time (source bootstrap interval 31.6–69.4%). Measured from forest fitting through prediction, the ratio was 93.6% (85.3–100.3%). These data support a rank-time improvement. The prespecified 10% threshold for a full-pipeline advantage was missed.

| Source | Inputs | Production fit to prediction, s | Comparator, s | Paired rank-time ratio | Paired complete-time ratio |
|---|---:|---:|---:|---:|---:|
| Residential Building, Tehran | 79 | 0.099 | 0.098 | .899 | 1.002 |
| UJIIndoorLoc | 520 | 0.757 | 0.749 | .527 | 1.004 |
| QSAR oral toxicity | 1,024 | 0.835 | 1.059 | .312 | .809 |
| Internet Advertisements | 1,558 | 0.540 | 0.578 | .319 | .943 |

Seconds are source medians of per-split median wall-clock times from three interleaved complete fits. Ratios use paired log times over five seeds before taking the exponential. The per-split comparator was chosen from the original interleaved folded native endpoint veto, straight scalar exhaustive evaluation, and one-thread exhaustive BLAS ranks, including the original unfolded straight scan on Tehran. BLAS won on UJI and the folded endpoint veto won on the other three. Directly measured whole-pipeline times include group updates, RF fitting, vote and acceptance dictionaries, masks, native metrics, and actual predictions. Data loading and compilation are outside the clock. All outputs were checked against the locked selection order, support and forecast array. The 4-source intervals enumerate all 256 ordered bootstrap source resamples. The resulting intervals are marginal to each contrast and apply to these four sources.

## Why an interval can be skipped

### Cost against the single impurity path

The MDI-only selector fits the identical forest and uses its own calibrated path. It omits the voting-sample path dictionary and union-rank calculation. Across 20 signed comparisons, fitting the two-path portfolio and forecasting took 1.293 times as long (source bootstrap 1.035–1.616). The portfolio retained .883 as many nominal columns. Native traversal of its truncated trees used .893 as many distinct, reachable input columns (.826–.965). Both methods reproduced their original forecast arrays. On oral toxicity the two paths took 0.823 s, against 0.489 s for MDI alone, and used .802 times as many reachable inputs. On the two regression tasks the differences in both runtime and reachable inputs were smaller.

| Source | Two-path / MDI time, s | Time ratio | Reachable width ratio |
|---|---:|---:|---:|
| Residential Building, Tehran | .096 / .093 | 1.044 | .956 |
| UJIIndoorLoc | .728 / .703 | 1.025 | .974 |
| QSAR oral toxicity | .823 / .489 | 1.760 | .802 |
| Internet Advertisements | .539 / .366 | 1.485 | .851 |

The timing and support counts are in `mdi-repeats.csv` and `mdi-paired.csv`. Forecast quality by source is reported in the sibling `wide-veto/source-scores.csv`. The fit-to-forecast times can be reproduced with the pipeline module's `--mdi` flag.

### Interval certificate

Let the fixed number of input variables be `p`, and let `c_h` be the current candidate's signed excess-loss coefficient at degree `h`. The all-size score after rate normalization is

`F(s) = Σ c_h binom(s−1,h−1)/binom(p−1,h−1)`, for integers `1 ≤ s ≤ p`.

For `h ≥ 2` the forward difference of the basis is `binom(s−1,h−2)/binom(p−1,h−1)`. It is nonnegative and nondecreasing. The `h=1` term is constant. On an interval of edges, enclose each basis difference by its minimum and maximum, and accumulate a lower and upper bound respecting the sign of each coefficient. If the lower bound is positive, the right endpoint is a maximizer. A negative upper bound gives the left endpoint. Indeterminate intervals are bisected. Short intervals are evaluated directly. The previously worst size and the two global endpoints can veto a candidate immediately. Every visited value above the incumbent plus tolerance is a valid rejection witness.

The implementation uses the actual stored binary64 basis columns to preserve the original computer arithmetic. A dyadic extrema tree over stored forward differences costs `O(pD)` preparation for `D` coefficients. Nonzero differences are rounded outwards by one floating-point step. Each candidate evaluates scalar sums in the same order as straight enumeration. With finite products and sums, writing `M = Σ_h |c_h| max_s |w_sh|`, the difference between a computed endpoint score and its exact-real dot product is bounded by the usual sequential multiply-add error `γ_(2D) M` plus a subnormal allowance. The signed derivative-bound dot products require at most `2M` on the corresponding scale. The implementation demands a sign margin of `16(D+2)(εM+η)`, where `ε` is binary64 machine epsilon and `η` is its smallest positive subnormal. This covers both endpoint errors and derivative-bound accumulation at forest degrees (at most nine stored columns). If all contributing stored basis columns are constant on an interval, the evaluated scores are bitwise constant there. Direct evaluation handles the remaining uncertain intervals. Worst-case work per curve is still `O(pD)`.

Structurally dormant input features have zero remaining loss increment in exact arithmetic. The release implementation additionally checks that adding each putatively dormant increment leaves every stored coefficient unchanged before coalescing it. It then keeps the earliest index and emits consecutive equivalent deletions until the next retained ineligible feature. Removing such a column changes neither the current coefficients nor a live dependency group, which preserves the scalar comparator's index tie rule.

## Replication and scope

The original interval experiment was registered in commit `e782e07`. Three interleaved complete ranks on every split produced `repeats.csv`. Its first implementation used the original wide study's structural dormancy rule. The ratio against each split's strongest timed comparator was .472 [.308, .751]. Against the fastest exhaustive implementation alone it was .243 [.090, .608]. It evaluated 5.69% as many size scores as the strongest comparator, and also performed the separately recorded derivative-interval tests. Adding the original shared, exhaustive acceptance clock gave an estimated total ratio of .990 [.974, .999].

Commit `371ea8e` registered direct complete runs with common optimized acceptance. It follows the original wide portfolio's MDI-first width tie break. The historical seven-source deployment's tie break uses worst loss after width, and its original source-level outputs have their own separate deployment checks. The first complete run with the structural dormancy rule gave .933 [.848, .996]. The production finite-precision check has the separately reported .936 [.853, 1.003] result. The two complete-run timing collections reuse the same four sources. The elapsed times refer to this CPU, dataset size, and depth-four forest.

From the thesis worktree, run

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_polynomial_check
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_polynomial_timing
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_polynomial_report
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_polynomial_pipeline --production
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_polynomial_parity
```

The scalar check includes 1,950 random and adversarial curves and tolerance decisions, with 526 strict interior maxima and 493 nonunique maxima. A separate fixture constructs a dormant column carrying a one-ulp update residue. The unmodified native rank and production rank had identical full orders on 125 historical and 30 prepared-data splits. All 20 additional wide orders matched their hash-locked receipts. These 175 path digests are in `production-paths.csv`. The timing scripts require the corresponding data and wide receipts. For prior unfolded and coalesced controls on the same sources, see the sibling `wide-veto` evidence directory.
