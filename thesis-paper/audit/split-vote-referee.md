# Independent review of the split-coverage selector

## Before release of the external predictions

I compared the implementation of the 100-tree XGBoost arms, the data loaders, the timestamped protocol and the score generator. The external classifiers withhold whole operators or identical covariate vectors. The two regressions train before the held year. The selector receives masked outer labels, estimates threshold agreement using training rows and uses the combined vote and width rows to choose its support. Those two subsets form one selection set in this experiment. Both methods then refit on all development rows. The common preprocessed arrays and the common release of each source are appropriate for this comparison.

The user-defined decision compares the end-to-end fit-through-prediction time and the number of columns. The same-dataset pairing is essential. Median paired time and median paired width excess can each pass on different seeds. The number of seeds satisfying both conditions, and the number satisfying both alongside the outer quality tolerance, must accompany the dataset count. An observed loss below 0.01 on a handful of tasks is a descriptive check. It does not establish statistical noninferiority.

The exhaustive Gain path is a faithful, strong validation maximizer within the Gain order. It evaluates every width. The bounded selector evaluates at most six, and is allowed to trade 0.005 of validation score for a smaller support. Either change can save time or reduce width without requiring a vote. The mass-scheduled Gain and pooled arms therefore address the scientific comparison. The exhaustive near-maximum width also deserves a separate sensitivity result. Its outer performance needs its own final refit unless the support coincides with an already evaluated arm. Reusing the absolute-maximum forecast would attach the wrong score.

The cost boundary includes ranking and real refits. Each seed supplies a single timed execution in a shared worker with rotating order. The exhaustive helper serializes the XGBoost configuration and checks the GPU after every fit. Other arms check it once. This minor implementation difference and one-off GPU initialization matter for ratios close to one. A relative gain of a few percent from such runs requires replication before it can sustain a speed claim against a practical control.

### Theory and novelty

For a voter with $d$ representatives, conditioning on the candidate being absent leaves $n-1$ columns, among which $n-d$ cannot represent the voter. The ratio of the two binomial coefficients gives the probability that a uniform size-$k$ coalition misses all other representatives. This is an exact calculation for the weighted threshold game. The minimum across sizes concerns shares of pivotal weight after within-size normalization. It supplies neither a bound on predictive loss after refitting nor a guarantee over individual missing-feature patterns. Pooled and minimum votes agree if all still-uncovered positive-weight voters have a common multiplicity. On the six original seed-42 runs of the geometric schedule every complete order changed between the two rules. Only Gisette and ISOLET changed their final support.

The voters are marginal quantile splits from a training sketch. They do not preserve the thresholds, conditioning path or sample distribution of actual XGBoost nodes. Empirical adjusted agreement between splits and submodular feature coverage have substantial precedents. The scientifically distinct question is whether worst-size normalized pivotal credit leads to better candidates under a small refit budget. [Surrogate minimal depth for variable selection](https://academic.oup.com/bioinformatics/article/35/19/3663/5368013) and [Liu et al. on submodular feature selection](https://people.ece.uw.edu/bilmes/p/mypubs/liu-submodfeature2013-icassp.pdf).

The implementation forms an all-pairs agreement matrix on up to eight indicators per feature. This is quadratic in their total number even at a fixed sketch size. The registered external inputs have nine to 95 columns. Timings on them cannot justify scaling claims for raw omics panels or tens of thousands of inputs.

The four external sources were purposively selected. The two temporal regressions provide one held time period each, even though the training sketch and timing can vary over five seeds. The turbine endpoint uses a physical site already present in an earlier study. The bankruptcy split lacks company and date IDs. Counts on these sources describe the four measured tasks.

## Review after release of all 100 external forecasts

I verified the identity and SHA-256 of every forecast, recovered the correct held rows separately for each seed, and recomputed macro one-versus-rest AUC, accuracy, uncensored R² and MAE with scikit-learn. I paired receipts directly without importing the study's scorer or its summary functions. All 100 actual runs entered the review.

| Source | Median width, Min / exhaustive Gain | Median seconds, Min / Gain | Median paired outer AUC or R² difference | Seeds passing speed and width | Seeds also passing quality |
|---|---:|---:|---:|---:|---:|
| Postures | 28 / 29 | 4.698 / 24.228 | −0.00094 | 5/5 | 5/5 |
| Bankruptcy | 30 / 61 | 1.378 / 17.187 | +0.00009 | 4/5 | 4/5 |
| Turbine CO | 2 / 2 | 0.769 / 1.485 | 0.00000 | 5/5 | 5/5 |
| Temperature | 16 / 14 | 1.071 / 3.370 | +0.00239 | 2/5 | 1/5 |

Three of four sources pass the original operational rule. The five repeated seeds yield 16/20 jointly faster and sufficiently narrow runs, of which 15 also meet the quality tolerance. Counting seeds would change the operational decision from a failure to a nominal 80% pass. The registered unit is the dataset. The external panel therefore fails.

The common 0.005 validation allowance materially changes the width comparison. Relative to the exhaustive near-best Gain widths, only Postures and Turbine meet the width criterion. Bankruptcy's median Gain width falls from 61 to 27. Its median paired width excess over the allowance becomes two columns. In the temperature series this excess is one.

Exactly two sources pass against each practical control. Postures and temperature pass against pooled, although on all ten of those paired fits the two algorithms deploy the same features and forecasts. On Turbine the minimum improves median R² by 0.04994 over pooled, and all five of its supports coincide with ordinary mass-scheduled Gain. The external results support a short native refit budget as the source of the large runtime saving against exhaustive Gain. The deployment evidence supplies no general advantage for the minimum rule.

**Verdict for the second iteration.** Reject a general-purpose algorithm paper claiming the registered success threshold or incremental predictive performance. The exact counting identity, the counterexample distinguishing voting rules and the measured cost/quality trade-offs can stand with their empirical scope. The four external sources have now been opened. A revised policy evaluated on them supplies a retrospective comparison. A prospective claim for that policy needs a separately frozen evaluation.

## Third iteration before release

The revised selector measures at most four ordinary Gain prefixes and freezes a validation floor. The anchor is the narrowest measured ordinary Gain prefix within 0.005 of the best measured value. Every accepted substitution remains inside this anchor and meets the frozen floor, up to floating-point tolerance $10^{-14}$. There are at most six validation fits and one deployment fit. Ordinary Gain with the same restricted search enjoys the same bounds. The comparison with an exhaustive prefix search answers the user's engineering question. The comparison with this restricted Gain search and with pooled votes tests the incremental effect of changing the order inside the anchor.

This guarantee is local to the measured reference prefixes. A nonmonotone curve can have a better unmeasured peak, an unmeasured narrower acceptable support, or both. A measured validation floor gives no population assurance for the refit. An empty set is never proposed. The revised algorithm changes the width search and the admissible support family as well as the place where voting is used. Differences between the rejected second iteration on four tasks and the third on five other tasks do not isolate any one of these changes.

The primary hardware also changes to single-thread CPU in the third panel. Each new comparison has the same device, common warmed process, three timings per seed and a rotated order. The second and third panels are unsuitable for an across-panels speed ratio. The common-slack exhaustive time adds the independent cost of its final fit and prediction to the measured exhaustive search. This captures its standalone work with the same shared search, although the alternative fit executes second.
