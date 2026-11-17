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

## Third iteration after release

All 375 receipts from the five prespecified tasks, five seeds, five timed arms and three repetitions passed independent checks. The verifier matched forecast hashes, raw-input row IDs, complete supports, all reference utilities and repeated forecast arrays before the first read of outer responses. All three selectors used exactly the same anchor in every task and seed. Their observed Gain reference scores agreed with the exhaustive control's scores at the same widths. The minimum selected a proper subset of its anchor in 10/25 cases. All five garment and all five shopping cases retained their entire anchor.

| Control | Sources passing speed and width | Sources also passing median quality | Sources at least 5% faster with width and quality | Same-seed successes, speed/width/quality | Changed final supports |
|---|---:|---:|---:|---:|---:|
| Exhaustive Gain, validation maximum | 5/5 | 5/5 | 5/5 | 23/25 | 18/25 |
| Exhaustive Gain, 0.005 allowance | 5/5 | 5/5 | 5/5 | 24/25 | 12/25 |
| Geometric Gain | 4/5 | 4/5 | 2/5 | 18/25 | 14/25 |
| Gain with identical anchor and two refinements | 0/5 | 0/5 | 0/5 | 2/25 | 9/25 |
| Mean with identical anchor and two refinements | 2/5 | 2/5 | 0/5 | 10/25 | 0/25 |

The predeclared engineering target is met against the named exhaustive comparator even with equal validation slack. All 25 task/seed pairs pass on time and width against both exhaustive controls. The five median time ratios against the validation-maximum control range from 0.536 to 0.745. The uniform quality claim fails at the run level. On Vowel, seed 42 lost 0.0242 AUC and 3.0 percentage points of accuracy. Seed 46 lost 0.0169 AUC and 8.0 points of accuracy. The repeated seeds reuse the same held speakers. Most prospective inputs have few columns, and this panel alone cannot validate an assertion about hundred-column performance.

The minimum never changed a final support or forecast relative to the pooled mean on the 25 prospective task/seed pairs. Its empirical incremental predictive benefit on these comparisons is exactly zero. Tetouan offers a case where the voted *order* matters. Against ordinary Gain with the identical anchor, it retained six columns rather than nine, gained 0.00134 median paired R² and took 1.199 times as long. The mean vote produced the same forecasts. The mechanism that reduced runtimes against exhaustive Gain is the bounded refit budget.

**Verdict.** The revised procedure meets the user's operational stopping rule on the prospective five-task panel. The evidence does not justify a broad methods claim that taking the minimum makes feature selection predictively superior, or that adding voting to this same-budget Gain procedure accelerates it. A manuscript may report the selection algorithm, its limited guarantee, the practical benchmark and the null ablation together. The tests do not establish a universal win or acceptance at a selective venue. [Verified predictions and timing ledger](../evidence/anchored_vote/receipts.csv), [paired seed scores](../evidence/anchored_vote/seeds.csv), [source comparisons](../evidence/anchored_vote/sources.csv).

## Manuscript review of the prospective results

I compared the canonical manuscript's abstract, methods and complete five-row table with the implementation, the recorded arm-level results and the original MinPI paper. The source successes, median columns, durations, AUC, accuracy, R² and MAE agree to publication rounding. The prose discloses both Vowel losses, the matched Gain control's time advantage, the 0/25 minimum-versus-mean support changes and the distinct earlier failure. It correctly distinguishes the fixed test cohorts from the resampled HCV test set.

Two technical statements required revision. The validation floor gives a lower bound on the selected score. It places no upper bound on improvement beyond the best observed anchor prefix. For equal gain and voting score the implementation chooses the later column in the supplied anchor order. That position need not be its higher original matrix index. I also requested explicit reuse of the initial model's total gains. All three corrections were communicated to the author. The prospective table gives marginal medians, and the pass rule uses paired differences. Its explicit caption correctly allows these summaries to differ.

The older ternary minimum of individual predictive marginals could have a negative harmful-feature witness. The new minimum acts on nonnegative normalized means of coverage marginals, averaged separately at three coalition sizes. Exact combinatorial counting for that change does not preserve the earlier interpretation. The manuscript should state this design change even though it cites both the original MinPI and earlier coverage work. [Original index definition](https://arxiv.org/abs/2610.33885).

**Submission judgment.** I would not submit the prospective-only draft as a top-conference demonstration of a superior minimum-voting algorithm. The primary benchmark supports the bounded search as an engineering option. The direct same-budget control was faster on all five datasets and the pooled game yielded the same predictions for every seed. A narrower empirical methods paper that makes this trade-off and the scope of the exact result its subject is scientifically defensible. Any later claim for higher-dimensional data needs its own paired measurements and an explicit account of timing sessions.
