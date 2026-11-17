# Audit of the sampled ridge MinPI experiment

## What a negative witness establishes

The binary quota vote is defined on a ridge support refitted on fitting rows and scored on distinct voting rows. A single change from a winning base to a losing extension attains the lower bound of the ternary minimum. This is an exact feature-level statement about the specified finite game. It does not establish that removing this feature improves an outer test score or that every support containing it is worse.

We independently refitted every support using scikit-learn's SVD solver for seven of the eight confirmation sources (5–9 columns), on each of the five splits. The 64-column leaves task exceeded this audit's declared width cap. The complete oracle records include 245 feature/split pairs, of which the exact minimum was negative for 47. The four-round sampler detected 14 of these (29.8%). Independent refits reproduced each of the 14 negative witnesses. The 33 missed negatives all received a sampled zero. The remaining 193 exact zeros and five exact positives received the corresponding sampled score. The split with seed 42 gave three detections among 12 exact negatives. Repeated features and seeds share data.

| Exact minimum | Sampled −1 | Sampled 0 | Sampled +1 |
|---|---:|---:|---:|
| −1 | 14 | 33 | 0 |
| 0 | 0 | 193 | 0 |
| +1 | 0 | 0 | 5 |

For each target, conditional on which block it occupies, the next blocks form prefixes of a uniform permutation of the remaining columns. A subset dynamic program propagates the probability mass of prefixes that have encountered no negative at the examined sizes. Averaging over the target block with probability proportional to block size gives the one-round detection probability. Enumeration of all 120 permutations of a five-column fixture reproduces the calculation. All 47 negative minima in this audit were reachable. Their one-round probabilities ranged from 0.00397 to 0.83492 (median 0.08307). Summing their four-round detection probabilities gives 17.01 expected discoveries over fresh shuffle seeds on these fixed games.

Among the 6,780 scheduled addition checks in the eight-source primary run, the algorithm evaluated 5,436, skipped 1,278 after encountering a losing base for an already nonpositive feature, and skipped 66 after finding a negative witness. Leaves contributed 3,840 scheduled checks with zero skips. Across 565 feature/split pairs the counts were 14 negative, 546 neutral and five positive. Most opportunities saved here came from the losing-base bound. A claim that early negative absorption explains a large empirical speedup would conflict with the recorded counts.

The [oracle implementation](../../experiments/minpi_compression/oracle_audit.py) and [feature-level records](../evidence/minpi_compression/audit/exact-minima.csv) permit reproduction without accessing outer labels.

Adding sampled MinPI to the coefficient tie order changed 8 of 40 rankings and 6 of 40 selected feature sets (three Glass, two Energy, one Slump). In the other five sources every seed retained the same columns as the coefficient control. Five of the 14 negative-certified feature/split pairs survived width selection (three Glass, two Energy). A downstream policy that always deletes a witnessed feature would implement a different selection rule.

## Design and provenance

The confirmation protocol specifies eight fixed numeric UCI sources, five group splits per source, six methods and three timing repetitions. Source archives carry SHA-256 hashes. Glass has 213 unique predictor rows among 214 observations. Yacht uses 22 hull geometries, Airfoil uses 33 angle/chord settings, and Energy uses 12 building geometries. Holding out entire geometries changes the regression prediction problem relative to splitting individual rows. WiFi evaluates additional observations in the same four rooms. Leaves uses only the 64 margin descriptors, with 100 known species represented in every partition. Slump predicts compressive strength from the seven mix variables. Glass and Slump exclude their record identifiers. Energy excludes the companion cooling-load response. Group-stratified classification requires the labels for splitting, but the ranking and width-selection code has no access to held-out class scores.

For each method, standardization and imputation are fitted only on its training portion. MinPI, the continuous-mean control and coefficient ranker use the same ridge fit and tie scores. The F filter and single-step RFE have access to the pooled fitting and voting data. Coefficient and F ranking are genuine low-cost alternatives to the new selector. Validation fits, full-model guard, outer refit and test set are otherwise shared. The grid of at most six widths (including full) compares these rankers at a common computational budget. It does not establish that the chosen support has minimum cardinality at the prescribed loss. Three repeats use the same splits and supports. Timing excludes data loading and splitting. Group repeats are not independent confirmation tasks.

The eight source populations were chosen purposively. Seven have only 5–9 columns. Four sources per outcome domain limit evidence for transfer. A within-source supervised width guard provides no noninferiority guarantee for test groups sampled from another part of feature space. There is no multiclass calibration, interaction oracle, nonlinear learner or high-dimensional width-scaling test in the predictive confirmation panel.

## Independent reproduction and primary decisions

We verified all eight archive checksums, 40 receipts, prediction payload digests, row partitions and separation of source groups. The [independent audit](../../experiments/minpi_compression/result_audit.py) replays the width rule from the stored curves, refits all 240 selected supports by SVD with train-only centering and scaling, and checks the resulting predictions. It computes AUC by the rank-sum identity with midranks, accuracy by argmax, and the regression metrics directly from residuals. Every support, score and median time agreed with the released records. The clock verification covers the aggregation of recorded times. It is not an independent timing run.

Cells below give paired MinPI-minus-baseline scores, paired geometric width ratios and paired geometric selection-time ratios. The underlying units are four sources per domain, each represented by five held-out group splits. The [paired intervals](../evidence/minpi_compression/audit/paired-effects.csv) enumerate the 256 ordered four-source bootstrap redraws with the five-seed source means fixed. They describe sensitivity to the empirical mix of sources. They have limited resolution for a small purposive panel. Source-level contrasts and [feature decisions](../evidence/minpi_compression/audit/vote-decisions.csv) expose the variation hidden by domain means.

| Task | Comparator | Width ratio | Selection time ratio | ΔAUC or ΔR² | Δaccuracy or Δscaled MAE |
|---|---|---:|---:|---:|---:|
| Classification | Full | 0.6894 | — | −0.00482 | −0.01292 |
| Classification | Coefficient | 1.0126 | 1.1493 | −0.00274 | +0.00814 |
| Classification | Univariate F | 1.2230 | 1.1361 | +0.00271 | +0.00371 |
| Classification | RFE | 0.9982 | 0.9988 | −0.00130 | +0.00598 |
| Classification | Continuous mean | 1.0606 | 0.9953 | +0.00392 | −0.00923 |
| Regression | Full | 0.5039 | — | +0.00660 | +0.00331 |
| Regression | Coefficient | 1.0000 | 1.5094 | −0.00447 | +0.00117 |
| Regression | Univariate F | 1.0930 | 1.4936 | +0.03458 | −0.01232 |
| Regression | RFE | 1.0718 | 1.0349 | +0.04054 | −0.01544 |
| Regression | Continuous mean | 1.0620 | 0.9326 | +0.00785 | +0.00721 |

The primary compression criterion passed in regression. It failed in classification. Relative to full, classification accuracy fell by 1.29 percentage points against an allowance of one point, with a source-bootstrap interval of [−2.21, −0.38] percentage points. Glass (−1.86 points) and Raisin (−2.56 points) incurred the largest losses. In regression, the source-bootstrap interval for the retained fraction was [0.251, 0.905], extending above the 0.85 threshold. Yacht retained one of six columns on all five splits and gave the same supports under coefficient ranking.

The registered secondary width comparison with RFE passes its classification point rule narrowly, by 0.18% in geometric width. Its interval was [0.870, 1.112] for the ratio. The 10% selection-speed target failed in both domains. In regression MinPI outpredicted RFE on average but used 7.18% more columns geometrically. A claim of a stable compression advantage over RFE or the coefficient ranking lacks empirical support. The matched-refit ranking microbenchmark addresses a distinct question about computing an identical sequence of votes.

## Zero-tolerance sensitivity

The sensitivity was added after pilot evaluation and before opening confirmation scores. Replaying the same stricter width guard for all six methods reproduced the 240 released sensitivity scores. Against full, the MinPI width ratios were 0.7708 for classification and 0.8123 for regression. The mean predictive changes met the original numerical loss allowances in both domains. In classification the paired source interval for the accuracy change was [−0.01395, 0], spanning the allowable loss. The stricter guard retained all columns on every split of Airfoil, Leaves, Slump and WiFi. In Yacht the mean width rose from 1.0 to 5.2 of six columns.

Relative to the coefficient ranker, the strict rule gave a regression width ratio of 0.9521, ΔR² of +0.02786 and Δscaled MAE of −0.01163. Both rankers chose identical supports in every regression split except Energy seeds 43 and 45. For seed 43, MinPI retained four of eight and improved R² from −0.988 to −0.432, still below the constant test-mean reference used to define R². At seed 45 its improvement was 0.00083. Across both domains the strict support changed in five of 40 comparisons (three Glass, two Energy). Selection-time measurements for the strict replay were not collected. The [sensitivity contrasts](../evidence/minpi_compression/audit/strict-paired-effects.csv) must remain separate from the primary decision.

The experiments support a computational claim about evaluating this ridge quota game and a regression compression result relative to full. They do not establish a broad performance benefit from adding voting to coefficient ranking. A predictive claim for wide tabular settings needs a width and accuracy evaluation on wider sources and a strong low-cost ranker under the same learner, support grid and guard. Where time, AUC or accuracy comparison is stated for the entire pipeline, include the ranking, guard and final refit in that cost.
