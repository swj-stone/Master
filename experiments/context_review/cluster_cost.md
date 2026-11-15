# Uncertainty and error costs for the Table 2 deletion vetoes

On 400 search-path proposals, expanding the two endpoint tests to the separating minimum detected 11 additional refit failures and rejected 20 more passing refits. That trade breaks even at a missed-failure cost of 20/11 false alarms. The paired task/fold bootstrap placed the lower end of the crossover range at 0.46 and left its upper end unbounded. On 379 separately drawn proposals the extra tests detected zero additional failures and rejected nine more passing refits. These counts concern vetoes on individual proposals. They do not measure the quality of the final selected feature set.

## Cohorts, outcome and resampling

The source contains 14 previously examined PMLB tasks with three outer folds each. For each fold and candidate support, the original experiment refitted the reduced model and its anchor on the outer development sample and evaluated both on the same outer test rows. A refit failed if relative AUC or accuracy loss exceeded 3%, or relative average precision loss exceeded 5% when the original rare-class condition applied. A veto of a failing refit is a true positive. A veto of a passing refit is a false positive. The tolerances and ballots are inherited from the original experiment. [Original design](../../configs/context_review/refit_target.json) · [Archived decisions](../../thesis-paper/evidence/reassessment/decisions.csv)

Search-path proposals have 189 failures and 211 passes. Drawn deletions have 163 failures and 216 passes. The two sources share 75 fold–support pairs and are reported separately. Within each source the Table 2 rate estimates pool counts before division, giving each proposal equal weight. TPR divides true positives by failures. FPR divides false positives by passes. A fold without failures contributes zero failure mass, and a fold without passes contributes zero passing mass. In each source three folds lacked failures and 11 lacked passes. All proposals passed on the drawn Satimage task. All proposals failed on Page Blocks and Phoneme in both sources.

The primary bootstrap drew 14 tasks with replacement, then drew three folds with replacement independently for each selected task occurrence. Every sampled fold carried all of its proposals, refit outcomes and rule decisions. The same draws were used for both cohorts and every rule. The sensitivity bootstrap used the same task draws and carried each sampled task's three folds intact. This preserves the observed dependence among its outer folds, including overlapping training rows. The 30,000 replicates used seed 15112026. Endpoints below are empirical 2.5th and 97.5th percentiles. All replicates contained failures and passes in each cohort. [Local design](../../configs/context_review/cluster_cost.json)

These are pointwise intervals for reweighting a convenience sample of tasks and folds. The fitted models, proposal paths, shared panels and outer test measurements are fixed inputs. The analysis does not establish coverage for a population of new tasks or for future refit losses.

## Conditional veto rates

Entries are percentages with paired task/fold bootstrap 95% intervals. Differencing the paired resamples gives the uncertainty on a rule comparison.

| Rule | Path TPR | Path FPR | Drawn TPR | Drawn FPR |
|---|---:|---:|---:|---:|
| Empty, point panel | 88.4 [78.6, 96.1] | 78.2 [63.7, 88.2] | 92.6 [85.0, 98.0] | 73.6 [65.9, 80.5] |
| Empty, four panels | 89.4 [77.5, 98.7] | 78.2 [66.4, 87.4] | 98.2 [95.7, 100] | 75.9 [68.4, 82.4] |
| Both endpoints | 91.5 [81.1, 99.4] | 82.5 [72.9, 90.0] | 98.8 [96.7, 100] | 78.2 [71.1, 84.6] |
| Nested minimum | 92.1 [81.6, 99.4] | 83.9 [74.7, 91.2] | 98.8 [96.7, 100] | 79.6 [72.5, 86.5] |
| Separating, point panel | 94.7 [85.3, 100] | 91.5 [84.0, 96.3] | 98.2 [95.8, 100] | 81.0 [74.5, 87.8] |
| Separating, pooled votes | 82.5 [67.2, 94.3] | 65.9 [50.6, 79.1] | 93.9 [85.4, 98.6] | 68.5 [58.5, 77.5] |
| Separating minimum | 97.4 [92.4, 100] | 91.9 [85.6, 96.2] | 98.8 [96.7, 100] | 82.4 [76.1, 88.7] |
| Size-matched random minimum | 96.3 [89.5, 100] | 92.9 [85.2, 97.6] | 98.8 [96.7, 100] | 84.7 [79.3, 90.1] |
| Veto when at least one quarter of columns are removed | 87.3 [75.7, 97.9] | 59.7 [39.4, 78.3] | 87.7 [77.8, 97.3] | 53.7 [46.0, 59.6] |

The following differences are separating minus endpoints in percentage points. The equal-task/fold target gives each task and each of its folds equal weight before selecting a proposal uniformly within the fold. It averages fold joint confusion proportions and then conditions on the outcome class. This gives pooled failure prevalences of 47.25% and 43.01% and equal-task/fold prevalences of 57.14% and 55.06% on path and drawn proposals respectively.

| Estimator and resampling | Path ΔTPR | Path ΔFPR | Drawn ΔTPR | Drawn ΔFPR |
|---|---:|---:|---:|---:|
| Pooled, tasks and folds | 5.82 [0, 13.56] | 9.48 [4.57, 15.14] | 0 [0, 0] | 4.17 [0.46, 9.14] |
| Pooled, whole tasks | 5.82 [0.72, 12.16] | 9.48 [5.56, 12.89] | 0 [0, 0] | 4.17 [1.00, 8.15] |
| Equal task/fold, tasks and folds | 3.85 [0, 10.03] | 9.26 [4.44, 14.73] | 0 [0, 0] | 3.97 [0.43, 8.66] |
| Equal task/fold, whole tasks | 3.85 [0.42, 9.16] | 9.26 [5.43, 12.56] | 0 [0, 0] | 3.97 [0.94, 7.63] |

The drawn ΔTPR interval collapses to zero because the rules agreed on **every observed failing proposal** in this cohort. Resampling existing folds cannot reveal an unobserved discordance. In the path cohort, Sonar supplied seven of the 11 additional detections. Omitting Sonar raises the path pairwise cost threshold from 20/11 to 19/4.

## Decision costs

Measure one false alarm at unit cost, and let *r* be the cost of missing one refit failure in the same units. The empirical mean loss is `(FP + r × FN)/N`. A false alarm rejects a deletion that passes the refit criterion. A miss allows a deletion that fails it. This marginal loss treats errors within each category alike. It does not price runtime, amount of feature reduction, loss severity or downstream consequences of continuing a search after a veto.

| Comparator | Separating has lower cost on the 400 path proposals when | Separating has lower cost on the 379 drawn proposals when |
|---|---|---|
| Both endpoints | r > 20/11 = 1.818 | No finite r. Equal misses and nine excess false alarms |
| No veto | r > 194/184 = 1.054 | r > 178/161 = 1.106 |
| Veto all | r < 17/5 = 3.400 | r < 38/2 = 19.000 |
| Veto if at least one quarter of anchor columns are removed | r > 68/19 = 3.579 | r > 62/18 = 3.444 |

At a displayed crossover the two policies tie. For the size-only rule the confusion counts (TP, FP, FN, TN) were (165, 126, 24, 85) on paths and (143, 116, 20, 100) on drawn deletions. The quarter threshold was fixed for this analysis from the original set of sampled deletion sizes. Its performance is a retrospective comparison on these tasks.

For path proposals the paired cost difference per 100 proposals is `(20 − 11r)/4`. For drawn proposals it is `900/379` units per 100 for every finite r. Positive values favor endpoints. These are paired losses per 100 proposals, distinct from conditional error-rate differences in percentage points.

| Miss cost r | Path cost difference per 100 [95% interval] | Drawn cost difference per 100 [95% interval] |
|---:|---:|---:|
| 1 | +2.25 [−3.40, +7.17] | +2.37 [+0.26, +4.95] |
| 2 | −0.50 [−10.11, +6.65] | +2.37 [+0.26, +4.95] |
| 5 | −8.75 [−30.91, +5.76] | +2.37 [+0.26, +4.95] |

For policies whose observed veto sets are nested, each resample gives a nonnegative number of extra detections and extra false alarms for the stricter policy. A draw with only extra false alarms yields an infinite crossover. A draw with both increments zero ties at every ratio and spans the entire range from zero to infinity. The 2.5th percentile of draw lower ends and the 97.5th percentile of draw upper ends give the reported crossover sensitivity range. For nonnested comparisons the output instead gives the direction of the observed inequality, counts of signed detection changes over draws, and paired cost intervals at eight fixed ratios. [Thresholds](../../results/context_review/cluster-cost/thresholds.csv) · [Paired costs](../../results/context_review/cluster-cost/paired-costs.csv)

For separating versus endpoints, 917 of 30,000 path resamples had extra false alarms and zero additional detections. Including them gives a range of [0.46, ∞]. Keeping whole tasks gives [0.56, 19.00]. Equal task/fold weighting gives an observed crossover of 1.80 and a task/fold range of [0.45, ∞]. On drawn proposals 29,325 task/fold resamples had only extra false alarms, while 675 tied for every ratio. No resample favored separating. Relative to no veto, the path and drawn task/fold crossover ranges are [0.47, 2.14] and [0.58, 1.96]. Relative to veto-all they are [0.96, ∞] and [4.29, ∞].

At deployment failure prevalence *p*, the expected cost is `(1 − p) × FPR + r × p × (1 − TPR)`. If the conditional rates transfer, the separating-versus-endpoints path crossover is `((1 − p)/p) × (20/211)/(11/189)`. At 10%, 25%, 50% and 75% failure prevalence the crossover is 14.66, 4.89, 1.63 and 0.54 respectively. These scenarios hold the conditional rates fixed and specify prevalence externally. [Prevalence sensitivity](../../results/context_review/cluster-cost/prevalence-sensitivity.csv)

## Lowest empirical cost across deterministic policies

Each row lists optimal policies in ascending *r*, with adjacent policies tying at the stated switch. The frontier uses exact rational comparison of every candidate policy against every other candidate. It identifies minima on these observations. Selecting a policy after examining this frontier requires new data to estimate its deployed performance.

| Policies | Cohort | Sequence and switches for pooled proposal counts |
|---|---|---|
| Eight Table 2 rules and both constants | Path | No veto → 0.891 → pooled votes → 1.964 → separating minimum → 3.400 → veto all |
| Eight Table 2 rules and both constants | Drawn | No veto → 0.967 → pooled votes → 2.286 → empty four panels → 5.000 → endpoints → 23.500 → veto all |
| Above plus fixed quarter-deletion rule | Path | No veto → 0.764 → quarter deletion → 3.542 → veto all |
| Above plus fixed quarter-deletion rule | Drawn | No veto → 0.811 → quarter deletion → 2.824 → empty four panels → 5.000 → endpoints → 23.500 → veto all |

With the size-only comparator included and each task/fold weighted equally, separating re-enters the path frontier for *r* from 3.28 to 3.93. It remains off the drawn frontier. Pairwise results against endpoints alone therefore do not identify the best policy among all the specified alternatives. [Exact frontiers](../../results/context_review/cluster-cost/frontier.csv)

## Reproduction

From the thesis checkout run `./thesis-python.ps1 -m experiments.context_review.cluster_cost`. The input SHA-256, seed, design, implementation hash and result hashes are recorded. The output includes the four confusion cells by fold and rule, conditional rates and joint masses, paired contrasts, deterministic frontiers and prevalence scenarios. [Implementation](cluster_cost.py) · [Fold counts](../../results/context_review/cluster-cost/fold-counts.csv) · [Rates](../../results/context_review/cluster-cost/rates.csv) · [Paired rate differences](../../results/context_review/cluster-cost/paired-rates.csv) · [Run manifest](../../results/context_review/cluster-cost/manifest.json)
