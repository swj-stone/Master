# Proper utility votes for coalition-mean predictions

On the six original prepared sources at seed 42, this study tests a signed vote before and after adding a feature to the expected output of a uniform-size background. A positive vote means that adding that feature to the mean predictor lowers the squared calibration margin error. The loss acts on the mean output. Obtaining the mean of the individual masks' squared losses would also require their output variances. Brute enumeration on a small overlapping path game checks the reported formula at every feature and requested size. The full-background vote equals the squared norm of actual full-model feature ablation.

| Ordering under the identical 20% fidelity sweep | Retained width / full ablation | Outer AUC change | Outer accuracy change |
|---|---:|---:|---:|
| Minimum of empty, middle, full backgrounds | 1.259 | −.0015 | −.0022 |
| Mean across these backgrounds | 1.001 | −.0005 | −.0037 |
| Lexicographic minimum of the three | 1.120 | −.0019 | −.0011 |
| Negative witness first, then full ablation within each class | 1.096 | +.0016 | +.0014 |
| Minimum across three equal ranges of background sizes | 1.092 | −.0005 | −.0017 |
| Negative witness across ranges, then full ablation | 1.057 | +.0006 | +.0006 |
| Mean across the three ranges | 1.001 | −.0001 | −.0000 |
| Direct exact recomputation after every deletion | .979 | +.0008 | +.0010 |

Retained-width ratios are geometric means over all six sources; AUC/accuracy changes average the five classification sources. At empty background, a feature never used at a tree root has exactly zero effect. The signed minimum therefore introduces ties; replacing the index tie-break by the other two votes helps but leaves a substantial width disadvantage. Per-source signed negatives range from zero on ISOLET to 21 on Madelon. The archived timing for the six alternative rank rules jointly charges construction of both point and band contributions; it is an upper bound on any single variant's rank calculation.

The separate future-panel objective preserves mean tree predictions at 25%, 50% and 100% sensor availability. For direct greedy at the ordinary full-teacher endpoint, 25% distortions ranged from .021 to .115 and 50% distortions from .056 to .139. The full-panel distortions reached .183–.200. Imposing the maximum of the three yielded the same greedy endpoint widths on all six sources. Scores for the mean-predictor and randomized row-level outages are present for every arm and are distinct evaluation settings. Evidence does not establish a robust-min advantage against full ablation in this study.

## Calibration-label scoring on all five seeds

The secondary diagnostic replaces margin-reconstruction utility by native log loss (classification) or mean squared error (regression) against the calibration outcomes. Other rows supply the full teacher fit and held-out outcomes. It measures the proper loss of the mean raw predictor after transforming to event probabilities. A terminal control measures the proper loss change of the full teacher upon ablating one column. The direct unsupervised terminal energy, pooled signed score, and negative-witness grouping serve as additional controls.

Under the common 20% distortion guard, literal minimum proper-score ordering retained 13.62% more columns than full-teacher ablation and 5.16% more than mean proper-score ordering. Its average classification AUC and accuracy changes against ablation were +.0001 and −.0009. At matched half-width, its AUC change against pooled scores was −.0038. At quarter-width this was −.0060, with accuracy +.0024. The quarter-width AUC change against pooled scores ranged from +.0055 on ISOLET to −.0271 on fraud, and Abalone lost .0728 R². These data do not support preference for the signed minimum across sources. All 30 paired splits and six rankings are retained in `proper-all.csv`, with matched comparisons in `proper-contrasts.csv`.
