# Nonlinear voting on the original six development datasets

These tests used the existing prepared splits and archived CUDA XGBoost receiver. The ranking and proxy models saw fit and vote rows. The strong-model width decisions used the disjoint width split and the archived full-model metric floors. This study did not evaluate outer predictions. All six sources had informed earlier method development.

## Ranking

One experiment formed fixed anchor, hit and miss triplets. A coalition received a vote when its retained coordinates made the anchor closer to the hit. Sampling eight permutations at three background sizes gave minimum and pooled net swings. A continuous neighbor-distance margin used the same triplets. Neighbor differences are related to [ReliefF](https://doi.org/10.1023/A:1025667309714). The triplet construction uses nearest different-class examples and a reference empirical-rank metric.

A second experiment fit 32 depth-seven random forest trees. Each tree stopped at the first unavailable split on a validation row. Its node gave one vote for a correct true-class versus alternative-class ordering, with half credit for a tie. Regression used training target quartiles for the proxy and the actual continuous target for the downstream model. We tested static ranking, conditional forward ranking, and demotion of negative features in the archived Gain ranking. The selected widths were as follows.

| Dataset, seed 42 | Gain | Triplet minimum | Tree minimum | Conditional tree minimum | Gain after negative demotion |
|---|---:|---:|---:|---:|---:|
| Abalone | 4 | 6 | 6 | 6 | 4 |
| Credit card fraud | 8 | — | 22 | 8 | 22 |
| Gisette | 25 | 50 | 75 | 50 | 25 |
| ISOLET | 50 | — | 100 | 100 | 100 |
| Madelon | 13 | 13 | 13 | 13 | 13 |
| Spambase | 14 | 28 | 28 | 28 | 14 |

The triplet experiment ended after four sources. On Madelon the old linear minimum needed 50 columns on this split. Both nonlinear games reached 13, a width that the corresponding pooled controls and Gain also reached. Tree-based demotion flagged 68 of 100 Madelon features and left the selected width at 13. The archived and new full-width inner scores agreed for every comparison. The [per-method measurements](rank-pilot.csv) contain native validation metrics. Ranking clocks describe each family computing all its experimental scores in one call. They cannot establish stand-alone speedups.

## Verifying candidate widths on the Gain path

The tree-game contrast can be evaluated from its paths. For a fixed row and tree, the vote at the reached node equals its root vote plus the sum of the vote increments along all reached edges. An edge with distinct prefix feature set P is reached when P is contained in the coalition. For a uniform size-r background excluding feature i in P, the chance that adding i activates this term is the ratio of the falling factorials (r)_(|P|−1) and (p−1)_(|P|−1). Repeated uses of the same feature count once in P. The direct traversal check covers every feature at every size on a four-feature multiclass tree.

For width pruning, a restricted tree loses the edge increments whose prefix is outside the retained set. The alternative computational proxy refit a depth-six CART model on each background and its intersection with the Gain prefix. The backgrounds occupied 25%, 50% and 100% of the features. For both proxies, the minimum rule used the largest measured loss from deleting the tail across the three tested sizes; the fixed-tree calculation used 12.5%, 50% and 87.5%, with 100% as its single-context control. Other sources supplied the maximum score of an accepted non-full width. Exceeding that value skipped the candidate on the held-out source. This threshold uses known width results on other historical datasets and carries no coverage guarantee.

| Pruning policy | Strong fits saved out of 129 | Splits with a wider result out of 30 |
|---|---:|---:|
| Deletion fraction | 36 | 4 |
| Discarded Gain mass | 20 | 5 |
| Forest impurity mass | 36 | 1 |
| Fixed-tree minimum vote | 22 | 1 |
| Fixed-tree pooled vote | 22 | 1 |
| Fixed-tree continuous full-context margin | 26 | 1 |
| CART refit minimum | 27 | 3 |
| CART refit pooled | 38 | 2 |
| CART refit on full context alone | 27 | 0 |

The reference fit the full model, then scanned the smaller widths in ascending order until one passed. The counts include the full model and exclude the common ranking and deployment fits. CART full-context screening used 2.70 s of measured weak fits and utility scoring across 30 splits, versus 5.77 s for its three-context minimum. These component clocks exclude data preparation. End-to-end timing would require prospective re-execution before making a speed claim. The raw tables report the [candidate measurements](guard-refit-inputs.csv), the [tree screen](guard-screen.csv) and the [refit screen](guard-refit-screen.csv).

A final comparison allowed at most two smaller widths plus the full fit. Historical source folds supplied a fixed pair minimizing log width. An adaptive controller chose the pair among the five calibration folds with the nearest vector of cheap scores. Fixed-tree minimum and pooled voting both used 73 strong checks and returned a wider model than the ascending reference on six splits, with a geometric width ratio of 1.112. The simple fixed pair used the same 73 checks, returned a wider model on seven splits, and had a width ratio of 1.123. The weak-CART minimum used 75 checks and returned a wider model on nine splits, with a ratio of 1.176. The evidence does not justify selecting any of these multi-context minimum variants for the independent panel. [Two-proposal results](guard-two-proposals.csv)

## Reproduction

From the thesis checkout, use the following PowerShell commands. The pilot stores results in the ignored worktree results directory.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_formula_check
.\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_pilot --game triplet --panels 8 --datasets madelon spambase gisette abalone
foreach ($game in 'tree','conditional','screen') {
    .\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_pilot --game $game --panels 32 --datasets madelon spambase gisette abalone credit_card_fraud isolet
}
.\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_guard
.\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_refit_guard
.\thesis-python.ps1 -m experiments.minpi_rework.nonlinear_export
```
