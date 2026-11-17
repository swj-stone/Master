# Compressed deployment of boosted trees on the six historical prepared sources

All 30 splits use their original development partitions and original 100-round depth-four XGBoost policy. Methods fit the same teacher once on the pooled fitting/voting rows, select features on the calibration rows, and stop at an unavailable split at deployment. Native training Hessian cover determines the margin at that stopping node. A row passed for prediction contains only the retained columns. The full traversal reproduces native margins on every calibration split. Subset predictions computed by grouping path increments agree with sparse traversal. A quadratic fidelity radius of 0.20 is relative to the Euclidean norm of the calibration margins above the empty-forest baseline. It does not bound the loss of accuracy on new rows.

In the table each cell gives mean retained width and two predictive metrics over five seeds. Classification uses AUC/accuracy. Abalone uses R²/MAE. Gain refit is the previously archived full development pipeline with a supervised width guard and a newly trained XGBoost on its selected columns. It solves a different deployment problem and has access to the full development set for training its final trees.

| Source | Exact greedy | Gain sweep | Full teacher | Gain refit |
|---|---|---|---|---|
| Madelon | 47.4 · .7904/.7146 | 65.8 · .7872/.7069 | 100 · .7783/.7046 | 20.4 · .8302/.7585 |
| Spambase | 19.2 · .9826/.9418 | 20.0 · .9827/.9418 | 55.8 · .9856/.9501 | 25.2 · .9840/.9459 |
| Abalone | 5.6 · .5260/1.5538 | 5.6 · .5283/1.5512 | 8 · .5361/1.5242 | 2.4 · .5225/1.5638 |
| Gisette | 47.6 · .9854/.9503 | 55.6 · .9857/.9527 | 100 · .9889/.9566 | 35.0 · .9849/.9521 |
| Credit-card fraud | 21.6 · .9733/.9994 | 25.0 · .9751/.9995 | 29 · .9772/.9995 | 10.8 · .9711/.9994 |
| ISOLET | 67.2 · .9915/.8071 | 76.4 · .9916/.8131 | 100 · .9930/.8314 | 75.0 · .9928/.8269 |

Over the 30 paired splits exact greedy uses 12.39% fewer features than Gain sweep and 19.09% fewer than Gain prefix in source-equal geometric means. The largest source-average accuracy loss against Gain sweep is 0.0060 on ISOLET. Against the full teacher ISOLET loses 0.0244 accuracy. Madelon's guarded, refitted Gain model exceeds the frozen greedy predictor in AUC by 0.0397. The loss and computational scope of each comparison must accompany any width result.

## Votes and exact choices

The exploratory adaptive directional heuristic checks one proposed deletion at a time. Its geometric mean width is 3.10% above exact greedy and 9.67% below Gain sweep. Adding rejected residuals as new voters changes four of 30 endpoints relative to starting with a single direction. Exact screening instead bounds the omitted singleton's squared norm using projections. Both the cached squared-norm control and the projection procedure reproduce every dense exact greedy deletion, including order, on all 30 splits. The efficient controls do not certify minimum cardinality.

The zero/one/four-dimensional projection ablation is available in `exact-one-axis.csv`. On ISOLET it recomputed 767.8, 38.2 and 37.8 singleton norms per split respectively, after the initial 100 singleton norms. Mean search times were 0.142, 0.109 and 0.121 seconds. On fraud the corresponding counts were 93.4, 6.4 and 6.4, and search times 0.083, 0.062 and 0.070 seconds. A single direction accounts for nearly all observed savings. End-to-end pipeline times also charge model fitting, tree extraction, necessary path caching and restricted prediction. The four-dimensional check is an extremum over the unit sphere of the four-bin span. Four scalar binary tests would implement a different bound.

## Evidence and replay

- `historical-42-123-456-789-1024.csv` and its control counterpart record split-level widths, times, metrics and candidate counts. The primary rows use the whole held-out partition and novel rows remove prepared-input repeats according to the archived manifest. `source-means.csv` aggregates each source separately.
- The corresponding JSON locks and receipt archives record selection, forecasts and supporting hashes. Forecasts are sealed under `results/minpi_rework/quota-tree`. Replays start with `experiments.minpi_rework.quota_compress_tree_run`; use its separate select and score actions.
- `exact-lazy-42-123-456-789-1024.csv` contains three interleaved repetitions of dense, cache-only, projection and two lazy alternatives. The latter index rows to propagate residual changes and compute anchor increments on demand. Their extra bookkeeping did not consistently reduce runtime.
- The separate 13-source preregistration and forecast lock are in `../quota-tree-confirmation`. The near-constant synthetic counterexample concerns the archived squared-slack Gain prefix. Use `experiments.minpi_rework.quota_compress_tree_v2.extract` for new experiments. The corrected entry point applies the same norm-scale tolerance as the exact search.
