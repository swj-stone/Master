# Pair deletion and single-column outages

All data in these experiments were used in earlier rounds of method development. The original six retain their prepared partitions and novel-input outer masks. The other 18 sources use the existing group-based partitions. The fixed-width outage comparison was chosen after inspecting earlier pilot results.

### Block deletion with the original minimum

Let $m(B)$ be the minimum benefit of adding a block $B$ over all backgrounds disjoint from that block. In any background, order the features of $B$ arbitrarily and telescope the block addition along this order. Every term in the sum is at least the minimum marginal of its feature. Thus $m(B)\geq\sum_{i\in B}m(i)$, and $m(B)/|B|\geq\min_{i\in B}m(i)$. When all singletons compete, the lowest average block minimum is always attained by a singleton. This applies even to interacting set functions.

With squared loss and centered additive shape functions, the minimum for a block follows by summing its shape functions and using their Gram matrix. We compared singletons with all pairs on a trimmed spline model. Mixed minimum deletion reproduced the single-column scores and choices on the original splits tested. Forcing two deletions per step broke the degeneracy but gave weak predictive performance. On Madelon, seed 42, forced pairs retained 98 of 100 columns and attained AUC .645. Pair-energy deletion retained 7 with AUC .691. Overlapping searches also tested the minimum benefit of deleting a block and a local adversary that can discard one additional feature. A local pair rule reached AUC .724 with 11 columns. The archived XGBoost gain model on the same novel inputs reached .870.

### Frozen additive model with a single unavailable input

The model fits four spline basis functions per input on the fitting rows with ridge penalty one. At deployment, an unavailable retained input contributes its training-mean fitted shape. The adversary can suppress one retained input or leave them all available. For one fitted support, the risk is the largest squared loss across these scenarios on the evaluation rows. This adversary picks a sensor for the whole set of observations. Clean predictions use all retained inputs. Normalized risk divides by the squared loss of the training-center constant on the same evaluation rows. Binary performance-quota ballots over different outages all pass at exactly those quotas accepted by the worst utility.

The fixed-width comparison starts from the largest-energy half of the feature shapes, rounded upward. Each arm tests all one-for-one swaps and takes at most two improving swaps. Their objectives use worst outage, mean of all outages including no failure, or no outage alone. The energy comparator keeps the starting support. The model, shape functions and imputation values stay frozen throughout voting and outer evaluation.

| Classification panel | Worst normalized loss vs mean | Source bootstrap 95% interval | Clean AUC difference | Clean accuracy difference |
|---|---:|---:|---:|---:|
| Nine previous UCI sources, five seeds | −.0450 | [−.0998, −.0014] | +.00004 | −.0063 |
| Five original sources, five seeds | −.0188 | [−.0504, −.0018] | −.0017 | −.0078 |

The intervals resample whole sources after averaging seeds and describe these exploratory panels. Against the no-outage swap objective, worst loss fell by .0470 and .0197, respectively. Against full-width models, clean accuracy at half width fell by .0524 and .0283. Regression did not give a stable clean-cost tradeoff. On Abalone, the method improved worst loss over mean by 1.873 normalized units, but clean R² fell by .206. The experiments did not measure failure rates or justify treating each public dataset as a sensor system.

A shared adaptive-width version separately repaired each energy prefix at each proposed width, then checked the clean AUC and accuracy floors or the clean R² and MAE floors, together with a worst-outage width-split loss no greater than full width. On the five original classification sources, it retained .363 of the full width, with clean AUC +.0030 and accuracy +.0034. Against mean swaps under the same guard, its width ratio was .952, AUC −.0024 and accuracy −.0018. On nine previous UCI classification sources its width ratio versus mean rose to 1.011 and its worst-loss difference was +.0106. Relative to full width its AUC fell .0122 and accuracy .0108 on this panel. The smaller model and clean-performance result on the original panel did not transfer to both panels.

The fixed-width classification outcome supports a narrow tradeoff between clean accuracy and worst single-column outage loss for frozen additive models. Routine intact-feature selection and the worst-case classification endpoint estimate different quantities. Other trials in the evidence include sequential block deletion, refitting the shape functions between stages, and using full-model outputs for outage calibration. Timings in the archived pilot characterize development runs with shared preparation and competing machine load.

### Reproduction

Use the thesis launcher from the isolated checkout. Score files are in `results/minpi_rework/<run>/scores.csv`. The compressed evidence contains rows from every trial and selected columns from replaying the main exchange experiment. The checks compare analytical margins, risk updates and swap choices against exhaustive small-model evaluation.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.pair_checks
.\thesis-python.ps1 -m experiments.minpi_rework.pair_outage --freeze --run my-outage-replay --names glass raisin wifi leaves airfoil yacht slump energy landsat biodeg aquatic power magic musk1 wall estate hardware fires --seeds 42 43 44 45 46 --methods exchange_min exchange_mean exchange_point energy full
.\thesis-python.ps1 -m experiments.minpi_rework.pair_summary --runs my-outage-replay --arm exchange_min --controls exchange_mean exchange_point energy full
```

For the original panel use `--names madelon spambase abalone gisette isolet credit_card_fraud --seeds 42 123 456 789 1024`. The adaptive-width comparison changes the `exchange_` prefix to `adaptive_`. Exact block minima and their alternative orientations use `pair_pilot`.
