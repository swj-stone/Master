# Backward quota paths on previously studied datasets

These experiments informed method development. The 18 UCI sources appeared in earlier studies in this worktree. The six original prepared datasets keep their original train, validation and test rows. Their scores below restrict test evaluation to previously identified novel inputs. In every trial the ridge receiver, outer split, width options and inner performance gate are shared across rankers. The Gain row trains the archived GPU XGBoost ranker and submits its feature ordering to the same ridge receiver.

For a feature in a sampled context let $a$ be the utility after fitting with that feature and $b$ its refitted utility without it. For a quota $t$, the addition ballot is $1[a\geq t]-1[b\geq t]$. Integrating the minimum ballot over $t\in[0,1]$ gives the common length of the positive intervals minus the union length of the negative intervals. A single $b>a$ therefore determines the negative sign. A positive sign requires $\min a>\max b$. Otherwise the sign is zero. This statement concerns the sampled contexts. Any strictly increasing utility transformation that maps into the quota domain preserves these signs.

The algorithm partitions current features at three sizes, with each feature covered by one quarter, one half and the full support when dimensions permit. Preceding a scoring stage, row and response centering use fitted means, and every ridge fit includes a free intercept. The first pilots scored actual held-out squared loss. Subsequent pilots scored pooled generalized cross-validation (GCV), using the exact ridge smoother trace. Ridge deletion identities give all required endpoints in a context from a common factorization. The sign rule uses the current full-support utility difference to discriminate within each sign, with conditional training cost resolving further ties. Its eager comparator evaluates the identical target in all scheduled contexts.

Round one used the older fixed width grid on 14 previously used sources, then compared utility definitions on 24 sources. Round two applied the same all-integer width guard to every method. One Cholesky factorization provides the nested predictions through incremental prefix contributions. Each width must pass both classification measures (AUC and accuracy within .01 of the full inner model) or both regression measures (R² within .02 and scaled MAE increase at most .02). Only inner width labels enter this decision. The optional zero-tolerance setting requires both measures to match or improve on the full inner model. The final model refits on all development rows. The complete histories retain every pilot in the accompanying compressed CSV.

### Five-seed comparisons

Ratios use paired geometric means, giving each source equal weight after aggregating seeds. Prediction differences are absolute. The original panel contains five classification sources and one regression source. The separate UCI panel contains nine classification and nine regression sources. Fixed CPU affinity and a single BLAS thread were used for five-seed timings. The UCI quarter-step comparison rotates execution order over three timing repetitions. Timings in the one-stage sensitivity and original-panel experiments use one observation per split.

| Quarter-step sign GCV versus comparator | Width ratio | Selection time ratio | Δ AUC | Δ accuracy | Δ R² | Δ scaled MAE |
|---|---:|---:|---:|---:|---:|---:|
| UCI classification, full-support GCV marginal | 1.018 | 1.201 | −.0032 | −.0023 | | |
| UCI classification, step-one ridge RFE | .925 | .726 | +.0013 | +.0049 | | |
| UCI regression, full-support GCV marginal | 1.011 | 1.371 | | | −.0039 | +.0012 |
| UCI regression, step-one ridge RFE | .949 | .878 | | | +.0079 | −.0009 |
| Original classification, full-support GCV marginal | 1.053 | 1.075 | −.0004 | −.0006 | | |
| Original classification, Gain ordering | .735 | .182 | −.0029 | −.0023 | | |
| Original regression, full-support GCV marginal | .944 | 1.299 | | | +.0044 | +.0005 |

On the UCI panel, quarter-step sign voting skipped 3,414 of 21,390 scheduled deletion edges; on the original panel it skipped 3,509 of 24,222. On all 90 UCI splits the eager and lazy algorithms gave identical orders, support sizes and predictions. Their selection-time ratios were 1.003 on classification and .966 on regression. The Cholesky factor and, for GCV, the smoother trace still require work on the context. Marginal savings did not translate into a broad wall-time advantage.

With one scoring stage and the zero-tolerance width guard, UCI classification retained .931 times the width of its full-support GCV comparator; AUC changed by −.0025 and accuracy by +.0023. For regression the ratio was 1.000, with R² −.0006 and scaled MAE +.0003. Compared with full-width ridge, this stricter selector retained .587 of classification features but lost .0126 AUC and .0142 accuracy. These observed losses exceed the original .01 allowances. Regression retained .677 with R² +.0003 and scaled MAE +.0011.

The original Madelon split also exposes a receiver limitation. With seed 42, a one-feature ridge model had AUC near .615 on novel inputs. The archived Gain ordering with an XGBoost width guard and XGBoost receiver attained .8696 using 13 columns. A faster linear receiver on this task would give up substantial predictive performance.

On the five original classification datasets, the same one-stage zero-tolerance procedure retained .514 times the full width, with AUC +.0001 and accuracy +.0002. It used .898 times the width of its single-context GCV control, with a selection-time ratio of 1.006, AUC −.0002 and accuracy +.0008. Against pooled coefficient ranking the width ratio reversed to 1.268. On the original regression task it retained the same number as single-context GCV and .944 of the full width.

These outcomes do not support selecting the quarter-step sign method for broad deployment. The mathematical absorption rule is exact for a finite family of real-valued endpoint scores. The implementation compares floating-point endpoints strictly. The practical benefit depends on the receiver, distribution of signs, acceptance tolerance, and the availability of a strong single-context control.

### Replay

`path_pilot.py` uses the previous local data manifests. `path_checks.py` compares the deletion identity with explicit ridge refits, the GCV update with the explicit hat matrix, prefix predictions with separate sklearn fits, and numerical quota integration with a direct sum over all endpoint intervals. To repeat the five-seed UCI sign comparison from this worktree, use a new run identifier.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.path_checks
.\thesis-python.ps1 -m experiments.minpi_rework.path_pilot --run path-uci-replay --cpu 4 --dense --names glass raisin wifi leaves airfoil yacht slump energy landsat biodeg aquatic power magic musk1 wall estate hardware fires --seeds 42 43 44 45 46 --methods signed_gcv signed_eager_gcv full_marginal_gcv pooled_coefficient rfe full --repeats 3
.\thesis-python.ps1 -m experiments.minpi_rework.path_effects --runs path-uci-replay --arms signed_gcv --controls full pooled_coefficient full_marginal_gcv signed_eager_gcv rfe
```

The original five-seed run used `--names madelon spambase abalone gisette isolet credit_card_fraud --seeds 42 123 456 789 1024 --gain`, with the same arms except eager GCV and a single timing repetition. One-stage sensitivity adds `@once` to the sign and full-support method names. Adding `--zero` activates the zero-loss guard.
