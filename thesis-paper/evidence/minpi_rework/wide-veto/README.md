# Wide-source replication of exact MinPI ranking

The study registration, four archival digests and both materiality criteria were committed as `5cad720` before training. All 20 dataset/seed jobs completed. All 360 timed trials (six implementations, three repeats per job) returned exactly the same ordered deletion path within a job. Forecasts, supports, split indices and timing receipts were sealed at `7130f31`. The independently verified forecast archive digest is [recorded here](archive-sha256.txt).

## Computational comparison

The principal ratio takes the fastest measured veto implementation and the fastest exhaustive implementation for that job, after taking each implementation's median over repetitions. The denominator includes native scalar and one-thread BLAS versions. Both families can coalesce columns that no longer occur in any active path group. Dormant columns have the same zero deletion effect. Their original IDs determine positions within ties, and every vote still ranges over all sizes up to the original p. Exhaustive evaluation without this optimization is much slower on the widest inputs.

| Source | Released width | Median rank seconds, veto / exhaustive | Paired rank ratio | Median fit-through-prediction seconds, veto / exhaustive | Paired complete ratio |
|---|---:|---:|---:|---:|---:|
| Internet Advertisements | 1,558 | .05016 / .28775 | .171 | 7.638 / 7.875 | .969 |
| QSAR oral toxicity | 1,024 | .31827 / .53283 | .584 | 5.932 / 6.175 | .964 |
| Tehran Residential Building | 79 | .00055 / .00076 | .754 | .17879 / .17901 | .999 |
| UJIIndoorLoc | 520 | .00874 / .00946 | .919 | 1.24370 / 1.24371 | 1.000 |

The geometric mean rank-time ratio was **.513** (source-bootstrap 95% interval .248–.832). This meets the preregistered rank-time target of at most .90 with the interval below one. The fit-through-prediction ratio was **.983** (.967–.999), which misses the .90 materiality target for complete runtime. Five paired log ratios were averaged per source. The four purposively chosen sources were then weighted equally; the interval describes sensitivity to those four sources. The complete time adds measured shared fitting, construction of the path game, evaluation of both guarded paths and restricted prediction to each separately timed ranking implementation. Input-file reads and JIT warmup are excluded.

The uncoalesced veto/straight-exhaustive rank ratio was .947 (.839–1.079). Coalescing both sides gave .447 (.215–.703) against the scalar loop, with the veto inspecting .327 (.169–.455) of its size comparisons. The primary envelope used coalesced BLAS for 12 jobs, coalesced scalar enumeration for six, and the uncoalesced scalar loop for two. It used coalesced veto for every job. The test establishes an advantage relative to the specified exhaustive computations. It places no lower bound on alternative exact polynomial-optimization algorithms.

## Native forest audit

A separate replay re-fitted each sklearn forest from its frozen training rows, changed its unavailable split nodes into terminal nodes, and called sklearn's native prediction routine. All 80 prediction arrays reproduced with maximum absolute discrepancy below 6.4×10⁻¹². Direct traversal of those pruned node arrays also reveals the distinction between nominal retained width and genuinely referenced input columns.

| Source | Distinct columns in full fitted forest | MDI nominal / reachable | MinPI nominal / reachable | Portfolio nominal / reachable |
|---|---:|---:|---:|---:|
| Internet Advertisements | 264.6 | 151.0 / 149.6 | 127.6 / 127.6 | 127.6 / 127.6 |
| QSAR oral toxicity | 537.8 | 269.0 / 256.0 | 286.6 / 214.0 | 218.6 / 213.4 |
| Tehran Residential Building | 71.2 | 30.4 / 30.4 | 42.6 / 40.4 | 29.0 / 29.0 |
| UJIIndoorLoc | 132.4 | 18.6 / 18.6 | 26.8 / 26.8 | 18.2 / 18.2 |

Unused inputs already reduce the full forest's actual acquisition requirements. Dropping a split feature can also make a retained descendant input unreachable. A physically unreachable feature can be omitted without changing any prediction, including when additional inputs become unavailable. The registered comparison used nominal widths; the native verification is a deployment diagnostic.

## Outer prediction after release

The table reports five-seed means for the full forest, the single-path MDI and minimum rules, and the portfolio that considers both paths. Each triple gives nominal width, AUC and accuracy for classifiers, or nominal width, R² and MAE for regressors. Tehran uses the released sale-price units and UJI uses metres.

| Source | Full forest | MDI | MinPI alone | MDI + MinPI |
|---|---:|---:|---:|---:|
| Internet Advertisements | 1,558 / .9507 / .9033 | 151.0 / .9468 / .9021 | 127.6 / .9449 / .9027 | 127.6 / .9449 / .9027 |
| QSAR oral toxicity | 1,024 / .8168 / .9202 | 269.0 / .8171 / .9198 | 286.6 / .8099 / .9195 | 218.6 / .8123 / .9193 |
| Tehran Residential Building | 79 / .0279 / 873.47 | 30.4 / −.0026 / 884.46 | 42.6 / .0082 / 880.28 | 29.0 / −.0006 / 883.44 |
| UJIIndoorLoc | 520 / .6069 / 31.59 | 18.6 / .5808 / 33.18 | 26.8 / .5809 / 33.06 | 18.2 / .5803 / 33.27 |

Relative to the full forest, source-mean AUC fell by .0058 and .0044. One of five seeds on each classification task lost more than the .01 calibration threshold on the outer rows; the largest single-seed AUC losses were .0196 and .0286, respectively. All classification accuracy losses were below .01. Both regression source means lost more than .02 R² (.0285 on Tehran, .0267 on UJI). Five Tehran and two UJI runs exceeded .02 R² loss individually. On UJI, the mean increase in scaled MAE was .0291 and two runs exceeded .02. Tehran's full forest already had low predictive skill. The acceptance guard does not ensure the same clean-prediction tolerance on new rows. The single MinPI path used more nominal inputs than MDI on three source means. The portfolio includes MDI as an option and its smallest feasible nominal width can only improve or tie MDI under the registered common guard.

The [source protocol](../../../../experiments/minpi_rework/wide_veto_protocol.md) describes target definitions, duplicates, the official later UJI validation cohort and chronological Tehran blocks. The five seeds for UJI share the official test and the five Tehran runs share all chronological partitions. [Timing trials](timings.csv), [paired jobs](paired-jobs.csv), [source-bootstrap timing contrasts](contrasts.csv), [native forest verification](native-audit.csv), [individual outcomes](scores.csv), [source means](source-scores.csv) and the [sealed receipt index](lock.json) accompany the analysis.
