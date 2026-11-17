# Time saved by an exact minimum vote

All 18 development and seven external sources were reconstructed at five splits each. The labels used here belong to fitting and voting rows. The replay checked the entire greedy order at every step and verified its suffixes and original evaluation counts against the four archived guards for each split. All 125 paths passed. It does not recompute held-out forecasts or scores.

For each candidate the production loop tries the prior worst size, the two endpoints, then the remaining sizes. Duplicate size visits are suppressed. Once its observed worst loss exceeds the incumbent plus the fixed tolerance, no unseen size could improve the candidate. The direct matched control computes every dot product in the same order and uses identical candidate ties. The BLAS control evaluates the dense candidate-by-size matrix. The exhaustive query count for one complete p-step ranking is p²(p+1)/2, counting endpoints. The tests separately cover p=1, p=2, exact ties and near ties.

| Scope | All 25 | Development 18 | External 7 |
|---|---:|---:|---:|
| Size queries, early / matched exhaustive | .410 [.344,.488] | .371 [.298,.465] | .529 [.454,.602] |
| Decision time, early / matched exhaustive | .570 [.476,.674] | .492 [.397,.603] | .831 [.780,.869] |
| Decision time without Python dispatch, same pair | .492 [.409,.584] | .424 [.341,.523] | .721 [.650,.777] |
| Ranking with original union updates, same pair | .976 [.947,.999] | .963 [.926,.989] | 1.012 [.994,1.042] |
| RF fit + path and moment construction + ranking, same pair | .9991 [.9982,.9997] | .9988 [.9976,.9996] | .9999 [.9998,1.0001] |
| Ranking with original union updates, early / dense BLAS | .858 [.786,.918] | .867 [.777,.934] | .836 [.719,.940] |

The decision clock includes per-step Python dispatch for both implementations unless explicitly removed. Three warmed repeats interleave the methods on one pinned CPU with one BLAS thread. Decision replays use the same stored coefficients at each greedy step and cycle identical paths enough times to resolve microseconds. Complete ranking is timed on live state and charges union updates. Construction is timed once per split and charged equally. The last construction-inclusive metric ends at ranking and excludes the width guard and prediction. Intervals resample whole sources after taking the median within split and averaging paired log ratios within source.

At the smallest source (Coimbra, p=9), seed 42 needed 206 of 405 possible queries and 5.4 vs 6.1 microseconds of total decision time. The full path took about 1.03 ms for either native version. At CT Location (p=96), the same split needed 150,930 of 446,976 queries and the full rank took 2.21 vs 3.18 ms. Tecator (p=100) needed only 74,500 of 505,000 queries, yet the full rank took 21.34 vs 22.67 ms because its state updates were more expensive. Width alone does not determine the saving.

## Straight-loop and update controls

The supplementary benchmark fuses the iterative native loop and applies the same union updates without allocating a temporary union vector for each active pair. All 125 orders remained unchanged. With identical early vetoes this update took .396 [.378,.419] times the original allocating-update rank time; the fit-to-rank ratio was .973 [.964,.982]. This improvement applies independently of the selector.

With the new update, early veto took .934 [.884,.974] times the full ranking time of the matched exhaustive native method. A stronger exhaustive control visits integer sizes in ascending order and avoids endpoint-order bookkeeping. Early veto took .961 [.923,.991] times its full ranking time on all sources and .999 [.994,1.004] on the seven external sources. Ratios by width were 1.003 [.996,1.013] for p≤20, .993 [.989,.998] for 21–50, and .867 [.786,.941] for p>50. All external sources have fewer than 20 variables. The observed larger-width saving is development evidence. The whole fit-to-rank ratio against straight exhaustive is .9992 [.9985,.9998].

The original-protocol data are in `repeats.csv`, `split-medians.csv`, `source-contrasts.csv` and `contrasts.csv`. Their `native-` counterparts record the stronger implementation controls. The corresponding executable modules and prespecifications are `predictive_veto_timing`, `predictive_veto_native` and `predictive_veto_report` under `experiments/minpi_rework`.
