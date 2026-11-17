# Supervised coalition votes and early refit termination

Seed 42, six prepared datasets and eight previously studied UCI sources. All timings below exclude common file I/O. Each source has equal weight. The full-reference XGBoost and its restricted models use the same fitting rows, 64 depth-four hist rounds, learning rate .12 and one CPU thread. The voting rows and width-selection rows are disjoint. The width grid is 1, ceil(p/8), ceil(p/4), ceil(p/2), p. The selected subset must pass both native metrics against the full model on width-selection rows. The six prepared splits are unchanged. These fourteen sources were previously observed in this research program.

The joint criterion compares each method with its matched direct Gain arm on the same source. It requires shorter mean runtime, an equal or smaller subset, and outer macro OVR AUC and accuracy losses of at most .01 each, or R² loss and scaled-MAE increase of at most .02 each. Scaled MAE uses the fitting target standard deviation. The 5% column replaces any time saving with at least 5%. These empirical thresholds measure observed differences on one split per source. They provide no population equivalence guarantee.

| Study and rule | Time ratio¹ | Width ratio¹ | Faster | Equal or smaller | Quality pass | Joint | Joint at 5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exhaustive grid, likelihood minimum | 1.046 | 1.000 | 1 | 11 | 10 | 0 | 0 |
| Exhaustive grid, likelihood pooled | 1.051 | .959 | 0 | 11 | 10 | 0 | 0 |
| Exhaustive grid, landmark minimum | 1.193 | .597 | 1 | 12 | 9 | 0 | 0 |
| Exhaustive grid, landmark pooled | 1.192 | .632 | 1 | 11 | 10 | 0 | 0 |
| Actual refit, intact checkpoint | .885 | 1.000 | 9 | 14 | 14 | 9 | 7 |
| Actual refit, pooled checkpoint | .909 | 1.000 | 10 | 14 | 14 | 10 | 9 |
| Actual refit, minimum checkpoint | .875 | 1.157 | 9 | 12 | 13 | 8 | 6 |
| Actual refit, minimum, zero-SE cutoff | .857 | 1.443 | 10 | 11 | 13 | 9 | 5 |
| Actual refit, minimum, two-SE cutoff | .903 | 1.051 | 7 | 13 | 14 | 6 | 4 |

¹ Geometric mean of per-source ratios to the corresponding direct Gain procedure. Counts out of 14. The grid study charges every stand-alone fit, rank and selected prediction. Duplicate supports share the same measured fit cost, charged to each method independently. Both supervised subset rankers have a single warm Numba compilation step excluded from times. In the refit study, the whole selector runs separately for every arm, including the full model, validation, scoring, callbacks and selected outer prediction. The direct Gain implementation stops on the first feasible width. Two shuffled runtime repetitions produce identical selected predictions. Reported times are their arithmetic mean per source. Tiny source timings are noisy, e.g. the minimum/direct ratios on Coimbra were 1.36 and .52.

### Where the procedures failed

Landmark voting uses the worst size-specific change in held-vote squared/Brier loss to rank features. The ranker kept strictly fewer columns on six sources. Of those six, Gisette (50 to 25) and Superconductor (21 to 11) met both quality limits. On Retina, 18 to 5 cost .072 AUC and .060 accuracy. On Music, 68 to 1 cost .146 R². The average classification AUC change of −.001 obscures these errors as well as a gain of .228 on Anuran. No source met the joint target for either minimum ranker on the grid or the single-candidate half-width control.

The second study interrupts the *actual* restricted XGBoost fit after round eight on an unfavorable vote. All arms use the same Gain ordering. Once an early veto removes a candidate it cannot return. Direct selection chooses the narrowest feasible prefix, so any veto procedure on this fixed path can only tie or exceed the direct width. If C more candidate fits are begun and V candidates receive a veto, the difference in completed boosting rounds is exactly 56V − 64C in favor of the veto arm. Operational savings can be smaller because the width, row count and number of class trees vary across fits and the votes take time.

Across the 14 sources, the primary minimum veto saved 1,384 of 3,776 boosting rounds but removed the direct winner on ISOLET (50 to 100 columns) and Madelon (13 to 50). On Madelon it also took .209 s against .155 s and lost .042 AUC and .031 accuracy. On ISOLET the larger model recovered .012 accuracy, but the target requires compactness too. The minimum passed the joint target on 3/6 prepared and 5/8 development sources. Pooled and intact checkpoints saved 1,064 rounds each, kept the direct winner and produced exactly its predictions on all 14 sources. The data support investigating partial-training futility checks. They do not support an advantage from taking the minimum across availability levels.

### Files and reproduction

- `paired.csv` holds per-source widths, wall times and labeled native scores for both arms of each comparison. `summary.csv` includes the single-feature, pooled, random and limited-width controls.
- `race/vetoes.csv` lists rejection counts, direct winners lost and actual boosting iterations by method and source. `race/scores.csv` separates the two full runtime repetitions. Selection receipts contain the individual rejected widths and retained columns.
- `archived_gpu_direct.csv` reproduces actual full-run wall time, retained widths and held scores from the original 100-round GPU, full-development direct Gain on the six prepared sources, seed 42. It differs in fitting and width search, metric acceptance, hardware and timing boundary. The matched comparisons above use the contemporary native direct arm.
- `experiments/refit_voting/protocol.md` and `race-protocol.md` contain the development designs. The first study's forecasts precede scoring in commit `4f25dae`. The second study's forecasts and split-index hashes precede scoring in `844c3b4`. Neither arm opened the separate four-source replication panel.

From the thesis worktree, recompute scored summaries with `./thesis-python.ps1 -m experiments.refit_voting.run score`, `./thesis-python.ps1 -m experiments.refit_voting.race score`, and `./thesis-python.ps1 -m experiments.refit_voting.analyze`. Each scoring program verifies forecast hashes. The corresponding `collect` commands reuse published receipts if present.

