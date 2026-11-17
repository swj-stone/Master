# Exact solver comparison on historical fitted games

Eight previously used datasets each supplied one linear additive predictor fit on the archived seed-42 training rows. The objective was squared error on the disjoint voting rows. Energy ranked the input columns. At width ceil(p/2), the leading columns above a 28-column boundary band were fixed in and those below were fixed out. For at most 28 original inputs, the entire feature set was available. The optimization result is conditional on this region and the fitted predictor. The benchmark evaluates computation. It does not supply new outer prediction measurements. [UCI sources](../../../../experiments/minpi_compression/data-credits.md) · [Original prepared splits](../../../../experiments/minpi_compression/legacy_gain_protocol.md)

The source list and optimization budget were committed before running the solvers. Forest Fires has 14 encoded columns, including four periodic month/day terms. The planning document listed 12 by mistake; the experiment used the full archived 14-column matrix at width seven. The final replay used compiled screening and enumeration from commit `ba095e0` and the two-word branch-and-bound from `a426c17`. [Native protocol](../../../../experiments/minpi_rework/scip_protocol.md) · [Secondary protocol](../../../../experiments/minpi_rework/scip_bnb_protocol.md) · [Implementation hashes](versions.json)

## Solvers and timing

* **Enumeration** traversed fixed-cardinality subsets in compiled code. In screened problems it pruned subsets inconsistent with certified precedence.
* **SCIP RLT** represented each pair product with a bounded continuous variable, its three McCormick inequalities and, for each feature, the identity that the incident products sum to (k−1) times its indicator. **SCIP convex** factored the regional positive semidefinite Gram matrix, bounded each factor coordinate by its attainable k-subset range, and minimized the equivalent squared-norm epigraph objective. Both used native presolve and default cuts/heuristics, one thread and a ten-second solver limit. Each native call received the same feasible one-swap local optimum for its unscreened regional game, projected onto the residual coordinates when applicable. [SCIP parameters](https://scipopt.org/doc/html/PARAMETERS.php) · [PySCIPOpt epigraph formulation](https://pyscipopt.readthedocs.io/en/stable/tutorials/expressions.html)
* **Specialized branch-and-bound** uses a cardinality upper plane. At a node with t slots, add to each available item's linear contribution half the sum of its t−1 largest remaining pair coefficients. The sum of the t largest resulting weights bounds every completion because each internal pair contributes to its two endpoints. The compiled code supports signed interactions and solves the complementary exclusion problem above half width. All three screening arms use the same regional starting incumbent. Direct enumeration and row ordering supply separate checks. [Gallo, Hammer and Simeone, *Quadratic knapsack problems*](https://doi.org/10.1007/BFb0120892)

Each source and arm had three timed repetitions in shuffled order, pinned to the same logical processor. Source loading and JIT compilation were excluded. Search time includes screening, incumbent construction where applicable, solver construction, optimization and extraction. Total time also includes fitting the additive predictor and constructing its quadratic game and region from in-memory data. The specialized solver was validated by brute force over 520 narrow game/width/precedence combinations, plus tests through 128 columns with ties, precedence and complementary widths. The real regional games used double-precision coefficients. A solver's optimal status refers to its numerical tolerances; the incumbent and dual bounds are also reported in the original objective units.

## Results

The following times, in seconds, sum the eight source medians across three repeats. The ratio in the last column uses the geometric mean of the eight within-source total-time ratios. SCIP time limits remain in the denominator.

| Solver | Direct search | Width-free search | Fixed-width search | Direct total | Fixed-width total | Fixed-width/direct total ratio |
|---|---:|---:|---:|---:|---:|---:|
| Compiled enumeration | 2.179 | 1.033 | 1.029 | 2.869 | 1.712 | 0.516 |
| SCIP RLT | 12.492 | 12.025 | 12.140 | 13.179 | 12.830 | 0.771 |
| SCIP convex epigraph | 12.076 | 11.864 | 11.774 | 12.738 | 12.457 | 0.749 |
| Specialized branch-and-bound | 0.155 | 0.156 | 0.158 | 0.789 | 0.806 | 1.034 |

All 216 native/enumeration results and all 72 specialized-solver results attained the direct enumerator's objective in the original double-precision game. SCIP RLT certified every arm within the budget. SCIP convex stopped at the limit for all three repetitions of each Leaves arm. Its reported relative gap was 25.22–25.42% even though its incumbents attained the separately enumerated optimum. The relative gap uses the solver's residual objective. The raw records give dual bounds translated to the original regional utility. Some native optimizer clock values round to whole seconds on this installation; use `opt_wall_s` for the high-resolution wall time spent inside the optimizer. [Native and enumeration runs](runs.csv) · [Source medians](by-source.csv) · [All backend contrasts](policy-summary.csv)

The specialized direct solver was fastest in summed search and total time. Fixed-width screening sped up its search on two sources (Fraud and Madelon) and increased the source-equal geometric mean search time by 16.9%. It increased the geometric mean total time by 3.4%. Fixed-width and width-free screening visited the same number of nodes on seven sources. On Forest Fires they visited 140 and 163, respectively. Direct search visited 37. Screening an excluded column from the 14-feature problem changed the solver from selecting seven to omitting six of the 13 residual columns. The relative cost of screening therefore depends on the downstream search algorithm and its orientation. The results do not establish a general end-to-end speed advantage over specialized branch-and-bound. [Specialized runs](bnb-runs.csv) · [Median costs and node counts](bnb-by-source.csv)

An earlier development replay used the initial NumPy screening implementation. Its native and enumeration source medians are retained separately. All numerical conclusions above use the frozen compiled implementation. [Earlier timings](prerevision-by-source.csv)

## Reproduction

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.scip_comparison
.\thesis-python.ps1 -m experiments.minpi_rework.scip_comparison --export
.\thesis-python.ps1 -m experiments.minpi_rework.scip_bnb_comparison check
.\thesis-python.ps1 -m experiments.minpi_rework.scip_bnb_comparison run
.\thesis-python.ps1 -m experiments.minpi_rework.scip_bnb_comparison export
```
