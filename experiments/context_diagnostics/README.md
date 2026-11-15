# Does the identity of a restoration change a deletion vote?

These two experiments compare rules on identical proposals. Their outcomes concern masked predictions of a frozen model. Subsequent refitting answers a different question. The [design](../../configs/context_diagnostics/design.json) was frozen locally before computing these outcomes, after the original six-dataset benchmark had been inspected. The design records the complete scenario grid, sources of randomness and denominators. The synthetic study distinguishes the data-generating process, the population target and observable ballot errors following the reporting approach of [Morris, White and Crowther](https://doi.org/10.1002/sim.8086). The distinction between a masking intervention and a downstream retrained predictor follows [Covert, Lundberg and Lee](https://jmlr.org/papers/v22/20-1316.html).

## Reproduce from the thesis checkout

Run the following from the `MastersThesis` directory, using the thesis Python launcher. The synthetic experiment requires NumPy and the project package. The archived-tree run requires the original prepared inputs, effective search records, and the CUDA XGBoost version used for that experiment. All data it reads are local. Each dataset–seed process has a one-hour watchdog. A completed run can resume from its saved receipts. The comparator and the restoration family use the same maximum number of distinct masks and the same sequence of cardinalities.

```powershell
.\thesis-python.ps1 -m experiments.context_diagnostics.synthetic
.\thesis-python.ps1 -m experiments.context_diagnostics.real
.\thesis-python.ps1 -m experiments.context_diagnostics.summarize
```

The default output is `results/thesis_benchmark/context-mechanism-r1/`. The synthetic subdirectory contains every decision, duplicate-layout controls included. The trees subdirectory contains the following.

| File | Contents |
|---|---|
| `seed-*/<dataset>/receipt.json` | Fixed proposals, tested and novel extensions, archived decision bits, and the mask-to-prediction index |
| `seed-*/<dataset>/predictions.npz` | Local ranking and reserved-row predictions, row indices, labels and four paired weights |
| `decisions.csv` | One row per candidate and rule, including the actual number of visited contexts, agreement on re-evaluation and number of held-out failures |
| `by-split.csv`, `by-dataset.csv`, `overall.csv` | Counts with explicit eligible denominators; pooled proposal counts weight tasks with more eligible candidates more heavily |
| `same-size-disagreement.csv`, `same-size-pairs.csv` | Variation across novel masks of the same cardinality and direction of pairwise decisions on both data partitions |

`summarize` checks original signatures, replays the original truncated votes, and recomputes alternative decisions from predictions. It cross-checks a prespecified sample of four mask predictions from every partition against scikit-learn's reference metrics. Use `--audit-all` to check every mask. A copy of the evidence can be analyzed with `--portable` to skip the hash comparison against the original archive. The verifier still checks the supplied mask predictions and archived ballot bits. These data checks are conditional on the supplied predictions.

## How to read the two experiments

The synthetic model uses independent signs, a three-feature anchor, five tail features and one independent noise sign. Dropping feature zero removes a weak main effect and, in the interaction settings, one or two fitted interactions. Its population squared error is computed two ways, using orthogonality and by enumerating all 512 feature/noise states. Each proposed deletion is tested against all 32 tail subsets to determine the population truth. The two mixed-sign terms in the opposed-interaction example expose a blind spot of a nested chain. The additive, correctly specified interaction and spurious-only mechanisms show what the extra contexts do when that mechanism is absent. Directed tail positions are enumerated, and common finite samples are shared across positions. There are 20 Monte Carlo data seeds, with dependent results within seed.

For real data, the candidate list is frozen from a previous MinPI trajectory, so decisions can be compared without confounding from subsequently different search paths. The ranking panel reproduces the original predictor and ballots. Re-evaluation uses rows reserved by the original split and a common set of finite unseen feature masks. The latter exclude all design masks used by the seven rules. Comparisons against the same-size random family separate the effect of the chosen identities from the number of questions asked. A false acceptance in these tables means that an accepted proposal fails on at least one *sampled* novel extension on the reserved panels. A rejection without a sampled failure provides limited evidence when only a few novel masks are available. A tail with no possible novel interior extension contributes to cost and coverage reporting only. Different candidates and contexts from the same fitted model share data and predictions.
