# Sampled MinPI compression experiments

| Comparison | Fixed design | Scores and implementation |
|---|---|---|
| Eight new sources, grouped splits | [Primary protocol](protocol.json) and [zero-tolerance sensitivity](zero_tolerance.json) | [Per-source method means](../../thesis-paper/evidence/minpi_compression/confirmation/sources.csv), [independent audit](../../thesis-paper/audit/minpi-compression-audit.md) |
| Matched batched and direct ridge solves | [Timing design](paired_cost.md) | [Wider inputs](../../thesis-paper/evidence/minpi_compression/paired-cost-wide.csv), [eight new sources](../../thesis-paper/evidence/minpi_compression/paired-cost-confirmation.csv) |
| Six wider sources, 16 existing folds | [Transfer protocol](wide_transfer.md) | [All methods and folds](../../thesis-paper/evidence/minpi_compression/wide-transfer/scores.csv) |
| Partial Boolean information and chain search | [Sharp bound and controls](dictator_replay.md), [registered search](chain_search.md) | [Domain-level closure effects](../../thesis-paper/evidence/minpi_compression/dictator/domain-effects.csv), [paired-query replay](chain_replay_notes.md) |
| Four new UCI sources for the chain comparison | [Protocol and data credits](chain_validation_protocol.md) | [Scores, random ablation and commands](../../thesis-paper/evidence/minpi_compression/chain-validation/README.md) |
| Greedy leave-one-out regularized least squares on these four sources | [Comparator and timing design](greedy_rls.md) | [Paired contrasts](../../thesis-paper/evidence/minpi_compression/greedy-rls/effects.csv) |
| One validated deletion from a coefficient-ranked prefix | [Registered witness, tail and random deletion controls](witness_guard.md) | [Optional refits and scores](../../thesis-paper/evidence/minpi_compression/witness-guard/README.md) |
| Thirty original splits, direct Gain, matched downstream models | [Ridge design](legacy_gain_protocol.md), [repeated-input check](legacy_gain_overlap.md), [boosted receiver](legacy_gain_receiver.md), [boosted width guard](legacy_gain_strong_guard.md) | [Original splits, results and commands](../../thesis-paper/evidence/minpi_compression/legacy-gain/README.md) |

The [offline release guide](release.md) covers verification of saved forecasts. Formal material includes the [ridge update](../../thesis-paper/formal/SampledMarginal.lean), [positive-index characterization](../../thesis-paper/formal/Dictator.lean), [partial-information bound](../../thesis-paper/formal/PartialUpper.lean) and [Wolfram Language calculation](../../thesis-paper/formal/SampledMarginal.wl).

From the checkout root, check the three Lean statements with the pinned Lean 4.35.0:

```powershell
Push-Location thesis-paper/formal
lean SampledMarginal.lean
lean Dictator.lean
lean PartialUpper.lean
Pop-Location
```

In a Wolfram Language session using the checkout root as working directory, evaluate `Get["thesis-paper/formal/SampledMarginal.wl"]`. The script checks the Schur equations and held-out loss using exact arithmetic; its expected value is `{True, True, True, True}`. The local symbolic checks were executed with Mathics3 10.0.1.

The confirmation panel contains four classification sources and four regression sources. Five grouped 80/20 splits are repeated on every source. Rankings are fitted within development data. A separate width-validation partition chooses the smallest eligible width; final models use all development rows. The pilot comprises three historical PMLB classification datasets and the scikit-learn diabetes regression dataset.

| UCI source | Target | Rows | Inputs | Distinct split groups |
|---|---|---:|---:|---:|
| Glass identification | Glass type | 214 | 9 | 213 |
| Raisin | Variety | 900 | 7 | 900 |
| Wireless indoor localization | Room, among the same four rooms | 2,000 | 7 | 2,000 |
| One hundred plant species | Species, using the margin descriptor view | 1,600 | 64 | 1,600 |
| Airfoil self-noise | Scaled sound pressure level | 1,503 | 5 | 33 angle/chord settings |
| Yacht hydrodynamics | Residuary resistance | 308 | 6 | 22 hulls |
| Concrete slump test | 28-day compressive strength | 103 | 7 | 103 |
| Energy efficiency | Heating load | 768 | 8 | 12 building geometries |

The split otherwise groups identical predictor vectors. The loader excludes sample IDs, unused target variables and alternate descriptor views. Validation and test partitions for the plant species task contain all 100 classes. The number of distinct engineering settings limits the precision of the three grouped regression evaluations.

From the isolated thesis checkout, run the following PowerShell commands in order. Downloading the eight UCI source archives requires network access. The three PMLB pilot files are available under `data/context_study/pmlb` in the existing thesis workspace. The source IDs, archive names, parser and SHA-256 digests are in `sources.py` and `protocol.json`.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.sources
foreach ($phase in @("pilot","confirmation")) {
    .\thesis-python.ps1 -m experiments.minpi_compression.benchmark select $phase
    .\thesis-python.ps1 -m experiments.minpi_compression.zero_tolerance select $phase
    .\thesis-python.ps1 -m experiments.minpi_compression.benchmark score $phase
    .\thesis-python.ps1 -m experiments.minpi_compression.zero_tolerance score $phase
    .\thesis-python.ps1 -m experiments.minpi_compression.contrasts $phase
    .\thesis-python.ps1 -m experiments.minpi_compression.export $phase
}
```

Selection writes supports, repeated clock readings, inner curves and predictions under `results/minpi_compression`. Score generation checks the per-job lock before comparing the frozen predictions with the outer labels. The `forecasts.zip` under each evidence phase contains the complete primary traces and prediction matrices. Selection reuses verified completed jobs. For a fresh timing replication, run in a clean thesis worktree with a fresh results directory. The benchmark reserves logical CPU 6 on Windows and one BLAS thread. Each log update reports source, seed, job time, experiment elapsed time and estimated time remaining.

The `scores.csv` rows are outer source/seed/method observations. Source summaries average the five seeds. The `contrasts.csv` files estimate paired differences and geometric ratios with equal source weights. Percentile intervals for the primary comparisons enumerate all ordered source resamples within a domain. Four purposively sampled sources per domain make these intervals descriptive; the regression pilot has only one source. The zero-tolerance sensitivity reuses the same primary ranks and changes the eligibility check for every method. It has a separate evidence directory and no timing claim.
