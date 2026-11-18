# Main fixed-budget experiment

The editable implementation lives in `src/minppi_mofs/exact_minpi.py`. The frozen experiment parameters are in `configs/minpi-study.json`; each case records the digest of the configuration, driver, new selector, and two reused source modules. The original PI implementations remain available in their historical modules. The formal comparison uses the compact selector's ranking component on the same witnesses as the new selector.

## Reproduction

Run from the `MastersThesis/history` directory with the shared Python environment.

```powershell
.\thesis-python.ps1 -m pytest tests/test_exact_minpi.py tests/test_piv3_compact_filter.py tests/test_piv2.py -q
.\thesis-python.ps1 -m experiments.minpi_study --pilot --datasets sparse_signal gisette
.\thesis-python.ps1 -m experiments.minpi_study
.\thesis-python.ps1 -m experiments.summarize_minpi_study
```

The driver reads the five local numeric CSVs in `data/raw` and verifies each against its registered `configs/<dataset>.yaml`. The generated benchmarks and three scikit-learn datasets are loaded in memory. The source files named by the official configuration are copied from the thesis data archive. The Communities and Crime CSV is the processed SHAPG snapshot. Only the columns from `population` through `LemasPctOfficDrugUn` are passed to the experiment. The remaining columns include crime rates and counts from which the target can be reconstructed. Missing targets or nonfinite inputs cause this study driver to fail. The selector used outside this benchmark handles missing feature values using its reference partition.

The eight nonsimulated sources are breast cancer, handwritten digits, diabetes, Madelon, Spambase, Communities and Crime, ISOLET, and Gisette. Madelon is a previously published artificial challenge dataset; distinguish it from observational sources. The three project-generated controls have eight planted informative coordinates, a two-coordinate XOR, and random labels, respectively. The global column permutation is independent of all labels and splits. No seed-specific result was used to choose any feature budget.

Every dataset has five separate fixed 75:25 train/test splits. Classification splits are label-stratified. Regression splits use a random permutation of rows. Different splits reuse observations. All supports use training data only. The selected widths equal the ceiling of one tenth of the number of input features. The chained selector first keeps forty percent using exact minPI and then ranks that shortlist by ExtraTrees importance; its final budget still refers to the original feature count.

The ridge learner uses training-only standardization and least-squares ridge classification or regression with alpha one and the LSQR solver. Trees use 48 extremely randomized trees with a square-root feature budget at each node. The standalone forest selector, its counterpart after the chain, and the prediction forest are fitted separately. The binary classification metric is AUC from the decision values or probabilities. Multiclass AUC is the unweighted average of one-versus-rest class AUCs. All classification cases also report accuracy and balanced accuracy, and regression cases report R² and MAE.

## Timing and outputs

The parent process launches one dataset/seed child at a time with a 3,600-second wall-clock deadline. Selection includes construction of the training witnesses and any supervised ranking or secondary forest fit. Each learner's time includes column gathering, training preprocessing, model fitting, and prediction. The selected support and its measured selection time are shared across the two downstream comparisons. Loading data, metric calculation, serialization, and process launch are excluded. Arm and learner order are randomized per split. The timing host uses an Intel Core Ultra 9 275HX with Windows 11 build 26200. BLAS, OpenMP, and the estimators use one thread.

`results/minpi-study/formal/<dataset>/s<seed>/record.json` contains original column IDs, selected positions in the permuted matrix, timings, metrics, and source digests. In the same directory, `predictions.npz` contains split indices, held-out labels, each arm's predictions and continuous scores, and the exact per-tier rates and Monte Carlo estimates. The generator writes complete cases atomically and checks source digests when resuming. Development runs are in `pilot`. Execution failures are recorded alongside the relevant case. Summaries in `summary` retain the paired datasets, seeds and learners.

- `split_metrics.csv` contains one row per fitted downstream model.
- `dataset_medians.csv` contains within-dataset medians across five splits.
- `paired_comparisons.csv` gives per-dataset paired primary-metric differences and time ratios. Its quality and speed flags use the frozen descriptive thresholds.
- `ablation_overlap.csv` compares support sets, identifies strict worst tiers, and records the discrepancy between Monte Carlo and exact scores.
- `support_stability.csv` reports all ten seed-pair Jaccard overlaps aggregated per dataset and arm.

For a fresh reproduction in a separate copy, leave the formal output directory empty. Reusing an existing case with changed source files raises an error instead of silently combining runs from different versions.


The [full-suite validation record](../../results/minpi-study/summary/test-status.json) lists the final counts, the four Windows symbolic-link skips, the certified-solver checks, and the SHA-256 digests of the underlying JUnit reports and logs.
