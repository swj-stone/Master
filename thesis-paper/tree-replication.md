# MinPI with incomplete feature availability

The manuscript is in `thesis-paper/minpi-tree.pdf`. Its LaTeX entry point is `thesis-paper/main.tex`. The standalone manuscript archive places the entry point and its bibliography, table and style alongside `main.pdf`. Build with pdfLaTeX, BibTeX, pdfLaTeX and pdfLaTeX in that directory.

The supervised MinPI study uses seven held-out UCI sources, each with seeds 42–46. The learner is a depth-limited random forest. For each nonempty acquisition size the selector computes the expected excess Brier loss (classification) or squared loss (regression) under a uniform mask of that size. A worst-size guard and intact validation metrics determine the accepted width. The registered portfolio includes an MDI path and a MinPI path. The analytic versus Monte Carlo comparison uses the same 35 source–seed splits and its own locked searches and forecasts. These are two experiments on seven study populations. [Data provenance and split units](../experiments/minpi_rework/validation_predictive.md) · [Supervised protocol](../experiments/minpi_rework/predictive_confirmation.md) · [Monte Carlo protocol](../experiments/minpi_rework/predictive_mc_confirmation.md)

The six original prepared tasks are development evidence for the supervised selector. The earlier 13-source, five-seed teacher-fidelity study and the separate five-source recalibration study had both been opened during development of this objective. Their original signed forecasts remain reproducible, and their outcomes retain the scope of their own protocols. [Eight-source protocol](../experiments/minpi_rework/validation_protocol.md) · [Wider protocol](../experiments/minpi_rework/validation_wide.md) · [Five-source provider](../experiments/minpi_rework/validation_exchange.md)

The public input files retain their original notices. Tecator spectra were recorded with a Tecator Infratec Food and Feed Analyzer. The Tecator file contains its redistribution terms and the request to send a preprint upon publication.

## Offline score and table replay

Unzip `minpi-tree-replication.zip` into a new directory. From its root, use a Python environment with the versions in `thesis-paper/tree-requirements.txt`. This workflow was exercised with Python 3.13 on Windows. Dependencies and typesetting tools are supplied by the runtime. The scorer uses the unmodified bytes in the source archives and the recorded forecasts. All 25 raw public inputs are included (7 primary and 18 earlier). Re-fetch commands for the registered inputs are `python -m experiments.minpi_rework.validation_predictive --fetch`, `python -m experiments.minpi_rework.validation_panel --fetch`, `python -m experiments.minpi_rework.validation_wide --fetch` and `python -m experiments.minpi_rework.validation_exchange --fetch`.

```powershell
$root = (Get-Location).Path
$env:PYTHONPATH = (Join-Path $root 'src') + [IO.Path]::PathSeparator + $root
python thesis-paper/tree_release.py verify --tex
```

On a POSIX shell, run `PYTHONPATH="$PWD/src:$PWD" python thesis-paper/tree_release.py verify`. Add `--tex` if TeX Live with the manuscript packages is available. A fresh checkout can use the shared interpreter through `thesis-python.ps1` instead of changing its editable installation.

Verification checks four public forecast archive hashes. For each of the 70 supervised study and simulation-comparison receipts it restores the frozen predictions, sparse path increments, sparse path incidence and JSON selection file. It rebuilds the held-out cardinality-risk and predictive scores, repeats the independent explicit-mask audit on one seed per source, reconstructs the benchmark timing summaries from signed measurement rows, and regenerates the typeset results from the frozen score tables. It also replays the earlier 65+25 locked runs, including scores from the two separately archived ranking controls. All 160 public receipts and 90 historical six-source arm-family receipts are checked. Calibration and timing observations retain their original hardware, masking and split conditions.

The exact feature-lattice and early-veto controls were added after the seven-source outcomes had been opened. The verifier checks 510 small synthetic lattice costs, refits all 35 public splits, and compares every released lattice support, risk and predictive score. Elapsed time varies on a new machine. For the veto benchmark it regenerates the six paired summary tables from both sets of repeated measurements on the 18 development and seven external sources. The original clocks are in `thesis-paper/evidence/minpi_rework/predictive-veto-timing/{repeats,native-repeats}.csv`.

## Recomputing models

The shared project package, search implementations, formal files and build specifications accompany the measurements. Run `python -m experiments.minpi_rework.predictive_mask_study --check` for small exact games. Run `python -m experiments.minpi_rework.compress_model_check`, `python -m experiments.minpi_rework.stratified_tree_checks` and `python -m experiments.minpi_rework.span_geometry` to exercise prediction and numerical invariants offline. The older XGBoost [deployment example](../experiments/minpi_rework/README.md) shows the retained-column predictor. Lean files under `thesis-paper/formal` use the supplied Lean toolchain version. Wolfram Language source and the available execution receipt accompany them.

For new supervised random-forest fits, use a separate fresh extraction before restoring results. Run `python -m experiments.minpi_rework.predictive_confirmation collect`, then `python -m experiments.minpi_rework.predictive_confirmation score`. To repeat the timing benchmark on the same seven sources run `python -m experiments.minpi_rework.predictive_mc_confirmation` and `python -m experiments.minpi_rework.predictive_mc_score`. These CPU runs use training-only imputation and disjoint fit/vote/acceptance/outer partitions. The Monte Carlo runner reports budgets skipped by its computational eligibility rule. New runs replace that extraction's original archives; retain the distributed copies for comparison.

To collect new veto timings, first restore the signed runs through `verify`. In that working copy, rename `thesis-paper/evidence/minpi_rework/predictive-veto-timing` and create an empty directory of the same name. Run `python -m experiments.minpi_rework.predictive_veto_timing`, `python -m experiments.minpi_rework.predictive_veto_native`, then `python -m experiments.minpi_rework.predictive_veto_report` with and without `--native`. Both collectors resume from existing rows and validate recovered orders and supports against the original ones.

For fresh public-cohort training, start from a second clean extraction **before restoring the historical predictions**. Keep the original ZIP elsewhere as the reference. From that copy run `python -m experiments.minpi_rework.quota_compress_tree_confirmation collect`, then `python -m experiments.minpi_rework.quota_compress_tree_confirmation score`. The separate five-source equivalent is `python -m experiments.minpi_rework.quota_compress_scale_confirmation collect` followed by `score`. Those earlier research drivers expect the registered CUDA XGBoost configuration and use Windows CPU affinity when running on Windows. Fresh fits have their own clocks and receipts. To reproduce the 65-run projection comparison on registered hardware, first restore the published runs and then run `python -m experiments.minpi_rework.span_benchmark`.

For the historical six, recorded forecasts, support decisions, receipt locks and scored results are included under `results/minpi_rework/quota-tree` and `thesis-paper/evidence/minpi_rework/quota-tree`. Their historical train/validation/test tables are supplied separately by the optional local `minpi-tree-prepared-six.zip`. Unpack it into the replication directory, preserving `results/icomp2026_formal/<run>/`. This supplies 30 prepared splits, their preparation manifests, cached historical receivers and the 30 supervised forest forecasts paired with the score rows in `predictive-prepared`. Use `python thesis-paper/tree_release.py verify --prepared` for original-six rescoring and `python -m experiments.minpi_rework.stratified_tree_benchmark` for all 95 registered computational workloads. A complete regeneration of the historical preprocessing, archived PIv1 baselines and earlier receiver training also needs their original input releases and protocols from the thesis project.

## Build the archives in the thesis checkout

From `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper`, run

```powershell
& .\MastersThesis\thesis-python.ps1 thesis-paper/tree_release.py build --with-prepared
```

This builds the manuscript from its current source in an empty directory, checks citations, and writes `thesis-paper/minpi-tree.pdf`, `packages/minpi-tree-source.zip`, `packages/minpi-tree-replication.zip` and the optional local prepared-data ZIP. Omit `--with-prepared` to regenerate the manuscript and main replication package alone. The ZIPs include byte-level content manifests. The local prepared-data supplement preserves upstream data restrictions. Only the specified cohorts and controls are part of this release.

Run `& .\MastersThesis\thesis-python.ps1 thesis-paper/tree_release.py verify-archives --prepared` from the same directory to check fresh extracted copies of all three archives. This requires a working Python environment, pdfLaTeX, BibTeX and pdftotext. The historical signed protocol contains original local paths. The research replication bundle therefore needs an anonymity review before release to a double-blind venue.
