# MinPI for input compression in boosted trees

The manuscript is in `thesis-paper/minpi-tree.pdf`. Its LaTeX entry point is `thesis-paper/main.tex`. The standalone manuscript archive places the entry point and its bibliography, table and style alongside `main.pdf`. Build with pdfLaTeX, BibTeX, pdfLaTeX and pdfLaTeX in that directory.

The primary cohort contains eight source datasets and a five-source wider extension. Five seeds per source give 65 teacher–calibration pairs. A separate five-source experiment tested scalar recalibration and alternative deletion rules on 25 splits. Its methods and predictions were locked separately. The six original prepared tasks support retrospective comparisons; the aggregate evidence names its development role. Human- and time-structured splits, target definitions, dataset attribution, and derivations are documented in the [eight-source protocol](../experiments/minpi_rework/validation_protocol.md), [wider protocol](../experiments/minpi_rework/validation_wide.md) and [five-source provider](../experiments/minpi_rework/validation_exchange.md). The source files retain their original notices. Tecator spectra were recorded with a Tecator Infratec Food and Feed Analyzer. The Tecator file contains its original redistribution terms and the request to send a preprint upon publication.

## Offline score and table replay

Unzip `minpi-tree-replication.zip` into a new directory. From its root, use a Python environment with the versions in `thesis-paper/tree-requirements.txt`. This workflow was exercised with Python 3.13 on Windows. Dependencies and typesetting tools are supplied by the runtime. The scorer uses the unmodified bytes in the source archives and the recorded forecasts. All 18 raw public inputs are included. Re-fetch commands for the registered inputs are `python -m experiments.minpi_rework.validation_panel --fetch`, `python -m experiments.minpi_rework.validation_wide --fetch` and `python -m experiments.minpi_rework.validation_exchange --fetch`.

```powershell
$root = (Get-Location).Path
$env:PYTHONPATH = (Join-Path $root 'src') + [IO.Path]::PathSeparator + $root
python thesis-paper/tree_release.py verify --tex
```

On a POSIX shell, run `PYTHONPATH="$PWD/src:$PWD" python thesis-paper/tree_release.py verify`. Add `--tex` if TeX Live with the manuscript packages is available. A fresh checkout can use the shared interpreter through `thesis-python.ps1` instead of changing its editable installation.

Verification checks the release manifest and the two historical ZIP hashes, restores the 90 signed per-split receipts and forecast files, then reruns the original scorers. It compares the numerical and nonnumerical columns in the released score and summary tables and reproduces the LaTeX result rows. The 13-source paired analyses also include independently saved per-split predictions for the two post-release ranking controls. The scorer uses the paid training and selection times in the original receipts. New timings are hardware dependent.

## Recomputing models

The shared project package, search implementations, deployment command, formal files and build specifications accompany the measurements. Run `python -m experiments.minpi_rework.compress_model_check`, `python -m experiments.minpi_rework.stratified_tree_checks` and `python -m experiments.minpi_rework.span_geometry` to exercise prediction and numerical invariants offline. The [deployment example](../experiments/minpi_rework/README.md) shows the retained-column predictor. Lean files under `thesis-paper/formal` use the supplied Lean toolchain version. Wolfram Language source and the available execution receipt accompany them.

For fresh public-cohort training, start from a second clean extraction **before restoring the historical predictions**. Keep the original ZIP elsewhere as the reference. From that copy run `python -m experiments.minpi_rework.quota_compress_tree_confirmation collect`, then `python -m experiments.minpi_rework.quota_compress_tree_confirmation score`. The separate five-source equivalent is `python -m experiments.minpi_rework.quota_compress_scale_confirmation collect` followed by `score`. These research drivers expect the registered CUDA XGBoost configuration and use Windows CPU affinity when running on Windows. Fresh fits have their own clocks and receipts. To reproduce the 65-run projection comparison on registered hardware, first restore the published runs and then run `python -m experiments.minpi_rework.span_benchmark`.

For the historical six, recorded forecasts, support decisions, receipt locks and scored results are included under `results/minpi_rework/quota-tree` and `thesis-paper/evidence/minpi_rework/quota-tree`. Their historical train/validation/test tables are supplied separately by the optional local `minpi-tree-prepared-six.zip`. Unpack it into the replication directory, preserving `results/icomp2026_formal/<run>/`. This supplies 30 prepared splits, their preparation manifests and cached original receiver forecasts. Use `python thesis-paper/tree_release.py verify --prepared` for original-six rescoring and `python -m experiments.minpi_rework.stratified_tree_benchmark` for all 95 registered computational workloads. A complete regeneration of the historical preprocessing, archived PIv1 baselines and earlier receiver training also needs their original input releases and protocols from the thesis project.

## Build the archives in the thesis checkout

From `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper`, run

```powershell
& .\MastersThesis\thesis-python.ps1 thesis-paper/tree_release.py build --with-prepared
```

This builds the manuscript from its current source in an empty directory, checks citations, and writes `thesis-paper/minpi-tree.pdf`, `packages/minpi-tree-source.zip`, `packages/minpi-tree-replication.zip` and the optional local prepared-data ZIP. Omit `--with-prepared` to regenerate the manuscript and main replication package alone. The ZIPs include byte-level content manifests. The local prepared-data supplement preserves upstream data restrictions. Only the specified cohorts and controls are part of this release.

Run `& .\MastersThesis\thesis-python.ps1 thesis-paper/tree_release.py verify-archives --prepared` from the same directory to check fresh extracted copies of all three archives. This requires a working Python environment, pdfLaTeX, BibTeX and pdftotext. The historical signed protocol contains original local paths. The research replication bundle therefore needs an anonymity review before release to a double-blind venue.
