# Reproduce the feature-level MinPI study

The archive contains the delivered PDF and its standalone LaTeX source ZIP in `thesis-paper/`. The [experiment guide](../../experiments/minpi_compression/README.md) identifies each comparison, its data and clock convention. [Primary audit](../../thesis-paper/audit/minpi-compression-audit.md) · [Four-source transfer](../../thesis-paper/evidence/minpi_compression/chain-validation/README.md) · [Six original datasets](../../thesis-paper/evidence/minpi_compression/legacy-gain/README.md).

## Offline verification

From the thesis checkout, the following command verifies the member hashes and executes the archived audit in a temporary extraction:

```powershell
.\thesis-python.ps1 thesis-paper/package_revision.py --verify-compression packages/thesis-minpi-compression.zip
```

Alternatively extract the ZIP yourself to a clean directory. Use a Python environment matching `pyproject.toml` and `requirements.txt`. From the extraction root run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) src) + [IO.Path]::PathSeparator + (Get-Location).Path
python -m experiments.minpi_compression.replay_release
```

This replays prediction and selection receipts from the eight-source confirmation, strict width sensitivity, four-source chain validation, the retrospective twelve-source deletion check and its prospective six-source transfer, six wider sources, and the original thirty matched Gain splits. For the transfer it checks the attainable compression bounds from the saved prefix and negative votes, the molecule-level scores on Musk and the hectares errors on Forest Fires. It also checks the frozen-support and XGBoost-width receiver comparisons against their stored forecasts. Timings in the published tables use the clocks saved at selection time. A replay verifies their aggregation. Measuring new wall times requires fresh selection runs on the desired hardware.

## Inputs and retraining

The [dataset credits](../../experiments/minpi_compression/data-credits.md) give a citation and provider license for each UCI source, the PMLB release and the origin of the scikit-learn pilot.

* The eighteen UCI source archives in `data/minpi_compression/` retain their original contents. The [eight-source loader](../../experiments/minpi_compression/sources.py) and [hashes](../../experiments/minpi_compression/protocol.json), [four-source acquisition script](../../experiments/minpi_compression/chain_validation_sources.py), and [six-source acquisition script](../../experiments/minpi_compression/witness_transfer_sources.py) allow re-acquisition from UCI. The [witness-transfer protocol](../../experiments/minpi_compression/witness_transfer.md) specifies chronological and group splits and the additional outcome units for Musk and Forest Fires. Musk's original `clean1.data.Z` requires a `gzip` executable on PATH with support for Unix compress. Pilot inputs are the three archived PMLB tables and scikit-learn's Diabetes dataset.
* Six UCI sources in `results/context_review/wide-uci/` supply the archived wider folds. Each fold has input and target arrays, their hashes and its source-specific layout. The new transfer uses up to the first 100 prepared columns. The [wide transfer protocol](../../experiments/minpi_compression/wide_transfer.md) and [paired cost design](../../experiments/minpi_compression/paired_cost.md) fix the two comparisons.
* The original prepared six-source training, validation and test CSVs for all five seeds are in the project's earlier `packages/thesis-piv1-evidence.zip`. Its recorded SHA-256 is `3a66b3673d8a4f6d3962ee4c5ad90a07c01b1cd97bff28504dab16ef592ec651`. For fresh runs on these splits, place the original archive under `packages/` and extract its `results/icomp2026_formal/` tree into the checkout at that relative path. The [original-split directions](../../thesis-paper/evidence/minpi_compression/legacy-gain/README.md) also specify the GPU setup. For offline checking the current release carries the measured outcomes, test targets and 60 original forecast files used by the historical comparison.

The project checkout uses `.\thesis-python.ps1`, which expects its shared `venv` in the parent directory. In a standalone extraction with another interpreter, replace this wrapper in the experiment commands with `python` after setting `PYTHONPATH` above. Extract `thesis-paper/minpi-source.zip` in a separate empty directory and run `latexmk -pdf main.tex` there to rebuild the paper. The Lean proofs and Wolfram Language check are in `thesis-paper/formal/`; the Lean toolchain version is pinned alongside them.

## Timing environment

Windows 11 (build 26200) on an Intel Core Ultra 9 275HX, 24 logical CPUs. The shared 64-bit Anaconda Python 3.13.9 environment contains NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, pandas 3.0.3, threadpoolctl 3.6.0 and XGBoost 3.3.0. The NumPy and SciPy libraries report pthreads OpenBLAS 0.3.31.188.0 and 0.3.31.dev respectively (Haswell dispatch). Ridge timings used one logical CPU and one BLAS thread; the primary, chain and direct-fit diagnostic runners record or specify the affinity separately. The original Gain and boosted receivers used an NVIDIA GeForce RTX 5070 Laptop GPU, driver 591.91, 8,151 MiB reported by `nvidia-smi`, and the XGBoost CUDA 12.9 build. The ridge study rotated methods over three repetitions; the matched nonlinear width-guard transfer used one complete timing per method and split. These clocks are conditional on this machine and implementation.
