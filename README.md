# MinPPI-MOFS — Min-Marginal Power Indices for Multi-Objective Feature Selection

Registered, reproducible experiments for the ICOMP 2026 submission.

This project implements **Min-Marginal Power Indices (MinPPI)**: a minimum-aggregation feature
attribution specialized to a binary utility-threshold voting game. For a fixed utility quota
$\alpha$, MinPPI labels each feature as harmful, neutral, or decisive at the worst-case level,
certified by an evaluated threshold-crossing coalition without a second score cutoff. Harmful
candidates pass through a validation gate before feature removal or multi-objective (NSGA-II)
prefiltering.

## Current Status

- A five-seed campaign is registered and sealed across six leakage-screened benchmark datasets:
  Spambase, Madelon, ISOLET, Gisette, Credit-Card-Fraud, Abalone (seeds `42, 123, 456, 789, 1024`).
- Communities & Crime was excluded after a post-run integrity audit (its target is reconstructible
  from retained outcome-component columns); it is kept only as a quarantined audit snapshot.
- Paper statistics are re-aggregated from canonical row-level records into a six-dataset
  publication view, hash-bound to the historical seven-dataset record.
- `conference-paper/figures/` accepts only manifest- and hash-verified formal figures.

## Repository Layout

```text
Master/
├── configs/                  formal frozen protocol (YAML)
├── data/raw/                 six formal CSVs + quarantined audit snapshot (git-ignored)
├── experiments/              pipeline runners, calibration, summarization, figures
├── src/minppi_mofs/          core algorithm library
├── tests/                    automated tests
├── conference-paper/         paper source and figure sync
├── demos/                    informal teaching demos
├── thesis/                   master's-thesis drafts (auxiliary)
└── results/                  resumable experiment outputs (git-ignored)
```

## Requirements

- Python 3.12+ (3.13 recommended).
- A CUDA-capable NVIDIA GPU, driver, and CUDA toolkit, with a CUDA-enabled XGBoost wheel —
  **required for the formal campaign** (Group 1 stages and the campaign run with `--backend cuda`).
- A LaTeX distribution providing `pdflatex` and `bibtex` — required for the Group 2 paper build.
- `git` — required for the formal provenance gate (the run verifies a clean working tree).
- Network access to download the six raw datasets during Group 1 acquisition (for example, the
  Credit-Card-Fraud raw snapshot is ~149 MB).
- Sufficient disk space for the raw snapshots and the resumable `results/` outputs.
- A clean git working tree before the formal run.

## Environment Setup

From a fresh clone, create the project-adjacent virtual environment first, then install and
verify. On Windows the interpreter is `venv\Scripts\python.exe`; on Linux/macOS it is
`venv/bin/python`.

```powershell
# one-time setup, run from the Master directory
python -m venv ../venv
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pip install --upgrade pip
& $PY -m pip install --no-build-isolation -r requirements.txt
& $PY -m pip check
& $PY -c "import minppi_mofs; print('import OK')"
```

## One-Click Execution

Prerequisites: complete the Environment Setup above (create and install the `../venv`), ensure a
CUDA environment is available, and keep the git working tree clean. All commands run from the
`Master` directory in PowerShell with the project venv. Only one campaign process may run at a
time; do not launch other GPU/model jobs concurrently.

### Group 1 — Acquisition, Calibration, Sealed Estimators, Screening

Runs environment acceptance, raw-data acquisition/verification, bootstrap, calibration, smoke,
the sealed estimator campaign, and validation-only screening. It fails fast if any sub-command
fails. The git working tree must be clean (`git status --short` empty) before the formal run;
re-run the same block after an interruption to resume.

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY experiments\run_icomp2026_group1.py --bootstrap-id bootstrap-s42-v2 --smoke-id smoke-s42-v2 --campaign-id icomp2026-formal-v2
if ($LASTEXITCODE -ne 0) { throw "Group 1 failed with exit $LASTEXITCODE" }
```

### Group 2 — Held-out Tests, Summarization, Formal Figures, Paper Build

Only run after confirming the Group 1 validation direction. Opens the registered held-out-test
access, completes ablation and the single campaign summary, regenerates the three formal figures,
and builds the paper PDF.

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
$OUT = "results\icomp2026_formal"
$CID = "icomp2026-formal-v2"
$POLICY = "$OUT\bootstrap-s42-v2\calibration\model_policy.json"
$FIGURES = "$OUT\$CID\figures\paper"

function Invoke-ProjectPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Write-Host "`n> $PY $($Arguments -join ' ')" -ForegroundColor Cyan
    & $PY @Arguments
    if ($LASTEXITCODE -ne 0) { throw "project Python failed with exit $LASTEXITCODE" }
}

Invoke-ProjectPython -Arguments @(
    "experiments\run_icomp2026_campaign.py", "--config-dir", "configs",
    "--output-root", $OUT, "--campaign-id", $CID,
    "--model-policy", $POLICY, "--through", "summarize",
    "--max-parallel", "3", "--allow-test-access"
)
Invoke-ProjectPython -Arguments @(
    "conference-paper\sync_figures.py", "--source", $FIGURES, "--check"
)
Invoke-ProjectPython -Arguments @(
    "conference-paper\sync_figures.py", "--source", $FIGURES,
    "--dest", "conference-paper\figures"
)

Push-Location conference-paper\paper
try {
    & pdflatex -halt-on-error icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "first pdflatex pass failed" }
    & bibtex icomp2026_conference
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
    & pdflatex -halt-on-error icomp2026_conference.tex
    & pdflatex -halt-on-error icomp2026_conference.tex
} finally {
    Pop-Location
}
```

Re-run the same block after an interruption; the campaign resumes unfinished jobs and skips
already-verified ones.

## Experiment Scripts and Recommended Order

All experiment scripts live under `experiments/`. The manifest pipeline is the only formal
execution path; run scripts directly only when the README instructs it.

### Recommended running order (two groups)

| Step | Script | Purpose |
|---|---|---|
| **Group 1** | `run_icomp2026_group1.py` | One-click entry: environment acceptance → data acquisition → bootstrap → calibration → smoke → sealed estimators → validation-only screening. Runs the `preprocess_*.py` and `calibrate_icomp2026_models.py` internally. |
| **Group 2** | `run_icomp2026_campaign.py` | Opens held-out-test access, runs removal/alpha/NSGA-II/ablation, and produces the single campaign summary (`--through summarize`). |
| | `make_paper_figures.py` | Generates the three formal figures from the canonical summary tables. |
| | `conference-paper/sync_figures.py` | Validates and copies the formal figure bundle into the paper. |
| | `pdflatex` + `bibtex` | Builds the paper PDF. |

### All scripts and their functions

**Formal entries (recommended to run, in order):**
- `run_icomp2026_group1.py` — complete Group 1 workflow (data → calibration → smoke → sealed estimators → screening).
- `run_icomp2026_campaign.py` — run the frozen protocol as three seed shards and summarize (Group 2).
- `make_paper_figures.py` — read-only deterministic paper figures from canonical tables.

**Supporting / monitoring:**
- `run_icomp2026_pipeline.py` — authoritative CLI controller for a single pipeline run (used by the entries; resumable with `--resume`).
- `monitor_icomp2026_campaign.py` — read-only campaign status snapshots (`--group initial` / `--group downstream`).
- `calibrate_icomp2026_models.py` — calibrate the frozen per-dataset XGBoost round policy on validation only.
- `screen_icomp2026_results.py` — provisional validation-only screening from a seed-42 run.
- `summarize_multifidelity_campaign.py` — validate shard provenance and build the campaign summary.

**Core library (imported, not run directly):**
- `icomp2026_stages.py` — manifest-authoritative stage execution engine.
- `icomp2026_estimator_bundle.py` — estimator bundle identities and schemas.
- `icomp2026_progress.py` — strict progress state and run logging.
- `native_process_guard.py` — authenticated completion receipts for worker processes.
- `summarize_icomp2026.py` — pure statistics and publication tables.

**Data acquisition (`experiments/preprocess_data/`):**
- `preprocess_abalone.py`, `preprocess_Spambase.py`, `preprocess_Madelon.py`, `preprocess_Gisette.py`, `preprocess_ISOLET.py`, `preprocess_Credit-Card-Fraud.py` — download and verify each formal raw CSV (SHA-256 fail-closed).
- `preprocess_crimedata.py` — quarantined Communities & Crime audit snapshot only (not formal).
- `raw_snapshot.py` — fail-closed raw snapshot helpers (imported).

**Diagnostics / auxiliary (optional, not part of the formal run):**
- `benchmark_icomp2026_backends.py` — validation-only throughput benchmark.
- `validate_grouped_acceleration.py` — training/validation-only acceptance gates for grouped acceleration.
- `forecast_icomp2026_48h.py` — measured runtime forecast ledger.
- `run_isolet_utility_pair.py` — ISOLET seed-42 paired utility ablation.
- `build_leakage_screened_publication.py` — historical compatibility; rebuilds the six-dataset view from the seven-dataset record (not needed for a new six-dataset campaign).

## Frozen Protocol (summary)

- **Datasets:** Spambase, Madelon, ISOLET, Gisette, Credit-Card-Fraud, Abalone.
- **Seeds:** `42, 123, 456, 789, 1024`.
- **Split:** train/validation/test = 0.64/0.16/0.20 (stratified for classification).
- **Preprocessing:** variance filter, MI selection (≤100 features), standardization — training-only fit.
- **Model:** XGBoost probabilities; utilities ROC-AUC / OVR-AUC / R²; default quota $\alpha=0.95$.
- **Backend:** CPU is the portable default; the formal campaign entry uses CUDA.
- Protocol changes require editing `configs/` and starting a new run; never modify an existing run.

## Authority Boundaries

| Content | Source of truth |
|---|---|
| Data / split / utility | `configs/<dataset>.yaml` |
| Seeds / budget / alpha / ablations / NSGA-II | `configs/icomp2026_pipeline.yaml` |
| Execution and resume | `experiments/run_icomp2026_*.py` |
| Formal completion | campaign manifest, audit, canonical metrics, summary manifest |
| Paper evidence | six-dataset `tables/publication/` |
| Paper figures | `figures/paper/` (manifest- and hash-verified) |

`demos/`, `thesis/`, and `configs_credit_probe/` are auxiliary research material and are not part
of the formal ICOMP evidence chain.

## Tests and Lint

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
& $PY -m ruff check src experiments tests conference-paper/sync_figures.py
```

## Citation

```bibtex
@inproceedings{minppi2026,
  title     = {Min-Marginal Power Indices for Multi-Objective Feature Selection},
  author    = {Anonymous},
  booktitle = {ICOMP},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```
