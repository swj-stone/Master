# MinPI — Min-Marginal Power Indices for Multi-Objective Feature Selection

Research code and experimental materials accompanying the manuscript by **Wenjie Shi and
Chi Zhao**, Saint Petersburg State University.

A mean marginal contribution describes whether a feature helps on average across coalitions. **MinPI asks
whether adding that feature can turn an acceptable model into an unacceptable one.** It
applies minimum marginal aggregation to a binary utility-threshold game. An observed
harmful label comes with an evaluated coalition that witnesses the threshold crossing.
These candidates support validation-gated feature deletion and prefiltering for
multi-objective feature selection.

The manuscript uses **MinPI** for the Min-Marginal Power Index. The distribution name
`minppi-mofs`, Python package `minppi_mofs`, configuration keys and `icomp2026_*` script
names are retained as implementation identifiers.

## Article and poster

This README follows the current final manuscript, including its six-dataset analysis.
The article's existing filename is retained; the ASCOMP 2026 poster is a separate artifact.

| Artifact | PDF | Editable LaTeX |
|---|---|---|
| Final manuscript | [Article](conference-paper/paper/icomp2026_conference.pdf) | [Article source](conference-paper/paper/icomp2026_conference.tex) |
| ASCOMP 2026 poster, 120 × 90 cm | [Poster](conference-paper/poster/ASCOMP_MinPI_Poster_Wenjie_SHI.pdf) | [Poster source](conference-paper/poster/latex/main.tex) |

The poster has a three-column layout, standard LaTeX mathematics and two upper-right QR
codes. [Print instructions](conference-paper/poster/PRINT_README.txt) accompany the source.
Historical articles, poster revisions and submission packages are kept separately in
`conference-paper/archive/` and `packages/`; compiler output and QA files live in `tmp/`.

## Method

Let $N$ be the analyzed feature set, $n=|N|$, and $U_{\mathrm{val}}(S)$ the validation
utility of a model fitted on training data using features $S$. With
$U_{\mathrm{val}}(N)>0$ and quota $\alpha\in(0,1]$, define

$$
v_\alpha(S)=\mathbf{1}\!\left[U_{\mathrm{val}}(S)\geq\alpha U_{\mathrm{val}}(N)\right],
\qquad
\Delta_i(S)=v_\alpha(S\cup\{i\})-v_\alpha(S).
$$

The exact minimum Shapley–Shubik and minimum Banzhaf analogues coincide because both
range over every coalition excluding $i$:

$$
\operatorname{MinPI}_i=\operatorname{MinSS}_i=\operatorname{MinBz}_i
=\min_{S\subseteq N\setminus\{i\}}\Delta_i(S)\in\{-1,0,+1\}.
$$

- **Harmful (−1):** at least one coalition changes from at or above the quota to below it when
  the feature is added. That evaluated coalition is the harmful witness.
- **Neutral (0):** all marginals are nonnegative and at least one is zero.
- **Decisive (+1):** every coalition changes from below the quota to at or above it.

These exact classes are relative to the model, data split, utility and quota. Finite-budget
estimators return the minimum observed marginal: a sampled −1 is witness-certified;
sampled 0 and +1 are estimates of the exact class.

### Estimation and selection

1. **Construct the game.** Fit preprocessing and coalition models on training data;
   evaluate the game on validation data. The empty coalition predicts training class
   priors for classification or the training response mean for regression.
2. **Estimate minimum marginals.** Algorithm A samples a coalition size uniformly, then
   a subset uniformly at that size. Algorithm B constructs a seed from a maximum-weight
   absolute-correlation tree and training-only mutual information, then independently
   drops seed members with probability 0.3 and adds up to five available features.
   Each trial restarts from the seed; both estimators stop when a −1 witness is found.
3. **Dispatch by analyzed dimension.** The removal experiments use exact enumeration for
   $n\leq20$, B for $20<n\leq75$ (A if correlation guidance is unavailable), and the
   featurewise minimum of A and B for $n>75$.
4. **Validate joint removal.** Collect $H=\{i:\widehat{\operatorname{MinPI}}_i=-1\}$.
   Remove all of $H$ when
   $U_{\mathrm{val}}(N\setminus H)\geq U_{\mathrm{val}}(N)-\varepsilon$; otherwise
   remove none. Use $\varepsilon=0.005$ for classification and $0.01$ for regression.
5. **Evaluate removal and search.** Compare retained subsets with matched-count controls
   on held-out data. The optional NSGA-II experiment compares full-space and
   MinPI-prefiltered search, selecting fronts on validation and evaluating them on test.

NSGA-II minimizes feature fraction and clipped validation utility loss:

$$
\left(\frac{|S|}{n},\;1-\operatorname{clip}(U_{\mathrm{val}}(S),0,1)\right).
$$

Both variants use the same analyzed dimension $n$ before prefiltering, population 40,
800 evaluations and hypervolume reference point $(1.1,1.1)$.

## Experimental design reported in the article

| Component | Setting |
|---|---|
| Benchmarks | Spambase, Madelon, ISOLET, Gisette, Credit-Card-Fraud, Abalone |
| Seeds | 42, 123, 456, 789, 1024 |
| Split | 64% training / 16% validation / 20% test; stratified for classification |
| Preprocessing | Training-only variance filtering, MI selection to at most 100 features, standardization |
| Learner | XGBoost; 100 boosting rounds, depth 4, learning rate 0.1, one CPU worker per fit, CUDA tree backend |
| Utility | ROC AUC for binary tasks; one-vs-rest AUC for ISOLET; $R^2$ for Abalone |
| Quota | Default $\alpha=0.95$; sweep 0.85, 0.90, 0.95, 0.99 at seed 42 |
| Sampling budget | 2, 5, 10, 25 draws per feature per estimator; ISOLET uses 2, 5, 10; removal uses the largest budget |
| Controls | Full model, SHAP-bottom, MinShap-bottom, Banzhaf-bottom, random-matched removal |
| Random baseline | 30 matched-count trials averaged within each seed |

The 100-round model setting was selected by training/validation-only calibration from
10, 20, 40 and 100 rounds. SHAP-bottom ranks mean absolute training TreeSHAP values;
MinShap-bottom and Banzhaf-bottom rank the minimum and mean continuous utility marginal
from shared sampled coalitions. All deletion controls use the accepted MinPI removal count.

The formal exact-reference estimator comparison is capped at 16 features and therefore
uses eight-feature Abalone. This benchmark cap is distinct from the dispatcher's
$n\leq20$ exact branch. Guidance/dispatcher ablations use seed 42 on Madelon, ISOLET,
Credit-Card-Fraud and Abalone; the utility ablation uses ISOLET at seed 42.

The original campaign included seven datasets. Communities & Crime was excluded from
the final analysis after outcome-component leakage was identified, and all reported
aggregates were recomputed for the six retained datasets. Current dataset configurations
list those six benchmarks; historical campaign records remain available for provenance.

## Main results

At $\alpha=0.95$, across 30 dataset–seed units, MinPI removed **12.53% of analyzed
features** with a mean held-out utility change of **−0.000313**. The table reports
five-seed means ± sample standard deviations; compression is relative to the analyzed
features after preprocessing. Here $\Delta U=U_{\mathrm{test}}(S_{\mathrm{ret}})-U_{\mathrm{test}}(N)$.

| Dataset | Removed features (%) | Held-out $\Delta U$ |
|---|---:|---:|
| Madelon | 43.60 ± 31.96 | +0.00469 ± 0.00497 |
| Abalone | 27.50 ± 25.62 | −0.00565 ± 0.01077 |
| Credit-Card-Fraud | 2.76 ± 2.89 | −0.00078 ± 0.00335 |
| Spambase | 0.72 ± 0.99 | −0.00003 ± 0.00007 |
| Gisette | 0.60 ± 0.89 | −0.00011 ± 0.00016 |
| ISOLET | 0.00 ± 0.00 | 0.00000 ± 0.00000 |

- **Matched removal:** Madelon achieved $\Delta U=+0.0047$ versus −0.0732 for random
  deletion of the same size; Abalone achieved −0.0056 versus −0.0370.
- **Prefiltered search:** the dataset-averaged held-out hypervolume change was +0.01584,
  with positive dataset-averaged differences in all five seeds (Holm-adjusted $p=0.4375$).
  Madelon contributed the largest change, +0.09515; outcomes varied across datasets.
- **Estimator comparison:** on Abalone at budget 25, mean harmful-set $F_1$ was 0.443
  for A and 0.313 for B. A exceeded B at every tested finite budget; the hybrid achieved
  $F_1=1$ at each budget by selecting exact enumeration.

The clearest gains occur where many features can be removed, especially Madelon.
Exact enumeration is the article's preferred option when feasible; beyond that, its
estimator analysis favors size-uniform A as the default. The hybrid routing above is
the fixed policy used in the reported removal experiments.

## Repository layout

```text
Master/
├── configs/                  dataset and experiment settings
├── data/                     raw datasets and local data artifacts
├── experiments/              runners, calibration, statistics and figure generation
├── src/minppi_mofs/           core implementation (legacy package name)
├── tests/                    implementation tests
├── conference-paper/
│   ├── paper/                final article PDF, LaTeX and bibliography
│   ├── figures/              article figures
│   ├── poster/               final poster, LaTeX, assets and print instructions
│   ├── presentation/         presentation materials
│   └── archive/              previous article and poster versions
├── results/                  experiment records and resumable outputs
├── packages/                 historical submission and review bundles
├── docs/                     design and revision history
├── tmp/                      compiler output, visual checks and scratch files
└── demos/                    teaching examples
```

## Requirements

- Python 3.12+ (3.13 recommended).
- A CUDA-capable NVIDIA GPU and compatible driver, with a CUDA-enabled XGBoost wheel —
  **required for the formal campaign** (Group 1 stages and the campaign run with `--backend cuda`).
- Article build: `latexmk`, `pdflatex` and `bibtex`. Poster build: XeLaTeX, Windows Arial
  and the included Latin Modern Math font. These are separate from experiment execution.
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

## Experiment execution

Prerequisites: complete the Environment Setup above (create and install the `../venv`), ensure a
CUDA environment is available, and keep the git working tree clean. All commands run from the
`Master` directory in PowerShell with the project venv. Only one campaign process may run at a
time; do not launch other GPU/model jobs concurrently. The commands below run the current
six-dataset configuration. Use a new set of run IDs for a new protocol or source revision;
reuse IDs only to resume their matching run. The article's reported results are the
historical six-dataset publication analysis described above.

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

### Group 2 — Held-out Tests, Summarization and Verified Figures

Only run after confirming the Group 1 validation direction. Opens the registered held-out-test
access, completes ablation and the single campaign summary, regenerates the three formal figures,
and exports a verified figure bundle. Article compilation is a separate step below.

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
    "--dest", "tmp\verified-paper-figures\$CID"
)
```

Re-run the same block after an interruption; the campaign resumes unfinished jobs and skips
already-verified ones.

The exported figures stay under `tmp/` for review. Updating the final manuscript's
`conference-paper/figures/` is an explicit publication update, separate from running
another experiment.

## Compile the article or poster

These commands use the local LaTeX sources and keep the resulting PDFs and compiler
intermediates under `tmp/`. They leave the final PDFs linked above unchanged.

```powershell
# Run from Master/
Push-Location conference-paper/paper
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../tmp/pdfs/article-final-build icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "Article build failed" }
} finally { Pop-Location }

New-Item -ItemType Directory -Force tmp/pdfs/ascomp-poster/latex-build | Out-Null
Push-Location conference-paper/poster/latex
try {
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/ascomp-poster/latex-build main.tex
    if ($LASTEXITCODE -ne 0) { throw "First poster pass failed" }
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/ascomp-poster/latex-build main.tex
    if ($LASTEXITCODE -ne 0) { throw "Second poster pass failed" }
} finally { Pop-Location }
```

Outputs: `tmp/pdfs/article-final-build/icomp2026_conference.pdf` and
`tmp/pdfs/ascomp-poster/latex-build/main.pdf`. The poster project includes its text,
formulas, background and math font; compiling it does not require rerunning experiments.

## Experiment Scripts and Recommended Order

All experiment scripts live under `experiments/`. The manifest pipeline is the only formal
execution path; run scripts directly only when the README instructs it.

### Recommended running order (two groups)

| Step | Script | Purpose |
|---|---|---|
| **Group 1** | `run_icomp2026_group1.py` | One-click entry: environment acceptance → data acquisition → bootstrap → calibration → smoke → sealed estimators → validation-only screening. Runs the `preprocess_*.py` and `calibrate_icomp2026_models.py` internally. |
| **Group 2** | `run_icomp2026_campaign.py` | Opens held-out-test access, runs removal/alpha/NSGA-II/ablation, and produces the single campaign summary (`--through summarize`). |
| | `make_paper_figures.py` | Generates the three formal figures from the canonical summary tables. |
| | `conference-paper/sync_figures.py` | Validates and copies the figure bundle to the specified destination. |
| Separate artifact build | `latexmk` / `xelatex` | Compiles the article / poster sources. |

### All scripts and their functions

**Formal entries (recommended to run, in order):**

- `run_icomp2026_group1.py` — complete Group 1 workflow (data → calibration → smoke → sealed estimators → screening).
- `run_icomp2026_campaign.py` — run the frozen protocol as three seed shards and summarize (Group 2).
- `make_paper_figures.py` — generates paper figures from canonical tables without changing experiment records.

**Supporting / monitoring:**

- `run_icomp2026_pipeline.py` — authoritative CLI controller for a single pipeline run (used by the entries; resumable with `--resume`).
- `monitor_icomp2026_campaign.py` — read-only campaign status snapshots (`--group initial` / `--group downstream`).
- `calibrate_icomp2026_models.py` — calibrate the frozen per-dataset XGBoost round policy on validation only.
- `screen_icomp2026_results.py` — provisional validation-only screening from a seed-42 run.
- `summarize_multifidelity_campaign.py` — validate shard provenance and build the campaign summary.

**Pipeline internals (imported, not run directly):**

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

## Authority Boundaries

| Content | Source of truth |
|---|---|
| Data / split / utility | `configs/<dataset>.yaml` |
| Seeds / budget / alpha / ablations / NSGA-II | `configs/icomp2026_pipeline.yaml` |
| Execution and resume | `experiments/run_icomp2026_*.py` |
| Formal completion | campaign manifest, audit, canonical metrics, summary manifest |
| Paper evidence | six-dataset `tables/publication/` |
| Paper figures | `figures/paper/` (manifest- and hash-verified) |
| Current manuscript definitions and reported results | `conference-paper/paper/icomp2026_conference.tex` |

The pipeline YAML has a portable CPU default; the formal campaign entry selects CUDA.
Dataset-specific estimator budgets override the generic budget list. Change configurations
under a new run ID when changing the protocol. `demos/` and archived packages are auxiliary
or historical material rather than the current article's result tables.

## Tests and Lint

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
& $PY -m ruff check src experiments tests conference-paper/sync_figures.py
```

## Citation

```bibtex
@misc{shi2026minpi,
  title     = {Min-Marginal Power Indices for Multi-Objective Feature Selection},
  author    = {Shi, Wenjie and Zhao, Chi},
  year      = {2026},
  note      = {Manuscript}
}
```
