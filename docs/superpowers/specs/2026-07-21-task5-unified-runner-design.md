# Task 5 Unified ICOMP 2026 Runner Design

Date: 2026-07-21

## Goal

Create one authoritative experiment runner for every new ICOMP 2026 paper
number.  The runner must make the protocol reproducible, resumable, auditable,
and complete before the one-GPU formal rerun starts.

The runner is an orchestration and protocol layer.  It reuses tested project
computation functions, but it does not invoke the legacy experiment scripts as
subprocesses and does not silently combine historical result directories.

## Global constraints

- The submission deadline is 2026-08-05.
- Execution uses one continuously available GPU; GPU jobs run sequentially.
- No formal experiment is run during Task 5 implementation.
- The real ShapG baseline and canonical `mi_correlation_guided` Algorithm B
  remain distinct.
- Split-first preprocessing, the unified XGBoost probability path, explicit
  utility selection, and corrected hypervolume are mandatory.
- Legacy runners remain for historical reference but cannot generate new paper
  numbers.
- No Git commit is made unless the user later requests one.

## 1. Component boundaries

### `experiments/run_icomp2026_pipeline.py`

This is the only public CLI for the rerun.  It parses arguments, loads the
resolved protocol, creates or resumes a run, configures logging, dispatches
stages, and returns a nonzero exit code when a stage has failed jobs.

It contains no estimator, model-training, feature-selection, statistics, or
plotting implementation.

### `src/minppi_mofs/experiment_protocol.py`

This module owns protocol mechanics that can be tested without datasets or a
GPU:

- configuration loading, merging, validation, hashing, and freezing;
- run identity and source fingerprinting;
- expected-job manifest generation;
- atomic result writes;
- per-stage and aggregate status transitions;
- resume decisions;
- audited access to test partitions;
- completeness checks before summarization.

### `experiments/icomp2026_stages.py`

This module implements the stage adapters.  Each adapter consumes a validated
stage context and calls tested functions from `src/minppi_mofs` or focused
helper functions extracted from legacy experiment scripts.  Adapters write the
canonical result schema through the protocol layer; they never create their own
timestamped result roots.

The first implementation remains in one stage module to meet the deadline.
It may be split later only if the module becomes difficult to review.

### `tests/test_experiment_protocol.py`

This file tests the protocol layer with temporary directories and stub stage
functions.  It must not train XGBoost or require a GPU.

## 2. CLI and run identity

The CLI is:

```text
--stage {validate,preprocess,estimator,removal,alpha,nsga2,ablation,summarize,figures,all}
--config-dir configs
--output-root results/icomp2026_rerun
--seeds 42 123 456 789 1024
--backend {cpu,cuda}
--run-id RUN_ID
--resume
--smoke
--stop-after-jobs N
```

For a new run, `--run-id` is optional.  The runner generates
`YYYYMMDDTHHMMSSZ-<protocol-hash-prefix>` when it is omitted.  For `--resume`,
`--run-id` is mandatory, so the runner never guesses which directory to reuse.
Run IDs must contain only letters, digits, `.`, `_`, and `-`.

`--stop-after-jobs` is a smoke-only recovery diagnostic.  It exits with code
75 after atomically completing exactly `N` jobs, leaving remaining jobs
pending.  Formal runs reject this option.

Creating a new run fails if its directory already exists.  Resuming fails if
the run directory does not exist or its frozen protocol/source fingerprint no
longer matches the current invocation.

The source fingerprint records:

- Git commit when available;
- whether the working tree is dirty;
- SHA-256 hashes of relevant `src/`, `experiments/`, and resolved config files;
- Python, package, XGBoost, CUDA, driver, and operating-system information.

A dirty tree is permitted because Tasks 3--5 are intentionally uncommitted,
but changing source or configuration after a run starts prevents resume.

## 3. Configuration model

`configs/icomp2026_pipeline.yaml` is the global protocol.  It contains:

- protocol version and active dataset IDs;
- default seeds and backend;
- allowed task/utility combinations;
- model parameters;
- MinPPI methods, alpha values, and estimator budgets;
- removal controls;
- NSGA-II population, evaluation budget, and fixed reference point;
- ablation definitions;
- smoke overrides.

Existing `configs/<dataset>.yaml` files become dataset descriptions.  Each
contains the canonical dataset ID, data paths, target, task, utility, split
parameters, feature-filter limit, and any justified dataset-specific budget.

The runner merges global and dataset configuration into a fully resolved
configuration.  Unknown keys, missing required keys, duplicate dataset IDs,
invalid tasks/utilities, invalid alpha values, seed mismatches, nonpositive
budgets, and an unfixed hypervolume reference point are validation errors.

The resolved configuration is copied to `configs_frozen/` before a computational
stage starts.  Later stages read the frozen copy, not the live `configs/`
directory.

## 4. Run manifest and job identity

Before execution, the runner expands the frozen protocol into every expected
`dataset × seed × method × stage-specific parameter` job.  Every job receives
a deterministic SHA-256-derived `job_id` and a manifest row containing:

- run ID and protocol hash;
- stage, dataset, seed, method, task, and utility;
- alpha or ablation value when applicable;
- budget identifier;
- expected result path.

The manifest, rather than filesystem globbing, is the authority for deciding
which combinations should exist.  `summarize` fails if any expected upstream
job is absent, failed, running, skipped without an approved reason, duplicated,
or associated with a different protocol hash.

## 5. Status transitions, atomic writes, and resume

Each stage writes `<stage>/status.json`; the root `status.json` is an aggregate
view.  A job may be `pending`, `running`, `success`, `failed`, or `skipped`.
Every transition records UTC time, attempt number, output path, and error
metadata when applicable.

Result data is written to a temporary file in the destination directory,
flushed, and installed with `os.replace`.  Status changes to `success` only
after the final result can be reopened and validated against the canonical
schema.

Resume behavior is strict:

- `success`: validate the existing output and skip execution;
- `failed`: increment the attempt and execute again;
- stale `running`: record interrupted recovery, increment the attempt, execute
  again;
- `pending`: execute;
- approved `skipped`: keep skipped only when the frozen protocol explicitly
  excludes the combination.

An invalid or missing successful output is converted to `failed` and rerun.
Configuration or source-fingerprint mismatch aborts the entire resume before
any job runs.

A failed job does not prevent independent jobs from running.  The stage records
all failures and returns a nonzero exit code after completing the remaining
jobs.  `--resume` is then used to retry only unsuccessful jobs.

## 6. Test-partition access audit

Stage adapters do not receive test arrays directly.  They receive train/validation
data plus an audited data-access object whose `load_test(reason=...)` method is
the only supported way to expose the test partition.

Allowed access is limited to:

- final removal evaluation after the harmful set and all controls are fixed;
- final evaluation of every predeclared alpha-grid point after all corresponding
  feature sets are fixed, without using those test values to choose alpha;
- final NSGA-II front evaluation after search and model selection finish.

`validate`, `preprocess`, `estimator`, and `ablation` cannot load test data for
selection.  The alpha stage receives test access only for its frozen grid's
final reporting pass.  `summarize` and `figures` consume saved final metrics
and do not reopen raw test partitions.

Every allowed or denied request appends an event to `protocol_audit.json` with
stage, dataset, seed, job ID, UTC time, reason, and decision.  A denied request
raises immediately and fails the job.  This is a protocol guard for project
code, not a security sandbox against arbitrary direct filesystem access; stage
code is reviewed and tested to use only the audited access object.

## 7. Stage responsibilities

### `validate`

Validate configuration, data availability, environment versions, CUDA support,
method combinations, output permissions, and source fingerprint.  It performs
no model fitting.

### `preprocess`

Run the split-first data protocol independently for every dataset and seed.
Write prepared partitions and `data_manifest.csv`, including row counts,
feature counts, label summaries, split indices/hashes, and fitted-transform
metadata.  Supervised selection and scaling fit on training data only.

### `estimator`

Compare exact MinPPI where feasible with Algorithm A and
`mi_correlation_guided` Algorithm B under declared evaluation budgets.  Use
train/validation only and record utility evaluations, runtime, detection
metrics, and estimator seeds.

### `removal`

Fix MinPPI harmful sets and matched controls using train/validation, then allow
one audited final test evaluation per fixed method/combination.  Record absolute
utility change, retained feature count, runtime, and raw per-seed results.

### `alpha`

Evaluate the predeclared alpha grid without selecting alpha on test data.
Alpha-dependent feature sets are fixed using train/validation; any reported test
evaluation follows the same one-shot final-evaluation rule as removal and is
not used to choose the paper's alpha.

### `nsga2`

Compare full and MinPPI-prefiltered search with identical population and total
evaluation budgets, corrected hypervolume, and one frozen reference point.
Search uses train/validation; final nondominated fronts receive audited test
evaluation only after search terminates.

### `ablation`

Run only the frozen utility, guidance, budget, and operator ablations.  Ablation
selection uses no test data.

### `summarize`

Verify manifest completeness first.  Then report raw five-seed values, mean,
standard deviation, median, 95% bootstrap interval, paired tests with Holm
correction where appropriate, and effect sizes.  Produce tables and
`latex_macros.tex` from the same canonical result records.

### `figures`

Read summary tables only and generate the paper figures.  It does not train
models or recompute metrics.

`all` executes these stages in dependency order and stops before a dependent
stage when its required upstream stage is incomplete.

## 8. Result layout

```text
results/icomp2026_rerun/<run_id>/
├── environment.json
├── source_fingerprint.json
├── resolved_protocol.yaml
├── configs_frozen/
├── run_manifest.csv
├── data_manifest.csv
├── protocol_audit.json
├── raw_metrics/
├── estimator/
├── removal/
├── alpha/
├── nsga2/
├── ablation/
├── tables/
├── figures/
├── latex_macros.tex
├── status.json
└── output.log
```

Smoke and formal runs use separate run IDs.  `resolved_protocol.yaml` records
`run_kind: smoke|formal`, and formal summarization refuses to combine smoke
records.

## 9. Smoke run

The Task 5 runtime acceptance test uses seed `42`, Spambase, and Abalone.  This
covers classification, regression, Algorithm B, exact small-feature behavior,
and the common XGBoost path while remaining fast.

Smoke overrides reduce estimator and NSGA-II budgets but do not change data
splits, utility definitions, model family, output schema, audit rules, or
reference-point semantics.

The smoke run must demonstrate:

- CPU validation and CUDA detection;
- canonical directory and metadata creation;
- deterministic manifest regeneration;
- valid atomic result files;
- an interrupted/failed job that succeeds on resume without rerunning completed
  jobs;
- allowed and denied test-access audit events;
- completeness failure when a manifest job is deliberately absent;
- successful summary only after all required jobs are restored.

Formal five-seed execution belongs to Task 6 and starts only after these checks
pass.

## 10. Tests and acceptance criteria

Unit and lightweight integration tests cover:

- CLI parsing and run-ID validation;
- global/dataset config merge and rejection of invalid protocols;
- deterministic hashes and manifests;
- atomic writes and schema validation;
- every status/resume branch;
- refusal to resume after source/config drift;
- allowed and denied test access with audit records;
- completeness detection for absent, duplicate, failed, and mismatched jobs;
- smoke overrides and separation from formal results;
- stage dependency ordering with stub executors;
- package, Ruff, compile, and existing full-suite regression checks.

Task 5 is complete only when all tests pass, the two-dataset CUDA smoke run
passes, resume is demonstrated, no test-access violation is unresolved, and no
formal experiment combination has been launched.

## Non-goals

- Do not rewrite MinPPI, NSGA-II, ShapG, or utility mathematics.
- Do not obtain new paper results during runner development.
- Do not automatically choose alpha, utility, model, reference point, or a
  favorable subset of seeds from observed test performance.
- Do not merge historical results into the formal rerun.
- Do not introduce distributed execution or concurrent GPU jobs.
