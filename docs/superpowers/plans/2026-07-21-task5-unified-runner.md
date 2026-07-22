# Task 5 Unified ICOMP 2026 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single authoritative, resumable, audited experiment runner that must pass a two-dataset CUDA smoke run before the ICOMP 2026 formal rerun begins.

**Architecture:** Keep orchestration in `experiments/run_icomp2026_pipeline.py`, pure protocol mechanics in `src/minppi_mofs/experiment_protocol.py`, and computational adapters in `experiments/icomp2026_stages.py`.  Every expected job comes from a frozen manifest; stage adapters return canonical records through atomic writers and receive test data only through an audited access object.

**Tech Stack:** Python 3.13, dataclasses, pathlib, hashlib, JSON/CSV/YAML, pandas, NumPy, SciPy, XGBoost 3.3, NetworkX, pytest, Ruff, matplotlib

## Global Constraints

- Submission deadline: 2026-08-05.
- One continuously available GPU; run no concurrent GPU jobs.
- Preserve all uncommitted Tasks 1--4 changes.
- Do not run any five-seed formal experiment during Task 5.
- Do not use legacy runner outputs as new paper results.
- Use split-first preprocessing, explicit task/utility, corrected hypervolume, and canonical `mi_correlation_guided` naming.
- Test utility cannot influence alpha, estimator budget, graph settings, model hyperparameters, removal strategy, or NSGA-II search.
- Do not commit to Git unless the user later requests it.

---

### Task 1: Define and validate the resolved experiment configuration

**Files:**
- Create: `src/minppi_mofs/experiment_protocol.py`
- Create: `configs/icomp2026_pipeline.yaml`
- Modify: `configs/spambase.yaml`
- Modify: `configs/madelon.yaml`
- Modify: `configs/isolet.yaml`
- Modify: `configs/gisette.yaml`
- Modify: `configs/credit_card_fraud.yaml`
- Modify: `configs/crimedata.yaml`
- Modify: `configs/abalone.yaml`
- Create: `tests/test_experiment_protocol.py`

**Interfaces:**
- Produces: `DatasetSpec`, `ResolvedProtocol`, `load_resolved_protocol(config_dir, *, seeds, backend, smoke)`
- Produces: `canonical_hash(value) -> str`
- Consumes: existing YAML files and CLI seed/backend overrides

- [x] **Step 1: Write failing configuration tests**

Add tests that create a temporary global config plus two dataset configs and assert the exact merged fields:

```python
def test_load_resolved_protocol_merges_global_and_dataset_configs(tmp_path):
    write_protocol_fixture(tmp_path)
    protocol = load_resolved_protocol(
        tmp_path, seeds=(42,), backend="cpu", smoke=True
    )
    assert protocol.protocol_version == "1.0"
    assert protocol.seeds == (42,)
    assert protocol.backend == "cpu"
    assert protocol.run_kind == "smoke"
    assert tuple(d.dataset_id for d in protocol.datasets) == (
        "spambase", "abalone"
    )
    assert protocol.datasets[0].task == "binary"
    assert protocol.datasets[0].utility == "roc_auc"
    assert protocol.datasets[1].task == "regression"
    assert protocol.datasets[1].utility == "r2"
```

Also test unknown global keys, missing target, invalid task/utility pairs, duplicate IDs, alpha outside `(0, 1]`, a nonpositive budget, seed duplication, and a hypervolume reference point other than the frozen two-number list `[1.1, 1.1]`.

- [x] **Step 2: Run the tests and confirm the missing API failure**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_experiment_protocol.py -q
```

Expected: collection fails because `minppi_mofs.experiment_protocol` does not exist.

- [x] **Step 3: Add immutable protocol value objects and strict validation**

Implement these public shapes:

```python
@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    raw_path: Path
    target: str
    task: str
    utility: str
    max_features: int | None
    variance_threshold: float
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass(frozen=True)
class ResolvedProtocol:
    protocol_version: str
    run_kind: str
    datasets: tuple[DatasetSpec, ...]
    seeds: tuple[int, ...]
    backend: str
    model_params: dict[str, object]
    alpha_default: float
    alpha_grid: tuple[float, ...]
    estimator_budgets: tuple[int, ...]
    nsga2: dict[str, object]
    removal: dict[str, object]
    ablations: dict[str, object]
    smoke: dict[str, object]


def load_resolved_protocol(
    config_dir: Path,
    *,
    seeds: tuple[int, ...] | None,
    backend: str | None,
    smoke: bool,
) -> ResolvedProtocol:
    global_config = _read_yaml(config_dir / "icomp2026_pipeline.yaml")
    selected_ids = tuple(global_config["datasets"])
    datasets = tuple(
        _load_dataset_spec(config_dir / f"{dataset_id}.yaml")
        for dataset_id in selected_ids
    )
    return _build_resolved_protocol(
        global_config,
        datasets,
        seeds_override=seeds,
        backend_override=backend,
        smoke=smoke,
    )


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

Do not add Pydantic or another dependency.  Reject unknown keys explicitly so a misspelled budget cannot be ignored.

- [x] **Step 4: Normalize the eight YAML files**

The global file must contain these top-level keys and frozen formal defaults:

```yaml
protocol_version: "1.0"
datasets: [spambase, madelon, isolet, gisette, credit_card_fraud, crimedata, abalone]
seeds: [42, 123, 456, 789, 1024]
backend: cpu
model:
  n_estimators: 100
  max_depth: 4
  learning_rate: 0.1
  n_jobs: 1
alpha:
  default: 0.95
  grid: [0.85, 0.90, 0.95, 0.99]
estimator:
  methods: [exact, random_mc, mi_correlation_guided, hybrid]
  budgets: [10, 25, 50, 100, 200]
  exact_max_features: 16
removal:
  controls: [full, minppi, shap_bottom, minshap_bottom, banzhaf_bottom, random_matched]
  random_repeats: 30
nsga2:
  variants: [full, minppi_prefilter]
  population: 40
  evaluations: 800
  ref_point: [1.1, 1.1]
ablations:
  utility: [accuracy, ovr_auc]
  guidance: [random, mi, correlation, mi_correlation]
  dispatcher: [exact, random_mc, mi_correlation_guided, hybrid]
smoke:
  datasets: [spambase, abalone]
  seeds: [42]
  estimator_budgets: [2]
  random_repeats: 2
  nsga2_population: 4
  nsga2_evaluations: 8
```

Each dataset YAML must use the same compact schema:

```yaml
dataset:
  id: spambase
  display_name: Spambase
  raw_path: data/raw/spambase.csv
  target: is_spam
  task: binary
  utility: roc_auc
preprocessing:
  max_features: 100
  variance_threshold: 0.01
split:
  train_frac: 0.64
  val_frac: 0.16
  test_frac: 0.20
```

Use `ovr_auc` for ISOLET and `r2` for Crimedata/Abalone.  Preserve dataset-specific raw paths, targets, variance thresholds, and feature caps from the current files.

- [x] **Step 5: Run focused and regression checks**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_experiment_protocol.py -q
..\venv\Scripts\ruff.exe check src\minppi_mofs\experiment_protocol.py tests\test_experiment_protocol.py
```

Expected: all new configuration tests pass and Ruff reports no errors.

---

### Task 2: Create run identity, layout, source fingerprint, and frozen metadata

**Files:**
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Modify: `tests/test_experiment_protocol.py`

**Interfaces:**
- Consumes: `ResolvedProtocol`, project root, output root, optional run ID
- Produces: `RunLayout`, `create_new_run(output_root, run_id, protocol)`
- Produces: `open_resumed_run(output_root, run_id, protocol, *, project_root)`
- Produces: `capture_source_fingerprint(project_root, protocol)` and `freeze_run_metadata(layout, protocol, *, project_root)`

- [x] **Step 1: Write failing run identity tests**

Cover valid IDs, rejection of separators/whitespace, refusal to overwrite an
existing run, mandatory run ID for resume, and fingerprint drift:

```python
def test_resume_rejects_source_drift(tmp_path, protocol):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    source = project / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    layout = create_new_run(tmp_path / "results", "run-1")
    freeze_run_metadata(layout, protocol, project_root=project)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        open_resumed_run(
            tmp_path / "results", "run-1", protocol,
            project_root=project,
        )
```

- [x] **Step 2: Confirm RED**

Run the Task 2 tests and confirm the new symbols are missing.

- [x] **Step 3: Implement layout and metadata**

Use this public layout:

```python
@dataclass(frozen=True)
class RunLayout:
    root: Path
    configs_frozen: Path
    raw_metrics: Path
    tables: Path
    figures: Path
    audit_path: Path
    manifest_path: Path
    status_path: Path
    log_path: Path
```

Validate run IDs with `re.fullmatch(r"[A-Za-z0-9._-]+", run_id)`.  An omitted
new-run ID becomes `YYYYMMDDTHHMMSSZ-<first 10 protocol-hash chars>`.

Fingerprint `src/**/*.py`, `experiments/**/*.py`, and resolved YAML content.
Record Git commit and dirty state as metadata, but compare file hashes for
resume because an intentionally dirty tree is allowed.

- [x] **Step 4: Freeze metadata atomically**

Write `resolved_protocol.yaml`, `environment.json`,
`source_fingerprint.json`, and copied dataset configs before manifest creation.
The environment record must include Python, NumPy, pandas, scikit-learn,
XGBoost, CUDA build information, OS, timezone, and the requested backend.

- [x] **Step 5: Verify Task 2**

Run focused tests, Ruff, and `python -m py_compile` for the protocol module.

---

### Task 3: Build deterministic job manifests and canonical result records

**Files:**
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Modify: `tests/test_experiment_protocol.py`

**Interfaces:**
- Produces: `JobSpec`, `ResultRecord`, `build_job_manifest(protocol, stages, *, data_manifest=None)`
- Produces: `atomic_write_json(path, value)` and `read_result_record(path, expected_job)`
- Consumes: the frozen `ResolvedProtocol`

- [x] **Step 1: Write failing manifest tests**

Assert stable IDs under dictionary reordering, unique IDs, smoke-only datasets,
and exact stage dimensions:

```python
def test_job_id_is_deterministic(protocol):
    first = build_job_manifest(protocol, stages=("preprocess", "removal"))
    second = build_job_manifest(protocol, stages=("preprocess", "removal"))
    assert [job.job_id for job in first] == [job.job_id for job in second]
    assert len({job.job_id for job in first}) == len(first)
```

Add an atomic-write test that monkeypatches `os.replace` and verifies the
temporary file and destination share a parent.  Add schema rejection tests for
wrong job ID, protocol hash, stage, dataset, seed, method, or non-finite metric.

- [x] **Step 2: Confirm RED**

Run only the manifest/result tests; expect missing-symbol failures.

- [x] **Step 3: Implement canonical job and result types**

```python
@dataclass(frozen=True)
class JobSpec:
    job_id: str
    protocol_hash: str
    stage: str
    dataset_id: str
    seed: int
    method: str
    parameters: dict[str, object]
    depends_on: tuple[str, ...]
    output_relpath: str


@dataclass(frozen=True)
class ResultRecord:
    job_id: str
    protocol_hash: str
    stage: str
    dataset_id: str
    seed: int
    method: str
    metrics: dict[str, float | int | str | bool | None]
    artifacts: dict[str, str]
    started_at: str
    finished_at: str
```

Compute `job_id` from canonical JSON of every field except `output_relpath` and
the ID itself.  Write one JSON record per job under
`raw_metrics/<stage>/<job_id>.json`.

- [x] **Step 4: Expand stage-specific job matrices**

Use explicit expansion rules:

- validate: one run-level job with dataset ID `__run__`, seed `0`, and method
  `environment_validation`;
- preprocess: dataset × seed, method `prepare_split`;
- estimator: dataset × seed × declared estimator method × budget, excluding
  exact when selected feature count exceeds `exact_max_features`;
- removal: dataset × seed × `selection_plan|full|minppi|shap_bottom|`
  `minshap_bottom|banzhaf_bottom|random_matched`; every evaluation method
  depends on the corresponding `selection_plan` job;
- alpha: dataset × seed × alpha value plus one `selection_plan` job; every
  alpha evaluation depends on that plan;
- nsga2: dataset × seed × `full|minppi_prefilter`;
- ablation: only combinations declared under the frozen ablation axes;
- summarize/figures: one run-level job with dataset ID `__run__` and seed `0`
  after upstream completeness.

Create stage manifests only when their dependencies are complete.  The
preprocess manifest is frozen first; estimator/removal/alpha/NSGA-II manifests
are then derived from the frozen `data_manifest.csv` and appended atomically to
the root manifest.  Never mutate a stage manifest after its first job starts.

- [x] **Step 5: Verify Task 3**

Run all protocol tests and `git diff --check`.

---

### Task 4: Implement status transitions, strict resume, and completeness gates

**Files:**
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Modify: `tests/test_experiment_protocol.py`

**Interfaces:**
- Produces: `StatusStore`, `ResumeDecision`, `assert_stage_complete(layout, manifest, required_stages)`
- Consumes: manifest jobs and canonical result records

- [x] **Step 1: Write a parameterized status test matrix**

```python
@pytest.mark.parametrize(
    ("state", "has_valid_output", "expected"),
    [
        ("pending", False, "execute"),
        ("running", False, "execute"),
        ("failed", False, "execute"),
        ("success", True, "skip"),
        ("success", False, "execute"),
    ],
)
def test_resume_decision(state, has_valid_output, expected, status_fixture):
    assert status_fixture.decide(state, has_valid_output).action == expected
```

Also assert attempt increments, stale-running recovery metadata, aggregate root
status, approved skips only, and completeness failure for absent, failed,
running, duplicate, schema-invalid, or foreign-protocol results.

- [x] **Step 2: Confirm RED**

Run the status/completeness tests and observe missing API failures.

- [x] **Step 3: Implement `StatusStore`**

Store per-stage status at `<stage>/status.json` and rebuild the root aggregate
after each transition.  Permit only:

```text
pending -> running
running -> success | failed
failed -> running
running(stale) -> running(new attempt)
```

Every transition stores UTC timestamp, attempt, result path, exception class,
message, and traceback path.  A `success` transition must call
`read_result_record` first.

- [x] **Step 4: Implement strict completeness**

`assert_stage_complete(layout, manifest, required_stages)` must compare the
manifest to validated records, never to averages or glob counts.  Return a
structured report for success; raise `IncompleteRunError` listing every missing
or invalid job otherwise.

- [x] **Step 5: Verify Task 4**

Run all protocol tests twice to prove status tests are isolated from prior
runs.

---

### Task 5: Add audited split storage and test-partition access control

**Files:**
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Create: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `PreparedSplitPaths`, `AuditedSplitAccess`, `AuditLog`
- Consumes: split CSV paths produced by preprocessing
- Guarantees: stage code cannot obtain test arrays through the supported API
  without an allowed stage/reason pair

- [x] **Step 1: Write failing allowed/denied audit tests**

```python
def test_estimator_cannot_access_test(tmp_path, prepared_paths):
    access = AuditedSplitAccess(
        prepared_paths,
        stage="estimator",
        job_id="job-estimator",
        audit=AuditLog.from_manifest(tmp_path / "protocol_audit.json", manifest),
    )
    access.load_train_validation()
    with pytest.raises(TestAccessViolation):
        access.load_test(reason="selection")


def test_removal_final_evaluation_is_audited(tmp_path, prepared_paths):
    audit = AuditLog.from_manifest(tmp_path / "protocol_audit.json", manifest)
    access = AuditedSplitAccess(
        prepared_paths, stage="removal", job_id="job-removal", audit=audit
    )
    X_test, y_test = access.load_test(reason="final_removal_evaluation")
    assert len(X_test) == len(y_test)
    assert audit.read_events()[-1]["decision"] == "allowed"
```

Cover alpha reason `final_alpha_grid_evaluation` and NSGA-II reason
`final_nsga2_front_evaluation`.  All other stage/reason pairs must be denied and
recorded.

- [x] **Step 2: Confirm RED**

Run the two new tests and confirm the audit classes are missing.

- [x] **Step 3: Implement immutable split paths and audit log**

```python
@dataclass(frozen=True)
class PreparedSplitPaths:
    dataset_id: str
    seed: int
    target: str
    train_csv: Path
    val_csv: Path
    test_csv: Path


_ALLOWED_TEST_ACCESS = {
    ("removal", "final_removal_evaluation"),
    ("alpha", "final_alpha_grid_evaluation"),
    ("nsga2", "final_nsga2_front_evaluation"),
}
```

`load_train_validation()` returns only train/validation arrays.  `load_test()`
records the event before returning or raising.  Use sequential atomic JSON
rewrites; Task 5 has a single process and does not need a lock service.

- [x] **Step 4: Add audit integrity checks**

Reject malformed events, unknown job IDs, foreign run IDs, and an allowed event
whose reason is not in the frozen whitelist.  Completeness must fail if any
denied event is unresolved.

- [x] **Step 5: Verify Task 5**

Run protocol plus stage tests and inspect the temporary audit files through
assertions only.

---

### Task 6: Implement the stage engine and authoritative CLI

**Files:**
- Create: `experiments/icomp2026_stages.py`
- Create: `experiments/run_icomp2026_pipeline.py`
- Modify: `tests/test_experiment_protocol.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `StageContext`, `StageRunSummary`, `execute_stage(stage, *, context, executors, stop_after_jobs=None)`
- Produces CLI `main(argv: Sequence[str] | None = None) -> int`
- Consumes: protocol, layout, manifest, status store, audited data access

- [x] **Step 1: Write failing CLI and stage-order tests**

Assert all declared choices, `--resume` requiring `--run-id`, smoke overrides,
one-GPU sequential execution, smoke-only `--stop-after-jobs`, and the exact
`all` order:

```python
EXPECTED_ORDER = (
    "validate", "preprocess", "estimator", "removal", "alpha",
    "nsga2", "ablation", "summarize", "figures",
)


def test_all_uses_dependency_order(monkeypatch, runner_fixture):
    seen = []
    monkeypatch.setattr(runner_fixture, "execute", lambda stage: seen.append(stage))
    assert runner_fixture.run("all") == 0
    assert tuple(seen) == EXPECTED_ORDER
```

Test that independent jobs continue after one failure, the stage returns
nonzero, completed jobs are skipped on resume, and dependent stages do not run
after incomplete prerequisites.

- [x] **Step 2: Confirm RED**

Run CLI/engine tests; expect import failures for both new experiment modules.

- [x] **Step 3: Implement the stage contract**

```python
@dataclass(frozen=True)
class StageContext:
    protocol: ResolvedProtocol
    layout: RunLayout
    job: JobSpec
    status: StatusStore
    audit: AuditLog


StageExecutor = Callable[[StageContext], ResultRecord]

STAGE_DEPENDENCIES = {
    "validate": (),
    "preprocess": ("validate",),
    "estimator": ("preprocess",),
    "removal": ("preprocess",),
    "alpha": ("preprocess",),
    "nsga2": ("preprocess", "removal"),
    "ablation": ("preprocess",),
    "summarize": ("estimator", "removal", "alpha", "nsga2", "ablation"),
    "figures": ("summarize",),
}
```

`execute_stage` iterates jobs in manifest order, marks running, calls the
registered executor, atomically writes/validates the result, marks success, and
captures failures without hiding them.

Before a job starts, require every `depends_on` job to be successful.  Leave a
dependent job pending when its prerequisite failed so a later resume can run it.
In smoke mode, `--stop-after-jobs N` exits with code 75 after exactly `N`
successful atomic writes.  Reject this option in formal mode.

- [x] **Step 4: Implement CLI lifecycle**

Parse arguments, resolve/freeze configuration for a new run or verify a resume,
initialize append-mode logging, build the manifest, dispatch the selected
stage(s), and return `0` only when requested work is complete.  Do not change
the process working directory and do not redirect global stdout/stderr objects.

- [x] **Step 5: Verify Task 6**

Run CLI tests, `--help`, Ruff, and compilation.  `--help` must work without
loading datasets or XGBoost models.

---

### Task 7: Implement validate and split-first preprocess stages

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `experiments/preprocess_data/preprocess_abalone.py`
- Modify: `experiments/preprocess_data/preprocess_Credit-Card-Fraud.py`
- Modify: `experiments/preprocess_data/preprocess_crimedata.py`
- Modify: `experiments/preprocess_data/preprocess_Gisette.py`
- Modify: `experiments/preprocess_data/preprocess_ISOLET.py`
- Modify: `experiments/preprocess_data/preprocess_Madelon.py`
- Modify: `experiments/preprocess_data/preprocess_Spambase.py`

**Interfaces:**
- Produces: `run_validate_job(context) -> ResultRecord`
- Produces: `run_preprocess_job(context) -> ResultRecord`
- Consumes: cached raw CSV and `prepare_split(X, y, *, task, seed, max_features, train_frac, val_frac, test_frac, variance_threshold)`

- [x] **Step 1: Write synthetic raw-data stage tests**

Create small binary and regression CSVs in `tmp_path`, execute preprocessing,
and assert three non-overlapping partitions, train-only standardization,
selected columns, hashes, row counts, and `data_manifest.csv` entries.  Assert
that missing raw files fail validation instead of triggering a download.

- [x] **Step 2: Confirm RED**

Run only validate/preprocess tests; registered handlers must still be absent.

- [x] **Step 3: Implement validation without model fitting**

Check raw paths, writable run directory, package versions, XGBoost build info,
CUDA visibility when requested, task/utility compatibility, and that all stage
methods exist.  Return these checks as structured metrics/artifacts.

- [x] **Step 4: Implement canonical preprocessing**

Load cached raw CSV, split `X` and target, coerce target from the explicit task,
call `prepare_split` using the frozen fractions/feature settings, and write:

```text
preprocess/<dataset_id>/s<seed>/train.csv
preprocess/<dataset_id>/s<seed>/val.csv
preprocess/<dataset_id>/s<seed>/test.csv
```

Write each partition atomically.  Add row-index hashes and file SHA-256 values
to `data_manifest.csv`.  Do not read the produced test CSV again in this stage.

- [x] **Step 5: Reduce old preprocessors to raw-data preparation only**

Remove calls that fit `prepare_split` or write formal processed splits.  They
may download/clean and cache `data/raw/<dataset>.csv`; the unified runner is the
only code that creates formal split artifacts.

- [x] **Step 6: Verify Task 7**

Run `tests/test_data_protocol.py`, the new stage tests, Ruff on preprocessing
scripts, and a CPU smoke preprocessing run using only seed 42.  This is not a
formal result run.

---

### Task 8: Add model-fit accounting and estimator/ablation stage adapters

**Files:**
- Modify: `src/minppi_mofs/voting_game.py`
- Modify: `tests/test_voting_game.py`
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `AccFullCache.model_fit_count`, `AccFullCache.coalition_count`
- Produces: `run_estimator_job(context)`, `run_ablation_job(context)`
- Consumes: exact, random MC, canonical Algorithm B, hybrid, `UtilityView`

- [x] **Step 1: Write fit-accounting tests**

Assert that empty-coalition priors do not increment fits, a new nonempty
coalition increments once, a cache hit does not increment, and two utility
views share the same count.

- [x] **Step 2: Implement public counters**

Initialize `_model_fit_count = 0`; increment immediately after each successful
nonempty model fit.  Expose read-only properties:

```python
@property
def model_fit_count(self) -> int:
    return self._model_fit_count

@property
def coalition_count(self) -> int:
    return len(self._preds)
```

Also allow frozen `device` and `tree_method` keys in `model_params`; do not
silently discard backend configuration.

- [x] **Step 3: Write estimator job tests with tiny data**

For 5 features, compare exact/A/B/hybrid outputs against exact and assert
`harmful_precision`, `harmful_recall`, `harmful_f1`, `score_agreement`,
`model_fit_count`, `coalition_count`, and `runtime_s` fields.  For more than the
exact threshold, assert exact is absent by manifest rule rather than reported as
zero.

- [x] **Step 4: Implement estimator jobs**

Build one `AccFullCache` per method/budget using train/validation only.  For
Algorithm B, build canonical MI/correlation guidance with explicit task.  Use
`K == K_adv == budget` for the declared maximum marginal-search budget and
record actual coalition/model-fit counts so efficiency comparisons use real
cost.  Compare to exact only when an exact record exists.

- [x] **Step 5: Implement frozen ablations**

Use a shared cache plus two `UtilityView` objects for ISOLET accuracy versus OVR
AUC.  For guidance ablation, run random, MI-only, correlation-only, and combined
guidance with identical budgets and coalition plans.  Dispatcher ablation uses
exact only below the feature threshold.  No ablation opens test data.

- [x] **Step 6: Verify Task 8**

Run voting-game, MinPPI, utility, integration, and new stage tests.  Confirm no
deprecated guidance warning appears in canonical stage execution.

---

### Task 9: Implement matched removal and alpha-sweep stages

**Files:**
- Modify: `src/minppi_mofs/baselines/shap_top_k.py`
- Modify: `src/minppi_mofs/baselines/min_shap.py`
- Modify: `src/minppi_mofs/baselines/banzhaf_fs.py`
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `fit_predict(X_train, y_train, X_eval, *, task, model_params, backend, seed)` and `evaluate_fixed_subset`
- Produces: `run_removal_job(context)`, `run_alpha_job(context)`
- Consumes: MinPPI and SHAP/MinShap/Banzhaf ranking baselines

- [ ] **Step 1: Write same-model prediction tests**

Monkeypatch XGBoost and assert binary/multiclass both call `XGBClassifier`,
regression calls `XGBRegressor`, classification returns probabilities, and
metrics are computed only by `compute_utility` from those predictions.

- [ ] **Step 2: Implement the canonical final-evaluation helper**

```python
def evaluate_fixed_subset(
    X_train_val: pd.DataFrame,
    y_train_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: tuple[str, ...],
    *,
    task: str,
    utility: str,
    model_params: dict[str, object],
    backend: str,
    seed: int,
) -> float:
    selected = list(features)
    predictions = fit_predict(
        X_train_val[selected],
        y_train_val,
        X_test[selected],
        task=task,
        model_params=model_params,
        backend=backend,
        seed=seed,
    )
    return compute_utility(y_test, predictions, metric=utility)
```

Fit once on train+validation, produce probability/continuous predictions, and
call `compute_utility`.  Apply the same model parameters/backend to full,
MinPPI, and every matched control.

Extend `compute_shap_topk_importance`, `compute_minshap_importance`, and
`compute_banzhaf_importance` with explicit `utility`, `model_params`, and
`backend` keywords.  Their internal XGBoost fits must use the frozen model
configuration and backend instead of hard-coded CPU defaults.  Add propagation
assertions to the same-model tests.

- [ ] **Step 3: Write matched-removal tests**

Use a fixed harmful set of size `k=2`; assert SHAP, MinShap, Banzhaf, and every
random control remove exactly two features, full removes zero, 0-harmful remains
a valid record, and test data is accessed once with reason
`final_removal_evaluation` only after all subsets are frozen.

- [ ] **Step 4: Implement matched removal**

Compute MinPPI on train/validation.  Rank controls from training data only.
Create 30 deterministic random matched sets in formal mode and the frozen smoke
count in smoke mode.  The `selection_plan` job writes every method's fixed
feature set and its hash before any evaluation job can load test data.  Each
method has its own manifest/result record and one audited final test read.
Return raw per-control utility, `delta_utility`, compression, selection time,
evaluation time, and model-fit counts; do not average random trials inside a
method record.

- [ ] **Step 5: Write and implement alpha-sweep tests**

Assert the complete frozen grid is run, no test-based argmax is computed, each
feature set is fixed by the alpha `selection_plan`, and each alpha result has
one audited read using `final_alpha_grid_evaluation`.  Output contains harmful
count, compression, validation/test utility, preservation, runtime, and
feature-set hash per alpha.

- [ ] **Step 6: Verify Task 9**

Run removal/alpha stage tests plus the full utility suite.  Search the new stage
module for direct reads of `test.csv`; only `AuditedSplitAccess` may contain
that operation.

---

### Task 10: Make NSGA-II utility/backend-aware and add the fair MOFS stage

**Files:**
- Modify: `src/minppi_mofs/baselines/nsga2_fs.py`
- Modify: `tests/test_nsga2_hv.py`
- Modify: `tests/test_nsga2_quick.py`
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Extends: `_evaluate_individual(mask, X_tr, y_tr, X_val, y_val, *, task, utility, model_params, backend, seed)`
- Extends: `nsga2_feature_selection(X_train, y_train, X_val, y_val, *, task, utility, model_params, backend, seed, pop_size, max_gen, p_crossover, p_mutation, ref_point, verbose)`
- Produces: `run_nsga2_job(context)`

- [ ] **Step 1: Write failing propagation and fairness tests**

Assert explicit utility reaches `compute_utility`, backend reaches XGBoost
parameters, the configured reference point reaches every HV computation, and
full/filtered variants receive identical population and total evaluation
budgets.

- [ ] **Step 2: Implement utility/backend propagation**

Keep the model family task-dependent only (classifier versus regressor) and
compute the chosen higher-is-better utility from predictions.  Convert the
second objective to `1 - clipped_utility` consistently.  Derive
`max_gen = evaluations // population - 1` after counting the initial population,
and reject a budget not divisible into a complete generation schedule.

- [ ] **Step 3: Write final-front audit tests**

Assert search sees train/validation only, both fronts freeze before test access,
the one allowed read uses `final_nsga2_front_evaluation`, and every nondominated
mask is retrained on train+validation before test scoring.

- [ ] **Step 4: Implement full versus prefiltered jobs**

For `minppi_prefilter`, include the MinPPI selection time in total runtime.
Write validation HV, held-out test HV, front points, selected masks, evaluation
count, convergence evaluation count, runtime, and dimensional reduction.
Always use the frozen `(1.1, 1.1)` reference point.

- [ ] **Step 5: Verify Task 10**

Run both NSGA-II test files and the new stage tests.  Reconfirm the known
two-point HV equals `0.28`.

---

### Task 11: Implement completeness-first summarization and read-only figures

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `experiments/make_paper_figures.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `run_summarize_job(context)`, `run_figures_job(context)`
- Consumes: validated canonical records only
- Produces: CSV tables, `latex_macros.tex`, and figures under the run directory

- [ ] **Step 1: Write completeness-before-statistics tests**

Monkeypatch the aggregation function and assert it is never called when one
manifest job is missing.  Add complete five-seed fixtures and assert raw values,
mean, standard deviation, median, deterministic 95% bootstrap interval, paired
Wilcoxon/permutation result, Holm-adjusted p-value, and effect size.

- [ ] **Step 2: Implement deterministic summary functions**

Use seed `42` for bootstrap resampling and implement Holm correction locally to
avoid another dependency.  Preserve every raw seed row.  Treat five seeds as
paired units and label tests descriptive because power is low.

- [ ] **Step 3: Generate LaTeX macros from the same tables**

Escape names, format metrics from typed columns, and generate macros such as
`\CreditCompression`, `\CreditDeltaUtility`, and `\NsgaMeanHvChange` only from
validated aggregate rows.  Store the exact source table/column in a companion
JSON provenance map.

- [ ] **Step 4: Make figure generation read-only**

Refactor `make_paper_figures.py` functions to accept explicit table paths and
output directory.  They must not load raw datasets, train models, or write to
legacy `results/figures`.  Smoke mode generates the same figure types from
smoke tables and labels them as smoke artifacts.

- [ ] **Step 5: Verify Task 11**

Run summary/figure tests twice and compare hashes for deterministic CSV, macro,
and PDF outputs where metadata permits stable hashes.

---

### Task 12: Run the Task 5 acceptance sequence and document the handoff

**Files:**
- Modify: `experiments/README.md`
- Modify: `configs/README.md`
- Modify: `ICOMP2026_experiment_redesign_submission_plan.md`

**Interfaces:**
- Produces: documented commands for validation, smoke, interruption recovery,
  and Task 6 handoff
- Does not produce formal paper results

- [ ] **Step 1: Run all local verification before GPU use**

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\ruff.exe check src tests experiments demos
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\experiment_protocol.py experiments\icomp2026_stages.py experiments\run_icomp2026_pipeline.py
git diff --check
```

Expected: all tests pass, Ruff/compile/diff checks succeed.

- [ ] **Step 2: Validate the real environment**

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage validate --config-dir configs --output-root results\icomp2026_rerun --backend cuda --run-id task5-validate
```

Expected: CUDA is visible, seven cached raw datasets/configs validate, and no
model fitting occurs.

- [ ] **Step 3: Run the two-dataset smoke pipeline**

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-smoke --smoke
```

Expected: only Spambase and Abalone run; every requested job succeeds; status,
manifest, audit, tables, macros, and figures exist under one run directory.

- [ ] **Step 4: Demonstrate resume without rerunning successes**

Create a separate controlled-interruption run, then resume it:

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --stop-after-jobs 1
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --resume
```

Expected: the first command exits with code 75 after one successful job.  The
second command runs only pending work; the prior successful output hash and
timestamp remain unchanged.

- [ ] **Step 5: Audit and cleanup the smoke handoff**

Verify no unresolved denied test event, no missing combination, and no formal
run record.  Keep smoke outputs labeled `run_kind: smoke`; do not copy them into
the conference paper.

- [ ] **Step 6: Update documentation**

Document the authoritative CLI, frozen config policy, output schema, resume
procedure, test-access meanings, and the exact Task 6 stage order.  Mark Task 5
complete only after the smoke evidence exists.

## Final verification commands

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_experiment_protocol.py tests\test_icomp2026_stages.py -q
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\ruff.exe check src tests experiments demos
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\experiment_protocol.py experiments\icomp2026_stages.py experiments\run_icomp2026_pipeline.py
git diff --check
```

## Task 6 handoff gate

Do not begin the five-seed rerun until all of the following are true:

- the full test suite passes;
- the CUDA smoke run passes for Spambase and Abalone;
- resume reruns only failed/stale jobs;
- every allowed/denied test access is auditable and no denied event is unresolved;
- summary refuses a deliberately incomplete manifest;
- source/config fingerprints are frozen;
- smoke outputs are isolated from formal output paths;
- the user reviews the Task 5 evidence and explicitly approves Task 6.
