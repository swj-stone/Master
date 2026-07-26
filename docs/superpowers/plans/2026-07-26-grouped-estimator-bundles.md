# Grouped Estimator Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved grouped nested estimator mode into the frozen protocol and materialize all canonical estimator jobs from recoverable dataset/seed bundles.

**Architecture:** A frozen `ExecutionSpec` selects legacy or `grouped_nested_v1`. A new experiment module builds one atomic bundle per dataset/seed using the cache and nested primitives from Plan 1, while `run_estimator_job` preserves canonical manifest jobs, score paths, result records, and legacy benchmark behavior.

**Tech Stack:** Python 3.13, dataclasses, YAML, canonical JSON, SQLite-backed Plan 1 cache, NumPy/pandas/XGBoost, pytest, PowerShell.

## Global Constraints

- Complete `2026-07-26-coalition-cache-nested-schedules.md` first.
- Formal budgets remain `[10, 25, 50, 100, 200]`; all current manifest jobs remain present.
- `exact_max_features=16` governs public exact jobs; hybrid may internally use exact through dimension 20.
- The new formal resolved protocol must freeze backend `cuda`; source default CPU is not silently changed.
- Bundle and cache identity include protocol, prepared split, feature order, model, backend, task, and seed.
- Only one GPU/model worker may execute.
- Preserve parent/worker native-teardown isolation.
- Do not alter or resume `results/icomp2026_rerun/icomp2026-formal-20260723` after source changes.
- Do not stage or commit; the user owns Git.
- Use `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\venv\Scripts\python.exe`.

## File map

- Modify `configs/icomp2026_pipeline.yaml`: add frozen execution settings.
- Modify `src/minppi_mofs/experiment_protocol.py`: parse, serialize, hash, freeze, and validate `ExecutionSpec`; expose canonical cache/bundle paths.
- Create `experiments/icomp2026_estimator_bundle.py`: group request, progress, bundle schema, build/resume, validation, and per-job projection.
- Create `experiments/icomp2026_progress.py`: shared append-only run logging,
  progress formatting, throttling, and ETA calculation without runner/stage
  circular imports.
- Modify `experiments/icomp2026_stages.py`: keep legacy estimator helper and route grouped mode through the bundle.
- Modify `experiments/run_icomp2026_pipeline.py`: append group progress to the existing log without weakening isolation.
- Create `tests/test_estimator_bundle.py`: group build, exact/hybrid reuse, canonical projections, progress, corruption, and resume.
- Modify `tests/test_experiment_protocol.py`: execution config and frozen-protocol tests.
- Modify `tests/test_icomp2026_stages.py`: runner behavior, legacy/grouped routing,
  progress logging, and result validation.
- Modify `tests/test_native_process_guard.py`: confirm progress integration does
  not change controller/worker isolation.
- Modify `tests/test_icomp2026_reporting.py`: new counter schema remains reportable.

---

### Task 1: Freeze execution mode in the protocol

**Files:**
- Modify: `configs/icomp2026_pipeline.yaml`
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Modify: `tests/test_experiment_protocol.py`

**Interfaces:**
- Produces: `ExecutionSpec`
- Adds: `ResolvedProtocol.execution`
- Produces: `RunLayout.cache_root` and `RunLayout.bundle_root`
- Consumed later by bundle and stage code.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_formal_protocol_freezes_grouped_execution(config_dir):
    protocol = load_resolved_protocol(
        config_dir, seeds=None, backend="cuda", smoke=False
    )
    assert protocol.execution == ExecutionSpec(
        mode="grouped_nested_v1",
        cache_schema="coalition_utility_v1",
        checkpoint_every_fits=50,
        progress_every_seconds=30,
    )


def test_execution_rejects_unknown_or_extra_keys(config_dir):
    edit_global_yaml(config_dir, execution={"mode": "auto"})
    with pytest.raises(ValueError, match="execution"):
        load_resolved_protocol(config_dir, seeds=None, backend=None, smoke=False)
```

Add a resolved-protocol hash test showing any execution field changes the
protocol hash and makes `open_resumed_run` reject a mismatch.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
Set-Location 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master'
& '..\venv\Scripts\python.exe' -m pytest tests/test_experiment_protocol.py -q -p no:cacheprovider -k execution
```

Expected: failure because `ExecutionSpec` and the `execution` key are absent.

- [ ] **Step 3: Add the frozen configuration**

Add:

```yaml
execution:
  mode: grouped_nested_v1
  cache_schema: coalition_utility_v1
  checkpoint_every_fits: 50
  progress_every_seconds: 30
```

Do not change the source `backend: cpu`; formal commands continue to pass
`--backend cuda`.

- [ ] **Step 4: Implement strict parsing**

Add:

```python
@dataclass(frozen=True)
class ExecutionSpec:
    mode: str
    cache_schema: str
    checkpoint_every_fits: int
    progress_every_seconds: int
```

Allow modes exactly `legacy` and `grouped_nested_v1`; require cache schema
`coalition_utility_v1`; require positive, non-bool integer intervals. Add
`execution` to `_GLOBAL_KEYS`, the resolved protocol, canonical serialization,
metadata freeze, equality, and protocol hashing.

Extend `RunLayout` and `_run_layout()` with:

```python
cache_root=root / "cache"
bundle_root=root / "estimator" / "_bundles"
```

Create those directories with the other canonical run directories. Reject
layout instances whose paths differ from `_run_layout(root)`.

- [ ] **Step 5: Run protocol and runner tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_experiment_protocol.py tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'protocol or runner or layout'
```

Expected: all selected tests pass.

- [ ] **Step 6: Review checkpoint**

Report the resolved execution block, hash mismatch test, layout paths, and test
totals. Do not stage or commit.

---

### Task 2: Define estimator group and bundle schemas

**Files:**
- Create: `experiments/icomp2026_estimator_bundle.py`
- Create: `tests/test_estimator_bundle.py`

**Interfaces:**
- Produces: `ESTIMATOR_BUNDLE_SCHEMA_VERSION`
- Produces: `EstimatorGroupRequest`
- Produces: `EstimatorCheckpoint`
- Produces: `EstimatorBundle`
- Produces: `group_request_from_context(context, prepared_manifest_row, feature_order)`
- Produces: `bundle_relpath(request)` and `progress_relpath(request)`

- [ ] **Step 1: Write failing schema tests**

```python
def test_group_request_identity_includes_every_frozen_input(
    stage_context, data_row, prepared_columns
):
    request = group_request_from_context(
        stage_context, data_row, tuple(prepared_columns)
    )
    assert request.dataset_id == stage_context.job.dataset_id
    assert request.seed == stage_context.job.seed
    assert request.budgets == (10, 25, 50, 100, 200)
    assert request.backend == "cuda"
    assert request.split_hashes == {
        "train": data_row["train_file_sha256"],
        "validation": data_row["val_file_sha256"],
    }
    assert bundle_relpath(request) == (
        "estimator/_bundles/spambase/s42/bundle.json"
    )
```

Add tests rejecting duplicate/unsorted budgets, mismatched job dataset/seed,
missing data-manifest hashes, unknown methods, and unsupported execution mode.

- [ ] **Step 2: Run and confirm module failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py -q -p no:cacheprovider
```

Expected: collection fails because the bundle module is absent.

- [ ] **Step 3: Implement frozen schema types**

Use:

```python
ESTIMATOR_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EstimatorGroupRequest:
    protocol_hash: str
    dataset_id: str
    seed: int
    task: str
    utility: str
    ordered_features: tuple[str, ...]
    split_hashes: Mapping[str, str]
    model_params: Mapping[str, object]
    backend: str
    methods: tuple[str, ...]
    budgets: tuple[int, ...]
    exact_max_features: int
    alpha: float
    cache_schema: str
    checkpoint_every_fits: int
    progress_every_seconds: int

    @property
    def digest(self) -> str:
        payload = {
            "protocol_hash": self.protocol_hash,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "task": self.task,
            "utility": self.utility,
            "ordered_features": list(self.ordered_features),
            "split_hashes": dict(self.split_hashes),
            "model_params": dict(self.model_params),
            "backend": self.backend,
            "methods": list(self.methods),
            "budgets": list(self.budgets),
            "exact_max_features": self.exact_max_features,
            "alpha": self.alpha,
            "cache_schema": self.cache_schema,
            "checkpoint_every_fits": self.checkpoint_every_fits,
            "progress_every_seconds": self.progress_every_seconds,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EstimatorCheckpoint:
    method: str
    budget: int
    scores: tuple[tuple[str, int], ...]
    logical_evaluation_count: int
    logical_unique_coalition_count: int
    physical_model_fit_count: int
    cache_hit_count: int
    schedule_digest: str | None
    component_runtime_s: float


@dataclass(frozen=True)
class EstimatorBundle:
    schema_version: int
    request_digest: str
    checkpoints: tuple[EstimatorCheckpoint, ...]
    physical_model_fit_count: int
    cache_hit_count: int
    group_runtime_s: float
```

Use canonical JSON conversion functions that reject unknown/missing fields,
non-finite times, duplicate method/budget pairs, noncanonical feature order,
invalid scores, and missing manifest-authorized pairs.

- [ ] **Step 4: Add canonical round-trip and conflict tests**

Write one bundle to a temporary canonical JSON file through
`atomic_write_json`, read it through `read_canonical_json`, validate it, and
assert equality. Mutate request digest, one score feature, a schedule digest,
and a counter in separate tests; each must fail with a specific `ValueError`.

- [ ] **Step 5: Run bundle schema tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py -q -p no:cacheprovider
```

Expected: schema tests pass.

- [ ] **Step 6: Review checkpoint**

Report bundle schema, canonical paths, validation failures, and test count. Do
not stage or commit.

---

### Task 3: Build one complete estimator bundle

**Files:**
- Modify: `experiments/icomp2026_estimator_bundle.py`
- Modify: `tests/test_estimator_bundle.py`

**Interfaces:**
- Produces: `build_estimator_bundle(request, X_train, y_train, X_val, y_val, store, progress_sink)`
- Consumes Plan 1: cache/view, nested schedules/checkpoints, exact estimator,
  MI/correlation guidance, and hybrid derivation.

Here `progress_sink` has type
`Callable[[minppi_mofs.nested_estimation.NestedProgress], None]`.

- [ ] **Step 1: Write a failing grouped-build test**

```python
def test_group_build_computes_components_once_and_emits_all_pairs(
    toy_request, toy_frames, counting_store
):
    bundle = build_estimator_bundle(
        toy_request,
        *toy_frames,
        store=counting_store,
        progress_sink=lambda event: None,
    )
    assert {(item.method, item.budget) for item in bundle.checkpoints} == {
        (method, budget)
        for method in toy_request.methods
        for budget in toy_request.budgets
        if method != "exact" or len(toy_request.ordered_features)
        <= toy_request.exact_max_features
    }
    guided = checkpoint_map(bundle, "mi_correlation_guided")
    hybrid = checkpoint_map(bundle, "hybrid")
    assert hybrid == guided  # toy dimension is 21..75
    assert counting_store.guided_build_count == 1
    assert counting_store.random_build_count == 1
```

Use separate parameterized tests for dimensions 8, 55, and 100.

- [ ] **Step 2: Run and confirm missing builder failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py -q -p no:cacheprovider -k group_build
```

Expected: failure because `build_estimator_bundle` is undefined.

- [ ] **Step 3: Implement component orchestration**

The builder must execute this exact component flow:

```python
random_schedule = make_nested_random_schedule(
    request.ordered_features, budgets=request.budgets, seed=request.seed
)
guided_schedule = make_nested_guided_schedule(
    request.ordered_features, budgets=request.budgets, seed=request.seed
)
exact_scores = (
    estimate_minppi_exact(
        X_train,
        X_val,
        y_train,
        y_val,
        alpha=request.alpha,
        task=request.task,
        utility=request.utility,
        acc_cache=default_view,
        seed=request.seed,
    )
    if feature_count <= 20
    else None
)
random_scores = estimate_random_checkpoints(
    random_schedule, default_view, progress=progress_sink
)
guided_scores = estimate_guided_checkpoints(
    guided_schedule,
    default_view,
    mi_scores=mi_scores,
    correlation_neighbors=correlation_neighbors,
    progress=progress_sink,
)

hybrid = derive_hybrid_checkpoints(
    feature_count=feature_count,
    checkpoints=request.budgets,
    exact_scores=exact_scores,
    random_scores=random_scores,
    guided_scores=guided_scores,
)
```

Only materialize public exact checkpoints when
`feature_count <= exact_max_features` and `"exact"` is authorized. Internal
exact for dimensions 17 through 20 is permitted only as a hybrid component.

Collect logical unique coalition keys per method/checkpoint from the nested
view. Physical fits and cache hits come from the shared store delta. Time each
component with `time.perf_counter()`. Execute components in the frozen order
`exact`, `random_mc`, `mi_correlation_guided`, then derived `hybrid`; capture
cache-counter snapshots at every budget boundary. Store cumulative
method/checkpoint physical fits and hits in `EstimatorCheckpoint`, and store the
complete group totals separately in `EstimatorBundle`. Hybrid checkpoint
physical fits are zero because it derives scores from completed components.

- [ ] **Step 4: Add Task 3 digest and counter tests**

For multiclass toy data, build accuracy and OVR-AUC views over one cache and
assert:

```python
assert accuracy_record.prediction_sha256 == auc_record.prediction_sha256
assert bundle.physical_model_fit_count < sum(
    item.logical_unique_coalition_count for item in bundle.checkpoints
)
```

Do not require a speed ratio in unit tests.

- [ ] **Step 5: Run grouped-build tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py tests/test_nested_estimation.py tests/test_coalition_cache.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Review checkpoint**

Report component invocation counts, public pair counts for dimensions 8/55/100,
physical/logical counters, and tests. Do not stage or commit.

---

### Task 4: Atomic ensure/resume and visible progress

**Files:**
- Modify: `experiments/icomp2026_estimator_bundle.py`
- Create: `experiments/icomp2026_progress.py`
- Modify: `experiments/run_icomp2026_pipeline.py`
- Modify: `tests/test_estimator_bundle.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `tests/test_native_process_guard.py`

**Interfaces:**
- Produces: `ensure_estimator_bundle(request, frames, run_root, progress_sink)`
- Produces from `icomp2026_progress.py`: `EstimatorProgressEvent`
- Produces: atomic `bundle.json` and `progress.json`

- [ ] **Step 1: Write interrupted-build recovery test**

Inject a fit hook that raises after 51 physical fits:

```python
with pytest.raises(InjectedStop):
    ensure_estimator_bundle(
        request=toy_request,
        frames=toy_frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
        fit_hook=stop_after(51),
    )

assert progress_path.exists()
assert not bundle_path.exists()
first_fit_count = sqlite_entry_count(cache_path)

bundle = ensure_estimator_bundle(
    request=toy_request,
    frames=toy_frames,
    run_root=tmp_path,
    progress_sink=lambda event: None,
    fit_hook=None,
)
assert bundle_path.exists()
assert sqlite_entry_count(cache_path) >= first_fit_count
assert resumed_physical_fits < cold_physical_fits
```

Also assert a different request digest or schedule digest fails instead of
resuming.

- [ ] **Step 2: Run and confirm missing ensure failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py -q -p no:cacheprovider -k resume
```

Expected: failure because `ensure_estimator_bundle` is undefined.

- [ ] **Step 3: Implement progress events and atomic publication**

Define in `experiments/icomp2026_progress.py`:

```python
@dataclass(frozen=True)
class EstimatorProgressEvent:
    request_digest: str
    schedule_digest: str
    dataset_id: str
    seed: int
    algorithm: str
    feature_index: int
    feature_count: int
    budget: int
    max_budget: int
    physical_model_fit_count: int
    cache_hit_count: int
    elapsed_s: float
    eta_s: float | None
```

Write canonical `progress.json` after each cache commit and at budget/feature
boundaries. Its exact schema is the event above plus
`last_completed_trial_index`; validate both digests before resume. Replaying
already cached trials is permitted only after this validation.

Track completion timestamps for the most recent 200 physical fits in a bounded
`deque`. When at least two timestamps exist, calculate recent throughput as
`(count - 1) / (last_timestamp - first_timestamp)` and ETA as the remaining
non-early-stopped master-schedule trial upper bound divided by that throughput.
Otherwise report `eta_s=None`. Emit a formatted line at least every
`progress_every_seconds`:

```text
[madelon seed=42 random] feature 37/100, K=100/200 fits=8421 cache_hits=3105 elapsed=21m ETA=34m
```

`ensure_estimator_bundle` supplies the builder with an `on_nested_progress`
adapter. The adapter combines `NestedProgress` with request digests, cache
counter snapshots, elapsed time, recent throughput, and the last completed
trial index to construct `EstimatorProgressEvent`, then calls
`emit_progress`. The builder does not import the runner or know the run log
path.

Build the final bundle in memory, validate it, then publish with
`atomic_write_json`. Reopen and validate the file before returning.

- [ ] **Step 4: Preserve parent/worker logging without a circular import**

Move the current append-only `_append_log` implementation into
`experiments/icomp2026_progress.py` as:

```python
def append_run_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")


def emit_progress(path: Path, event: EstimatorProgressEvent) -> None:
    line = format_progress(event)
    print(line, flush=True)
    append_run_log(path, f"progress {line}")
```

Import `append_run_log` into the runner for lifecycle messages and
`emit_progress` into the stage/bundle integration. Do not import the runner
from stages. Do not create another process or touch the native worker receipt.

- [ ] **Step 5: Add corruption and completed-reuse tests**

Assert:

- completed bundle returns without fitting;
- malformed `progress.json` fails closed;
- corrupt SQLite entry fails closed;
- conflicting canonical bundle fails closed;
- successful bundle survives a fresh Python object; and
- progress formatting never prints labels or data values.

- [ ] **Step 6: Run recovery and runner tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py tests/test_icomp2026_stages.py tests/test_native_process_guard.py -q -p no:cacheprovider -k 'progress or resume or native or worker'
```

Expected: all tests pass.

- [ ] **Step 7: Review checkpoint**

Report interruption boundary, reused entries, progress example, bundle
publication checks, and tests. Do not stage or commit.

---

### Task 5: Route canonical estimator jobs through bundles

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `tests/test_estimator_bundle.py`

**Interfaces:**
- Produces: `_run_legacy_estimator_job(context)`
- Produces: `_run_grouped_estimator_job(context)`
- `run_estimator_job(context)` dispatches solely by `protocol.execution.mode`.

- [ ] **Step 1: Write routing tests**

```python
def test_grouped_estimator_job_projects_bundle_checkpoint(monkeypatch, context):
    monkeypatch.setattr(
        stages, "ensure_estimator_bundle", lambda *args, **kwargs: BUNDLE
    )
    result = stages.run_estimator_job(context_for("random_mc", budget=25))
    checkpoint = next(
        item
        for item in BUNDLE.checkpoints
        if item.method == "random_mc" and item.budget == 25
    )
    assert result.metrics["budget"] == 25
    assert result.metrics["logical_evaluation_count"] == EXPECTED_LOGICAL
    assert (
        result.metrics["physical_model_fit_count"]
        == checkpoint.physical_model_fit_count
    )
    assert (
        result.metrics["group_physical_model_fit_count"]
        == BUNDLE.physical_model_fit_count
    )
    assert result.artifacts["scores"].endswith("/random_mc/b25/scores.json")


def test_legacy_mode_does_not_open_bundle(monkeypatch, legacy_context):
    monkeypatch.setattr(
        stages,
        "ensure_estimator_bundle",
        lambda *args, **kwargs: pytest.fail("bundle used"),
    )
    assert stages.run_estimator_job(legacy_context).metrics["budget"] == 25
```

- [ ] **Step 2: Run and confirm routing failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'grouped_estimator or legacy_mode'
```

Expected: failures because mode dispatch is not implemented.

- [ ] **Step 3: Extract the legacy helper without behavior change**

Move the current body of `run_estimator_job` into
`_run_legacy_estimator_job`. Keep validation, old schedule generation, cache
construction, artifact schema, runtime, and exact-agreement metrics unchanged.
Make the public function dispatch:

```python
def run_estimator_job(context: StageContext) -> ResultRecord:
    if context.protocol.execution.mode == "legacy":
        return _run_legacy_estimator_job(context)
    if context.protocol.execution.mode == "grouped_nested_v1":
        return _run_grouped_estimator_job(context)
    raise ValueError("unsupported execution mode")
```

- [ ] **Step 4: Implement grouped projection**

Load prepared train/validation data and the rich manifest row, build the group
request, ensure the bundle, find exactly one matching checkpoint, and publish
the existing score path. Preserve:

```json
{
  "protocol_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "job_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "dataset_id": "spambase",
  "seed": 42,
  "method": "random_mc",
  "budget": 25,
  "scores": [{"feature": "word_freq_make", "score": -1}],
  "model_fit_count": 123,
  "coalition_count": 124,
  "logical_evaluation_count": 625,
  "logical_unique_coalition_count": 123,
  "physical_model_fit_count": 80,
  "cache_hit_count": 43,
  "group_request_digest": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "group_physical_model_fit_count": 25000,
  "group_cache_hit_count": 12000,
  "group_runtime_s": 4200.0,
  "schedule_digest": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
```

The checkpoint physical-fit/hit fields are cumulative within that method and
budget under the bundle's fixed component execution order. The group-prefixed
fields describe the complete bundle and are deduplicated by
`group_request_digest` in reporting. Validate feature order and canonical job
identity before publication.

- [ ] **Step 5: Add all-method/all-budget projection test**

Create a smoke manifest, project every estimator job from two toy bundles, and
assert every canonical score path and `ResultRecord` validates. Assert exact
jobs are absent above `exact_max_features`, and hybrid does not create a new
physical fit.

- [ ] **Step 6: Run stage tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_estimator_bundle.py tests/test_icomp2026_stages.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Review checkpoint**

Report legacy/grouped routing evidence, projected job count, schemas, and test
totals. Do not stage or commit.

---

### Task 6: Reporting compatibility and grouped smoke verification

**Files:**
- Modify: `experiments/summarize_icomp2026.py`
- Modify: `tests/test_icomp2026_reporting.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `README.md`

**Interfaces:**
- Consumes grouped estimator metrics.
- Produces summaries that label logical algorithmic work and physical cached
  execution separately.

- [ ] **Step 1: Write failing reporting tests**

Build estimator records with:

```python
metrics = {
    "budget": 25,
    "model_fit_count": 100,
    "coalition_count": 101,
    "logical_evaluation_count": 2500,
    "logical_unique_coalition_count": 100,
    "physical_model_fit_count": 40,
    "cache_hit_count": 60,
    "runtime_s": 12.5,
    "group_request_digest": "a" * 64,
    "group_physical_model_fit_count": 400,
    "group_cache_hit_count": 600,
    "group_runtime_s": 60.0,
}
```

Assert the summary keeps existing estimator comparison columns and adds
unambiguously named physical/cache columns. Reject missing grouped fields only
when execution mode is grouped.

- [ ] **Step 2: Run and confirm reporting failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_reporting.py -q -p no:cacheprovider -k grouped
```

Expected: grouped schema is not yet recognized.

- [ ] **Step 3: Implement mode-aware validation and summaries**

Keep `model_fit_count` as deterministic logical unique nonempty coalitions.
Add physical fits, hits, and group runtime to machine-readable tables. Never
sum `group_runtime_s` once per projected job; deduplicate it by group request
digest before producing total physical runtime.

- [ ] **Step 4: Document execution semantics**

Add a README section with:

- legacy vs grouped mode;
- nested budget definition;
- logical vs physical counters;
- formal CUDA override requirement;
- new run-id requirement after source changes; and
- prohibition on mixing the stopped formal run with grouped results.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_experiment_protocol.py tests/test_estimator_bundle.py tests/test_icomp2026_stages.py tests/test_icomp2026_reporting.py tests/test_native_process_guard.py -q -p no:cacheprovider
& '..\venv\Scripts\python.exe' -m pytest tests -q -p no:cacheprovider
```

Expected: both commands pass with only the repository's accepted skip.

- [ ] **Step 6: Run a new grouped smoke**

Use a new run ID:

```powershell
& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-estimator-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage validate

& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-estimator-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage preprocess `
  --resume

& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-estimator-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage estimator `
  --resume
```

Expected: public exit code 0; preprocess and estimator statuses complete;
bundles exist for both smoke dataset/seed groups; all projected estimator jobs
succeed; progress lines appear; physical fits are below summed logical unique
fits. Never point these commands at `icomp2026-formal-20260723`.

- [ ] **Step 7: Review checkpoint**

Report focused/full tests, smoke run ID, status counts, bundle/cache audit,
logical/physical counters, and source changes. Do not stage or commit.
