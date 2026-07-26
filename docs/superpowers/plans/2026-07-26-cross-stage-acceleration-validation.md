# Cross-Stage Acceleration and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse validated grouped estimator work in removal, alpha, ablation, and NSGA-II, then enforce the approved scientific non-inferiority and speed gates before a new formal run.

**Architecture:** Stage adapters reopen the run-scoped coalition cache with strict split/namespace identity and consume canonical estimator artifacts only through validated helpers. NSGA-II memoizes duplicate selected-feature tuples without changing its 800 logical evaluations or random evolution. A validation command produces a canonical training/validation-only acceptance report and a final smoke/resume audit.

**Tech Stack:** Python 3.13, Plan 1 cache, Plan 2 bundles, NumPy/pandas/XGBoost, NSGA-II implementation, canonical JSON/CSV, pytest, PowerShell.

## Global Constraints

- Complete both preceding 2026-07-26 acceleration plans first.
- Do not reduce datasets, seeds, estimator budgets, removal controls, alpha grid,
  NSGA-II population/evaluations, or ablations.
- Selection may access only training and validation namespaces.
- Final test cache opens only after existing audited selection freeze.
- NSGA-II performs all 800 logical evaluations; duplicate masks may avoid only
  physical fits.
- Preserve CUDA as the explicitly resolved formal backend and serialize GPU work.
- The stopped `icomp2026-formal-20260723` run is read-only baseline evidence.
- Acceptance comparisons use cold new builds, not completed bundles or warm
  caches.
- No Git staging or commits; the user owns Git.
- Use `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\venv\Scripts\python.exe`.

## File map

- Modify `experiments/icomp2026_stages.py`: cache factory, estimator-result authority, removal/alpha/ablation reuse, and final-test namespace.
- Modify `src/minppi_mofs/baselines/_coalition_utility.py`: optional shared cache-backed evaluator for identical internal splits.
- Modify `src/minppi_mofs/baselines/banzhaf_fs.py` and `min_shap.py`: accept one supplied coalition evaluator.
- Modify `src/minppi_mofs/baselines/nsga2_fs.py`: objective memoization and diagnostics without RNG changes.
- Modify `tests/test_icomp2026_stages.py`, `tests/test_task9_modeling.py`, and `tests/test_nsga2_hv.py`.
- Modify `experiments/summarize_icomp2026.py` and `tests/test_icomp2026_reporting.py`: cross-stage physical/logical accounting.
- Create `experiments/validate_grouped_acceleration.py`: exact, non-inferiority, speed, digest, and cold-cache acceptance report.
- Create `tests/test_validate_grouped_acceleration.py`: report schema, thresholds, no-test-access, and command behavior.
- Modify `ICOMP2026_experiment_redesign_submission_plan.md` and `README.md`: execution and acceptance record.

---

### Task 1: Reopen the authorized selection cache from stage context

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `_selection_cache_identity(context, data_row, X_train, y_train, X_val, y_val)`
- Produces: `_open_selection_cache(context, frames, data_row)`
- Produces: `_open_final_test_cache(context, train_val, test, data_row, reason)`
- Consumed by removal, alpha, ablation, and NSGA-II tasks.

- [ ] **Step 1: Write failing namespace and identity tests**

```python
def test_selection_cache_identity_uses_prepared_content(
    context, prepared_row, prepared_frames
):
    X_train, y_train, X_val, y_val = prepared_frames
    identity = _selection_cache_identity(
        context, prepared_row, X_train, y_train, X_val, y_val
    )
    assert identity.namespace == "selection_validation"
    assert identity.train_features_sha256 == hash_numeric_frame(X_train)
    assert identity.train_target_sha256 == hash_numeric_array(y_train)
    assert identity.eval_features_sha256 == hash_numeric_frame(X_val)
    assert identity.eval_target_sha256 == hash_numeric_array(y_val)
    assert identity.backend == "cuda"


def test_final_test_cache_requires_audited_reason(context, prepared_row):
    with pytest.raises(TestAccessViolation):
        _open_final_test_cache(
            context, train_val, test, prepared_row, reason="selection"
        )
```

- [ ] **Step 2: Run and confirm helper failures**

Run:

```powershell
Set-Location 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master'
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'selection_cache_identity or final_test_cache'
```

Expected: failures because the helpers are absent.

- [ ] **Step 3: Implement manifest-bound cache factories**

First validate the rich data-manifest row against the loaded prepared files,
then build `CoalitionCacheIdentity` from deterministic hashes of the loaded
feature frames and target arrays, protocol, dataset authority, ordered columns,
task classes, model parameters, backend, and installed library versions. Use:

```text
cache/selection_validation/<dataset>/s<seed>/coalitions.sqlite3
cache/final_test/<dataset>/s<seed>/coalitions.sqlite3
```

Selection cache construction must not load `test.csv` or read a test hash beyond
the already frozen manifest metadata. Final-test construction must call the
existing audited `_task9_load_test` path with its approved reason before opening
the final-test namespace.

- [ ] **Step 4: Add collision and leakage tests**

Assert different validation file hash, model parameter, backend, feature order,
or namespace cannot open the same database. Pre-create a valid final-test cache
and assert a selection context still cannot access it.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py tests/test_coalition_cache.py -q -p no:cacheprovider -k 'cache or test_access'
```

Expected: selected tests pass.

- [ ] **Step 6: Review checkpoint**

Report cache paths, identities, audited reasons, and leakage tests. Do not stage
or commit.

---

### Task 2: Reuse estimator authority in removal

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Produces: `_completed_estimator_checkpoint(context, method, budget, expected_features)`
- Removal selection consumes the maximum-budget hybrid checkpoint.
- Changes stage dependencies so `removal`, `alpha`, and `ablation` require both
  `preprocess` and `estimator`.

- [ ] **Step 1: Write failing removal-reuse test**

```python
def test_cross_stage_reuse_requires_completed_estimator():
    assert STAGE_DEPENDENCIES["removal"] == ("preprocess", "estimator")
    assert STAGE_DEPENDENCIES["alpha"] == ("preprocess", "estimator")
    assert STAGE_DEPENDENCIES["ablation"] == ("preprocess", "estimator")


def test_removal_uses_completed_max_budget_hybrid(monkeypatch, removal_context):
    monkeypatch.setattr(
        stages,
        "_completed_estimator_checkpoint",
        lambda *args, **kwargs: HYBRID_CHECKPOINT,
    )
    monkeypatch.setattr(
        stages,
        "estimate_minppi_hybrid",
        lambda *args, **kwargs: pytest.fail("hybrid recomputed"),
    )
    record = stages._run_removal_plan(removal_context)
    document = read_removal_plan(record)
    assert minppi_scores(document) == HYBRID_CHECKPOINT.scores
    assert document["methods"][MINPPI_INDEX]["physical_model_fit_count"] == 0
```

- [ ] **Step 2: Run and confirm recomputation**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k removal_uses_completed
```

Expected: failure because removal currently calls `estimate_minppi_hybrid`.

- [ ] **Step 3: Implement validated estimator lookup**

Update `STAGE_DEPENDENCIES` with the exact tuples asserted above. Add a runner
test that completes preprocess but not estimator, requests removal, and receives
a dependency issue without opening a coalition cache or writing a removal
artifact.

Find the unique successful estimator job matching dataset, seed, method, and
budget. Validate its `ResultRecord`, score artifact, protocol hash, feature
order, schedule digest, logical counters, and grouped execution mode. Return an
immutable checkpoint object. Missing, failed, duplicate, or conflicting
authority raises `ValueError`.

- [ ] **Step 4: Replace only the removal MinPPI computation**

Use maximum-budget hybrid scores to determine harmful features. Preserve the
three baseline computations, random matched controls, subset hashes, and
selection-time schema. Record MinPPI physical fits as zero in this stage and
carry its logical unique coalition count from the estimator artifact.

- [ ] **Step 5: Run removal tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k removal
```

Expected: all removal tests pass.

- [ ] **Step 6: Review checkpoint**

Report estimator authority validation, removed recomputation, counters, and
tests. Do not stage or commit.

---

### Task 3: Share baseline coalition utilities when identities match

**Files:**
- Modify: `src/minppi_mofs/baselines/_coalition_utility.py`
- Modify: `src/minppi_mofs/baselines/banzhaf_fs.py`
- Modify: `src/minppi_mofs/baselines/min_shap.py`
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_task9_modeling.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Adds optional: `coalition_utility: Callable[[set[str]], float] | None`
- Produces: `make_shared_baseline_utility(X, y, task, utility, model_params, backend, seed, store, diagnostics) -> CoalitionUtility`

- [ ] **Step 1: Write one-supplied-evaluator tests**

```python
def test_banzhaf_and_minshap_use_supplied_evaluator(monkeypatch, toy_data):
    evaluator = RecordingEvaluator()
    banzhaf = compute_banzhaf_importance(
        *toy_data,
        coalition_utility=evaluator,
        **POLICY,
    )
    minshap = compute_minshap_importance(
        *toy_data,
        coalition_utility=evaluator,
        **POLICY,
    )
    assert banzhaf.keys() == minshap.keys()
    assert evaluator.physical_fit_count == len(set(evaluator.coalitions))
```

Assert a supplied evaluator with the wrong feature universe is rejected before
sampling.

- [ ] **Step 2: Run and confirm unexpected-keyword failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_task9_modeling.py -q -p no:cacheprovider -k supplied_evaluator
```

Expected: failure because the baselines do not accept the parameter.

- [ ] **Step 3: Add the optional evaluator boundary**

When `coalition_utility is None`, retain current internal split and behavior.
When supplied, validate a protocol exposing its frozen feature order and use it
for all coalition calls. Do not allow task, utility, model, backend, seed, or
split parameters to be silently overridden.

- [ ] **Step 4: Construct one baseline-internal cache in removal**

Reproduce the current deterministic sample cap and 80/20 internal split once.
Build a `baseline_internal` identity from the resulting arrays. Supply the same
cache-backed evaluator to Banzhaf and MinShap. SHAP bottom remains separate.

- [ ] **Step 5: Add equality and fit-reduction tests**

On fixed toy data, assert legacy and supplied-evaluator scores are equal within
`1e-12`, while physical fits with the shared evaluator do not exceed the sum of
separate legacy fits. Verify diagnostics distinguish logical coalitions,
physical fits, and hits.

- [ ] **Step 6: Run baseline and stage tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_task9_modeling.py tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'banzhaf or minshap or removal'
```

Expected: all selected tests pass.

- [ ] **Step 7: Review checkpoint**

Report legacy equality, physical fit reduction, diagnostics, and tests. Do not
stage or commit.

---

### Task 4: Re-evaluate alpha thresholds from cached utilities

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Alpha plan consumes the maximum nested schedules and one selection cache.
- Produces no new physical fits when the estimator cache is complete.

- [ ] **Step 1: Write failing zero-new-fit alpha test**

```python
def test_alpha_grid_reuses_complete_estimator_cache(alpha_context):
    warm_estimator_group(alpha_context)
    before = cache_stats(alpha_context)
    record = stages._run_alpha_plan(alpha_context)
    after = cache_stats(alpha_context)
    assert after.physical_model_fit_count == before.physical_model_fit_count
    assert after.cache_hit_count > before.cache_hit_count
    assert len(read_alpha_plan(record)["entries"]) == len(
        alpha_context.protocol.alpha_grid
    )
```

- [ ] **Step 2: Run and confirm new fits occur**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k alpha_grid_reuses
```

Expected: current alpha plan creates a fresh `AccFullCache`, so the assertion
fails.

- [ ] **Step 3: Use nested schedules with per-alpha views**

Open the selection cache, rebuild deterministic random/guided schedules from the
same request digest, and evaluate each alpha through a new
`CoalitionUtilityView`. Derive hybrid scores at maximum budget and evaluate the
retained subset through the same cache. Verify schedule digests match the
estimator bundle before any lookup.

- [ ] **Step 4: Preserve alpha artifact semantics**

Keep alpha tokens, feature order, harmful counts, compression, retained hashes,
validation utility, and four canonical evaluation jobs. Add physical fits and
hits without assigning the group runtime four times.

- [ ] **Step 5: Run alpha and cache tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py tests/test_nested_estimation.py tests/test_coalition_cache.py -q -p no:cacheprovider -k alpha
```

Expected: all selected tests pass; warmed alpha test reports zero new fits.

- [ ] **Step 6: Review checkpoint**

Report schedule-digest check, four alpha results, fit delta, hits, and tests. Do
not stage or commit.

---

### Task 5: Reuse estimator results in dispatcher and guidance ablations

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Dispatcher ablation consumes corresponding maximum-budget estimator scores.
- `mi_correlation` guidance consumes the maximum-budget guided scores.
- Other guidance variants use the persistent selection cache.

- [ ] **Step 1: Write dispatcher and guidance reuse tests**

```python
@pytest.mark.parametrize(
    "value",
    ["exact", "random_mc", "mi_correlation_guided", "hybrid"],
)
def test_dispatcher_ablation_projects_estimator(value, context, monkeypatch):
    monkeypatch.setattr(
        stages,
        "estimate_minppi_" + value.replace("mi_correlation_guided", "guided"),
        lambda *args, **kwargs: pytest.fail("estimator recomputed"),
        raising=False,
    )
    result = stages._run_dispatcher_ablation(context_for(value))
    assert result.metrics["physical_model_fit_count"] == 0


def test_mi_correlation_guidance_reuses_guided_checkpoint(context):
    result = stages._run_guidance_ablation(context_for("mi_correlation"))
    assert result.metrics["physical_model_fit_count"] == 0
```

- [ ] **Step 2: Run and confirm recomputation failures**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'dispatcher_ablation_projects or mi_correlation_guidance_reuses'
```

Expected: current implementations recompute.

- [ ] **Step 3: Implement artifact projection**

Use `_completed_estimator_checkpoint` and require exact dataset, seed, method,
budget, task, utility, features, protocol, and schedule identity. Materialize
ablation-specific score artifacts with their own job IDs while recording zero
new physical fits and the inherited logical cost.

- [ ] **Step 4: Cache the remaining guidance variants**

Generate each variant's approved nested guided schedule. Use the selection
cache for coalition utilities, retain separate schedule digests, and report
only actual new fits/hits. Do not reuse scores across different guidance
signals.

- [ ] **Step 5: Revalidate the ISOLET paired utility path**

Assert accuracy and OVR-AUC use the same cache records and prediction digests,
the secondary utility has zero physical fit delta, and the existing paired
artifact remains the single authority for both utility jobs.

- [ ] **Step 6: Run ablation tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py tests/test_utility.py -q -p no:cacheprovider -k ablation
```

Expected: all selected tests pass.

- [ ] **Step 7: Review checkpoint**

Report projected dispatcher values, guidance cache deltas, ISOLET digest
evidence, and tests. Do not stage or commit.

---

### Task 6: Memoize NSGA-II masks without changing evolution

**Files:**
- Modify: `src/minppi_mofs/baselines/nsga2_fs.py`
- Modify: `tests/test_nsga2_hv.py`
- Modify: `tests/test_nsga2_quick.py`

**Interfaces:**
- Adds: `utility_evaluator: Callable[[tuple[str, ...]], float] | None = None`
- Adds to `NSGA2Result`: `logical_evaluation_count`, `unique_subset_count`, and
  `memoization_hit_count`.

- [ ] **Step 1: Write duplicate-mask memoization test**

Use a two-feature toy problem, where eight logical evaluations must contain
duplicates because only four masks exist:

```python
def test_duplicate_masks_are_fitted_once_without_changing_logical_count(
    monkeypatch, toy_data
):
    evaluator = RecordingSubsetEvaluator()
    result = nsga2_feature_selection(
        *toy_data,
        pop_size=4,
        max_gen=1,
        utility_evaluator=evaluator,
        seed=42,
    )
    assert result.logical_evaluation_count == 8
    assert result.unique_subset_count == len(set(evaluator.selected_features))
    assert result.unique_subset_count <= 4
    assert result.memoization_hit_count == 8 - result.unique_subset_count
```

Also run legacy and memoized versions with a deterministic evaluator and assert
identical masks, fitness values, HV history, convergence generation, and final
front.

- [ ] **Step 2: Run and confirm missing parameters**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nsga2_hv.py -q -p no:cacheprovider -k duplicate_masks
```

Expected: failure for the unexpected evaluator parameter or missing result
counters.

- [ ] **Step 3: Implement local selected-feature memoization**

Inside `nsga2_feature_selection`, create:

```python
utility_by_features: dict[tuple[str, ...], float] = {}
logical_evaluation_count = 0
memoization_hit_count = 0
```

Every `_evaluate_individual` call increments logical count. Canonicalize the
selected features in frozen original order. On first occurrence, call the
supplied evaluator or existing `evaluate_fixed_subset`; on duplicates, reuse
the scalar. Do not call RNG in the cache branch.

Extend the result type explicitly:

```python
@dataclass
class NSGA2Result:
    pareto_front: list
    hv: float
    runtime_s: float
    converged_gen: int
    hv_history: list[float]
    logical_evaluation_count: int
    unique_subset_count: int
    memoization_hit_count: int
```

Set `unique_subset_count = len(utility_by_features)` and require
`logical_evaluation_count == unique_subset_count + memoization_hit_count`.
Update every in-repository `NSGA2Result` construction.

- [ ] **Step 4: Add exact-budget and evolution tests**

For the formal relationship `population=40`, `evaluations=800`, assert the
derived generation count still produces exactly 800 logical evaluations. Keep
the existing initial-population-only test green. Assert memoization does not
change an all-unique-mask run.

- [ ] **Step 5: Run NSGA tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nsga2_hv.py tests/test_nsga2_quick.py -q -p no:cacheprovider
```

Expected: all NSGA-II tests pass.

- [ ] **Step 6: Review checkpoint**

Report logical evaluations, unique masks, physical fits, hit count, and
evolution equality. Do not stage or commit.

---

### Task 7: Connect NSGA-II to validation and final-test caches

**Files:**
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `tests/test_nsga2_hv.py`

**Interfaces:**
- Validation front search consumes a selection-cache evaluator.
- Final front evaluation consumes a separately authorized final-test evaluator.

- [ ] **Step 1: Write validation and test namespace tests**

```python
def test_nsga_search_uses_selection_cache_only(context):
    run_nsga_front_freeze(context)
    assert audit_test_reads(context) == ()
    assert cache_namespaces(context) == {"selection_validation"}


def test_nsga_final_front_opens_test_cache_after_freeze(context):
    freeze_both_fronts(context)
    run_nsga2_job(context)
    assert audit_test_reads(context) == ("final_nsga2_front_evaluation",)
    assert "final_test" in cache_namespaces(context)
```

- [ ] **Step 2: Run and confirm cache integration failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_icomp2026_stages.py -q -p no:cacheprovider -k 'nsga_search_uses_selection_cache or nsga_final_front_opens'
```

Expected: current NSGA path calls uncached fixed-subset evaluation.

- [ ] **Step 3: Supply the validation evaluator**

In `_freeze_nsga2_front`, open the selection cache and pass:

```python
utility_evaluator=lambda selected: cache.evaluate(set(selected), dataset.utility)
```

Keep the cardinality denominator, feature masks, population, exact evaluation
budget, RNG seed, HV reference point, and front schema unchanged. Add logical,
physical, unique-mask, and hit diagnostics.

- [ ] **Step 4: Supply the final-test evaluator**

Only after both validation fronts are frozen and reread, call the existing
audited test loader, construct train+validation and test hashes, and open
`final_test`. Reuse duplicate final-front subsets without changing masks or
front order.

- [ ] **Step 5: Run NSGA and stage tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nsga2_hv.py tests/test_nsga2_quick.py tests/test_icomp2026_stages.py -q -p no:cacheprovider -k nsga
```

Expected: all selected tests pass; logical evaluation count remains exact.

- [ ] **Step 6: Review checkpoint**

Report validation/test namespaces, audited test reason, logical/physical counts,
front equality, and tests. Do not stage or commit.

---

### Task 8: Build the scientific and speed acceptance command

**Files:**
- Create: `experiments/validate_grouped_acceleration.py`
- Create: `tests/test_validate_grouped_acceleration.py`

**Interfaces:**
- Produces CLI arguments: `--config-dir`, repeatable
  `--legacy-run name=absolute_path`, repeatable
  `--candidate-run name=absolute_path`, and `--report`
- Produces canonical JSON report with `passed`, `gates`, and `evidence`.
- Never loads held-out test data.

- [ ] **Step 1: Write report-schema and threshold tests**

```python
def test_classification_noninferiority_threshold():
    gate = classification_gate(legacy=0.900, candidate=0.895)
    assert gate.passed
    assert classification_gate(legacy=0.900, candidate=0.8949).passed is False


def test_speed_gate_requires_cold_group_runtime():
    assert speed_gate(
        dataset_id="spambase",
        legacy_runtime_s=300.0,
        candidate_runtime_s=120.0,
        cache_state="cold",
    ).passed
    with pytest.raises(ValueError, match="cold"):
        speed_gate(
            dataset_id="spambase",
            legacy_runtime_s=300.0,
            candidate_runtime_s=120.0,
            cache_state="warm",
        )
```

Add R² `0.01`, compression `0.02`, Spambase `2.5x`, Madelon `3.5x`,
prediction-digest equality, and zero-harmful Abalone convention tests.

- [ ] **Step 2: Run and confirm module failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_validate_grouped_acceleration.py -q -p no:cacheprovider
```

Expected: collection fails because the validation module is absent.

- [ ] **Step 3: Implement pure gate functions and canonical report**

Define immutable `GateResult` and these exact threshold rules:

```python
@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    detail: str


def classification_gate(legacy: float, candidate: float) -> GateResult:
    delta = candidate - legacy
    return GateResult(
        "classification_utility",
        delta >= -0.005 - 1e-12,
        delta,
        "delta >= -0.005",
        f"candidate={candidate:.12g}, legacy={legacy:.12g}",
    )


def r2_gate(legacy: float, candidate: float) -> GateResult:
    delta = candidate - legacy
    return GateResult(
        "r2",
        delta >= -0.01 - 1e-12,
        delta,
        "delta >= -0.01",
        f"candidate={candidate:.12g}, legacy={legacy:.12g}",
    )


def compression_gate(
    legacy_compression: float,
    candidate_compression: float,
    legacy_utility: float,
    candidate_utility: float,
) -> GateResult:
    compression_delta = candidate_compression - legacy_compression
    utility_delta = candidate_utility - legacy_utility
    passed = compression_delta >= -0.02 - 1e-12 or utility_delta >= 0.005 - 1e-12
    return GateResult(
        "compression",
        passed,
        compression_delta,
        "compression delta >= -0.02 or utility delta >= 0.005",
        f"utility_delta={utility_delta:.12g}",
    )


def speed_gate(
    dataset_id: str,
    legacy_runtime_s: float,
    candidate_runtime_s: float,
    cache_state: str,
) -> GateResult:
    if cache_state != "cold":
        raise ValueError("speed evidence must use a cold candidate cache")
    thresholds = {"spambase": 2.5, "madelon": 3.5}
    if dataset_id not in thresholds:
        raise ValueError("speed dataset is not authorized")
    factor = legacy_runtime_s / candidate_runtime_s
    return GateResult(
        f"{dataset_id}_speed",
        factor >= thresholds[dataset_id],
        factor,
        thresholds[dataset_id],
        f"legacy_s={legacy_runtime_s:.12g}, candidate_s={candidate_runtime_s:.12g}",
    )


def prediction_digest_gate(
    accuracy_digest: str, ovr_auc_digest: str
) -> GateResult:
    return GateResult(
        "isolet_prediction_digest",
        accuracy_digest == ovr_auc_digest,
        accuracy_digest,
        ovr_auc_digest,
        "accuracy and OVR-AUC must derive from one prediction array",
    )
```

Implement `abalone_exact_gate` over ordered feature-score mappings. Compute
score agreement as the fraction equal to exact. Compute harmful recall over
features with exact score `-1`; when exact has no harmful feature, define recall
as 1.0 only if the candidate has no false harmful feature. Pass only when the
candidate five-seed mean agreement and harmful recall are each at least their
legacy five-seed mean, within `1e-12`.

The CLI exits 0 only when every required gate passes, exits 1 for a valid failed
comparison, and exits 2 for malformed/missing evidence. Write the report
atomically.

- [ ] **Step 4: Enforce no-test-access**

The command reads estimator, removal selection, validation utility, bundle,
cache-stat, and environment artifacts only. Reject paths or artifact fields
identified as test results. Add a test monkeypatching `_task9_load_test` to fail
if called and assert the validation command still completes.

- [ ] **Step 5: Add Abalone and pilot evidence readers**

Read:

- legacy Abalone exact/random/guided/hybrid results for all five seeds;
- candidate nested equivalents;
- Spambase seed 42 cold group runtime and hybrid removal validation utility;
- Madelon seed 42 equivalent evidence; and
- ISOLET seed 42 paired prediction digests.

For Spambase and Madelon legacy evidence, the stopped formal run has estimator
scores but no removal stage. Derive the removed subset directly from the
validated maximum-budget hybrid score artifact, compute compression from the
frozen feature count, and evaluate that retained subset once on the frozen
training/validation split through the canonical modeling boundary. Apply the
same derivation to the grouped candidate so the utility comparison is paired
and does not require held-out test data.

Require exactly the evidence names `abalone`, `spambase`, `madelon`, and
`isolet` on each side. Validate every run internally against its own protocol
hash, then compare new/legacy data-manifest split hashes, dataset, seed, method,
budget, feature order, model parameters, utility, and resolved CUDA backend.
The hashes of a legacy and grouped protocol are expected to differ because
execution mode is frozen; do not require cross-mode protocol-hash equality.
Accept the stopped formal run as Spambase/Madelon legacy evidence only after its
persisted source-fingerprint document, resolved protocol, data manifest,
statuses, and canonical result records validate internally. Do not compare its
source fingerprint to the now-modified working tree and do not attempt to
resume it.

- [ ] **Step 6: Run acceptance-command tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_validate_grouped_acceleration.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 7: Review checkpoint**

Report threshold boundary tests, exit-code behavior, no-test evidence, and test
count. Do not stage or commit.

---

### Task 9: Full verification, cold pilots, smoke, and resume audit

**Files:**
- Modify: `ICOMP2026_experiment_redesign_submission_plan.md`
- Modify: `README.md`
- Generated, not committed by Codex: new run directories and acceptance report

**Interfaces:**
- Consumes all completed implementation tasks.
- Produces evidence required for explicit user approval of a new formal run.

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
Set-Location 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master'
& '..\venv\Scripts\python.exe' -m pytest tests -q -p no:cacheprovider
```

Expected: all tests pass with only the repository's accepted skip. Stop on any
failure; do not start GPU pilots.

- [ ] **Step 2: Run the missing legacy reference scopes**

Create a new legacy-mode validation run ID using a temporary copied config whose
only execution change is `mode: legacy`. Run all five Abalone seeds on
train/validation only. Create a second legacy scope for ISOLET seed 42 through
the paired utility artifact. Do not edit the stopped formal directory.

Expected: exact and hybrid references plus random/guided maximum-budget scores
exist for all five Abalone seeds, and ISOLET accuracy/OVR-AUC share the legacy
paired prediction source. Use the stopped formal run only for its already
complete Spambase seed 42 and Madelon seed 42 estimator evidence.

- [ ] **Step 3: Run cold grouped Abalone candidate**

Use a fresh grouped run ID and empty run-local cache. Run the same five seeds
and verify exact/hybrid agreement and candidate-vs-legacy exact gates.

- [ ] **Step 4: Run cold Spambase, Madelon, and ISOLET pilots**

Use separate new run IDs or a dedicated validation protocol that freezes only
the stated dataset/seed scope:

```text
Spambase seed 42
Madelon seed 42
ISOLET seed 42 shared utility
```

The resolved backend must be CUDA. Record cold group time, physical/logical
counts, removal validation utility/compression, and prediction digests.

- [ ] **Step 5: Generate the acceptance report**

Run:

```powershell
& '..\venv\Scripts\python.exe' experiments/validate_grouped_acceleration.py `
  --config-dir configs `
  --legacy-run 'abalone=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\legacy-abalone' `
  --legacy-run 'spambase=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_rerun\icomp2026-formal-20260723' `
  --legacy-run 'madelon=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_rerun\icomp2026-formal-20260723' `
  --legacy-run 'isolet=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\legacy-isolet-s42' `
  --candidate-run 'abalone=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\grouped-abalone' `
  --candidate-run 'spambase=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\grouped-spambase-s42' `
  --candidate-run 'madelon=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\grouped-madelon-s42' `
  --candidate-run 'isolet=C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\results\icomp2026_acceleration_validation\grouped-isolet-s42' `
  --report 'results/icomp2026_acceleration_validation/acceptance.json'
```

Expected: exit 0 and `"passed": true`. If a gate fails, preserve evidence,
report the exact gate, and stop before smoke/formal execution.

- [ ] **Step 6: Run end-to-end grouped smoke**

Use a new ID:

```powershell
& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-nested-full-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage all
```

Expected: public exit 0; every smoke manifest job succeeds; summary and figures
exist; denied test-access audit is clean; one GPU worker at a time.

- [ ] **Step 7: Run controlled interruption/resume smoke**

Use a different ID:

```powershell
& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-nested-resume-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage all `
  --stop-after-jobs 4
```

Expected first invocation: exit 75. Then run:

```powershell
& '..\venv\Scripts\python.exe' experiments/run_icomp2026_pipeline.py `
  --config-dir configs `
  --output-root results/icomp2026_rerun `
  --run-id grouped-nested-resume-smoke-v1 `
  --backend cuda `
  --smoke `
  --stage all `
  --resume
```

Expected: exit 0; completed artifacts from the first invocation remain
byte-identical; the interrupted bundle/cache resumes without identity conflict.

- [ ] **Step 8: Update project documentation**

Record:

- the approved grouped execution design and plan links;
- exact test totals;
- acceptance report path and every gate result;
- measured Spambase/Madelon speed factors;
- physical/logical fit reductions;
- smoke and resume run IDs;
- the old formal run's read-only baseline status; and
- the exact new formal command, marked as requiring explicit user approval.

- [ ] **Step 9: Final pre-formal review checkpoint**

Provide the user:

- full-suite output;
- acceptance report;
- smoke/resume status audit;
- estimated full runtime from measured cold throughput;
- disk-space estimate for cache and results;
- exact PowerShell formal command with a fresh run ID; and
- a statement that no Git write was performed.

Do not start the formal run until the user explicitly approves it.
