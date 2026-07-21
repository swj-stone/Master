# Task 3 Non-blocking Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit utility-override validation and make the E4 runner refuse results when the two utility passes do not reuse exactly the same prediction-cache keys.

**Architecture:** Keep `AccFullCache` as the task-aware validation boundary. Add a small pure cache-key guard to the E4 runner and exercise it both directly and through two complete deterministic MinPPI runs. Do not change coalition sampling.

**Tech Stack:** Python 3, pytest, pandas, NumPy, XGBoost, Ruff

## Global Constraints

- Do not modify MinPPI sampling behavior.
- Do not run the formal E4 experiment.
- Do not commit changes to Git.
- Preserve the shared raw-prediction cache and `UtilityView` API.

---

### Task 1: Validate utility overrides at evaluation time

**Files:**
- Modify: `tests/test_utility.py`
- Modify: `src/minppi_mofs/voting_game.py:201-208`

**Interfaces:**
- Consumes: `_validate_task_utility(task: str, utility: str) -> None`
- Produces: `AccFullCache.evaluate(S, utility=...)` raising `ValueError` for incompatible task/metric pairs before metric computation

- [ ] **Step 1: Write failing parameterized tests**

Add tests that construct valid classification and regression caches, then call `evaluate` with an incompatible override (`r2` for classification and `accuracy` for regression). Assert a clear `ValueError`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_utility.py -k "EvaluateRejects" -q
```

Expected: both tests fail because `evaluate` reaches `compute_utility` without validating the override.

- [ ] **Step 3: Implement the validation boundary**

Resolve the metric, call `_validate_task_utility(self._task, metric)`, then retrieve predictions and call `compute_utility`.

- [ ] **Step 4: Verify GREEN**

Run the same focused command and expect both tests to pass.

---

### Task 2: Fail closed when E4 utility passes diverge

**Files:**
- Modify: `tests/test_utility.py`
- Modify: `experiments/e4_isolet_utility_ablation.py:123-145`

**Interfaces:**
- Produces: `_assert_same_prediction_keys(before: set[tuple[str, ...]], after: set[tuple[str, ...]]) -> None`
- Raises: `RuntimeError` listing the number of added and missing coalition keys when the sets differ

- [ ] **Step 1: Write failing guard tests**

Test that identical key sets are accepted and that added or missing keys raise `RuntimeError` with `added=` and `missing=` counts.

- [ ] **Step 2: Verify RED**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_utility.py -k "PredictionKeyGuard" -q
```

Expected: collection/import failure because the guard does not exist.

- [ ] **Step 3: Implement and invoke the guard**

Define the pure helper in E4. Snapshot `set(cache._preds)` before the OVR pass, snapshot it again afterward, and call the helper before reporting or saving results.

- [ ] **Step 4: Verify GREEN**

Run the focused guard tests and expect them to pass.

---

### Task 3: Cover complete two-utility MinPPI execution

**Files:**
- Modify: `tests/test_utility.py`

**Interfaces:**
- Consumes: `AccFullCache`, `UtilityView`, `estimate_minppi_random_mc`, and `_assert_same_prediction_keys`
- Produces: deterministic integration coverage showing that the guard rejects the currently known utility-dependent trajectory divergence

- [ ] **Step 1: Add deterministic synthetic multiclass integration test**

Generate the known counterexample with NumPy seed 2, 75 rows, 8 features, a 50/25 train-validation split, and two complete shared-coalition runs using seed 42, `K=8`, and `per_coalition=2`. Snapshot cache keys between the accuracy and OVR-AUC runs and assert `_assert_same_prediction_keys` raises.

- [ ] **Step 2: Run the integration test**

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_utility.py -k "FullRunCacheConsistency" -q
```

Expected: pass only when the E4 guard detects the divergent full-run key sets.

- [ ] **Step 3: Run complete verification**

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\python.exe -m ruff check src\minppi_mofs\voting_game.py experiments\e4_isolet_utility_ablation.py tests\test_utility.py
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\voting_game.py experiments\e4_isolet_utility_ablation.py tests\test_utility.py
git diff --check
```

Expected: all commands exit with status 0; the only accepted pytest warning is the existing `.pytest_cache` permission warning.
