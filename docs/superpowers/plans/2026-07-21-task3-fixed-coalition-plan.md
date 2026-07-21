# Task 3 Fixed Coalition Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shared-coalition MinPPI utility-independent by generating and reusing one immutable coalition-feature evaluation plan.

**Architecture:** Add a deterministic plan generator and validator to `min_ppi.py`. Shared Monte Carlo executes every step without utility-dependent skip/break branches. E4 constructs one plan and supplies it to both `UtilityView` runs, while the existing cache-key guard verifies the invariant.

**Tech Stack:** Python 3, NumPy, pandas, XGBoost, pytest, Ruff

## Global Constraints

- Preserve existing calls that do not pass `shared_plan`.
- Do not change exact or per-feature Monte Carlo behavior.
- Do not run the formal E4 experiment.
- Do not commit changes to Git.
- Both utilities must reuse the same prediction-cache keys, with zero fits on the second pass.

---

### Task 1: Define and test immutable shared plans

**Files:**
- Modify: `tests/test_min_ppi.py`
- Modify: `src/minppi_mofs/min_ppi.py`

**Interfaces:**
- Produces: `SharedCoalitionStep`, `SharedCoalitionPlan`
- Produces: `make_shared_coalition_plan(features, *, K, per_coalition, seed) -> SharedCoalitionPlan`

- [ ] **Step 1: Add failing plan-generation tests**

Test that the same features and seed return equal tuple-based plans, all evaluated features are absent from their coalition, and every feature receives either at least two base evaluations or the supplemental total of five.

- [ ] **Step 2: Verify RED**

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_min_ppi.py -k "SharedCoalitionPlan" -q
```

Expected: import failure because `make_shared_coalition_plan` does not exist.

- [ ] **Step 3: Implement the immutable generator**

Use one `np.random.default_rng(seed)`, preserve feature order from the input sequence, produce `K` base steps, then add utility-independent supplemental steps for features with fewer than two base hits until they have five total hits.

- [ ] **Step 4: Verify GREEN**

Run the focused command and expect all plan-generation tests to pass.

---

### Task 2: Execute and validate a supplied plan

**Files:**
- Modify: `tests/test_min_ppi.py`
- Modify: `src/minppi_mofs/min_ppi.py:125-249`
- Modify: `src/minppi_mofs/__init__.py`

**Interfaces:**
- Extends: `estimate_minppi_random_mc(..., shared_plan: Optional[SharedCoalitionPlan] = None)`
- Produces: validation rejecting non-shared use, unknown coalition features, unknown evaluated features, and evaluated features already present in their coalition

- [ ] **Step 1: Add failing plan-validation and execution tests**

Pass an explicit valid plan and assert the estimator evaluates every planned marginal. Pass invalid plans and assert `ValueError` occurs before cache evaluation. Pass a plan with `shared_coalitions=False` and assert `ValueError`.

- [ ] **Step 2: Verify RED**

Run the focused validation tests. Expected: failure because `shared_plan` is not accepted.

- [ ] **Step 3: Implement fixed execution**

When shared mode is selected, use the supplied plan or call `make_shared_coalition_plan`. Validate the full plan before evaluation. Iterate all planned features without checking current MinPPI and without breaking on marginal `-1`. Keep the minimum and hit counts exactly as before.

- [ ] **Step 4: Export the generator and types**

Expose `SharedCoalitionPlan`, `SharedCoalitionStep`, and `make_shared_coalition_plan` from `minppi_mofs.__init__`.

- [ ] **Step 5: Verify GREEN and existing MinPPI tests**

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_min_ppi.py -q
```

Expected: all tests pass.

---

### Task 3: Use one plan in E4 and prove zero second-pass fits

**Files:**
- Modify: `tests/test_utility.py`
- Modify: `experiments/e4_isolet_utility_ablation.py`

**Interfaces:**
- Consumes: `make_shared_coalition_plan`
- Preserves: `_assert_same_prediction_keys(before, after)`

- [ ] **Step 1: Change the full-run regression test to the desired behavior**

Generate one plan, pass it to both complete utility runs, assert the cache-key sets are equal, assert the guard accepts them, and spy on `cache._model.fit` to assert the OVR-AUC pass makes zero fit calls.

- [ ] **Step 2: Verify RED**

Run the full-run test. Expected: failure because `shared_plan` is not yet wired into E4/current estimator behavior.

- [ ] **Step 3: Integrate the plan into E4**

Generate the plan once from `X_tr.columns`, `K_ABLATION`, `PER_COALITION`, and `SEED`; pass the same object to both estimator calls. Record plan step count in metadata.

- [ ] **Step 4: Run Task 3 verification**

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_utility.py tests\test_min_ppi.py tests\test_voting_game.py -q
```

Expected: all tests pass and the full-run test records zero second-pass fits.

- [ ] **Step 5: Run final verification**

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\python.exe -m ruff check src\minppi_mofs\min_ppi.py src\minppi_mofs\voting_game.py src\minppi_mofs\__init__.py experiments\e4_isolet_utility_ablation.py tests\test_min_ppi.py tests\test_utility.py
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\min_ppi.py src\minppi_mofs\voting_game.py src\minppi_mofs\__init__.py experiments\e4_isolet_utility_ablation.py tests\test_min_ppi.py tests\test_utility.py
git diff --check
```

Expected: every command exits with status 0; only the existing `.pytest_cache` permission warning is accepted.
