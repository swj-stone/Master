# Task 4 Feature Graph and Guidance Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a strict weighted correlation MST and migrate Algorithm B to an honest `mi_correlation_guided` API while retaining deprecated compatibility entry points.

**Architecture:** Keep the real ShapG baseline separate. Repair its shared graph helper, expose canonical MI/correlation guidance functions from the existing integration module, route Algorithm B through canonical names, and update active experiments and the conference paper.

**Tech Stack:** Python 3, pandas, NumPy, NetworkX, scikit-learn, pytest, Ruff, LaTeX source

## Global Constraints

- Preserve all current uncommitted Task 3 changes.
- Do not rename the real ShapG baseline.
- Do not run formal experiments.
- Do not commit changes to Git.
- Active code and paper text must use `mi_correlation_guided`; old names may appear only in compatibility wrappers/tests and the real ShapG baseline.

---

### Task 1: Make the feature graph a strict spanning tree

**Files:**
- Modify: `tests/test_feature_graph.py`
- Modify: `src/minppi_mofs/baselines/shapg.py:65-118`

**Interfaces:**
- Preserves: `_create_minimal_edge_graph(W, reverse=True) -> (adj, red)`
- Guarantees: for `n > 1`, `adj` is connected and has exactly `n - 1` undirected edges

- [x] Write failing tests changing edge-count assertions from `>= n-1` to `== n-1` and asserting the adjacency graph is a tree.
- [x] Run `pytest tests/test_feature_graph.py -k "SpanningTree or MinimalEdge" -q`; confirm failure on the cycle-edge implementation.
- [x] Move adjacency and reduced-weight writes inside the `if union(i, j)` block.
- [x] Re-run the focused tests; confirm pass.

---

### Task 2: Preserve correlation weights and neighbor order

**Files:**
- Modify: `tests/test_feature_graph.py`
- Modify: `tests/test_shapg_integration.py`
- Modify: `src/minppi_mofs/baselines/shapg.py:121-140`
- Modify: `src/minppi_mofs/shapg_integration.py`

**Interfaces:**
- Produces: weighted `nx.Graph` edges whose `weight` equals the accepted absolute correlation
- Produces: `get_correlation_neighbors(...)` ordered by descending edge weight

- [x] Add failing tests that compare graph weights with `red`, reject the all-one-weight behavior, preserve zero-weight accepted edges, and verify neighbor lists follow descending weights.
- [x] Run the focused graph/neighbor tests; confirm failure because `build_feature_graph` used binary adjacency.
- [x] Build NetworkX edges manually from accepted adjacency entries while assigning weights from `red`; add all feature nodes explicitly.
- [x] Introduce canonical `get_correlation_neighbors`; retain `get_correlated_neighbors` as a warning wrapper.
- [x] Re-run focused tests; confirm pass.

---

### Task 3: Select MI implementation from explicit task

**Files:**
- Modify: `tests/test_shapg_integration.py`
- Modify: `src/minppi_mofs/shapg_integration.py`

**Interfaces:**
- Extends: `compute_mi_guidance_signal(X, y, *, task, seed=42, ...)`
- Produces: `build_mi_correlation_guidance(X, y, *, task, top_k=10, seed=42)`

- [x] Add failing spy tests proving 26-class data uses `mutual_info_classif`, regression uses `mutual_info_regression`, and invalid task raises before either estimator runs.
- [x] Run `pytest tests/test_shapg_integration.py -k "ExplicitTask" -q`; confirm failure because `task` was not implemented.
- [x] Validate task against `binary`, `multiclass`, and `regression`; remove dtype/cardinality inference.
- [x] Add canonical build/rank/filter functions and deprecated legacy wrappers.
- [x] Update existing tests to canonical calls and add warning tests for legacy wrappers.
- [x] Run the complete integration test file; confirm pass.

---

### Task 4: Canonical Algorithm B names and repository migration

**Files:**
- Modify: `tests/test_min_ppi.py`
- Modify: `src/minppi_mofs/min_ppi.py`
- Modify: `src/minppi_mofs/__init__.py`
- Modify: `experiments/eval_harmful_utility.py`
- Modify: `experiments/run_mofs_verify.py`
- Modify: `experiments/run_pareto_eval.py`
- Modify: `experiments/make_figures.py`
- Modify: `experiments/make_paper_figures.py`
- Modify: `conference-paper/paper/icomp2026_conference.tex`

**Interfaces:**
- Produces: `estimate_minppi_mi_correlation_guided(..., correlation_neighbors=None, mi_scores=None)`
- Produces: dispatcher method `mi_correlation_guided`
- Preserves with warnings: `estimate_minppi_mi_guided`, method `mi_guided`, `shapg_neighbors`, and `shapg_scores`

- [x] Add failing canonical estimator/dispatcher tests and deprecated alias warning tests.
- [x] Implement the canonical estimator, canonical hybrid/main parameters, conflict detection when both canonical and legacy parameters are supplied, and compatibility wrappers.
- [x] Export canonical guidance and estimator names from `minppi_mofs.__init__`.
- [x] Update all active experiments to canonical imports, variables, parameters, method string, and explicit task.
- [x] Replace Algorithm B “ShapG-guided” claims in the paper and figure labels with “MI-correlation-guided”; retain genuine ShapG baseline references.
- [x] Search the repository for active misleading names and inspect every remaining hit.
- [x] Run Task 4 tests, then the complete test suite.
- [x] Run Ruff, Python compilation, `git diff --check`, and final terminology search.

## Verification Commands

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_feature_graph.py tests\test_shapg_integration.py tests\test_min_ppi.py -q
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\python.exe -m ruff check src experiments tests
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\baselines\shapg.py src\minppi_mofs\shapg_integration.py src\minppi_mofs\min_ppi.py src\minppi_mofs\__init__.py
git diff --check
```
