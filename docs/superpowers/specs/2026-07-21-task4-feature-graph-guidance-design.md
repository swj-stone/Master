# Task 4 Feature Graph and Guidance Naming Design

Date: 2026-07-21

## Goal

Complete Task 4 by making the correlation graph a strict weighted maximum spanning tree, selecting the MI estimator from an explicit task, and naming Algorithm B according to its actual MI-plus-correlation implementation.

## 1. Strict weighted feature graph

`_create_minimal_edge_graph` will write an edge to the adjacency and reduced-weight matrices only when Union-Find accepts it. For `n > 1`, the result must be connected and contain exactly `n - 1` undirected edges.

The reduced matrix stores absolute correlation strength for accepted edges. `build_feature_graph` will construct the NetworkX graph from that weighted matrix and explicitly preserve all feature nodes, including the single-feature case.

The real ShapG baseline remains named ShapG and continues to use this graph helper.

## 2. Canonical MI-correlation guidance API

The implementation remains in `shapg_integration.py` for import compatibility, but its canonical public functions become:

```python
compute_mi_guidance_signal(X, y, *, task, seed=42)
get_correlation_neighbors(X, *, top_k=10, corr_method="pearson", seed=42)
build_mi_correlation_guidance(X, y, *, task, top_k=10, seed=42)
rank_by_mi(scores)
filter_top_by_mi(X, y, *, task, top_k=50, seed=42)
```

`task` must be one of `binary`, `multiclass`, or `regression`. Binary and multiclass tasks use `mutual_info_classif`; regression uses `mutual_info_regression`. The implementation will no longer infer task type from label dtype or number of distinct values.

Legacy helper names remain as wrappers that emit `DeprecationWarning`:

- `get_correlated_neighbors`
- `build_guidance_from_data`
- `rank_by_shapg`
- `filter_top_by_shapg`

The already accurate name `compute_mi_guidance_signal` is retained. Its formerly ignored ShapG-specific `depth` and `m` keywords remain temporarily accepted for compatibility and emit `DeprecationWarning` when explicitly supplied; canonical calls omit them.

## 3. Canonical Algorithm B API

The canonical estimator becomes:

```python
estimate_minppi_mi_correlation_guided(
    ...,
    correlation_neighbors=None,
    mi_scores=None,
)
```

The canonical dispatcher method is `method="mi_correlation_guided"`. Hybrid estimation and all internal code use `correlation_neighbors` and `mi_scores`.

Compatibility behavior:

- `estimate_minppi_mi_guided` remains a deprecated wrapper.
- `method="mi_guided"` remains a deprecated dispatcher alias.
- Legacy `shapg_neighbors` and `shapg_scores` are accepted only through compatibility paths and emit `DeprecationWarning`.
- The real ShapG baseline under `minppi_mofs.baselines.shapg` is unchanged in name.

## 4. Experiment and paper migration

All active experiment runners will import and call `build_mi_correlation_guidance`, pass the explicit dataset task, use canonical variable names, and request `method="mi_correlation_guided"`.

The conference paper will replace Algorithm B claims of “ShapG-guided” with “MI-correlation-guided”. Its theorem and structural condition will refer to correlation-graph neighbors and MI-ranked baseline features. Mentions of ShapG as a separate baseline or related method remain unchanged.

## 5. Tests

The change will follow four TDD cycles:

1. Strict MST tests fail on the current cycle-edge implementation, then pass after the Union-Find write-order fix.
2. Weighted-graph and neighbor-order tests fail because current weights are all one, then pass after graph construction uses the reduced-weight matrix.
3. Explicit-task tests prove 26-class labels call `mutual_info_classif`, regression calls `mutual_info_regression`, and invalid tasks fail clearly.
4. Canonical-name tests cover the new estimator, dispatcher, parameters, exports, deprecated aliases, active experiment source, and paper terminology.

Final verification includes the Task 4 test group, the complete test suite, Ruff, compilation, `git diff --check`, and a repository search for misleading active ShapG-guidance names.

## Non-goals

- No change to the actual ShapG baseline name or ranking experiments.
- No formal experiment execution.
- No new Algorithm B mathematical guarantee beyond honest terminology.
- No Git commit.
