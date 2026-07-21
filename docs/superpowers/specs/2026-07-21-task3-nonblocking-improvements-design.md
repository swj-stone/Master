# Task 3 Non-blocking Improvements Design

Date: 2026-07-21

## Scope

This change improves validation and failure detection around the Task 3 utility ablation without changing MinPPI coalition sampling. The known blocking issue—utility-dependent adaptive sampling producing different coalition trajectories—remains deliberately out of scope.

## Changes

### 1. Validate explicit utility overrides

`AccFullCache.evaluate(S, utility=...)` will validate the selected metric against the cache task before computing it. Invalid classification/regression combinations must raise a clear `ValueError` instead of reaching a downstream shape error.

The existing default utility path remains unchanged.

### 2. Add a full-run regression test

Add an integration-level test that runs `estimate_minppi_random_mc` twice over one shared prediction cache, once through an accuracy `UtilityView` and once through an OVR-AUC `UtilityView`.

The test will use deterministic synthetic multiclass data known to trigger utility-dependent trajectory divergence. It will assert that the second run adds prediction-cache entries. This records the current blocking behavior explicitly; it is not a completion test for the eventual fixed-schedule design.

### 3. Fail closed in the E4 runner

After the second utility pass, E4 will compare the prediction-cache key set before and after the run. If new coalition predictions were created, it will raise `RuntimeError` and refuse to save results or metadata claiming identical predictions.

Comparing key sets, rather than only cache lengths, also detects replacement or divergence with the same number of entries.

## Tests

The implementation will follow TDD:

1. Add a failing test for invalid `evaluate(..., utility=...)` overrides.
2. Add a regression test demonstrating full-run trajectory divergence.
3. Extract a small E4 cache-consistency guard and add a failing unit test for added or changed keys.
4. Implement the minimum production changes required for the tests to pass.
5. Run Task 3 tests, the complete test suite, Ruff, compilation, and `git diff --check`.

## Non-goals

- No fixed coalition schedule.
- No paired multi-utility MinPPI estimator.
- No formal E4 experiment run.
- No Git commit.
