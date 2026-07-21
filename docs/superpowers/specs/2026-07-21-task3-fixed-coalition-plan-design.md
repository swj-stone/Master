# Task 3 Fixed Coalition Plan Design

Date: 2026-07-21

## Goal

Complete Task 3 by guaranteeing that utility ablations evaluate the same coalition-feature comparisons and therefore consume exactly the same cached model predictions.

## Problem

The shared-coalition implementation currently samples from the same seed twice, but its control flow depends on utility outcomes. A feature already assigned MinPPI `-1` is skipped, and supplemental sampling stops when a negative marginal is found. Accuracy and OVR-AUC can therefore advance the random generator differently and evaluate different coalitions.

The existing E4 cache-key guard detects this divergence and refuses to save results, but it does not prevent divergence.

## Architecture

### Immutable shared-coalition plan

`src/minppi_mofs/min_ppi.py` will define:

```python
SharedCoalitionStep = tuple[frozenset[str], tuple[str, ...]]
SharedCoalitionPlan = tuple[SharedCoalitionStep, ...]

def make_shared_coalition_plan(
    features: Sequence[str],
    *,
    K: int,
    per_coalition: int,
    seed: int,
) -> SharedCoalitionPlan: ...
```

Each step stores one coalition and the absent features whose marginal contributions must be evaluated against it. The tuple/frozenset representation makes the schedule immutable after construction.

Plan generation will reproduce the current shared sampling budget:

1. Generate `K` base coalitions.
2. Select at most `per_coalition` absent features for each base coalition.
3. Track feature coverage.
4. For features with fewer than two base evaluations, generate enough dedicated steps to reach five total evaluations.

Generation never reads utility values and never stops early.

### Plan execution

`estimate_minppi_random_mc` will accept:

```python
shared_plan: Optional[SharedCoalitionPlan] = None
```

When `shared_coalitions=True`, it executes the supplied plan or creates one from the existing `K`, `per_coalition`, and `seed` arguments. Every planned marginal is evaluated. Once MinPPI reaches `-1`, later planned comparisons still run so another utility view consumes the identical prediction keys; the stored minimum remains `-1`.

Supplying `shared_plan` while `shared_coalitions=False`, or supplying a plan containing unknown features or a feature already inside its coalition, raises `ValueError` before evaluation.

Per-feature Monte Carlo mode remains unchanged.

### E4 integration

`experiments/e4_isolet_utility_ablation.py` will generate one plan before either utility run and pass the same object to both calls. The existing cache-key guard remains as a fail-closed invariant.

After the accuracy run, the OVR-AUC run must perform zero new model fits and add zero prediction-cache keys.

## Compatibility

- Existing calls without `shared_plan` remain valid.
- `K`, `per_coalition`, and `seed` retain their current meaning.
- Removing early exits may perform extra cache lookups/evaluations, but it does not change the mathematical minimum once `-1` has been observed.
- Exact and per-feature Monte Carlo algorithms are out of scope.
- No formal experiment is run and no Git commit is created.

## Tests

1. Same inputs and seed produce identical immutable plans.
2. Plan steps contain valid absent features and provide the required supplemental coverage.
3. Invalid plans are rejected before model evaluation.
4. Two complete utility runs using the same plan produce identical cache-key sets.
5. A fit spy confirms the second utility run performs zero fits.
6. The E4 cache-key guard accepts the completed paired run.
7. Existing MinPPI, utility, protocol, lint, and compilation checks remain green.
