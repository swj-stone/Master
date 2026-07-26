# Grouped nested experiment acceleration design

## Context

The ICOMP 2026 formal pipeline expands the frozen protocol into 1,338 jobs:

- 1 environment-validation job;
- 35 preprocessing jobs;
- 550 estimator jobs;
- 245 removal jobs;
- 175 alpha jobs;
- 70 NSGA-II jobs;
- 260 ablation jobs;
- 1 summarization job; and
- 1 figure-generation job.

The formal run `icomp2026-formal-20260723` is preserved as an immutable
pre-optimization baseline. It is currently stopped, with no live Python worker,
after 97 of 550 estimator jobs succeeded. One stale estimator status entry says
`running`; it is not evidence of a live process. The last durable update was
2026-07-25.

Those 97 jobs performed approximately 835,000 XGBoost fits and accumulated
approximately 38.6 hours of measured estimator runtime. The completed Spambase
jobs used 627,254 fits over 28.2 hours. Every estimator job creates a new
in-memory `AccFullCache`, so cache contents are discarded at the job boundary.
The same dataset and seed are consequently evaluated again for every method,
budget, and later pipeline stage.

The current implementation also makes budget-specific random streams. A
`K=25` job is not a reusable continuation of the `K=10` job because one global
random generator is consumed feature by feature, and the number of draws for an
earlier feature changes the stream seen by later features. Algorithm B has the
same cross-budget coupling. The five budgets therefore repeat work instead of
forming checkpoints on one statistically valid master schedule.

The user selected scientific non-inferiority rather than byte-for-byte
compatibility with the old stochastic outputs. Data splits, model parameters,
seeds, maximum budgets, utilities, stage coverage, and held-out-test policy must
remain unchanged. The random schedule may be redesigned into deterministic,
nested prefixes when the new design is validated as non-inferior.

## Goals

The design must:

1. reduce physical XGBoost fits without reducing any configured experimental
   budget;
2. retain all seven datasets, five seeds, five estimator budgets, model
   settings, removal controls, alpha values, NSGA-II evaluations, and ablations;
3. make every lower estimator budget a strict prefix of the maximum-budget
   schedule;
4. reuse mathematically identical estimator components instead of fitting them
   again for hybrid, removal, alpha, and dispatcher ablations;
5. cache coalition utilities safely across job and stage boundaries;
6. preserve Task 3's guarantee that different classification utilities are
   computed from the same model probabilities;
7. preserve the held-out-test access boundary;
8. provide visible progress, ETA, cache statistics, and fine-grained resume;
9. retain the canonical job manifest and artifact paths needed by audit and
   paper-generation code; and
10. pass explicit scientific non-inferiority and speed gates before a new formal
    run is authorized.

## Non-goals

This change does not:

- reduce the number of datasets, seeds, budgets, controls, alpha values, NSGA-II
  logical evaluations, or ablation values;
- reduce `n_estimators`, tree depth, or other frozen XGBoost parameters;
- adopt adaptive CPU/GPU routing in the initial implementation;
- allow simultaneous GPU model workers;
- import incomplete in-memory state from the old formal run;
- mix old and new formal artifacts;
- weaken source-fingerprint, protocol-hash, manifest, status, dependency, or
  denied-test-access checks;
- access the held-out test set during selection or optimization; or
- stage, commit, reset, clean, branch, or otherwise manage Git.

## Considered approaches

### Persistent cache only

A run-scoped disk cache could preserve the existing budget-specific stochastic
plans and reuse only coalitions that happen to be identical. This has the
smallest semantic change but retains most repeated work across budgets and does
not exploit the mathematical duplication in hybrid jobs. It is insufficient as
the primary deadline-oriented solution.

### Grouped nested execution with persistent coalition utilities

This is the selected approach. One internal computation group produces every
estimator method and budget for a dataset/seed pair. Random and guided searches
run to the maximum budget once and emit lower-budget checkpoints. Hybrid results
are derived from their mathematical components. A persistent utility cache
supports interruption recovery and reuse by later stages.

Using already completed formal results as an empirical cost model, this
structure reduces Spambase estimator fits by approximately 3.0 times and
Madelon seed 42 fits by approximately 5.26 times before counting cross-stage or
NSGA-II reuse.

### Adaptive CPU/GPU scheduling

Small independent XGBoost fits may run faster as parallel CPU processes than as
serialized CUDA launches. However, CPU and CUDA histogram implementations may
produce slightly different probabilities, and concurrent scheduling adds a new
reproducibility and resource-control surface. This remains an optional,
separately benchmarked future optimization and is not enabled by this design.

## Computation-group authority

The internal estimator computation key is:

```text
(
    dataset_id,
    seed,
    prepared_split_hash,
    ordered_feature_hash,
    model_configuration_hash,
    backend,
    task,
)
```

The public manifest remains job-oriented. Existing estimator jobs and their
canonical output paths are not collapsed or removed. On the first pending
estimator job for a computation group, the executor ensures that a recoverable
group bundle exists. It builds or resumes the bundle as necessary, then
materializes the current job's canonical score artifact and `ResultRecord`.
Later jobs in that group validate and read the completed bundle.

The bundle must contain enough provenance to prove that it belongs to the
current frozen protocol and prepared split. It must not be accepted merely
because a file with the expected name exists.

## Deterministic nested schedules

### Independent feature streams

Each feature receives a deterministic random stream derived from:

```text
SeedSequence(
    protocol_seed,
    stable_algorithm_identifier,
    feature_position,
    schedule_schema_version,
)
```

Feature names and positions come from the frozen prepared feature order.
Changing the budget does not change another feature's draws.

Random Monte Carlo and guided search use different stable algorithm identifiers.
Their schedules remain method-specific while being reproducible across process
restarts.

### Budget checkpoints

The master schedule is generated for:

```text
K_max = max(estimator_budgets) = 200
```

The executor records scores after the first 10, 25, 50, 100, and 200 trials for
each feature. Therefore:

- budget 10 uses exactly the first 10 eligible samples;
- budget 25 adds samples 11 through 25;
- budget 50 adds samples 26 through 50;
- budget 100 adds samples 51 through 100; and
- budget 200 adds samples 101 through 200.

If a feature reaches the minimum possible marginal value `-1`, later samples
cannot change its MinPPI. The executor may stop physically evaluating that
feature while materializing `-1` at all later checkpoints. This is an exact
monotonicity optimization, not a budget reduction.

For tiny feature universes, the existing exact-enumeration cap remains
authoritative. Logical counts must report the protocol-defined eligible
evaluations even when a score is already mathematically final.

### Exact and hybrid derivation

Exact MinPPI is independent of the nominal estimator budget. For an eligible
dataset/seed pair it is computed once and materialized for all five exact jobs.
Manifest eligibility remains governed by `exact_max_features=16`.

Hybrid is derived without new model fits:

- for at most 20 selected features, hybrid reuses exact;
- for 21 through 75 selected features, hybrid reuses guided;
- above 75 selected features, hybrid takes the feature-wise minimum of random
  and guided checkpoint scores.

When the selected dimension is 17 through 20, the group may compute exact once
as an internal hybrid component even though the manifest contains no public
exact-method job. It must not materialize an unauthorized exact result.

The derivation is tested against a direct hybrid execution using the same new
schedules.

## Coalition utility cache

### Storage model

The pipeline gains a run-scoped `CoalitionUtilityCache` backed by SQLite and an
in-memory hot layer. SQLite transactions provide atomic checkpoints and make
partial group work reusable after interruption. The default commit interval is
50 new physical fits.

The persistent cache stores task-appropriate utility scalars rather than every
complete validation probability matrix. Retaining all probabilities for tens of
thousands of ISOLET coalitions would make the cache unacceptably large.

For each nonempty coalition, the evaluator performs one model fit and one
prediction call. Before releasing the prediction array, it computes every
protocol-authorized utility for that task:

- binary classification: accuracy, ROC-AUC, and PR-AUC;
- multiclass classification: accuracy and OVR-AUC; and
- regression: R-squared and the existing normalized RMSE utility
  `1 / (1 + RMSE)`.

The row also stores the prediction shape, dtype, and SHA-256 digest. The digest
is audit evidence that multiple utilities came from one prediction array; it is
not used as a cache key.

The empty coalition remains a deterministic no-fit case based on training-label
priors or the training-target mean.

### Cache identity

A cache key includes:

- cache schema version;
- protocol hash;
- dataset ID and seed;
- evaluation namespace;
- hashes of training features, training labels, evaluation features, and
  evaluation labels;
- ordered feature-universe hash;
- canonical sorted coalition;
- task and class definition;
- complete model parameters;
- backend and relevant library versions; and
- model random seed.

Utility is not part of the model/prediction identity. One cache row can expose
multiple utility values computed from the same predictions.

### Integrity rules

Rows use canonical serialization and a checksum. On open and read, the cache
validates its schema, run authority, keys, scalar finiteness, prediction
metadata, and checksum. A mismatch or corrupt row fails closed. The system does
not silently reuse, delete, or repair suspect scientific state.

Bundle publication is atomic. An incomplete bundle cannot be mistaken for a
completed group. Cache commits may precede bundle completion because their
content-addressed rows are independently validated.

### Test isolation

Training/validation selection and final test evaluation use different
namespaces and different data hashes. The test namespace cannot be opened
through a selection-stage context. Existing audited test-access authorization
remains required before any test data are loaded.

Caching cannot turn a denied test access into an allowed read, and
training/validation cache rows cannot satisfy a test lookup.

## Cross-stage reuse

The following reuse is authorized when all cache and artifact identities match:

- the maximum-budget hybrid estimator supplies the MinPPI component of the
  removal selection plan;
- cached validation coalition utilities supply the alpha grid, where only the
  winning threshold changes;
- maximum-budget estimator scores supply matching dispatcher ablations;
- the `mi_correlation` guidance ablation reuses the matching guided estimator
  schedule and scores;
- other guidance variants reuse any identical coalition utilities while
  retaining their distinct guidance schedules; and
- NSGA-II validation evaluations reuse identical feature-subset utilities.

Removal baselines remain methodologically distinct. Their model-based coalition
evaluations may use the shared utility cache only when their subset, model,
split, task, utility source, backend, and seed identities are exactly equal.

Final test evaluations may reuse identical test-scope subsets only after all
selection artifacts governing them are frozen and authorized.

## NSGA-II memoization

NSGA-II continues to perform the frozen 800 logical evaluations for each
variant. Population generation, tournament selection, crossover, mutation,
non-dominated sorting, crowding distance, and hypervolume calculations retain
their existing random sequence and behavior.

The objective evaluator canonicalizes each feature mask to its selected-feature
tuple and queries the coalition utility cache. Duplicate masks receive the
identical cached objective without another XGBoost fit. This optimization must
not skip a logical evaluation or consume additional random draws.

The two variants may share a validation utility only when their selected
original feature sets are identical, even if their mask dimensions differ.

## Metrics and runtime accounting

Caching makes the old single `model_fit_count` insufficient. New artifacts
record:

- `logical_evaluation_count`: evaluations required by the algorithmic schedule;
- `logical_unique_coalition_count`: unique coalitions needed by that method and
  checkpoint;
- `physical_model_fit_count`: XGBoost fits actually performed while producing
  the recorded scope;
- `cache_hit_count`: successful lookups avoiding a fit;
- `group_runtime_s`: actual wall time to produce or resume the group bundle;
- component and checkpoint elapsed times;
- schedule digest; and
- prediction digests where shared-utility provenance is required.

For backward compatibility, `model_fit_count` is defined as the logical unique
nonempty coalition count for that method/checkpoint. It is deterministic and
does not depend on manifest execution order. Actual hardware cost and
acceleration claims use `physical_model_fit_count` and measured wall time.

Summaries and paper tables must label logical algorithmic cost separately from
physical cached execution cost. No runtime is fabricated or allocated to a job
that did not physically perform it.

## Progress and recovery

The group builder emits a concise progress line at least every 30 seconds and at
budget checkpoints. It includes:

- dataset and seed;
- algorithm;
- feature position;
- current and maximum budget;
- physical fits;
- cache hits;
- elapsed time; and
- an ETA derived from recent physical-fit throughput.

Progress state is atomically checkpointed with the master-schedule digest and
the last completed feature/trial boundary. On resume, the executor validates
the protocol, split, model, feature order, schedule digest, and cache schema
before continuing.

The design must preserve the existing parent/worker native-teardown isolation.
There remains at most one GPU/model worker.

## Configuration

The resolved protocol gains a frozen execution block equivalent to:

```yaml
execution:
  mode: grouped_nested_v1
  cache_schema: coalition_utility_v1
  checkpoint_every_fits: 50
  progress_every_seconds: 30
```

The legacy mode remains available only for controlled equivalence and speed
benchmarks. A formal run freezes `grouped_nested_v1`; it cannot resume under a
different execution mode or cache schema.

No adaptive backend selection is added. The source configuration currently has
a CPU default, while the stopped formal run's resolved protocol froze CUDA
through the runner's explicit backend authority. The new formal invocation must
likewise resolve and freeze CUDA explicitly; this design does not silently
change the source default or select a backend per coalition.

## Failure behavior

The pipeline fails closed when:

- cache metadata, schema, checksum, or content is invalid;
- a bundle conflicts with the frozen manifest or protocol;
- a schedule digest changes across resume;
- a canonical per-job artifact conflicts with its group bundle;
- a test-scope lookup is attempted without audited authorization;
- a logical/physical counter is inconsistent; or
- an output contains non-finite or malformed metrics.

A diagnostic must identify the dataset, seed, scope, and violated invariant
without printing sensitive data. Automatic deletion or mutation of the old
formal run is forbidden.

## Test strategy

Implementation follows red-green-refactor. Tests cover:

### Nested schedule invariants

- seed reproducibility;
- different seeds produce different schedules;
- feature streams are independent of configured budget;
- each lower budget is a strict prefix of the maximum schedule;
- checkpoint results equal standalone execution with the same new schedule;
- early `-1` propagation is exact; and
- schedule digests detect feature-order or schema changes.

### Derivation invariants

- exact is computed once and materialized for every budget;
- hybrid equals exact, guided, or the random/guided feature-wise minimum at the
  documented dispatcher boundaries;
- group bundles emit every canonical manifest job exactly once; and
- bundle artifacts validate after interruption and resume.

### Cache invariants

- cached and uncached utilities agree within `1e-12`;
- one fit and one prediction produce all task-authorized utilities;
- ISOLET accuracy and OVR-AUC share one prediction digest;
- changes to split, data, feature order, coalition, model, backend, seed, task,
  class definition, protocol, or schema prevent a hit;
- transaction recovery loses at most the uncommitted batch;
- corrupt metadata, rows, and checksums fail closed; and
- logical and physical counters remain deterministic.

### Stage and leakage invariants

- removal, alpha, and matching ablations reuse only authorized state;
- alpha changes thresholds without retraining identical coalitions;
- duplicate NSGA-II masks avoid physical fits without changing the 800 logical
  evaluations or random evolution;
- test and validation namespaces cannot alias; and
- denied test access remains denied even when test-scope cache files exist.

The complete existing test suite must remain green.

## Scientific and performance acceptance gates

### Abalone exact benchmark

Across all five seeds:

- exact and hybrid results must agree;
- maximum-budget harmful-feature recall against exact must not be lower than the
  old schedule's five-seed mean; and
- maximum-budget score agreement against exact must not be lower than the old
  schedule's five-seed mean.

If exact identifies no harmful feature for a seed, harmful-feature recall is
defined as 1.0 only when the approximation also reports no false harmful
feature; score agreement remains independently defined over all features.

### Real-data pilot

Before a new formal run, execute:

- Spambase seed 42;
- Madelon seed 42; and
- the ISOLET seed 42 shared-utility check.

The optimized path is accepted only when:

- classification validation utility of the maximum-budget hybrid-derived
  removal subset is no more than 0.005 below the corresponding legacy result;
- a later regression comparison allows at most 0.01 lower R-squared;
- removal compression is no more than 0.02 below legacy unless accompanied by
  at least 0.005 higher validation utility;
- Spambase estimator wall time improves by at least 2.5 times;
- Madelon estimator wall time improves by at least 3.5 times; and
- ISOLET accuracy and OVR-AUC carry the same prediction digest.

The pilot uses training and validation data only. It must not inspect the
held-out test set. Speed factors compare a cold new group build with the
durable legacy measurements on the same machine and backend. Reading an already
completed bundle or warm cache is not a valid speed measurement.

### Pipeline acceptance

After unit, full-suite, exact, and pilot gates pass:

1. run a new end-to-end smoke ID;
2. audit manifest completeness, statuses, progress, cache counters, denied test
   access, summary tables, and figures;
3. run a controlled interruption/resume smoke and verify stable completed
   artifacts; and
4. obtain explicit user approval before starting a new formal run ID.

The new formal run executes every frozen scientific condition from scratch
under `grouped_nested_v1`. The old `icomp2026-formal-20260723` directory remains
separate evidence and is not relabelled, overwritten, or mixed with the new
results.

## Expected outcome

The estimator-only empirical upper bounds are approximately 3.0 times for
Spambase and 5.26 times for the completed Madelon seed. Cross-stage utility
reuse and NSGA-II duplicate-mask memoization should improve the full pipeline
further. A conservative full-pipeline target is 4 to 8 times faster, subject to
the measured acceptance gates rather than treated as a guarantee.

Scientific breadth and configured budgets remain intact. The intended paper
claim is not that fewer experiments were run, but that deterministic paired
budget schedules and content-validated memoization eliminated redundant model
training.

## Approved decisions

The user approved:

1. scientific non-inferiority rather than byte-identical old stochastic output;
2. grouped nested execution as the primary acceleration method;
3. run-scoped coalition utility caching and strict test isolation;
4. the stated validation and speed gates; and
5. the implementation scope, compatibility rules, and user-owned Git workflow.
