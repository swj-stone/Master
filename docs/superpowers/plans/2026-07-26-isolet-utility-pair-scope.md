# ISOLET Utility Pair Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated CUDA-only runner that creates a new manifest-authoritative ISOLET seed-42 run containing only validation, preprocessing, and the canonical paired utility ablation jobs.

**Architecture:** A new `experiments/run_isolet_utility_pair.py` module owns the narrow scope. It obtains every `JobSpec` from `build_job_manifest`, initializes the complete scoped manifest before status exists, delegates validation and preprocessing to the existing stage dispatcher, revalidates the pair against the produced rich data manifest, and dispatches the two ablation records with a dedicated dependency rule of `validate + preprocess`. The ordinary pipeline and its `ablation -> preprocess + estimator` dependency remain untouched.

**Tech Stack:** Python 3.10+, pytest, existing experiment protocol/status/audit APIs.

## Global Constraints

- Do not run GPU work while implementing or testing.
- Do not mutate an existing run.
- Do not stage, commit, or otherwise write Git state.
- Accept only one ISOLET dataset, seed `42`, backend `cuda`, and utility values `accuracy` plus `ovr_auc`.
- Support both `legacy` and `grouped_nested_v1` execution configs without changing the paired scientific artifact.
- Never read the held-out test split.

---

### Task 1: Canonical scoped-manifest builder

**Files:**
- Create: `experiments/run_isolet_utility_pair.py`
- Test: `tests/test_isolet_utility_pair_scope.py`

**Interfaces:**
- Produces: `build_isolet_utility_pair_manifest(protocol) -> tuple[JobSpec, ...]`
- Produces: `validate_isolet_utility_pair_manifest(protocol, manifest, data_manifest) -> tuple[JobSpec, JobSpec]`

- [ ] Write tests that reject wrong dataset, seed, backend, value set, duplicate jobs, and modified/noncanonical fields.
- [ ] Run the tests and confirm failure because the module/API does not exist.
- [ ] Implement the smallest pure builder/validator using only `build_job_manifest` output.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Durable dedicated dispatcher

**Files:**
- Modify: `experiments/run_isolet_utility_pair.py`
- Test: `tests/test_isolet_utility_pair_scope.py`

**Interfaces:**
- Produces: `IsoletUtilityPairRunner.run() -> int`
- Consumes: existing `execute_stage`, `StatusStore`, `AuditLog`, `run_ablation_job`, and canonical JSON publication.

- [ ] Write tests for manifest/status/result/audit persistence, validate+preprocess-only dependency, shared artifact reuse, resume validation, tamper failure, and no test-split access.
- [ ] Confirm the tests fail for the missing dispatcher.
- [ ] Implement new-run/resume orchestration and a pair-only status transition loop.
- [ ] Confirm focused tests pass without invoking XGBoost or CUDA.

### Task 3: CLI and regression verification

**Files:**
- Modify: `experiments/run_isolet_utility_pair.py`
- Test: `tests/test_isolet_utility_pair_scope.py`

**Interfaces:**
- Produces: CLI flags `--config-dir`, `--output-root`, `--run-id`, and `--resume`.

- [ ] Write parser/main tests proving CUDA is forced and existing-run overwrite is rejected.
- [ ] Confirm RED, implement the isolated CLI entry point, then confirm GREEN.
- [ ] Run the new test file, targeted utility/stage tests, and Ruff.
- [ ] Review the diff against every global constraint and report exact commands and job count.
