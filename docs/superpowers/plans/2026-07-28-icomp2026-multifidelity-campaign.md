# ICOMP 2026 Multi-Fidelity Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only calibration and single-seed screening workflow whose frozen per-dataset model policy is reused by the five-seed main-table campaign.

**Architecture:** Keep the existing manifest-authoritative pipeline as the execution engine. Add a frozen model-policy boundary, a calibration command that produces that policy from seed 42 train/validation data, a screen-report command that evaluates all datasets without reading test, and a campaign identity that permits seed-sharded runs to be combined only when their scientific settings match.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn, XGBoost, PyYAML, SQLite, pytest.

## Global Constraints

- Do not read test data during calibration or screening.
- Use candidate rounds `[10, 20, 40, 100]`.
- Use formal estimator checkpoints `[10, 25, 50]`; reserve K=100 for sensitivity.
- Use seed 42 for calibration/screening and `[42, 123, 456, 789, 1024]` for the final campaign.
- All methods within one dataset/seed use the same frozen model policy.
- Classification utilities reuse the same prediction probabilities.
- Do not import or mutate artifacts from `icomp2026-formal-20260727-v3`.
- Do not commit Git changes; the user owns all commits.

---

### Task 1: Frozen model-policy and campaign identity

**Files:**
- Create: `src/minppi_mofs/multifidelity.py`
- Create: `tests/test_multifidelity.py`

**Interfaces:**
- Produces: `CalibrationRules`, `DatasetCalibration`, `ModelPolicy`, `load_model_policy(path, *, expected_datasets)`, `write_model_policy(path, policy)`, `select_candidate_round(calibrations, rules)`, `model_params_for(protocol, dataset_id)`, `campaign_evidence_payload(protocol)`, and `campaign_evidence_digest(protocol)`.
- Consumes: canonical JSON hashing and atomic JSON writing from `minppi_mofs.experiment_protocol` only at call time to avoid import cycles.

- [ ] **Step 1: Write failing validation and selection tests**

```python
def test_selects_smallest_candidate_that_meets_every_fidelity_rule():
    rules = CalibrationRules.default()
    rows = (
        DatasetCalibration.for_test(10, full_utility_loss=0.02),
        DatasetCalibration.for_test(20, full_utility_loss=0.004),
        DatasetCalibration.for_test(40, full_utility_loss=0.001),
        DatasetCalibration.for_test(100, full_utility_loss=0.0),
    )
    assert select_candidate_round(rows, rules) == 20
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python.exe -m pytest tests/test_multifidelity.py -q -p no:cacheprovider`

Expected: import failure because `minppi_mofs.multifidelity` does not exist.

- [ ] **Step 3: Implement immutable schemas and fail-closed JSON validation**

The model-policy document must contain:

```python
{
    "schema_version": 1,
    "calibration_seed": 42,
    "candidate_rounds": [10, 20, 40, 100],
    "rules": {...},
    "datasets": {
        "isolet": {
            "selected_n_estimators": 20,
            "reference_n_estimators": 100,
            "calibrations": [...]
        }
    }
}
```

Reject missing/extra fields, non-finite values, unknown datasets, a reference other than 100, and a selected round not present in the candidates.

- [ ] **Step 4: Implement campaign evidence identity**

The digest includes datasets, preprocessing/split rules, backend, per-dataset model params, alpha, estimator methods/budgets, removal controls and NSGA-II settings. It excludes the active seed shard and run ID so seed 42 and the remaining four seeds can be validated as one campaign.

- [ ] **Step 5: Run focused tests**

Run: `python.exe -m pytest tests/test_multifidelity.py -q -p no:cacheprovider`

Expected: all Task 1 tests pass.

---

### Task 2: Protocol and execution alignment

**Files:**
- Modify: `configs/icomp2026_pipeline.yaml`
- Modify: `src/minppi_mofs/experiment_protocol.py`
- Modify: `experiments/run_icomp2026_pipeline.py`
- Modify: `experiments/icomp2026_estimator_bundle.py`
- Modify: `experiments/icomp2026_stages.py`
- Modify: `tests/test_experiment_protocol.py`
- Modify: `tests/test_icomp2026_stages.py`
- Modify: `tests/test_estimator_bundle.py`

**Interfaces:**
- Consumes: `ModelPolicy` and `model_params_for`.
- Produces: `load_resolved_protocol(..., model_policy_path=None)` and CLI `--model-policy PATH`.

- [ ] **Step 1: Write failing protocol tests**

Cover:

```python
protocol = load_resolved_protocol(
    config_dir,
    seeds=(42,),
    backend="cuda",
    smoke=False,
    model_policy_path=policy_path,
)
assert protocol.model_params_by_dataset["isolet"]["n_estimators"] == 20
assert protocol.model_params_by_dataset["madelon"]["n_estimators"] == 40
```

Also verify the model-policy digest changes the resolved protocol hash and a foreign/missing dataset policy is rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `python.exe -m pytest tests/test_experiment_protocol.py -q -p no:cacheprovider`

Expected: `load_resolved_protocol` rejects the new argument.

- [ ] **Step 3: Add model-policy fields without weakening strict config validation**

Add to `ResolvedProtocol`:

```python
model_params_by_dataset: dict[str, dict[str, object]]
model_policy_digest: str | None
```

Without a policy, each dataset maps to a copy of the global model config. With a policy, only `n_estimators` is overridden; `max_depth`, `learning_rate`, and `n_jobs` remain globally frozen.

- [ ] **Step 4: Route every model construction through `model_params_for`**

Replace direct `context.protocol.model_params` use in estimator caches, utility ablation, removal, alpha and NSGA-II with the dataset-bound helper. Preserve existing behavior when no policy is supplied.

- [ ] **Step 5: Add CLI support and frozen metadata**

Add:

```text
--model-policy PATH
```

The resolved protocol already serializes the mapping and digest, so new-run/resume validation remains fail closed.

- [ ] **Step 6: Change the formal estimator grid to `[10, 25, 50]`**

Retain K=100 only in the design/sensitivity path; do not make it part of every seed's production bundle.

- [ ] **Step 7: Run protocol, bundle and stage tests**

Run:

```powershell
python.exe -m pytest tests/test_experiment_protocol.py tests/test_estimator_bundle.py tests/test_icomp2026_stages.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

---

### Task 3: Validation-only calibration command

**Files:**
- Create: `experiments/calibrate_icomp2026_models.py`
- Create: `tests/test_icomp2026_calibration.py`

**Interfaces:**
- Consumes: a completed preprocess-only run root, frozen config directory, backend, and output policy path.
- Produces: a canonical model-policy JSON plus `calibration_summary.md`.

- [ ] **Step 1: Write failing deterministic-panel tests**

Verify:

- every feature appears in exactly two marginal pairs;
- coalition sizes cover 25% and 75% of the remaining features;
- the same panel is reused for all candidate rounds;
- no function accepts or opens a test path.

- [ ] **Step 2: Write failing selection-metric tests**

Use synthetic utility maps to verify full-utility loss, Spearman correlation, winning agreement, marginal-sign agreement, harmful recall and false-harmful rate.

- [ ] **Step 3: Implement prepared train/validation loading**

Read and hash-check only `train_relpath` and `val_relpath` from `data_manifest.csv`. Never resolve or read `test_relpath`.

- [ ] **Step 4: Implement candidate evaluation**

For each unique coalition:

```python
predictions = fit_predict(
    X_train[selected],
    y_train,
    X_val[selected],
    task=dataset.task,
    model_params={**base_params, "n_estimators": rounds},
    backend=backend,
    seed=42,
)
utility = compute_utility(y_val, predictions, metric=dataset.utility)
```

Use the canonical constant predictor for the empty coalition. Record seconds/fit and all fidelity metrics.

- [ ] **Step 5: Write model policy and human summary atomically**

The Markdown summary lists the selected rounds, speed ratio, each rule outcome and explicit fallback-to-100 reasons.

- [ ] **Step 6: Run focused tests**

Run: `python.exe -m pytest tests/test_icomp2026_calibration.py tests/test_multifidelity.py -q -p no:cacheprovider`

Expected: all calibration tests pass.

---

### Task 4: Single-seed screen report and witness extraction

**Files:**
- Create: `experiments/screen_icomp2026_results.py`
- Create: `tests/test_icomp2026_screening.py`

**Interfaces:**
- Consumes: one completed seed-42 estimator run and its frozen model policy.
- Produces: `screen_report.json`, `screen_report.md`, seven dataset claim cards and a campaign runtime forecast.

- [ ] **Step 1: Write failing conclusion-grade tests**

Cover `strong_support`, `partial_support`, `neutral`, `counterexample`, and `insufficient_evidence`.

- [ ] **Step 2: Write failing witness tests**

Given cached coalition records, return a harmful witness only when:

```python
utility(S) >= alpha * utility(full)
utility(S | {feature}) < alpha * utility(full)
```

Select the witness with the largest continuous utility drop and record both coalitions.

- [ ] **Step 3: Implement validation-only removal checks**

For the K=50 hybrid harmful set, evaluate on validation:

- full subset;
- MinPPI-kept subset;
- five deterministic random-matched subsets.

No test path may be opened. Record compression and utility delta.

- [ ] **Step 4: Implement per-dataset claim cards**

Each card contains data shape, model rounds, fit/runtime counters, harmful counts, witness details, K=25/50 stability, validation removal results, interpretability strength and the fixed conclusion grade.

- [ ] **Step 5: Implement aggregate claim matrix**

Report which claims have directionally supporting, mixed, neutral, counterexample or insufficient evidence. Label every output as provisional and validation-only.

- [ ] **Step 6: Run focused tests**

Run: `python.exe -m pytest tests/test_icomp2026_screening.py -q -p no:cacheprovider`

Expected: all screening tests pass.

---

### Task 5: Seed-sharded campaign validation and runtime budget

**Files:**
- Create: `experiments/summarize_multifidelity_campaign.py`
- Create: `tests/test_multifidelity_campaign.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: seed-42 run plus one or more confirmation run roots.
- Produces: a validated five-seed campaign index and combined main-table input.

- [ ] **Step 1: Write failing shard-compatibility tests**

Reject shards with overlapping seeds, different model-policy digests, different dataset settings, different K/methods or changed source fingerprints. Accept exactly `{42,123,456,789,1024}` with one common campaign evidence digest.

- [ ] **Step 2: Implement runtime forecast**

Use:

```python
estimated_s = estimated_physical_fits * measured_seconds_per_fit * 1.30
```

Emit P0/P1/P2 totals and fail preflight when P0 exceeds 96 GPU hours.

- [ ] **Step 3: Implement campaign index**

The index records each shard root, protocol hash, source fingerprint, seed set, model-policy digest, evidence digest and completion state. It never copies or mutates shard artifacts.

- [ ] **Step 4: Document commands**

Document bootstrap preprocessing, calibration, seed-42 screen, other-seed confirmation, resume, monitoring and reporting commands.

- [ ] **Step 5: Run focused tests**

Run: `python.exe -m pytest tests/test_multifidelity_campaign.py tests/test_icomp2026_reporting.py -q -p no:cacheprovider`

Expected: all campaign/reporting tests pass.

---

### Task 6: Regression verification and first screen run

**Files:**
- Modify only files found failing by the tests above.
- Produce runtime artifacts under a new `results/icomp2026_multifidelity/` campaign root.

**Interfaces:**
- Consumes: all Tasks 1–5.
- Produces: verified code and the first all-dataset seed-42 directional report.

- [ ] **Step 1: Run fast regression suite**

Run:

```powershell
python.exe -m pytest tests -q -p no:cacheprovider
```

Expected: all non-environment-blocked tests pass.

- [ ] **Step 2: Run bootstrap validate/preprocess for seed 42**

Use a new run ID and CUDA backend. Do not resume or mutate the old formal run.

- [ ] **Step 3: Run calibration**

Generate the frozen model policy and inspect every fallback or failed threshold.

- [ ] **Step 4: Run a model-policy smoke test**

Run the existing smoke pipeline with the generated policy before starting the seven-dataset screen.

- [ ] **Step 5: Run the all-dataset seed-42 estimator**

Use K `[10,25,50]`, CUDA and the frozen policy. Monitor progress and preserve checkpoints.

- [ ] **Step 6: Generate the screen report**

Produce the seven claim cards and answer:

- Does the direction support the paper's three claims?
- Which datasets are strong/mixed/neutral/counterexamples?
- Are the witnesses concrete enough to explain?
- Is the forecast for four remaining seeds within the deadline?

- [ ] **Step 7: Stop at the freeze gate**

Do not access test or start the remaining seeds until the user reviews the validation-only screen report.
