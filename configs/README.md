# configs/

`icomp2026_pipeline.yaml` is the authoritative global protocol. The seven
dataset files (`spambase.yaml`, `madelon.yaml`, `isolet.yaml`, `gisette.yaml`,
`credit_card_fraud.yaml`, `crimedata.yaml`, and `abalone.yaml`) supply only the
dataset identity, raw-data authority, target/task/utility, train-only
preprocessing settings, and split fractions.

The frozen formal defaults include seeds `[42, 123, 456, 789, 1024]`, one common
backend and XGBoost model configuration, alpha grid, estimator budgets, matched
removal controls, NSGA-II budget/reference point, and declared ablations. Unknown
or misspelled keys are rejected rather than ignored. CLI overrides are resolved
before a new run freezes `resolved_protocol.yaml` and the copied configs.

`--smoke` selects only Spambase and Abalone, seed 42, and the frozen low budgets.
It also sets `run_kind: smoke`; smoke outputs cannot be promoted or copied into a
formal result directory. `--stop-after-jobs` is accepted only in smoke mode.

Resume must specify the same run ID and the same resolved protocol. Source or
configuration drift is rejected before pending work runs. A prior successful job
is skipped only when its canonical result still validates; its artifact and
successful attempt must remain unchanged.

Task 12 was accepted with the serialized CUDA runs `task5-smoke-recovery` and
`task5-resume-check`. The older `task5-smoke` is immutable completed-but-nonzero
diagnostic evidence, not acceptance evidence. The public CLI uses one parent and
one synchronous worker; only Windows `0xC0000409` after a nonce-valid zero receipt
is contained. Exit 75 and all ordinary failures remain nonzero.

Only one GPU/model worker may run at any time. Formal five-seed execution must
use a new, explicitly formal run ID and must not begin without the user's explicit
approval after reviewing the Task 12 evidence.
