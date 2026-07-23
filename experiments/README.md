# experiments/

## Authoritative ICOMP 2026 runner

`run_icomp2026_pipeline.py` is the only runner for new ICOMP 2026 evidence. It
accepts:

```text
--stage {validate,preprocess,estimator,removal,alpha,nsga2,ablation,summarize,figures,all}
--config-dir <path>
--output-root <path>
--seeds <seed...>
--backend {cpu,cuda}
--run-id <id>
--resume
--smoke
--stop-after-jobs <N>  # smoke only
```

For script execution, the public process is a lightweight controller. It starts
exactly one worker with the same interpreter and arguments, waits synchronously,
and validates a nonce-bound completion receipt written after `main()` returns.
On Windows it converts only native exit `0xC0000409` to public success, and only
when that authenticated receipt records `main()` return code 0. Controlled stop
75 and every ordinary nonzero exit are propagated unchanged; absent, malformed,
or mismatched receipts fail closed. The parent imports no `minppi_mofs` or stage
module, so model and CUDA state exist only in the single worker.

Never run two GPU/model commands at once. Each command below must finish and the
GPU process must exit before the next command starts.

## Task 12 acceptance evidence

`results/icomp2026_rerun/task5-smoke` is immutable diagnostic evidence. Its 55
jobs and durable artifacts completed, but its original public invocation ended
nonzero during native CUDA/XGBoost interpreter teardown. It is not the acceptance
run and must not be deleted, overwritten, resumed, or relabelled as passing.

The accepted post-fix smoke command was:

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-smoke-recovery --smoke
```

On 2026-07-23 it returned public exit 0. The worker reproduced the post-success
`0xC0000409` teardown and the parent emitted the documented containment warning.
The durable audit found 55/55 successes, no failed/pending/running/skipped jobs,
24 allowed and zero denied/unresolved test-access events, a complete six-file
summary bundle, and three non-empty paper PDFs. The frozen protocol and summary
both identify the run as `smoke`.

Controlled interruption and resume were proven, in this order:

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --stop-after-jobs 1
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --resume
```

The first command returned 75 and the resume returned public exit 0. The first
completed validate job remained `success`, attempt 1. Its result record and both
declared artifacts retained identical bytes, SHA-256, and modification times;
after resume all 55 jobs were successful and every successful attempt remained
1.

## Output and audit contract

Each run owns frozen protocol/environment/source metadata, `run_manifest.csv`,
root and per-stage `status.json`, canonical records under `raw_metrics/`,
`protocol_audit.json`, `output.log`, summary tables, and figures. A summary is
valid only after manifest-based completeness succeeds. Allowed held-out test
access is limited to final removal, alpha-grid, and NSGA-II-front evaluation;
every other request is denied and any denial requires an explicit audited
resolution before publication.

Smoke and formal results must use different run IDs and directories. Smoke
tables, macros, and figures are validation artifacts and must never be copied
into the paper as formal results. No five-seed formal command is authorized by
Task 12; it requires the user's explicit approval in a later task.

Legacy scripts remain historical or narrowly scoped utilities. They are not an
alternative source of new paper results.
