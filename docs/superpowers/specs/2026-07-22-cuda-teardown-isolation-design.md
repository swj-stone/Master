# CUDA teardown isolation design

## Context

During Task 12, the authoritative smoke command completed all 55 manifest jobs,
wrote a clean root status (55 successes, no failed, pending, running, skipped, or
unresolved jobs), produced the summary bundle and all three paper figures, and
logged `finish stage=all exit_code=0`. About five seconds later, the Python
process terminated on Windows with `0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`)
and no Python traceback.

Read-only investigation found no matching Windows Event Log, WER, Reliability,
ReportArchive, ReportQueue, or CrashDumps record. The failure did not reproduce
with the runner import, XGBoost 3.3.0 CUDA-build import and build information,
Agg PDF rendering, the combined import stack, or the exact three-PDF workload.
The material difference was that the failed process had fitted CUDA XGBoost
models. The supported fault boundary is therefore native CUDA/XGBoost teardown
after the Python pipeline has durably completed; the exact DLL cannot be named
without a crash dump or another full GPU reproduction.

The existing `task5-smoke` directory is immutable evidence and must not be
deleted, replaced, or relabelled as a passing OS-level invocation.

## Goal and non-goals

The public CLI must isolate native teardown from its controlling process while
preserving all current command-line arguments, exit codes, run-directory
semantics, logs, resume guarantees, and single-GPU serialization.

This change does not repair or hide corruption during a pipeline stage. It does
not weaken manifest, status, artifact, source-drift, or protocol-drift checks.
It does not authorize a formal run, a second simultaneous GPU process, or reuse
of the failed smoke run as the post-fix acceptance run.

## Considered approaches

1. **Parent/worker isolation with an authenticated completion receipt
   (selected).** The public process launches exactly one worker using the same
   interpreter and arguments. The worker writes a private receipt immediately
   after `main()` returns. The parent propagates ordinary exit codes and only
   contains the known Windows fail-fast code when the receipt proves that
   `main()` returned zero.
2. **Accept durable output despite the non-zero OS exit.** This preserves the
   current code but fails the Task 12 requirement that the command itself exit
   zero and leaves future automation unable to distinguish a completed run from
   a genuine crash.
3. **Call `os._exit` after pipeline completion.** This is smaller but bypasses
   normal interpreter teardown in every successful run and could conceal other
   cleanup defects. It is rejected.

## Architecture and data flow

`experiments/run_icomp2026_pipeline.py` retains its existing importable `main()`
API. Only script execution gains a lightweight controller:

1. The public process creates a fresh private temporary directory and generates
   a cryptographically random nonce.
2. It starts one child process with `sys.executable`, the same script path, and
   the original argument vector. Private environment variables carry a worker
   marker, receipt path, and nonce. No user-facing recovery flag is introduced.
3. The worker calls the existing `main()` exactly once. Immediately after it
   returns, the worker atomically writes a JSON receipt containing a fixed schema
   version, the nonce, and the integer return code. It then exits normally via
   `SystemExit`, allowing native teardown to run.
4. The parent waits synchronously. There is never more than one worker and never
   more than one GPU/model process.
5. The parent validates the receipt schema, nonce, types, and allowed location,
   then decides the public exit code using the rules below. The temporary
   directory is removed by the parent after the decision; it is not an experiment
   artifact and cannot alter run fingerprints or resume state.

The worker environment marker is accepted only together with a parent-created
receipt path and nonce. Recursive child spawning is therefore prevented without
adding a public CLI option.

## Exit-code rules

- Child exits normally and the valid receipt matches the child exit code: return
  that code unchanged, including controlled-stop code 75.
- Child exits with Windows `0xC0000409`, and a valid receipt says `main()` returned
  0: emit a concise warning to stderr and return 0.
- Child exits with `0xC0000409`, but the receipt is absent, malformed, has the
  wrong nonce/schema, or records a non-zero result: propagate failure.
- Child exits with any other non-zero code: propagate failure, even if a receipt
  exists.
- Child exits zero without a valid matching zero receipt: fail closed because
  the controller cannot prove that the intended worker path completed.
- Process-launch, receipt-I/O, or validation errors fail closed with a stable
  non-zero controller code and an actionable stderr message.

Containment is intentionally restricted to Windows and the single observed
fail-fast status. Linux/macOS signals and unrelated Windows native exits are not
converted to success.

## Security and integrity properties

The receipt is control-plane evidence, not scientific evidence. A random nonce
binds it to one parent invocation, a private temporary directory prevents
run-directory collisions, and atomic replacement prevents partial JSON from
being accepted. The parent never infers success merely from the existence of
experiment output. The receipt must prove that the existing Python `main()`
finished successfully before the known teardown failure occurred.

The experiment run directory remains authoritative for scientific completeness.
Task 12 will still separately audit its manifest, root and per-stage statuses,
denied-access resolutions, summary bundle, figures, and smoke/formal isolation.

## Test strategy

Implementation follows red-green-refactor. Tests exercise the controller with a
small helper child rather than CUDA:

- normal return 0 with a valid receipt;
- controlled-stop return 75 with a matching receipt;
- ordinary non-zero child failure;
- simulated `0xC0000409` after a valid zero receipt;
- the same fail-fast code with missing, malformed, stale-nonce, wrong-schema, or
  non-zero receipts;
- zero child exit without a receipt;
- recursion prevention and exact argument/interpreter forwarding;
- cleanup of the private temporary directory;
- the existing importable `main()` tests and CLI subprocess tests remain green.

The fail-fast return code will be injected at the subprocess boundary. Tests do
not intentionally crash Python and do not need a GPU.

## Task 12 recovery and acceptance

After unit and full-suite verification, use a new run ID for one serialized
post-fix smoke run; preserve `task5-smoke` as diagnostic evidence. The new run
must exit zero publicly and pass the same durable artifact/status audit. Then run
the controlled stop/resume check with its dedicated run ID, proving exit 75,
successful resume, and byte-for-byte stability of the completed artifact.

Documentation must distinguish the failed diagnostic run from the passing
post-fix smoke run and record the native-teardown containment behavior. No formal
five-seed experiment starts until the user explicitly approves it after Task 12
is complete.
