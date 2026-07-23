# CUDA Teardown Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public ICOMP 2026 pipeline CLI survive the observed post-completion Windows CUDA/XGBoost fail-fast teardown without converting genuine pipeline or worker failures into success.

**Architecture:** Keep the importable pipeline `main()` unchanged and add a small standard-library isolation module. A parent CLI launches one worker with the same interpreter and arguments; the worker atomically records an authenticated return-code receipt after `main()` returns, and the parent contains only Windows `0xC0000409` paired with a valid zero receipt.

**Tech Stack:** Python 3.12 standard library, pytest, Ruff, existing ICOMP 2026 runner and protocol/status stores.

## Global Constraints

- Preserve `results/icomp2026_rerun/task5-smoke` unchanged as failed-invocation diagnostic evidence.
- Never run two GPU/model processes concurrently; all acceptance commands are serialized.
- Do not start the formal five-seed experiment in this plan.
- Do not add a public recovery flag or weaken manifest, artifact, source-drift, protocol-drift, or resume validation.
- Contain only Windows `0xC0000409` after a schema-valid, nonce-matching receipt records `main()` return code 0.
- Keep `experiments/native_process_guard.py` standard-library-only and dispatch the parent before any `minppi_mofs` or stage import; importing a submodule of `minppi_mofs` executes its XGBoost-loading package initializer and is therefore forbidden in the parent.
- Propagate controlled-stop code 75 and every ordinary non-zero worker exit.
- The parent-created temporary receipt directory must be cleaned after every decision path.
- Git staging, commits, resets, cleans, branches, and worktrees belong to the user; agents may only inspect `git diff`, `git status`, and `git diff --check`.

---

### Task 1: Receipt validation and exit decision

**Files:**
- Create: `experiments/native_process_guard.py`
- Create: `tests/test_native_process_guard.py`

**Interfaces:**
- Produces: `WorkerContract(receipt_path: Path, nonce: str)`, `worker_contract(environ: Mapping[str, str]) -> WorkerContract | None`, `write_worker_receipt(contract: WorkerContract, exit_code: int) -> None`, and `decide_public_exit(worker_returncode: int, receipt_code: int | None) -> tuple[int, bool]`.
- The returned Boolean is true only when a post-success fail-fast was contained and the caller must print a warning.

- [x] **Step 1: Write failing contract and decision tests**

Add tests that specify strict behavior before production code exists:

```python
import json
from pathlib import Path

import pytest

from experiments.native_process_guard import (
    RECEIPT_SCHEMA_VERSION,
    WorkerContract,
    decide_public_exit,
    read_worker_receipt,
    worker_contract,
    write_worker_receipt,
)


def test_worker_contract_requires_complete_private_environment(tmp_path: Path) -> None:
    assert worker_contract({}) is None
    with pytest.raises(ValueError, match="incomplete worker contract"):
        worker_contract({"MINPPI_PIPELINE_WORKER": "1"})
    contract = worker_contract(
        {
            "MINPPI_PIPELINE_WORKER": "1",
            "MINPPI_PIPELINE_RECEIPT": str(tmp_path / "receipt.json"),
            "MINPPI_PIPELINE_NONCE": "a" * 64,
        }
    )
    assert contract == WorkerContract(tmp_path / "receipt.json", "a" * 64)


def test_receipt_round_trip_is_schema_and_nonce_bound(tmp_path: Path) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    write_worker_receipt(contract, 75)
    assert read_worker_receipt(contract) == 75
    payload = json.loads(contract.receipt_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "nonce": "b" * 64,
        "exit_code": 75,
    }


@pytest.mark.parametrize("returncode", [-1073740791, 3221226505])
def test_known_windows_fail_fast_is_contained_only_after_zero_receipt(
    returncode: int,
) -> None:
    assert decide_public_exit(returncode, 0) == (0, True)
    assert decide_public_exit(returncode, None) == (returncode, False)
    assert decide_public_exit(returncode, 2) == (returncode, False)


@pytest.mark.parametrize("code", [0, 2, 70, 75])
def test_matching_ordinary_worker_exit_is_propagated(code: int) -> None:
    assert decide_public_exit(code, code) == (code, False)


def test_zero_exit_without_valid_receipt_fails_closed() -> None:
    assert decide_public_exit(0, None) == (70, False)


def test_unrelated_native_failure_is_never_contained() -> None:
    assert decide_public_exit(-1073741819, 0) == (-1073741819, False)
```

Also parameterize `read_worker_receipt()` over malformed JSON, wrong schema,
wrong nonce, Boolean/non-integer exit codes, extra keys, and a missing file; each
case must return `None` rather than accepting ambiguous evidence.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_native_process_guard.py -q -p no:cacheprovider
```

Expected: collection fails because `experiments.native_process_guard` does not
exist. This is the required RED state.

- [x] **Step 3: Implement the minimal receipt and decision primitives**

Create the module with these constants and exact validation rules:

```python
RECEIPT_SCHEMA_VERSION = 1
CONTROLLER_ERROR = 70
WINDOWS_FAIL_FAST = 0xC0000409
WORKER_MARKER_ENV = "MINPPI_PIPELINE_WORKER"
WORKER_RECEIPT_ENV = "MINPPI_PIPELINE_RECEIPT"
WORKER_NONCE_ENV = "MINPPI_PIPELINE_NONCE"


@dataclass(frozen=True)
class WorkerContract:
    receipt_path: Path
    nonce: str
```

`worker_contract()` returns `None` only when all three private variables are
absent. A partial contract raises `ValueError`; a complete contract requires
marker `"1"`, a 64-character lowercase hexadecimal nonce, and a non-directory
receipt path. `write_worker_receipt()` validates `type(exit_code) is int`, writes
UTF-8 JSON plus newline to a sibling temporary file, flushes and `os.fsync()`s it,
then publishes with `os.replace()`.

`read_worker_receipt()` returns `None` for any I/O, JSON, schema, exact-key,
nonce, or type violation. Compare the nonce with `secrets.compare_digest()`.

Normalize the native code with `worker_returncode & 0xFFFFFFFF`. The only
special decision is normalized `WINDOWS_FAIL_FAST` plus receipt 0, which returns
`(0, True)`. Matching ordinary codes are propagated, zero without a valid receipt
returns `(CONTROLLER_ERROR, False)`, and all other non-zero child codes are
propagated.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the focused pytest command from Step 2, then:

```powershell
..\venv\Scripts\ruff.exe check experiments\native_process_guard.py tests\test_native_process_guard.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [x] **Step 5: Inspect the Task 1 diff without mutating Git**

Run `git diff --check` and `git status --short`. Do not stage or commit; hand the
Task 1 diff to a fresh reviewer before Task 2.

---

### Task 2: Parent/worker CLI integration

**Files:**
- Modify: `experiments/native_process_guard.py`
- Modify: `experiments/run_icomp2026_pipeline.py`
- Modify: `tests/test_native_process_guard.py`
- Modify: `tests/test_icomp2026_stages.py`

**Interfaces:**
- Consumes: Task 1 receipt/decision functions.
- Produces: `run_isolated_cli(script: Path, argv: Sequence[str], *, executable: str = sys.executable, environ: Mapping[str, str] | None = None, process_runner: Callable[..., CompletedProcess] = subprocess.run) -> int` and a script-only `_entrypoint() -> int`.

- [x] **Step 1: Write failing controller tests**

Use an injected process runner so tests exercise real receipt parsing and cleanup
without crashing Python. The fake runner must assert that the executable, fully
resolved script path, and original arguments are forwarded exactly; it reads the
three private environment variables, writes a real receipt when requested, and
returns `subprocess.CompletedProcess` with the selected code.

Required cases:

```python
def test_isolated_cli_returns_zero_and_removes_receipt_directory(...): ...
def test_isolated_cli_propagates_controlled_stop_75(...): ...
def test_isolated_cli_contains_fail_fast_after_valid_zero_receipt(...): ...
def test_isolated_cli_propagates_fail_fast_without_receipt(...): ...
def test_isolated_cli_rejects_stale_nonce_receipt(...): ...
def test_isolated_cli_propagates_ordinary_nonzero_without_receipt(...): ...
def test_isolated_cli_returns_70_when_zero_worker_omits_receipt(...): ...
def test_isolated_cli_returns_70_when_process_launch_raises_oserror(...): ...
```

Assert the containment warning names `0xC0000409` and says the pipeline `main()`
completed. Assert that launch/receipt errors produce actionable stderr and no
temporary directory remains.

Add a runner entrypoint test that monkeypatches `main`, `worker_contract`,
`write_worker_receipt`, and `run_isolated_cli` only at the boundary:

```python
def test_entrypoint_worker_calls_main_once_and_records_result(monkeypatch): ...
def test_entrypoint_parent_delegates_original_argv_without_calling_main(monkeypatch): ...
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_native_process_guard.py tests\test_icomp2026_stages.py -q -p no:cacheprovider
```

Expected: the new tests fail because `run_isolated_cli()` and `_entrypoint()` do
not exist.

- [x] **Step 3: Implement the synchronous parent controller**

`run_isolated_cli()` must:

```python
with tempfile.TemporaryDirectory(prefix="minppi-pipeline-") as temp_dir:
    contract = WorkerContract(Path(temp_dir) / "receipt.json", secrets.token_hex(32))
    child_environment = dict(os.environ if environ is None else environ)
    child_environment.update(contract.as_environment())
    completed = process_runner(
        [executable, str(script.resolve()), *argv],
        env=child_environment,
        check=False,
    )
    receipt_code = read_worker_receipt(contract)
    public_code, contained = decide_public_exit(completed.returncode, receipt_code)
```

Catch only launch/controller `OSError` and `ValueError`, print a single concise
message to stderr, and return 70. If `contained` is true, print the documented
warning. Do not capture worker stdout/stderr and do not retry.

- [x] **Step 4: Integrate the script entrypoint**

In `experiments/run_icomp2026_pipeline.py`, preserve `main()` and add:

```python
def _entrypoint(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    contract = worker_contract(os.environ)
    if contract is None:
        return run_isolated_cli(Path(__file__), arguments)
    result = main(arguments)
    write_worker_receipt(contract, result)
    return result


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
```

Remove the old direct `raise SystemExit(main())` block. Keep the existing early
`--help` path so help remains fast and side-effect free. A malformed partial
worker contract must fail before `main()` runs and must not create an experiment
run.

- [x] **Step 5: Verify GREEN and backward compatibility**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_native_process_guard.py tests\test_icomp2026_stages.py -q -p no:cacheprovider
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --help
..\venv\Scripts\ruff.exe check src tests experiments demos
..\venv\Scripts\python.exe -m py_compile experiments\native_process_guard.py experiments\run_icomp2026_pipeline.py
git diff --check
```

Expected: pytest passes, help exits 0 without creating a run, Ruff and compile
pass, and the diff has no whitespace errors.

- [x] **Step 6: Run the complete CPU test gate**

Run:

```powershell
..\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Expected: all tests pass with only the repository's known skip. Hand the Task 2
diff and exact output to a fresh reviewer; do not stage or commit.

---

### Task 3: Recover and complete Task 12 acceptance

**Files:**
- Modify: `experiments/README.md`
- Modify: `configs/README.md`
- Modify: `ICOMP2026_experiment_redesign_submission_plan.md`
- Modify: `docs/superpowers/plans/2026-07-21-task5-unified-runner.md`
- Create: `results/icomp2026_rerun/task5-smoke-recovery/` through the runner
- Create: `results/icomp2026_rerun/task5-resume-check/` through the runner

**Interfaces:**
- Consumes: the isolated public CLI from Task 2.
- Produces: one passing CUDA smoke run, one proven controlled-stop/resume run,
  updated operator documentation, and completed Task 12 evidence.

- [x] **Step 1: Re-run the non-GPU local gate immediately before acceptance**

Run the complete pytest, Ruff, `py_compile`, and `git diff --check` commands from
Task 2. Record exact counts and exit codes.

Observed 2026-07-23: 691 passed, 1 skipped; pytest, Ruff, expanded six-file
`py_compile`, and `git diff --check` all exited 0.

- [x] **Step 2: Run one new serialized CUDA smoke**

Run exactly:

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-smoke-recovery --smoke
```

Expected public exit: 0. Do not alter `task5-smoke`. If the worker again exits
with `0xC0000409`, require the parent warning and a valid zero receipt decision;
if any other failure occurs, stop and return to systematic debugging.

Observed: public exit 0 after 738.6 seconds. The only worker reproduced
`0xC0000409` after `main()` completed and the parent emitted the required
containment warning. GPU compute-process count never exceeded one.

- [x] **Step 3: Audit the new smoke artifacts**

Prove from durable files that all 55 manifest jobs are terminal successes, no
stage has failed/pending/running/unapproved skipped jobs, the protocol audit has
no unresolved denied access, the summary bundle is complete, all three PDFs are
non-empty, and `run_kind` is smoke. Confirm no formal result directory was
created.

Observed: 55/55 successful jobs and 73 declared artifact references validated;
all root/per-stage non-success counts were zero. The audit contained 24 allowed,
zero denied, and zero unresolved accesses. Six summary files and three non-empty
PDFs matched their manifest bytes and hashes. Only the recovery smoke directory
was added, and the old `task5-smoke` key hashes, bytes, mtimes, status, and log
were unchanged.

- [x] **Step 4: Prove controlled interruption and immutable resume**

Run exactly, sequentially:

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --stop-after-jobs 1
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --run-id task5-resume-check --smoke --resume
```

Require exit 75 from the first command and exit 0 from the second. Before resume,
record SHA-256, byte size, modification timestamp, state, and attempt number for
the first completed job's declared artifact and status entry. After resume, prove
the artifact bytes/hash and successful attempt are unchanged and all remaining
jobs completed.

Observed public exits: 75 then 0. The first validate result and its two declared
artifacts retained identical SHA-256, bytes, and UTC mtimes; state remained
`success`, attempt remained 1. The completed run had 55 successes and the attempt
distribution was `{1: 55}`. The successful resumed worker emitted the same
restricted post-success containment warning.

- [x] **Step 5: Update operator and submission documentation**

Document:

- the parent/worker behavior and exact containment restriction;
- `task5-smoke` as a completed-but-nonzero diagnostic run, never as acceptance;
- `task5-smoke-recovery` as the passing acceptance run;
- the exact stop/resume commands and evidence;
- one-GPU serialization, smoke/formal separation, and the requirement for
  explicit user approval before the formal five-seed run.

Mark Task 12 checkboxes complete only after Steps 1–4 pass. Do not mark Task 13
or any formal-run item complete.

- [x] **Step 6: Final verification and handoff**

Run fresh:

```powershell
..\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
..\venv\Scripts\ruff.exe check src tests experiments demos
..\venv\Scripts\python.exe -m py_compile src\minppi_mofs\experiment_protocol.py experiments\native_process_guard.py experiments\icomp2026_stages.py experiments\summarize_icomp2026.py experiments\make_paper_figures.py experiments\run_icomp2026_pipeline.py
git diff --check
git status --short
```

Have a fresh read-only reviewer inspect the implementation, tests, Task 12
artifacts, and documentation. Resolve every Critical or Important finding and
repeat affected verification. Report modified files and commands to the user,
leave Git untouched, and wait for explicit authorization before Task 13/formal
execution.
