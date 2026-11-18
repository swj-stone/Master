"""Authenticated completion receipts for isolated native worker processes."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


RECEIPT_SCHEMA_VERSION = 1
CONTROLLER_ERROR = 70
WINDOWS_FAIL_FAST = 0xC0000409
WORKER_MARKER_ENV = "MINPPI_PIPELINE_WORKER"
WORKER_RECEIPT_ENV = "MINPPI_PIPELINE_RECEIPT"
WORKER_NONCE_ENV = "MINPPI_PIPELINE_NONCE"

_RECEIPT_KEYS = frozenset({"schema_version", "nonce", "exit_code"})
_NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class WorkerContract:
    """Same-user parent controller's private worker invocation contract.

    This contract prevents accidental recursion; it is not a security boundary
    against a malicious process running as the same operating-system user.
    """

    receipt_path: Path
    nonce: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_path, Path):
            raise TypeError("receipt_path must be a Path")
        if type(self.nonce) is not str:
            raise TypeError("nonce must be a string")
        if _NONCE_PATTERN.fullmatch(self.nonce) is None:
            raise ValueError("nonce must be 64 lowercase hexadecimal characters")

    def as_environment(self) -> dict[str, str]:
        """Return the private variables needed to invoke exactly one worker."""

        return {
            WORKER_MARKER_ENV: "1",
            WORKER_RECEIPT_ENV: str(self.receipt_path),
            WORKER_NONCE_ENV: self.nonce,
        }


def worker_contract(environ: Mapping[str, str]) -> WorkerContract | None:
    """Read the same-user parent controller's private environment contract.

    These variables are an invocation contract, not a security boundary against
    a malicious process running as the same operating-system user.
    """

    names = (WORKER_MARKER_ENV, WORKER_RECEIPT_ENV, WORKER_NONCE_ENV)
    present = tuple(name in environ for name in names)
    if not any(present):
        return None
    if not all(present):
        raise ValueError("incomplete worker contract")

    marker = environ[WORKER_MARKER_ENV]
    receipt_path = Path(environ[WORKER_RECEIPT_ENV])
    nonce = environ[WORKER_NONCE_ENV]
    if marker != "1":
        raise ValueError("invalid worker marker")
    if _NONCE_PATTERN.fullmatch(nonce) is None:
        raise ValueError("invalid worker nonce")
    if receipt_path.is_dir():
        raise ValueError("worker receipt path must not be a directory")
    return WorkerContract(receipt_path, nonce)


def write_worker_receipt(contract: WorkerContract, exit_code: int) -> None:
    """Durably publish a worker completion receipt with atomic replacement."""

    if type(exit_code) is not int:
        raise TypeError("exit_code must be an integer")
    if not 0 <= exit_code <= 255:
        raise ValueError("exit_code must be in range 0..255")

    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "nonce": contract.nonce,
        "exit_code": exit_code,
    }
    temporary_path = contract.receipt_path.with_name(
        f".{contract.receipt_path.name}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as receipt:
            json.dump(payload, receipt, sort_keys=True)
            receipt.write("\n")
            receipt.flush()
            os.fsync(receipt.fileno())
        os.replace(temporary_path, contract.receipt_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def read_worker_receipt(contract: WorkerContract) -> int | None:
    """Return a valid nonce-bound receipt code, or ``None`` on any violation."""

    try:
        payload = json.loads(contract.receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None

    if type(payload) is not dict or set(payload) != _RECEIPT_KEYS:
        return None
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != RECEIPT_SCHEMA_VERSION
    ):
        return None
    if (
        type(payload["nonce"]) is not str
        or _NONCE_PATTERN.fullmatch(payload["nonce"]) is None
    ):
        return None
    if not secrets.compare_digest(payload["nonce"], contract.nonce):
        return None
    if (
        type(payload["exit_code"]) is not int
        or not 0 <= payload["exit_code"] <= 255
    ):
        return None
    return payload["exit_code"]


def decide_public_exit(
    worker_returncode: int,
    receipt_code: int | None,
) -> tuple[int, bool]:
    """Map a worker/receipt result to the fail-closed public process result."""

    normalized_returncode = worker_returncode & 0xFFFFFFFF
    if normalized_returncode == WINDOWS_FAIL_FAST and receipt_code == 0:
        return 0, True
    if worker_returncode == 0 and receipt_code != 0:
        return CONTROLLER_ERROR, False
    return worker_returncode, False


def run_isolated_cli(
    script: Path,
    argv: Sequence[str],
    *,
    executable: str = sys.executable,
    environ: Mapping[str, str] | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> int:
    """Run one synchronous worker and fail closed without capturing its streams."""

    try:
        with tempfile.TemporaryDirectory(prefix="minppi-pipeline-") as temp_dir:
            contract = WorkerContract(
                Path(temp_dir) / "receipt.json",
                secrets.token_hex(32),
            )
            child_environment = dict(os.environ if environ is None else environ)
            child_environment.update(contract.as_environment())
            completed = process_runner(
                [executable, str(script.resolve()), *argv],
                env=child_environment,
                check=False,
            )
            receipt_code = read_worker_receipt(contract)
            public_code, contained = decide_public_exit(
                completed.returncode,
                receipt_code,
            )
    except (OSError, ValueError) as exc:
        print(f"pipeline isolation controller error: {exc}", file=sys.stderr)
        return CONTROLLER_ERROR
    normalized_returncode = completed.returncode & 0xFFFFFFFF
    if (
        normalized_returncode == WINDOWS_FAIL_FAST
        and receipt_code != 0
    ):
        print(
            "warning: refusing to contain Windows teardown failure 0xC0000409 "
            "without a valid zero completion receipt",
            file=sys.stderr,
        )
    if completed.returncode == 0 and public_code == CONTROLLER_ERROR:
        print(
            "pipeline isolation controller error: worker exited zero "
            "without a valid matching completion receipt",
            file=sys.stderr,
        )
    if contained:
        print(
            "warning: contained Windows teardown failure 0xC0000409 "
            "after pipeline main() completed successfully",
            file=sys.stderr,
        )
    return public_code
