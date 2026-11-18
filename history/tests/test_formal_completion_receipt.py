"""Exercise the formal run's watchdog with on-disk, nonce-bound worker receipts."""
import json
from unittest.mock import Mock

import pytest

from experiments.native_process_guard import (
    CONTROLLER_ERROR,
    WINDOWS_FAIL_FAST,
    WorkerContract,
    write_worker_receipt,
)
from experiments.run_thesis_sampling import supervise


@pytest.mark.parametrize(
    "raw_exit,receipt_code,stale_nonce,has_result,status,public_exit,contained",
    [
        (0, 0, False, True, "completed", 0, False),
        (0, None, False, True, "error", CONTROLLER_ERROR, False),
        (0, 1, False, True, "error", CONTROLLER_ERROR, False),
        (WINDOWS_FAIL_FAST, 0, False, True, "completed", 0, True),
        (WINDOWS_FAIL_FAST - 2**32, 0, False, True, "completed", 0, True),
        (WINDOWS_FAIL_FAST, None, False, True, "error", WINDOWS_FAIL_FAST, False),
        (WINDOWS_FAIL_FAST, 0, True, True, "error", WINDOWS_FAIL_FAST, False),
        (7, 0, False, True, "error", 7, False),
        (0, 0, False, False, "error", 0, False),
    ],
)
def test_formal_watchdog_receipt(tmp_path, raw_exit, receipt_code, stale_nonce, has_result,
                                 status, public_exit, contained):
    contract = WorkerContract(tmp_path / "receipt.json", "a" * 64)
    worker = Mock(exitcode=raw_exit)
    worker.is_alive.return_value = False

    def publish():
        if has_result:
            (tmp_path / "result.json").write_text('{"k": 5}', encoding="utf-8")
        if receipt_code is not None:
            signer = WorkerContract(contract.receipt_path, "b" * 64) if stale_nonce else contract
            write_worker_receipt(signer, receipt_code)

    worker.start.side_effect = publish
    outcome = supervise(worker, tmp_path, None, lambda _: None, completion_contract=contract)
    assert outcome["status"] == status
    assert outcome["exit_code"] == raw_exit
    assert outcome["public_exit_code"] == public_exit
    assert outcome["completion_receipt_code"] == (None if stale_nonce else receipt_code)
    assert outcome["teardown_fail_fast_contained"] is contained
    assert outcome["result"] == ({"k": 5} if has_result else {})
    assert json.loads((tmp_path / "watchdog.json").read_bytes()) == outcome
