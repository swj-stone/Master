"""Tests for the native worker completion contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import TracebackType
from typing import TextIO

import pytest

import experiments.native_process_guard as guard_module
from experiments.native_process_guard import (
    RECEIPT_SCHEMA_VERSION,
    WorkerContract,
    decide_public_exit,
    read_worker_receipt,
    worker_contract,
    write_worker_receipt,
)


def test_standalone_guard_import_does_not_load_xgboost() -> None:
    project_root = Path(__file__).parents[1]
    probe = """
import sys
import experiments.native_process_guard
print('XGBOOST_LOADED=' + str('xgboost' in sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "XGBOOST_LOADED=False" in completed.stdout


def test_worker_contract_requires_path_instance() -> None:
    with pytest.raises(TypeError, match="receipt_path"):
        WorkerContract("receipt.json", "a" * 64)  # type: ignore[arg-type]


def test_worker_contract_requires_string_nonce(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="nonce"):
        WorkerContract(tmp_path / "receipt.json", 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("nonce", ["a" * 63, "A" * 64, "g" * 64, "é" * 64])
def test_worker_contract_requires_lowercase_hex_nonce(
    tmp_path: Path,
    nonce: str,
) -> None:
    with pytest.raises(ValueError, match="nonce"):
        WorkerContract(tmp_path / "receipt.json", nonce)


def test_worker_contract_requires_complete_private_environment(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize(
    ("marker", "nonce"),
    [
        ("0", "a" * 64),
        ("true", "a" * 64),
        ("1", "a" * 63),
        ("1", "A" * 64),
        ("1", "g" * 64),
    ],
)
def test_worker_contract_rejects_invalid_complete_environment(
    tmp_path: Path,
    marker: str,
    nonce: str,
) -> None:
    with pytest.raises(ValueError):
        worker_contract(
            {
                "MINPPI_PIPELINE_WORKER": marker,
                "MINPPI_PIPELINE_RECEIPT": str(tmp_path / "receipt.json"),
                "MINPPI_PIPELINE_NONCE": nonce,
            }
        )


def test_worker_contract_rejects_directory_receipt_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="receipt path"):
        worker_contract(
            {
                "MINPPI_PIPELINE_WORKER": "1",
                "MINPPI_PIPELINE_RECEIPT": str(tmp_path),
                "MINPPI_PIPELINE_NONCE": "a" * 64,
            }
        )


def test_receipt_round_trip_is_schema_and_nonce_bound(tmp_path: Path) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)

    write_worker_receipt(contract, 75)

    assert read_worker_receipt(contract) == 75
    receipt_text = contract.receipt_path.read_text(encoding="utf-8")
    assert receipt_text.endswith("\n")
    assert json.loads(receipt_text) == {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "nonce": "b" * 64,
        "exit_code": 75,
    }


def test_write_worker_receipt_fsyncs_before_sibling_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    real_open = Path.open
    real_fsync = os.fsync
    real_replace = os.replace
    events: list[str] = []
    replace_arguments: list[tuple[Path, Path]] = []

    class RecordingHandle:
        def __init__(self, handle: TextIO) -> None:
            self._handle = handle

        def __enter__(self) -> RecordingHandle:
            self._handle.__enter__()
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            return self._handle.__exit__(exception_type, exception, traceback)

        def write(self, text: str) -> int:
            return self._handle.write(text)

        def flush(self) -> None:
            events.append("flush")
            self._handle.flush()

        def fileno(self) -> int:
            return self._handle.fileno()

    def record_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> RecordingHandle:
        return RecordingHandle(real_open(path, *args, **kwargs))

    def record_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        real_fsync(file_descriptor)

    def record_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        events.append("replace")
        replace_arguments.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(guard_module.os, "fsync", record_fsync)
    monkeypatch.setattr(guard_module.os, "replace", record_replace)

    write_worker_receipt(contract, 0)

    assert events == ["flush", "fsync", "replace"]
    assert len(replace_arguments) == 1
    source, destination = replace_arguments[0]
    assert source.parent == contract.receipt_path.parent
    assert source != contract.receipt_path
    assert destination == contract.receipt_path


@pytest.mark.parametrize("failure_point", ["json.dump", "os.fsync", "os.replace"])
def test_write_worker_receipt_failure_preserves_target_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    original_receipt = "existing receipt\n"
    contract.receipt_path.write_text(original_receipt, encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"{failure_point} failed")

    owner_name, attribute_name = failure_point.split(".")
    owner = guard_module.json if owner_name == "json" else guard_module.os
    monkeypatch.setattr(owner, attribute_name, fail)

    with pytest.raises(OSError, match=failure_point):
        write_worker_receipt(contract, 0)

    assert contract.receipt_path.read_text(encoding="utf-8") == original_receipt
    assert list(tmp_path.iterdir()) == [contract.receipt_path]


@pytest.mark.parametrize("exit_code", [True, 1.0, "1"])
def test_write_worker_receipt_requires_an_integer(
    tmp_path: Path,
    exit_code: object,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)

    with pytest.raises(TypeError, match="exit_code"):
        write_worker_receipt(contract, exit_code)  # type: ignore[arg-type]


@pytest.mark.parametrize("exit_code", [-1, 256])
def test_write_worker_receipt_rejects_out_of_range_exit_code(
    tmp_path: Path,
    exit_code: int,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)

    with pytest.raises(ValueError, match="0..255"):
        write_worker_receipt(contract, exit_code)


@pytest.mark.parametrize(
    "receipt_text",
    [
        None,
        "{",
        "[]",
        json.dumps(
            {"schema_version": 2, "nonce": "b" * 64, "exit_code": 0}
        ),
        json.dumps(
            {"schema_version": True, "nonce": "b" * 64, "exit_code": 0}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "c" * 64, "exit_code": 0}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "b" * 64, "exit_code": True}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "b" * 64, "exit_code": 1.0}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "b" * 64, "exit_code": "0"}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "b" * 64, "exit_code": -1}
        ),
        json.dumps(
            {"schema_version": 1, "nonce": "b" * 64, "exit_code": 256}
        ),
        json.dumps(
            {
                "schema_version": 1,
                "nonce": "b" * 64,
                "exit_code": 0,
                "extra": "ambiguous",
            }
        ),
        json.dumps({"schema_version": 1, "nonce": "b" * 64}),
    ],
    ids=[
        "missing-file",
        "malformed-json",
        "non-object-json",
        "wrong-schema",
        "boolean-schema",
        "wrong-nonce",
        "boolean-exit",
        "float-exit",
        "string-exit",
        "negative-exit",
        "exit-over-255",
        "extra-key",
        "missing-key",
    ],
)
def test_read_worker_receipt_rejects_ambiguous_evidence(
    tmp_path: Path,
    receipt_text: str | None,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    if receipt_text is not None:
        contract.receipt_path.write_text(receipt_text, encoding="utf-8")

    assert read_worker_receipt(contract) is None


def test_read_worker_receipt_rejects_non_ascii_nonce(tmp_path: Path) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    contract.receipt_path.write_text(
        json.dumps(
            {"schema_version": 1, "nonce": "é" * 64, "exit_code": 0}
        ),
        encoding="utf-8",
    )

    assert read_worker_receipt(contract) is None


def test_read_worker_receipt_rejects_json_integer_over_python_limit(
    tmp_path: Path,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    contract.receipt_path.write_text(
        '{"schema_version": 1, "nonce": "'
        + "b" * 64
        + '", "exit_code": '
        + "9" * 5000
        + "}",
        encoding="utf-8",
    )

    assert read_worker_receipt(contract) is None


def test_read_worker_receipt_rejects_deeply_nested_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = WorkerContract(tmp_path / "receipt.json", "b" * 64)
    contract.receipt_path.write_text(
        "[" * 5000 + "0" + "]" * 5000,
        encoding="utf-8",
    )

    def reject_deep_json(_document: str) -> object:
        raise RecursionError("JSON nesting is too deep")

    monkeypatch.setattr(guard_module.json, "loads", reject_deep_json)

    assert read_worker_receipt(contract) is None


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


@pytest.mark.parametrize("receipt_code", [None, 2, 75])
def test_zero_exit_without_valid_matching_receipt_fails_closed(
    receipt_code: int | None,
) -> None:
    assert decide_public_exit(0, receipt_code) == (70, False)


def test_ordinary_nonzero_failure_is_propagated_without_matching_receipt() -> None:
    assert decide_public_exit(2, None) == (2, False)
    assert decide_public_exit(75, 0) == (75, False)


def test_unrelated_native_failure_is_never_contained() -> None:
    assert decide_public_exit(-1073741819, 0) == (-1073741819, False)


def test_isolated_cli_returns_zero_and_removes_receipt_directory(
    tmp_path: Path,
) -> None:
    script = tmp_path / "pipeline.py"
    receipt_directories: list[Path] = []
    invocations = 0

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal invocations
        invocations += 1
        assert command == [
            "test-python",
            str(script.resolve()),
            "--stage",
            "validate",
        ]
        assert set(kwargs) == {"env", "check"}
        assert kwargs["check"] is False
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CALLER_VALUE"] == "preserved"
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        write_worker_receipt(contract, 0)
        return subprocess.CompletedProcess(command, 0)

    result = guard_module.run_isolated_cli(
        script,
        ("--stage", "validate"),
        executable="test-python",
        environ={"CALLER_VALUE": "preserved"},
        process_runner=process_runner,
    )

    assert result == 0
    assert invocations == 1
    assert len(receipt_directories) == 1
    assert not receipt_directories[0].exists()


def test_isolated_cli_propagates_controlled_stop_75(tmp_path: Path) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        write_worker_receipt(contract, 75)
        return subprocess.CompletedProcess(command, 75)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 75
    assert not receipt_directories[0].exists()


def test_isolated_cli_contains_fail_fast_after_valid_zero_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        write_worker_receipt(contract, 0)
        return subprocess.CompletedProcess(command, -1073740791)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 0
    assert not receipt_directories[0].exists()
    stderr = capsys.readouterr().err
    assert "0xC0000409" in stderr
    assert "pipeline main() completed" in stderr


def test_isolated_cli_propagates_fail_fast_without_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        return subprocess.CompletedProcess(command, 3221226505)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 3221226505
    assert not receipt_directories[0].exists()
    stderr = capsys.readouterr().err
    assert stderr.count("\n") == 1
    assert "0xC0000409" in stderr
    assert "refusing to contain" in stderr
    assert "valid zero completion receipt" in stderr


def test_isolated_cli_rejects_stale_nonce_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        stale_nonce = ("0" if contract.nonce[0] != "0" else "1") + contract.nonce[1:]
        write_worker_receipt(
            WorkerContract(contract.receipt_path, stale_nonce),
            0,
        )
        return subprocess.CompletedProcess(command, -1073740791)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == -1073740791
    assert not receipt_directories[0].exists()
    stderr = capsys.readouterr().err
    assert stderr.count("\n") == 1
    assert "0xC0000409" in stderr
    assert "refusing to contain" in stderr


@pytest.mark.parametrize("receipt_kind", ["malformed", "nonzero"])
def test_isolated_cli_refuses_fail_fast_with_unacceptable_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    receipt_kind: str,
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        if receipt_kind == "malformed":
            contract.receipt_path.write_text("{", encoding="utf-8")
        else:
            write_worker_receipt(contract, 75)
        return subprocess.CompletedProcess(command, -1073740791)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == -1073740791
    assert not receipt_directories[0].exists()
    stderr = capsys.readouterr().err
    assert stderr.count("\n") == 1
    assert "0xC0000409" in stderr
    assert "refusing to contain" in stderr
    assert "valid zero completion receipt" in stderr


def test_isolated_cli_propagates_ordinary_nonzero_without_receipt(
    tmp_path: Path,
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        return subprocess.CompletedProcess(command, 2)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 2
    assert not receipt_directories[0].exists()


def test_isolated_cli_returns_70_when_zero_worker_omits_receipt(
    tmp_path: Path,
) -> None:
    receipt_directories: list[Path] = []

    def process_runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        return subprocess.CompletedProcess(command, 0)

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 70
    assert not receipt_directories[0].exists()


def test_isolated_cli_returns_70_when_process_launch_raises_oserror(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_directories: list[Path] = []
    invocations = 0

    def process_runner(
        _command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal invocations
        invocations += 1
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        contract = worker_contract(environment)
        assert contract is not None
        receipt_directories.append(contract.receipt_path.parent)
        raise OSError("launch unavailable")

    result = guard_module.run_isolated_cli(
        tmp_path / "pipeline.py",
        (),
        process_runner=process_runner,
    )

    assert result == 70
    assert invocations == 1
    assert not receipt_directories[0].exists()
    assert "launch unavailable" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mode", "expected_returncode"),
    [("0", 0), ("75", 75), ("missing", 70)],
)
def test_real_parent_worker_round_trip_and_cleanup(
    tmp_path: Path,
    mode: str,
    expected_returncode: int,
) -> None:
    project_root = Path(__file__).parents[1]
    helper = tmp_path / "isolated_helper.py"
    worker_log = tmp_path / f"workers-{mode}.txt"
    helper.write_text(
        """
from __future__ import annotations

import os
import sys
from pathlib import Path

from experiments.native_process_guard import run_isolated_cli, worker_contract

arguments = tuple(sys.argv[1:])
contract = worker_contract(os.environ)
if contract is None:
    raise SystemExit(run_isolated_cli(Path(__file__), arguments))

with Path(os.environ["WORKER_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write("worker\\n")
    handle.write(str(contract.receipt_path.parent) + "\\n")
if arguments == ("missing",):
    raise SystemExit(0)

import experiments.run_icomp2026_pipeline as pipeline

pipeline.main = lambda _argv: int(arguments[0])
raise SystemExit(pipeline._entrypoint(arguments))
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath
    environment["TEMP"] = str(tmp_path)
    environment["TMP"] = str(tmp_path)
    environment["TMPDIR"] = str(tmp_path)
    environment["WORKER_LOG"] = str(worker_log)

    completed = subprocess.run(
        [sys.executable, str(helper), mode],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    worker_lines = worker_log.read_text(encoding="utf-8").splitlines()
    assert worker_lines[0] == "worker"
    assert len(worker_lines) == 2
    receipt_directory = Path(worker_lines[1])
    assert receipt_directory.parent == tmp_path
    assert receipt_directory.name.startswith("minppi-pipeline-")
    assert not receipt_directory.exists()
    assert list(tmp_path.glob("minppi-pipeline-*")) == []
