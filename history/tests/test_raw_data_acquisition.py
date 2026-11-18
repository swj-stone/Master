from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parents[1]
PREPROCESSOR_DIR = PROJECT_ROOT / "experiments" / "preprocess_data"


def _load_script(filename: str):
    path = PREPROCESSOR_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset_config(
    project_root: Path,
    *,
    filename: str,
    dataset_id: str,
    raw_name: str,
    payload: bytes,
) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / filename
    config_path.write_text(
        "\n".join(
            (
                "dataset:",
                f"  id: {dataset_id}",
                f"  raw_path: data/raw/{raw_name}",
                f"  raw_sha256: {hashlib.sha256(payload).hexdigest()}",
                f"  raw_byte_count: {len(payload)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize(
    ("script_name", "config_name", "dataset_id", "raw_name", "payload"),
    (
        (
            "preprocess_Spambase.py",
            "spambase.yaml",
            "spambase",
            "spambase.csv",
            b"word_freq_make,is_spam\n0.0,0\n0.0,0\n",
        ),
        (
            "preprocess_Credit-Card-Fraud.py",
            "credit_card_fraud.yaml",
            "credit_card_fraud",
            "credit_card_fraud.csv",
            b"V1,Amount,Class\n0.0,10.0,0\n0.0,10.0,0\n",
        ),
    ),
)
def test_registered_existing_raw_snapshot_is_never_rewritten(
    tmp_path: Path,
    script_name: str,
    config_name: str,
    dataset_id: str,
    raw_name: str,
    payload: bytes,
) -> None:
    """Catches acquisition scripts rewriting a registered cache via deduplication."""
    _write_dataset_config(
        tmp_path,
        filename=config_name,
        dataset_id=dataset_id,
        raw_name=raw_name,
        payload=payload,
    )
    raw_path = tmp_path / "data" / "raw" / raw_name
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(payload)
    module = _load_script(script_name)

    module.main(project_root=tmp_path)

    assert raw_path.read_bytes() == payload


def test_mismatched_existing_raw_snapshot_fails_closed_without_rewrite(
    tmp_path: Path,
) -> None:
    """Catches a mismatched cache being silently normalized or replaced."""
    registered = b"word_freq_make,is_spam\n0.0,0\n0.0,0\n"
    mismatched = b"word_freq_make,is_spam\n0.0,0\n"
    _write_dataset_config(
        tmp_path,
        filename="spambase.yaml",
        dataset_id="spambase",
        raw_name="spambase.csv",
        payload=registered,
    )
    raw_path = tmp_path / "data" / "raw" / "spambase.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(mismatched)
    module = _load_script("preprocess_Spambase.py")

    with pytest.raises(RuntimeError, match="registered raw snapshot"):
        module.main(project_root=tmp_path)

    assert raw_path.read_bytes() == mismatched


def test_missing_raw_snapshot_is_published_only_after_integrity_check(
    tmp_path: Path,
) -> None:
    """Catches partial or wrong downloads being published as the raw cache."""
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )

    payload = b"feature,target\n1,0\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="toy.csv",
        payload=payload,
    )
    snapshot = load_registered_snapshot(config_path, expected_dataset_id="toy")

    path = materialize_registered_snapshot(
        snapshot, lambda staging: staging.write_bytes(payload)
    )

    assert path == tmp_path / "data" / "raw" / "toy.csv"
    assert path.read_bytes() == payload
    assert not tuple(path.parent.glob(".toy.csv.*.tmp"))


def test_wrong_download_is_removed_without_publishing_target(tmp_path: Path) -> None:
    """Catches a source change bypassing the registered snapshot authority."""
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )

    expected = b"feature,target\n1,0\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="toy.csv",
        payload=expected,
    )
    snapshot = load_registered_snapshot(config_path, expected_dataset_id="toy")

    with pytest.raises(RuntimeError, match="registered raw snapshot"):
        materialize_registered_snapshot(
            snapshot, lambda staging: staging.write_bytes(b"wrong")
        )

    assert not snapshot.path.exists()
    assert not tuple(snapshot.path.parent.glob(".toy.csv.*.tmp"))


def test_concurrent_conflicting_winner_is_never_overwritten(tmp_path: Path) -> None:
    """Catches a competing publisher being overwritten after candidate validation."""
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )

    expected = b"feature,target\n1,0\n"
    conflicting = b"conflicting concurrent publisher\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="toy.csv",
        payload=expected,
    )
    snapshot = load_registered_snapshot(config_path, expected_dataset_id="toy")

    def publish_conflict_after_download(staging: Path) -> None:
        staging.write_bytes(expected)
        snapshot.path.write_bytes(conflicting)

    with pytest.raises(RuntimeError, match="registered raw snapshot"):
        materialize_registered_snapshot(snapshot, publish_conflict_after_download)

    assert snapshot.path.read_bytes() == conflicting
    assert not tuple(snapshot.path.parent.glob(".toy.csv.*.tmp"))


def test_concurrent_identical_winner_is_reused_without_replacement(
    tmp_path: Path,
) -> None:
    """Catches replacement of an identical cache won by another process."""
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )

    payload = b"feature,target\n1,0\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="toy.csv",
        payload=payload,
    )
    snapshot = load_registered_snapshot(config_path, expected_dataset_id="toy")
    winner = tmp_path / "concurrent-winner.csv"
    winner.write_bytes(payload)
    hardlink_probe = tmp_path / "hardlink-probe.csv"
    try:
        os.link(winner, hardlink_probe)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    hardlink_probe.unlink()

    def publish_identical_after_download(staging: Path) -> None:
        staging.write_bytes(payload)
        os.link(winner, snapshot.path)

    path = materialize_registered_snapshot(snapshot, publish_identical_after_download)

    assert path == snapshot.path
    assert os.path.samefile(winner, snapshot.path)
    assert not tuple(snapshot.path.parent.glob(".toy.csv.*.tmp"))


def test_registered_raw_path_must_stay_inside_project_data_raw(
    tmp_path: Path,
) -> None:
    """Catches a tracked config redirecting acquisition outside data/raw."""
    from experiments.preprocess_data.raw_snapshot import load_registered_snapshot

    payload = b"feature,target\n1,0\n"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "toy.yaml"
    config_path.write_text(
        "\n".join(
            (
                "dataset:",
                "  id: toy",
                "  raw_path: ../outside.csv",
                f"  raw_sha256: {hashlib.sha256(payload).hexdigest()}",
                f"  raw_byte_count: {len(payload)}",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="data/raw"):
        load_registered_snapshot(config_path, expected_dataset_id="toy")


def test_registered_raw_path_must_be_direct_child_of_data_raw(
    tmp_path: Path,
) -> None:
    """Catches nested configured paths escaping through a directory symlink."""
    from experiments.preprocess_data.raw_snapshot import load_registered_snapshot

    payload = b"feature,target\n1,0\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="nested/toy.csv",
        payload=payload,
    )

    with pytest.raises(RuntimeError, match="directly inside data/raw"):
        load_registered_snapshot(config_path, expected_dataset_id="toy")


def test_registered_raw_snapshot_rejects_symbolic_link(tmp_path: Path) -> None:
    """Catches a valid digest outside the project being accepted through a link."""
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )

    payload = b"feature,target\n1,0\n"
    config_path = _write_dataset_config(
        tmp_path,
        filename="toy.yaml",
        dataset_id="toy",
        raw_name="toy.csv",
        payload=payload,
    )
    outside = tmp_path / "outside.csv"
    outside.write_bytes(payload)
    raw_path = tmp_path / "data" / "raw" / "toy.csv"
    raw_path.parent.mkdir(parents=True)
    try:
        raw_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    snapshot = load_registered_snapshot(config_path, expected_dataset_id="toy")

    with pytest.raises(RuntimeError, match="regular file"):
        materialize_registered_snapshot(snapshot, lambda _: None)

    assert raw_path.is_symlink()
    assert outside.read_bytes() == payload


def test_credit_acquisition_pins_openml_id_and_crlf_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches mutable name lookup or host-dependent line endings changing bytes."""
    module = _load_script("preprocess_Credit-Card-Fraud.py")
    source_columns = [f"source_{index}" for index in range(29)]
    frame = pd.DataFrame([[*([0.0] * 29), "0"]], columns=[*source_columns, "Class"])
    header = ",".join([*(f"V{i}" for i in range(1, 29)), "Amount", "Class"])
    row = ",".join([*("0.0" for _ in range(29)), "0"])
    payload = f"{header}\r\n{row}\r\n".encode("utf-8")
    _write_dataset_config(
        tmp_path,
        filename="credit_card_fraud.yaml",
        dataset_id="credit_card_fraud",
        raw_name="credit_card_fraud.csv",
        payload=payload,
    )
    observed: dict[str, object] = {}

    def fake_fetch_openml(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(module, "fetch_openml", fake_fetch_openml)

    module.main(project_root=tmp_path)

    assert observed == {
        "args": (),
        "kwargs": {
            "data_id": 1597,
            "as_frame": True,
            "parser": "auto",
            "cache": False,
        },
    }
    assert (tmp_path / "data" / "raw" / "credit_card_fraud.csv").read_bytes() == payload
