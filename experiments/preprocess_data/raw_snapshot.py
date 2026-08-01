"""Fail-closed acquisition helpers for registered raw CSV snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import uuid

import yaml


@dataclass(frozen=True)
class RegisteredRawSnapshot:
    """The byte-level raw-data authority declared by one dataset config."""

    dataset_id: str
    path: Path
    sha256: str
    byte_count: int


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_registered_snapshot(
    config_path: Path, *, expected_dataset_id: str
) -> RegisteredRawSnapshot:
    """Load only the raw-snapshot fields needed by an acquisition script."""
    source = Path(config_path).resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read dataset config: {source}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("dataset"), dict):
        raise RuntimeError(f"dataset config is malformed: {source}")
    dataset = document["dataset"]
    dataset_id = dataset.get("id")
    raw_path = dataset.get("raw_path")
    digest = dataset.get("raw_sha256")
    byte_count = dataset.get("raw_byte_count")
    if dataset_id != expected_dataset_id:
        raise RuntimeError(
            f"dataset config ID mismatch: expected {expected_dataset_id!r}, "
            f"got {dataset_id!r}"
        )
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise RuntimeError(f"dataset raw_path is invalid: {source}")
    if not _is_sha256(digest):
        raise RuntimeError(f"dataset raw_sha256 is invalid: {source}")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise RuntimeError(f"dataset raw_byte_count is invalid: {source}")
    path = Path(raw_path)
    project_root = source.parent.parent.resolve()
    raw_root = (project_root / "data" / "raw").resolve()
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".csv":
        raise RuntimeError(
            f"dataset raw_path must be a relative CSV inside data/raw: {source}"
        )
    path = Path(os.path.abspath(project_root / path))
    if path.parent != raw_root:
        raise RuntimeError(
            f"dataset raw_path must be a CSV directly inside data/raw: {source}"
        )
    return RegisteredRawSnapshot(
        dataset_id=dataset_id,
        path=path,
        sha256=digest,
        byte_count=byte_count,
    )


def verify_registered_snapshot(
    snapshot: RegisteredRawSnapshot, *, candidate: Path | None = None
) -> None:
    """Reject any candidate whose bytes differ from the registered authority."""
    path = snapshot.path if candidate is None else Path(candidate)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"registered raw snapshot must be a regular file for "
            f"{snapshot.dataset_id}: {path}"
        )
    try:
        byte_count = path.stat().st_size
        digest_builder = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest_builder.update(chunk)
    except OSError as exc:
        raise RuntimeError(
            f"registered raw snapshot is missing for {snapshot.dataset_id}: {path}"
        ) from exc
    digest = digest_builder.hexdigest()
    if byte_count != snapshot.byte_count or digest != snapshot.sha256:
        raise RuntimeError(
            "registered raw snapshot mismatch for "
            f"{snapshot.dataset_id}: expected bytes={snapshot.byte_count} "
            f"sha256={snapshot.sha256}, got bytes={byte_count} sha256={digest}; "
            "the existing file was left unchanged"
        )


def materialize_registered_snapshot(
    snapshot: RegisteredRawSnapshot,
    producer: Callable[[Path], object],
) -> Path:
    """Reuse a valid cache or atomically publish a verified new download."""
    if snapshot.path.exists() or snapshot.path.is_symlink():
        verify_registered_snapshot(snapshot)
        print(f"Using registered cache {snapshot.path}")
        return snapshot.path

    snapshot.path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.path.parent.is_symlink() or not snapshot.path.parent.is_dir():
        raise RuntimeError(
            "registered raw snapshot parent must be a regular directory for "
            f"{snapshot.dataset_id}: {snapshot.path.parent}"
        )
    staging = snapshot.path.with_name(f".{snapshot.path.name}.{uuid.uuid4().hex}.tmp")
    published = True
    try:
        producer(staging)
        verify_registered_snapshot(snapshot, candidate=staging)
        try:
            os.link(staging, snapshot.path)
        except FileExistsError:
            published = False
            verify_registered_snapshot(snapshot)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    action = "Published registered cache" if published else "Using registered cache"
    print(f"{action} {snapshot.path}")
    return snapshot.path
