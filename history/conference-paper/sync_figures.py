"""Validate and transactionally copy the canonical formal figure bundle.

After a publication-eligible campaign finishes, this utility verifies the
campaign-bound manifest and copies its three canonical PDFs into
``conference-paper/figures`` as one complete directory transaction.

Run from the project root with the existing project interpreter::

    ../venv/Scripts/python.exe conference-paper/sync_figures.py --source <campaign>/figures/paper --check
    ../venv/Scripts/python.exe conference-paper/sync_figures.py --source <campaign>/figures/paper
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Canonical figure names emitted by the formal summary. All must be PDF.
REQUIRED_FIGURES: tuple[str, ...] = (
    "fig1_overview.pdf",
    "fig2_cd_heatmap.pdf",
    "fig3_prefilter.pdf",
)
FIGURE_MANIFEST = "figure_bundle_manifest.json"
CAMPAIGN_SUMMARY = "campaign_summary.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FigureBundleError(ValueError):
    """A formal figure bundle failed closed validation."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    paper_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Sync paper figures into conference-paper/figures/"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
        help="Formal <campaign>/figures/paper directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=paper_root / "figures",
        help="Destination directory (default: conference-paper/figures)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the formal bundle without copying it",
    )
    return parser.parse_args()


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FigureBundleError(f"{label} is not a regular file: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FigureBundleError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise FigureBundleError(f"{label} must be a JSON object")
    return document


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _validate_manifest_schema(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        set(manifest) != {"schema_version", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["files"], list)
    ):
        raise FigureBundleError("figure bundle manifest schema is invalid")

    entries = manifest["files"]
    if len(entries) != len(REQUIRED_FIGURES):
        raise FigureBundleError("figure bundle manifest file list is incomplete")
    for expected_name, entry in zip(REQUIRED_FIGURES, entries, strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"filename", "sha256", "byte_length"}
            or entry.get("filename") != expected_name
            or type(entry.get("byte_length")) is not int
            or entry["byte_length"] <= 0
            or not _is_sha256(entry.get("sha256"))
        ):
            raise FigureBundleError("figure bundle manifest file metadata is invalid")
    return entries


def _validate_campaign_summary(source: Path, manifest_payload: bytes) -> None:
    summary_path = source.parents[1] / CAMPAIGN_SUMMARY
    summary = _read_json_object(summary_path, label=CAMPAIGN_SUMMARY)
    if (
        type(summary.get("schema_version")) is not int
        or summary["schema_version"] != 1
    ):
        raise FigureBundleError("campaign_summary.json schema_version must be 1")
    if summary.get("publication_allowed") is not True:
        raise FigureBundleError("campaign_summary.json publication_allowed is not true")
    expected_digest = summary.get("figure_bundle_manifest_sha256")
    if not _is_sha256(expected_digest):
        raise FigureBundleError(
            "campaign_summary.json figure_bundle_manifest_sha256 is invalid"
        )
    actual_digest = hashlib.sha256(manifest_payload).hexdigest()
    if expected_digest != actual_digest:
        raise FigureBundleError(
            "campaign_summary.json figure_bundle_manifest_sha256 does not match"
        )


def validate_formal_bundle(source: Path) -> dict[str, bytes]:
    """Return verified PDF payloads for one publication-eligible campaign."""
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise FigureBundleError(f"formal figure bundle is not a regular directory: {source}")

    entries = tuple(source.iterdir())
    expected_names = {*REQUIRED_FIGURES, FIGURE_MANIFEST}
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise FigureBundleError(
            "formal figure bundle must contain the exact regular-file PDF set "
            "and figure_bundle_manifest.json"
        )

    manifest_path = source / FIGURE_MANIFEST
    manifest_payload = manifest_path.read_bytes()
    manifest = _read_json_object(manifest_path, label="figure bundle manifest")
    manifest_entries = _validate_manifest_schema(manifest)

    payloads: dict[str, bytes] = {}
    for entry in manifest_entries:
        filename = entry["filename"]
        payload = (source / filename).read_bytes()
        if not payload.startswith(b"%PDF-"):
            raise FigureBundleError(f"{filename} does not start with %PDF-")
        if len(payload) != entry["byte_length"]:
            raise FigureBundleError(f"{filename} byte length does not match manifest")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise FigureBundleError(f"{filename} sha256 does not match manifest")
        payloads[filename] = payload

    _validate_campaign_summary(source, manifest_payload)
    return payloads


def _restore_previous_destination(
    *, staging_path: Path, destination: Path, backup_path: Path | None
) -> None:
    """Restore the prior directory by inspecting which rename took effect."""
    if backup_path is None:
        if destination.exists():
            if staging_path.exists():
                raise RuntimeError(
                    "cannot safely undo first sync: both staging and "
                    "destination exist"
                )
            os.replace(destination, staging_path)
        return
    if not backup_path.exists():
        return
    if destination.exists():
        if staging_path.exists():
            raise RuntimeError(
                "cannot safely restore destination: both staging and destination exist"
            )
        os.replace(destination, staging_path)
    os.replace(backup_path, destination)


def _sync_verified_payloads(
    payloads: dict[str, bytes], *, destination: Path
) -> None:
    """Replace *destination* atomically, retaining the old set for rollback."""
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        not destination.is_dir() or destination.is_symlink()
    ):
        raise FigureBundleError(
            f"destination must be a regular directory when it exists: {destination}"
        )

    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-staging-", dir=destination_parent)
    )
    backup_path: Path | None = None
    try:
        for name in REQUIRED_FIGURES:
            (staging_path / name).write_bytes(payloads[name])

        if destination.exists():
            backup_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}-backup-", dir=destination_parent
                )
            )
            backup_path.rmdir()

        try:
            if backup_path is not None:
                os.replace(destination, backup_path)
            os.replace(staging_path, destination)
        except BaseException:
            _restore_previous_destination(
                staging_path=staging_path,
                destination=destination,
                backup_path=backup_path,
            )
            raise

        if backup_path is not None:
            shutil.rmtree(backup_path)
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)


def main() -> int:
    """Validate before optionally synchronising a complete formal figure set."""
    args = parse_args()
    try:
        payloads = validate_formal_bundle(args.source)
    except FigureBundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            f"{len(payloads)}/{len(REQUIRED_FIGURES)} formal figures valid "
            f"in {args.source}"
        )
        return 0

    try:
        _sync_verified_payloads(payloads, destination=args.dest)
    except FigureBundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for name in REQUIRED_FIGURES:
        print(f"  copied  {name}")
    print(
        f"\n{len(payloads)}/{len(REQUIRED_FIGURES)} figures synced to {args.dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
