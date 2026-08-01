"""Behavioural tests for the paper-figure synchronisation CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "conference-paper" / "sync_figures.py"
CANONICAL_FIGURES = (
    "fig1_overview.pdf",
    "fig2_cd_heatmap.pdf",
    "fig3_prefilter.pdf",
)
MANIFEST_NAME = "figure_bundle_manifest.json"


def load_script_module():
    """Load the CLI module for direct transaction-failure testing."""
    spec = importlib.util.spec_from_file_location("sync_figures", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sync(*args: Path | str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the real CLI against isolated directories."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _manifest_for(source: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "files": [
            {
                "filename": name,
                "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
                "byte_length": (source / name).stat().st_size,
            }
            for name in CANONICAL_FIGURES
        ],
    }


def _write_manifest(
    source: Path,
    manifest: object,
    *,
    update_campaign_summary: bool = True,
) -> None:
    payload = _canonical_json_bytes(manifest)
    (source / MANIFEST_NAME).write_bytes(payload)
    if update_campaign_summary:
        summary = {
            "schema_version": 1,
            "publication_allowed": True,
            "figure_bundle_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }
        (source.parents[1] / "campaign_summary.json").write_bytes(
            _canonical_json_bytes(summary)
        )


def write_formal_source(tmp_path: Path) -> Path:
    """Create the real campaign-relative bundle shape accepted by the CLI."""
    source = tmp_path / "campaign" / "figures" / "paper"
    source.mkdir(parents=True)
    for index, name in enumerate(CANONICAL_FIGURES, start=1):
        (source / name).write_bytes(
            f"%PDF-1.4\nfixture-{index}\n%%EOF\n".encode("ascii")
        )
    _write_manifest(source, _manifest_for(source))
    return source


def test_uses_exact_canonical_figure_set() -> None:
    """Changing or omitting a paper figure must fail this contract check."""
    module = load_script_module()

    assert module.REQUIRED_FIGURES == CANONICAL_FIGURES


def test_defaults_are_anchored_to_project_root_not_working_directory(
    monkeypatch, tmp_path: Path
) -> None:
    """Launching elsewhere must still locate the project's figure locations."""
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    args = module.parse_args()

    assert args.source == PROJECT_ROOT / "results" / "figures"
    assert args.dest == PROJECT_ROOT / "conference-paper" / "figures"


def test_missing_figure_fails_without_creating_partial_destination(tmp_path: Path) -> None:
    """Validation must complete before a destination is created or populated."""
    source = write_formal_source(tmp_path)
    (source / CANONICAL_FIGURES[1]).unlink()
    destination = tmp_path / "destination"

    result = run_sync("--source", source, "--dest", destination, cwd=tmp_path)

    assert result.returncode != 0
    assert not destination.exists()
    assert "formal figure bundle" in result.stderr.lower()


def test_extra_bundle_entry_is_rejected(tmp_path: Path) -> None:
    """A stale or untracked source file must not enter a formal bundle."""
    source = write_formal_source(tmp_path)
    (source / "notes.txt").write_text("not registered", encoding="utf-8")

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "exact" in result.stderr.lower()


def test_symlinked_pdf_is_rejected(tmp_path: Path) -> None:
    """Every source artifact must be a regular file, never a symlink."""
    source = write_formal_source(tmp_path)
    figure = source / CANONICAL_FIGURES[0]
    target = tmp_path / "outside.pdf"
    target.write_bytes(figure.read_bytes())
    figure.unlink()
    try:
        figure.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "regular" in result.stderr.lower()


def test_pdf_magic_is_required_even_when_manifest_matches(tmp_path: Path) -> None:
    """Hash-consistent arbitrary bytes must not be accepted as a PDF."""
    source = write_formal_source(tmp_path)
    (source / CANONICAL_FIGURES[0]).write_bytes(b"plain text pretending to be PDF")
    _write_manifest(source, _manifest_for(source))

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "%pdf-" in result.stderr.lower()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update({"unexpected": True}),
        lambda manifest: manifest["files"].reverse(),
        lambda manifest: manifest["files"][0].update({"unexpected": True}),
        lambda manifest: manifest["files"][0].update({"byte_length": True}),
        lambda manifest: manifest["files"][0].update({"sha256": "not-a-digest"}),
    ],
    ids=(
        "manifest-extra-key",
        "wrong-entry-order",
        "entry-extra-key",
        "boolean-byte-length",
        "invalid-sha256",
    ),
)
def test_manifest_schema_is_fail_closed(tmp_path: Path, mutate) -> None:
    """Malformed metadata must be rejected before any paper file is touched."""
    source = write_formal_source(tmp_path)
    manifest = _manifest_for(source)
    mutate(manifest)
    _write_manifest(source, manifest)

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


@pytest.mark.parametrize("field", ["byte_length", "sha256"])
def test_manifest_must_match_each_pdf(tmp_path: Path, field: str) -> None:
    """A stale PDF length or digest must invalidate the whole bundle."""
    source = write_formal_source(tmp_path)
    manifest = _manifest_for(source)
    if field == "byte_length":
        manifest["files"][1][field] += 1
    else:
        manifest["files"][1][field] = "0" * 64
    _write_manifest(source, manifest)

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert field.replace("_", " ") in result.stderr.lower()


def test_missing_campaign_summary_is_rejected(tmp_path: Path) -> None:
    """A bundle without its campaign publication decision is not publishable."""
    source = write_formal_source(tmp_path)
    (source.parents[1] / "campaign_summary.json").unlink()

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "campaign_summary.json" in result.stderr


def test_campaign_summary_must_allow_publication(tmp_path: Path) -> None:
    """Engineering-only or incomplete campaigns must not update the paper."""
    source = write_formal_source(tmp_path)
    summary_path = source.parents[1] / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_allowed"] = False
    summary_path.write_bytes(_canonical_json_bytes(summary))

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "publication_allowed" in result.stderr


def test_campaign_summary_must_bind_exact_manifest_bytes(tmp_path: Path) -> None:
    """The campaign decision must identify this exact figure manifest."""
    source = write_formal_source(tmp_path)
    summary_path = source.parents[1] / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["figure_bundle_manifest_sha256"] = "0" * 64
    summary_path.write_bytes(_canonical_json_bytes(summary))

    result = run_sync("--source", source, "--check", cwd=tmp_path)

    assert result.returncode != 0
    assert "figure_bundle_manifest_sha256" in result.stderr


def test_check_is_read_only_for_an_existing_destination(tmp_path: Path) -> None:
    """--check may inspect artifacts but must not create or alter any output."""
    source = write_formal_source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "existing.txt").write_bytes(b"unchanged")
    before_names = {path.name for path in tmp_path.iterdir()}

    result = run_sync(
        "--source", source, "--dest", destination, "--check", cwd=tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "existing.txt").read_bytes() == b"unchanged"
    assert {path.name for path in tmp_path.iterdir()} == before_names


def test_sync_replaces_destination_with_only_canonical_pdfs(tmp_path: Path) -> None:
    """Successful publication removes stale targets and excludes the manifest."""
    source = write_formal_source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "stale.pdf").write_bytes(b"old")
    expected = [(source / name).read_bytes() for name in CANONICAL_FIGURES]

    result = run_sync("--source", source, "--dest", destination, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert tuple(sorted(path.name for path in destination.iterdir())) == CANONICAL_FIGURES
    assert [(destination / name).read_bytes() for name in CANONICAL_FIGURES] == expected


@pytest.mark.parametrize("interrupt_after_promotion", (False, True))
def test_interrupted_first_install_restores_absent_destination(
    monkeypatch, tmp_path: Path, interrupt_after_promotion: bool
) -> None:
    """A failed first promotion must leave no partially installed directory."""
    module = load_script_module()
    source = write_formal_source(tmp_path)
    destination = tmp_path / "destination"
    real_replace = os.replace
    interrupted = False

    def interrupt_promotion(source_path: Path, target_path: Path) -> None:
        nonlocal interrupted
        if interrupted:
            real_replace(source_path, target_path)
            return
        interrupted = True
        if interrupt_after_promotion:
            real_replace(source_path, target_path)
        raise KeyboardInterrupt("simulated interrupted first install")

    monkeypatch.setattr(module.os, "replace", interrupt_promotion)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--source", str(source), "--dest", str(destination)],
    )

    with pytest.raises(KeyboardInterrupt, match="interrupted first install"):
        module.main()

    assert not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == {"campaign"}


def _write_existing_destination(destination: Path) -> dict[str, bytes]:
    destination.mkdir()
    old_contents: dict[str, bytes] = {}
    for index, name in enumerate(CANONICAL_FIGURES, start=1):
        old_contents[name] = f"old-pdf-{index}".encode()
        (destination / name).write_bytes(old_contents[name])
    (destination / "keep.txt").write_bytes(b"keep-me")
    return old_contents


def _assert_existing_destination_restored(
    tmp_path: Path, destination: Path, old_contents: dict[str, bytes]
) -> None:
    assert {
        name: (destination / name).read_bytes() for name in CANONICAL_FIGURES
    } == old_contents
    assert (destination / "keep.txt").read_bytes() == b"keep-me"
    assert {path.name for path in tmp_path.iterdir()} == {"campaign", "destination"}


def test_commit_failure_rolls_back_the_complete_existing_destination(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed staging promotion must restore every previous target."""
    module = load_script_module()
    source = write_formal_source(tmp_path)
    destination = tmp_path / "destination"
    old_contents = _write_existing_destination(destination)
    real_replace = os.replace
    replace_count = 0

    def fail_second_replace(source_path: Path, target_path: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated commit failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(module.os, "replace", fail_second_replace)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--source", str(source), "--dest", str(destination)],
    )

    with pytest.raises(OSError, match="simulated commit failure"):
        module.main()

    _assert_existing_destination_restored(tmp_path, destination, old_contents)


def test_interrupt_after_backup_rename_restores_existing_destination(
    monkeypatch, tmp_path: Path
) -> None:
    """An interruption after the first rename must not strand only a backup."""
    module = load_script_module()
    source = write_formal_source(tmp_path)
    destination = tmp_path / "destination"
    old_contents = _write_existing_destination(destination)
    real_replace = os.replace
    replace_count = 0

    def interrupt_after_first_replace(source_path: Path, target_path: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        real_replace(source_path, target_path)
        if replace_count == 1:
            raise KeyboardInterrupt("simulated interruption after backup rename")

    monkeypatch.setattr(module.os, "replace", interrupt_after_first_replace)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--source", str(source), "--dest", str(destination)],
    )

    with pytest.raises(KeyboardInterrupt, match="after backup rename"):
        module.main()

    _assert_existing_destination_restored(tmp_path, destination, old_contents)
