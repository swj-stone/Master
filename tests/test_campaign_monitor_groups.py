from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import sys

import pytest

import experiments.monitor_icomp2026_campaign as campaign_monitor


_FIGURES = (
    "fig1_overview.pdf",
    "fig2_cd_heatmap.pdf",
    "fig3_prefilter.pdf",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_formal_figure_bundle(campaign_root: Path) -> None:
    bundle = campaign_root / "figures" / "paper"
    bundle.mkdir(parents=True)
    for index, name in enumerate(_FIGURES, start=1):
        (bundle / name).write_bytes(
            f"%PDF-1.4\nmonitor-{index}\n%%EOF\n".encode("ascii")
        )
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "filename": name,
                "sha256": hashlib.sha256((bundle / name).read_bytes()).hexdigest(),
                "byte_length": (bundle / name).stat().st_size,
            }
            for name in _FIGURES
        ],
    }
    manifest_payload = _canonical_json_bytes(manifest)
    (bundle / "figure_bundle_manifest.json").write_bytes(manifest_payload)
    (campaign_root / "campaign_summary.json").write_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "publication_allowed": True,
                "figure_bundle_manifest_sha256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
            }
        )
    )


def _write_campaign(tmp_path: Path) -> tuple[Path, str]:
    output_root = tmp_path / "results"
    campaign_id = "campaign"
    campaign_root = output_root / campaign_id
    campaign_root.mkdir(parents=True)
    (campaign_root / "campaign_plan.json").write_text(
        json.dumps({"run_ids": ["run-one"]}), encoding="utf-8"
    )
    run_root = output_root / "run-one"
    run_root.mkdir()
    stages = {}
    for stage in (
        "validate",
        "preprocess",
        "estimator",
        "removal",
        "alpha",
        "nsga2",
        "ablation",
    ):
        stages[stage] = {
            "pending": 0,
            "running": 0,
            "success": 1,
            "failed": 0,
            "skipped": 0,
        }
    (run_root / "status.json").write_text(
        json.dumps({"stages": stages}), encoding="utf-8"
    )
    return output_root, campaign_id


def test_initial_monitor_view_only_reports_sealed_initial_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root, campaign_id = _write_campaign(tmp_path)
    (output_root / campaign_id / "campaign_index.json").write_text(
        "{}", encoding="utf-8"
    )
    screening = output_root / campaign_id / "screening"
    screening.mkdir()
    (screening / "screen_report.json").write_text("{}", encoding="utf-8")

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            "--group",
            "initial",
        )
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "  validate: 1/1 success" in output
    assert "  preprocess: 1/1 success" in output
    assert "  estimator: 1/1 success" in output
    assert "  removal:" not in output
    assert "  nsga2:" not in output
    assert "campaign-index: available" in output
    assert "screening-report: available" in output


def test_downstream_monitor_view_reports_test_stages_and_publication_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root, campaign_id = _write_campaign(tmp_path)
    _write_formal_figure_bundle(output_root / campaign_id)

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            "--group",
            "downstream",
        )
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "  removal: 1/1 success" in output
    assert "  alpha: 1/1 success" in output
    assert "  nsga2: 1/1 success" in output
    assert "  ablation: 1/1 success" in output
    assert "  estimator:" not in output
    assert "campaign-summary: publication_allowed=true, figures=ready" in output


def test_downstream_monitor_rejects_digest_without_actual_figure_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root, campaign_id = _write_campaign(tmp_path)
    (output_root / campaign_id / "campaign_summary.json").write_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "publication_allowed": True,
                "figure_bundle_manifest_sha256": "a" * 64,
            }
        )
    )

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            "--group",
            "downstream",
        )
    )

    assert exit_code == 1
    assert "figures=not ready" in capsys.readouterr().out


class _FlushVisibleBuffer(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.visible = ""

    def flush(self) -> None:
        self.visible = self.getvalue()
        super().flush()


def test_watch_flushes_each_snapshot_before_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, campaign_id = _write_campaign(tmp_path)
    stream = _FlushVisibleBuffer()
    monkeypatch.setattr(sys, "stdout", stream)

    def inspect_then_interrupt(_seconds: int) -> None:
        assert "run-one:" in stream.visible
        raise KeyboardInterrupt

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            "--watch",
        ),
        sleep=inspect_then_interrupt,
    )

    assert exit_code == 130
