"""Repository-level consistency checks for the canonical Task 4 API."""

from __future__ import annotations

from pathlib import Path

import minppi_mofs


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXPERIMENTS = (
    ROOT / "experiments" / "eval_harmful_utility.py",
    ROOT / "experiments" / "run_mofs_verify.py",
    ROOT / "experiments" / "run_pareto_eval.py",
)


def test_package_exports_canonical_guidance_api() -> None:
    canonical_names = (
        "estimate_minppi_mi_correlation_guided",
        "build_mi_correlation_guidance",
        "get_correlation_neighbors",
        "rank_by_mi",
        "filter_top_by_mi",
    )

    assert all(hasattr(minppi_mofs, name) for name in canonical_names)


def test_active_experiments_use_only_canonical_guidance_names() -> None:
    forbidden = (
        "build_guidance_from_data",
        'method="mi_guided"',
        "shapg_neighbors=",
        "shapg_scores=",
    )

    for path in ACTIVE_EXPERIMENTS:
        source = path.read_text(encoding="utf-8")
        assert "build_mi_correlation_guidance" in source
        assert 'method="mi_correlation_guided"' in source
        assert not any(token in source for token in forbidden)


def test_paper_separates_algorithm_b_from_shapg_baseline() -> None:
    paper = (
        ROOT / "conference-paper" / "paper" / "icomp2026_conference.tex"
    ).read_text(encoding="utf-8")

    assert "MI-Correlation-Guided" in paper
    assert "ShapG-guided" not in paper
    assert "ShapG is retained as a baseline" in paper
