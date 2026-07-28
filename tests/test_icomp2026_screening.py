from __future__ import annotations

from pathlib import Path

import pytest

from experiments.screen_icomp2026_results import (
    ClaimEvidence,
    CoalitionValue,
    classify_conclusion,
    find_harmful_witness,
    read_frozen_run_seeds,
    select_screen_budget_pair,
)


def test_conclusion_grades_cover_directional_outcomes() -> None:
    assert (
        classify_conclusion(
            ClaimEvidence(True, 3, 0.95, 3, 0.01, 0.005)
        )
        == "strong_support"
    )
    assert (
        classify_conclusion(
            ClaimEvidence(True, 3, 0.60, 1, 0.0, 0.0)
        )
        == "partial_support"
    )
    assert (
        classify_conclusion(
            ClaimEvidence(True, 0, 1.0, 0, 0.0, 0.0)
        )
        == "neutral"
    )
    assert (
        classify_conclusion(
            ClaimEvidence(True, 3, 0.80, 2, -0.03, 0.01)
        )
        == "counterexample"
    )
    assert (
        classify_conclusion(
            ClaimEvidence(False, 0, 0.0, 0, 0.0, 0.0)
        )
        == "insufficient_evidence"
    )


def test_harmful_witness_requires_threshold_crossing_and_maximizes_drop() -> None:
    records = (
        CoalitionValue(("a", "b", "c"), 1.0),
        CoalitionValue(("b",), 0.97),
        CoalitionValue(("a", "b"), 0.90),
        CoalitionValue(("c",), 0.99),
        CoalitionValue(("a", "c"), 0.80),
        CoalitionValue((), 0.50),
        CoalitionValue(("a",), 0.40),
    )

    witness = find_harmful_witness(
        "a",
        records,
        full=("a", "b", "c"),
        alpha=0.95,
    )

    assert witness is not None
    assert witness.without == ("c",)
    assert witness.with_feature == ("a", "c")
    assert witness.utility_drop == 0.19
    assert witness.threshold == 0.95


def test_harmful_witness_rejects_nonwinning_base_or_nonlosing_addition() -> None:
    records = (
        CoalitionValue(("a", "b"), 1.0),
        CoalitionValue(("b",), 0.94),
        CoalitionValue(("a",), 0.96),
        CoalitionValue((), 0.50),
    )

    assert (
        find_harmful_witness(
            "a", records, full=("a", "b"), alpha=0.95
        )
        is None
    )


def test_screen_uses_the_two_highest_available_budget_checkpoints() -> None:
    assert select_screen_budget_pair((10, 25)) == (10, 25)
    assert select_screen_budget_pair((10, 25, 50)) == (25, 50)


def test_screen_reads_complete_seed_shard_authority(
    tmp_path: Path,
) -> None:
    (tmp_path / "resolved_protocol.yaml").write_text(
        "seeds:\n- 42\n- 123\n",
        encoding="utf-8",
    )

    assert read_frozen_run_seeds(tmp_path) == (42, 123)

    (tmp_path / "resolved_protocol.yaml").write_text(
        "seeds:\n- 42\n- 42\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed"):
        read_frozen_run_seeds(tmp_path)
