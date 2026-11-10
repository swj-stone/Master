"""Confirmation operates only on frozen, validated candidate masks."""

import pytest

from minppi_mofs.removal_confirmation import select_confirmed_removal


FEATURES = list("abcd")
FULL = 15
OFFICIAL = {FULL: (1., .99, .98), 14: (.999, .988, .981), 12: (1., .99, .98)}


def test_confirmation_veto_selects_surviving_incumbent_with_frozen_scores():
    confirmation = {FULL: .89, 14: .888, 12: .87}
    result = select_confirmed_removal(FEATURES, {"structured": 14, "sign_worst": 12},
                                      OFFICIAL, .005, confirmation.__getitem__)
    assert result["selected_mask"] == 14 and result["selected_source"] == "structured"
    assert result["removed_features"] == ["a"] and result["removed_count"] == 1
    assert result["selected_scores"] == list(OFFICIAL[14])
    assert result["selected_confirmation_score"] == .888
    assert result["selected_confirmation_delta"] == pytest.approx(-.002)
    by_mask = {row["mask"]: row for row in result["trials"]}
    assert by_mask[12]["official_gate_passed"] and not by_mask[12]["confirmation_gate_passed"]
    assert not by_mask[12]["accepted"]


def test_both_gates_are_required_and_full_remains_available():
    scores = {**OFFICIAL, 14: (1., .90, 1.)}
    confirmation = {FULL: .89, 14: .90, 12: .87}
    result = select_confirmed_removal(FEATURES, {"structured": 14, "sign_worst": 12},
                                      scores, .005, confirmation.__getitem__)
    assert result["selected_source"] == "full_set" and result["selected_mask"] == FULL
    assert result["removed_features"] == [] and result["selected_confirmation_delta"] == 0
    assert [row["mask"] for row in result["trials"] if row["accepted"]] == [FULL]


@pytest.mark.parametrize("same_mask,expected_source", [(15, "full_set"), (14, "structured")])
def test_identical_masks_share_fit_and_have_fixed_names(same_mask, expected_source):
    seen = []
    def confirm(mask):
        assert mask not in seen
        seen.append(mask)
        return .9
    result = select_confirmed_removal(FEATURES, {"sign_worst": same_mask, "structured": same_mask},
                                      OFFICIAL, .005, confirm)
    assert seen == list(dict.fromkeys([FULL, same_mask]))
    assert result["selected_source"] == expected_source
    assert result["trials"][-1]["sources"][-2:] == ["structured", "sign_worst"]


def test_rank_by_sparsity_then_confirmation_delta_then_official_worst_half():
    scores = {**OFFICIAL, 13: (1., .991, .981)}
    def chosen(sign_mask, confirm_delta):
        values = {FULL: 1., 14: 1., sign_mask: 1. + confirm_delta}
        return select_confirmed_removal(FEATURES, {"structured": 14, "sign_worst": sign_mask},
                                        scores, .005, values.__getitem__)["selected_source"]
    assert chosen(12, -.004) == "sign_worst"  # larger accepted removal wins
    assert chosen(13, -.003) == "structured"  # equal size; compare new holdout
    assert chosen(13, 0.) == "sign_worst"  # equal new score; compare old halves


@pytest.mark.parametrize("masks", [
    {"structured": 0, "sign_worst": 1}, {"structured": 1, "sign_worst": 16},
    {"structured": True, "sign_worst": 12}, {"sign_worst": 12},
])
def test_invalid_frozen_candidates_rejected_before_refits(masks):
    with pytest.raises(ValueError):
        select_confirmed_removal(FEATURES, masks, OFFICIAL, .005, lambda mask: pytest.fail("unexpected fit"))


def test_bad_official_scores_or_holdout_scores_are_rejected():
    candidates = {"structured": 14, "sign_worst": 12}
    for scores in ({}, {**OFFICIAL, 12: (.8, .8)}, {**OFFICIAL, 14: (float("nan"), .9, .9)}):
        with pytest.raises(ValueError):
            select_confirmed_removal(FEATURES, candidates, scores, .005, lambda mask: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError, match="confirmation"):
        select_confirmed_removal(FEATURES, candidates, OFFICIAL, .005, lambda mask: float("inf"))
