"""Guarded updates may only improve the locked validation comparison."""

import pytest

from minppi_mofs.guarded_update import guarded_update


BASE = (1., 1., 1.)


@pytest.mark.parametrize("signed,scores,adopt", [
    (12, (.998, .998, .998), True),  # strictly greater compression at equal scores
    (13, (.998, .999, .998), True),  # equal compression and one strict score improvement
    (13, (.998, .998, .998), False),  # different masks with equivalent objectives
    (13, (.998, .998 + 5e-13, .998), False),
    (12, (1., 1., .997), False),  # a single worse half blocks the larger deletion
    (12, (1., 1., .9), False),  # fails the original full-feature gate
    (15, BASE, False),  # improved utility but less compression
    (14, (.998, .998, .998), False),  # same mask
])
def test_only_strict_validation_dominance_updates_the_incumbent(signed, scores, adopt):
    result = guarded_update(list("abcd"), 14, signed, {14: (.998, .998, .998), signed: scores}, BASE, .005)
    assert result["selected_mask"] == (signed if adopt else 14)
    assert result["selected_source"] == ("sign_worst" if adopt else "structured")
    assert result["predicates"]["adopt_challenger"] is adopt
    assert result["selected_scores"] == list(scores if adopt else (.998, .998, .998))
    assert result["candidate_masks"] == {"structured": 14, "sign_worst": signed}
    assert result["challenger_minus_incumbent_removed_count"] == 3 - signed.bit_count()


def test_invalid_or_unverified_incumbents_cannot_anchor_comparison():
    for mask in (0, 16, 1.5, True):
        with pytest.raises(ValueError, match="mask"):
            guarded_update(list("abcd"), mask, 12, {12: BASE}, BASE, .005)
    with pytest.raises(ValueError, match="incumbent"):
        guarded_update(list("abcd"), 14, 12, {14: (1., .90, 1.), 12: BASE}, BASE, .005)
    with pytest.raises(ValueError, match="full"):
        guarded_update(list("abcd"), 15, 12, {15: (.999, 1., 1.), 12: BASE}, BASE, .005)
    with pytest.raises(ValueError, match="scores"):
        guarded_update(list("abcd"), 14, 12, {14: BASE}, BASE, .005)
