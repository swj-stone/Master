"""The sign-aware experiment uses two predeclared utility penalties on identical masks."""

import pytest

from minppi_mofs.sign_aware_removal import select_sign_aware_removals


def test_witnesses_and_zeros_are_examined_once_and_other_signs_are_retained():
    features = list("abcdefgh")
    minima = dict(zip(features, [-1, 0, 0, 0, 0, 1, None, 0], strict=True))
    seen, published = [], []
    gains = dict(zip(features, [99, 1, 2, 3, 4, 0, 0, 5], strict=True))

    def evaluate(mask):
        assert mask not in seen
        seen.append(mask)
        if not mask & 1:  # The negative witness is essential in this joint subset.
            return (.8, .8, .8)
        return (1., 1., 1.)

    result = select_sign_aware_removals(features, {"a": {"before": 1}}, minima,
                                        gains, gains, evaluate, (1., 1., 1.), .01, publish=published.append)
    assert result["eligible_features"] == result["candidate_features"] == list("abcdeh")
    assert result["nomination"][0]["witness"] == {"before": 1}
    assert result["removed_features"] == list("bcde")
    assert result["selected_method"] == "sign_worst"
    assert result["witness_only"]["removed_features"] == ["a"]
    assert not result["witness_only"]["accepted"]
    assert set(result["methods"]) == {"sign_pooled", "sign_worst", "matched_gain", "matched_treeshap"}
    assert result["trials"] == published and len(seen) == len(published) <= 21
    assert all(row["mask"] & (1 << 5) and row["mask"] & (1 << 6)
               for row in result["trials"] if row["is_sign_candidate"])
    assert result["structured_pool"]["removed_features"] == list("abcdeh")


def test_variant_winners_use_their_predeclared_loss_with_no_extra_evaluations():
    features = list("abcdefgh")
    importance = dict.fromkeys(features, 0)

    def evaluate(mask):
        n_removed = 8 - mask.bit_count()
        if n_removed == 7:
            return (1., .991, 1.)  # pooled objective .875, worst-fold objective -.025
        if n_removed == 4:
            return (1., 1., 1.)
        return (.7, .7, .7)

    result = select_sign_aware_removals(features, {}, dict.fromkeys(features, 0), {},
                                        importance, evaluate, (1., 1., 1.), .01)
    assert result["methods"]["sign_pooled"]["removed_features"] == list("abcdefg")
    assert result["removed_features"] == result["methods"]["sign_worst"]["removed_features"] == list("abcd")
    assert result["methods"]["sign_pooled"]["objective"] == pytest.approx(.875)
    assert result["methods"]["sign_worst"]["objective"] == pytest.approx(.5)
    assert result["eligible_features"] == features and result["candidate_features"] == features[:-1]
    assert len(result["trials"]) == 5


def test_rank_fusion_uses_tie_midranks_and_input_order():
    features = list("abcd")
    # All gain values tie. SHAP is therefore the sole discriminator of the fused ranking.
    shap = dict(a=3, b=2, c=1, d=1)
    result = select_sign_aware_removals(features, {}, dict.fromkeys(features, 0), {}, shap,
                                        lambda mask: (1., 1., 1.), (1., 1., 1.), 0.)
    assert result["candidate_pools"]["witness_first_fused"] == ["c", "d", "b"]
    assert result["candidate_pools"]["eligible_gain"] == ["a", "b", "c"]
    assert result["removed_features"] == list("bcd")  # deterministic first-ranked path tie


@pytest.mark.parametrize("minima", [{"a": 1, "b": None}, {"a": 1}, {}])
def test_no_supported_deletions_keep_full_baseline_even_when_controls_succeed(minima):
    result = select_sign_aware_removals(["a", "b"], {}, minima, {}, dict(a=0, b=0),
                                        lambda mask: (1., 1., 1.), (1., 1., 1.), .01)
    assert result["selected_mask"] == 3 and all(not t["is_sign_candidate"] for t in result["trials"])
    assert result["methods"]["matched_gain"]["removed_count"] == 1
    assert result["structured_pool"]["mask"] == 3 and not result["candidate_features"]


def test_external_controls_evaluate_unrestricted_ranks_under_same_loss_rule():
    features = list("abcd")
    minima = dict(a=1, b=None, c=0, d=0)
    # The optimal two-column removal on each control can remove protected columns.
    gain, shap = dict(a=0, b=1, c=2, d=3), dict(a=1, b=3, c=0, d=2)
    evaluations = []
    def evaluate(mask):
        assert mask not in evaluations
        evaluations.append(mask)
        if mask in (12, 10):  # control optima remove a+b or a+c
            return (1., 1., 1.)
        if mask == 3:  # best sign-safe joint removal
            return (1., .998, 1.)
        return (.9, .9, .9)

    result = select_sign_aware_removals(features, {}, minima, gain, shap, evaluate, (1., 1., 1.), .01)
    assert result["selected_mask"] == 3
    assert result["methods"]["matched_gain"]["mask"] == 12
    assert result["methods"]["matched_treeshap"]["mask"] == 10
    assert result["candidate_features"] == ["c", "d"]
    assert 12 in evaluations and not next(t for t in result["trials"] if t["mask"] == 12)["is_sign_candidate"]
    assert set(result["external_candidate_features"]) == set(features)
    assert set(result["all_candidate_features"]) == set(features)


def test_all_negative_cannot_delete_every_feature_and_zero_margin_rejects_a_bad_fold():
    features = ["a", "b"]
    result = select_sign_aware_removals(features, set(features), dict.fromkeys(features, -1),
                                        {}, dict.fromkeys(features, 0), lambda mask: (1., 1.-1e-13, 1.),
                                        (1., 1., 1.), 0.)
    assert result["selected_mask"] == 3 and not result["witness_only"]["feasible"]
    assert len(result["trials"]) == 1 and not result["trials"][0]["accepted"]


def test_rejects_missing_or_inconsistent_sign_evidence_and_missing_shap():
    args = (["a", "b"], {"a"}, {"a": 0, "b": 0}, {}, dict(a=0, b=1))
    tail = (lambda mask: (1., 1., 1.), (1., 1., 1.), .01)
    with pytest.raises(ValueError, match="witness"):
        select_sign_aware_removals(*args, *tail)
    with pytest.raises(ValueError, match="sample_minima"):
        select_sign_aware_removals(["a", "b"], {}, None, {}, dict(a=0, b=1), *tail)
    with pytest.raises(ValueError, match="TreeSHAP"):
        select_sign_aware_removals(["a", "b"], {}, {"a": 0}, {}, {"a": 0}, *tail)
