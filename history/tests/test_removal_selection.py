"""Joint deletion is accepted only for a measured, nonempty retained subset."""

import numpy as np
import pandas as pd
import pytest


def test_duplicate_is_nominated_with_its_retained_representative_without_a_witness():
    from minppi_mofs.removal_selection import nominate_removals

    x, y, z = np.array(list(np.ndindex(2, 2, 2))).T
    train = pd.DataFrame({"x": x, "copy": -x, "y": y, "z": z})
    nomination = nominate_removals(train, dict(x=2, copy=2, y=0.1, z=0.5), {})
    assert [e["feature"] for e in nomination] == ["copy", "y", "z"]
    proxy = nomination[0]
    assert (proxy["reason"], proxy["representative"]) == ("correlated_proxy", "x")
    assert proxy["abs_corr"] == pytest.approx(1)


def test_witnesses_are_first_and_do_not_become_proxy_representatives():
    from minppi_mofs.removal_selection import nominate_removals

    train = pd.DataFrame({f: [0, 1, 2, 3] for f in ["bad", "rep", "copy"]})
    records = {"bad": {"before_utility": 0.8, "after_utility": 0.7}}
    nominations = nominate_removals(train, {"bad": 4, "rep": 3, "copy": 0}, records)
    assert [e["feature"] for e in nominations] == ["bad", "copy", "rep"]
    assert nominations[0]["reason"] == "witness"
    assert nominations[0]["witness"] == records["bad"]
    assert nominations[1]["representative"] == "rep"


def test_many_proxies_respect_pool_cap_but_every_witness_survives():
    from minppi_mofs.removal_selection import nominate_removals

    features = [str(i) for i in range(10)]
    train = pd.DataFrame({f: [1, 2, 3] for f in features})
    assert len(nominate_removals(train, {}, set())) == 8
    witnessed = set(features[:9])
    result = nominate_removals(train, {}, witnessed)
    assert {r["feature"] for r in result} == witnessed
    assert all(r["reason"] == "witness" for r in result)


def test_reported_union_contains_gain_only_deletions_outside_structured_nomination():
    from minppi_mofs.removal_selection import select_robust_removals

    features, structured = list("abcdefgh"), list("bcdefg")
    nominations = [{"feature": f, "reason": "low_gain"} for f in structured]
    gains = dict(zip(features, range(8)))
    result = select_robust_removals(features, nominations, gains, lambda mask: (1., 1., 1.), (1., 1., 1.), .01)
    assert result["removed_features"] == structured
    assert result["candidate_pools"]["gain_only"] == list("abcdef")
    assert result["candidate_pools"]["witness_proxy_gain"] == structured
    assert result["structured_pool"]["removed_features"] == structured
    assert "witness_proxy_gain:pool" in result["structured_pool"]["sources"]
    assert result["candidate_features"] == structured
    assert result["external_candidate_features"] == list("abcdef")
    assert result["all_candidate_features"] == list("abcdefg")
    assert all(set(t["removed_features"]) <= set(result["all_candidate_features"]) for t in result["trials"])


def test_scalar_joint_gate_finds_intermediate_prefix_without_monotonicity():
    from minppi_mofs.removal_selection import select_removals

    features = list("abcdefgh")
    nomination = [{"feature": f, "reason": "low_gain"} for f in features[:6]]
    scores = {6: 0.2, 4: 0.98, 2: 0.8, 1: 1.0}
    seen = []
    def evaluate(mask):
        assert mask not in seen
        seen.append(mask)
        return scores[8 - mask.bit_count()]

    result = select_removals(features, nomination, evaluate, 1.0, .03)
    assert result["removed_features"] == list("abcd")
    assert result["retained_features"] == list("efgh")
    assert [t["removed_count"] for t in result["trials"]] == [6, 4, 2, 1]
    assert len(seen) == 4


def test_robust_gate_reserves_a_witness_and_rejects_pooled_success_with_a_bad_fold():
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcdefgh")
    noms = [{"feature": f, "reason": "witness" if f == "a" else "low_gain"} for f in features[:6]]
    gains = {f: i for i, f in enumerate(features)}
    seen, published = [], []
    def evaluate(mask):
        assert mask and mask not in seen
        seen.append(mask)
        if not (mask & 1):  # deleting a loses despite its harmful witness elsewhere
            return (.5, .5, .5)
        if 8 - mask.bit_count() == 5:
            return (.99, .95, 1.)
        return (1., 1., 1.)

    result = select_robust_removals(features, noms, gains, evaluate, (1., 1., 1.), .02, publish=published.append)
    assert result["methods"]["witness_proxy_gain"]["removed_count"] == 0
    assert result["methods"]["reserve_witness"]["removed_count"] == 4
    assert result["methods"]["gain_only"]["removed_count"] == 0
    assert result["removed_features"] == list("bcde")
    assert (result["pooled_only_removed_count"], result["robust_removed_count"]) == (5, 4)
    assert not result["witness_only"]["accepted"]
    assert published == result["trials"]
    assert len(seen) == len(result["trials"]) < 15
    assert any(t["pooled_accepted"] and not t["accepted"] for t in published)


def test_worst_fold_slack_then_fixed_method_order_break_equal_size_ties():
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcd")
    noms = [{"feature": f, "reason": "low_gain"} for f in "bcd"]
    gains = {f: i for i, f in enumerate(features)}
    def run(prefer_d):
        def evaluate(mask):
            score = .99 if prefer_d and mask == 8 else .98
            return (1., score, score)
        return select_robust_removals(features, noms, gains, evaluate, (1., 1., 1.), .03,
                                      sample_minima=dict.fromkeys(features, 0), shap_importance=gains)

    assert run(True)["selected_method"] == "witness_zero_shap"  # retain d
    assert run(False)["selected_method"] == "witness_proxy_gain"


def test_rescue_bisects_failed_blocks_with_no_witness_and_checks_the_whole_cumulative_mask():
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcdefgh")
    noms = [{"feature": f, "reason": "low_gain"} for f in "abcdef"]
    gains = {f: i for i, f in enumerate(features)}
    def evaluate(mask):
        # All three strict prefix lists would delete a first, yet b-f are jointly dispensable.
        score = .99 if mask & 1 else .5
        return (score, score, score)

    result = select_robust_removals(features, noms, gains, evaluate, (1., 1., 1.), .02)
    assert result["methods"]["gain_only"]["removed_count"] == 0
    assert result["methods"]["witness_proxy_gain"]["removed_count"] == 0
    rescue = result["methods"]["reserve_witness"]
    assert rescue["removed_count"] == 5 and rescue["retained_features"] == list("agh")
    assert len([t for t in result["trials"] if any(s.startswith("reserve_witness:") for s in t["sources"])]) <= 8


def test_single_feature_and_all_witnesses_keep_at_least_one_feature():
    from minppi_mofs.removal_selection import select_removals

    def unexpected(mask):
        pytest.fail(f"unexpected evaluation of {mask}")

    single = select_removals(["x"], [{"feature": "x", "reason": "witness"}], unexpected, .7, .01)
    assert single["selected_mask"] == 1 and single["removed_features"] == []
    assert not single["witness_only"]["feasible"]
    queried = []
    def veto(mask):
        queried.append(mask)
        return .1

    all_witness = [{"feature": f, "reason": "witness"} for f in "abc"]
    result = select_removals(list("abc"), all_witness, veto, .7, .01)
    assert 0 not in queried and result["removed_features"] == []
    assert result["selected_source"] == "full_set"


def test_gate_boundary_uses_recorded_absolute_tolerance():
    from minppi_mofs.removal_selection import select_removals

    args = (["a", "b"], [{"feature": "a", "reason": "witness"}])
    assert select_removals(*args, lambda mask: .9 - 5e-13, 1., .1)["removed_features"] == ["a"]
    assert select_removals(*args, lambda mask: .9 - 2e-12, 1., .1)["removed_features"] == []


def test_only_sampled_zeros_supplement_witnesses_and_controls_cannot_choose_the_final_mask():
    from minppi_mofs.removal_selection import nominate_removals, select_robust_removals

    features = list("abcdef")
    x, y, z = np.array(list(np.ndindex(2, 2, 2))).T
    train = pd.DataFrame(dict(a=x, b=x, c=y, d=z, e=x*y, f=y*z))
    gains = dict(a=0, b=3, c=2, d=0, e=1, f=1)
    minima = dict(a=1, b=0, c=0, d=None)  # e and f also unseen
    nominations = nominate_removals(train, gains, {}, sample_minima=minima)
    assert {r["feature"] for r in nominations} == {"b","c"}
    assert next(r for r in nominations if r["feature"] == "b")["representative"] == "a"
    result = select_robust_removals(features, nominations, gains, lambda mask: (1.,1.,1.), (1.,1.,1.), .01,
                                    sample_minima=minima, shap_importance=dict(a=0,b=3,c=4,d=.1,e=.2,f=.3))
    assert result["candidate_features"] == ["b","c"]
    assert result["removed_features"] == ["b","c"]
    assert result["methods"]["treeshap_only"]["removed_count"] == 5
    assert "a" in result["methods"]["treeshap_only"]["removed_features"]
    assert "d" in result["methods"]["gain_only"]["removed_features"]
    assert result["candidate_pools"]["witness_zero_shap"] == ["b","c"]


def test_a_certified_removal_has_priority_over_more_zero_only_removals():
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcdefgh")
    noms = [{"feature": f, "reason": "witness" if f == "a" else "low_gain"} for f in "abcdef"]
    minima = {f: (-1 if f == "a" else 0) for f in features}
    def evaluate(mask):
        # H is safe alone. Combining it with any other removal is vetoed.
        safe = bool(mask & 1) or mask == 254
        return (float(safe),)*3
    result = select_robust_removals(features, noms, {}, evaluate, (1.,1.,1.), .01, sample_minima=minima,
                                    shap_importance={f: i for i, f in enumerate(features)})
    assert result["methods"]["reserve_witness"]["removed_count"] == 5
    assert result["removed_features"] == ["a"] and result["removed_witness_count"] == 1


@pytest.mark.parametrize("witness_passes, expected_removals", [(True, "abc"), (False, "ab")])
def test_structured_only_replays_the_portfolio_path_and_skips_unrelated_validation_fits(
    witness_passes, expected_removals,
):
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcdefgh")
    nominees = [{"feature": f, "reason": "witness" if f in "abc" else "low_gain"} for f in "abcdef"]
    minima = {f: (-1 if f in "abc" else 0) for f in features}
    gain = {f: 8 - i for i, f in enumerate(features)}
    shap = dict(zip("cfdeghab", range(8)))

    def run(standalone):
        queried = []
        def evaluate(mask):
            assert mask not in queried
            queried.append(mask)
            removed = "".join(f for i, f in enumerate(features) if not mask & (1 << i))
            fails_one_fold = removed in {"abcdef", "abcd"} or (removed == "abc" and not witness_passes)
            return (1.0, .95 if fails_one_fold else 1.0, 1.0)
        kwargs = {"structured_only": True} if standalone else {"shap_importance": shap}
        result = select_robust_removals(features, nominees, gain, evaluate, (1., 1., 1.), .01,
                                        sample_minima=minima, **kwargs)
        return result, set(queried)

    full, all_masks = run(False)
    only, queried = run(True)
    expected = {t["mask"]: t for t in full["trials"]
                if any(source.startswith("witness_proxy_gain:") for source in t["sources"])}
    assert queried == set(expected) < all_masks
    for t in only["trials"]:
        assert (t["scores"], t["accepted"]) == (expected[t["mask"]]["scores"], expected[t["mask"]]["accepted"])
    path = full["methods"]["witness_proxy_gain"]
    assert only["selected_mask"] == path["selected_mask"]
    assert only["removed_features"] == list(expected_removals)
    assert only["selected_source"] == path["selected_source"]
    assert only["pooled_only_removed_count"] == path["pooled_only_removed_count"]
    assert list(only["methods"]) == ["witness_proxy_gain"]
    assert only["external_candidate_features"] == []
    assert not only["conditional_rescue_used"]


def test_structured_mode_rescues_zeros_only_after_all_its_prefixes_and_witness_set_are_vetoed():
    from minppi_mofs.removal_selection import select_robust_removals

    features = list("abcdefgh")
    nominations = [{"feature": f, "reason": "witness" if f == "a" else "low_gain"} for f in "abcdef"]
    minima = {f: -1 if f == "a" else 0 for f in features}
    queried = []

    def score(mask):
        queried.append(mask)
        return (1., 1., 1.) if mask & 1 else (1., .5, 1.)

    out = select_robust_removals(features, nominations, {}, score, (1., 1., 1.), .01,
                                 sample_minima=minima, structured_only=True)
    assert out["conditional_rescue_used"]
    assert out["methods"]["witness_proxy_gain"]["removed_count"] == 0
    assert out["selected_method"] == "reserve_witness" and out["removed_features"] == list("bcdef")
    assert not out["witness_only"]["accepted"]
    assert all(not mask & 1 for mask in queried[:-1]) and queried[-1] == out["selected_mask"]
    assert len(queried) == len(set(queried))
    assert out["external_candidate_features"] == []
