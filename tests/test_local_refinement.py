"""The local search ranks training-only exchanges and evaluates each distinct subset once."""

import math

import pytest

from minppi_mofs.local_refinement import select_local_refinement


def run_search(features, signs, mask, evaluate, *, gains=None, shap=None, recovery=None,
               retained_gain=None, reference=(1., 1., 1.), full_scores=(1., 1., 1.), publish=None):
    return select_local_refinement(
        list(features), dict(zip(features, signs, strict=True)), mask,
        gains or {}, shap or dict.fromkeys(features, 0), recovery or {}, retained_gain or {},
        evaluate, reference, full_scores, .01, publish=publish)


def test_swap_candidates_preserve_cardinality_witness_and_protected_columns():
    # Retain abcd; d is positive, h is unobserved and retained, e is a removed witness.
    features, signs = "abcdefgh", [0, -1, 0, 1, -1, 0, 0, None]
    incumbent = 0b10001111
    seen, published = [], []
    def evaluate(mask):
        assert mask not in seen
        seen.append(mask)
        assert mask & 0b10001000 == 0b10001000 and not mask & 0b00010000
        return (1., 1., 1.)
    result = run_search(features, signs, incumbent, evaluate, gains={"a": 3,"b": 2},
                        shap={"a":3,"b":2,"c":1,"d":0,"e":0,"f":2,"g":1,"h":0},
                        recovery={"f":.7,"g":.6}, retained_gain={"a":1,"b":3,"c":2},
                        publish=published.append)
    assert len(seen) == len(set(seen)) == len(published) == 8
    assert result["trials"] == published
    assert result["restore_candidates"] == ["f", "g"]
    assert all(row["mask"].bit_count() == incumbent.bit_count() - (not row["restored_features"])
               for row in published)
    assert all(len(row["restored_features"]) in {0,1,2} and
               len(row["newly_removed_features"]) == (len(row["restored_features"]) or 1)
               for row in published)
    assert all(row["mask"] == incumbent for row in result["methods"].values())
    assert set(result["candidate_features"]) == set("abcefg")


def test_choose_robust_pooled_and_pure_drop_from_same_measurements():
    # Retain ab, restore from cd. Gain and SHAP tie for both paths.
    scores = {6:(1.02, 1.02, 1.02), 12:(1.06, 1.04, 1.001), 10:(1.2, 1.1, .999),
              2:(1.01, 1.01, 1.01)}
    observed = []
    def evaluate(mask):
        observed.append(mask)
        return scores[mask]
    result = run_search("abcd", [0]*4, 3, evaluate, recovery={"c":.9, "d":.8})
    assert observed == [6, 12, 10, 2]
    assert all(len(r["sources"]) == 2 for r in result["trials"])
    assert result["selected_method"] == "local_robust" and result["selected_mask"] == 6
    assert result["methods"]["local_pooled"]["mask"] == 12
    assert result["methods"]["local_drop"]["mask"] == 2
    refused = next(r for r in result["trials"] if r["mask"] == 10)
    assert refused["full_gate_passed"] and not refused["incumbent_gate_passed"] and not refused["accepted"]


def test_tie_breaks_and_too_few_columns_for_two_by_two():
    # Restore ties on recovery broken by SHAP, gain and finally original feature order.
    # Only the one retained column can be exchanged; removing it alone is impossible.
    shap = dict(a=1, b=2, c=3, d=3, e=3)
    result = run_search("abcde", [0]*5, 1, lambda mask:(1.01,)*3,
                        gains={"d":1,"e":1}, shap=shap, recovery=dict.fromkeys("bcde", .5))
    assert result["restore_candidates"] == list("decb")
    assert [r["mask"] for r in result["trials"]] == [8,16]
    assert result["selected_mask"] == 8 and result["methods"]["local_drop"]["mask"] == 1


def test_only_drops_when_no_restorable_zero_witness_first_only_on_equal_rank():
    # Only d has been removed, with a negative witness.
    ranking = run_search("abcd", [0,-1,0,-1], 7, lambda _: (1.1,)*3,
                         retained_gain={"a":0,"b":1,"c":1})
    assert not ranking["restore_candidates"]
    assert {r["mask"] for r in ranking["trials"]} == {5,6}
    assert all(row["op"] == "drop_one" for row in ranking["trials"])


def test_missing_or_protected_sign_cannot_be_deleted():
    result = run_search("abc", [1,None,0], 7, lambda _: (1.,1.,1.))
    assert len(result["trials"]) == 1 and result["trials"][0]["mask"] == 3
    assert not result["trials"][0]["half_mean_improved"]
    with pytest.raises(ValueError, match="protected"):
        run_search("abc", [1,None,0], 1, lambda _: ())
    with pytest.raises(ValueError, match="recovery"):
        run_search("ab", [0,0], 1, lambda _: (), recovery={"b": math.inf})
    with pytest.raises(ValueError, match="incumbent"):
        run_search("ab", [0,0], 3, lambda _: (), reference=(.9,1.,1.))


def test_deadline_stops_on_a_complete_trial_and_excludes_unexamined_candidates():
    allowance = iter([True, False])
    seen = []
    result = select_local_refinement(
        list("abcd"), dict.fromkeys("abcd", 0), 3, {}, dict.fromkeys("abcd", 0),
        {"c":1, "d":0}, {}, lambda mask: seen.append(mask) or (1.02,)*3, (1.,)*3, (1.,)*3, .01,
        allow_next=lambda: next(allowance))
    assert result["evaluated_count"] == len(seen) == 1 and result["planned_count"] == 4
    assert result["budget_stopped"] and result["selected_mask"] == 6
    assert result["candidate_features"] == list("acd")


def test_frontier_scores_only_eligible_prior_masks_without_more_fits():
    fs = list("abcdefg")
    signs = dict(zip(fs, [0,0,0,0,-1,0,1], strict=True))
    incumbent = 0b1000011  # a,b,g retained. c,d,f may be restored, e remains out.
    old = [
        {"mask": 0b1100001, "scores": (1.04,)*3, "test_utility": -999},  # f was ranked third
        {"mask": 0b0010011, "scores": (1.2,)*3},  # restores e and drops protected g
        {"mask": 0b1100011, "scores": (1.1,)*3},  # larger retained subset
    ]
    def never(_):
        pytest.fail("No local validation fit is allowed")
    result = select_local_refinement(
        fs, signs, incumbent, {}, dict.fromkeys(fs, 0), {"c":3,"d":2,"f":1}, {}, never,
        (1.,)*3, (1.,)*3, .01, previous_trials=old+[old[0]], allow_next=lambda: False)
    assert result["selected_mask"] == incumbent and result["budget_stopped"]
    assert result["methods"]["validation_frontier"]["mask"] == 0b1100001
    assert result["methods"]["validation_frontier"]["score_origin"] == "previous_validation"
    assert result["frontier_previous_count"] == 1 and result["evaluated_count"] == 0
    skipped = select_local_refinement(
        fs, signs, incumbent, {}, dict.fromkeys(fs, 0), None, None, never, (1.,)*3, (1.,)*3, .01,
        previous_trials=old, skip_ranking=True)
    assert skipped["methods"]["validation_frontier"] == result["methods"]["validation_frontier"]
    assert skipped["budget_stopped"] and skipped["planned_count"] is None and not skipped["planned_masks"]
    with pytest.raises(ValueError, match="inconsistent"):
        select_local_refinement(
            fs, signs, incumbent, {}, dict.fromkeys(fs, 0), {"c":3,"d":2,"f":1}, {}, never,
            (1.,)*3, (1.,)*3, .01, previous_trials=old+[dict(old[0], scores=(.98,)*3)],
            allow_next=lambda: False)
