import pytest

from experiments.compare_fixed_predictor import recipe_parameters, validate_decision, summarize


def test_later_rounds_bind_the_actual_checkpoints_and_count_one_shared_fit():
    policy = {"model_params": {"n_estimators": 5}}
    params, fits = recipe_parameters("rounds", policy)
    assert fits == 1 and params == dict(predictor_recipe="rounds", original_rounds=5, blend_checkpoints=[8,10])
    assert recipe_parameters("column_blend", policy) == ({"colsample_bynode": .5}, 2)


def test_predictor_decision_requires_each_score_and_matches_the_original_validation():
    old = [.8, .79, .81]
    improved = [.805, .797, .814]
    d = dict(mask="0x7", original_scores=old, blend_scores=improved,
             validation_deltas=[a-b for a,b in zip(improved, old)], adopt_blend=True)
    assert validate_decision(d, old, "0x7") == "blend"
    with pytest.raises(ValueError, match="original validation"):
        validate_decision(d, [.76,.77,.75], "0x7")
    d["blend_scores"] = [.805,.789,.824]
    d["validation_deltas"] = [a-b for a,b in zip(d["blend_scores"],old)]
    with pytest.raises(ValueError, match="adoption"):
        validate_decision(d, old, "0x7")


def test_success_requires_two_gains_and_no_new_breach_or_material_decline():
    rows = [dict(dataset=str(i), status="completed", margin=.005, delta=d, new_breach=False,
                 test_delta_vs_original_full=0., within_original_full_margin=True, ci_low=0., ci_high=0.)
            for i,d in enumerate([.0006,.0008,0.,0.,0.,0.])]
    assert summarize(rows, 6)["success"]
    assert not summarize(rows[:-1], 6)["success"]
    rows[-1]["delta"] = -.0006
    assert not summarize(rows, 6)["success"]
    rows[-1].update(delta=0., new_breach=True)
    assert not summarize(rows, 6)["success"]
