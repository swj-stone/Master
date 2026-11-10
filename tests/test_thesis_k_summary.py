import csv
import math

import pytest

from experiments.thesis_k_summary import update_k_reports


def read(run, filename):
    with (run / filename).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def job(seed, gains, *, dataset="toy", width=8):
    checks = {}
    for k, test in enumerate(gains, 1):
        removed = (["a", "b"], ["a", "c"], ["a", "c", "d"])[k - 1]
        witnesses = {} if k == 1 else {"a": {"round": 2}}
        time_s = [10., 24., 40.][k - 1]
        decision = {"original_scores": [.78, .77, .79], "blend_scores": [.8, .81, .79], "adopt_blend": True}
        checks[str(k)] = dict(
            phase="complete", dataset=dataset, seed=seed, k=k, n_features=width, policy={"utility": "roc_auc"},
            margin=.005, removed_count=len(removed), retained_count=width-len(removed), removed_features=removed,
            proposed_count=width-1, witness_count=len(witnesses), witnesses=witnesses, sampling_stop="round_limit",
            planned_trials=k*width, resolved_trials=k*width, sampled_draws=k*width-len(witnesses),
            sample_fits=k*width, sample_elapsed_s=time_s, sampling_cache_hits_from_selection=k-1,
            checkpoint_selection_s=3.*k, checkpoint_selection_fits=2, checkpoint_test_s=1., checkpoint_test_fits=1,
            predictor_decisions=dict(full=decision, guarded=decision),
            method_test_utilities=dict(full_original=.8, full_blend=.805, full_chosen=.805,
                                       guarded_original=test-.001, guarded_blend=test, guarded_chosen=test))
    return {"status": "completed", "elapsed_s": 75., "result": {"checkpoints": checks}}


def test_reports_pair_seeds_compare_masks_and_never_count_unfinished_jobs(tmp_path):
    a = job(42, [.8, .806, .807])
    b = job(123, [.8, .802, .800], width=10)
    repair = {("toy",42):a, ("toy",123):b, ("toy",456):{"status":"interrupted","elapsed_s":4., "result":a["result"]}}
    update_k_reports(tmp_path, [("toy",42), ("toy",123), ("toy",456), ("toy",789)], repair)
    detail = read(tmp_path, "summary.csv")
    assert len(detail) == 12
    assert [r["status"] for r in detail[-6:]] == ["interrupted"] * 3 + ["pending"] * 3
    assert all(r["guarded_chosen_test"] == "" for r in detail[-6:])
    assert len(read(tmp_path, "marginal.csv")) == 8
    first, second = read(tmp_path, "marginal.csv")[:2]
    assert first["removed_delta"] == "0" and first["removal_set_symmetric_difference"] == "2"
    assert float(first["removal_set_jaccard"]) == pytest.approx(1/3)
    assert first["new_negative_witnesses"] == "1" and first["additional_sample_fits"] == "8"
    assert float(first["additional_sampling_s"]) == 14.
    assert float(second["guarded_chosen_test_delta"]) == pytest.approx(.001)
    ds = read(tmp_path, "dataset_summary.csv")[2]
    assert ds["completed"] == "2" and ds["expected"] == "4"
    assert float(ds["compression_fraction_mean"]) == pytest.approx((3/8+3/10)/2)
    transition = read(tmp_path, "marginal_summary.csv")[1]
    assert transition["completed"] == "2" and transition["expected"] == "4"
    assert transition["pending"] == transition["interrupted"] == "1"
    assert transition["test_gains"] == transition["test_losses"] == "1"
    assert transition["test_practical_gains"] == transition["test_practical_losses"] == "1"
    assert float(transition["guarded_chosen_test_delta_mean"]) == pytest.approx(-.0005)
    assert float(transition["guarded_chosen_test_delta_sd"]) == pytest.approx(.003/math.sqrt(2))
    assert transition["second_difference_n"] == "2"
    assert float(transition["second_difference_guarded_chosen_test_mean"]) == pytest.approx((-.005-.004)/2)


def test_zero_cost_unchanged_transition_and_pending_tables(tmp_path):
    jobs = [("toy",42)]
    update_k_reports(tmp_path, jobs, {})
    assert len(read(tmp_path, "summary.csv")) == 3 and all(r["completed"] == "0" for r in read(tmp_path, "dataset_summary.csv"))
    out = job(42, [.8,.8,.8])
    a, b = (out["result"]["checkpoints"][str(k)] for k in (2,3))
    for key in ("removed_features","removed_count","retained_count","sampled_draws","sample_fits","sample_elapsed_s"):
        b[key] = a[key]
    update_k_reports(tmp_path, jobs, {("toy",42):out})
    pair = read(tmp_path, "marginal.csv")[1]
    assert pair["removal_set_jaccard"] == "1.0" and pair["guarded_chosen_test_gain_per_sampling_minute"] == ""
    summary = read(tmp_path, "marginal_summary.csv")[1]
    assert summary["test_practical_ties"] == "1" and summary["guarded_chosen_test_delta_sd"] == ""
    assert summary["pooled_test_gain_per_sampling_minute"] == ""


def test_rejects_unpaired_checkpoint_identities(tmp_path):
    result = job(42, [.8,.81,.82])
    result["result"]["checkpoints"]["2"]["seed"] = 789
    with pytest.raises(ValueError, match="identity"):
        update_k_reports(tmp_path, [("toy",42)], {("toy",42):result})
