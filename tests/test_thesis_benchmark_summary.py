import csv
import math

from experiments.thesis_benchmark_summary import write_dataset_summary


def test_dataset_summary_uses_paired_seed_deltas_and_records_incomplete_jobs(tmp_path):
    jobs = [("a", 42), ("a", 123), ("a", 456), ("b", 42), ("b", 123)]
    rows = [
        dict(dataset="a", seed=42, status="completed", metric="roc_auc", n_features=4, removed_count=2,
             compression_fraction=.5, full_original_test=.79, guarded_original_test=.8,
             full_chosen_test=.80, guarded_chosen_test=.81, selection_delta=.01, predictor_delta=.01,
             net_delta=.02, matched_full_delta=.01, guarded_recipe="extended_average", wall_s=6.),
        dict(dataset="a", seed=123, status="completed", metric="roc_auc", n_features=4, removed_count=1,
             compression_fraction=.25, full_original_test=.79, guarded_original_test=.82,
             full_chosen_test=.79, guarded_chosen_test=.82, selection_delta=.03, predictor_delta=0.,
             net_delta=.03, matched_full_delta=.03, guarded_recipe="original", wall_s=10.),
        dict(dataset="a", seed=456, status="timed_out", wall_s=900., predictor_delta=99.),
        dict(dataset="b", seed=42, status="interrupted", wall_s=2.),
    ]
    write_dataset_summary(tmp_path, rows, jobs)
    with (tmp_path / "dataset_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        a, b = csv.DictReader(f)
    assert (a["dataset"], a["completed"], a["timed_out"], a["pending"], a["all_completed"]) == ("a", "2", "1", "0", "False")
    assert a["guarded_extended_count"] == "1" and float(a["selection_delta_mean"]) == .02
    assert math.isclose(float(a["selection_delta_sd"]), .02 / math.sqrt(2))
    assert math.isclose(float(a["predictor_delta_mean"]), .005)
    assert float(a["wall_s_mean"]) == 8. and float(a["attempted_wall_s"]) == 916.
    assert (b["dataset"], b["interrupted"], b["pending"], b["guarded_chosen_test_mean"]) == ("b", "1", "1", "")


def test_one_completed_seed_has_no_sample_standard_deviation(tmp_path):
    row = dict(dataset="a", seed=123, status="completed", wall_s=3., guarded_chosen_test=.85)
    write_dataset_summary(tmp_path, [row], [("a", 123)])
    with (tmp_path / "dataset_summary.csv").open(encoding="utf-8-sig") as f:
        result = next(csv.DictReader(f))
    assert result["all_completed"] == "True"
    assert result["guarded_chosen_test_mean"] == "0.85" and result["guarded_chosen_test_sd"] == ""
