from __future__ import annotations

from experiments.benchmark_icomp2026_backends import (
    BenchmarkCandidate,
    choose_fastest_equivalent,
    make_benchmark_workload,
    make_structural_panel,
)


def test_benchmark_workload_is_deterministic_nested_and_nonempty() -> None:
    features = tuple(f"f{index}" for index in range(1, 101))

    workload = make_benchmark_workload(
        features,
        seed=42,
        fit_count=8,
    )

    assert workload == make_benchmark_workload(
        features,
        seed=42,
        fit_count=8,
    )
    assert len(workload) == 8
    assert tuple(len(coalition) for coalition in workload[:4]) == (
        25,
        50,
        75,
        100,
    )
    assert set(workload[0]) < set(workload[1])
    assert set(workload[1]) < set(workload[2])
    assert set(workload[2]) < set(workload[3])
    assert workload[:4] == workload[4:]


def test_fastest_candidate_must_preserve_reference_utilities() -> None:
    reference = BenchmarkCandidate("cuda", n_jobs=1, workers=1)
    reports = (
        {
            "candidate": reference.label,
            "fits_per_second": 1.0,
            "max_abs_utility_delta_to_reference": 0.0,
        },
        {
            "candidate": "cuda-n1-w2",
            "fits_per_second": 1.8,
            "max_abs_utility_delta_to_reference": 0.0,
        },
        {
            "candidate": "cpu-n8-w1",
            "fits_per_second": 3.0,
            "max_abs_utility_delta_to_reference": 0.002,
        },
    )

    selected = choose_fastest_equivalent(
        reports,
        reference=reference,
        utility_tolerance=1e-6,
    )

    assert selected == "cuda-n1-w2"


def test_candidate_label_is_stable_and_validated() -> None:
    assert BenchmarkCandidate("cpu", n_jobs=8, workers=2).label == "cpu-n8-w2"


def test_structural_panel_stratifies_targets_over_full_feature_order() -> None:
    features = tuple(f"f{index}" for index in range(1, 21))

    panel = make_structural_panel(features, target_feature_count=4, seed=42)

    assert tuple(row.feature for row in panel[::2]) == (
        "f1",
        "f7",
        "f13",
        "f20",
    )
    assert len(panel) == 8
    assert {row.fraction for row in panel} == {0.25, 0.75}
