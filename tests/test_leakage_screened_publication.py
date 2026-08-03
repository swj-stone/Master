from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import yaml

import experiments.make_paper_figures as figure_module
import experiments.run_icomp2026_group1 as group1_module
from experiments.build_leakage_screened_publication import (
    EXCLUDED_DATASET_ID,
    RETAINED_DATASETS,
    audit_crimedata_target_construction,
    filter_registered_exclusion,
)


def _composite_frame(*, leaking: bool) -> pd.DataFrame:
    outcome_fields = (
        "murders",
        "murdPerPop",
        "rapes",
        "rapesPerPop",
        "robberies",
        "robbbPerPop",
        "assaults",
        "assaultPerPop",
        "ViolentCrimesPerPop",
        "burglaries",
        "burglPerPop",
        "larcenies",
        "larcPerPop",
        "autoTheft",
        "autoTheftPerPop",
        "arsons",
        "arsonsPerPop",
    )
    frame = pd.DataFrame({field: [0.0] * 4 for field in outcome_fields})
    frame["burglPerPop"] = [1.0, 2.0, 3.0, 4.0]
    frame["larcPerPop"] = [2.0, 3.0, 4.0, 5.0]
    frame["autoTheftPerPop"] = [3.0, 4.0, 5.0, 6.0]
    frame["arsonsPerPop"] = [4.0, 5.0, 6.0, 7.0]
    frame["state_encoded"] = [0.1, 0.2, 0.3, 0.4]
    if leaking:
        frame["nonViolPerPop"] = frame[
            ["burglPerPop", "larcPerPop", "autoTheftPerPop", "arsonsPerPop"]
        ].sum(axis=1)
    else:
        frame["nonViolPerPop"] = [100.0, 1.0, 200.0, 2.0]
    return frame


def test_crimedata_target_construction_audit_requires_reproducible_leakage() -> None:
    audit = audit_crimedata_target_construction(
        _composite_frame(leaking=True), tolerance=0.1, minimum_fraction=0.95
    )

    assert audit == {
        "dataset_id": "crimedata",
        "reason_code": "outcome_component_leakage",
        "post_hoc": True,
        "target": "nonViolPerPop",
        "component_features": [
            "burglPerPop",
            "larcPerPop",
            "autoTheftPerPop",
            "arsonsPerPop",
        ],
        "outcome_role_features": [
            "murders",
            "murdPerPop",
            "rapes",
            "rapesPerPop",
            "robberies",
            "robbbPerPop",
            "assaults",
            "assaultPerPop",
            "ViolentCrimesPerPop",
            "burglaries",
            "burglPerPop",
            "larcenies",
            "larcPerPop",
            "autoTheft",
            "autoTheftPerPop",
            "arsons",
            "arsonsPerPop",
        ],
        "globally_supervised_features": ["state_encoded"],
        "row_count": 4,
        "matched_row_count": 4,
        "match_fraction": 1.0,
        "absolute_tolerance": 0.1,
        "median_absolute_error": 0.0,
        "construction_correlation": 1.0,
        "construction_r_squared": 1.0,
        "target_global_mean": 16.0,
        "target_global_mean_match_count": 0,
    }

    with pytest.raises(ValueError, match="target-construction leakage was not reproduced"):
        audit_crimedata_target_construction(
            _composite_frame(leaking=False),
            tolerance=0.1,
            minimum_fraction=0.95,
        )


def test_registered_exclusion_retains_exact_six_dataset_publication_scope() -> None:
    raw_rows = [
        {"dataset_id": dataset, "value": index}
        for index, dataset in enumerate((*RETAINED_DATASETS, EXCLUDED_DATASET_ID))
    ]
    front_rows = [
        {"dataset_id": dataset, "point_index": index}
        for index, dataset in enumerate((*RETAINED_DATASETS, EXCLUDED_DATASET_ID))
    ]
    original_raw = deepcopy(raw_rows)
    original_front = deepcopy(front_rows)

    retained_raw, retained_front = filter_registered_exclusion(raw_rows, front_rows)

    assert {row["dataset_id"] for row in retained_raw} == set(RETAINED_DATASETS)
    assert {row["dataset_id"] for row in retained_front} == set(RETAINED_DATASETS)
    assert raw_rows == original_raw
    assert front_rows == original_front

    with pytest.raises(ValueError, match="exact registered seven-dataset source scope"):
        filter_registered_exclusion(
            [*raw_rows, {"dataset_id": "foreign", "value": 0}], front_rows
        )


def test_formal_pipeline_excludes_quarantined_dataset() -> None:
    with open("configs/icomp2026_pipeline.yaml", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)

    assert tuple(protocol["datasets"]) == RETAINED_DATASETS
    assert EXCLUDED_DATASET_ID not in protocol["estimator"]["budgets_by_dataset"]
    assert EXCLUDED_DATASET_ID not in protocol["stage_scopes"]["alpha"]["datasets"]
    assert figure_module._CANONICAL_DATASETS == tuple(sorted(RETAINED_DATASETS))
    assert tuple(dataset_id for dataset_id, _, _ in group1_module.DATASETS) == tuple(
        sorted(RETAINED_DATASETS)
    )
