import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from experiments.compare_thesis_sign_aware import audit_confirmation, operating_point, relation
from minppi_mofs.removal_confirmation import select_confirmed_removal


def test_operating_points_keep_equal_and_noncomparable_cases_distinct():
    baseline = dict(removed=5, test_utility=.8)
    def compare(removed, utility):
        return relation(dict(removed=removed, test_utility=utility), baseline)

    assert compare(5, .8) == "equivalent"
    assert compare(5, .81) == compare(6, .8) == "dominates"
    assert compare(4, .8) == "dominated"
    assert compare(6, .79) == compare(4, .81) == "tradeoff"


def test_only_unrestricted_controls_can_drop_sampled_positive():
    entry = dict(mask=2, removed_count=1, removed_features=["positive"], scores=[.7]*3,
                 test_delta=0, test_utility=.7)
    result = dict(features=["positive","zero"], sample_minima={"positive":1, "zero":0},
                  full_validation_scores=[.7]*3, full_test_utility=.7,
                  method_masks={"matched_gain":2, "sign_worst":2},
                  method_results={"matched_gain":entry, "sign_worst":entry},
                  method_test_utilities={"matched_gain":.7, "sign_worst":.7})
    assert operating_point(result, "matched_gain", .005)["removed"] == 1
    with pytest.raises(ValueError, match="protected feature"):
        operating_point(result, "sign_worst", .005)


def test_confirmation_audit_checks_actual_training_rows_and_second_gate(tmp_path):
    y = np.repeat([0, 1], 12)
    pd.DataFrame({"x":np.arange(24), "y":y}).to_csv(tmp_path/"train.csv", index=False)
    fit, holdout = train_test_split(np.arange(24), test_size=.25, random_state=42, stratify=y)
    split = dict(seed=42, stratified=True, sizes=[18, 6],
                 index_sha256=[hashlib.sha256(np.sort(a).astype("<i8").tobytes()).hexdigest()
                               for a in (fit, holdout)])
    masks = {"structured":3, "sign_worst":2}
    scores = {3:.9, 2:.8}
    record = select_confirmed_removal(("a","b"), masks, {3:(.9,)*3, 2:(.9,)*3}, .005, scores.__getitem__)
    record.update(split=split, candidate_masks=masks)
    result = dict(dataset="sample", confirmation=record, features=["a","b"], confirmation_fits=2,
                  confirmation_scores={hex(m):v for m,v in scores.items()}, selection={"selected_mask":3},
                  method_masks={"confirmed":3}, full_validation_scores=[.9]*3, full_test_utility=.85,
                  retained_test_utility=.85, gate={"noninferiority_margin":.005}, policy={"task":"binary"})
    authority = {"train_relpath":"train.csv", "target":"y"}
    assert audit_confirmation(result, tmp_path, authority)["confirmation_source"] == "full_set"
    record["trials"][1]["confirmation_gate_passed"] = True
    with pytest.raises(ValueError, match="confirmation gates"):
        audit_confirmation(result, tmp_path, authority)
