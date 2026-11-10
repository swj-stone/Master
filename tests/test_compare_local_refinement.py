import numpy as np
import pytest
import hashlib
from sklearn.model_selection import train_test_split

from experiments.compare_local_refinement import paired_interval, stability_summary, verify_fold_halves


def test_bootstrap_pairs_the_same_rows_and_preserves_both_classes():
    y = np.array([0, 0, 0, 1])
    incumbent = np.column_stack((y, 1-y))
    better = incumbent[:, ::-1]
    assert paired_interval(y, better, incumbent, "roc_auc", draws=12) == pytest.approx((1., 1.))
    assert paired_interval(y, better, better, "roc_auc", draws=12) == (0., 0.)


def test_stability_requires_multiple_improvements_and_no_losses_across_all_six():
    rows = [dict(test_delta_vs_incumbent=d, margin=.005, removed=8, incumbent_removed=8,
                 within_test_margin=True, ci_low=0., paired_sampling=True) for d in [.0006,.0007,0.,0.,0.,0.]]
    assert stability_summary(rows, 6)["success"]
    assert not stability_summary(rows[:5], 6)["success"]
    rows[-1]["test_delta_vs_incumbent"] = -.0006
    assert not stability_summary(rows, 6)["success"]
    rows[-1].update(test_delta_vs_incumbent=0., removed=7)
    assert not stability_summary(rows, 6)["success"]


def test_fold_audit_matches_opposite_half_hashes_and_partial_training_count():
    y = np.tile([0,1], 800)
    sides = [np.sort(v) for v in train_test_split(np.arange(len(y)),test_size=.5,random_state=42,stratify=y)]
    def h(rows):
        return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
    sampled = [half[np.sort(np.random.default_rng(42).choice(len(half),512,replace=False))]
               for half in sides]
    split = dict(seed=42, stratified=True, sizes=[800,800],
                 index_space="zero-based positions in frozen training CSV",
                 index_sha256=[h(v) for v in sides], shap_rows=[512],
                 shap_index_sha256=[h(sampled[1])])
    verify_fold_halves(y, "binary", split, 1)
    split["shap_index_sha256"] = [h(sampled[0])]
    with pytest.raises(ValueError, match="SHAP"):
        verify_fold_halves(y, "binary", split, 1)
