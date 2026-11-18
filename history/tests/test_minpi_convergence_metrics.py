import numpy as np
import pytest

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.piv3_compact_witness import _decision, _measure, _sample


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("draw", ["bootstrap", "panels"])
def test_scores_and_min_votes_match_weighted_reference(task, draw):
    rng = np.random.default_rng(831)
    classes = 5 if task == "multiclass" else 2
    y = np.tile(np.arange(classes), 24) if task != "regression" else rng.normal(size=96)
    utility = {"regression": "r2", "binary": "roc_auc", "multiclass": "ovr_auc"}[task]
    rows, weights = _sample(y, task, 45)
    y = y[rows]
    if draw == "panels":
        assignment = np.repeat(np.arange(3), len(y) // 3)
        weights = np.vstack([rng.uniform(.2, 1., len(y)) * b for b in
                             [np.ones(len(y)), *(assignment == i for i in range(3))]])
    fast = make_scorer(y, weights, task, utility, True)
    expected, actual = [], []
    for _ in range(6):
        if task == "regression":
            pred = y + rng.normal(size=len(y))
        else:
            # Ties and a constant score column exercise half-credit handling.
            pred = rng.integers(0, 5, size=(len(y), classes)).astype(float)
            pred[:, -1] = 2.
        reference = [_measure(y, pred, w, task, True) for w in weights]
        measured = fast(pred)
        expected.append(reference)
        actual.append(measured)
        for truth, got in zip(reference, measured, strict=True):
            assert got == pytest.approx(truth, rel=1e-12, abs=1e-12)
    for scores in (expected, actual):
        assert _decision(scores[:3], scores[:3], scores[0][0], "min")[0]
    actual_decision = _decision(actual[:3], actual[3:], actual[0][0], "min")
    assert actual_decision == _decision(expected[:3], expected[3:], expected[0][0], "min")


def test_missing_classes_raise_instead_of_undefined_auc_passing_a_vote():
    with pytest.raises(ValueError, match="positive and negative mass"):
        make_scorer(np.array([0, 0, 1, 1]), [[1, 1, 1, 1], [1, 1, 0, 0]],
                    "binary", "roc_auc", True)
