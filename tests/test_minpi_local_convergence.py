"""Regression tests for the experimental singleton-deletion certificate."""
import numpy as np
import pandas as pd
import pytest

from experiments.test.minpi_local_convergence import _panel_weights, select
from minppi_mofs.piv3_compact_witness import _decision, _measure, _sample


class MaskModel:
    """Predict a known signal exactly only on supports satisfying the supplied rule."""
    def __init__(self, columns, good):
        self.columns = columns
        self.good = good
        self.calls = []

    def predict_proba(self, x):
        indices = np.flatnonzero(np.any(x != 0., axis=0))
        mask = frozenset(self.columns[i] for i in indices)
        self.calls.append(mask)
        p = .5 + .4*x[:, indices[0]] if len(indices) and self.good(mask) else np.full(len(x), .5)
        return np.column_stack((1-p, p))


def controlled_data(columns):
    labels = np.tile([0, 1], 30)
    frame = pd.DataFrame(np.tile(2*labels[:, None]-1, (1, len(columns))), columns=columns)
    # Each median is zero, so the model can read the simulated support exactly.
    return frame, labels


@pytest.mark.parametrize("mode", ["legacy", "full"])
def test_terminal_certificate_replays_every_remaining_column_and_context(mode):
    cols = list("abcdef")
    data, y = controlled_data(cols)
    # b matters in all contexts. Removing c is safe in the small context alone.
    def good(mask):
        return "b" in mask and ("e" not in mask or "c" in mask)

    model = MaskModel(cols, good)
    anchor = dict(order=cols, k=4)
    selected, info = select(anchor, model, data, data, y, task="binary", utility="roc_auc",
                            search_seed=42, vote_mode=mode)
    assert selected == ["b", "c"]  # The leading Gain feature (a) was eligible and deleted.
    assert info["distinct_contexts"] == 2  # Here the medium and outer additions coincide.
    assert len(model.calls) == len(set(model.calls)) == info["voting_calls"]
    assert info["stop_reason"] == "no_accepted_singleton"
    assert info["accepted_per_sweep"] == [2, 0]
    last = info["sweeps"][-1]
    assert info["final_sweep_covered"] == last["start_k"] == last["end_k"] == len(selected)
    assert {item["column"] for item in last["decisions"]} == set(selected)

    # Reconstruct all three contexts, including duplicate ones, and evaluate every
    # possible final deletion with the original sklearn-based scores and vote.
    if mode == "legacy":
        rows, weights = _sample(y, "binary", np.random.SeedSequence([42, 659]))
    else:
        rows, weights, _ = _panel_weights(y, "binary", 42)
    def scores(mask):
        x = data.to_numpy()[rows].copy()
        x[:, [j for j, name in enumerate(cols) if name not in mask]] = 0
        pred = model.predict_proba(x)
        return [_measure(y[rows], pred, w, "binary", False) for w in weights]

    initial = scores(set(cols[:4]))[0]
    for column in selected:
        before = [set(selected) | set(tail) for tail in ([], cols[4:8], cols[4:])]
        accepted, _ = _decision([scores(s) for s in before], [scores(s-{column}) for s in before],
                                initial, "min")
        assert not accepted
    c, = [d for d in last["decisions"] if d["column"] == "c"]
    assert c["decisive_rule"] == "context_1"  # Middle context vetoes a safe small-context deletion.


@pytest.mark.parametrize("mode", ["legacy", "full"])
def test_previous_rejection_is_reconsidered_after_another_deletion(mode):
    cols = list("abcd")
    data, y = controlled_data(cols)
    allowed = [set("abc"), set("ac"), set("a")]
    model = MaskModel(cols, lambda mask: (mask & set("abc")) in allowed)
    selected, info = select(dict(order=cols, k=3), model, data, data, y, task="binary",
                            utility="roc_auc", search_seed=42, vote_mode=mode)
    assert selected == ["a"]
    assert info["stop_reason"] == "minimum_size" and info["final_sweep_covered"] == 0
    assert [s["removed"] for s in info["sweeps"]] == [["b"], ["c"]]
    c_history = [d["accepted"] for sweep in info["sweeps"] for d in sweep["decisions"] if d["column"] == "c"]
    assert c_history == [False, True]
    assert len(model.calls) == len(set(model.calls))


def test_one_column_avoids_all_predictions():
    cols = ["a"]
    data, y = controlled_data(cols)
    model = MaskModel(cols, lambda mask: True)
    selected, info = select(dict(order=cols, k=1), model, data, data, y, task="binary",
                            utility="roc_auc", search_seed=42, vote_mode="full")
    assert selected == cols and info["stop_reason"] == "minimum_size"
    assert info["proposals"] == info["voting_calls"] == info["sweeps_completed"] == 0
