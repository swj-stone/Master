import numpy as np
import pandas as pd
import pytest

from minppi_mofs import fold_consensus as fc


def setup_run(monkeypatch, *, nrows=30):
    features = list("abcde") + ["[f]", "g", "h"]
    X = pd.DataFrame(np.zeros((nrows, len(features))), columns=features)
    y = np.resize([0, 1], nrows)
    fits, holds = [], []

    class FakeModel:
        def fit(self, x, labels):
            self.index = len(fits)
            fits.append(tuple(x.index))
            assert list(x.columns) == [f"f{i}" for i in range(8)]
            assert set(labels) == {0, 1}

        def get_booster(self):
            return self

        def get_score(self, *, importance_type):
            assert importance_type == "gain"
            return {f"f{i}": float(i + 1) for i in range(8)}

    monkeypatch.setattr(fc, "make_model", lambda **_: FakeModel())

    def shap(model, hold, *, seed, max_rows):
        assert seed == 42 and len(hold) <= max_rows == 512
        assert not set(hold.index) & set(fits[model.index])
        holds.append(tuple(hold.index))
        return np.arange(8)

    monkeypatch.setattr(fc, "fitted_tree_shap_importance", shap)
    # d and h removed; d is a negative witness. g positive, e unseen.
    signs = dict(zip(features, [0, 0, 0, -1, None, 0, 1, 0], strict=True))
    kwargs = dict(policy=dict(task="binary", model_params={}, backend="cpu", seed=42))
    return (X, y, signs, 0b01110111), kwargs, fits, holds


def test_opposite_training_halves_and_protected_subsets(monkeypatch):
    args, kwargs, fits, holds = setup_run(monkeypatch, nrows=1200)
    seen = []

    def evaluate(mask):
        assert mask & 0b01010000 == 0b01010000 and not mask & 0b00001000
        assert mask.bit_count() <= args[-1].bit_count()
        seen.append(mask)
        return (1.04, 1.02, 1.02)

    result = fc.select_fold_consensus(*args, evaluate, (1.0,) * 3, (1.0,) * 3, 0.01, **kwargs)
    assert set(fits[0]).isdisjoint(fits[1]) and set(fits[0] + fits[1]) == set(range(1200))
    assert len(holds[0]) == len(holds[1]) == 512
    assert set(holds[0]) <= set(fits[1]) and set(holds[1]) <= set(fits[0])
    assert result["ranking_fits"] == 2 and len(result["fold_rankings"]) == 2
    assert result["evaluated_count"] == len(set(seen)) == len(seen) <= 5
    assert result["methods"]["guarded_incumbent"]["mask"] == args[3]
    assert result["methods"]["fold_consensus"]["mask"] == result["selected_mask"]
    assert result["consensus_order"] == list("abcd") + ["[f]", "h"]
    assert result["removed_features"] == ["a", "b", "d"]


def test_fold_disagreement_uses_worst_rank_and_stable_ties():
    first = dict(a=0.1, b=0.5, c=0.4, d=0.4)
    second = dict(a=0.9, b=0.5, c=0.5, d=0.5)
    assert fc._order_consensus(list("abcd"), set("abcd"), [first, second]) == list("cdba")


@pytest.mark.parametrize(
    "allowances,expected_fits", [([False], 0), ([True, False], 1), ([True, True, False], 2)]
)
def test_partial_budget_keeps_incumbent_without_unverified_fit(monkeypatch, allowances, expected_fits):
    args, kwargs, fits, _ = setup_run(monkeypatch)
    allowed = iter(allowances)
    out = fc.select_fold_consensus(
        *args,
        lambda _: pytest.fail("no candidate fit budget"),
        (1.0,) * 3,
        (1.0,) * 3,
        0.01,
        allow_next=lambda: next(allowed),
        **kwargs,
    )
    assert out["budget_stopped"] and out["selected_mask"] == args[3]
    assert len(fits) == out["ranking_fits"] == expected_fits and out["evaluated_count"] == 0


def test_choose_only_improvements_with_no_loss_on_three_slices(monkeypatch):
    args, kw, *_ = setup_run(monkeypatch)
    out = fc.select_fold_consensus(*args, lambda _: (1.5, 0.999, 1.5), (1.0,) * 3, (1.0,) * 3, 0.01, **kw)
    assert all(not r["accepted"] for r in out["trials"])
    assert out["selected_mask"] == args[3] and not out["budget_stopped"]
