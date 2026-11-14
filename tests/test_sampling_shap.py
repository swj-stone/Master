from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.baselines.sampling_shap import select


class SyntheticModel:
    def __init__(self):
        self.calls = []

    def predict(self, x):
        self.calls.append(len(x))
        return 2 * x[:, 1]

    def predict_proba(self, x):
        self.calls.append(len(x))
        score = 1.0 / (1.0 + np.exp(-x[:, 1]))
        return np.column_stack((.1 + .5 * score, .6 - .4 * score, .3 - .1 * score))


def context(task, width=3):
    rng = np.random.default_rng(18)
    columns = [f'col_{i}' for i in range(width)]
    train = pd.DataFrame(rng.normal(size=(160, width)), columns=columns)
    rank = pd.DataFrame(rng.normal(size=(48, width)), columns=columns)
    return SimpleNamespace(seed=42, columns=columns, task=task, full_model=SyntheticModel(),
                           train=train, ranking=rank, yr=np.arange(len(rank)) % 3,
                           progress=lambda *args, **kwargs: None)


@pytest.mark.parametrize('task', ['regression', 'multiclass'])
def test_official_sampler_uses_only_model_predictions_and_counts_pairs(task):
    ctx = context(task)
    choice = select(ctx)
    diag = choice.diagnostics
    assert choice.selected is None
    assert np.argmax(choice.scores) == 1
    np.testing.assert_allclose(choice.scores[[0, 2]], 0, atol=1e-10)
    assert diag['paired_samples'] == 32 * 3 * 12
    assert diag['sampling_prediction_rows'] == 2 * diag['paired_samples']
    assert diag['prediction_rows'] == 128 + 32 + diag['sampling_prediction_rows']
    assert diag['prediction_rows'] == sum(ctx.full_model.calls)
    assert diag['prediction_calls'] == len(ctx.full_model.calls)
    assert diag['background_prediction_rows'] == len(diag['background_rows']) == 128
    assert diag['explained_prediction_rows'] == len(diag['ranking_rows']) == 32
    assert len(set(diag['ranking_rows'])) == 32
    if task == 'multiclass':
        assert set(ctx.yr[diag['ranking_rows']]) == {0, 1, 2}


def test_selector_is_reproducible_and_restores_numpy_random_state():
    ctx = context('multiclass')
    before = np.random.get_state()
    expected_after = np.random.RandomState()
    expected_after.set_state(before)
    first = select(ctx)
    second = select(ctx)
    actual_after = np.random.RandomState()
    actual_after.set_state(np.random.get_state())
    np.testing.assert_array_equal(actual_after.random_sample(10), expected_after.random_sample(10))
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.diagnostics == second.diagnostics
