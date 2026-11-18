import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.backward_minpi import _Deletion, _deletion_cost, backward_path_from_training, backward_top_from_training
from minppi_mofs.environment_minpi import InsufficientEnvironmentsError


@pytest.mark.parametrize("n,p", [(18, 7), (7, 18)])
def test_deletion_and_anchor_update_equal_two_explicit_refits(n, p):
    rng = np.random.default_rng(31)
    fit, check = rng.normal(size=(n, p)), rng.normal(size=(10, p))
    target, teacher = rng.normal(size=(n, 3)), rng.normal(size=(len(check), 3))
    fit[:, 3] = fit[:, 2]
    state = _Deletion(fit, check, target)
    def loss(cols):
        model = Ridge(alpha=1, solver="svd", fit_intercept=False).fit(fit[:, cols], target)
        return np.square(teacher - model.predict(check[:, cols])).sum()
    for drop in ([], [4]):
        cols = np.delete(np.arange(p), drop)
        base = loss(cols)
        params = (state.prediction, state.beta, state.diag, state.innovation) if not drop else state.after_one(4)
        score = _deletion_cost(teacher, *params)
        for j in cols:
            assert score[j] == pytest.approx(loss(cols[cols != j]) - base, abs=2e-11)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_complete_order_matches_explicit_teacher_fidelity_oracle(task):
    rng = np.random.default_rng(54)
    X = rng.normal(size=(80, 9))
    z = rng.normal(size=len(X)) + 2 * X[:, 0] - 2.5 * X[:, 2]
    y = np.digitize(z, [-1, .5]) if task == "classification" else z
    groups = np.repeat(np.arange(8), 10)
    counts = (1, 2, 5, 7, 9)
    actual, info = backward_path_from_training(X, y, counts, task=task, groups=groups, ordered_groups=True)
    folds = []
    for fold in info["folds"]:
        tr, te = fold["fit_rows"], fold["held_rows"]
        assert groups[tr].max() < groups[te].min()
        scaler = StandardScaler().fit(X[tr])
        A, H = scaler.transform(X[tr]), scaler.transform(X[te])
        if task == "classification":
            target = (y[tr, None] == np.unique(y)).astype(float)
            classes, code, amount = np.unique(y[te], return_inverse=True, return_counts=True)
            weight = 1 / (len(classes) * amount[code])
        else:
            target = y[tr, None].copy()
            weight = np.full(len(te), 1 / len(te))
        target -= target.mean(axis=0)
        teacher = Ridge(alpha=1, fit_intercept=False, solver="svd").fit(A, target).predict(H).reshape(len(te), -1)
        folds.append((A, H, target, teacher, weight))
    def losses(cols):
        out = []
        for A, H, target, teacher, weight in folds:
            pred = np.zeros_like(teacher)
            if len(cols):
                model = Ridge(alpha=1., fit_intercept=False, solver="svd").fit(A[:, cols], target)
                pred = model.predict(H[:, cols]).reshape(teacher.shape)
            e = (weight * np.square(teacher).sum(axis=1)).sum()
            out.append((weight * np.square(teacher - pred).sum(axis=1)).sum() / e)
        return np.array(out)
    live = list(range(X.shape[1]))
    removed = []
    while len(live) > 1:
        baseline = losses(live)
        worst = {j: max(losses([f for f in live if f != j]) - baseline) for j in live}
        anchor = min(live, key=lambda j: (worst[j], -j))
        size = len(live) // 2
        other = [f for f in live if f != anchor]
        baseline = losses(other)
        for j in other:
            worst[j] = max(worst[j], max(losses([f for f in other if f != j]) - baseline))
        batch = [anchor] + sorted(other, key=lambda j: (worst[j], -j))[:size-1]
        removed += batch
        live = [f for f in live if f not in batch]
    elimination = removed + live
    for h in counts:
        np.testing.assert_array_equal(actual[h], sorted(elimination[-h:]))
        single = backward_top_from_training(X, y, h, task=task, groups=groups, ordered_groups=True)[0]
        np.testing.assert_array_equal(actual[h], single)


def test_flat_teacher_ties_and_insufficient_environments():
    X = np.tile(np.arange(8., dtype=float), (36, 1))
    X[:, 1] = np.nan
    y = np.tile([0, 1, 2], 12)
    cols, info = backward_top_from_training(X, y, 3)
    assert cols.tolist() == [0, 1, 2]
    assert set(info["normalization"]) == {"target_energy"}
    with pytest.raises(InsufficientEnvironmentsError):
        backward_top_from_training(X, y, 3, groups=np.ones(len(y)))
    assert backward_top_from_training(X, y, 8, groups=np.ones(len(y)))[0].size == 8
