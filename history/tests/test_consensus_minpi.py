import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.consensus_minpi import ConsensusMinPISelector, consensus_path_from_training, consensus_top_from_training
from minppi_mofs.environment_minpi import environment_path_from_training


def test_true_group_paths_match_independent_environment_orders_and_borda_ranks():
    rng = np.random.default_rng(241)
    X = rng.normal(size=(180, 9))
    y = X[:, 1] + .7 * X[:, 6] + rng.normal(size=len(X))
    groups = np.repeat(np.arange(9), 20)
    result, info = consensus_path_from_training(X, y, (2, 6), groups=groups,
        ordered_groups=True, task="regression", random_state=15)
    assert info["aggregation"] == "min" and len(info["paths"]) == 3
    rank_sum = np.zeros(9)
    for path in info["paths"]:
        ref, _ = environment_path_from_training(X, y, range(1, 9), task="regression",
            groups=groups, ordered_groups=True, random_state=path["seed"])
        for h in ref:
            np.testing.assert_array_equal(np.sort(path["order"][:h]), ref[h])
        for fold in path["folds"]:
            assert groups[fold["fit_rows"]].max() < groups[fold["held_rows"]].min()
        rank_sum[path["order"]] += np.arange(9)
    order = np.argsort(rank_sum, kind="stable")
    for h, chosen in result.items():
        np.testing.assert_array_equal(chosen, np.sort(order[:h]))
        separate, second = consensus_top_from_training(X, y, h, task="regression", random_state=15,
            groups=groups, ordered_groups=True)
        np.testing.assert_array_equal(chosen, separate)
        assert info == second


def test_iid_mean_game_matches_brute_force_held_refits_with_prior_column_removed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(65, 7))
    y = np.digitize(X[:, 2] - X[:, 0] + .5*rng.normal(size=len(X)), [-.8,.4])
    result, info = consensus_path_from_training(X,y,(3,5),n_paths=3,random_state=13)
    assert info["aggregation"] == "mean" and info["distinct_paths"] > 1
    assert len({p["seed"] for p in info["paths"]}) == 3
    for path in info["paths"]:
        folds = []
        for fold in path["folds"]:
            tr, te = fold["fit_rows"], fold["held_rows"]
            scaler = StandardScaler().fit(X[tr])
            fit, check = scaler.transform(X[tr]), scaler.transform(X[te])
            targets = (y[:,None]==np.unique(y)).astype(float)
            targets -= targets[tr].mean(axis=0)
            _, c, counts = np.unique(y[te], return_inverse=True, return_counts=True)
            w = 1 / counts[c]
            w /= (w * np.square(targets[te]).sum(axis=1)).sum()
            folds.append((fit, check, targets[tr], targets[te], w))
        def mean_loss(cols):
            losses = []
            for fit, check, fit_y, check_y, w in folds:
                pred = np.zeros_like(check_y)
                if cols:
                    model = Ridge(alpha=1,solver="svd",fit_intercept=False).fit(fit[:,cols],fit_y)
                    pred = model.predict(check[:,cols])
                losses.append((w * np.square(check_y - pred).sum(axis=1)).sum())
            return np.mean(losses)
        selected = []
        for size, stop in [(0,1),(1,2),(2,4),(4,7)]:
            gains = {}
            prefixes = [selected] if size==0 else [selected[:-1],selected]
            pool = [j for j in range(7) if j not in selected]
            for j in pool:
                gains[j] = min(mean_loss(s)-mean_loss(s+[j]) for s in prefixes)
            selected.extend(sorted(pool, key=lambda j: -gains[j])[:stop-size])
        assert selected == path["order"]
    assert set(result[3]).issubset(result[5])


def test_blocking_groups_are_disjoint_without_treating_them_as_environments():
    rng = np.random.default_rng(49)
    X = rng.normal(size=(90, 8))
    y = np.tile([0,1,2],30)
    groups = np.repeat(np.arange(30),3)
    sel = ConsensusMinPISelector(tolerance=1)
    ert = clone(sel).fit(X,y,groups=groups,environment_groups=False)
    info = ert.diagnostics_
    assert info["rule"] == "consensus" and info["consensus_path"]["aggregation"] == "mean"
    disc = np.array(info["discovery_rows"])
    held = np.concatenate(info["validation_rows"])
    assert set(groups[disc]).isdisjoint(groups[held])
    for path in info["consensus_path"]["paths"]:
        for fold in path["folds"]:
            assert set(groups[disc[fold["fit_rows"]]]).isdisjoint(groups[disc[fold["held_rows"]]])
    test = clone(sel).fit(X,y,groups=groups,environment_groups=False)
    assert test.diagnostics_ == info
    np.testing.assert_array_equal(test.transform(X), ert.transform(X))
    with pytest.raises(ValueError):
        sel.fit(X,y,groups=groups,environment_groups=False,ordered_groups=True)
