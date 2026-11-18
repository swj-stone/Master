"""Reference checks for signed Banzhaf and the published ShapG approximation."""
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd

from minppi_mofs.baselines import banzhaf_fs, shapg


def context(x, fn, *, mode="native", active=None):
    return SimpleNamespace(mode=mode, train=x, columns=x.columns.tolist(), seed=42, active=active,
                           values=lambda masks: [fn(set(mask)) for mask in masks],
                           progress=lambda phase, **fields: None)


def test_banzhaf_against_paired_conditional_means_and_known_nulls():
    x = pd.DataFrame(np.arange(32).reshape(8, 4), columns=list("abcd"))
    value = lambda s: 3 * (0 in s) - 2 * (1 in s) + 1.75 * ({0,2} <= s)
    rng = np.random.default_rng(42)
    first = rng.integers(0, 2, size=(32, 4)).astype(bool)
    masks = np.vstack([first,~first])
    v = np.array([value(set(np.flatnonzero(m))) for m in masks])
    expected = [v[masks[:,i]].mean()-v[~masks[:,i]].mean() for i in range(4)]
    out = banzhaf_fs.select(context(x, value))
    np.testing.assert_allclose(out.scores, expected)
    frozen = banzhaf_fs.select(context(x, value, mode="frozen",active=(0,1,2)))
    np.testing.assert_allclose(frozen.scores, [*expected[:3],0])
    assert out.diagnostics["sampled_masks"] == 64


def test_pearson_edge_cover_matches_algorithm_one_lexicographic_sort():
    x = pd.DataFrame(np.random.default_rng(5).normal(size=(18, 8)))
    x[6] = x[1]
    x[7] = 2.
    graph = shapg._paper_neighbors(x)
    # Pandas' pairwise accumulator can differ in the last bit for an exact
    # duplicate column, depending on which endpoint is accumulated first.
    corr = x.corr().abs().fillna(0).to_numpy().round(13)
    edges = sorted(itertools.combinations(range(len(corr)),2),key=lambda pair: (-corr[pair],pair))
    ref = [set() for _ in corr]
    for i,j in edges:
        if not ref[i] or not ref[j]:
            ref[i].add(j)
            ref[j].add(i)
    assert graph == ref


def test_shapg_scores_use_published_coefficients_and_same_local_game(monkeypatch):
    neighbors = [{1,2,3,4,5}, {0}, {0}, {0}, {0}, {0}]
    monkeypatch.setattr(shapg, "_paper_neighbors", lambda x: neighbors)
    x = pd.DataFrame(np.eye(6), columns=list("abcdef"))
    w = np.array([1,-2,3,-4,5,0], dtype=float)
    value = lambda s: 1. + sum(w[i] for i in s)
    # At the center degree=5 and each of the 8 local marginal evaluations
    # equals w[0], weighted by 6/(4*2^5). Neighbor scores are unscaled.
    expected = w.copy()
    expected[0] *= 8*6/(4*32)
    actual = shapg.select(context(x, value))
    np.testing.assert_allclose(actual.scores, expected)
    frozen = shapg.select(context(x, value, mode="frozen",active=tuple(range(5))))
    np.testing.assert_allclose(frozen.scores, expected)
    assert actual.diagnostics["graph_edges"] == 5

