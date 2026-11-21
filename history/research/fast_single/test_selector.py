import numpy as np
import pytest
from scipy import sparse

from research.fast_single.selector import ROWS, select_fast
from research.peer_crossover.single import select_single


def objective(z, target, ids):
    if not ids:
        return np.sum(target**2)
    a = z[:, ids]
    penalty = np.sqrt(z.shape[0])
    coef = np.linalg.solve(a.T @ a + penalty*np.eye(len(ids)), a.T @ target)
    return np.sum((target-a @ coef)**2) + penalty*np.sum(coef**2)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_scores_against_refitted_objectives(task):
    rng = np.random.default_rng(5201)
    n, p = 90, 46
    x = rng.normal(size=(n, p))
    x[np.abs(x)<.8] = 0.
    x[:, -1] = 0.
    raw = x[:, 1]-x[:, 6]+rng.normal(size=n)
    if task == "classification":
        y = np.digitize(raw, [-.5, .5])
        target = np.eye(3)[y]
    else:
        y = raw
        target = (y/y.std())[:, None]
    target -= target.mean(axis=0)
    z = x-x.mean(axis=0)
    scale = z.std(axis=0)
    np.divide(z,scale,out=z,where=scale>0)
    cols, stats = select_fast(x, y, task=task, seed=541)
    peers = stats['peers']
    empty = np.array([objective(z,target,[])-objective(z,target,[j]) for j in range(p)])
    peer_gains = np.array([np.mean([objective(z,target,[k])-objective(z,target,[k,j])
                                   for k in peers if k!=j]) for j in range(p)])
    np.testing.assert_allclose(stats['tier_scores'], [empty,peer_gains], rtol=2e-10, atol=1e-10)
    votes = np.minimum(empty,peer_gains)
    floor = n*np.sum(target**2)/((n-1)*(n+np.sqrt(n)))
    w = votes**2/(votes+floor)
    count = int(np.ceil(w.sum()**2/(w@w)))
    assert stats['null_scale'] == pytest.approx(floor)
    np.testing.assert_array_equal(cols, np.sort(np.argsort(-votes,kind='stable')[:count]))
    ix_sparse, stats_sparse = select_fast(sparse.csr_matrix(x), y, task=task, seed=541)
    np.testing.assert_array_equal(ix_sparse, cols)
    np.testing.assert_allclose(stats_sparse['tier_scores'],stats['tier_scores'],rtol=1e-10,atol=1e-11)


def test_low_width_reproduces_paper_and_sketched_rows_are_repeatable():
    rng = np.random.default_rng(79)
    x = rng.normal(size=(ROWS+211, 77)).astype(np.float32)
    y = rng.normal(size=len(x))
    old, reference = select_single(x[:, :32],y,task="regression",seed=15)
    new, actual = select_fast(x[:, :32],y,task="regression",seed=15)
    np.testing.assert_array_equal(new,old)
    np.testing.assert_array_equal(actual["rank"],reference["rank"])
    ix, stats = select_fast(x,y,task="regression",seed=15)
    repeat, repeated = select_fast(x,y,task="regression",seed=15)
    np.testing.assert_array_equal(repeat,ix)
    np.testing.assert_array_equal(stats["tier_scores"],repeated["tier_scores"])
    assert stats["sketch_size"] == ROWS
    assert len(np.unique(stats["peers"])) == 8


def test_no_signal_and_sparse_large_constant_offsets():
    rng = np.random.default_rng(7299)
    x = rng.normal(size=(80,55)) + 1.0e11
    y = x[:,2]-x[:,8]
    ix, stats = select_fast(x,y,task="regression",seed=22)
    ix_sparse, s_sparse = select_fast(sparse.csr_matrix(x),y,task="regression",seed=22)
    np.testing.assert_array_equal(ix,ix_sparse)
    np.testing.assert_allclose(s_sparse['rank'],stats['rank'],rtol=1e-10)
    full, null = select_fast(x,np.ones(len(x)),task="regression")
    np.testing.assert_array_equal(full,np.arange(x.shape[1]))
    np.testing.assert_array_equal(null['rank'],np.zeros(x.shape[1]))


@pytest.mark.parametrize("field", ["data", "indices", "indptr", "all"])
@pytest.mark.parametrize("foreign_endian", [False, True], ids=["native", "swapped"])
def test_strided_csr_buffers(field, foreign_endian):
    rng = np.random.default_rng(710)
    x = rng.normal(size=(48, 40))
    x[np.abs(x) < 1.] = 0.
    y = x[:, 1] - x[:, 7] + .1*rng.normal(size=len(x))
    ref = sparse.csr_matrix(x)
    expected, stats = select_fast(ref, y, task="regression", seed=1129)
    view = ref.copy()
    for attr in ("data", "indices", "indptr"):
        if field not in (attr, "all"):
            continue
        values = getattr(ref, attr)
        dtype = values.dtype.newbyteorder("S") if foreign_endian else values.dtype
        backing = np.zeros(2*values.size, dtype=dtype)
        backing[::2] = values
        setattr(view, attr, backing[::2])
    cols, observed = select_fast(view, y, task="regression", seed=1129)
    np.testing.assert_array_equal(observed["tier_scores"], stats["tier_scores"])
    assert observed["selected_width"] == stats["selected_width"]
    np.testing.assert_array_equal(cols, expected)
