"""Numeric identities used by the stagewise deletion path."""
import numpy as np
from sklearn.linear_model import Ridge

from experiments.minpi_rework.path_guard import prefix_increments
from minppi_mofs.local_vote_path import DeletionGame, GCVGame, path_order, quota_integral
from minppi_mofs.sampled_linear_minpi import _utility


def check():
    rng = np.random.default_rng(4391)
    x, z = rng.normal(size=(80, 9)), rng.normal(size=(30, 9))
    x[:, 4] = x[:, 3] + 1e-9 * x[:, 4]
    z[:, 4] = z[:, 3] + 1e-9 * z[:, 4]
    y, t = rng.normal(size=(80, 3)), rng.normal(size=(30, 3))
    for ridge in (.01, 1., 20.):
        game = DeletionGame(x,y,z,t,ridge)
        pooled = GCVGame(x,y,ridge)
        base, directions, slopes = prefix_increments(x,y,z,ridge)
        for width in range(1,x.shape[1]+1):
            pred = Ridge(alpha=ridge).fit(x[:,:width],y).predict(z[:,:width])
            np.testing.assert_allclose(base + directions[:,:width]@slopes[:width],pred,atol=1e-11,rtol=0)
        for cols in (np.array([2]), np.array([4, 3, 7, 1]), np.arange(9)):
            def refit(ids):
                hat = (Ridge(alpha=ridge).fit(x[:,ids], y).predict(z[:,ids])
                       if len(ids) else y.mean(0))
                return _utility(np.sum((hat-t)**2) / len(t), game.scale)
            base, without, _ = game.edges(cols)
            np.testing.assert_allclose(base, refit(cols), atol=1e-11, rtol=0)
            np.testing.assert_allclose(without, [refit(np.delete(cols,j)) for j in range(len(cols))],
                                       atol=1e-11, rtol=0)
            def direct_gcv(ids):
                design = x[:,ids] - x[:,ids].mean(axis=0)
                hat = design @ np.linalg.solve(design.T@design + ridge*np.eye(len(ids)), design.T)
                hat += 1/len(x)
                risk = np.sum((y-hat@y)**2) / len(x) / (1-np.trace(hat)/len(x))**2
                return _utility(risk, pooled.scale)
            u, v, _ = pooled.edges(cols)
            np.testing.assert_allclose(u, direct_gcv(cols), atol=1e-11)
            np.testing.assert_allclose(v, [direct_gcv(np.delete(cols,j)) for j in range(len(cols))],atol=1e-11)
    a, b = rng.uniform(size=(5, 47)), rng.uniform(size=(5, 47))
    # Brute-force integration on the endpoints provides an exact finite-sum reference.
    edges = np.unique(np.r_[a.ravel(), b.ravel(), 0., 1.])
    ref = np.zeros(a.shape[1])
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        theta = (right+left)/2
        ref += (right-left)*((a >= theta).astype(int) - (b >= theta)).min(axis=0)
    np.testing.assert_allclose(quota_integral(a,b), ref, atol=1e-13, rtol=0)
    result, meta = path_order(x, y[:,0], z, t[:,0], task="regression")
    assert sorted(result) == list(range(9))
    assert meta["stages"][-1]["size"] == 2
    for suffix in ("", "_gcv"):
        lazy, la = path_order(x,y[:,0],z,t[:,0],task="regression",mode="signed"+suffix)
        eager, ea = path_order(x,y[:,0],z,t[:,0],task="regression",mode="signed_eager"+suffix)
        np.testing.assert_array_equal(lazy,eager)
        assert la["skipped_edges"] + la["deletion_edges"] == ea["deletion_edges"]
    print("Exact refit, context integral, and nested support checks passed")


if __name__ == "__main__":
    check()
