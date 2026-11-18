import numpy as np
import pytest

from research.lowdim.method import fast_support


def objective(Z, Y, columns):
    if not columns:
        return float(np.square(Y).sum())
    A = Z[:, columns]
    penalty = np.sqrt(len(Z))
    coefficients = np.linalg.solve(A.T@A+penalty*np.eye(len(columns)), A.T@Y)
    return np.square(Y-A@coefficients).sum()+penalty*np.square(coefficients).sum()


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("kind", ["float32","float64"])
def test_matches_all_refits(task, kind):
    rng = np.random.default_rng(73645)
    X = rng.normal(size=(47,9)).astype(kind)[:,::-2]
    X[:,1] = X[:,0]
    X[:,2] = 3.
    y = X[:,0].astype(float)*0.6+X[:,3]*0.1+rng.normal(size=len(X))
    if task == "classification":
        y = np.searchsorted([-.2,.8],y)
        Y = np.eye(3)[y]
    else:
        Y = y[:,None]/y.std()
    Y -= Y.mean(axis=0)
    Z = X.astype(float)
    Z -= Z.mean(axis=0)
    scale = Z.std(axis=0)
    Z /= np.where(scale>0,scale,1.)
    cols, info = fast_support(X,y,task=task)
    v0 = [objective(Z,Y,[])-objective(Z,Y,[j]) for j in range(X.shape[1])]
    v1 = [np.mean([objective(Z,Y,[k])-objective(Z,Y,[k,j])
                   for k in range(X.shape[1]) if j!=k]) for j in range(X.shape[1])]
    np.testing.assert_allclose(info["tier_scores"], [v0,v1], atol=2e-11)
    vote = np.minimum(v0,v1)
    k = int(np.ceil(vote.sum()**2/(vote@vote)))
    assert len(cols) == k
    boundary = np.sort(info["rank"])[-k]
    assert (info["rank"][cols] >= boundary).all()


def test_constant_response_and_shifted_regression():
    X = np.random.default_rng(75642).normal(size=(89, 6))
    cols, _ = fast_support(X, np.ones(len(X)), task="regression")
    np.testing.assert_array_equal(cols, np.arange(X.shape[1]))
    y = 0.7*X[:,0] + X[:,1] - X[:,2]
    a, info= fast_support(X, y, task="regression")
    b, shifted= fast_support(X+8e7, y*(-1e240), task="regression")
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(info['rank'],shifted['rank'],rtol=2e-7,atol=1e-8)


def test_seeded_row_sample_and_single_feature():
    rng = np.random.default_rng(1567)
    X = rng.normal(size=(6503, 9))
    y = X[:,0]+rng.normal(size=len(X))
    _, first = fast_support(X,y,task="regression",seed=314)
    _, again = fast_support(X,y,task="regression",seed=314)
    _, changed = fast_support(X,y,task="regression",seed=159)
    np.testing.assert_array_equal(first["rank"], again["rank"])
    assert not np.array_equal(first["rank"], changed["rank"])
    cols, _ = fast_support(X[:,:1], y, task="regression")
    np.testing.assert_array_equal(cols,[0])
