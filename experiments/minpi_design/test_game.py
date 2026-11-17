"""Compare the endpoint formula and branch pruning against independent calculations."""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from experiments.minpi_design.game import Design, Game
from experiments.minpi_design.path_native import bisect


def test_all_mask_sizes_and_dual():
    rng = np.random.default_rng(211)
    x = rng.normal(size=(31,5))
    x[:,1] = .8*x[:,0]+.2*x[:,1]
    y = x[:,:2]+.2*rng.normal(size=(31,2))
    y -= y.mean(axis=0)
    design,z = Design.fit(x,"spline")
    d = design.center.shape[1]
    p = x.shape[1]
    game = Game(z,y,d)
    beta = rng.normal(size=(p*d,2))
    block = beta.reshape(p,d,2)
    normalizer = np.linalg.norm(y)**2/len(y)
    penalty = np.linalg.norm(design.whitening @ block)**2/len(y)
    parts = np.einsum("npd,pdc->npc",z.reshape(len(x),p,d),block)
    regrets = []
    for s in range(1,p+1):
        t = (s-1)/(p-1)
        H = (1-t)*np.eye(p*d)+t*game.gram
        loss = np.sum(beta*(H@beta-2*game.cross))
        observed = np.mean([np.linalg.norm(y-parts[:,ids].sum(axis=1))**2/len(y)
                            for ids in combinations(range(p),s)])
        np.testing.assert_allclose((observed-normalizer)*p/s+penalty,loss,rtol=1e-10)
        optimum = np.sum(game.cross*np.linalg.solve(H,game.cross))
        regrets.append(loss+optimum)
    assert max(regrets) <= max(regrets[0],regrets[-1])+1e-10
    # Independently solve the epigraph on a fixed retained subset via SLSQP.
    keep = np.array([0,2,4])
    ix = game.coordinates(keep)
    graph = game.profile(keep)
    value,fitted = graph.state(graph.optimum())
    gram = game.gram[np.ix_(ix,ix)]
    def regret(w,e):
        fc = w[:-1].reshape(-1,2)
        hfc = fc if e==0 else gram @ fc
        return np.sum(fc*(hfc-2*game.cross[ix]))+game.reference[e]
    constraints = [dict(type="ineq",fun=lambda w,e=e: w[-1]-regret(w,e)) for e in (0,1)]
    initial = np.r_[np.zeros(fitted.size),max(game.reference)+1.]
    opt = minimize(lambda w: w[-1],initial,constraints=constraints,method="SLSQP",
                   options=dict(ftol=1e-12,maxiter=300))
    assert opt.success
    np.testing.assert_allclose(opt.fun,value,atol=2e-8)
    np.testing.assert_allclose(max(regret(np.r_[fitted.ravel(),value],e) for e in (0,1)),value,atol=1e-9)


def test_pruning_under_correlations_and_ties():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(44,8))
        x[:,4] = x[:,2] if seed%2 else 0.
        y = rng.normal(size=(len(x),2))
        _,z = Design.fit(x,"linear" if seed%2 else "spline")
        game = Game(z,y-y.mean(axis=0),z.shape[1]//x.shape[1])
        a,ac = game.deletion_path("min")
        b,bc = game.deletion_path("min_exhaustive")
        for first,second in zip(a,b,strict=True):
            np.testing.assert_array_equal(first,second)
        assert ac["exact"]<=bc["exact"]==bc["possible"]


def test_bisection_width():
    grid = [1,2,4,8,16,29]
    for threshold in grid:
        assert bisect(grid,lambda k: k>=threshold)==threshold
    assert bisect([1],lambda k:False)==1
