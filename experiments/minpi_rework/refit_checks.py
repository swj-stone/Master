"""Audit paired thresholds, the raw-column refit and masked outer-label invariance."""
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework.refit_delivery import paired_guard,refit_select,game_stats
from minppi_mofs.additive_minpi import fit_additive,quadratic
from minppi_mofs.exchange_minpi import integer_game


def main():
    rng=np.random.default_rng(193)
    x=rng.normal(size=(160,9)); x[::11,1]=np.nan
    y=(x[:,0]>.2).astype(int)
    idx=dict(fit=np.arange(80),vote=np.arange(80,112),width=np.arange(112,136),test=np.arange(136,160))
    pool=np.arange(112)
    for target,task in [(y,"classification"),(x[:,0]+.1*rng.normal(size=len(x)),"regression")]:
        model=fit_additive(x[pool],target[pool],task,"spline")
        parts=model.parts(x[pool])-model.center
        g,d=quadratic(parts,parts.sum(1))
        eg,ed,s=integer_game(g,d)
        energy,ag,ad,u=game_stats(model,x,pool,True)
        np.testing.assert_allclose(energy,g.diagonal(),atol=1e-13)
        np.testing.assert_array_equal(eg,ag); np.testing.assert_array_equal(ed,ad)
        np.testing.assert_allclose(s,u)
        rec,pred=refit_select(x,target,task,idx,42,"width")
        for policy,key in (("adaptive","support"),("fixed","fixed")):
            cols=rec[key]; raw=x[:,cols]
            narrow=fit_additive(raw[pool],target[pool],task,"spline")
            expected=narrow.intercept+narrow.parts(raw[idx["test"]]).sum(axis=1)
            if task=="regression": expected=expected.ravel()
            np.testing.assert_allclose(pred[policy],expected,atol=1e-12)
        unused=np.setdiff1d(np.arange(x.shape[1]),rec["support"])
        disturbed=x.copy(); disturbed[np.ix_(idx["test"],unused)]=1e6
        masked=target.copy(); masked[idx["test"]]=-1
        replay,output=refit_select(disturbed,masked,task,idx,42,"width")
        assert rec["support"]==replay["support"]
        np.testing.assert_array_equal(pred["adaptive"],output["adaptive"])
    yy=np.arange(200)%2
    anchor=np.eye(2)[yy]
    candidate=anchor.copy(); candidate[:1]=candidate[:1,::-1]
    ref=b.metrics(yy,anchor,"classification",1.)
    accepted,c=paired_guard(yy,anchor,ref,candidate,"classification",1.)
    assert accepted and abs(c["acc_loss"]-.005)<1e-12 and abs(c["acc_se"]-.005)<1e-12
    candidate[:2]=anchor[:2,::-1]
    assert not paired_guard(yy,anchor,ref,candidate,"classification",1.)[0]
    print("Raw-column sklearn ridge refits, packed game, paired guard, and masked-label invariance passed")


if __name__=="__main__":
    with threadpool_limits(1): main()
