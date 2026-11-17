"""Voting-sample optimal supports at every width on tractable feature lattices."""
from time import perf_counter
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy import sparse
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import quota_coalition_rf as rf, predictive_guard as guard
from experiments.minpi_rework import validation_predictive as fresh
from experiments.minpi_rework.predictive_mask_study import centered
from experiments.minpi_rework.predictive_union_control import Terms
from minppi_mofs.availability_minpi import probabilities
from minppi_mofs.predictive_quota import PredictiveGame


OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-lattice"


def optimal(terms):
    """Subset zeta transforms yield the globally least worst-sized risk at each width."""
    p=terms.p
    if p>20:
        raise ValueError("Exact lattice backend is limited to 20 columns")
    inc=terms.incidence
    masks=np.asarray(inc@(1<<np.arange(p,dtype=np.int64))).ravel()
    degrees=np.diff(inc.indptr)
    d=int(degrees.max(initial=0))
    size=1<<p
    positive=np.arange(1,p+1)
    by_size=probabilities(p,d)[1:]*p/positive[:,None]
    # For each union size, sum all its weights contained in each support mask.
    total=np.bincount(degrees,weights=terms.mass,minlength=d+1)
    kept=np.zeros((d+1,size))
    np.add.at(kept,(degrees,masks),terms.mass)
    for bit in range(p):
        width=1<<bit
        blocks=kept.reshape(d+1,-1,2*width)
        blocks[:,:,width:]+=blocks[:,:,:width]
    costs=np.empty(size)
    for lo in range(0,size,4096):
        hi=min(size,lo+4096)
        costs[lo:hi]=(by_size@(total[:,None]-kept[:,lo:hi])).max(axis=0)
    ids=np.arange(size,dtype=np.uint32)
    lengths=np.bitwise_count(ids)
    chosen=[]
    minima=[]
    for k in range(p+1):
        batch=np.flatnonzero(lengths==k)
        small=costs[batch].min()
        mask=batch[np.flatnonzero(costs[batch]<=small+terms.tolerance)[0]]
        chosen.append(np.flatnonzero((mask>>np.arange(p))&1))
        minima.append(small)
    return chosen,np.array(minima),costs


def compare(source,name,seed):
    x,y,task,idx=source.split(name,seed)
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    v=model.game(x[idx["vote"]])
    z=centered(model,y[idx["vote"]],task)
    tick=perf_counter()
    terms=Terms.build(v.a,v.incidence,z)
    supports,values,costs=optimal(terms)
    elapsed=perf_counter()-tick
    vg=PredictiveGame.from_paths(v.a,v.incidence,z)
    greedy,_=vg.ranked_path()
    votes=vg.path_curves(greedy)[:,1:]*(v.p/np.arange(1,v.p+1))
    gap=(votes.max(axis=1)[::-1]-values)/max(vg.null,1e-100)
    if gap.min() < -1e-8:
        raise ArithmeticError("Greedy beats global optimum")
    provider=b.ROOT/"results/minpi_rework/predictive-confirmation"/f"{name}-{seed}"
    rec=json.loads((provider/"selection.json").read_bytes())["records"]
    cal=model.game(x[idx["width"]])
    cg=PredictiveGame.from_paths(cal.a,cal.incidence,centered(model,y[idx["width"]],task))
    calibrated=[]
    for k,cols in enumerate(supports):
        mask=np.zeros(v.p,bool)
        mask[cols]=True
        loss=(cg.conditional_risk(mask)[1:]*(v.p/np.arange(1,v.p+1))).max()
        assert abs(costs[int(np.sum(1<<cols))]-values[k])<=terms.tolerance+1e-7*max(1,abs(values[k]))
        calibrated.append(loss/max(cg.null,1e-100))
    scale_train=max(float(y[idx["fit"]].std()),1e-12)
    yval=y[idx["width"]]
    full_scores=b.metrics(yval,model.native(x[idx["width"]]),task,scale_train)
    for k,cols in enumerate(supports):
        if calibrated[k]<=.02+1e-12 and b.eligible(b.metrics(yval,model.predict(x[np.ix_(idx["width"],cols)],cols),
                                     task,scale_train),full_scores,task):
            break
    mdi=rec["mdi_min"]
    lattice=dict(k=int(k),cal_worst=calibrated[k],support=supports[k].tolist())
    options={"lattice":lattice,"mdi":mdi}
    best=min(options,key=lambda m:(options[m]["k"],options[m]["cal_worst"],m))
    candidate=options[best]
    arms={"lattice":lattice,"mdi_plus_lattice":candidate,"mdi_plus_greedy":rec["mdi_plus_min_min"],"mdi":mdi}
    truth=source._partition(name,seed)[1]
    ys=truth[idx["test"]]
    test=x[idx["test"]]
    game=model.game(test)
    test_game=PredictiveGame.from_paths(game.a,game.incidence,centered(model,ys,task))
    scale=max(float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
    rows=[]
    for m,item in arms.items():
        cols=np.asarray(item["support"],int)
        mask=np.zeros(v.p,bool)
        mask[cols]=True
        risk=(test_game.conditional_risk(mask)[1:]*(v.p/np.arange(1,v.p+1))).max()/max(test_game.null,1e-100)
        preds=model.predict(test[:,cols],cols)
        rows.append(dict(dataset=name,seed=seed,method=m,task=task,p=v.p,k=len(cols),outer_worst=risk,
                         max_vote_gap=gap.max(),median_vote_gap=np.median(gap),widths_with_gap=int((gap>1e-8).sum()),
                         lattice_s=elapsed,support=cols.tolist(),**b.metrics(ys,preds,task,scale)))
    return rows


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--check",action="store_true")
    if cli.parse_args().check:
        rng=np.random.default_rng(521)
        for p in range(1,9):
            ids=np.arange(1,1<<p)
            inc=sparse.csr_matrix((ids[:,None]>>np.arange(p))&1)
            a=sparse.csc_matrix(rng.integers(-2,3,(6,len(ids))).astype(float))
            target=rng.integers(-4,5,6)
            tv=Terms.build(a,inc,target)
            pg=PredictiveGame.from_paths(a,inc,target)
            _,_,costs=optimal(tv)
            for i in range(1<<p):
                s=((i>>np.arange(p))&1).astype(bool)
                direct=np.max(pg.conditional_risk(s)[1:]*p/np.arange(1,p+1))
                np.testing.assert_allclose(costs[i],direct,rtol=1e-9,atol=1e-8)
        print("510 lattice costs agree with direct recounts")
        return
    OUT.mkdir(exist_ok=True,parents=True)
    rows=[]
    for name in fresh.SOURCES:
        for seed in fresh.SEEDS:
            rows+=compare(fresh,name,seed)
        pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)
        print(name,"all five sizes audited",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
