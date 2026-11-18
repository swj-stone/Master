"""Frozen independent confirmation: redundant signal measurements versus diffuse signals."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.feature_game_eval import panels as benchmark

ARMS = ("minpi","full","F","random","marginal","ridge_weights","representatives")
SHAPES = ((2048,64), (4096,256))
REPEATS = len(ARMS)
MASTER_SEED = 2026111807


def panel(mechanism, task, shape, replicate):
    n, p = SHAPES[shape]
    rng = np.random.default_rng([MASTER_SEED,shape,replicate,mech_id(mechanism),int(task=="classification")])
    # Draw each complete data-generating instance, including its own coefficients.
    total = n + 1024
    X = rng.standard_normal((total,p))
    groups = np.full(p,-1,dtype=int)
    if mechanism == "proxy":
        latent = rng.standard_normal((total,8))
        beta = rng.uniform(.7,1.3,8) * rng.choice([-1,1],8)
        beta /= np.linalg.norm(beta)
        for j in range(32):
            groups[j] = j//4
            X[:,j] = latent[:,j//4] + .15*X[:,j]
        response = latent @ beta
    else:
        response = X @ (rng.choice([-1,1],p)/np.sqrt(p))
    response += .35*rng.standard_normal(total)
    if task=="classification":
        response = (response>0).astype(int)
    order = rng.permutation(p)
    X = np.ascontiguousarray(X[:,order])
    return SimpleNamespace(name=mechanism, task=task, seed=64063+replicate, X=X[:n],y=response[:n],
                           Xt=X[n:],truth=response[n:], group_ids=groups[order])


def mech_id(mechanism):
    return {"proxy":0,"diffuse":1}[mechanism]


def select_control(panel, arm, width):
    X,y = panel.X,panel.y
    if arm=="ridge_weights":
        Z = StandardScaler().fit_transform(X)
        model = (RidgeClassifier if panel.task=="classification" else Ridge)(alpha=1.,solver="cholesky").fit(Z,y)
        score = np.linalg.norm(np.atleast_2d(model.coef_),axis=0)
        return np.sort(np.argsort(-score,kind="stable")[:width])
    if arm=="representatives":
        # Supervised relevance from all training rows; correlations use a fixed sketch.
        score = (f_classif if panel.task=="classification" else f_regression)(X,y)[0]
        order = np.argsort(-score,kind="stable")
        rows = np.random.default_rng(panel.seed).choice(len(X), min(len(X),256),replace=False)
        vectors = X[rows].copy()
        vectors -= vectors.mean(axis=0)
        vectors /= np.maximum(np.linalg.norm(vectors,axis=0),np.finfo(float).tiny)
        vectors = np.ascontiguousarray(vectors.T)
        blocked = np.zeros(X.shape[1],dtype=bool)
        support = []
        for j in order:
            if not blocked[j]:
                support.append(j)
                if len(support)==width:
                    break
                blocked |= np.abs(vectors @ vectors[j]) >= .90
                blocked[j] = True
        if len(support)<width:
            support.extend(order[~np.isin(order,support)][:width-len(support)])
        return np.sort(support)
    return benchmark.select(arm,panel,width)


def execute(panel, arm, width):
    start=perf_counter_ns()
    cols=select_control(panel,arm,width)
    selected=perf_counter_ns()
    pred,_=benchmark.fit(panel,cols)
    end=perf_counter_ns()
    return cols,pred,(selected-start)/1e6,(end-start)/1e6


def run(out,pass_id,reference):
    parameters = dict(master_seed=MASTER_SEED, repeats=REPEATS, shapes=SHAPES,
                      methods=ARMS,tasks=["classification","regression"],replicates=list(range(1,9)),
                      mechanism=["proxy","diffuse"], test_rows=1024,
                      solver="StandardScaler/Cholesky, alpha=1",
                      acceptance="proxy class: in each task >=12/16 cases jointly reduce, preserve full within .02,"
                                 " speed over Full in both passes and score strictly > F and random; >=5/8 at each shape."
                                 " Compare versus ridge weights, correlation representatives and marginal on all cases."
                                 " A general baseline advantage needs evidence against these controls as well.",
                      source_sha256=benchmark.protocol()["source_sha256"] |
                        {"research/feature_game_eval/confirmation.py":benchmark.checksum(Path(__file__))})
    parameters=json.loads(json.dumps(parameters))
    if reference and json.loads((reference/"protocol.json").read_text(encoding="utf-8"))!=parameters:
        raise ValueError("Confirmation protocol mismatch")
    out.mkdir(parents=True,exist_ok=False)
    (out/"protocol.json").write_text(json.dumps(parameters,indent=2),encoding="utf-8")
    raw=[]
    cases=[]
    for mech in parameters["mechanism"]:
        for task in parameters["tasks"]:
            for shape in (0,1):
                for rep in parameters["replicates"]:
                    d=panel(mech,task,shape,rep)
                    key=dict(mechanism=mech,task=task,rows=len(d.y),p=d.X.shape[1],replicate=rep)
                    case=f"{mech}-{task}-{shape}-{rep}.npz"
                    arrays=dict(truth=d.truth,group_ids=d.group_ids,input_hashes=np.asarray(
                        [benchmark.digest(x) for x in (d.X,d.y,d.Xt,d.truth)]))
                    width=None
                    for arm in ARMS:
                        cols,pred,_,_=execute(d,arm,width)
                        if arm=="minpi":
                            width=len(cols)
                        assert len(np.unique(cols))==len(cols)==(d.X.shape[1] if arm=="full" else width)
                        arrays[arm+"_cols"],arrays[arm+"_pred"]=cols,pred
                        score,_,_=benchmark.deployment.score_predictions(mech,d.truth,pred,None,task,None)
                        groups=np.unique(d.group_ids[cols])
                        cases.append(dict(**key,method=arm,width=len(cols),score=score,
                                          latent_groups=int((groups>=0).sum()) if mech=="proxy" else None))
                    if reference:
                        with np.load(reference/case) as old:
                            for k,v in arrays.items():
                                np.testing.assert_array_equal(old[k],v)
                    np.savez_compressed(out/case,**arrays)
                    rotation=np.random.default_rng([MASTER_SEED,pass_id,shape,rep]).permutation(ARMS)
                    for r in range(REPEATS):
                        for pos,arm in enumerate(np.roll(rotation,r)):
                            cols,pred,tselect,ttotal=execute(d,arm,width)
                            np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
                            np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
                            raw.append(dict(**key,method=arm,repeat=r,position=pos,select_ms=tselect,total_ms=ttotal))
                    print(case,flush=True)
    pd.DataFrame(raw).to_csv(out/"trials.csv",index=False)
    pd.DataFrame(cases).to_csv(out/"cases.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--pass-id",required=True,type=int,choices=(1,2))
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    if bool(args.reference)!=(args.pass_id==2):
        parser.error("Pass 2 must reference pass 1")
    with threadpool_limits(1):
        run(args.output,args.pass_id,args.reference)
