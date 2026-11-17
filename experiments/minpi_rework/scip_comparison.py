"""Matched native SCIP and compiled enumeration on predeclared regional games."""
import argparse
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyscipopt
from pyscipopt import Model, quicksum
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain
from experiments.minpi_rework.forward import development_split
from minppi_mofs import exchange_minpi as ex
from minppi_mofs.additive_minpi import fit_additive, quadratic


NAMES = ("glass","fires","wall","credit_card_fraud","landsat","biodeg","leaves","madelon")
MODES = ("direct","unrestricted","width")
BACKENDS = ("enum","rlt","convex")
RUN = benchmark.ROOT / "results/minpi_rework/scip-comparison-ba095e0"
EVIDENCE = benchmark.ROOT / "thesis-paper/evidence/minpi_rework/scip-comparison"
SECONDS = 10.


def game(data):
    x,y,task,idx = data
    fit,vote = idx["fit"],idx["vote"]
    teacher = fit_additive(x[fit],y[fit],task,"linear")
    parts = teacher.parts(x[vote])-teacher.center
    baseline = teacher.intercept+teacher.center.sum(axis=0)
    target = np.eye(len(baseline))[y[vote]] if task=="classification" else y[vote,None]
    gram,single = quadratic(parts,target-baseline)
    p=len(single)
    width=(p+1)//2
    band=min(p,28)
    left=min(max(width-band//2,0),p-band)
    order=np.lexsort((np.arange(p),-np.diag(gram)))
    keep,free=order[:left],order[left:left+band]
    return (gram[np.ix_(free,free)], single[free]-2*gram[np.ix_(free,keep)].sum(axis=1),
            width-left, p, width, keep, free, ex.value(gram,single,keep))


def warm_start(g,d,k):
    """A strict one-swap local maximizer respects every valid dominance edge."""
    chosen=np.arange(len(d))<k
    mask=g.copy()
    np.fill_diagonal(mask,0.)
    floor=128*len(d)*np.finfo(float).eps*max(np.abs(d).max(), np.abs(g).max(),1e-100)
    while True:
        ins,outs=np.flatnonzero(~chosen),np.flatnonzero(chosen)
        score=d-2*mask[:,chosen].sum(axis=1)
        diff=score[ins,None]-score[None,outs]+2*mask[np.ix_(ins,outs)]
        i,j=np.unravel_index(np.argmax(diff), diff.shape)
        if diff[i,j]<=floor:
            return chosen
        chosen[ins[i]],chosen[outs[j]]=True,False


def native(g,d,k,edges,initial,scale,backend):
    """Native optimality statements use solver tolerances; independently check objective."""
    model=Model("fixed_width")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads",1)
    model.setRealParam("limits/time",SECONDS)
    p=len(d)
    x=[model.addVar(vtype="B") for _ in range(p)]
    model.addCons(quicksum(x)==k)
    if edges is not None:
        for i,j in zip(*np.nonzero(edges)):
            model.addCons(x[i]>=x[j])
    z=initial.astype(float)
    values=list(zip(x,z))
    factor_error=0.
    if backend=="rlt":
        products={}
        for i in range(p):
            for j in range(i):
                w=products[i,j]=model.addVar(lb=0.,ub=1.)
                model.addCons(w<=x[i])
                model.addCons(w<=x[j])
                model.addCons(w>=x[i]+x[j]-1)
                values.append((w,z[i]*z[j]))
        for i in range(p):
            pairs=(products[max(i,j),min(i,j)] for j in range(p) if j!=i)
            model.addCons(quicksum(pairs)==(k-1)*x[i])
        obj=quicksum(float(d[i]/scale)*x[i] for i in range(p))
        obj-=quicksum(float(2*g[i,j]/scale)*w for (i,j),w in products.items())
    else:
        h=g/scale
        eig,v=np.linalg.eigh(h)
        assert min(eig)>-1e-10*max(1.,np.max(abs(eig)))
        root=np.sqrt(np.maximum(eig,0))[:,None]*v.T
        factor_error=float(np.max(abs(root.T@root-h)))
        aux=[]
        for r in root:
            ordered=np.sort(r)
            y=model.addVar(lb=float(ordered[:k].sum()-1e-10),ub=float(ordered[-k:].sum()+1e-10))
            model.addCons(y==quicksum(float(i)*j for i,j in zip(r,x)))
            values.append((y,float(r@z)))
            aux.append(y)
        t=model.addVar(lb=0.)
        model.addCons(quicksum(y*y for y in aux)<=t)
        values.append((t,float(np.sum((root@z)**2))))
        obj=quicksum(float((d[i]+g[i,i])/scale)*x[i] for i in range(p))-t
    model.setObjective(obj,"maximize")
    sol=model.createSol()
    for variable,val in values:
        model.setSolVal(sol,variable,float(val))
    accepted=model.addSol(sol)
    if not accepted:
        raise ArithmeticError("Native solver declined the common starting support")
    t=perf_counter()
    model.optimize()
    opt_wall=perf_counter()-t
    status=str(model.getStatus())
    solution=model.getBestSol()
    cols=np.array([i for i in range(p) if model.getSolVal(solution,x[i])>.5],dtype=int)
    assert len(cols)==k
    return cols,dict(status=status,gap=float(model.getGap()),nodes=int(model.getNNodes()),
        opt_s=float(model.getSolvingTime()),opt_wall_s=opt_wall,
        dual=float(model.getDualbound()*scale), primal=float(model.getPrimalbound()*scale),
        factor_error=factor_error, incumbent_accepted=accepted)


def one(data,mode,backend):
    begin=perf_counter()
    g,d,k,p,width,fixed,band,base=game(data)
    ready=perf_counter()
    scale=max(np.abs(g).max(),np.abs(d).max(),1e-100)
    if backend!="enum":
        initial=warm_start(g,d,k)
    t=perf_counter()
    s=(ex.Screen(np.empty(0,int),np.arange(len(d)),None,0)
       if mode=="direct" else ex.screen(g,d,k,mode))
    screen_s=perf_counter()-t
    f,kept=s.free,s.keep
    remain=k-len(kept)
    fixed_value=ex.value(g,d,kept)
    if backend!="enum":
        assert initial[kept].all() and initial[f].sum()==remain
        if s.order is not None and len(f):
            i,j=np.nonzero(s.order)
            assert (initial[f[i]]>=initial[f[j]]).all()
    residual=(g[np.ix_(f,f)],d[f]-2*g[np.ix_(f,kept)].sum(axis=1))
    defaults=dict(status="optimal",gap=0.,nodes=0,opt_s=0.,opt_wall_s=0.,factor_error=0.,
                  incumbent_accepted=None)
    if not 0<remain<len(f):
        best=np.arange(len(f)) if remain==len(f) else np.empty(0,int)
        info=defaults
    elif backend=="enum":
        t=perf_counter()
        best=ex.enumerate_best(*residual,remain,s.order)
        spent=perf_counter()-t
        info={**defaults,"opt_s":spent,"opt_wall_s":spent}
    else:
        best,info=native(*residual,remain,s.order,initial[f],scale,backend)
    local=np.sort(np.r_[kept,f[best]])
    objective=ex.value(g,d,local)
    upper=base+fixed_value+info.pop("dual") if "dual" in info else base+objective
    lower=base+fixed_value+info.pop("primal") if "primal" in info else base+objective
    columns=np.sort(np.r_[fixed,band[local]])
    total=perf_counter()-begin
    return dict(**info,mode=mode,backend=backend,p=p,width=width,band=len(d),local_width=k,
        free=len(f),fixed_in=len(kept),edges=int(s.order.sum()) if s.order is not None else 0,
        candidate_sets=math.comb(len(f),remain),objective=base+objective,primal=lower,upper=upper,
        scale=scale,screen_s=screen_s,teacher_s=ready-begin,search_s=total-(ready-begin),
        total_s=total,support=" ".join(map(str,columns)))


def prepare(name,jobs):
    if (name,42) in jobs:
        return legacy_gain.read_split(*jobs[name,42],42)
    return development_split(name,42)


def warm():
    g,d=np.eye(4),np.arange(4,dtype=float)
    ex.screen(g,d,2,"width")
    ex.screen(g,d,2,"unrestricted")
    ex.enumerate_best(g,d,2)
    e=np.zeros((4,4),bool)
    e[1,0]=True
    ex.enumerate_best(g,d,2,e)


def signature():
    from minppi_mofs import additive_minpi,exchange_bnb
    paths=(Path(ex.__file__),Path(additive_minpi.__file__),Path(exchange_bnb.__file__))
    model=Model()
    return dict(expected_core_commit="ba095e0",
        source_sha256={p.name:sha256(p.read_text(encoding="utf8").encode()).hexdigest() for p in paths},
        pyscipopt=pyscipopt.__version__,native_scip=".".join(str(m()) for m in
            (model.getMajorVersion, model.getMinorVersion, model.getTechVersion)),
        native_presolve_rounds=model.getParam("presolving/maxrounds"),native_feastol=model.getParam("numerics/feastol"),
        native_clocktype=model.getParam("timing/clocktype"),logical_processor=8,seconds=SECONDS)


def run(names, repeats):
    RUN.mkdir(parents=True,exist_ok=True)
    jobs=legacy_gain.jobs()
    # JIT startup is reported separately; all measured methods use warm libraries.
    t=perf_counter()
    warm()
    warm_s=perf_counter()-t
    stamp=signature()
    for name in names:
        data=prepare(name,jobs)
        for rep in repeats:
            out=RUN/f"{name}-{rep}.json"
            if out.exists():
                continue
            arms=[(m,b) for m in MODES for b in BACKENDS]
            rng=np.random.default_rng(7400+73*rep+NAMES.index(name))
            records=[]
            for idx in rng.permutation(len(arms)):
                m,b=arms[idx]
                result=one(data,m,b)
                records.append(dict(dataset=name,repeat=rep,seed=42,**result))
                print(f"{name} {rep} {b}/{m} {result['status']} {result['search_s']:.3f}s",flush=True)
            truth=next(r["objective"] for r in records if r["backend"]=="enum" and r["mode"]=="direct")
            for r in records:
                r["oracle_gap"]=truth-r["objective"]
                tol=2e-6*max(r["scale"],abs(truth),1e-10)
                assert r["oracle_gap"]>=-tol and r["upper"]>=truth-tol, (name,r,truth)
                if r["status"]=="optimal":
                    assert abs(r["oracle_gap"])<=tol,(name,r,truth)
            out.write_text(json.dumps(dict(jit_warm_s=warm_s,signature=stamp,rows=records)),encoding="utf-8")


def export():
    data=pd.DataFrame([row for n in NAMES for rep in range(3)
        for row in json.loads((RUN/f"{n}-{rep}.json").read_bytes())["rows"]])
    assert len(data)==216 and not data.duplicated(["dataset","repeat","mode","backend"]).any()
    EVIDENCE.mkdir(exist_ok=True,parents=True)
    stamps=[json.loads((RUN/f"{n}-{rep}.json").read_bytes())["signature"] for n in NAMES for rep in range(3)]
    assert all(stamp==stamps[0] for stamp in stamps)
    (EVIDENCE/"versions.json").write_text(json.dumps(stamps[0],indent=2),encoding="utf8")
    data.to_csv(EVIDENCE/"runs.csv",index=False)
    agg=data.assign(certified=data.status=="optimal").groupby(["dataset","backend","mode"]).agg(
        certified=("certified","sum"),search_s=("search_s","median"),total_s=("total_s","median"),
        opt_s=("opt_s","median"),opt_wall_s=("opt_wall_s","median"),
        nodes=("nodes","median"),gap=("gap","max"),oracle_gap=("oracle_gap","max"))
    agg.to_csv(EVIDENCE/"by-source.csv")
    print(agg.groupby(["backend","mode"])[["certified","search_s","total_s"]].sum().round(4))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names",nargs="+",choices=NAMES,default=NAMES)
    p.add_argument("--repeats",nargs="+",type=int,choices=range(3),default=range(3))
    p.add_argument("--export",action="store_true")
    args=p.parse_args()
    if os.name=="nt" and not args.export:
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<8)):
            raise ctypes.WinError()
    with threadpool_limits(1):
        (export() if args.export else run(args.names,args.repeats))
