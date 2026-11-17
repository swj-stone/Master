"""Exploratory single-sensor outage game for a deployed additive predictor.

An unavailable column contributes its training-mean fitted shape. The event may
affect any single retained column, or none. The guard and all comparators receive
the same scenarios. Previously opened data only.
"""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework import forward
from minppi_mofs.additive_minpi import fit_additive, quadratic


def outage_order(gram,single,rule,pair):
    p=len(single)
    active=np.arange(p)
    hazard=single-2*(gram.sum(axis=1)-gram.diagonal())
    dropped=[]
    while len(active)>1:
        l=r=active
        widths=np.ones(len(active),dtype=int)
        harm=hazard[active]
        if pair and len(active)>2:
            a,c=np.triu_indices(len(active),k=1)
            i,j=active[a],active[c]
            joint=hazard[i]+hazard[j]+2*gram[i,j]
            # All arms inspect p singleton and at most p pair candidates. Use
            # point damage to shortlist eligible pairs before any outage vote.
            short=np.argsort(joint,kind="stable")[:len(active)]
            l,r=np.r_[active,i[short]],np.r_[active,j[short]]
            harm=np.r_[harm,joint[short]]
            widths=np.r_[widths,np.full(len(short),2)]
        if rule=="point":
            score=harm/widths
        else:
            gaps=hazard[active]+2*gram[np.ix_(l,active)]
            pairs=l!=r
            gaps[pairs]+=2*gram[np.ix_(r[pairs],active)]
            mask=(active[None,:]==l[:,None]) | (active[None,:]==r[:,None])
            if rule=="min":
                gaps[mask]=-np.inf
                worst=np.maximum(0.,gaps.max(axis=1))
                reference=max(0.,max(hazard[active]))
            elif rule=="mean":
                gaps[mask]=0.
                worst=gaps.sum(axis=1)/(len(active)-widths+1)
                reference=hazard[active].sum()/(len(active)+1)
            else:
                raise ValueError(rule)
            score=(harm+worst-reference)/widths
        k=int(np.argmin(score))
        block=sorted(set((l[k],r[k])), key=lambda j:(hazard[j],j))
        dropped.extend(block)
        hazard+=2*gram[:,block].sum(axis=1)
        active=active[~np.isin(active,block)]
    return np.array((dropped+active.tolist())[::-1],dtype=int)


def exchange_half(gram,single,rule,rounds=2,size=None):
    """At a prescribed width, make at most two improving 1-for-1 exchanges from energy."""
    p=len(single)
    order=np.argsort(-gram.diagonal(),kind="stable")
    size=(p+1)//2 if size is None else size
    kept=order[:size].copy()
    for _ in range(rounds):
        absent=np.setdiff1d(np.arange(p),kept)
        if not len(absent): break
        removed=np.repeat(kept,len(absent))
        added=np.tile(absent,len(kept))
        hazard=single-2*gram[:,kept].sum(axis=1)
        hazard[kept]+=2*gram.diagonal()[kept]
        added_gap=hazard[added]+2*gram[removed,added]
        delta=hazard[removed]-added_gap
        if rule!="point":
            gaps=hazard[kept]+2*(gram[np.ix_(removed,kept)]-gram[np.ix_(added,kept)])
            gaps[np.arange(len(added)),np.repeat(np.arange(size),len(absent))]=added_gap
            if rule=="min":
                delta+=np.maximum(0.,gaps.max(axis=1))-max(0.,hazard[kept].max())
            elif rule=="mean":
                delta+=(gaps.sum(axis=1)-hazard[kept].sum())/(size+1)
            else: raise ValueError(rule)
        choice=int(np.argmin(delta))
        if delta[choice]>=-1e-12*(1+np.max(np.abs(hazard))): break
        kept[kept==removed[choice]]=added[choice]
    selected=np.isin(order,kept)
    return np.r_[order[selected],order[~selected]]


def support_risk(gram,single,base,cols):
    cross=gram[np.ix_(cols,cols)]
    off_diag=cross.sum(axis=1)-cross.diagonal()
    loss=base-single[cols].sum()+off_diag.sum()
    return loss+max(0.,np.max(single[cols]-2*off_diag))


def shape(train,query,x,y,task,kind):
    model=fit_additive(x[train],y[train],task,kind)
    center=model.parts(x[train]).mean(axis=0)
    return model.parts(x[query])-center,model.intercept+center.sum(axis=0)


def stats(parts,target,offset):
    centered=target-offset
    gram,single=quadratic(parts,centered)
    scale=max(np.finfo(float).eps,np.sum(centered*centered)/len(parts))
    return gram,single,scale


def risks(gram,single,loss0,order):
    """Exact loss for each prefix, intact and maximum across one outage or none."""
    loss=loss0
    harm=np.zeros(len(order))
    point,worst=[],[]
    for i,j in enumerate(order):
        other=order[:i]
        harm[:i]-=2*gram[j,other]
        harm[i]=single[j]-2*gram[j,other].sum()
        loss-=harm[i]
        point.append(loss)
        worst.append(loss+max(0.,harm[:i+1].max()))
    return np.array(point),np.array(worst)


def pilot(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    manifest=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            start=perf_counter()
            if (name,seed) in manifest:
                root,source=manifest[name,seed]
                x,y,task,idx=legacy.read_split(root,source,seed)
                outer=pd.read_csv(root/source["test_relpath"],usecols=[source["target"]])[source["target"]].to_numpy()
                with np.load(legacy.DEST/"novel-input-rows.npz") as a:
                    mask=a[f"{name}-{seed}"]
            else:
                x,y,task,idx=forward.development_split(name,seed)
                outer=y[idx["test"]].copy()
                y=y.copy()
                y[idx["test"]]=-1
                mask=np.ones(len(outer),dtype=bool)
            fit,vote,width,test=(idx[k] for k in ("fit","vote","width","test"))
            pool=np.sort(np.r_[fit,vote])
            dev=np.sort(np.r_[pool,width])
            classes=np.unique(y[fit])
            def encode(arr):
                return np.eye(len(classes))[arr] if task=="classification" else arr[:,None]
            # All arms receive identical precomputed sufficient statistics.
            prep_clock=perf_counter()
            if args.freeze:
                fixed=fit_additive(x[fit],y[fit],task,args.kind)
                center=fixed.parts(x[fit]).mean(axis=0)
                off=offset=test_offset=fixed.intercept+center.sum(axis=0)
                sp=fixed.parts(x[vote])-center
                z=fixed.parts(x[width])-center
            else:
                sp,off=shape(fit,vote,x,y,task,args.kind)
                z,offset=shape(pool,width,x,y,task,args.kind)
            train_target=sp.sum(axis=1)+off if args.teacher else encode(y[vote])
            width_target=z.sum(axis=1)+offset if args.teacher else encode(y[width])
            gram,single,_=stats(sp,train_target,off)
            g,d,base=stats(z,width_target,offset)
            # Outer outcomes enter scoring only after orders, guards and k are frozen below.
            full_pred=offset+z.sum(axis=1)
            scale=max(np.finfo(float).eps,y[pool].std())
            if task=="regression": full_pred=full_pred.ravel()
            full=b.metrics(y[width],full_pred,task,scale)
            common_time=perf_counter()-prep_clock
            decisions=[]
            for method in args.methods:
                rank_start=perf_counter()
                if method=="energy":
                    order=np.argsort(-gram.diagonal(),kind="stable")
                elif method=="full":
                    order=np.arange(len(single))
                elif method.startswith("exchange_"):
                    order=exchange_half(gram,single,method.split("_",1)[1])
                elif method.startswith("adaptive_"):
                    order=np.argsort(-gram.diagonal(),kind="stable")
                else:
                    family,rule=method.split("_")
                    order=outage_order(gram,single,rule,family=="pair")
                rank_s=perf_counter()-rank_start
                _,worst=risks(g,d,base,order)
                p=len(order)
                guard=p
                pred=np.broadcast_to(offset,(len(z),len(offset))).copy()
                if method!="full":
                    for k,j in enumerate(order[:-1],1):
                        if method.startswith("adaptive_"):
                            proposal=exchange_half(gram,single,method.split("_",1)[1],size=k)[:k]
                            bound=support_risk(g,d,base,proposal)
                        else:
                            bound=worst[k-1]
                            pred+=z[:,j]
                        # Require the candidate's worst held-out outage loss to be
                        # no larger than the full reference; no extra tolerance.
                        if bound>worst[-1]+1e-12*base:
                            continue
                        if method.startswith("adaptive_"):
                            pred=offset+z[:,proposal].sum(axis=1)
                        if task=="classification" and (pred.argmax(1)==y[width]).mean()<full["accuracy"]-.01:
                            continue
                        metrics=b.metrics(y[width],pred if task=="classification" else pred.ravel(),task,scale)
                        if b.eligible(metrics,full,task):
                            guard=k
                            break
                picked=proposal if method.startswith("adaptive_") and guard<p else order[:guard]
                selection_s=common_time+perf_counter()-rank_start
                if method.startswith("adaptive_"):
                    order=exchange_half(gram,single,method.split("_",1)[1])
                decisions.append((method,order,picked,rank_s,selection_s))
            if args.freeze:
                test_parts=fixed.parts(x[test[mask]])-center
            else:
                test_parts,test_offset=shape(dev,test[mask],x,y,task,args.kind)
            a,h,test_base=stats(test_parts,encode(outer[mask]),test_offset)
            ta,th,teacher_base=stats(test_parts,test_parts.sum(axis=1)+test_offset,test_offset)
            for method,order,picked,clock,selection_s in decisions:
                half=(len(order)+1)//2 if method!="full" else len(order)
                for kind,cols in (("guard",picked),("half",order[:half])):
                    k=len(cols)
                    intact,worst=risks(a,h,test_base,cols)
                    teacher_intact,teacher_worst=risks(ta,th,teacher_base,cols)
                    pred=test_offset+test_parts[:,cols].sum(axis=1)
                    if task=="regression": pred=pred.ravel()
                    metric=b.metrics(outer[mask],pred,task,max(1e-12,y[dev].std()))
                    rows.append(dict(dataset=name,seed=seed,task=task,method=method,choice=kind,
                                      selected=" ".join(str(j) for j in sorted(cols)),
                                      p=len(order),k=k,rank_s=clock,selection_s=selection_s,worst=worst[k-1]/test_base,
                                      intact=intact[k-1]/test_base,teacher_worst=teacher_worst[k-1]/teacher_base,
                                      teacher_intact=teacher_intact[k-1]/teacher_base,**metric))
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run",required=True)
    cli.add_argument("--names",nargs="+",required=True)
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--kind",choices=["spline","linear"],default="spline")
    cli.add_argument("--freeze",action="store_true",help="Use one fitted learner throughout post-hoc compression.")
    cli.add_argument("--teacher",action="store_true",help="Use intact full predictions for ranking/inner outage loss.")
    cli.add_argument("--methods",nargs="+",default=["single_min","single_mean","single_point","pair_min","pair_mean","pair_point","energy","full"])
    with threadpool_limits(1):
        pilot(cli.parse_args())
