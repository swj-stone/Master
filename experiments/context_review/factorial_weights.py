"""Separate row changes and reweighting on the original endpoint-pass proposals."""
import json

import numpy as np
import pandas as pd

from experiments.context_review.factorial import DESIGN, OUT, RAW, ROOT, scores, sha


def weights(y, point, classes, common, stream):
    matched = np.zeros(len(y))
    draws = np.zeros((3,len(y)))
    for label,standard_mass in zip(classes,common,strict=True):
        ids=np.flatnonzero(y==label)
        matched[ids]=standard_mass/len(ids)
        mass=point[ids].sum()
        for draw in draws:
            counts=np.bincount(stream.integers(len(ids),size=len(ids)),minlength=len(ids))
            draw[ids]=mass*counts/len(ids)
    return matched,draws


def one(frozen,task_no):
    name,seed=frozen.iloc[0][["dataset","seed"]]
    folder=RAW/f"seed-{seed}"/name
    d=json.loads((folder/"receipt.json").read_bytes())
    assert sha(folder/"predictions.npz")==d["predictions_sha256"]
    assert d["task"]!="regression"  # The ranking-endpoint-pass subset has no eligible regression task.
    ids=frozen.proposal.to_numpy(dtype=int)
    ix=np.array(d["lookup"])[np.r_[0,ids+1]]
    used,inv=np.unique(ix,return_inverse=True)
    lookup=inv.reshape(ix.shape)
    with np.load(folder/"predictions.npz") as store:
        p=store["predictions"][used]
        sets=[(store[tag+"_y"],store[tag+"_w"]) for tag in ("rank","guard")]
    classes=np.unique(sets[0][0])
    assert np.array_equal(classes,np.unique(sets[1][0]))
    common=np.mean([[wt[0,y==c].sum() for c in classes] for y,wt in sets],axis=0)
    answer=[]
    for slot,((y,orig),panels) in enumerate(zip(sets,np.split(p,[len(sets[0][0])],axis=1),strict=True)):
        tag=("rank","guard")[slot]
        singleton_and_boot=[*orig]
        families={"original":list(range(4)),"point":[0],"common_mass_point":[4]}
        for replicate in range(5):
            stream=np.random.default_rng(np.random.SeedSequence([5119,int(seed),task_no,slot,replicate]))
            matched,draws=weights(y,orig[0],classes,common,stream)
            if replicate==0:
                singleton_and_boot.append(matched)
            first=len(singleton_and_boot)
            families[f"redraw_{replicate}"]=[0,first,first+1,first+2]
            singleton_and_boot.extend(draws)
        ws=np.array(singleton_and_boot)
        np.testing.assert_allclose(ws.sum(axis=1),1,atol=1e-12)
        values=scores(panels,y,ws,d["task"],d["rare"])[lookup]
        tol=np.array([.03,.03,.05] if d["rare"] else [.03,.03])
        votes=(values[:1]-values[1:]<=tol*np.abs(values[:1])+1e-12).all(axis=-1)
        for method,columns in families.items():
            fail=votes[:,:,columns].sum(axis=-1)<=len(columns)/2
            for i,rec in enumerate(frozen.itertuples(index=False)):
                end=bool(fail[i,d["rules"]["ends"]].any())
                interior=bool(fail[i,d["novel"]].any())
                if method=="original":
                    assert end==getattr(rec,tag+"_ends") and interior==getattr(rec,tag+"_interior")
                answer.append(dict(dataset=name,seed=seed,proposal=rec.proposal,partition=tag,method=method,
                                   original_guard_ends=rec.guard_ends,ends=int(end),interior=int(interior)))
    return answer


def main():
    config=json.loads(DESIGN.read_bytes())
    spec=ROOT/"configs/context_review/factorial_weights.json"
    data=pd.read_csv(OUT/"factorial-candidates.csv")
    fixed=data[(data.novel>0)&(data.rank_ends==0)]
    assert len(fixed)==26
    pin={p.relative_to(ROOT).as_posix():sha(p) for p in (spec,OUT/"factorial-candidates.csv",ROOT/"experiments/context_review/factorial_weights.py")}
    (OUT/"factorial-weights-pins.json").write_text(json.dumps(pin,indent=2),encoding="utf8")
    rows=[]
    for (ds,_),part in fixed.groupby(["dataset","seed"]):
        rows.extend(one(part,config["datasets"].index(ds)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"factorial-weights.csv",index=False)
    group=["method","partition"]
    for name,subset in [("all",result),("both",result[result.original_guard_ends==0])]:
        totals=subset.groupby(group).agg(n=("proposal","size"),ends=("ends","sum"),interior=("interior","sum"))
        totals.to_csv(OUT/f"factorial-weights-{name}.csv")
        print(totals)


if __name__=="__main__":
    main()
