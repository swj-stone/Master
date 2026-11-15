"""Seal two additional ANOVA exchange slates using the identical fitted-fold recipe."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review import filter_neighborhood as first
from experiments.context_review.forest_predict import deadline, new_forest, receiver, transform
from experiments.context_review.receiver_transport import SOURCE, member

ROOT = bg.ROOT
DESIGN = ROOT / "configs/context_review/neighborhood_replicates.json"
DEST = ROOT / "thesis-paper/evidence/neighborhood_replicates"
RUN = ROOT / "results/context_review/neighborhood-replicates"


def draw(order,width,seed,source,fold):
    anchor = order[:width]
    if width<2:
        return [tuple(sorted(anchor))]
    keep = width-max(1,width//4)
    prefix = anchor[:keep]
    remaining = anchor[keep:]
    count = keep*len(remaining)
    rng = np.random.default_rng(np.random.SeedSequence([seed,source,fold]))
    subset = range(count) if count<=7 else rng.choice(count,7,replace=False)
    alternatives = {tuple(sorted(prefix))}
    for offset in subset:
        a,b = divmod(int(offset),len(remaining))
        alternatives.add(tuple(sorted((set(prefix)-{prefix[a]}) | {remaining[b]})))
    ranks = {c:i for i,c in enumerate(order)}
    return [tuple(sorted(anchor)),*sorted(alternatives,key=lambda s:(sum(ranks[c] for c in s),s))]


def scores(x,y,train,proposal,outer,guard,rec,union,first_arrays,first_points):
    old = {tuple(s):i for i,s in enumerate(rec["supports"])}
    additional = [s for s in union if s not in old]
    names = rec["metrics"]
    nold, tensors, points, gprobs, oprobs = len(old),{},{},{},{}
    for s,i in old.items():
        tensors[s] = first_arrays["tensor"][i]
        if i:
            points[s] = first_points[i-1]
        gprobs[s] = first_arrays[f"g{i}"]
        oprobs[s] = first_arrays[f"o{i}"]
    if additional:
        seed = 20421+rec["fold"]
        rare = "ap" in names
        xt,xp = transform(x,train,x[proposal])
        voter = new_forest(seed).fit(xt,y[train])
        anchor = rec["supports"][0]
        family = primary.separating_contexts(rec["order"][len(anchor):])
        ballot = primary.build_ballot(voter,xt,xp,y[proposal],seed,anchor,rare)
        point = primary.build_ballot(voter,xt,xp,y[proposal],seed,anchor,rare,point=True)
        for support in [tuple(anchor),*additional]:
            tensor = np.array([[[p[name] for name in names] for p in ballot.score(frozenset(support)|e)]
                               for e in family])
            if support==tuple(anchor):
                np.testing.assert_array_equal(tensor,tensors[support])
            tensors[support] = tensor
            if support != tuple(anchor):
                points[support] = primary.rating(point,support,[frozenset()],"point")
        inner = np.sort(np.r_[train,proposal])
        g,_,_ = receiver(x,y,inner,x[guard],additional,seed)
        o,_,_ = receiver(x,y,np.arange(len(y)),outer,additional,seed)
        gprobs.update(zip(additional,g,strict=True))
        oprobs.update(zip(additional,o,strict=True))
    return (np.array([tensors[s] for s in union]),np.array([points[s] for s in union[1:]]),
            [gprobs[s] for s in union],[oprobs[s] for s in union],len(additional),nold)


def nominate(tensor,points,names,slate):
    if len(slate)==1:
        return {m:0 for m in ("vote","point","slack")}
    target = tensor[slate]
    allowance = abs(target[0])*np.array([.05 if n=="ap" else .03 for n in names])
    diff = target[1:]-target[0]
    votes = (-diff<=allowance+1e-12).all(axis=3).sum(axis=2).min(axis=1)
    margins = ((diff+allowance+1e-12)/np.maximum(allowance,1e-12)).min(axis=3)
    full = np.sort(margins,axis=2)[:,:,1].min(axis=1)
    np.testing.assert_array_equal(votes>=3,full>=0)
    zs = {"vote":votes,"slack":full,"point":points[np.array(slate[1:])-1]}
    return {arm:slate[1+int(np.argmax(z))] for arm,z in zs.items()}


def forecast(key,rec,first_arrays,points,design):
    src = ROOT/"results/context_review"/SOURCE[rec["panel"]]/key.split("/",1)[1]
    assert bg.digest(src/"inputs.npz")==rec["inputs_sha"]
    with np.load(src/"inputs.npz") as data:
        x,y,train,proposal,outer,guard = (data[k] for k in ("x","y","train","proposal","x_outer","guard"))
    slates = [draw(rec["order"],len(rec["supports"][0]),seed,rec["id"],rec["fold"]) for seed in design["seeds"]]
    assert slates[0]==list(map(tuple,rec["supports"]))
    union = list(dict.fromkeys(s for slate in slates for s in slate))
    ids = {s:i for i,s in enumerate(union)}
    slates = [[ids[s] for s in slate] for slate in slates]
    tensor,point,g,o,new,original = scores(x,y,train,proposal,outer,guard,rec,union,first_arrays,points)
    winners = [nominate(tensor,point,rec["metrics"],slate) for slate in slates]
    assert winners[0]==rec["winners"]
    out = BytesIO()
    np.savez_compressed(out,tensor=tensor,points=point,guard_y=first_arrays["guard_y"],test_row=first_arrays["test_row"],
                        **{f"g{i}":p for i,p in enumerate(g)},**{f"o{i}":p for i,p in enumerate(o)})
    record = {k:rec[k] for k in ("panel","dataset","fold","id","metrics","targets_sha","inputs_sha")}
    record.update(supports=union,slates=slates,winners=winners,original_count=original,new_unique=new,
                  member_sha=sha256(out.getbuffer()).hexdigest())
    return record,out.getvalue()


def predict():
    DEST.mkdir(exist_ok=True,parents=True)
    original = bg.read(first.DEST/"prediction-lock.json")
    old_points = bg.read(first.DEST/"point-scores.json")
    design = bg.read(DESIGN)
    assert bg.digest(first.DEST/"predictions.zip")==original["archive_sha"]
    assert bg.digest(first.DEST/"prediction-lock.json")==old_points["prediction_seal"]
    campaign = dict(design_sha=bg.digest(DESIGN),code_sha=bg.digest(Path(__file__)),
                    original_sha=old_points["prediction_seal"],seeds=design["seeds"])
    records = {}
    start = perf_counter()
    with ZipFile(first.DEST/"predictions.zip") as src, threadpool_limits(limits=1):
        for i,(key,meta) in enumerate(original["folds"].items(),1):
            folder = RUN/key
            folder.mkdir(parents=True,exist_ok=True)
            path,receipt = folder/"predictions.npz",folder/"receipt.json"
            if not receipt.exists():
                began = perf_counter()
                timer = Timer(3600, deadline,args=(key,))
                timer.daemon = True
                timer.start()
                try:
                    data = src.read(key+".npz")
                    assert sha256(data).hexdigest()==meta["member_sha"]
                    with np.load(BytesIO(data)) as first_arrays:
                        meta_new,data = forecast(key,meta,first_arrays,old_points["scores"][key],design)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(path)
                    meta_new.update(campaign=campaign, additional_seconds=perf_counter()-began)
                    bg.store(receipt,meta_new)
                finally:
                    timer.cancel()
            rec = bg.read(receipt)
            assert rec["campaign"]==campaign and rec["member_sha"]==bg.digest(path)
            records[key] = rec
            elapsed = perf_counter()-start
            print(f"[{i:02}/46] {key} new={rec['new_unique']} {rec['additional_seconds']:.1f}s "
                  f"elapsed={elapsed:.1f}s ETA={elapsed*(46-i)/i:.1f}s",flush=True)
    tmp = DEST/"predictions.tmp"
    with ZipFile(tmp,"w") as archive:
        for key in records:
            member(archive,key+".npz",(RUN/key/"predictions.npz").read_bytes())
    out = DEST/"predictions.zip"
    tmp.replace(out)
    bg.store(DEST/"prediction-lock.json",dict(**campaign,archive_sha=bg.digest(out),folds=records))


if __name__=="__main__":
    predict()
