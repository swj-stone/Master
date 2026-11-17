"""Cleveland fixed-forest test acquisition under Turney's archived Ontario fee schedule."""
import argparse
from decimal import Decimal
from functools import cache
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as bench
from experiments.minpi_rework import quota_coalition_rf as forest
from experiments.minpi_rework.predictive_lattice import optimal, Terms
from experiments.minpi_rework.predictive_mask_study import centered, curve
from minppi_mofs.predictive_quota import PredictiveGame

ARCHIVE = bench.ROOT / "data/minpi_rework/field-cost-45.zip"
EXPECTED = "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc"
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
OUT = bench.ROOT / "thesis-paper/evidence/minpi_rework/field-cost"
NAMES = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal".split()
SYMBOLS = {1: {"male":1, "fem":0}, 2: {"angina":1, "abnang":2, "notang":3, "asympt":4},
           5: {"true":1, "fal":0}, 6: {"norm":0, "abn":1, "hyp":2},
           8: {"true":1, "fal":0}, 10: {"up":1, "flat":2, "down":3},
           12: {"norm":3, "fix":6, "rev":7}}
TASK = "classification"


def fetch():
    if not ARCHIVE.exists():
        with urlopen(URL, timeout=50) as response:
            data = response.read()
        from hashlib import sha256
        if sha256(data).hexdigest() != EXPECTED:
            raise ValueError("Cleveland source archive has changed")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(data)
    if bench.digest(ARCHIVE) != EXPECTED:
        raise ValueError("Cleveland source archive digest mismatch")


def source(p, seed, *, hide):
    fetch()
    with ZipFile(ARCHIVE) as z:
        lines = z.read("cleve.mod").decode("ascii").splitlines()
        rows = [line.split() for line in lines if line.strip() and not line.startswith("%")]
        rows = [r for r in rows if "?" not in r]
        if len(rows) != 296 or not all(len(r)==15 for r in rows):
            raise ValueError("Unexpected cleve.mod complete-case schema")
        x = np.array([[SYMBOLS[i][v] if i in SYMBOLS else float(v)
                       for i, v in enumerate(row[:13])] for row in rows], dtype=np.float64)
        if {r[13] for r in rows} != {"sick", "buff"}:
            raise ValueError("Unexpected diagnostic coding")
        y = np.array([r[13]=="sick" for r in rows], dtype=int)
        def schedule(kind):
            lines = z.read("costs/heart-disease."+kind).decode("ascii").splitlines()
            return {a.strip():v.strip().rstrip(".") for a,v in
                    (i.split(":") for i in lines if ":" in i)}
        discounts = {a:tuple(int(100*Decimal(v.strip())) for v in values.split(","))
                     for a,values in schedule("expense").items()}
        groups = schedule("group")
        prices = {a:int(100*Decimal(v)) for a,v in schedule("cost").items()}
        assert set(discounts)==set(prices)==set(NAMES)
        assert all(prices[a]==discounts[a][0] for a in NAMES)
    primary = [i for i in range(13) if NAMES[i]!="ca"]
    _, first, labels = np.unique(x[:,primary], axis=0, return_index=True, return_inverse=True)
    ids = np.arange(len(first))
    def split(g, frac, salt):
        return train_test_split(g, test_size=frac, random_state=seed+salt, stratify=y[first[g]])
    dev, test = split(ids,.20,0)
    remain, width = split(dev,.20,35)
    fit, vote = split(remain,.25,71)
    idx = {key:np.flatnonzero(np.isin(labels,g)) for key,g in
           dict(fit=fit,vote=vote,width=width,test=test).items()}
    if hide:
        y[idx["test"]] = -1
    take = primary if p==12 else list(range(13))
    return x[:,take], y, idx, [NAMES[i] for i in take], discounts, groups


def bits(mask, p):
    return np.flatnonzero((mask >> np.arange(p)) & 1)


def panel_fees(names, discounts, groups):
    shared = {}
    for name in names:
        full, incremental = discounts[name]
        if name in groups:
            group = groups[name]
            assert shared.setdefault(group, full-incremental)==full-incremental
        else:
            assert full == incremental
    ids = np.arange(1<<len(names))
    fees = np.zeros(len(ids), dtype=int)
    group_masks = {g:0 for g in shared}
    for i, name in enumerate(names):
        fees += discounts[name][1]*((ids >> i) & 1)
        if name in groups:
            group_masks[groups[name]] |= 1<<i
    for g, mask in group_masks.items():
        fees += ((ids & mask)!=0)*shared[g]
    return fees


def canonical_masks(model):
    size = 1<<model.p
    masks = np.zeros(size, dtype=np.int64)
    for required in model.sets:
        mask = sum(1<<j for j in required)
        masks[mask] = mask
    for i in range(model.p):
        w = 1<<i
        v = masks.reshape(-1, 2*w)
        v[:,w:] |= v[:,:w]
    assert np.array_equal(masks[masks], masks)
    return masks


def cfs_path(p, fees, loss):
    """Algorithm 1, using marginal group charge and the specified loss/AIC."""
    current = 0
    path = [current]
    while current != (1<<p)-1:
        choices = [(j,current|(1<<j)) for j in range(p) if not current & (1<<j)]
        baseline = loss(current)
        gains = [(baseline-loss(new))/(fees[new]-fees[current]) for _,new in choices]
        winner = int(np.argmax(gains))
        if gains[winner] <= 0:
            break
        current = choices[winner][1]
        path.append(current)
    return list(dict.fromkeys([*path,(1<<p)-1]))


def deletion_path(order):
    mask = (1<<len(order))-1
    path = [mask]
    for j in order:
        mask ^= 1<<int(j)
        path.append(mask)
    return path


@njit(cache=True)
def tests_traversed(x, keep, feature, threshold, left, right):
    queries = np.zeros(len(x), np.int64)
    for row in range(len(x)):
        for tree in range(len(feature)):
            node = 0
            while left[tree,node] >= 0:
                j = feature[tree,node]
                if not (keep & (1<<j)):
                    break
                queries[row] |= 1<<j
                node = left[tree,node] if x[row,j] <= threshold[tree,node] else right[tree,node]
    return queries


def collect_fit(p, seed):
    x,y,idx,names,discounts,groups = source(p,seed,hide=True)
    assert (y[idx["test"]]==-1).all()
    model = forest.fit(x[idx["fit"]],y[idx["fit"]],TASK,seed)
    fees = panel_fees(names, discounts, groups)
    canon = canonical_masks(model)
    full = len(fees)-1
    vote = model.game(x[idx["vote"]])
    game = PredictiveGame.from_paths(vote.a,vote.incidence,centered(model,y[idx["vote"]],TASK))
    paths = {rank:deletion_path(game.ranked_path(rank)[0]) for rank in ("min","pooled")}
    paths["mdi"] = deletion_path(np.argsort(model.model.feature_importances_,kind="stable"))
    train = np.r_[idx["fit"],idx["vote"]]
    xfit = (x[train]-x[train].mean(axis=0))/np.maximum(x[train].std(axis=0),1e-12)
    @cache
    def aic(mask):
        cols = bits(mask,p)
        if len(cols):
            fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2500, tol=1e-9)
            fit.fit(xfit[:,cols],y[train])
            logits = fit.decision_function(xfit[:,cols])
        else:
            prior = y[train].mean()
            logits = np.full(len(train), np.log(prior/(1-prior)))
        return 2*(len(cols)+1+np.sum(np.logaddexp(0, logits)-y[train]*logits))
    @cache
    def brier(mask):
        cols = bits(mask,p)
        pred = model.predict(x[np.ix_(idx["vote"],cols)],cols)
        return np.square(np.eye(2)[y[idx["vote"]]]-pred).sum()
    paths["cfs_aic"] = cfs_path(p, fees, aic)
    paths["cfs_brier"] = cfs_path(p, fees, brier)
    calibration = model.game(x[idx["width"]])
    terms = Terms.build(calibration.a, calibration.incidence, centered(model,y[idx["width"]],TASK))
    _,_,worst = optimal(terms)
    risk = worst/max(terms.null,1e-100)
    @cache
    def clean(mask):
        cols = bits(mask,p)
        pred = model.predict(x[np.ix_(idx["width"],cols)],cols)
        return bench.metrics(y[idx["width"]],pred,TASK,1.)
    full_performance = clean(full)
    def choose(available, policy):
        allowed = [int(m) for m in available if worst[m]<=.02*terms.null+1e-12*max(1.,terms.null)]
        key = (lambda m:(m.bit_count(),risk[m],m)) if policy=="width" else (
               lambda m:(fees[canon[m]],risk[m],int(canon[m]).bit_count(),m))
        return next(m for m in sorted(allowed,key=key) if bench.eligible(clean(m),full_performance,TASK))
    arms = {"full":full}
    for policy in ("width","fee"):
        selected = {name:choose(path,policy) for name,path in paths.items()}
        arms.update({name+"_"+policy:v for name,v in selected.items()})
        for name in ("min","pooled","cfs_aic","cfs_brier"):
            m0,m1 = selected["mdi"],selected[name]
            if policy=="width":
                winner = min((m0,m1),key=lambda m:(m.bit_count(),risk[m]))
            else:
                winner = min((m0,m1),key=lambda m:(fees[canon[m]],risk[m],int(canon[m]).bit_count()))
            arms["mdi_plus_"+name+"_"+policy] = winner
    arms["all_subsets_fee"] = choose(range(full+1),"fee")
    assert all(fees[canon[arms["all_subsets_fee"]]]<=fees[canon[m]] for m in arms.values())
    metadata = {}
    predictions = {}
    for name, mask in arms.items():
        physical = int(canon[mask])
        cols = bits(physical,p)
        test_x = model._array(x[idx["test"]],np.arange(p))
        pred = model.predict(test_x[:,cols],cols)
        raw_cols = bits(mask,p)
        np.testing.assert_array_equal(pred, model.predict(test_x[:,raw_cols],raw_cols))
        actual = tests_traversed(test_x,physical,model.feature,model.threshold,model.left,model.right)
        assert ((actual&~physical)==0).all()
        metadata[name] = dict(nominal_mask=int(mask),mask=physical,width=len(cols),fee=fees[physical]/100,
                              avg_traversed_fee=fees[actual].mean()/100,guard=float(risk[mask]))
        predictions[name] = pred
    return dict(p=p,seed=seed,features=names,idx={k:v.tolist() for k,v in idx.items()},arms=metadata), predictions


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    entries = {}
    path = OUT/"forecast-receipts.zip"
    with ZipFile(path,"w",ZIP_DEFLATED) as archive:
        for p in (12,13):
            for seed in range(42,47):
                info, pred = collect_fit(p,seed)
                key = f"p{p}-{seed}"
                archive.writestr(key+".json",json.dumps(info))
                buffer = BytesIO()
                np.savez_compressed(buffer,**pred)
                archive.writestr(key+".npz",buffer.getvalue())
                entries[key] = len(pred)
                print(key,len(pred),"forecasts",flush=True)
    code = Path(__file__)
    lock = dict(archive=bench.digest(path),source=EXPECTED,selector=bench.digest(code),
                protocol=bench.digest(code.with_name("field_cost_protocol.md")),jobs=entries)
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2),encoding="utf8")


def score():
    lock = json.loads((OUT/"lock.json").read_bytes())
    path = OUT/"forecast-receipts.zip"
    assert len(lock["jobs"])==10 and bench.digest(path)==lock["archive"]
    rows = []
    with ZipFile(path) as archive:
        for key,count in lock["jobs"].items():
            info = json.loads(archive.read(key+".json"))
            p,seed = info["p"],info["seed"]
            x,y,idx,_,_,_ = source(p,seed,hide=False)
            assert info["idx"]=={k:v.tolist() for k,v in idx.items()}
            model = forest.fit(x[idx["fit"]],y[idx["fit"]],TASK,seed)
            yy = y[idx["test"]]
            raw = model.game(x[idx["test"]])
            game = PredictiveGame.from_paths(raw.a,raw.incidence,centered(model,yy,TASK))
            with np.load(BytesIO(archive.read(key+".npz"))) as f:
                assert len(f.files)==count
                for arm, rec in info["arms"].items():
                    cols = bits(rec["mask"],p)
                    np.testing.assert_array_equal(model.predict(x[np.ix_(idx["test"],cols)],cols),f[arm])
                    worst = curve(game,cols).max()
                    rows.append(dict(p=p,seed=seed,arm=arm,**rec,held_worst=worst,
                        exceeds_quota=worst>.02+1e-12,**bench.metrics(yy,f[arm],TASK,1.)))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT/"scores.csv",index=False)
    fields = ["width","fee","avg_traversed_fee","held_worst","exceeds_quota","auc","accuracy"]
    scores.groupby(["p","arm"])[fields].mean().to_csv(OUT/"means.csv")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["fetch","collect","score"])
    with threadpool_limits(1):
        {"fetch":fetch,"collect":collect,"score":score}[parser.parse_args().action]()
