"""External-refit audit using raw PMLB tables and public library APIs."""
import argparse
from functools import lru_cache
from itertools import combinations, islice
from math import ceil, comb
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from rounds.pmlb_baselines import DATA, ROOT, RUN, auc, compact_gain, fit, read, sha

HERE = Path(__file__).resolve().parent
STUDIES = ROOT/"results/context_review"
NAMES = read(RUN/"protocol-lock.json")["design"]["datasets"]
RULES = ["point", "empty", "endpoints", "nested", "separating", "random", "pooled", "one_panel_min"]


def split_data(name):
    table = DATA/name/f"{name}.tsv.gz"
    source = read(RUN/"sources.json")["sources"][name][table.name]["sha256"]
    assert sha(table) == source
    data = pd.read_csv(table, sep="\t")
    raw = data.pop("target").to_numpy()
    x = data.to_numpy(dtype=np.float32)
    for f, (dev, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=20261115).split(x,raw)):
        seed = 20421+f
        train, rank = train_test_split(dev, test_size=.25, stratify=raw[dev], random_state=seed)
        classes, sizes = np.unique(raw[train], return_counts=True)
        if len(classes)==2 and sizes[0]<sizes[1]:
            classes = classes[::-1]
        y = pd.Categorical(raw,categories=classes).codes
        original = RUN/name/f"fold-{f}"
        archive = read(original/"selection.json")
        for field, known in (("class_order",classes),("train",train),("rank",rank),("test",test)):
            np.testing.assert_array_equal(archive[field],known)
        assert read(original/"completed.json")["selection_sha256"]==sha(original/"selection.json")
        yield f,x,y,dev,train,rank,test,archive


def scores(y,p,rare):
    result = dict(auc=auc(y,p), accuracy=float((p.argmax(axis=1)==y).mean()))
    if rare:
        result["ap"] = average_precision_score(y,p[:,1])
    return result


def fails(anchor,reduced,t):
    return any(anchor[k]-reduced[k] > (0.05 if k=="ap" else t)*abs(anchor[k])+1e-12 for k in anchor)


def panel_weights(y,seed):
    labels, sizes = np.unique(y,return_counts=True)
    quota = np.zeros_like(sizes)
    left = min(256,len(y))
    for j,idx in enumerate(np.argsort(sizes,kind="stable")):
        quota[idx] = min(left//(len(labels)-j),sizes[idx])
        left -= quota[idx]
    rng = np.random.default_rng(np.random.SeedSequence([seed,659]))
    rows = np.sort(np.concatenate([rng.choice(np.where(y==c)[0],q,replace=False) for c,q in zip(labels,quota)]))
    weights = np.zeros((4,len(rows)))
    for c,q,n in zip(labels,quota,sizes):
        ids = np.where(y[rows]==c)[0]
        weights[0,ids] = (n/len(y))/q
        for w in weights[1:]:
            w[ids] = (n/len(y))*np.bincount(rng.integers(q,size=q),minlength=q)/q
    return rows, weights


def metrics_on_panels(y,p,ws,rare):
    # Calculate weighted concordance through midrank probabilities for each class.
    correct = (np.argmax(p,axis=1)==y)
    acc = (ws@correct)/ws.sum(axis=1)
    columns = [1] if p.shape[1]==2 else range(p.shape[1])
    all_auc = []
    for c in columns:
        _, bucket = np.unique(p[:,c], return_inverse=True)
        pos = y==c
        auc_c = []
        for w in ws:
            good = np.bincount(bucket,weights=w*pos)
            bad = np.bincount(bucket,weights=w*(~pos))
            auc_c.append((good @ (bad.cumsum()-bad/2))/(good.sum()*bad.sum()))
        all_auc.append(auc_c)
    out = [dict(auc=a,accuracy=b) for a,b in zip(np.mean(all_auc,axis=0),acc)]
    if rare:
        for r,w in zip(out,ws):
            r["ap"]=average_precision_score(y,p[:,1],sample_weight=w)
    return out


def selected_cohorts(rec,root_seed):
    anchor = frozenset(rec["selected"]["gain"])
    union = set()
    for method in ("antichain","min","point"):
        current = anchor
        for proposed in rec["traces"][method]["trace"]:
            reduced = current-frozenset(proposed["block"])
            if reduced and reduced<anchor:
                union.add(tuple(sorted(reduced)))
            if proposed["accepted"]:
                current = reduced
        assert sorted(current)==rec["selected"][method]
    paths=set(Random(117034+root_seed).sample(sorted(union), k=min(12,len(union))))
    rng=Random(47113+root_seed)
    size=len(anchor)
    arbitrary=set()
    for d in sorted({1,ceil(size/4),size//2} - {0,size}):
        choices=set()
        if comb(size,d)<=4:
            choices.update(combinations(sorted(anchor),d))
        else:
            while len(choices)<4:
                choices.add(tuple(sorted(rng.sample(sorted(anchor),d))))
        arbitrary.update(tuple(sorted(anchor-set(c))) for c in choices)
    return (paths,arbitrary)


def predictions(x,y,train,test,seed):
    pre=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    a,b=pre.transform(x[train]),pre.transform(x[test])
    @lru_cache(maxsize=None)
    def estimate(cols):
        return fit(a[:,list(cols)],y[train],seed).predict_proba(b[:,list(cols)])
    return estimate


def judge(gr):
    sep,nested,random = (np.asarray(gr[k],dtype=int) for k in ("separating","nested","random"))
    strict = lambda m: bool((m.sum(axis=1) < 3).any())
    return dict(point=not sep[0,0],empty=strict(sep[:1]), endpoints=strict(sep[[0,-1]]),
                nested=strict(nested),separating=strict(sep),random=strict(random),
                pooled=int(sep.sum())<=sep.size//2,one_panel_min=not sep[:,0].all())


def ballot_reader(x,y,train,rank,seed,rare):
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    a,b=imp.transform(x[train]),imp.transform(x[rank])
    model=fit(a,y[train],seed)
    bgrows=np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    reference=np.median(a[bgrows],axis=0)
    rows,weights=panel_weights(y[rank],seed)
    b=b[rows]
    @lru_cache(maxsize=None)
    def readmask(frozen):
        cols=list(sorted(frozen))
        xx=np.tile(reference,(len(b),1)); xx[:,cols]=b[:,cols]
        return metrics_on_panels(y[rank][rows],model.predict_proba(xx),weights,rare)
    return readmask


def target(name,bundle,fresh,refit_masks):
    f,x,y,dev,train,rank,test,old=bundle
    folder=STUDIES/"refit-target"/name/f"fold-{f}"
    done=read(folder/"completed.json")
    assert done["predictions"]==sha(folder/"predictions.npz") and done["selection"]==sha(folder/"selection.json")
    design=read(folder/"selection.json")
    ref=frozenset(design["anchor"])
    assert sorted(ref)==old["selected"]["gain"]
    sets=selected_cohorts(old,f+10*NAMES.index(name))
    for label,items in zip(("path","uniform_deletions"),sets):
        assert items==set(map(tuple,design["cohorts"][label]))
    union,overlap=sets[0]|sets[1],sets[0]&sets[1]
    assert sorted(union)==list(map(tuple,design["supports"]))
    rare = len(np.unique(y))==2 and min(np.bincount(y[rank]))<=len(rank)*.05
    seed=20421+f
    estimate=predictions(x,y,dev,test,seed) if fresh else None
    masks=ballot_reader(x,y,train,rank,seed,rare) if refit_masks else None
    rows=[]; audits=dict(dataset=name,fold=f,paths=len(sets[0]),uniform=len(sets[1]),
                         overlap=len(overlap),unique=len(union),refitted=0,mask_ballots=0)
    with np.load(folder/"predictions.npz") as frozen:
        np.testing.assert_array_equal(frozen["row_id"],test); np.testing.assert_array_equal(frozen["y"],y[test])
        anchor=scores(y[test],frozen["anchor"],rare)
        if fresh:
            np.testing.assert_array_equal(estimate(tuple(sorted(ref))),frozen["anchor"]); audits["refitted"]+=1
        for i,(s,gr,archived) in enumerate(zip(design["supports"],design["ballots"],design["decisions"],strict=True)):
            for key,old_arm in (("nested","min"),("random","random_contexts"),("separating","antichain")):
                assert design["contexts"][key]==old["traces"][old_arm]["context_extensions"]
                if refit_masks:
                    for e,observed in zip(design["contexts"][key],gr[key],strict=True):
                        targets=masks(frozenset(s)|frozenset(e)); original=masks(ref|frozenset(e))
                        assert [not fails(a,b,.03) for a,b in zip(original,targets)]==observed
                        audits["mask_ballots"]+=len(observed)
            votes=judge(gr)
            assert votes==archived
            p=frozen[f"s{i}"]
            if fresh:
                np.testing.assert_array_equal(estimate(tuple(s)),p); audits["refitted"]+=1
            reduced=scores(y[test],p,rare)
            for pop in ("path","uniform_deletions"):
                if tuple(s) not in set(map(tuple,design["cohorts"][pop])):
                    continue
                for t in (.01,.03,.05):
                    rows.append(dict(dataset=name,fold=f,subset=i,cohort=pop,relative_threshold=t,
                        failure=fails(anchor,reduced,t),loss_auc=anchor["auc"]-reduced["auc"],
                        auc_only=fails({"auc":anchor["auc"]},reduced,t),
                        accuracy_only=fails({"accuracy":anchor["accuracy"]},reduced,t),
                        loss_accuracy=anchor["accuracy"]-reduced["accuracy"],**votes))
    return rows,audits


def gain(name,bundle):
    f,x,y,dev,train,rank,test,_=bundle
    folder=STUDIES/"exhaustive-gain"/name/f"fold-{f}"
    rec=read(folder/"selection.json"); done=read(folder/"completed.json")
    assert sha(folder/"predictions.npz")==done["predictions"] and sha(folder/"selection.json")==done["selection"]
    seed=20421+f
    pre=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    a,b=pre.transform(x[train]),pre.transform(x[rank])
    model=fit(a,y[train],seed); gains=model.get_booster().get_score(importance_type="gain")
    idx=sorted(range(x.shape[1]),key=lambda i:(-gains.get(f"f{i}",0),-i))
    assert idx==rec["order"]
    all_scores=[]
    for k in range(1,len(idx)+1):
        cols=sorted(idx[:k])
        model_k=model if k==len(idx) else fit(a[:,cols],y[train],seed)
        all_scores.append(auc(y[rank],model_k.predict_proba(b[:,cols])))
    np.testing.assert_allclose(all_scores,rec["rank_auc"],rtol=0,atol=1e-12)
    best=0
    for j,score in enumerate(all_scores):
        if score>all_scores[best]+1e-14:
            best=j
    cols=tuple(sorted(idx[:best+1])); assert list(cols)==rec["chosen"]
    pred=predictions(x,y,dev,test,seed)(cols)
    with np.load(folder/"predictions.npz") as preds:
        np.testing.assert_array_equal(y[test],preds["y"]); np.testing.assert_array_equal(test,preds["row_id"])
        np.testing.assert_array_equal(preds["probability"],pred)
    outcome=scores(y[test],pred,False)
    assert all(abs(v-done["row"][m])<1e-12 for m,v in outcome.items())
    return dict(dataset=name,fold=f,k=best+1,width=x.shape[1],fits=len(idx)+1,**outcome)


def guard(name,bundle):
    f,x,y,dev,train,rank,test,_=bundle
    folder=STUDIES/"refit-guard"/name/f"fold-{f}"
    counts=np.bincount(y[rank])
    if np.any(counts<2):
        assert not (folder/"completed.json").exists()
        return dict(dataset=name,fold=f,status="ineligible",ranking_class_counts=str(counts.tolist()),
                    inner_refits=0,outer_refits=0,verified_selections=0,guard_size=0)
    pre=read(folder/"proposals.json"); sel=read(folder/"selection.json"); done=read(folder/"completed.json")
    assert sha(folder/"proposals.json")==sel["proposals_sha256"] and sha(folder/"selection.json")==done["selection"]
    assert sha(folder/"predictions.npz")==done["predictions"]
    proposal,validation=train_test_split(rank,test_size=.5,stratify=y[rank],random_state=39823+f)
    for field,ids in (("train",train),("test",test),("guard",validation),("proposal",proposal)):
        np.testing.assert_array_equal(pre[field],ids)
    groups=[set(z) for z in (train,test,validation,proposal)]
    assert all(not(a&b) for a,b in combinations(groups,2))
    seed=20421+f
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr=imp.transform(x[train]),imp.transform(x[proposal])
    rank_cols,k=compact_gain(fit(xt,y[train],seed), xt,xr,y[proposal],seed)
    assert pre["order"]==rank_cols and pre["candidates"]["anchor"]==sorted(rank_cols[:k])
    cand=pre["candidates"]
    assert cand["half"]==sorted(rank_cols[:ceil(k/2)]) and cand["same_width"]==sorted(rank_cols[:len(cand["vote"])])
    inner=predictions(x,y,np.sort(np.r_[train,proposal]),validation,seed)
    final=predictions(x,y,dev,test,seed)
    rare=len(np.unique(y))==2 and min(np.bincount(y[proposal]))<=len(proposal)*.05
    assert rare==pre["rare"]
    records={r["arm"]:r for r in done["rows"]}
    with np.load(folder/"predictions.npz") as probs:
        for a,vals in (("row_id",test),("y",y[test]),("guard_row",validation),("guard_y",y[validation])):
            np.testing.assert_array_equal(probs[a],vals)
        def inner_scores(arm):
            p=inner(tuple(cand[arm])); np.testing.assert_array_equal(p,probs["inner_"+arm])
            s=scores(y[validation],p,rare)
            assert all(abs(v-sel["guard_metrics"][arm][key])<1e-12 for key,v in s.items())
            return s
        base=inner_scores("anchor")
        reference=scores(y[test],final(tuple(cand["anchor"])),rare)
        for arm in sel["selected"]:
            raw=arm.removesuffix("_guard")
            guarded=arm!=raw
            reject=fails(base,inner_scores(raw),.03)
            support=cand["anchor"] if guarded and reject else cand[raw]
            assert sel["selected"][arm]==support and records[arm]["fallback"]==bool(guarded and reject)
            p=final(tuple(support)); np.testing.assert_array_equal(p,probs[arm])
            s=scores(y[test],p,rare)
            assert all(abs(v-records[arm][m])<1e-12 for m,v in s.items())
            assert records[arm]["exceeds"]==fails(reference,s,.03)
    return dict(dataset=name,fold=f,status="complete",ranking_class_counts=str(counts.tolist()),
                inner_refits=inner.cache_info().misses,outer_refits=final.cache_info().misses,
                verified_selections=len(sel["selected"]),guard_size=len(validation))


def challenger(name,bundle):
    f,x,y,dev,train,rank,test,_=bundle
    folder=STUDIES/"challenger"/name/f"fold-{f}"
    if min(np.bincount(y[rank]))<2:
        assert not (folder/"completed.json").is_file()
        return dict(dataset=name,fold=f,status="ineligible")
    seed=20421+f
    predecessor=STUDIES/"refit-guard"/name/f"fold-{f}"
    base=read(predecessor/"proposals.json"); settled=read(predecessor/"selection.json")
    prior=read(folder/"proposals.json"); final=read(folder/"selection.json")
    done=read(folder/"completed.json")
    assert done["selection"]==sha(folder/"selection.json") and done["predictions"]==sha(folder/"predictions.npz")
    assert final["proposals"]==sha(folder/"proposals.json")
    proposal=np.asarray(base["proposal"]); validation=np.asarray(base["guard"]); rare=base["rare"]
    order=base["order"]; anchor=frozenset(base["candidates"]["anchor"]); tail=order[len(anchor):]
    live=frozenset(base["candidates"]["vote"])
    ranked=[c for c in reversed(order) if c in live]
    if len(tail)<2:
        family=list(dict.fromkeys((frozenset(),frozenset(tail))))
    else:
        q=2
        while comb(q,q//2)<len(tail):
            q+=1
        signatures=list(islice(combinations(range(q),q//2),len(tail)))
        family=[frozenset()]+[frozenset(c for c,s in zip(tail,signatures) if i in s) for i in range(q)]+[frozenset(tail)]
    readmask=ballot_reader(x,y,train,proposal,seed,rare)
    min_scores=[]
    if len(live)>1:
        for c in ranked:
            values=[]
            for ext in family:
                ref=readmask(anchor|ext)
                can=readmask((live-{c})|ext)
                values.append(sum(not fails(a,b,.03) for a,b in zip(ref,can)))
            min_scores.append(min(values))
        best=max(min_scores); best_c=ranked[min_scores.index(best)]
        candidates={"maximin":sorted(live-{best_c}),"weak_gain":sorted(live-{ranked[0]})}
        assert prior["counts"]["min_yes"]==best and prior["counts"]["removed"]==best_c
        assert prior["eager"]["visits"]==len(ranked)*len(family)
    else:
        best=None
        candidates={a:sorted(live) for a in ("maximin","weak_gain")}
    assert prior["candidates"]==candidates
    assert prior["eager_masks"]==readmask.cache_info().misses
    inner=predictions(x,y,np.sort(np.r_[train,proposal]),validation,seed)
    outer=predictions(x,y,dev,test,seed)
    with np.load(folder/"predictions.npz") as probs:
        anchor_score=scores(y[validation],inner(tuple(sorted(anchor))),rare)
        original_score=scores(y[test],outer(tuple(sorted(anchor))),rare)
        rows={r["arm"]:r for r in done["rows"]}
        for arm in final["selected"]:
            raw=arm.removesuffix("_guard")
            s=candidates[raw]
            fresh=inner(tuple(s))
            np.testing.assert_array_equal(fresh,probs["inner_"+raw])
            guard_reject=fails(anchor_score,scores(y[validation],fresh,rare),.03)
            if raw!=arm and guard_reject:
                s=settled["selected"]["vote_guard"]
            assert final["selected"][arm]==s
            fresh=outer(tuple(s))
            np.testing.assert_array_equal(fresh,probs[arm])
            outcome=scores(y[test],fresh,rare)
            assert all(abs(rows[arm][k]-v)<1e-12 for k,v in outcome.items())
            assert rows[arm]["exceeds"]==fails(original_score,outcome,.03)
    return dict(dataset=name,fold=f,status="complete",candidates=len(min_scores),best=best,
                tied_best=sum(m==best for m in min_scores),different=candidates["maximin"]!=candidates["weak_gain"],
                eager_distinct=readmask.cache_info().misses,inner_refits=inner.cache_info().misses,
                outer_refits=outer.cache_info().misses)


def sensitivity(df):
    values=[]
    rates=[]
    specific=[]
    for (pop,t),sub in df.groupby(["cohort","relative_threshold"]):
        for field in ("failure","auc_only","accuracy_only"):
            hit=sub[field]
            extra=sub.separating & ~sub.endpoints
            assert not (sub.endpoints & ~sub.separating).any()
            per_fold=sub[["dataset","fold"]].assign(tp=extra & hit,fp=extra & ~hit).groupby(["dataset","fold"]).mean()
            macro=per_fold.groupby("dataset").mean().mean()
            specific.append(dict(cohort=pop,threshold=t,target=field,failures=int(hit.sum()),
                                 extra_tp=int((extra&hit).sum()),extra_fp=int((extra&~hit).sum()),
                                 equal_task_extra_tp=macro.tp,equal_task_extra_fp=macro.fp))
        for rule in RULES+["always_veto","always_allow"]:
            v=pd.Series(rule=="always_veto",index=sub.index) if rule.startswith("always") else sub[rule].astype(bool)
            fail=sub.failure
            flags=sub[["dataset","fold"]].assign(tp=(v&fail),fp=(v&~fail),fn=(~v&fail),tn=(~v&~fail))
            fold_mean=flags.groupby(["dataset","fold"]).mean()
            per_task=fold_mean.groupby("dataset").mean()
            macro=per_task.mean()
            totals=flags[["tp","fp","fn","tn"]].sum()
            values.append(dict(cohort=pop,threshold=t,rule=rule,n=len(sub),**totals.to_dict(),
                               **{"equal_task_"+k:v for k,v in macro.items()}))
            rates.extend(dict(dataset=task,cohort=pop,threshold=t,rule=rule,**line.to_dict())
                         for task,line in per_task.iterrows())
    pd.DataFrame(values).to_csv(HERE/"refit-review-sensitivity.csv",index=False)
    pd.DataFrame(rates).to_csv(HERE/"refit-review-by-task.csv",index=False)
    pd.DataFrame(specific).to_csv(HERE/"refit-review-by-target.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope",choices=["target","gain","guard","challenger"])
    parser.add_argument("--fresh",action="store_true",help="Refit all proposed deployment supports")
    parser.add_argument("--ballots",action="store_true",help="Rebuild every observed ranking-panel ballot")
    args=parser.parse_args()
    checks,rows=[],[]
    for name in NAMES:
        for data in split_data(name):
            if args.scope=="gain":
                checks.append(gain(name,data))
            elif args.scope=="guard":
                checks.append(guard(name,data))
            elif args.scope=="challenger":
                checks.append(challenger(name,data))
            else:
                r,report=target(name,data,args.fresh,args.ballots)
                rows.extend(r); checks.append(report)
        print(f"{args.scope}: {name} ({len(checks)}/42)",flush=True)
    assert len(checks)==42
    # Leave the full model/ballot audit receipt intact when replaying probabilities.
    if args.scope!="target" or (args.fresh and args.ballots):
        pd.DataFrame(checks).to_csv(HERE/f"refit-review-{args.scope}.csv",index=False)
    if rows:
        allrows=pd.DataFrame(rows)
        keys=["dataset","fold","subset","cohort"]
        current=allrows[allrows.relative_threshold==.03].sort_values(keys).reset_index(drop=True)
        saved=pd.read_csv(STUDIES/"refit-target/decisions.csv").sort_values(keys).reset_index(drop=True)
        for field in keys+RULES:
            np.testing.assert_array_equal(current[field],saved[field])
        np.testing.assert_array_equal(current.failure,saved.fail)
        for field in ("loss_accuracy","loss_auc"):
            np.testing.assert_allclose(current[field],saved[field],rtol=0,atol=1e-12)
        sensitivity(allrows)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
