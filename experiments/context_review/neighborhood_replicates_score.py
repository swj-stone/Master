"""Compare all three exchange slates with the fold and source as the paired units."""
from itertools import combinations
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import filter_neighborhood as original
from experiments.context_review.filter_neighborhood_score import METRICS, PAIRS, support_law
from experiments.context_review.forest_replay import comparisons, read_member, read_metrics
from experiments.context_review.neighborhood_replicates import DESIGN, DEST, ROOT, nominate
from experiments.context_review.refit_opportunity import outcome


def records():
    lock = bg.read(DEST/"prediction-lock.json")
    assert lock["design_sha"] == bg.digest(DESIGN) and lock["archive_sha"] == bg.digest(DEST/"predictions.zip")
    assert lock["original_sha"] == bg.digest(original.DEST/"prediction-lock.json")
    first = bg.read(original.DEST/"prediction-lock.json")
    assert set(lock["folds"])==set(first["folds"])
    folds,overlap,details = [],[],[]
    with ZipFile(DEST/"predictions.zip") as archive, ZipFile(ROOT/"thesis-paper/evidence/receiver/outcomes.zip") as labels:
        for key,rec in lock["folds"].items():
            ident = {k:rec[k] for k in ("panel","dataset","fold")}
            ref = first["folds"][key]
            assert rec["inputs_sha"]==ref["inputs_sha"] and rec["targets_sha"]==ref["targets_sha"]
            supports = rec["supports"]
            assert [supports[i] for i in rec["slates"][0]]==ref["supports"]
            size = len(supports)
            assert len({tuple(s) for s in supports})==size
            with read_member(archive,key,rec["member_sha"]) as p, read_member(labels,key,rec["targets_sha"]) as t:
                np.testing.assert_array_equal(p["test_row"],t["test_row"])
                tensor,points = p["tensor"],p["points"]
                metrics = read_metrics(p,p["guard_y"],t["y"],size,"ap" in rec["metrics"])
            guards = np.array([outcome(g,metrics[0][0],.03) for g,o in metrics])
            outer = np.array([outcome(o,metrics[0][1],.03) for g,o in metrics])
            savings = np.array([1-len(cols)/len(supports[0]) for cols in supports])
            scores = {m:np.array([o.get(m,np.nan) for g,o in metrics]) for m in ("auc","accuracy","ap")}
            scores.update(u=savings*outer,breach=~outer,saved=savings)
            for i,cols in enumerate(supports):
                details.append(dict(**ident,index=i,columns=" ".join(map(str,cols)),guard=guards[i],outer=outer[i],
                                    in_draws=" ".join(str(j+1) for j,s in enumerate(rec["slates"]) if i in s)))
            for a,b in combinations(range(3),2):
                sa,sb = [set(rec["slates"][j][1:]) for j in (a,b)]
                inter = len(sa&sb)
                union = len(sa|sb)
                overlap.append(dict(**ident,draw_a=a+1,draw_b=b+1,intersection=inter,union=union,
                                    jaccard=inter/union if union else 1.,identical=sa==sb,
                                    shared_beyond_prefix=max(0,inter-1)))
            for j,slate in enumerate(rec["slates"]):
                derived = nominate(tensor,points,rec["metrics"],slate)
                assert derived==rec["winners"][j]
                options = slate[1:] or [0]
                choices = {"prefix":[options[0]],"uniform":options,**{k:[v] for k,v in derived.items()}}
                for delivery in ("guarded","unchecked"):
                    for method,selected in choices.items():
                        law = support_law(selected,guards,delivery=="guarded",size)
                        folds.append(dict(**ident,draw=j+1,seed=lock["seeds"][j],delivery=delivery,
                            method=method, nominees=" ".join(map(str,selected)),
                            **{m:float(law@s) for m,s in scores.items()}))
    return pd.DataFrame(folds),pd.DataFrame(overlap),pd.DataFrame(details)


def replay():
    folds,overlap,candidates = records()
    # Enforce consistency with the original single draw for every deployed outcome.
    keys = ["panel","dataset","fold","delivery","method"]
    first = pd.read_csv(original.DEST/"policies.csv").set_index(keys).sort_index()
    draw1 = folds[folds.draw==1].set_index(keys).sort_index()
    assert draw1.index.equals(first.index)
    np.testing.assert_allclose(draw1[list(METRICS)],first[list(METRICS)],atol=1e-12,rtol=0,equal_nan=True)
    by = ["panel","delivery"]
    by_draw = folds.groupby([*by,"draw","method","dataset"],as_index=False)[list(METRICS)].mean()
    draw_means = by_draw.groupby([*by,"draw","method"],as_index=False)[list(METRICS)].mean()
    draw_contrasts = comparisons(by_draw,[*by,"draw"],PAIRS,("u","breach"))
    fold_means = folds.groupby(keys,as_index=False)[list(METRICS)].mean()
    tasks = fold_means.groupby([*by,"dataset","method"],as_index=False)[list(METRICS)].mean()
    means = tasks.groupby([*by,"method"],as_index=False)[list(METRICS)].mean()
    contrasts = comparisons(tasks,by,PAIRS,("u","breach","auc","accuracy"))
    for file,frame in (("policies",folds),("candidates",candidates),("overlap",overlap),
                       ("draw-tasks",by_draw),("draw-means",draw_means),("draw-contrasts",draw_contrasts),
                       ("tasks",tasks),("means",means),("contrasts",contrasts)):
        frame.to_csv(DEST/(file+".csv"),index=False)
    print(contrasts.query("delivery=='guarded' and method=='slack' and reference=='vote' and metric=='u'").to_string(index=False))


if __name__=="__main__":
    replay()
