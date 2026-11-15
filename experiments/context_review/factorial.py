"""Paired, retrospective row-panel by restoration-family replay on frozen proposals."""
import argparse
from collections import Counter
from hashlib import file_digest
from itertools import combinations
import json
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs/context_review/factorial.json"
RAW = ROOT / json.loads(DESIGN.read_bytes())["source"]
OUT = ROOT / "results/context_review/factorial"
CELLS = ["rank_ends", "rank_interior", "guard_ends", "guard_interior"]


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def scores(predictions, y, weights, task, rare):
    """Weighted AUC from negative-score empirical CDF at each positive prediction."""
    value = []
    for pred in predictions:
        if task == "regression":
            mu = weights @ y
            sse = weights @ ((pred-y)**2)
            var = np.sum(weights * (y[None,:] - mu[:,None])**2, axis=1)
            clipped = np.maximum(0, 1-sse/np.where(var>0, var, 1))
            cols = [-weights @ np.abs(pred-y), np.where(var>0, clipped, sse==0)]
        else:
            aucs = []
            for j in ([1] if task == "binary" else range(pred.shape[1])):
                neg = np.flatnonzero(y != j)
                pos = np.flatnonzero(y == j)
                neg = neg[np.argsort(pred[neg,j], kind="stable")]
                sorted_neg, positive = pred[neg,j], pred[pos,j]
                left = np.searchsorted(sorted_neg, positive, side="left")
                right = np.searchsorted(sorted_neg, positive, side="right")
                cdf = np.pad(np.cumsum(weights[:,neg], axis=1), ((0,0),(1,0)))
                pair_mass = weights[:,pos].sum(axis=1) * cdf[:,-1]
                aucs.append(np.sum(weights[:,pos]*(cdf[:,left]+cdf[:,right])/2, axis=1)/pair_mass)
            cols = [np.mean(aucs,axis=0), weights @ (np.argmax(pred,axis=1)==y)]
            if rare:
                cols.append(np.array([average_precision_score(y,pred[:,1],sample_weight=w) for w in weights]))
        value.append(np.stack(cols,axis=-1))
    answer = np.array(value)
    if not np.isfinite(answer).all():
        raise ValueError("Undefined panel metric")
    return answer


def panel_votes(pred, y, weights, receipt):
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    assert len(weights)==4 and np.all(weights>=0)
    # Normalize to the standard weighted mean even for tiny summation differences.
    metric = scores(pred, y, weights/weights.sum(axis=1)[:,None], receipt["task"], receipt["rare"])
    aligned = metric[np.array(receipt["lookup"])]
    anchor = aligned[:1]
    tolerance = np.array([.03,.03,.05] if metric.shape[-1] == 3 else [.03,.03])
    return np.all(anchor-aligned[1:] <= tolerance*np.abs(anchor)+1e-12, axis=-1)


def check_receipt(path, d):
    for name, expected in d["source"].items():
        assert sha(ROOT/name)==expected
    assert sha(path.parent/"predictions.npz")==d["predictions_sha256"]
    sources = list(map(Path,d["source"]))
    raw = json.loads((ROOT/next(s for s in sources if s.name=="estimator.json")).read_bytes())
    metadata = ROOT/next(s for s in sources if s.name=="metadata.json")
    meta = json.loads(metadata.read_bytes())
    names = {name:i for i,name in enumerate(meta["features"])}
    anchor = frozenset(d["anchor"])
    assert anchor == frozenset(names[s] for s in raw["anchor_order"])
    live, history = anchor, {}
    for step in raw["trace"]:
        proposed = live.difference(names[s] for s in step["block"])
        assert len(live)==step["before"] and proposed and proposed!=live
        assert history.setdefault(proposed,step["votes"])==step["votes"]
        if step["accepted"]:
            live = proposed
    indices = np.linspace(0,len(history)-1,min(len(history),12),dtype=int)
    records = list(history.items())
    assert [sorted(records[i][0]) for i in indices] == d["candidates"]
    assert [records[i][1] for i in indices] == d["original_votes"]
    extensions = list(map(frozenset,d["extensions"]))
    masks = list(map(frozenset,d["masks"]))
    assert len(extensions)==len(set(extensions))
    assert all(e<=set(d["tail"]) and not e & anchor for e in extensions)
    assert [extensions[i] for i in d["rules"]["ends"]]==list(dict.fromkeys([frozenset(),frozenset(d["tail"])]))
    assert not set(d["novel"]) & {i for rule in d["rules"].values() for i in rule}
    for support, lookup in zip([anchor,*map(frozenset,d["candidates"])],d["lookup"],strict=True):
        assert [support|e for e in extensions] == [masks[i] for i in lookup]
    y_path = metadata.parent/"val_y.npy"
    assert sha(y_path)==meta["sha256"]["val_y.npy"]
    return np.load(y_path), len(records)


def calculate(name, seed):
    path = RAW/f"seed-{seed}"/name/"receipt.json"
    d = json.loads(path.read_bytes())
    assert d["dataset"]==name and int(d["seed"])==seed
    validation_y, all_candidates = check_receipt(path,d)
    with np.load(path.parent/"predictions.npz", allow_pickle=False) as f:
        p = f["predictions"]
        n = len(f["rank_rows"])
        assert not set(f["rank_rows"]) & set(f["guard_rows"])
        for tag in ("rank","guard"):
            rows = f[tag+"_rows"]
            assert len(rows)==len(set(rows))
            np.testing.assert_array_equal(validation_y[rows],f[tag+"_y"])
        rank = panel_votes(p[:,:n],f["rank_y"],f["rank_w"],d)
        guard = panel_votes(p[:,n:],f["guard_y"],f["guard_w"],d)
    novel, ends = d["novel"], d["rules"]["ends"]
    rows, checks, signs = [], [], []
    for index,(r,g) in enumerate(zip(rank,guard,strict=True)):
        archived = d["original_votes"][index]
        np.testing.assert_array_equal(r[d["rules"]["nested"][:len(archived)]], archived)
        row = dict(dataset=name, seed=seed, proposal=index, unique_trajectory=all_candidates,
                   anchor=len(d["anchor"]), tail=len(d["tail"]), endpoints=len(ends), novel=len(novel))
        yr, yg = r.sum(axis=-1), g.sum(axis=-1)
        for tag, yes in (("rank",yr),("guard",yg)):
            row[f"{tag}_ends"] = int(min(yes[ends])<3)
            row[f"{tag}_interior"] = int(min(yes[novel])<3) if novel else None
            row[f"{tag}_interior_fails"] = int(sum(yes[novel]<3))
            row[f"{tag}_interior_tied_panels"] = int(sum(yes[novel]==2))
            if len(novel)>=len(ends):
                row[f"{tag}_matched_budget_failure"] = 1-comb(int(sum(yes[novel]>=3)),len(ends))/comb(len(novel),len(ends))
        for rule, family in d["rules"].items():
            for tag, yes in (("rank",yr),("guard",yg)):
                fail = sum(yes[family])<=2*len(family) if rule=="pooled" else min(yes[family])<3
                row[f"{tag}_{rule}_veto"] = int(fail)
            checks.append(dict(dataset=name,seed=seed,proposal=index,rule=rule,
                               accept=1-row[f"rank_{rule}_veto"],guard_design_accept=1-row[f"guard_{rule}_veto"],
                               rank_novel_failing=row["rank_interior_fails"],guard_novel_failing=row["guard_interior_fails"]))
        group = Counter()
        for i,j in combinations(novel,2):
            if len(d["extensions"][i])!=len(d["extensions"][j]):
                continue
            sr = int(yr[i]>=3)-int(yr[j]>=3)
            sg = int(yg[i]>=3)-int(yg[j]>=3)
            group[sr,sg]+=1
        for (sr,sg),count in sorted(group.items()):
            signs.append(dict(dataset=name,seed=seed,proposal=index,rank_direction=sr,guard_direction=sg,pairs=count))
        rows.append(row)
    folder = OUT/f"seed-{seed}"/name
    folder.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(folder/"factorial-candidates.csv",index=False)
    pd.DataFrame(checks).to_csv(folder/"factorial-decisions.csv",index=False)
    pd.DataFrame(signs,columns=["dataset","seed","proposal","rank_direction","guard_direction","pairs"]).to_csv(
        folder/"factorial-directions.csv",index=False)


def summarize(rows):
    reports=[]
    labels=["all probeable","ranking endpoints pass","ranking and guard endpoints pass"]
    for ds,whole in [("ALL",rows),*rows.groupby("dataset")]:
        probe=whole[whole.novel.gt(0)]
        for label, part in zip(labels,[probe,probe[probe.rank_ends.eq(0)],
                                        probe[probe.rank_ends.eq(0)&probe.guard_ends.eq(0)]],strict=True):
            stats=dict(dataset=ds,condition=label,total=len(whole),eligible=len(probe),subset=len(part),
                       guard_endpoint_veto_all=int(whole.guard_ends.sum()),
                       ranking_endpoint_veto_all=int(whole.rank_ends.sum()))
            stats.update({col:int(part[col].sum()) for col in CELLS})
            for col in ["rank_matched_budget_failure","guard_matched_budget_failure"]:
                stats[col]=part[col].mean()
            for col in ("nested","separating","random_matched"):
                veto=part[f"rank_{col}_veto"].eq(1)
                fails=part.guard_interior.eq(1)
                stats.update({f"{col}_flagged_fail":int((veto & fails).sum()),
                              f"{col}_flagged_no_fail":int((veto & ~fails).sum())})
            reports.append(stats)
    return pd.DataFrame(reports)


def replay():
    spec = json.loads(DESIGN.read_bytes())
    names,seeds=spec["datasets"],spec["seeds"]
    parts=[(name,seed) for name in names for seed in seeds]
    files={RAW/f"seed-{seed}"/name/"receipt.json" for name,seed in parts}
    pinfiles=files|{DESIGN,Path(__file__),RAW/"decisions.csv"}
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(pinfiles)}
    OUT.mkdir(exist_ok=True,parents=True)
    pin=OUT/"factorial-pins.json"
    if pin.exists() and json.loads(pin.read_bytes())!=pins:
        raise ValueError("Frozen replay or source changed")
    pin.write_text(json.dumps(pins,indent=2),encoding="utf8")
    for name,seed in parts:
        subprocess.run([sys.executable,"-m","experiments.context_review.factorial","--worker",name,str(seed)],
                       check=True,timeout=3600)
        print(f"Checked {name}/{seed}",flush=True)
    def collect(key):
        paths=[OUT/f"seed-{s}"/n/f"factorial-{key}.csv" for n,s in parts]
        chunks=[pd.read_csv(p) for p in paths]
        return pd.concat([c for c in chunks if len(c)],ignore_index=True)
    rows,checks,signs=map(collect,("candidates","decisions","directions"))
    assert len(rows)==274 and rows.novel.gt(0).sum()==216
    expected=pd.read_csv(RAW/"decisions.csv")
    keys=["dataset","seed","proposal","rule"]
    pd.testing.assert_frame_equal(expected[checks.columns].sort_values(keys).reset_index(drop=True),
                                  checks.sort_values(keys).reset_index(drop=True),check_dtype=False)
    rows.to_csv(OUT/"factorial-candidates.csv",index=False)
    summarize(rows).to_csv(OUT/"factorial-summary.csv",index=False)
    eligible=rows[rows.novel.gt(0)]
    eligible.groupby(["dataset",*CELLS]).size().rename("proposals").to_csv(OUT/"factorial-joint.csv")
    signs.groupby(["dataset","rank_direction","guard_direction"]).pairs.sum().to_csv(OUT/"factorial-pairs.csv")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        calculate(args.worker[0],int(args.worker[1]))
    else:
        replay()
