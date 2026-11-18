"""Verify FSDD supports, outcomes, held-out entities, scores and paired timing."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from research.conditional_dense.fsdd.features import read_corpus
from research.conditional_dense.fsdd.study import METHODS


def checksums(archive, filenames):
    result = {}
    with ZipFile(archive) as z:
        for member in z.namelist():
            path = Path(member)
            if path.parent.name=="recordings" and path.suffix==".wav":
                rate,samples = wavfile.read(io.BytesIO(z.read(member)))
                result[path.name] = hashlib.sha256(str(rate).encode()+samples.tobytes()).digest()
    return [result[name] for name in filenames]


def unit_ba(y, pred, units):
    return sum(balanced_accuracy_score(y[units==g],pred[units==g]) for g in np.unique(units))/len(np.unique(units))


def audit(first,second,out,archive):
    out.mkdir(parents=True,exist_ok=False)
    states = [json.loads((p/"protocol.json").read_text()) for p in (first,second)]
    assert [s.pop("process") for s in states] == [1,2]
    assert states[0] == states[1]
    manifest = states[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["corpus_sha256"]
    X,y,units,names,meta = read_corpus(archive)
    assert meta == manifest["corpus"]
    hashes = checksums(archive,names)
    with np.load(first/"index.npz") as a,np.load(second/"index.npz") as b:
        for field, expected in (("digit",y),("speaker",units),("filename",names)):
            np.testing.assert_array_equal(a[field],expected)
            np.testing.assert_array_equal(b[field],expected)
    counts = pd.crosstab(units,y)
    assert counts.shape == (6,10) and counts.eq(50).all().all()
    counts.to_csv(out/"speaker-digit-counts.csv")
    raw = pd.concat([pd.read_csv(p/"trials.csv").assign(process=i) for i,p in enumerate((first,second),1)])
    assert len(raw)==600 and not raw.duplicated(["process","fold","method","repeat"]).any()
    assert (raw.select_ms+raw.slice_ms+raw.fit_ms+raw.predict_ms <= raw.total_ms+1e-6).all()
    for _, d in raw.groupby(["process","fold","method"]):
        assert sorted(d.position)==sorted(d.repeat)==list(range(10))
    medians = raw.groupby(["fold","method","process"])[["ba","k","select_ms","slice_ms","fit_ms","predict_ms","total_ms"]].median()
    medians.reset_index().to_csv(out/"process-medians.csv",index=False)
    pairs = []; splits = []; draws = []; saved_arrays = 0; seen = set()
    for meta in manifest["folds"]:
        fold = meta["fold"]
        per = raw[raw.fold==fold]
        assert per.groupby("method").ba.nunique().eq(1).all()
        pred_score = per.groupby("method").ba.first()
        times = medians.loc[fold].reset_index().pivot(index="method",columns="process",values="total_ms")
        with np.load(first/f"fold-{fold}.npz") as a,np.load(second/f"fold-{fold}.npz") as b:
            assert set(a.files)==set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field],b[field])
            saved_arrays += len(a.files)
            tr,te = a["train_rows"],a["test_rows"]
            np.testing.assert_array_equal(np.sort(np.r_[tr,te]),np.arange(len(X)))
            assert not (set(te)&seen)
            seen.update(te)
            assert sorted(set(units[te])) == meta["test_people"]
            assert not set(units[tr]) & set(units[te])
            wave_dup = set(hashes[i] for i in tr) & set(hashes[i] for i in te)
            row_dup = set(X[i].tobytes() for i in tr) & set(X[i].tobytes() for i in te)
            splits.append(dict(fold=fold,test_speakers=",".join(meta["test_people"]),train_rows=len(tr),
                               test_rows=len(te),shared_waves=len(wave_dup),shared_vectors=len(row_dup)))
            assert np.array_equal(a["truth"],y[te]) and np.array_equal(a["units"],units[te])
            for arm in METHODS:
                cols = a[arm+"_cols"]
                assert np.array_equal(cols,np.unique(cols))
                assert cols.min()>=0 and cols.max()<208 and len(cols)==per[per.method==arm].k.iloc[0]
                np.testing.assert_allclose(unit_ba(y[te],a[arm+"_pred"],units[te]),pred_score[arm],rtol=0,atol=1e-12)
            for choice in ("old","new"):
                width = meta["widths"][choice]
                rand_scores = []
                for draw in range(20):
                    cols = np.sort(np.random.default_rng([61341,fold,meta["seed"],draw]).permutation(208)[:width])
                    np.testing.assert_array_equal(cols,a[f"{choice}_random{draw}_cols"])
                    s = unit_ba(y[te],a[f"{choice}_random{draw}_pred"],units[te])
                    rand_scores.append(s); draws.append(dict(fold=fold,method=choice,draw=draw,score=s))
                row = dict(fold=fold,seed=meta["seed"],method=choice,p=208,k=width,score=pred_score[choice],
                    random20_median=np.median(rand_scores),random20_q25=np.quantile(rand_scores,.25),
                    random20_q75=np.quantile(rand_scores,.75),outscored_random20=int(np.sum(pred_score[choice]>np.array(rand_scores)+1e-12)))
                for n,ref in (("full","full"),("F",choice+"_F"),("sketch_F",choice+"_sketch_F"),
                             ("random",choice+"_random"),("half_F","half_F"),("old","old"),("new","new")):
                    row["score_"+n] = pred_score[ref]
                    row["delta_"+n] = pred_score[choice]-pred_score[ref]
                    row["k_"+n] = int(per[per.method==ref].k.iloc[0])
                    for process in (1,2):
                        row[f"ms{process}_method"] = times.loc[choice,process]
                        row[f"ms{process}_"+n] = times.loc[ref,process]
                        row[f"ratio{process}_"+n] = times.loc[choice,process]/times.loc[ref,process]
                    row["faster_"+n] = max(row["ratio1_"+n],row["ratio2_"+n])<1
                    row["better_faster_"+n] = row["faster_"+n] and row["delta_"+n]>1e-12
                row["full_reduced_faster"] = width<208 and row["faster_full"]
                row["full_old_tolerance_joint"] = row["full_reduced_faster"] and row["delta_full"]>=-.02-1e-12
                row["both_F_methods_and_full_joint"] = row["better_faster_F"] and row["better_faster_sketch_F"] and row["full_old_tolerance_joint"]
                pairs.append(row)
        assert all(bd["boundary_gap"]<=1e-10 for bd in meta["F_boundaries"].values())
    assert seen==set(range(len(X)))
    cases = pd.DataFrame(pairs);cases.to_csv(out/"cases.csv",index=False)
    pd.DataFrame(draws).to_csv(out/"random20.csv",index=False)
    pd.DataFrame(splits).to_csv(out/"split-audit.csv",index=False)
    summary = []
    for method,g in cases.groupby("method"):
        row = dict(method=method,source_count=1,folds=len(g),score=g.score.mean(),widths=g.k.tolist(),
                    full_reduced_faster=int(g.full_reduced_faster.sum()),full_joint=int(g.full_old_tolerance_joint.sum()),
                    joint_and_both_F_methods=int(g.both_F_methods_and_full_joint.sum()))
        for ref in ("full","F","sketch_F","random","half_F","old","new"):
            row["mean_delta_"+ref]=g["delta_"+ref].mean()
            row["strict_wins_"+ref]=int((g["delta_"+ref]>1e-12).sum())
            row["faster_"+ref]=int(g["faster_"+ref].sum())
            row["both_"+ref]=int(g["better_faster_"+ref].sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    report=dict(pipelines=len(raw),cross_process_arrays=saved_arrays,disjoint_outer_folds=3,source_count=1,
        official_archive=manifest["corpus_sha256"],matrix=manifest["corpus"]["matrix_sha256"],
        matching_F_supports=sum(bd["n_alternate_ties"]==0 for fold in manifest["folds"] for bd in fold["F_boundaries"].values()),
        f_supports_checked=15,preparation_ms=[json.loads((p/"preparation.json").read_text())["common_wave_to_matrix_ms"] for p in (first,second)])
    (out/"verification.json").write_text(json.dumps(report,indent=2))
    print(cases[["fold","method","k","score","delta_full","delta_F","delta_sketch_F","ratio1_full","ratio2_full",
                 "ratio1_F","ratio2_F","ratio1_sketch_F","ratio2_sketch_F","outscored_random20","both_F_methods_and_full_joint"]].round(4).to_string(index=False))
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("first","second","out","archive"):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    with threadpool_limits(1):
        audit(args.first,args.second,args.out,args.archive)
