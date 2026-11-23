"""Check the archived Sampled Single-peer manuscript numbers and rebuild its figures in isolation."""
import argparse
import hashlib
from itertools import combinations
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/minpi-lowdim"
EVIDENCE = ROOT / "research/fast_single"
PEER = ROOT / "research/lowdim/peer_evidence"
RUNS = ROOT / "results/minpi-study"
ARCHIVED_PEER = RUNS / "lowdim-peer-ablation-20261118a"
KEY = ["cohort", "dataset", "fold", "seed"]
ARMS = ("fast", "single", "F_fast", "random_fast", "F_single", "random_single", "full")
EPS = 1e-12


def read(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def wins(diff):
    return tuple(int(v.sum()) for v in (diff > EPS, abs(diff) <= EPS, diff < -EPS))


def ratio(frame, a, b, clock="total_ms"):
    return np.maximum(frame[f"{clock}1_{a}"] / frame[f"{clock}1_{b}"],
                      frame[f"{clock}2_{a}"] / frame[f"{clock}2_{b}"])


def absolute(frame, arm, clock="total_ms"):
    return ((frame[f"{clock}1_{arm}"] + frame[f"{clock}2_{arm}"]) / 2).median()


def joint(frame, arm, tolerance=.02):
    return ((frame["k_" + arm] < frame.p) & (ratio(frame, arm, "full") < 1)
            & (frame.score_full - frame["score_" + arm] <= tolerance + EPS))


def verify_primary(main):
    assert main.case_id.tolist() == list(range(405)) and not main.duplicated(KEY).any()
    assert len(main.drop_duplicates(["train_sha","target_sha","test_sha","truth_sha"])) == 319
    assert (main.n_train.min(), main.n_train.max(), main.p.min(), main.p.max()) == (100, 182276, 4, 54877)
    assert main.groupby("cohort").size().to_dict() == {
        "original":69, "extra":57, "formal":40, "four_stage_ridge":30, "dense":19, "spectra":36,
        "mechanism":64, "archive_supplement":33, "text":38, "conference_fraud":1, "external":12, "fresh":6,
    }
    assert main.k_full.equals(main.p)
    for owner in ("fast", "single"):
        for base in ("F", "random"):
            assert main["k_" + owner].equals(main[f"k_{base}_{owner}"])
    assert (main.k_fast < main.p).sum() == 403
    assert [int(joint(main, a).sum()) for a in ("fast","single","F_fast","random_fast")] == [215,163,131,152]
    assert joint(main,"fast",0).sum() == 143
    assert ((main.k_fast < main.p) & (ratio(main,"fast","full") < 1)).sum() == 320
    low, wide = main[main.p <= 32], main[main.p > 32]
    assert (len(low), len(wide)) == (74,331)
    assert low.cols_sha_fast.equals(low.cols_sha_single) and (low.cols_sha_fast == low.cols_sha_F_fast).sum() == 57
    assert [joint(low, a).sum() for a in ("fast","F_fast","random_fast")] == [25,2,23]
    assert [joint(wide, a).sum() for a in ("fast","F_fast","random_fast")] == [190,129,129]
    assert [(ratio(wide, a, "full") < 1).sum() for a in ("fast","random_fast")] == [291,323]
    saving = [round(100*(1-ratio(low,"fast",b,c).median()),1)
              for b in ("F_fast", "random_fast") for c in ("select_ms", "total_ms")]
    assert saving == [82.3,14.6,-52.2,-4.]
    assert wins(main.score_fast-main.score_random_fast) == (286,10,109)
    assert (main.score_fast > main.score_F_fast + EPS).sum() == 159
    rr = ratio(main,"fast","random_fast")
    assert (round(rr.median(),2), (rr < 1).sum(), (rr <= 1.1).sum()) == (1.11,16,177)
    assert ((rr <= 1.1) & joint(main,"fast")).sum() == 83
    assert round(ratio(main,"fast","F_fast").median(),2) == .78
    assert (ratio(wide,"fast","single") < 1).sum() == 319 and wins(wide.score_fast-wide.score_single) == (122,9,200)
    assert [round(100*(main["k_"+a]/main.p).median(),1) for a in ("fast","single")] == [43.8,49.2]
    for task, row in {"binary":(187,161,113,73,1), "multiclass":(62,41,35,22,1), "regression":(156,118,67,59,7)}.items():
        sub = main[main.task == task]
        assert (len(sub), joint(sub,"fast",np.inf).sum(), joint(sub,"fast").sum(),
                *[((sub.score_fast > sub["score_"+b] + EPS) & (ratio(sub,"fast",b)<1)).sum()
                  for b in ("F_fast", "random_fast")]) == row
    for name, counts, scores, times in (("abalone",(12,8,8,7),[.52886,.52870,.52886,.51086],
                                        [1.169,1.107,1.339,1.154]),
                                       ("diabetes",(11,5,7,4),[.46770,.48443,.46770,.40045],
                                        [1.008,1.046,1.276,1.004])):
        group = main[main.dataset.str.lower() == name]
        ordered = ("fast","full","F_fast","random_fast")
        assert (len(group),group.k_fast.min(),group.k_fast.max(),(ratio(group,"fast","full")<1).sum())==counts
        assert group.cols_sha_fast.equals(group.cols_sha_F_fast)
        assert [round(group["score_"+a].mean(),5) for a in ordered] == scores
        assert [round(absolute(group,a),3) for a in ordered] == times
    largest = wide[wide.p > 2048]
    assert len(largest) == 43
    assert [round(absolute(largest,a,c),2) for c in ("select_ms","total_ms")
            for a in ("single","fast")] == [119.56,2.16,127.03,10.94]
    reg = wide[wide.task == "regression"]
    assert len(reg)==115 and tuple((reg.score_full-reg.score_fast).quantile([.5,.9]).round(4)) == (.0076,.5134)
    print("Verified 405 configurations, 319 distinct inputs, matched widths and all three manuscript tables.")


def verify_ablation_transfer(main, transfer):
    ab = read(EVIDENCE/"evidence/ablation-cases.csv").set_index("case_id")
    wide = main[main.p>32].set_index("case_id")
    for name in ("p","k_fast","score_fast","score_full","cols_sha_fast","pred_sha_fast","decision_sha_fast"):
        pd.testing.assert_series_equal(ab[name], wide[name])
    assert (ab.k_marginal==ab.k_fast).all() and (ab.cols_sha_fast!=ab.cols_sha_marginal).sum()==152
    assert wins(ab.score_fast-ab.score_marginal)==(81,197,53)
    assert wins(ab.score_fast-ab.score_unadjusted)==(105,33,193)
    assert [round(100*(ab["k_"+a]/ab.p).median(),1) for a in ("unadjusted","fast")]==[42.2,37.5]
    text = ab[ab.train_nnz_full!=""]
    nnz = text.train_nnz_fast.astype(float)/text.train_nnz_random.astype(float)
    assert len(text)==38 and nnz.gt(1).all() and round(nnz.median(),2)==3.06
    sparse_rows = wide.loc[text.index]
    assert wins(sparse_rows.score_fast-sparse_rows.score_random_fast) == (38,0,0)
    assert round(ratio(sparse_rows,"fast","random_fast").median(),2)==1.66
    mango=ab[ab.dataset=="24-Mango_S3_dm"]
    assert [(int(mango["k_"+a].min()),int(mango["k_"+a].max()),round(mango["score_"+a].median(),3))
            for a in ("fast","unadjusted")] == [(23,31,.359),(32,44,.710)]
    assert len(transfer)==12*7 and not transfer.duplicated(["dataset","seed","arm"]).any()
    assert set(transfer.arm)==set(ARMS)
    indexed=transfer.set_index(["arm","dataset","seed"]).sort_index()
    rows=indexed.loc["fast"].copy()
    for base, column in (("random_fast","random"),("F_fast","F"),("full","full")):
        ref=indexed.loc[base]
        if base!="full":
            pd.testing.assert_series_equal(rows.k,ref.k)
        gap=rows.score-ref.score
        times=np.maximum(rows.total_ms_1/ref.total_ms_1,rows.total_ms_2/ref.total_ms_2)
        np.testing.assert_allclose(gap,rows["delta_score_vs_"+column].astype(float),rtol=0,atol=EPS)
        np.testing.assert_allclose(times,rows["worst_ratio_vs_"+column].astype(float),rtol=0,atol=EPS)
        rows["delta_score_vs_"+column]=gap
        rows["worst_ratio_vs_"+column]=times
    assert rows.groupby("dataset").size().eq(3).all()
    ok=rows.k.lt(rows.p) & rows.loss_vs_full.le(.02+EPS) & rows.worst_ratio_vs_full.lt(1.)
    assert rows.assign(ok=ok).groupby("dataset").ok.sum().to_dict()==dict(gasoline=1,olive=3,real_sim=0,yarn=0)
    med=rows.groupby("dataset").median(numeric_only=True)
    assert (med.at["olive","k"],round(med.at["olive","loss_vs_full"],4),
            round(med.at["olive","worst_ratio_vs_full"],2)) == (163,.0104,.82)
    assert (med.at["real_sim","k"], round(med.at["real_sim","delta_score_vs_random"],3),
            round(med.at["real_sim","delta_score_vs_full"],3),round(med.at["real_sim","worst_ratio_vs_full"],2),
            round(med.at["real_sim","worst_ratio_vs_random"],2)) == (1536,.139,-.059,.18,2.16)
    print("Verified all 331 wide-input ablations, 38 sparse nonzero counts and 12 supplementary configurations.")


def score(metric, y, pred, decision=None):
    if metric.lower()=="r2":
        return r2_score(y,pred)
    if metric.lower()=="auc":
        return roc_auc_score(y,decision)
    if metric.lower()=="ba":
        return balanced_accuracy_score(y,pred)
    raise ValueError(metric)


def verify_subsets(main, run, write_path):
    rows=read(PEER/"cases.csv")
    assert len(rows)==31 and not rows.duplicated(KEY).any()
    assert (rows.replaced_empty>0).sum()==7 and wins(rows.score_new-rows.score_empty)==(2,28,1)
    assert (rows.score_new-rows.random32_median>EPS).sum()==26
    main=main.set_index(KEY)
    subsets={}
    draws=enumerations=0
    for r in rows.itertuples():
        key=(r.cohort,r.dataset,r.fold,r.seed)
        ref=main.loc[key]
        name=f"{r.cohort}-{r.dataset}-{r.fold}-{r.seed}.npz"
        baseline=(RUNS/"lowdim-formal-20261118a-pass1"/f"{r.dataset}-{r.seed}.npz" if r.cohort=="formal"
                  else RUNS/"lowdim-core-20261118a-pass1"/name)
        with np.load(run/name) as recorded, np.load(baseline) as old:
            kept=recorded["new_cols"]
            assert r.k==ref.k_fast and r.p==ref.p and abs(r.score_new-ref.score_fast)<=EPS
            assert hashlib.sha256(kept.tobytes()).hexdigest()==ref.cols_sha_fast
            decisions=recorded.get("random32_decision",[None]*32)
            preds=recorded["random32_pred"]
            y=old["truth"]
            keys=recorded["random32_cols"]
            assert keys.shape==(32,r.k) and np.diff(keys,axis=1).min(initial=1)>0 and keys.min()>=0 and keys.max()<r.p
            vals=[score(r.metric,y,p,d) for p,d in zip(preds,decisions,strict=True)]
            np.testing.assert_allclose(vals,recorded["random32_scores"],atol=EPS,rtol=0)
            assert abs(np.median(vals)-r.random32_median)<=EPS
            draws+=len(vals)
            if r.dataset!="abalone":
                continue
            cs,vs=recorded["exhaustive_cols"],recorded["exhaustive_scores"]
            np.testing.assert_array_equal(cs,list(combinations(range(10),8)))
            np.testing.assert_allclose(vs,[r2_score(y,p) for p in recorded["exhaustive_pred"]],rtol=0,atol=EPS)
            items=[(r.fold," ".join(map(str,c)),v,int(np.array_equal(c,kept))) for c,v in zip(cs,vs,strict=True)]
            if r.fold in subsets:
                assert items==subsets[r.fold]
            subsets[r.fold]=items
            assert 1+sum(v>r.score_new+EPS for v in vs)==(4,1,11)[r.fold-1]
            enumerations+=len(vs)
    assert (draws,enumerations)==(992,405)
    rebuilt=pd.DataFrame([v for f in sorted(subsets) for v in subsets[f]],
                         columns=["fold","columns_zero_based","test_r2","minpi_selected"])
    published = PEER/"abalone-subset-scores.csv"
    if published.exists() or write_path is None:
        pd.testing.assert_frame_equal(rebuilt,read(published),check_exact=True)
    if write_path:
        with write_path.open("x",encoding="utf8",newline="") as stream:
            rebuilt.to_csv(stream,index=False,lineterminator="\n")
    print("Verified 31 Figure 2 supports, 992 random-subset predictions and all 405 exhaustive predictions.")


def render():
    names=("lowdim-runtime","subset-quality","matched-curves")
    with TemporaryDirectory(prefix="single-paper-") as directory:
        stage=Path(directory)
        paper=stage/"paper/minpi-lowdim"
        paper.mkdir(parents=True)
        shutil.copy2(PAPER/"main.tex",paper)
        data=[EVIDENCE/"evidence-extension/all-405.csv", PEER/"cases.csv", PEER/"abalone-subset-scores.csv"]
        for f in data:
            target=stage/f.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,target)
        for file in ("plot_runtime.py","plot_subsets.py","plot_matched_curves.py"):
            script=shutil.copy2(PAPER/file,paper)
            subprocess.run([sys.executable,script],check=True,cwd=stage)
        for name in names:
            for ext in ("pdf","png"):
                rel=Path("figures")/f"{name}.{ext}"
                assert (paper/rel).read_bytes()==(PAPER/rel).read_bytes(), f"Regenerated image differs: {rel}"
        for _ in range(2):
            p=subprocess.run(["xelatex","-interaction=nonstopmode","-halt-on-error","main.tex"],cwd=paper,
                             capture_output=True)
            if p.returncode:
                raise RuntimeError(p.stdout[-4000:].decode(errors="replace"))
        text=["pdftotext","-layout"]
        assert subprocess.check_output([*text,str(paper/"main.pdf"),"-"]) == subprocess.check_output([*text,str(PAPER/"main.pdf"),"-"])
        for prefix,source in (("archived",PAPER),("rebuilt",paper)):
            subprocess.run(["pdftoppm","-r","96","-png",str(source/"main.pdf"),str(stage/prefix)],check=True)
        old,new=sorted(stage.glob("archived-*.png")),sorted(stage.glob("rebuilt-*.png"))
        assert old and len(old)==len(new) and all(a.read_bytes()==b.read_bytes() for a,b in zip(old,new,strict=True))
        print(f"Rebuilt all three figure PDFs and PNGs byte for byte. Rebuilt PDF matches text and all {len(old)} pages at 96 dpi.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--render",action="store_true",help="Also rebuild three figures and the manuscript in a temporary directory.")
    cli.add_argument("--peer-run",type=Path,default=ARCHIVED_PEER,help="31-case low-dimensional ablation run to check.")
    cli.add_argument("--write-subsets",type=Path,help="Export the 135 Figure 2 points to a new file using the per-seed archives.")
    args=cli.parse_args()
    main=read(EVIDENCE/"evidence-extension/all-405.csv")
    verify_primary(main)
    verify_ablation_transfer(main,read(EVIDENCE/"evidence/transfer.csv"))
    verify_subsets(main,args.peer_run,args.write_subsets)
    if args.render:
        render()
