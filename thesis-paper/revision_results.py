"""Assemble the revised tables and paired figure from published fold-level records."""
import argparse
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from make_tables import write_table


PAPER = Path(__file__).resolve().parent
BASE = PAPER.parent
STUDIES = BASE / "results/thesis_benchmark"
DATA = PAPER / "evidence/revision"
FILES = {
    "external": STUDIES/"prospective-pmlb-contexts-r1/per-fold.csv",
    "rfecv": STUDIES/"prospective-pmlb-rfecv-r1/per-fold.csv",
    "coded_prefix": STUDIES/"prospective-pmlb-antichain-matched-r1/per-fold.csv",
    "grouped": STUDIES/"prospective-pmlb-duplicate-groups-r1/per-fold.csv",
    "challenge": BASE/"results/context_study/nips2003/scores.csv",
    "working": BASE/"results/context_study/working-set/scores.csv",
    "schedules": BASE/"results/context_study/schedules/scores.csv",
    "random_families": BASE/"results/context_study/random-families/per-dataset.csv",
    "fixed_margins": BASE/"results/context_study/fixed-margins/per-dataset.csv",
    "matched_selectors": BASE/"results/context_study/matched-selectors/per-fold.csv",
    "exhaustive_gain": BASE/"results/context_review/exhaustive-gain/per-fold.csv",
    "tree_diagnostics": STUDIES/"context-mechanism-r1/trees/by-split.csv",
    "synthetic": STUDIES/"context-mechanism-r1/synthetic/summary.csv",
}
ARM_NAMES = {"full":"All features", "l1":"L1 logistic support", "rfecv":"RFECV",
             "exhaustive_gain":"Exhaustive direct Gain", "gain":"Compact Gain anchor",
             "point":"Empty, one panel", "empty_majority":"Empty, four panels", "all_point":"Nested, one panel",
             "pooled":"Nested, pooled", "min":"Nested minimum", "antichain":"Separating minimum",
             "random_contexts":"Matched random masks", "matched_gain":"Prefix at nested width",
             "coded_prefix":"Prefix at separating width", "rfe_same_k":"RFE at separating width",
             "anova_same_k":"ANOVA at separating width"}

DONATED = {"matched_gain", "coded_prefix", "rfe_same_k", "anova_same_k"}


def grid(data, names, replicates, arms, rep):
    expected={(n,r,a) for n in names for r in replicates for a in arms}
    actual=set(data[["dataset",rep,"arm"]].itertuples(index=False,name=None))
    if actual!=expected or len(actual)!=len(data):
        raise ValueError(f"Incomplete or repeated folds: missing {list(expected-actual)[:6]}")


def external():
    main=pd.read_csv(DATA/"external.csv")
    rf=pd.read_csv(DATA/"rfecv.csv")
    rf["arm"]="rfecv"
    datasets=json.loads((BASE/"configs/context_study/prospective.json").read_bytes())["datasets"]
    design=list(ARM_NAMES)
    extra={"rfecv","exhaustive_gain","coded_prefix","rfe_same_k","anova_same_k"}
    grid(main,datasets,range(3),[a for a in design if a not in extra],"fold")
    grid(rf,datasets,range(3),["rfecv"],"fold")
    exhaustive=pd.read_csv(DATA/"exhaustive_gain.csv").assign(arm="exhaustive_gain")
    grid(exhaustive,datasets,range(3),["exhaustive_gain"],"fold")
    frames=[main,rf,exhaustive]
    matched=pd.read_csv(DATA/"coded_prefix.csv")
    matched["arm"]="coded_prefix"
    grid(matched,datasets,range(3),["coded_prefix"],"fold")
    frames.append(matched)
    others=pd.read_csv(DATA/"matched_selectors.csv")
    grid(others,datasets,range(3),["rfe_same_k","anova_same_k"],"fold")
    donors=main[main.arm=="antichain"].set_index(["dataset","fold"]).k
    for _, rows_at_width in pd.concat([others,matched]).groupby("arm"):
        if not rows_at_width.set_index(["dataset","fold"]).k.sort_index().equals(donors.sort_index()):
            raise ValueError("A conditional selector received a different width")
    frames.append(others)
    data=pd.concat(frames,ignore_index=True)
    data["kept_pct"]=100*data.k/data.width
    columns=["auc","accuracy","kept_pct","total_s"]
    means=data.groupby(["dataset","arm"])[columns].mean().reset_index()
    means.to_csv(DATA/"external-by-dataset.csv",index=False)
    summary=means.groupby("arm")[columns].mean()
    rows=[]
    for key in design:
        if key in summary.index:
            a=summary.loc[key]
            rows.append([ARM_NAMES[key], f"{a.auc:.4f}", f"{100*a.accuracy:.2f}",
                         f"{a.kept_pct:.1f}", "---" if key in DONATED else f"{a.total_s:.2f}"])
    write_table("external","Fourteen external classification tasks. The three outer folds are averaged "
                "within each task, then tasks receive equal weight. Seconds include selector and final predictor "
                "costs within a process. Conditional selectors receive a width from another method, "
                "so a standalone runtime is unavailable.",
                ["Method","AUC",r"Acc. \%",r"Kept \%","Seconds"],rows)
    auc=means.pivot(index="dataset",columns="arm",values="auc")
    effects=[]
    for comparator in ["min","random_contexts"]:
        delta=auc["antichain"]-auc[comparator]
        pos=(delta>1e-12).sum()
        neg=(delta< -1e-12).sum()
        p=binomtest(int(pos),int(pos+neg)).pvalue if pos+neg else 1.
        effects.append(dict(comparator=comparator,mean=delta.mean(),median=delta.median(),
                            low=delta.min(),high=delta.max(),wins=pos,ties=len(delta)-pos-neg,losses=neg,p=p))
    first,second=sorted(effects,key=lambda a:a["p"])
    first["holm_p"]=min(1.,2*first["p"])
    second["holm_p"]=min(1.,max(first["holm_p"],second["p"]))
    pd.DataFrame(effects).to_csv(DATA/"external-contrasts.csv",index=False)
    figure(main,datasets)


def figure(data,datasets):
    fig,axes=plt.subplots(1,2,figsize=(6.1,4.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.23,right=.99,top=.91,bottom=.12,wspace=.13)
    pivot=data.pivot(index=["dataset","fold"],columns="arm",values="auc")
    labels = {"mfeat_zernike":"Mfeat-Zernike","movement_libras":"Libras","new_thyroid":"New thyroid",
              "wine_quality_red":"Red wine","page_blocks":"Page blocks"}
    for ax,arm,title in zip(axes,["min","random_contexts"],["Versus nested masks","Versus matched random masks"],strict=True):
        ax.axvline(0,linewidth=.8,color="0.48",ls="--")
        for i,name in enumerate(datasets):
            effect=100*(pivot.loc[name,"antichain"]-pivot.loc[name,arm])
            ax.hlines(i,effect.min(),effect.max(),color="#7999a3",linewidth=1.45)
            ax.scatter(effect.mean(),i,s=21,color="#00586e",zorder=3)
        ax.set_title(title,fontsize=9)
        ax.tick_params(labelsize=8,length=2)
        ax.tick_params(axis="y",length=0)
        ax.spines[["top","left","right"]].set_visible(False)
    axes[0].set_yticks(range(len(datasets)),[labels.get(n,n.capitalize()) for n in datasets])
    axes[0].invert_yaxis()
    fig.supxlabel("Difference in outer-fold AUC × 100",fontsize=9,y=.025)
    (PAPER/"figures").mkdir(exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype":42}):
        fig.savefig(PAPER/"figures/external-pairs.pdf",metadata={"Title":"Paired outer-fold AUC differences"})
    plt.close(fig)


def challenge():
    data=pd.read_csv(DATA/"challenge.csv")
    methods={"raw_full":"All raw features","screen_full":"All screened features","l1":"L1 logistic support",
             "anchor":"Coefficient anchor","point":"Empty, one panel","nested":"Nested minimum",
             "separating":"Separating minimum","random_matched":"Matched random masks",
             "prefix_at_separating_k":"Prefix at separating width"}
    tasks=["arcene","dexter","dorothea"]
    grid(data,tasks,[42,123,456,789,1024],methods,"seed")
    means=data.groupby(["arm","dataset"])[["k","ber","auc","accuracy"]].mean()
    means.to_csv(DATA/"challenge-by-dataset.csv")
    rows=[]
    for key,label in methods.items():
        row=[label]
        for task in tasks:
            avg=means.loc[(key,task)]
            row.extend([f"{avg.k:g}", f"{100*avg.ber:.1f}"])
        rows.append(row)
    write_table("challenge","The three previously unused NIPS feature-selection challenge datasets, "
                "evaluated on their public validation samples. For each dataset the first entry is "
                "the mean number of features and the second is balanced error rate in percent "
                "(lower is better). The five inner splits share a validation sample.",
                ["Method","Arcene $k$","BER","Dexter $k$","BER","Dorothea $k$","BER"],rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true",help="Copy raw run summaries into paper evidence first")
    args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if args.collect:
        for name,source in FILES.items():
            copyfile(source,DATA/f"{name}.csv")
    external()
    challenge()
    print((DATA/"external-contrasts.csv").read_text(encoding="utf8"))


if __name__=="__main__":
    main()
