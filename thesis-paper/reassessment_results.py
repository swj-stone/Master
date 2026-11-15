"""Build proposal/refit tables and the per-task veto figure from portable evidence."""
import argparse
from pathlib import Path
from shutil import copyfile

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from make_tables import write_table

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RUN = ROOT/"results/context_review"
EVIDENCE = PAPER/"evidence/reassessment"
INPUTS = {"decisions":RUN/"refit-target/decisions.csv", "exposure":RUN/"refit-target/exposure.csv",
          "refit_guard":RUN/"refit-guard/per-fold.csv","challenger":RUN/"challenger/per-fold.csv",
          "eligibility":RUN/"refit-guard/eligibility.csv","challenger_costs":RUN/"challenger/costs.csv",
          "factorial":RUN/"factorial/factorial-summary.csv","har":RUN/"external-har-locked-r1/scores.csv",
          "har_by_subject":RUN/"external-har-locked-r1/by-subject.csv",
          "har_anchor_random":RUN/"external-har-random-columns/scores.csv",
          "timing":RUN/"timing-process/paired-comparisons.csv","timing_counts":RUN/"timing-process/one-pass-counts.csv",
          "drift":RUN/"drift-locked/summary.csv","drift_batches":RUN/"drift-locked/by-batch.csv",
          "har_history":RUN/"matched-history/har-summary.csv","gas_history":RUN/"matched-history/gas-summary.csv",
          "refit_grid":RUN/"refit-vote-grid/per-fold.csv","refit_grid_costs":RUN/"refit-vote-grid/costs.csv"}


def refits():
    decision = pd.read_csv(EVIDENCE/"decisions.csv")
    assert not decision.duplicated(["dataset","fold","cohort","subset"]).any()
    exposure = pd.read_csv(EVIDENCE/"exposure.csv")
    assert len(exposure)==42 and len(decision.drop_duplicates(["dataset","fold","subset"]))==exposure.unique.sum()
    rules={"point":"Empty, point panel","empty":"Empty, four panels","endpoints":"Both endpoints",
           "nested":"Nested minimum","one_panel_min":"Separating, point panel","pooled":"Separating, pooled votes",
           "separating":"Separating minimum","random":"Size-matched random minimum"}
    rows=[]
    for rule,label in rules.items():
        cells=[label]
        for name in ("path","uniform_deletions"):
            group=decision[decision.cohort==name]
            assert len(group)==exposure[name].sum()
            y,v=group.fail,group[rule]
            counts=[(y&v).sum(),(~y&v).sum(),(y&~v).sum(),(~y&~v).sum()]
            cells.extend(str(x) for x in counts)
        rows.append(cells)
    lines=[r"\begin{table}[tb]",r"\centering",
           r"\caption{Deletion vetoes against the observed loss after refitting candidate and anchor. "
           r"A positive outcome exceeds the joint tolerance on the outer test rows. "
           r"Counts weight each sampled proposal equally. The two sources share 75 fold--support pairs.}",
           r"\label{tab:refits}",r"\small\setlength{\tabcolsep}{2.5pt}",r"\begin{tabular}{@{}lrrrr@{\quad}rrrr@{}}",r"\toprule",
           r" & \multicolumn{4}{c}{Search path ($n=400$)} & \multicolumn{4}{c}{Drawn ($n=379$)} \\",
           r" & TP & FP & FN & TN & TP & FP & FN & TN \\",r"\midrule"]
    lines.extend(" & ".join(r)+r" \\" for r in rows)
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/refit-decision.tex").write_text("\n".join(lines)+"\n",encoding="utf8")


def adjudication():
    status=pd.read_csv(EVIDENCE/"eligibility.csv")
    assert len(status)==42 and status.eligible.sum()==41
    expected=set(status[status.eligible][["dataset","fold"]].itertuples(index=False,name=None))
    df=pd.concat([pd.read_csv(EVIDENCE/f"{file}.csv") for file in ("refit_guard","challenger")])
    for _,group in df.groupby("arm"):
        assert len(group)==41 and set(group[["dataset","fold"]].itertuples(index=False,name=None))==expected
    means=df.groupby(["dataset","arm"])[["auc","accuracy","retained"]].mean().groupby("arm").mean()
    exceeds=df.groupby("arm").exceeds.sum()
    labels={"anchor":"Compact Gain anchor","vote":"Vote","vote_guard":"Vote, refit guard",
            "same_width_guard":"Same-width prefix, guard","half_guard":"Half-width prefix, guard",
            "maximin_guard":"Maximin challenger, guard","weak_gain_guard":"Weakest Gain deletion, guard"}
    rows=[]
    for arm,label in labels.items():
        data=means.loc[arm]
        rows.append([label,f"{data.auc:.4f}",f"{100*data.accuracy:.2f}",f"{100*data.retained:.1f}",
                     f"{int(exceeds.loc[arm])}/41"])
    write_table("heldout-adjudication","Refit checks on a separate inner partition. "
                "AUC, accuracy and retained fraction average available folds within each of 14 tasks, "
                "then weight tasks equally. The last column counts outer folds exceeding the loss "
                "tolerance relative to the refitted anchor. One red-wine fold had an indivisible ranking class.",
                ["Method","AUC",r"Acc. \%",r"Kept \%","Exceedances"],rows)
    path=PAPER/"tables/heldout-adjudication.tex"
    path.write_text(path.read_text(encoding="utf8").replace("tab:heldout-adjudication","tab:adjudication"),
                    encoding="utf8")


def incremental_vetoes():
    frame=pd.read_csv(EVIDENCE/"decisions.csv")
    assert not (frame.endpoints & ~frame.separating).any()
    frame["caught"]=frame.separating & ~frame.endpoints & frame.fail
    frame["false_alarm"]=frame.separating & ~frame.endpoints & ~frame.fail
    counts=frame.groupby(["dataset","cohort"])[["caught","false_alarm"]].sum()
    counts.to_csv(EVIDENCE/"incremental-vetoes.csv")
    names=sorted(frame.dataset.unique())
    labels={"mfeat_zernike":"Mfeat-Zernike","movement_libras":"Libras","new_thyroid":"New thyroid",
            "page_blocks":"Page blocks","wine_quality_red":"Red wine"}
    colors=("#ac5427","#15646c")
    fig,axes=plt.subplots(1,2,figsize=(6.1,3.75),sharey=True)
    fig.subplots_adjust(left=.21,right=.985,top=.88,bottom=.17,wspace=.15)
    for ax,cohort,title in zip(axes,["path","uniform_deletions"],
                                ["Search-path proposals","Drawn deletions"],strict=True):
        sub=counts.xs(cohort,level="cohort").loc[names]
        y=range(len(names))
        ax.barh(y,-sub.false_alarm,color=colors[0],height=.5)
        ax.barh(y,sub.caught,color=colors[1],height=.5)
        ax.axvline(0,color=".3",lw=.7)
        ax.set(xlim=(-4.65,7.65),xticks=[-4,-2,0,2,4,6],xticklabels=["4","2","0","2","4","6"])
        ax.tick_params(labelsize=8,length=2)
        ax.tick_params(axis="y",length=0)
        ax.set_title(title,fontsize=9)
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].set_yticks(range(len(names)),[labels.get(n,n.capitalize()) for n in names])
    axes[0].invert_yaxis()
    fig.legend([Patch(color=c) for c in colors],["Additional false alarms (left)","Additional losses caught (right)"],
               ncol=2,loc="lower center",frameon=False,fontsize=8)
    with matplotlib.rc_context({"pdf.fonttype":42}):
        fig.savefig(PAPER/"figures/refit-vetoes.pdf",metadata={"Title":"Incremental vetoes against refitted outcomes"})
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collect",action="store_true",help="Refresh inputs from the audited working-tree runs")
    args=p.parse_args()
    EVIDENCE.mkdir(exist_ok=True,parents=True)
    (PAPER/"tables").mkdir(exist_ok=True)
    (PAPER/"figures").mkdir(exist_ok=True)
    if args.collect:
        for key,path in INPUTS.items():
            copyfile(path,EVIDENCE/f"{key}.csv")
    refits()
    adjudication()
    incremental_vetoes()
