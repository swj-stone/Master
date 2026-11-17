"""Produce paper tables and paired source summaries from scored and timed receipts."""
import json
import numpy as np
import pandas as pd
from experiments.minpi_rework import quota_compress_tree_confirmation as conf
from experiments.minpi_rework import refit_confirm_score, validation_panel, validation_wide
from experiments.minpi_rework.optimization_report import contrast
from experiments.minpi_compression.benchmark import ROOT, digest


PAPER=ROOT/"thesis-paper"
OUT=PAPER/"evidence/minpi_rework/stratified-tree"
SCORES=conf.DEST
DISPLAYS={"anuran":"Anuran","mice":"Mice","retina":"Retinopathy","coimbra":"Coimbra","appliances":"Appliances",
          "superconductor":"Superconductor","grid":"Grid","music":"Music","derma":"Dermatology","arrhy":"Arrhythmia",
          "robotlp":"Robot LP1","ctloc":"CT location","news":"Online News"}


def fmt(x,digits):
    return f"{x:.{digits}f}"


def groups():
    records=[]
    for panel,source,name,seed in conf.jobs():
        if name not in {"anuran","mice","ctloc"}:
            continue
        _,y,task,idx=source._partition(name,seed)
        unit=source._data(name)[3][idx["test"]]
        _,inv,n=np.unique(unit,return_inverse=True,return_counts=True)
        key=conf.OUT/f"{name}-{seed}"
        stamp=json.loads((key/"selection.json").read_bytes())
        assert digest(key/"forecasts.npz")==stamp["forecast"]
        scale=np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])
        with np.load(key/"forecasts.npz") as ar:
            for m in ("directions","gain_sweep","full"):
                for weighting,weights in (("rows",None),("subjects",1/n[inv])):
                    metrics=refit_confirm_score.score_arrays(y[idx["test"]],ar[m],task,scale,weights)
                    records.append(dict(dataset=name,seed=seed,method=m,weighting=weighting,**metrics))
    pd.DataFrame(records).to_csv(OUT/"subject-scores.csv",index=False)


def main():
    raw=pd.read_csv(SCORES/"scores.csv")
    actual=pd.read_csv(OUT/"timings.csv")
    keys=["panel","dataset","seed"]
    data=raw[raw.method=="directions"].copy()
    times=actual[actual.method=="all"].groupby(keys,as_index=False)[["search_s","pipeline_s"]].median()
    data=data.drop(columns=["search_s","pipeline_s"]).merge(times,on=keys,validate="one_to_one")
    data["method"]="stratified"
    base=raw[raw.method=="gain_sweep"].copy()
    native=pd.read_csv(SCORES/"native-costs.csv").groupby(keys,as_index=False).median(numeric_only=True)
    base=base.drop(columns=["search_s","pipeline_s"]).merge(native[keys+["compiled_s","matched_gain_pipeline_s"]],on=keys)
    base=base.rename(columns={"compiled_s":"search_s","matched_gain_pipeline_s":"pipeline_s"})
    full=raw[raw.method=="full"].drop(columns=["pipeline_s"]).merge(native[keys+["native_full_pipeline_s"]],on=keys)
    full=full.rename(columns={"native_full_pipeline_s":"pipeline_s"})
    rank=pd.read_csv(SCORES/"rank-control-scores.csv").rename(columns={"rank_and_search_s":"search_s"})
    direct=pd.read_csv(SCORES/"compiled-exact-partial.csv")
    faster=direct[direct.method=="partial"].groupby(keys,as_index=False).seconds.median()
    cheaper=data.merge(faster,on=keys)
    cheaper["pipeline_s"]=cheaper.pipeline_s-cheaper.search_s+cheaper.seconds
    cheaper["search_s"]=cheaper.seconds
    cheaper["method"]="exact_partial"
    both=pd.concat([data,base,full,rank,cheaper],ignore_index=True)
    both["selection_s"]=both.search_s
    both["deployment_s"]=both.pipeline_s
    summaries=[]
    for m in ("gain_sweep","full","treeshap","full_ablation","exact_partial"):
        summaries+=contrast(both,"stratified",m)
    pd.DataFrame(summaries).to_csv(OUT/"contrasts.csv",index=False)
    fields=["p","k","auc","accuracy","r2","mae","scaled_mae","search_s","pipeline_s"]
    means=both.groupby(["panel","dataset","method"])[fields].mean()
    means.to_csv(OUT/"source-results.csv")
    lines=[]
    display=[]
    for task in ("classification","regression"):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{"+("Classification (AUC, accuracy)" if task=="classification"
                        else "Regression ($R^2$, MAE in source units)")+r"}}\\")
        for panel,dataset in data.loc[data.task==task,["panel","dataset"]].drop_duplicates().itertuples(index=False):
            a=means.loc[panel,dataset]
            quality=("auc","accuracy") if task=="classification" else ("r2","mae")
            pair=lambda m: " / ".join(fmt(a.loc[m,col],3) for col in quality)
            row=[DISPLAYS[dataset],str(int(a.loc["full","p"])),fmt(a.loc["gain_sweep","k"],1),
                 fmt(a.loc["stratified","k"],1),pair("full"),pair("gain_sweep"),pair("stratified")]
            lines.append(" & ".join(row)+r"\\")
            display.append(row)
    (PAPER/"minpi-tree-rows.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
    groups()


if __name__=="__main__":
    main()
