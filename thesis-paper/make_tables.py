"""Build the manuscript tables from complete seed-level evidence, without model fitting."""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
TABLES = HERE / "tables"
DATASETS = {"abalone": "Abalone", "spambase": "Spambase", "madelon": "Madelon",
            "credit_card_fraud": "Fraud", "gisette": "Gisette", "isolet": "ISOLET"}
SEEDS = {42, 123, 456, 789, 1024}
NAMES = {"min": "MinPI", "mean": "Pooled votes", "point": "One panel",
         "gain": "Compact Gain", "minpi": "Five-block MinPI", "matched_gain": "Matched prefix",
         "full": "Full", "pfi": "Permutation", "sampling_shap": "SamplingSHAP",
         "minshap": "MinShap", "banzhaf": "Banzhaf", "shapg": "ShapG",
         "tree_shap_top_k": "TreeSHAP", "nsga2": "NSGA-II"}


def write_table(key, caption, headings, rows):
    TABLES.mkdir(exist_ok=True)
    lines = [r"\begin{table}[tb]", r"\centering", r"\caption{" + caption + "}",
             r"\label{tab:" + key + "}", r"\small",
             r"\begin{tabular}{@{}l" + "r" * (len(headings)-1) + "@{}}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    for row in rows:
        if row is None:
            lines.append(r"\midrule")
        else:
            if len(row) != len(headings):
                raise ValueError(f"{key}: unexpected row width")
            lines.append(" & ".join(map(str, row)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / (key + ".tex")).write_text("\n".join(lines)+"\n", encoding="utf-8")


def check_grid(frame, methods, seeds=SEEDS):
    required = {(d, s, m) for d in DATASETS for s in seeds for m in methods}
    keys = list(frame[["dataset", "seed", "arm"]].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys) or set(keys) != required or not frame.status.eq("completed").all():
        raise ValueError(f"Incomplete or duplicated benchmark grid: {methods}")
    if not frame.wall_s.between(0, 3600, inclusive="neither").all():
        raise ValueError("Unusable timing in the final grid")
    if (frame.k > frame.width).any() or (frame.k < 0).any():
        raise ValueError("Invalid feature count")


def transfer():
    data = pd.read_csv(HERE / "transfer/per-seed.csv")
    methods = ["anchor", "min", "mean", "point", "matched_prefix", "full"]
    expected = {(d,s,m) for d in ("breast_cancer","wine","diabetes")
                for s in range(20271,20276) for m in methods}
    if set(data[["dataset","seed","arm"]].itertuples(index=False,name=None)) != expected or len(data)!=90:
        raise ValueError("Incomplete external evaluation")
    rows = []
    for name, label in (("breast_cancer", "Breast cancer"),("wine", "Wine"),("diabetes","Diabetes")):
        ds = data[data.dataset == name]
        metric = "mae" if name == "diabetes" else "auc"
        avg = ds.groupby("arm").mean(numeric_only=True)
        f = ".2f" if metric == "mae" else ".4f"
        rows.append([label, f"{int(ds.width.iloc[0])}", f"{avg.loc['min','retained']:.1f}",
                     *[format(avg.loc[a,metric],f) for a in ["min","mean","point","matched_prefix","anchor","full"]]])
    write_table("transfer", "Linear predictors on three external data sets, averaged over five "
                "prespecified splits. Test scores are AUC for the two classifiers and MAE for "
                r"Diabetes (lower is better). $p$ is the input width and $k$ is MinPI's retained width. "
                "Prefix receives the realized MinPI width.", ["Data set", "$p$", "$k$", "MinPI",
                "Pooled", "Point", "Prefix", "Anchor", "Full"], rows)


def primary():
    data = pd.read_csv(EVIDENCE / "primary.csv")
    controls = pd.read_csv(EVIDENCE / "controls.csv")
    baseline = pd.read_csv(EVIDENCE / "historical.csv").rename(columns={"method":"arm"})
    regular, eager = data[data.arm != "eager"], data[data.arm == "eager"]
    check_grid(regular, ["min","mean","point"])
    check_grid(eager, ["eager"], {42})
    check_grid(controls, ["gain","minpi","matched_gain"])
    check_grid(baseline, ["full","gain","pfi","sampling_shap","minshap","banzhaf","shapg","tree_shap_top_k","nsga2"])
    rows = []
    for source, methods in ((pd.concat([regular,controls]),["min","mean","point","gain","minpi"]),
                            (baseline,["full","pfi","tree_shap_top_k","sampling_shap","nsga2","banzhaf","shapg","minshap"])):
        if rows:
            rows.append(None)
        for arm in methods:
            ds = source[source.arm == arm]
            rows.append([NAMES[arm], f"{100*(ds.k/ds.width).mean():.1f}", f"{ds.wall_s.mean():.2f}",
                         f"{ds.fits.mean():.1f}", f"{ds.auc.mean():.4f}", f"{ds.accuracy.mean():.4f}",
                         f"{ds.ap.mean():.4f}", f"{ds.mae.mean():.3f}"])
    write_table("baselines", "Five splits of each of six tasks. The upper block uses the current "
                "study runs and its controls. The lower block uses archived runs on the same "
                "splits. Sec. is mean cold elapsed time. AUC and accuracy average the five "
                "classification tasks. AP and MAE concern Fraud and Abalone, respectively.",
                ["Method", r"Kept \%", "Sec.", "Fits", "AUC", "Acc.", "AP", "MAE"], rows)
    rows = []
    for name,label in DATASETS.items():
        x = pd.concat([regular,controls])
        ds = x[x.dataset == name].set_index(["seed","arm"]).sort_index()
        metric = "mae" if name == "abalone" else "ap" if name == "credit_card_fraud" else "auc"
        f = ".3f" if metric == "mae" else ".4f"
        avg = ds.groupby("arm").mean(numeric_only=True)
        delta = ds.xs("min",level="arm")[metric] - ds.xs("matched_gain",level="arm")[metric]
        rows.append([label, f"{avg.loc['gain','k']:.1f}", f"{avg.loc['min','k']:.1f}",
                     *[format(avg.loc[a,metric],f) for a in ["min","gain","mean","point"]],
                     format(delta.mean(),"+"+f)])
    write_table("datasets", r"Test scores and average columns before and after MinPI. "
                r"Abalone reports MAE ($\downarrow$) and Fraud reports AP ($\uparrow$). The "
                r"remaining rows report AUC ($\uparrow$). $\Delta_k$ is MinPI minus a Gain "
                r"prefix of exactly its selected size, paired within each split.",
                ["Data set", "$k_A$", "$k$", "MinPI", "Anchor", "Pooled", "Point", r"$\Delta_k$"], rows)
    rows = []
    lazy = regular[(regular.arm=="min") & (regular.seed==42)].set_index("dataset")
    direct = eager.set_index("dataset")
    for name, label in DATASETS.items():
        a,b = lazy.loc[name], direct.loc[name]
        if a.k != b.k or a.proposals != b.proposals:
            raise ValueError("The eager and short-circuit trajectories disagree")
        rows.append([label, f"{int(a.proposals)}", f"{int(a.witness_predictions)}", f"{int(b.witness_predictions)}",
                     f"{a.selector_s:.3f}", f"{b.selector_s:.3f}", f"{a.wall_s:.2f}", f"{b.wall_s:.2f}"])
    write_table("mechanism", "Short-circuit and exhaustive evaluation on split 42. The same "
                "proposal path and final features are required in each pair. Calls count distinct "
                "witness masks, excluding anchor construction. Selection and cold times are in "
                "seconds. Each time is one measurement.", ["Data set","Trials","Calls S","Calls E",
                "Sel. S","Sel. E","Cold S","Cold E"],rows)
    paired_figure(regular[regular.arm=="min"], controls[controls.arm=="gain"])


def paired_figure(method, gain):
    """Retain the five paired observations even when scores or widths coincide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    keys=["dataset","seed"]
    data=method.merge(gain,on=keys,suffixes=("_min","_gain"),validate="one_to_one")
    fig,(left,right)=plt.subplots(1,2,figsize=(6.0,2.67),sharey=True)
    fig.subplots_adjust(left=.24,right=.98,top=.86,bottom=.21,wspace=.23)
    for i,name in enumerate(DATASETS):
        group=data[data.dataset==name].sort_values("seed")
        metric="r2" if name=="abalone" else "ap" if name=="credit_card_fraud" else "auc"
        gain_removed=100*(group.k_gain-group.k_min)/group.width_min
        effect=100*(group[metric+"_min"]-group[metric+"_gain"])
        for axis,values in [(left,gain_removed),(right,effect)]:
            offsets=[-.17,-.085,0,.085,.17]
            axis.hlines(i,values.min(),values.max(),color="#79939e",lw=1.1,zorder=1)
            axis.scatter(values,[i+j for j in offsets],s=12,color="#075577",zorder=3,lw=0)
    labels=[r"Abalone ($R^2$)","Spambase","Madelon","Fraud (AP)","Gisette","ISOLET"]
    left.set_yticks(range(len(DATASETS)),labels)
    left.invert_yaxis()
    left.set_title("(a) Additional pruning",loc="left",fontsize=10)
    right.set_title("(b) Held-out score change",loc="left",fontsize=10)
    left.set_xlabel("Percentage points of input width",fontsize=8)
    right.set_xlabel("Score difference × 100",fontsize=8)
    for axis in (left,right):
        axis.axvline(0,color="#3b3b3b",lw=.7,ls="--",zorder=0)
        axis.tick_params(axis="both",labelsize=8,length=2)
        axis.spines[["top","right","left"]].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    out=HERE/"figures"
    out.mkdir(exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype":42}):
        fig.savefig(out/"paired.pdf",metadata={"Title":"Paired differences from the compact Gain anchor"})
    plt.close(fig)


def width_table():
    data = pd.read_csv(EVIDENCE / "width.csv").set_index(["dataset", "arm"])
    datasets = {"madelon": 500, "gisette": 5000, "isolet": 617}
    arms = ["min", "gain", "full"]
    expected = {(name, arm) for name in datasets for arm in arms}
    if set(data.index) != expected or len(data) != len(expected) or not data.seed.eq(42).all():
        raise ValueError("The original-width stress test requires all nine outcomes")
    rows = []
    for name, width in datasets.items():
        group = data.loc[name]
        if not group.width.eq(width).all():
            raise ValueError(f"Unexpected stress-test width for {name}")
        scores, times, counts = [], [], []
        for arm in arms:
            row = group.loc[arm]
            complete = row.status == "completed"
            scores.append(f"{row.auc:.4f}" if complete else r"---")
            times.append(f"{row.wall_s:.1f}" if complete else r"---")
            counts.append(str(int(row.k)) if complete else r"---")
        rows.append([DATASETS[name], str(width), counts[0], counts[1], *scores, "/".join(times)])
    write_table("width", "Original-width inputs at seed 42, using exactly the same rows as "
                "the corresponding prepared splits. Test AUCs and cold times use a fresh fit for each "
                "arm. Times in the final column are seconds for MinPI, Gain and Full, in that order.",
                ["Data set", "$p$", "$k_{M}$", "$k_{G}$", "MinPI", "Gain", "Full", "Cold sec."], rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-only",action="store_true")
    args=parser.parse_args()
    transfer()
    if not args.transfer_only:
        primary()
        if (EVIDENCE / "width.csv").is_file():
            width_table()
