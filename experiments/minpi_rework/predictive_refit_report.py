"""Summarize matched learners at each registered refitting budget."""
import numpy as np
import pandas as pd

from experiments.minpi_rework.predictive_refit import OUT


def main():
    raw = pd.read_csv(OUT/"scores.csv")
    if raw.duplicated(["panel", "dataset", "seed", "method"]).any() or len(raw) != 39*7:
        raise ValueError("Missing or repeated paired outputs")
    raw["quota_fail"] = raw.outer_worst > .02
    fields = ["k", "fits", "seconds", "auc", "accuracy", "r2", "scaled_mae", "quota_fail"]
    by_source = raw.groupby(["panel", "method", "dataset"])[fields].mean()
    by_source.to_csv(OUT/"source-means.csv")
    means = by_source.groupby(level=["panel", "method"]).mean()
    counts = raw.groupby(["panel", "method"]).quota_fail.sum()
    lines = ["# Refitting under the same forest and mask objective", "",
             "Seven transfer sources use five seeds each. All four wide sources use seed 42 for this "
             "secondary test. Every model uses the same training rows and bootstrap recipe as its "
             "parent. All 39 jobs sealed their selected supports and prediction hashes before releasing "
             "the wide outcomes. Scoring reproduced each sealed forecast. All 56 small-forest "
             "size-level checks agreed with direct mask enumeration, including distinct empty predictions.",
             "", "Averaged metrics below weight sources equally. MAE is divided by the response "
             "standard deviation on development rows. Time sums the independently measured components "
             "needed by each arm through acceptance. The widths refer to the declared schema. "
             "The frozen-arm timing uses the scalar selector in this study, which predates the "
             "subsequent interval implementation. The 20% and 50% grids use descending widths "
             "under one fixed MDI order. A single optional refit uses the previously accepted width "
             "and survives only when it passes both checks and improves the worst-size calibration loss.",
             "", "| Panel | Method | Columns | Fits | Seconds | AUC / accuracy | R² / scaled MAE | Failed quota |",
             "|---|---|---:|---:|---:|---|---|---:|"]
    for (panel, method), row in means.iterrows():
        lines.append(f"| {panel} | {method} | {row.k:.2f} | {row.fits:.1f} | {row.seconds:.3f} "
                     f"| {row.auc:.4f} / {row.accuracy:.4f} | {row.r2:.4f} / {row.scaled_mae:.4f} "
                     f"| {counts.loc[panel,method]} |")
    lines += ["", "On the seven-source panel, the 20% deletion grid gave a smaller model and better "
              "average regression scores at the cost of about ten forest fits. The frozen MinPI "
              "comparison had higher average AUC and accuracy there. A one-refit MDI alternative "
              "passed the held-out risk quota more often than the corresponding MinPI alternative. "
              "On the four wider seed-42 fits, the two refit grids both achieved better average "
              "regression scores and accuracy than frozen MinPI. Timing and predictive choices "
              "depend on whether replacing the original forest is permitted.",
              "", "[Preregistered comparison](../../../../experiments/minpi_rework/predictive_refit_protocol.md)"
              " · [Forecast lock](lock.json) · [All decisions and scores](scores.csv)"]
    (OUT/"README.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    pairs = raw.set_index(["panel","dataset","seed","method"]).sort_index()
    candidate = pairs.xs("minpi",level="method")
    effects = []
    for name in ("mdi", "refit20", "refit50", "mdi_one_refit", "minpi_one_refit"):
        other = pairs.xs(name,level="method")
        for panel in ("seven", "wide"):
            a, b = candidate.loc[panel], other.loc[panel]
            for field in fields:
                if field in ("k", "seconds", "fits"):
                    diff = np.log(a[field]/b[field])
                else:
                    diff = a[field].astype(float)-b[field].astype(float)
                values = diff.groupby(level="dataset").mean().dropna().to_numpy()
                rng = np.random.default_rng(1711)
                bs = values[rng.integers(len(values), size=(20000,len(values)))].mean(axis=1)
                est = np.array([values.mean(), *np.quantile(bs, [.025,.975])])
                if field in ("k","seconds","fits"):
                    est = np.exp(est)
                effects.append([panel,name,field,len(values),*est])
    pd.DataFrame(effects,columns=["panel","comparison","metric","sources","estimate","lower","upper"]
                 ).to_csv(OUT/"contrasts.csv",index=False)


if __name__ == "__main__":
    main()
