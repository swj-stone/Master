"""Pair the measured source outcomes to actual direct Gain; write auditable detail and summaries."""
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd

from experiments.refit_voting import run

MEASURES = ["k", "seconds", "auc", "accuracy", "r2", "scaled_mae"]


def pairs(frame, baseline, round_name):
    averaged = frame.groupby(["panel", "dataset", "task", "method"], as_index=False)[MEASURES].mean()
    key = ["panel", "dataset", "task"]
    reference = averaged[averaged.method == baseline].drop(columns="method")
    merged = averaged.merge(reference, on=key, suffixes=("", "_gain"), validate="many_to_one")
    merged["study"] = round_name
    merged["time_ratio"] = merged.seconds / merged.seconds_gain
    merged["width_ratio"] = merged.k / merged.k_gain
    for score in ("auc", "accuracy", "r2", "scaled_mae"):
        merged[score+"_delta"] = merged[score] - merged[score+"_gain"]
    quality = np.where(merged.task == "classification",
                       (merged.auc_delta >= -.01) & (merged.accuracy_delta >= -.01),
                       (merged.r2_delta >= -.02) & (merged.scaled_mae_delta <= .02))
    merged["quality_pass"] = quality
    merged["narrow"] = merged.k <= merged.k_gain
    merged["faster"] = merged.time_ratio < 1.
    merged["joint"] = merged.quality_pass & merged.narrow & merged.faster
    merged["joint_5pct_speed"] = merged.quality_pass & merged.narrow & (merged.time_ratio <= .95)
    return merged


def summary(rows):
    return dict(n=len(rows), width_gm=np.exp(np.log(rows.width_ratio).mean()),
                time_gm=np.exp(np.log(rows.time_ratio).mean()), faster=int(rows.faster.sum()),
                no_wider=int(rows.narrow.sum()), quality=int(rows.quality_pass.sum()),
                joint=int(rows.joint.sum()), joint_5pct_speed=int(rows.joint_5pct_speed.sum()),
                fewer=int((rows.k < rows.k_gain).sum()),
                mean_auc_delta=rows.auc_delta.mean(), mean_accuracy_delta=rows.accuracy_delta.mean(),
                mean_r2_delta=rows.r2_delta.mean(), mean_scaled_mae_delta=rows.scaled_mae_delta.mean())


def archived_reference():
    """Keep the legacy full-development, 100-round GPU reference separately identified."""
    frame = pd.read_csv(run.ROOT / "docs/thesis/archives/piv1/final_results_by_seed.csv")
    frame = frame.loc[frame.seed.eq(run.SEED)].copy()
    with ZipFile(run.ROOT / "packages/thesis-piv1-evidence.zip") as archive:
        paths = [p for p in archive.namelist() if "PIv1-all-seeds-" in p]
        widths = []
        for name in frame.dataset:
            suffix = f"/seed-{run.SEED}/{name}/direct_gain/result.json"
            path, = [p for p in paths if p.endswith(suffix)]
            result = json.loads(archive.read(path))
            widths.append(result["retained"])
    frame["gain_k"] = widths
    cols = ["dataset", "seed", "gain_k", "gain_wall_s", "gain_test_auc", "gain_test_accuracy",
            "gain_test_r2", "gain_test_mae"]
    frame[cols].to_csv(run.DEST / "archived_gpu_direct.csv", index=False)


def main():
    archived_reference()
    ranking = pd.read_csv(run.DEST / "scores.csv")
    race = pd.read_csv(run.DEST / "race" / "scores.csv")
    data = [pairs(ranking[ranking.budget == b], "gain", "ranking-"+b) for b in ("grid", "half")]
    data.append(pairs(race, "direct", "refit-race"))
    results = pd.concat(data, ignore_index=True)
    results.to_csv(run.DEST / "paired.csv",index=False)
    records = []
    for (study,method),group in results.groupby(["study","method"]):
        records.append(dict(study=study,method=method,**summary(group)))
    pd.DataFrame(records).to_csv(run.DEST / "summary.csv",index=False)

    veto_details = []
    for panel,name in run.data_jobs():
        directory = run.DEST / "race" / f"{panel}-{name}"
        receipt = json.loads((directory / "choices.json").read_bytes())
        fold = [row for row in receipt["rows"] if row["repeat"] == 0]
        direct = next(r for r in fold if r["method"] == "direct")
        for row in fold:
            veto_details.append(dict(panel=panel, dataset=name, method=row["method"], k=row["k"],
                                     direct_k=direct["k"], vetoes=len(row["rejected"]),
                                     incorrect_veto_of_direct_winner=direct["k"] in row["rejected"],
                                     actual_rounds=row["rounds"], direct_rounds=direct["rounds"],
                                     rounds_saved=direct["rounds"]-row["rounds"]))
    pd.DataFrame(veto_details).to_csv(run.DEST / "race" / "vetoes.csv",index=False)


if __name__ == "__main__":
    main()

