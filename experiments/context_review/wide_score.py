"""Verify the Git-committed pre-outcome seal, then score the six-source panel."""
import hashlib
import json
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.context_review.transport_predict import ARMS
from experiments.context_review.transport_score import fold_rows
from experiments.context_review.wide_data import EVIDENCE, ROOT, RUN, SPEC, read
from experiments.context_study.data import digest

CONTRASTS = (("slack", "point"), ("vote", "gain"), ("slack", "gain"),
             ("slack", "vote"), ("point", "gain"))
METRICS = ["u", "breach", "auc", "accuracy", "saved", "retained", "passed", "k", "anchor_k",
           "wall_s", "nomination_s", "masks", "total_fits", "guard_candidate_fits", "eligible"]


def check_commit(file):
    relative = file.relative_to(ROOT).as_posix()
    checked_in = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if file.suffix == ".json":
        assert json.loads(checked_in) == read(file), f"Uncommitted prediction lock: {relative}"
    else:
        assert hashlib.sha256(checked_in).hexdigest() == digest(file), f"Uncommitted package: {relative}"


def validate_package():
    seal_file, package = (EVIDENCE / name for name in ("prediction-lock.json", "predictions.zip"))
    check_commit(seal_file)
    check_commit(package)
    seal = read(seal_file)
    assert digest(package) == seal["package_sha256"]
    status = read(EVIDENCE / "fold-status.json")
    expected = {f"{name}/fold-{fold}" for name, spec in read(SPEC)["sources"].items()
                for fold in range(spec["folds"])}
    assert set(status) == expected and set(seal["folds"]) == {k for k in status if status[k]["feasible"]}
    assert seal["planned"] == len(status) and seal["feasible"] == len(seal["folds"])
    with ZipFile(package) as archive:
        protocol = archive.read("protocol.json")
        assert hashlib.sha256(protocol).hexdigest() == seal["protocol"]
        assert json.loads(protocol) == seal["dependencies"]
        assert set(archive.namelist()) == {"protocol.json"} | {
            f"{key}/{file}" for key, files in seal["folds"].items() for file in files}
        for key, files in seal["folds"].items():
            for file, checksum in files.items():
                assert hashlib.sha256(archive.read(f"{key}/{file}")).hexdigest() == checksum
            choice = json.loads(archive.read(f"{key}/selection.json"))
            assert choice["layout"] == files["layout.json"] == status[key]["layout"]
            assert choice["predictions"] == files["predictions.npz"] and choice["protocol"] == seal["protocol"]
    return seal


def collect(seal):
    model_pack = EVIDENCE / "predictions.zip"
    outcome_pack = EVIDENCE / "outcomes.zip"
    if not outcome_pack.exists():
        with ZipFile(outcome_pack.with_suffix(".partial"), "w", compression=ZIP_DEFLATED) as archive:
            for key in seal["folds"]:
                archive.write(RUN / key / "targets.npz", f"{key}/targets.npz")
        outcome_pack.with_suffix(".partial").replace(outcome_pack)
    rows, candidates, manifest = [], [], []
    with ZipFile(model_pack) as package, ZipFile(outcome_pack) as outcomes, TemporaryDirectory() as temp:
        base = Path(temp)
        for key in seal["folds"]:
            dest = base / key
            dest.mkdir(parents=True)
            for file in ("predictions.npz", "layout.json", "selection.json"):
                (dest / file).write_bytes(package.read(f"{key}/{file}"))
            (dest / "targets.npz").write_bytes(outcomes.read(f"{key}/targets.npz"))
            layout = read(dest / "layout.json")
            assert digest(dest / "targets.npz") == layout["targets_sha256"]
            records, pool, split = fold_rows(dest)
            with np.load(dest / "targets.npz") as targets, np.load(dest / "predictions.npz") as probs:
                rare = read(dest / "selection.json")["choices"]["gain"]["rare"]
                for entry in records:
                    entry["rare_ap_active"] = int(rare)
                    entry["positive_label"] = layout["classes"][-1] if len(layout["classes"]) == 2 else None
                    for target, name in ((entry["arm"], "ap"), ("anchor", "anchor_ap")):
                        entry[name] = average_precision_score(targets["y"], probs[target][:, 1]) if rare else None
            rows.extend(records)
            candidates.extend(pool)
            manifest.append({**split, "split_type": layout["split"], "dates": json.dumps(layout["time_ranges"])})
    return pd.DataFrame(rows), pd.DataFrame(candidates), pd.DataFrame(manifest)


def paired(df, left, right, draws):
    group = ["dataset", "fold"]
    a, b = [df[df.arm == arm].set_index(group).sort_index() for arm in (left, right)]
    assert a.index.equals(b.index)
    delta = a[METRICS] - b[METRICS]
    tasks = delta.groupby("dataset").mean()
    values = tasks.u.to_numpy()
    mean = np.mean(values)
    low, high = np.percentile(values[draws].mean(1), [2.5, 97.5])
    omitted = [(values.sum() - value) / (len(values) - 1) for value in values]
    # Any half-width U lies in [0,.5]. Sensitivity under independent task sampling only.
    radius = np.sqrt(np.log(40) / (2 * len(values)))
    result = dict(arm=left, reference=right, n_tasks=len(values), n_folds=len(a),
                  u_low=low, u_high=high, hoeffding_low=max(-.5, mean-radius),
                  hoeffding_high=min(.5, mean+radius), loo_low=min(omitted), loo_high=max(omitted),
                  changed_nominee=int((a.nominee != b.nominee).sum()),
                  changed_selected=int((a.selected != b.selected).sum()), wins=int((values > 0).sum()),
                  ties=int((values == 0).sum()), losses=int((values < 0).sum()),
                  fold_wins=int((delta.u > 0).sum()), fold_ties=int((delta.u == 0).sum()),
                  fold_losses=int((delta.u < 0).sum()), **tasks.mean().to_dict())
    tasks.insert(0, "contrast", f"{left}-{right}")
    return result, tasks.reset_index()


def score():
    seal = validate_package()
    df, pool, splits = collect(seal)
    assert len(df) == len(ARMS) * seal["feasible"] and not df.duplicated(["dataset", "fold", "arm"]).any()
    df.to_csv(EVIDENCE / "fold-results.csv", index=False)
    pool.to_csv(EVIDENCE / "candidate-scores.csv", index=False)
    splits.to_csv(EVIDENCE / "fold-manifest.csv", index=False)
    tasks = df.groupby(["dataset", "arm"])[METRICS].mean()
    tasks.to_csv(EVIDENCE / "task-results.csv")
    summary = tasks.groupby("arm").mean()
    summary.to_csv(EVIDENCE / "summary.csv")
    totals = df.groupby("arm")[["eligible", "passed", "breach", "masks", "total_fits",
                                "guard_candidate_fits", "wall_s", "nomination_s"]].sum()
    totals.to_csv(EVIDENCE / "counts.csv")
    n = df.dataset.nunique()
    assert n >= 2
    draws = np.random.default_rng(33119).integers(n, size=(30000, n))
    contrasts, details = zip(*(paired(df, left, right, draws) for left, right in CONTRASTS), strict=True)
    pd.concat(details, ignore_index=True).to_csv(EVIDENCE / "task-contrasts.csv", index=False)
    pd.DataFrame(contrasts).to_csv(EVIDENCE / "contrasts.csv", index=False)
    print("Safe saved fraction by task and policy:\n", tasks.u.unstack())
    print("Equal-task means:\n", summary.to_string(float_format=lambda v: f"{v:.5f}"))
    print("Observed resource and event totals:\n", totals)
    print("Paired U differences:\n", pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    score()
