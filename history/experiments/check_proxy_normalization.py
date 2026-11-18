"""Audit corrected proxy normalization on every archived development and transfer panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_benchmark as dev
from experiments import adaptive_minpi_transfer as transfer
from experiments import budget_minpi_deployments as deployment
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction
from minppi_mofs.pilot_minpi import pilot_top_from_training
from minppi_mofs.proxy_minpi import _unit_columns, proxy_top_from_training
from minppi_mofs.sketch_minpi import reduced_width


RESULTS = ROOT / "results/minpi-study"
ARCHIVES = {"development": RESULTS/"adaptive-69-20261117a",
            "transfer": RESULTS/"adaptive-transfer-20261117a"}


def invalid_edges(X, folds, parent):
    """Check the empirical proxy criterion, without inferring causal interchangeability."""
    children = np.flatnonzero(parent != np.arange(len(parent)))
    if not len(children):
        return 0
    anchors = parent[children]
    cols, remap = np.unique(np.r_[children, anchors], return_inverse=True)
    left, right = np.split(remap, 2)
    valid = np.ones(len(left), dtype=bool)
    correlations = []
    for key in ("fit_rows", "held_rows"):
        z = _unit_columns(X[np.ix_(folds[0][key], cols)])
        corr = np.einsum("ij,ij->j", z[:, left], z[:, right])
        fourth = np.einsum("ij,ij->j", z*z, z*z)
        valid &= (np.abs(corr) >= .9) & (fourth[left] <= .25) & (fourth[right] <= .25)
        correlations.append(corr)
    return int(np.sum(~valid | (correlations[0]*correlations[1] <= 0)))


def evaluate(part, panel, cols):
    if part == "development":
        pred, decisions = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols,
                                                panel.task, panel.seed)
        score, _, _ = deployment.score_predictions(panel.name, panel.gold, pred, panel.units,
                                                   panel.task, decisions)
    else:
        pred = prediction(panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
        score = metric(panel.truth, pred, panel.task)
    return pred, score


def compare(part, panel, run, old_scores, out, verify):
    stem = f"{panel.name}-{panel.fold}-{panel.seed}"
    archive_path = run/(stem + ".npz")
    revised_path = out/(part + "-" + stem + ".npz")
    rows, arrays = [], {}
    ordered = panel.name in ("gas", "secom") if part == "development" else panel.ordered
    params = dict(task=panel.task, random_state=panel.seed, groups=panel.groups, ordered_groups=ordered)
    with np.load(archive_path) as old:
        if part == "transfer":
            np.testing.assert_array_equal(old["input_hashes"], [digest(a) for a in (
                panel.X, panel.y, panel.Xt, panel.truth)])
        for name in ("pilot_auto" if part == "development" else "pilot", "pilot_75"):
            if name+"_cols" not in old:
                continue
            count = reduced_width(panel.X.shape[1], .75) if name.endswith("75") else None
            original, prior = pilot_top_from_training(panel.X, panel.y, count, **params)
            np.testing.assert_array_equal(original, old[name+"_cols"])
            if name+"_diagnostics" in old:
                saved = json.loads(old[name+"_diagnostics"].item())
                np.testing.assert_array_equal(prior["parent"], saved["parent"])
            cols, fixed = proxy_top_from_training(panel.X, panel.y, count, **params)
            for role in ("fit_rows", "held_rows"):
                np.testing.assert_array_equal(prior["folds"][0][role], fixed["folds"][0][role])
            pred, score = evaluate(part, panel, cols)
            reference = old[name+"_pred"]
            previous_score = old_scores.loc[(panel.name, panel.fold, panel.seed, name), "score"]
            full_score = old_scores.loc[(panel.name, panel.fold, panel.seed, "full"), "score"]
            arrays[name+"_cols"], arrays[name+"_pred"], arrays[name+"_parent"] = cols, pred, fixed["parent"]
            invalid_old = invalid_edges(panel.X, prior["folds"], prior["parent"])
            invalid_new = invalid_edges(panel.X, fixed["folds"], fixed["parent"])
            assert invalid_new == 0
            fields = dict(part=part, dataset=panel.name, fold=panel.fold, seed=panel.seed, method=name,
                          p=panel.X.shape[1], old_width=len(original), width=len(cols),
                          support_changed=not np.array_equal(original, cols),
                          changed_predictions=int(np.count_nonzero(reference != pred)),
                          max_prediction_change=float(np.max(np.abs(reference - pred))),
                          old_score=previous_score, score=score, score_delta=score-previous_score,
                          full_score=full_score, reduced_within=len(cols)<panel.X.shape[1] and score-full_score>=-.02,
                          old_groups=prior["groups_found"], groups=fixed["groups_found"],
                          changed_parent=int(np.count_nonzero(prior["parent"] != fixed["parent"])),
                          old_invalid_edges=invalid_old, invalid_edges=invalid_new,
                          support_sha256=digest(cols), prediction_sha256=digest(pred))
            rows.append(fields)
    if verify:
        with np.load(revised_path) as saved:
            assert set(saved) == set(arrays)
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, saved[key])
    else:
        np.savez_compressed(revised_path, **arrays)
    return rows


def run(out, verify):
    files = [Path(__file__), ROOT/"src/minppi_mofs/proxy_minpi.py",
             ROOT/"src/minppi_mofs/pilot_minpi.py", ROOT/"src/minppi_mofs/moment_minpi.py",
             ROOT/"src/minppi_mofs/sketch_minpi.py", ROOT/"src/minppi_mofs/utility_minpi.py",
             ROOT/"experiments/adaptive_minpi_benchmark.py", ROOT/"experiments/adaptive_minpi_transfer.py",
             ROOT/"experiments/moment_minpi_benchmark.py", ROOT/"experiments/budget_minpi_deployments.py",
             ROOT/"experiments/minpi_study.py", ROOT/"experiments/stable_minpi_stress.py"]
    for run_path in ARCHIVES.values():
        files.extend([run_path/"cases.csv", run_path/"protocol.json", *sorted(run_path.glob("*.npz"))])
    hashes = {f.relative_to(ROOT).as_posix(): file_hash(f) for f in files}
    spec = dict(inputs=hashes, timings=False, dev=69, transfer=57, configurations_arms=207,
                invalid_edge="Fails abs(r)>=.90, matching sign or participation >=4 on either "
                             "archived row half with corrected normalization.")
    manifest = out/"protocol.json"
    if verify:
        assert json.loads(manifest.read_text(encoding="utf-8")) == spec
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    rows = []
    for part, panels in (("development", dev.previous.panels()), ("transfer", transfer.panels())):
        run_path = ARCHIVES[part]
        archive = pd.read_csv(run_path/"cases.csv").set_index(["dataset", "fold", "seed", "method"])
        inputs = []
        for panel in panels:
            if part == "development":
                inputs.append(dev.input_hashes(panel))
            rows.extend(compare(part, panel, run_path, archive, out, verify))
        if inputs:
            pd.testing.assert_frame_equal(pd.DataFrame(inputs), pd.read_csv(run_path/"inputs.csv"), check_dtype=False)
    cases = pd.DataFrame(rows)
    assert cases.groupby("part").size().to_dict() == {"development": 138, "transfer": 69}
    if verify:
        pd.testing.assert_frame_equal(cases, pd.read_csv(out/"cases.csv"), check_dtype=False)
        cases[["part","dataset","fold","seed","method","support_sha256","prediction_sha256"]].to_csv(
            out/"fresh-process-audit.csv", index=False)
    else:
        cases.to_csv(out/"cases.csv", index=False)
        cases["prediction_changed"] = cases.changed_predictions != 0
        cases["grouping_changed"] = cases.changed_parent != 0
        cases.groupby(["part","method"]).agg(configs=("dataset", "size"), supports_changed=("support_changed","sum"),
            predictions_changed=("prediction_changed","sum"), grouping_changed=("grouping_changed","sum"),
            previous_invalid_edges=("old_invalid_edges","sum"), revised_invalid_edges=("invalid_edges","sum"),
            reduced_within=("reduced_within","sum")).to_csv(out/"summary.csv")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
