"""Quality and end-to-end cost of one frozen feature selector on 126 archived panels."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_benchmark as original
from experiments import adaptive_minpi_transfer as extra
from experiments import budget_minpi_deployments as deployment
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, digest
from minppi_mofs.stable_minpi import _impute_validation
from research.algorithms.coalition_residual import select_features

ARCHIVE = ROOT / "results/minpi-study"
ARMS = ("minpi", "full", "F", "random", "marginal")
KEY = ["cohort", "dataset", "fold", "seed"]
VARIATES = 5
REPEATS = 5


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol():
    paths = [
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "research/algorithms/coalition_residual.py",
        "experiments/compare_predictive_minpi.py",
        "experiments/adaptive_minpi_benchmark.py",
        "experiments/moment_minpi_benchmark.py",
        "experiments/adaptive_minpi_transfer.py",
        "experiments/budget_minpi_deployments.py",
        "experiments/minpi_backward_confirmation.py",
        "experiments/fast_minpi_five_source.py",
        "experiments/stable_minpi_stress.py",
        "src/minppi_mofs/stable_minpi.py",
        "research/evaluation/historical-panels.csv",
    ]
    return dict(arms=list(ARMS), random_quality_draws=VARIATES, timed_repeats=REPEATS,
                solver="Cholesky when min(n,p)<=2048, else LSQR; based on original shape for all arms",
                python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                threads=1, source_sha256={name:checksum(ROOT/name) for name in paths})


def panels():
    known = pd.read_csv(ARCHIVE/"adaptive-69-20261117a/inputs.csv").set_index(["dataset","fold","seed"])
    for panel in original.previous.panels():
        key = (panel.name, panel.fold, panel.seed)
        if original.input_hashes(panel) != dict(zip(["dataset","fold","seed"],key)) | known.loc[key].to_dict():
            raise ValueError(f"Original input fingerprints differ on {key}")
        yield "original", panel, panel.gold, panel.units, "original"
    for panel in extra.panels():
        case = f"{panel.name}-{panel.fold}-{panel.seed}"
        with np.load(ARCHIVE/"adaptive-transfer-20261117a"/f"{case}.npz") as archive:
            np.testing.assert_array_equal(
                [digest(a) for a in (panel.X, panel.y, panel.Xt, panel.truth)], archive["input_hashes"])
        yield "extra", panel, panel.truth, None, panel.family


def select(arm, panel, width):
    X, y = panel.X, panel.y
    if arm == "full":
        return np.arange(X.shape[1])
    if arm == "minpi":
        return select_features(X, y, task=panel.task, random_state=int(panel.seed))[0]
    if arm == "marginal":
        return select_features(X, y, task=panel.task, random_state=int(panel.seed),
                               mode="marginal", count=width)[0]
    if arm == "F":
        if panel.name in deployment.ADDITIONAL:
            X, _ = _impute_validation(X, X[:0])
        return choose("f", X, y, panel.task, panel.seed, width)
    if arm == "random":
        return np.sort(np.random.default_rng(panel.seed).permutation(X.shape[1])[:width])
    if arm.startswith("random_"):
        seed = [28631, panel.seed, panel.fold, int(arm.split("_")[1])]
        return np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width])
    raise ValueError(arm)


def fit(panel, cols):
    n, p = panel.X.shape
    A, B = (panel.X, panel.Xt) if len(cols)==p else (panel.X[:,cols], panel.Xt[:,cols])
    if panel.name in deployment.ADDITIONAL:
        A, B = _impute_validation(A, B)
    solver = "cholesky" if min(n,p) <= 2048 else "lsqr"
    cls = RidgeClassifier if panel.task == "classification" else Ridge
    model = make_pipeline(StandardScaler(), cls(alpha=1., solver=solver, max_iter=250, tol=.001))
    model.fit(A, panel.y)
    if panel.task == "regression":
        return model.predict(B), None
    decision = model.decision_function(B)
    idx = (decision > 0).astype(int) if decision.ndim == 1 else decision.argmax(axis=1)
    return model.classes_[idx], decision


def pipeline(panel, arm, width):
    start = perf_counter_ns()
    cols = select(arm, panel, width)
    selected = perf_counter_ns()
    pred, decision = fit(panel, cols)
    end = perf_counter_ns()
    return cols, pred, decision, (selected-start)/1e6, (end-start)/1e6


def run(stage, out, quality, pass_id):
    state = protocol()
    if stage == "cost":
        if state != json.loads((quality/"protocol.json").read_text(encoding="utf-8")):
            raise ValueError("Need the original selector and harness for cost replay")
    out.mkdir(parents=True, exist_ok=False)
    (out/"protocol.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    scores, unit_scores, timings = [], [], []
    counts = {"original":0,"extra":0}
    registry = pd.read_csv(ROOT/"research/evaluation/historical-panels.csv")
    expected = set(registry[["panel","dataset","fold","seed"]].itertuples(index=False,name=None))
    encountered = set()
    for cohort, panel, truth, test_units, family in panels():
        n, p = panel.X.shape
        cfg = dict(cohort=cohort, dataset=panel.name, fold=panel.fold, seed=panel.seed)
        case = "-".join(str(x) for x in cfg.values())
        key = tuple(cfg.values())
        if key not in expected or key in encountered:
            raise ValueError(f"Unregistered/duplicate panel {key}")
        encountered.add(key)
        counts[cohort] += 1
        if stage == "quality":
            arrays = {"truth":truth, "input_fingerprints":np.array([digest(a) for a in (panel.X,panel.y,panel.Xt,truth)])}
            if test_units is not None:
                arrays["test_units"] = test_units
            width = None
            for arm in (*ARMS, *(f"random_{j}" for j in range(VARIATES))):
                cols, pred, decision, _, _ = pipeline(panel, arm, width)
                if arm == "minpi":
                    width = len(cols)
                assert len(cols)==(p if arm=="full" else width)
                assert np.array_equal(cols,np.unique(cols)) and 0 <= min(cols) <= max(cols) < p
                score, units, ap = deployment.score_predictions(panel.name,truth,pred,test_units,panel.task,decision)
                arrays[arm+"_columns"], arrays[arm+"_pred"] = cols,pred
                if panel.name == "secom":
                    arrays[arm+"_decision"] = decision
                scores.append(dict(**cfg,family=family,task=panel.task,train=n,test=len(truth),
                                   p=p,method=arm,width=len(cols),score=score,secom_ap=ap))
                unit_scores.extend(dict(**cfg,method=arm,unit=unit,score=v) for unit,v in units.items())
            np.savez_compressed(out/f"{case}.npz", **arrays)
        else:
            with np.load(quality/f"{case}.npz") as source:
                reference = {k:source[k] for arm in ARMS for k in (arm+"_columns",arm+"_pred")}
                if panel.name == "secom":
                    reference.update({arm+"_decision":source[arm+"_decision"] for arm in ARMS})
            width = len(reference["minpi_columns"])
            base = np.random.default_rng([5399,panel.seed,panel.fold,pass_id]).permutation(ARMS)
            for repeat in range(-1,REPEATS):
                for position, arm in enumerate(np.roll(base,repeat)):
                    cols, pred, decision, select_ms,total_ms = pipeline(panel,arm,width)
                    np.testing.assert_array_equal(reference[arm+"_columns"],cols)
                    np.testing.assert_array_equal(reference[arm+"_pred"],pred)
                    if panel.name=="secom":
                        np.testing.assert_array_equal(reference[arm+"_decision"],decision)
                    if repeat >= 0:
                        timings.append(dict(**cfg,method=arm,repeat=repeat,position=position,
                                            select_ms=select_ms,total_ms=total_ms))
        print(f"{case} ({len(encountered)}/126)", flush=True)
    if encountered != expected or counts != {"original":69,"extra":57}:
        raise ValueError(f"Incomplete evaluation: {counts}")
    if protocol() != state:
        raise ValueError("Method or dependency changed during the run")
    if stage == "quality":
        pd.DataFrame(scores).to_csv(out/"scores.csv",index=False)
        pd.DataFrame(unit_scores).to_csv(out/"units.csv",index=False)
    else:
        pd.DataFrame(timings).to_csv(out/"times.csv",index=False)
    (out/"complete.json").write_text(json.dumps(dict(counts=counts,stage=stage,pass_id=pass_id)),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["quality","cost"])
    parser.add_argument("output",type=Path)
    parser.add_argument("--quality",type=Path)
    parser.add_argument("--pass-id",type=int,choices=(1,2),default=1)
    args = parser.parse_args()
    if args.stage=="cost" and args.quality is None:
        parser.error("Cost replay requires an independent quality archive")
    with threadpool_limits(1):
        run(args.stage,args.output,args.quality,args.pass_id)
