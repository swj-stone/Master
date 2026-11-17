"""Run two unmodified scikit-learn selectors against the frozen anchored MinPI after release."""
import argparse
import ctypes
from hashlib import sha256
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import PredefinedSplit
from threadpoolctl import threadpool_limits

from experiments.anchored_vote import audit, select as policy
from experiments.minpi_replication import direct_gain
from experiments.minpi_replication.second_panel import data
from experiments.split_vote import path as width_path
from minppi_mofs import modeling, split_vote, utility


OUT = audit.OUT / "standard"
ARMS = ("min", "select_from_model", "rfecv")
REPS = 3


def digest(raw):
    return sha256(raw).hexdigest()


def signature():
    files = [Path(__file__), *(Path(mod.__file__) for mod in (policy, data, direct_gain, width_path,
                                                              modeling, split_vote, utility, audit)),
             Path(data.__file__).with_name("sources.json")]
    ledger = {str(f.relative_to(audit.ROOT)): digest(f.read_bytes()) for f in files}
    for estimator in (SelectFromModel, RFECV, PredefinedSplit):
        f = Path(inspect.getfile(estimator))
        ledger[estimator.__name__] = dict(source=digest(f.read_bytes()), signature=str(inspect.signature(estimator)))
    return dict(hashes=ledger, sklearn=sklearn.__version__, xgboost=xgboost.__version__, cpu_mask=256,
                repetitions=REPS)


def native(task, y, seed):
    kind = task if task == "regression" else ("binary" if len(np.unique(y)) == 2 else "multiclass")
    return modeling.make_model(task=kind, model_params=direct_gain.MODEL_PARAMS, backend="cpu", seed=seed)


def matches_signature(recorded, active):
    if recorded == active:
        return True
    # Only the reporting columns and this verification path changed after timing.
    compat = json.loads((OUT/"report_compatibility.json").read_bytes())
    key = str(Path(__file__).resolve().relative_to(audit.ROOT))
    if recorded["hashes"][key] != compat["measured"] or active["hashes"][key] != compat["reporting"]:
        return False
    return recorded == {**active, "hashes": {**active["hashes"], key: compat["measured"]}}


def baseline(x, y, task, idx, seed, method):
    started = perf_counter()
    fit = np.sort(idx["fit"])
    valid = np.sort(np.r_[idx["vote"], idx["width"]])
    dev = np.sort(np.r_[fit, valid])
    estimator = native(task, y[fit], seed)
    details = {}
    if method == "select_from_model":
        selector = SelectFromModel(estimator).fit(x[fit], y[fit])
        cols = selector.get_support(indices=True)
        final = direct_gain.fit_native(x[np.ix_(dev, cols)], y[dev], task, seed, "cpu")
        pred = direct_gain.predict_native(final, x[np.ix_(idx["test"], cols)], task)
        elapsed = perf_counter() - started
        assert np.isclose(selector.threshold_, selector.estimator_.feature_importances_.mean())
        np.testing.assert_array_equal(cols, np.flatnonzero(selector.estimator_.feature_importances_ >= selector.threshold_))
        details["threshold"] = float(selector.threshold_)
    else:
        split = PredefinedSplit(test_fold=np.where(np.isin(dev, fit), -1, 0))
        scoring = "r2" if task == "regression" else ("roc_auc" if len(np.unique(y[fit])) == 2 else "roc_auc_ovr")
        selector = RFECV(estimator, step=.2, min_features_to_select=1, cv=split, scoring=scoring, n_jobs=1)
        selector.fit(x[dev], y[dev])
        pred = selector.predict(x[idx["test"]]) if task == "regression" else selector.predict_proba(x[idx["test"]])
        elapsed = perf_counter() - started
        cols = selector.get_support(indices=True)
        final = selector.estimator_
        train_ix, valid_ix = next(split.split())
        np.testing.assert_array_equal(dev[train_ix], fit)
        np.testing.assert_array_equal(dev[valid_ix], valid)
        details.update(scoring=scoring, widths=selector.cv_results_["n_features"].tolist(),
                       cv_scores=selector.cv_results_["mean_test_score"].tolist())
        assert np.isfinite(details["cv_scores"]).all()
    params = json.loads(final.get_booster().save_config())["learner"]["generic_param"]
    assert params["device"] == "cpu" and int(params["nthread"]) == 1
    return dict(k=len(cols), selected=cols.tolist(), seconds=elapsed, details=details), pred


def measure():
    kernel = ctypes.windll.kernel32
    if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(256)):
        raise RuntimeError("Could not pin to logical CPU 8")
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(actual), ctypes.byref(system)) or actual.value != 256:
        raise RuntimeError("Affinity check failed")
    OUT.mkdir(parents=True, exist_ok=True)
    sig = signature()
    manifest = OUT/"signature.json"
    if manifest.exists():
        if not matches_signature(json.loads(manifest.read_bytes()), sig):
            raise RuntimeError("Measurement code or environment changed")
    else:
        manifest.write_text(json.dumps(sig, indent=2),encoding="utf8")
    frozen = pd.read_csv(audit.OUT/"seeds.csv").query("method == 'min'").set_index(["dataset","seed"])
    warm = np.random.default_rng(5393).normal(size=(96,6)).astype("float32")
    idx = dict(fit=np.arange(48), vote=np.arange(48,64), width=np.arange(64,80), test=np.arange(80,96))
    for arm in ARMS:
        if arm == "min":
            policy.select(warm, np.arange(96)%2, "classification", idx, 42, "min", "cpu")
        else:
            baseline(warm, np.arange(96)%2, "classification", idx, 42, arm)
    for si, seed in enumerate(data.SEEDS):
        for ni, name in enumerate(data.SOURCES):
            dest = OUT/f"{name}-{seed}"
            dest.mkdir(exist_ok=True)
            receipt = dest/"receipt.json"
            forecasts = dest/"predictions.npz"
            if receipt.exists():
                old = json.loads(receipt.read_bytes())
                if not matches_signature(old["signature"],sig) or old["forecast"] != digest(forecasts.read_bytes()):
                    raise RuntimeError(f"Stale {receipt}")
                continue
            x, y, task, idx = data.split(name,seed)
            x = np.asarray(x, np.float32)
            assert (y[idx["test"]] == -1).all() if task == "classification" else np.isnan(y[idx["test"]]).all()
            recs = []
            preds = {}
            clock = perf_counter()
            for rep in range(REPS):
                for order, arm in enumerate(np.roll(ARMS, (rep+si+ni)%len(ARMS))):
                    if arm == "min":
                        rec, pred = policy.select(x,y,task,idx,seed,"min","cpu")
                    else:
                        rec, pred = baseline(x,y,task,idx,seed,arm)
                    if rep:
                        np.testing.assert_array_equal(preds[arm], pred)
                    else:
                        preds[arm] = pred
                    pred_sha = digest(np.ascontiguousarray(pred).tobytes())
                    if arm == "min":
                        old = frozen.loc[name,seed]
                        assert pred_sha == old.predictions_sha256 and rec["selected"] == json.loads(old["columns"])
                    recs.append(dict(method=arm, repeat=rep, order=order, k=rec["k"], selected=rec["selected"],
                                     pred_sha=pred_sha, seconds=rec["seconds"], details=rec.get("details", {})))
            np.savez_compressed(forecasts, indices=idx["test"], **preds)
            obs = dict(dataset=name, seed=seed, task=task, signature=sig, forecast=digest(forecasts.read_bytes()), records=recs)
            receipt.write_text(json.dumps(obs,allow_nan=False), encoding="utf8")
            print(f"{name}/{seed}: {perf_counter()-clock:.2f}s, {1+si*len(data.SOURCES)+ni}/25",flush=True)


def report():
    rows, times = [], []
    for name in data.SOURCES:
        for seed in data.SEEDS:
            folder = OUT/f"{name}-{seed}"
            obs = json.loads((folder/"receipt.json").read_bytes())
            archive = folder/"predictions.npz"
            assert digest(archive.read_bytes()) == obs["forecast"] and matches_signature(obs["signature"], signature())
            truth = data.truth(name,seed)
            with np.load(archive) as preds:
                _, _, task, idx = data.split(name,seed)
                np.testing.assert_array_equal(preds["indices"],idx["test"])
                for method in ARMS:
                    rr = [r for r in obs["records"] if r["method"]==method]
                    assert sorted(r["repeat"] for r in rr)==list(range(REPS))
                    assert sorted(r["order"] for r in rr)==list(range(len(ARMS)))
                    pred_sha = digest(np.ascontiguousarray(preds[method]).tobytes())
                    assert all(r["pred_sha"]==pred_sha and r["selected"]==rr[0]["selected"] for r in rr)
                    times.extend(dict(dataset=name,seed=seed,method=method,repeat=r["repeat"],order=r["order"],
                                      seconds=r["seconds"]) for r in rr)
                    rows.append(dict(dataset=name,seed=seed,task=task,method=method,k=rr[0]["k"],
                                     columns=json.dumps(rr[0]["selected"]),seconds=np.median([r["seconds"] for r in rr]),
                                     **audit.score(truth,preds[method],task)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"seeds.csv",index=False)
    pd.DataFrame(times).to_csv(OUT/"timings.csv",index=False)
    result = []
    for baseline_name in ARMS[1:]:
        joined = frame[frame.method=="min"].merge(frame[frame.method==baseline_name],on=["dataset","seed"],
                                                 suffixes=("_min","_control"),validate="one_to_one")
        for name, g in joined.groupby("dataset"):
            speed = g.seconds_min-g.seconds_control
            excess = g.k_min-g.k_control-np.maximum(1.,.05*g.k_control)
            diff = g.primary_min-g.primary_control
            output = dict(dataset=name, control=baseline_name, median_paired_time_ratio=np.median(g.seconds_min/g.seconds_control),
                         median_width_difference=np.median(g.k_min-g.k_control), median_quality_difference=np.median(diff),
                         pass_speed_width=bool(speed.median()<0 and excess.median()<=0),
                         pass_speed_width_seeds=int(((speed<0)&(excess<=0)).sum()),
                         pass_joint=bool(speed.median()<0 and excess.median()<=0 and diff.median()>=-.01),
                         pass_seeds=int(((speed<0)&(excess<=0)&(diff>=-.01)).sum()))
            output.update({k:g[k].median() for k in ("seconds_min","seconds_control","k_min","k_control",
                           "primary_min","primary_control","accuracy_min","accuracy_control","mae_min","mae_control")})
            result.append(output)
    summary = pd.DataFrame(result)
    summary.to_csv(OUT/"sources.csv",index=False)
    print(summary[["dataset","control","median_paired_time_ratio","median_width_difference","median_quality_difference","pass_joint","pass_seeds"]].round(4).to_string(index=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("measure","report"))
    with threadpool_limits(1):
        (measure if cli.parse_args().phase=="measure" else report)()

