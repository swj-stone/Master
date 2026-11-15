"""Evaluate fixed tree-selected supports with a logistic guard and final learner."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
import warnings
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.refit_opportunity import outcome

ROOT = bg.ROOT
EVIDENCE = ROOT / "thesis-paper/evidence"
DEST = EVIDENCE / "receiver"
DESIGN = ROOT / "configs/context_review/receiver_transport.json"
SOURCE = {"first": "transport-uci", "wide": "wide-uci"}
ARMS = ("gain", "vote", "point", "slack")
METRICS = ("u", "breach", "accepted", "saved", "auc", "accuracy", "ap", "joint_ceiling")
CONTRASTS = (("slack", "point"), ("slack", "uniform"), ("vote", "gain"),
             ("point", "uniform"), ("point", "gain"))


def member(archive, key, data):
    # Fixed ZIP timestamps make repeated calculations byte-comparable.
    archive.writestr(ZipInfo(key), data, compress_type=ZIP_DEFLATED)


def frozen_folds():
    first = bg.read(EVIDENCE / "transport/prediction-lock.json")["selections"]
    wide = bg.read(EVIDENCE / "wide/prediction-lock.json")["folds"]
    if len(first) != 30 or len(wide) != 16:
        raise ValueError("Incomplete campaign")
    return [("first", key, h) for key, h in sorted(first.items())] + [
        ("wide", key, row["selection.json"]) for key, row in sorted(wide.items())]


def transformed(x, y, fitting, evaluation, supports):
    # Transformations act independently on columns, so they can be reused for all supports.
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    train = impute.fit_transform(np.asarray(x[fitting], dtype=np.float64))
    scale = StandardScaler().fit(train)
    train = scale.transform(train)
    valid = scale.transform(impute.transform(np.asarray(evaluation, dtype=np.float64)))
    probabilities, diagnostics = [], []
    for support in supports:
        model = LogisticRegression(C=1., solver="lbfgs", tol=1e-6, max_iter=2000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train[:, support], y[fitting])
        np.testing.assert_array_equal(model.classes_, np.arange(len(model.classes_)))
        probabilities.append(model.predict_proba(valid[:, support]))
        diagnostics.append({"iterations": int(np.max(model.n_iter_)),
                            "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught)})
    return probabilities, diagnostics


def predict(panel, key, selection_sha):
    src = ROOT / "results/context_review" / SOURCE[panel] / key
    if bg.digest(src / "selection.json") != selection_sha:
        raise ValueError(f"Selection mismatch for {panel}/{key}")
    select = bg.read(src / "selection.json")
    layout = bg.read(src / "layout.json")
    assert bg.digest(src / "layout.json") == select["layout"]
    assert bg.digest(src / "inputs.npz") == layout["inputs_sha256"]
    anchor = select["choices"]["gain"]["anchor"]
    supports = [anchor] + [c["columns"] for c in select["pool"]]
    nominees = {arm: supports.index(select["choices"][arm]["nominee"]) for arm in ARMS}
    assert len(set(map(tuple, supports))) == len(supports)
    assert all(i != 0 for i in nominees.values()) or len(supports) == 1
    with np.load(src / "inputs.npz") as inputs:
        x, y, outer, fitting, prop, guard, test_row = [
            inputs[k] for k in ("x", "y", "x_outer", "train", "proposal", "guard", "test_row")]
    inner = np.sort(np.r_[fitting, prop])
    dev = np.arange(len(y))
    np.testing.assert_array_equal(np.sort(np.r_[inner, guard]), dev)
    g, guard_runs = transformed(x, y, inner, x[guard], supports)
    o, outer_runs = transformed(x, y, dev, outer, supports)
    arrays = BytesIO()
    np.savez_compressed(arrays, guard_y=y[guard], guard=guard, test_row=test_row,
                        **{f"g{i}": p for i, p in enumerate(g)}, **{f"o{i}": p for i, p in enumerate(o)})
    data = arrays.getvalue()
    info = dict(panel=panel, dataset=layout["dataset"], fold=layout["fold"], rare=select["choices"]["gain"]["rare"],
                supports=supports, nominees=nominees, input_sha=layout["inputs_sha256"], layout_sha=select["layout"],
                selection_sha=selection_sha, targets_sha=layout["targets_sha256"], predictions_sha=sha256(data).hexdigest(),
                guard_runs=guard_runs, outer_runs=outer_runs)
    return info, data


def fit():
    DEST.mkdir(exist_ok=True, parents=True)
    start = perf_counter()
    jobs = frozen_folds()
    records = {}
    temp = DEST / "predictions.tmp"
    with ZipFile(temp, "w") as archive, threadpool_limits(limits=1):
        for n, (panel, key, selection_sha) in enumerate(jobs, 1):
            rec, data = predict(panel, key, selection_sha)
            full_key = f"{panel}/{key}"
            member(archive, full_key + ".npz", data)
            records[full_key] = rec
            elapsed = perf_counter() - start
            print(f"[{n:02}/{len(jobs)}] {full_key}: elapsed {elapsed:.1f}s, ETA {elapsed*(len(jobs)-n)/n:.1f}s", flush=True)
    path = DEST / "predictions.zip"
    temp.replace(path)
    bg.store(DEST / "prediction-lock.json", dict(design_sha=bg.digest(DESIGN), sklearn=sklearn.__version__,
             forecast_archive_sha=bg.digest(path), folds=records))


def copy_outcomes():
    lock = bg.read(DEST / "prediction-lock.json")
    assert len(lock["folds"]) == 46 and bg.digest(DEST / "predictions.zip") == lock["forecast_archive_sha"]
    temp = DEST / "outcomes.tmp"
    with ZipFile(temp, "w") as archive:
        for full_key, rec in lock["folds"].items():
            panel, key = full_key.split("/", 1)
            src = ROOT / "results/context_review" / SOURCE[panel] / key / "targets.npz"
            data = src.read_bytes()
            assert sha256(data).hexdigest() == rec["targets_sha"]
            member(archive, full_key + ".npz", data)
    temp.replace(DEST / "outcomes.zip")


def replay():
    lock = bg.read(DEST / "prediction-lock.json")
    assert lock["design_sha"] == bg.digest(DESIGN)
    assert bg.digest(DEST / "predictions.zip") == lock["forecast_archive_sha"]
    assert len(lock["folds"]) == 46
    candidates, policies = [], []
    with ZipFile(DEST / "predictions.zip") as pred, ZipFile(DEST / "outcomes.zip") as truth:
        for full_key, rec in lock["folds"].items():
            name = full_key + ".npz"
            p, labels = pred.read(name), truth.read(name)
            assert sha256(p).hexdigest() == rec["predictions_sha"] and sha256(labels).hexdigest() == rec["targets_sha"]
            with np.load(BytesIO(p)) as arrays, np.load(BytesIO(labels)) as outer:
                np.testing.assert_array_equal(arrays["test_row"], outer["test_row"])
                guards, scores = [], []
                for i in range(len(rec["supports"])):
                    for phase, y, out in (("g", arrays["guard_y"], guards), ("o", outer["y"], scores)):
                        probs = arrays[f"{phase}{i}"]
                        np.testing.assert_array_equal(np.unique(y), np.arange(probs.shape[1]))
                        assert np.isfinite(probs).all()
                        np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-12)
                        out.append(bg.metrics(y, probs, rec["rare"]))
            ident = {field: rec[field] for field in ("panel", "dataset", "fold")}
            finals = []
            for i, support in enumerate(rec["supports"]):
                guard_ok = outcome(guards[i], guards[0], .03)
                outer_ok = outcome(scores[i], scores[0], .03)
                j = i if guard_ok else 0
                returned = scores[j]
                saved = 1 - len(rec["supports"][j]) / len(rec["supports"][0])
                failed = not outcome(returned, scores[0], .03)
                finals.append(dict(u=saved * (not failed), breach=int(failed), accepted=int(j != 0), saved=saved,
                                   auc=returned["auc"], accuracy=returned["accuracy"], ap=returned.get("ap", np.nan)))
                candidates.append(dict(**ident, index=i, columns=" ".join(map(str, support)), guard_ok=guard_ok,
                                       outer_ok=outer_ok, **{"guard_"+k: v for k, v in guards[i].items()},
                                       **{"outer_"+k: v for k, v in scores[i].items()},
                                       guard_iter=rec["guard_runs"][i]["iterations"], outer_iter=rec["outer_runs"][i]["iterations"],
                                       convergence_warning=rec["guard_runs"][i]["convergence_warning"] or rec["outer_runs"][i]["convergence_warning"]))
            vals = pd.DataFrame(finals)
            available = vals.iloc[1:] if len(vals) > 1 else vals
            arms = {arm: vals.iloc[i] for arm, i in rec["nominees"].items()}
            arms["uniform"] = available.mean()
            for arm, result in arms.items():
                policies.append(dict(**ident, method=arm, joint_ceiling=available.u.max(), **result.to_dict()))
    candidate_frame, folds = pd.DataFrame(candidates), pd.DataFrame(policies)
    candidate_frame.to_csv(DEST / "candidates.csv", index=False)
    folds.to_csv(DEST / "folds.csv", index=False)
    tasks = folds.groupby(["panel", "method", "dataset"])[list(METRICS)].mean()
    tasks.to_csv(DEST / "tasks.csv")
    tasks.groupby(["panel", "method"]).mean().to_csv(DEST / "means.csv")
    contrasts = []
    for panel in SOURCE:
        for arm, ref in CONTRASTS:
            delta = tasks.loc[panel, arm] - tasks.loc[panel, ref]
            for metric in ("u", "breach", "auc", "accuracy"):
                values = delta[metric].to_numpy()
                assert np.isfinite(values).all()
                draws = np.random.default_rng(33119).integers(len(values), size=(30_000, len(values)))
                lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
                contrasts.append(dict(panel=panel, method=arm, reference=ref, metric=metric, mean=values.mean(),
                                      low95=lo, high95=hi, improved=int((values > 1e-14).sum()), worse=int((values < -1e-14).sum())))
    pd.DataFrame(contrasts).to_csv(DEST / "contrasts.csv", index=False)
    print(tasks.groupby(["panel", "method"]).mean().round(5))
    print(f"{len(folds)//5} folds, {len(candidate_frame)} anchor/candidate pairs assessed at both stages, "
          f"{candidate_frame.convergence_warning.sum()} had a convergence warning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score", "replay"))
    mode = parser.parse_args().mode
    if mode == "fit":
        fit()
    else:
        if mode == "score":
            copy_outcomes()
        replay()
