"""Deploy the frozen exhaustive gain prefix within .005 of the validation maximum."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_replication import data
from experiments.minpi_replication.direct_gain import fit_native, predict_native
from experiments.split_vote.report import EXTERNAL_ARMS, ORIGINAL_ARMS
from experiments.split_vote.run import ORIGINAL, ROOT, RUN, digest, load

EVIDENCE = ROOT / "thesis-paper/evidence/minpi_replication"


def registered(panel):
    if panel == "external":
        return "four-frozen", data.SOURCES, data.SEEDS, EXTERNAL_ARMS
    if panel == "development":
        return "six-paired", ORIGINAL, legacy.CONFIG["seeds"], ORIGINAL_ARMS
    raise ValueError(panel)


def read_panel(panel):
    tag, names, seeds, arms = registered(panel)
    base = {}
    hashes = {}
    versions = set()
    for name in names:
        for seed in seeds:
            for arm in arms:
                folder = RUN/tag/f"{name}-{seed}"/arm
                path = folder/"receipt.json"
                report = json.loads(path.read_bytes())
                if (name,seed,arm) != (report["dataset"],report["seed"],report["method"]):
                    raise ValueError(f"Wrong receipt identity: {path}")
                if digest(folder/"forecast.npz") != report["forecast_sha256"]:
                    raise ValueError(f"Changed predictions: {folder}")
                versions.add(json.dumps(report["version"],sort_keys=True))
                hashes[f"{name}-{seed}/{arm}"] = digest(path)
                if arm == "exhaustive":
                    base[name,seed] = folder, report
    if len(versions) != 1:
        raise ValueError("Mixed versions across the registered panel")
    return base, sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def prefix(report):
    r = report["record"]
    order, curve = r["gain_order"], np.array(r["width_curve"])
    p = report["p"]
    if sorted(order) != list(range(p)) or curve.shape != (p,) or not np.isfinite(curve).all():
        raise ValueError("Invalid exhaustive path")
    best, best_k = -np.inf, 0
    for k, value in enumerate(curve,1):
        if value > best + 1e-14:
            best, best_k = value, k
    near_k = int(np.flatnonzero(curve >= curve.max()-.005-1e-14)[0])+1
    support = sorted(order[:near_k])
    if (r["near_width"],r["near_support"],r["near_validation_utility"]) != (near_k,support,curve[near_k-1]):
        raise ValueError("Matched .005 rule disagrees with exhaustive receipt")
    if (r["retained"],r["support"],r["validation_utility"]) != (best_k,sorted(order[:best_k]),best):
        raise ValueError("Archived primary argmax disagrees with exhaustive receipt")
    return support


def deploy(report, primary, x, y, task, idx, seed, device="cuda"):
    support = prefix(report)
    test = idx["test"]
    if len(primary) != len(test) or not np.isfinite(primary).all():
        raise ValueError("Invalid primary forecast")
    reused = support == report["record"]["support"]
    if reused:
        return primary.copy(), None
    dev = np.sort(np.concatenate([idx[key] for key in ("fit", "vote", "width")]))
    if np.intersect1d(dev,test).size or len(np.unique(dev)) != len(dev):
        raise ValueError("Invalid development partition")
    started = perf_counter()
    fitted = fit_native(x[np.ix_(dev,support)], y[dev], task, seed, device)
    pred = predict_native(fitted, x[np.ix_(test,support)], task)
    elapsed = perf_counter()-started
    if primary.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("Supplementary prediction array is invalid")
    return pred, elapsed


def filenames(panel,name,seed):
    folder = RUN/f"common-slack-{panel}"/f"{name}-{seed}"
    return folder/"receipt.json",folder/"forecast.npz"


def sources(name, seed, manifest):
    if name in data.SOURCES:
        return json.loads(data.MANIFEST.read_bytes())[name]["sha256"]
    row = manifest[name,seed][1]
    return ":".join(row[f"{part}_file_sha256"] for part in ("train","val","test"))


def verify(receipt, forecast, pair, panel_sha, source, test):
    meta = json.loads(receipt.read_bytes())
    folder, report = pair
    if (meta["receipt_sha256"],meta["panel_sha256"],meta["forecasts_sha256"],meta["source"],
        meta["support"],meta["dataset"],meta["seed"]) != (
        digest(folder/"receipt.json"),panel_sha,digest(forecast),source,prefix(report),
        report["dataset"],report["seed"]):
        raise ValueError(f"Stale supplementary receipt: {receipt}")
    with np.load(forecast) as a, np.load(folder/"forecast.npz") as base:
        np.testing.assert_array_equal(a["test"],test)
        if a["primary"].shape != base["primary"].shape or not np.isfinite(a["primary"]).all():
            raise ValueError(f"Invalid supplementary forecast: {forecast}")
        reused = meta["support"] == report["record"]["support"]
        if reused != meta["reused"] or reused != (meta["refit_predict_s"] is None):
            raise ValueError(f"Invalid supplementary deployment status: {receipt}")
        if reused:
            np.testing.assert_array_equal(a["primary"],base["primary"])
    return meta


def freeze(panel):
    jobs, panel_sha = read_panel(panel)
    manifests = legacy.jobs() if panel=="development" else {}
    for (name,seed),pair in jobs.items():
        folder, report = pair
        x,y,task,idx = load(name,seed)
        source = sources(name,seed,manifests)
        receipt, forecast = filenames(panel,name,seed)
        if not receipt.exists():
            receipt.parent.mkdir(exist_ok=True, parents=True)
            with np.load(folder/"forecast.npz") as a:
                pred,elapsed = deploy(report,a["primary"],x,y,task,idx,seed)
            with forecast.with_suffix(".tmp").open("wb") as out:
                np.savez_compressed(out,primary=pred,test=idx["test"])
            forecast.with_suffix(".tmp").replace(forecast)
            meta = dict(dataset=name,seed=seed,support=prefix(report),source=source,
                        receipt_sha256=digest(folder/"receipt.json"),panel_sha256=panel_sha,
                        forecasts_sha256=digest(forecast),reused=elapsed is None,refit_predict_s=elapsed)
            receipt.with_suffix(".tmp").write_text(json.dumps(meta,allow_nan=False),encoding="utf8")
            receipt.with_suffix(".tmp").replace(receipt)
        verify(receipt,forecast,pair,panel_sha,source,idx["test"])
        print(f"sealed {name}/{seed} {digest(forecast)}",flush=True)


def metrics(y,pred,task,scale):
    if task=="regression":
        mae = mean_absolute_error(y,pred)
        return dict(primary=r2_score(y,pred),mae=mae,scaled_mae=mae/scale)
    np.testing.assert_allclose(pred.sum(axis=1),1.,atol=2e-6)
    scores = pred[:,1] if pred.shape[1]==2 else pred
    auc = roc_auc_score(y,scores,multi_class="ovr",average="macro")
    return dict(primary=auc,accuracy=accuracy_score(y,pred.argmax(axis=1)))


def archive_parity(archive, name, seed, r, pred, truth):
    match, = [p for p in archive.namelist() if "PIv1-all-seeds-" in p and
              p.endswith(f"/seed-{seed}/{name}/direct_gain/frozen.json")]
    folder = match.removesuffix("frozen.json")
    frozen = json.loads(archive.read(match))
    ranking = json.loads(archive.read(folder+"ranking.json"))
    names = ranking["feature_order"]
    past = pd.read_csv(BytesIO(archive.read(folder+"validation_curve.csv")))["pooled_utility"].to_numpy()
    with np.load(BytesIO(archive.read(folder+"predictions.npz"))) as a:
        np.testing.assert_array_equal(a["y"],truth)
        archived = a["selected"]
    return dict(archive_order_equal=[names[i] for i in r["gain_order"]]==ranking["descending_gain"],
                archive_support_equal=[names[i] for i in r["support"]]==frozen["selected"],
                archive_max_curve_abs=float(np.max(np.abs(past-r["width_curve"]))),
                archive_prediction_equal=np.array_equal(pred,archived),
                archive_max_prediction_abs=float(np.max(np.abs(pred-archived))))


def verified_archive():
    path = ROOT/"packages/thesis-piv1-evidence.zip"
    expected = (ROOT/"docs/thesis/archives/piv1/evidence.sha256").read_text().split()[0]
    check = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1<<20):
            check.update(chunk)
    if check.hexdigest() != expected:
        raise ValueError("The original PIv1 evidence archive has changed")
    return ZipFile(path)


def score(panel):
    jobs,panel_sha = read_panel(panel)
    manifests = legacy.jobs() if panel=="development" else {}
    sealed = {}
    for (name,seed),pair in jobs.items():
        _,y,_,idx = load(name,seed)
        receipt,forecast = filenames(panel,name,seed)
        meta = verify(receipt,forecast,pair,panel_sha,sources(name,seed,manifests),idx["test"])
        sealed[name,seed] = meta,forecast,idx["test"],float(np.std(y[idx["fit"]]))
    rows = []
    with (verified_archive() if manifests else nullcontext()) as archive:
        for (name,seed),(folder,report) in jobs.items():
            meta,forecast,test,scale = sealed[name,seed]
            if name in data.SOURCES:
                truth = data._records(name).target[test]
            else:
                directory,manifest = manifests[name,seed]
                path = directory/manifest["test_relpath"]
                if digest(path) != manifest["test_file_sha256"]:
                    raise ValueError(f"Held labels changed: {path}")
                truth = pd.read_csv(path,usecols=[manifest["target"]]).iloc[:,0].to_numpy()
            with np.load(forecast) as a, np.load(folder/"forecast.npz") as b:
                near,primary = a["primary"],b["primary"]
            task,r = report["task"],report["record"]
            rows.append(dict(dataset=name,seed=seed,task=task,k=len(meta["support"]),
                             validation=r["near_validation_utility"],primary_k=r["retained"],
                             primary_validation=r["validation_utility"],reused=meta["reused"],
                             primary_method_s=r["worker_s"],extra_refit_predict_s=meta["refit_predict_s"],
                             near_method_s=r["worker_s"] if meta["reused"] else None,
                             primary_forecast_sha256=report["forecast_sha256"],
                             near_forecast_with_rows_sha256=meta["forecasts_sha256"],
                             **metrics(truth,near,task,scale),
                             **{f"argmax_{k}":v for k,v in metrics(truth,primary,task,scale).items()},
                             **(archive_parity(archive,name,seed,r,primary,truth) if archive else {})))
    path = EVIDENCE/f"common-slack-{panel}.csv"
    pd.DataFrame(rows).to_csv(path.with_suffix(".tmp"),index=False)
    path.with_suffix(".tmp").replace(path)
    print(path,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","score"])
    parser.add_argument("panel",choices=["external","development"])
    args = parser.parse_args()
    with threadpool_limits(1):
        (freeze if args.action=="freeze" else score)(args.panel)
