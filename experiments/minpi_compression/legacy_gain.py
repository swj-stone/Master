"""Evaluate the original six prepared datasets with the same ridge downstream selector."""
import argparse
import csv
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as bench
from minppi_mofs.modeling import make_model


ORIGINAL = bench.ROOT / "docs/thesis/archives/piv1/protocol.json"
CONFIG = json.loads(ORIGINAL.read_bytes())
RUN = bench.RUN / "legacy-gain"
DEST = bench.EVIDENCE / "legacy-gain"
METHODS = ("full", "coefficient", "gain", "minpi")
PROTOCOL = Path(__file__).with_name("legacy_gain_protocol.md")


def jobs():
    tasks = {}
    for manifest in sorted((bench.ROOT / "results/icomp2026_formal").glob("*/data_manifest.csv")):
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name, seed = row["dataset_id"], int(row["seed"])
                if name in CONFIG["datasets"] and seed in CONFIG["seeds"]:
                    if (name, seed) in tasks:
                        raise ValueError(f"Duplicate prepared split {name}/{seed}")
                    tasks[name, seed] = manifest.parent, row
    if len(tasks) != 30:
        raise ValueError("Expected the prepared splits for all six sources and five seeds")
    return tasks


def read_split(directory, row, seed):
    names = json.loads(row["selected_features_json"])
    target = row["target"]
    parts, ys = [], []
    for part in ("train", "val", "test"):
        path = directory / row[part + "_relpath"]
        if bench.digest(path) != row[part + "_file_sha256"]:
            raise ValueError(f"Prepared partition differs from manifest: {path}")
        columns = names if part == "test" else names + [target]
        frame = pd.read_csv(path, usecols=columns)
        parts.append(frame[names].to_numpy(dtype=float))
        if part != "test":
            ys.append(frame[target].to_numpy())
    y = np.concatenate(ys)
    train = np.arange(len(parts[0]))
    val = len(train) + np.arange(len(parts[1]))
    classification = row["task"] != "regression"
    if classification:
        y = y.astype(int)
    vote, width = train_test_split(val, test_size=.5, random_state=701 + seed,
                                  stratify=y[val] if classification else None)
    test = len(y) + np.arange(len(parts[2]))
    return np.concatenate(parts), y, "classification" if classification else "regression", dict(
        fit=train, vote=np.sort(vote), width=np.sort(width), test=test)


def gain_rank(method, x, y, task, idx, seed):
    if method != "gain":
        raise ValueError(method)
    fit = idx["fit"]
    problem = ("binary" if len(np.unique(y[fit])) == 2 else "multiclass") if task == "classification" else task
    model = make_model(task=problem, seed=seed, backend=CONFIG["backend"], model_params=CONFIG["model_params"])
    model.fit(x[fit], y[fit])
    booster = model.get_booster()
    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"Archived Gain requires CUDA, got {device}")
    gains = booster.get_score(importance_type="gain")
    order = sorted(range(x.shape[1]), key=lambda i: (gains.get(f"f{i}", 0.), i), reverse=True)
    return np.asarray(order), dict(device=device)


def archived_gain(archive, name, seed, features):
    tail = f"seed-{seed}/{name}/direct_gain/ranking.json"
    path, = [p for p in archive.namelist() if "PIv1-all-seeds-" in p and p.endswith(tail)]
    receipt = json.loads(archive.read(path))
    return [features.index(f) for f in receipt["descending_gain"]]


def signature():
    return {**bench.stamp(), "protocol": bench.digest(PROTOCOL), "legacy_gain": bench.digest(Path(__file__)),
            "original": bench.digest(ORIGINAL)}


def select():
    tasks = sorted(jobs().items())
    start = perf_counter()
    RUN.mkdir(exist_ok=True, parents=True)
    lock = {}
    with ZipFile(bench.ROOT / "packages/thesis-piv1-evidence.zip") as archive:
        for j, ((name, seed), (directory, row)) in enumerate(tasks):
            sub = RUN / f"{name}-{seed}"
            sub.mkdir(exist_ok=True)
            receipt = sub / "selection.json"
            if receipt.exists():
                info = json.loads(receipt.read_bytes())
                current = signature()
                accepted = [current]
                versions = json.loads(Path(__file__).with_name("legacy_gain_primary_compat.json").read_bytes())
                if current["legacy_gain"] == versions["revised"]:
                    accepted.append({**current, "legacy_gain":versions["original"]})
                if info["signature"] not in accepted or bench.digest(sub / "predictions.npz") != info["forecast"]:
                    raise ValueError(f"Stale evidence at {sub}")
                lock[sub.name] = bench.digest(receipt)
                continue
            timer = Timer(3600, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            try:
                begin = perf_counter()
                x, y, task, idx = read_split(directory, row, seed)
                expected = archived_gain(archive, name, seed, json.loads(row["selected_features_json"]))
                recs, out = [], {}
                order = np.random.default_rng(seed+j).permutation(METHODS)
                for repeat in range(3):
                    for method in np.roll(order, repeat):
                        if method == "gain":
                            # Inject only the comparator's ranker and use the locked width/deployment path verbatim.
                            with patch.object(bench, "ranked", gain_rank):
                                rec, pred = bench.select_one(method, x, y, task, idx, seed)
                            rec["archive_order_identical"] = rec["order"] == expected
                        else:
                            rec, pred = bench.select_one(method, x, y, task, idx, seed)
                        if repeat:
                            np.testing.assert_array_equal(pred, out[method])
                        recs.append(dict(**rec, repeat=repeat))
                        out[method] = pred
                    print(f"{name}/{seed} repeat {repeat+1}/3 elapsed {perf_counter()-begin:.1f}s", flush=True)
                np.savez_compressed(sub / "predictions.npz", **out)
                info = dict(dataset=name, seed=seed, task=task, signature=signature(),
                            inputs={p:row[p+"_file_sha256"] for p in ("train", "val", "test")},
                            scale=max(float(np.std(y)),1e-12), forecast=bench.digest(sub / "predictions.npz"),
                            vote=idx["vote"].tolist(), width=idx["width"].tolist(), rows=recs)
                tmp = receipt.with_suffix(".tmp")
                tmp.write_text(json.dumps(info), encoding="utf8")
                tmp.replace(receipt)
                lock[sub.name] = bench.digest(receipt)
                elapsed = perf_counter() - start
                print(f"sealed [{j+1}/{len(tasks)}] {name}/{seed} {perf_counter()-begin:.1f}s; "
                      f"elapsed {elapsed:.1f}s ETA {elapsed*(len(tasks)-j-1)/(j+1):.1f}s", flush=True)
            finally:
                timer.cancel()
    (RUN / "lock.json").write_text(json.dumps(lock,indent=2), encoding="utf8")


def independent_metrics(y, pred, task, scale):
    if task == "classification":
        scores = []
        for c, column in enumerate(pred.T):
            positive = y == c
            n = positive.sum()
            # Rank midpoints can sum above 1e9 on a large negative class.
            total_rank = rankdata(np.asarray(column,dtype=np.float64))[positive].sum(dtype=np.float64)
            scores.append((total_rank - n*(n+1)/2) / (n*(len(y)-n)))
        return dict(auc=np.mean(scores), accuracy=np.mean(y == pred.argmax(axis=1)))
    mae = np.mean(np.abs(y-pred))
    return dict(r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2), mae=mae, scaled_mae=mae/scale)


def score():
    lock = json.loads((RUN / "lock.json").read_bytes())
    tasks = jobs()
    if set(lock) != {f"{n}-{s}" for n,s in tasks}:
        raise ValueError("Incomplete prediction lock")
    rows = []
    for (name, seed), (directory, row) in tasks.items():
        sub = RUN / f"{name}-{seed}"
        assert lock[sub.name] == bench.digest(sub / "selection.json")
        info = json.loads((sub / "selection.json").read_bytes())
        assert bench.digest(sub / "predictions.npz") == info["forecast"]
        target_file = directory / row["test_relpath"]
        assert info["inputs"]["test"] == bench.digest(target_file)
        y = pd.read_csv(target_file, usecols=[row["target"]])[row["target"]].to_numpy()
        with np.load(sub / "predictions.npz") as predictions:
            for method in METHODS:
                runs = [r for r in info["rows"] if r["method"] == method]
                rec = runs[0]
                assert all(r["k"] == rec["k"] and r["order"] == rec["order"] for r in runs)
                metrics = independent_metrics(y,predictions[method],info["task"],info["scale"])
                reference = bench.metrics(y,predictions[method],info["task"],info["scale"])
                np.testing.assert_allclose(list(metrics.values()),list(reference.values()),atol=1e-12,rtol=0)
                rows.append(dict(dataset=name, seed=seed, task=info["task"], method=method, k=rec["k"], p=rec["p"],
                    retained=rec["k"]/rec["p"], selected=" ".join(map(str,sorted(rec["order"][:rec["k"]]))),
                    archive_order_identical=rec.get("archive_order_identical"),negative=rec.get("minimum",[]).count(-1),
                    **{k: np.median([r[k] for r in runs]) for k in ("rank_s","selection_s","deployment_s")}, **metrics))
    frame = pd.DataFrame(rows)
    DEST.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEST / "scores.csv", index=False)
    frame.groupby(["dataset","method"]).mean(numeric_only=True).drop(columns="seed").to_csv(DEST / "sources.csv")
    (DEST / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf8")
    result = []
    for task, group in frame.groupby("task"):
        idx = group.set_index(["dataset","seed","method"])
        ours = idx.xs("minpi",level="method")
        for baseline in ("coefficient", "full", "gain"):
            other = idx.xs(baseline,level="method").reindex(ours.index)
            fields = ["auc","accuracy"] if task == "classification" else ["r2","scaled_mae","mae"]
            diff = pd.DataFrame({m: ours[m]-other[m] for m in fields})
            ratios = ["retained", "deployment_s"] if baseline == "full" else ["retained","selection_s","deployment_s"]
            for m in ratios:
                diff[m] = np.log(ours[m]/other[m])
            effect = diff.groupby(level="dataset").mean().mean()
            effect[ratios] = np.exp(effect[ratios])
            result.append(dict(task=task,baseline=baseline,same_support=int((ours.selected==other.selected).sum()),
                               **effect.to_dict()))
    summary = pd.DataFrame(result)
    summary.to_csv(DEST / "contrasts.csv",index=False)
    print(summary.round(5).to_string(index=False))
    print("Gain rankings identical to archive:",frame[frame.method=="gain"].archive_order_identical.sum(),"/30")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "score"])
    action = parser.parse_args().action
    if os.name == "nt":
        import ctypes
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<8)):
            raise ctypes.WinError()
    with threadpool_limits(limits=1):
        (select if action == "select" else score)()


if __name__ == "__main__":
    run()
