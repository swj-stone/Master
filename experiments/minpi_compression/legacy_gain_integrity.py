"""Check original row IDs and count exact prepared predictor overlap without accessing labels."""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.minpi_compression.legacy_gain import DEST, METHODS, RUN, independent_metrics, jobs
from experiments.minpi_compression.benchmark import digest


def audit():
    rows = []
    novel_rows = {}
    for (name, seed), (folder,row) in jobs().items():
        meta = json.loads((folder / row["train_relpath"]).with_name("split_metadata.json").read_bytes())
        names = json.loads(row["selected_features_json"])
        assert names == meta["selected_features"]
        splits = ("train", "val", "test")
        all_indices = np.concatenate([meta[s+"_indices"] for s in splits])
        assert np.array_equal(np.sort(all_indices), np.arange(int(row["raw_row_count"])))
        # Store binary records to check equality of the entire predictor vector, including all 100 columns.
        sets = []
        for s in splits:
            a = pd.read_csv(folder / row[s+"_relpath"], usecols=names)[names].to_numpy(dtype=np.float64)
            assert np.isfinite(a).all() and len(a) == len(meta[s+"_indices"])
            a[a == 0] = 0.0
            a = np.ascontiguousarray(a)
            sets.append(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel())
        train,val,test = sets
        overlap = np.isin(test, np.unique(np.concatenate((train,val))))
        novel_rows[f"{name}-{seed}"] = ~overlap
        rows.append(dict(dataset=name, seed=seed, p=len(names), test_rows=len(test),
                         rows_with_exact_dev_match=int(overlap.sum()), unique_original_ids=len(all_indices)))
    DEST.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / "split-overlap.csv",index=False)
    np.savez_compressed(DEST / "novel-input-rows.npz", **novel_rows)
    print(frame.groupby("dataset")[["test_rows","rows_with_exact_dev_match"]].mean().round(2))


def score():
    lock = json.loads((RUN / "lock.json").read_bytes())
    rows = []
    with np.load(DEST / "novel-input-rows.npz") as masks:
        assert len(lock) == 30 and set(masks.files) == set(lock)
        for (name,seed), (folder,row) in jobs().items():
            key = f"{name}-{seed}"
            run = RUN / key
            assert digest(run / "selection.json") == lock[key]
            state = json.loads((run / "selection.json").read_bytes())
            assert digest(run / "predictions.npz") == state["forecast"]
            path = folder / row["test_relpath"]
            assert digest(path) == state["inputs"]["test"]
            novel = masks[key]
            y = pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()[novel]
            with np.load(run / "predictions.npz") as predictions:
                for method in METHODS:
                    vals = independent_metrics(y,predictions[method][novel],state["task"],state["scale"])
                    rows.append(dict(dataset=name,seed=seed,task=state["task"],method=method,rows=len(y),**vals))
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / "novel-input-scores.csv", index=False)
    table = frame.groupby(["dataset","method"]).mean(numeric_only=True)
    diff = table.xs("minpi",level="method") - table.xs("gain",level="method")
    print(diff[["auc","accuracy","r2","mae"]].round(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit","score"])
    (audit if parser.parse_args().action == "audit" else score)()
