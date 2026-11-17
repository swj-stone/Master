"""Export the 30 locked split traces and independently rescore the portable forecasts."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from experiments.minpi_compression import legacy_gain as study
from experiments.minpi_compression.benchmark import digest


def export():
    lock = json.loads((study.RUN / "lock.json").read_bytes())
    stage = study.DEST / "forecasts.partial"
    with ZipFile(stage, "w", ZIP_DEFLATED) as archive:
        archive.write(study.RUN / "lock.json", "lock.json")
        for (name, seed), (source, row) in study.jobs().items():
            key = f"{name}-{seed}"
            folder = study.RUN / key
            assert digest(folder / "selection.json") == lock[key]
            state = json.loads((folder / "selection.json").read_bytes())
            assert digest(folder / "predictions.npz") == state["forecast"]
            for name in ("selection.json","predictions.npz"):
                archive.write(folder/name, key+"/"+name)
            targets = source / row["test_relpath"]
            assert digest(targets) == state["inputs"]["test"]
            y = pd.read_csv(targets, usecols=[row["target"]]).to_numpy().ravel()
            payload = BytesIO()
            np.save(payload, y, allow_pickle=False)
            archive.writestr(key + "/targets.npy", payload.getvalue())
    stage.replace(study.DEST / "forecasts.zip")


def verify():
    table = pd.read_csv(study.DEST / "scores.csv").set_index(["dataset","seed","method"])
    with ZipFile(study.DEST / "forecasts.zip") as archive:
        lock = json.loads(archive.read("lock.json"))
        assert len(lock) == 30 and len(archive.namelist()) == 91 and archive.testzip() is None
        for key, expected in lock.items():
            meta = archive.read(key + "/selection.json")
            assert sha256(meta).hexdigest() == expected
            state = json.loads(meta)
            pred = archive.read(key + "/predictions.npz")
            assert sha256(pred).hexdigest() == state["forecast"]
            y = np.load(BytesIO(archive.read(key + "/targets.npy")), allow_pickle=False)
            with np.load(BytesIO(pred)) as preds:
                for method in study.METHODS:
                    row = table.loc[(state["dataset"], state["seed"], method)]
                    runs = [r for r in state["rows"] if r["method"] == method]
                    assert len(runs) == 3 and all(row.k == r["k"] for r in runs)
                    for m in ("rank_s","selection_s","deployment_s"):
                        np.testing.assert_allclose(np.median([r[m] for r in runs]), row[m], atol=1e-12,rtol=0.)
                    vals = study.independent_metrics(y,preds[method],state["task"],state["scale"])
                    for m, v in vals.items():
                        np.testing.assert_allclose(v,row[m],atol=1e-12,rtol=0.)
    print(f"Verified 120 scores, supports' widths and three-repeat clocks on {len(lock)} sealed splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export","verify"))
    if parser.parse_args().action == "export":
        export()
    verify()
