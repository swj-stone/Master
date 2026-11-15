"""Prepare three stratified splits per source, withholding test targets from fitting."""
from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.context_review.transport_data import DATA, EVIDENCE, RUN, SPEC, read
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest


MEMBERS = {
    "banknote": "data_banknote_authentication.txt", "blood": "transfusion.data",
    "dry_bean": "DryBeanDataset/Dry_Bean_Dataset.arff", "htru2": "HTRU_2.csv",
    "mammographic": "mammographic_masses.data", "qsar": "biodeg.csv",
    "rice": "Rice_Cammeo_Osmancik.arff", "seeds": "seeds_dataset.txt",
    "steel": "Faults.NNA", "vertebral": "column_3C.dat",
}


def load_rows(name):
    with ZipFile(DATA / f"{name}.zip") as archive:
        raw = archive.read(MEMBERS[name])
    if MEMBERS[name].endswith(".arff"):
        records, _ = loadarff(StringIO(raw.decode("latin1")))
        frame = pd.DataFrame.from_records(records)
        label = frame.pop(frame.columns[-1]).str.decode("utf8")
    else:
        sep = r"\s+" if name in {"seeds", "steel", "vertebral"} else ";" if name == "qsar" else ","
        frame = pd.read_csv(BytesIO(raw), sep=sep, header=0 if name == "blood" else None,
                            na_values=["?"], encoding="utf8")
        if name == "steel":
            targets = frame.iloc[:, -7:].to_numpy()
            assert np.isin(targets, [0, 1]).all() and (targets.sum(axis=1) == 1).all()
            label = pd.Series(targets.argmax(axis=1))
            frame = frame.iloc[:, :-7]
        else:
            label = frame.pop(frame.columns[-1])
    assert label.notna().all()
    x = frame.to_numpy(dtype=np.float32)
    assert np.isfinite(x[~np.isnan(x)]).all()
    return x, label.to_numpy()


def prepare():
    sources = {r["dataset"]: r for r in read(EVIDENCE / "sources.json")}
    for name, number, _, n, p in read(SPEC)["sources"]:
        archive_hash = digest(DATA / f"{name}.zip")
        assert archive_hash == sources[name]["sha256"]
        x, labels = load_rows(name)
        assert x.shape == (n, p), (name, x.shape, n, p)
        feature_hash = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
        outer = StratifiedKFold(3, shuffle=True, random_state=20261115)
        for fold, (dev, test) in enumerate(outer.split(x, labels)):
            root = RUN / name / f"fold-{fold}"
            if (root / "selection.json").exists():
                raise ValueError("Prepare only before fitting: this fold already has frozen selections.")
            root.mkdir(parents=True, exist_ok=True)
            d = np.arange(len(dev))
            train, rank = train_test_split(d, test_size=.25, random_state=20421+fold,
                                          stratify=labels[dev])
            classes, counts = np.unique(labels[dev[train]], return_counts=True)
            if len(classes) == 2 and counts[0] < counts[1]:
                classes = classes[::-1]
            code = {label: i for i, label in enumerate(classes)}
            encoded_dev = np.array([code[g] for g in labels[dev]])
            proposal, guard = train_test_split(rank, test_size=.5, random_state=39823+fold,
                                               stratify=encoded_dev[rank])
            assert all(np.array_equal(np.unique(encoded_dev[rows]), np.arange(len(classes)))
                       for rows in (train, proposal, guard))
            assert np.array_equal(np.sort(np.r_[train, proposal, guard]), d)
            assert np.array_equal(np.sort(np.r_[dev, test]), np.arange(n))
            np.savez_compressed(root / "inputs.npz", x=x[dev], y=encoded_dev, x_outer=x[test],
                                train=train, proposal=proposal, guard=guard, test_row=test)
            np.savez_compressed(root / "targets.npz", test_row=test,
                                y=np.array([code[g] for g in labels[test]]))
            store(root / "layout.json", dict(dataset=name, id=number, fold=fold, classes=classes.tolist(),
                  source=archive_hash, member=MEMBERS[name], n=n, p=p,
                  train=dev[train].tolist(), proposal=dev[proposal].tolist(), guard=dev[guard].tolist(),
                  test=test.tolist(), inputs_sha256=digest(root / "inputs.npz"),
                  targets_sha256=digest(root / "targets.npz"),
                  missing=int(np.isnan(x).sum()), duplicated_inputs=int(n-len(np.unique(feature_hash))),
                  overlap_test_rows=int(np.isin(feature_hash[test], feature_hash[dev]).sum())))
        print(f"{name}: {n} rows, {p} columns, three splits frozen", flush=True)


if __name__ == "__main__":
    prepare()
