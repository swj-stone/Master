"""Acquire the six source-fixed archives and construct isolated development/outer files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import subprocess
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.context_review.transport_data import ROOT, read
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest

SPEC = ROOT / "configs/context_review/wide_panel.json"
DATA = ROOT / "data/context_study/wide-uci"
RUN = ROOT / "results/context_review/wide-uci"
EVIDENCE = ROOT / "thesis-paper/evidence/wide"


def download(item):
    name, meta = item
    url = f"https://archive.ics.uci.edu/static/public/{meta['id']}/{meta['slug']}.zip"
    path = DATA / f"{name}.zip"
    if not path.exists():
        request = Request(url, headers={"User-Agent": "Academic feature-selection replication"})
        for attempt in range(6):
            try:
                with urlopen(request, timeout=45) as response:
                    raw = response.read()
                break
            except OSError as exc:
                print(name, type(exc).__name__, "retry", attempt + 1, flush=True)
                if attempt == 5:
                    raise
                sleep(attempt + 1)
        with ZipFile(BytesIO(raw)) as archive:
            assert archive.testzip() is None
        staged = path.with_suffix(".partial")
        staged.write_bytes(raw)
        staged.replace(path)
    with ZipFile(path) as archive:
        members = archive.namelist()
    print(name, path.stat().st_size, members, flush=True)
    return dict(dataset=name, url=url, id=meta["id"], doi=meta["doi"], creator=meta["attribution"],
                license="CC BY 4.0", bytes=path.stat().st_size, sha256=digest(path), members=members)


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, read(SPEC)["sources"].items()))
    path = EVIDENCE / "sources.json"
    if path.exists():
        assert read(path) == manifest, "Published source bytes changed"
    store(path, manifest)


def load_rows(name):
    """Read source targets only to build fixed partitions and separate target files."""
    dates = official = None
    with ZipFile(DATA / f"{name}.zip") as archive:
        if name == "cnae":
            frame = pd.read_csv(archive.open("CNAE-9.data"), header=None)
            y = frame.pop(0)
        elif name == "ozone":
            frame = pd.read_csv(archive.open("eighthr.data"), header=None, na_values="?")
            dates = pd.to_datetime(frame.pop(0), format="%m/%d/%Y").to_numpy()
            y = frame.pop(frame.columns[-1]).astype(int)
        elif name == "parkinson":
            # libarchive requires a seekable file when reading this RAR.
            inner = DATA / "pd_speech_features.rar"
            inner.write_bytes(archive.read("pd_speech_features.rar"))
            csv = subprocess.run(["tar", "-xOf", str(inner), "pd_speech_features.csv"],
                                 check=True, capture_output=True).stdout
            frame = pd.read_csv(BytesIO(csv), header=1)
            assert len(frame) == 756 and frame.groupby("id").size().eq(3).all()
            assert frame.groupby("id")["class"].nunique().eq(1).all()
            y = frame.groupby("id", sort=True)["class"].first()
            frame = frame.drop(columns=["gender", "class"]).groupby("id", sort=True).mean()
            assert len(frame) == 252 and frame.index.equals(y.index)
        elif name == "secom":
            frame = pd.read_csv(archive.open("secom.data"), header=None, sep=r"\s+")
            labels = pd.read_csv(archive.open("secom_labels.data"), header=None, sep=" ")
            y = labels[0]
            dates = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S").to_numpy()
        elif name == "semeion":
            frame = pd.read_csv(archive.open("semeion.data"), header=None, sep=r"\s+")
            labels = frame.iloc[:, -10:].to_numpy()
            assert np.isin(labels, [0, 1]).all() and (labels.sum(1) == 1).all()
            frame, y = frame.iloc[:, :-10], pd.Series(labels.argmax(1))
        elif name == "urban":
            training, testing = [pd.read_csv(archive.open(f"urban+land+cover/{part}.csv"))
                                 for part in ("training", "testing")]
            assert set(training.columns) == set(testing.columns)
            official = len(training)
            frame = pd.concat([training, testing[training.columns]], ignore_index=True)
            y = frame.pop("class")
        else:
            raise ValueError(name)
    x = frame.to_numpy(dtype=np.float32)
    assert len(x) == len(y) and not y.isna().any()
    assert x.shape[1] >= 50 and np.isfinite(x[~np.isnan(x)]).all()
    assert dates is None or (len(dates) == len(y) and not pd.isna(dates).any())
    return x, y.to_numpy(), dates, official, list(map(str, frame.columns))


def splits(labels, method, dates, official):
    if method == "stratified":
        return list(StratifiedKFold(3, shuffle=True, random_state=20261115)
                    .split(np.arange(len(labels)), labels))
    if method == "official":
        return [(np.arange(official), np.arange(official, len(labels)))]
    ticks = np.unique(dates)
    cuts = [2 * len(ticks) // 5, 3 * len(ticks) // 5, 4 * len(ticks) // 5, len(ticks)]
    return [(np.flatnonzero(np.isin(dates, ticks[:a])), np.flatnonzero(np.isin(dates, ticks[a:b])))
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def inner(labels, dates, fold):
    ids = np.arange(len(labels))
    if dates is None:
        train, rank = train_test_split(ids, test_size=.25, random_state=20421 + fold, stratify=labels)
        proposal, guard = train_test_split(rank, test_size=.5, random_state=39823 + fold,
                                          stratify=labels[rank])
    else:
        ticks = np.unique(dates)
        a, b = 3 * len(ticks) // 4, 7 * len(ticks) // 8
        train, proposal, guard = [np.flatnonzero(np.isin(dates, t))
                                  for t in (ticks[:a], ticks[a:b], ticks[b:])]
    assert np.array_equal(np.sort(np.r_[train, proposal, guard]), ids)
    return train, proposal, guard


def prepare():
    sources = {row["dataset"]: row for row in read(EVIDENCE / "sources.json")}
    status = {}
    for name, spec in read(SPEC)["sources"].items():
        sha = digest(DATA / f"{name}.zip")
        assert sha == sources[name]["sha256"]
        x, y, dates, official, columns = load_rows(name)
        outer = splits(y, spec["split"], dates, official)
        assert len(outer) == spec["folds"]
        hashes = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
        for fold, (dev, test) in enumerate(outer):
            key = f"{name}/fold-{fold}"
            dest = RUN / key
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "selection.json").exists():
                raise ValueError("Cannot regenerate a sealed layout")
            assert not np.intersect1d(dev, test).size
            train, prop, guard = inner(y[dev], None if dates is None else dates[dev], fold)
            roles = {"train": dev[train], "proposal": dev[prop], "guard": dev[guard], "test": test}
            if dates is not None:
                for a, b in (("train", "proposal"), ("proposal", "guard"), ("guard", "test")):
                    assert dates[roles[a]].max() < dates[roles[b]].min()
            classes, counts = np.unique(y[dev[train]], return_counts=True)
            missing = [key for key, rows in roles.items()
                       if not np.array_equal(np.unique(y[rows]), np.unique(y))]
            if missing:
                status[key] = {"feasible": False, "reason": "Missing class in " + ", ".join(missing)}
                continue
            if len(classes) == 2 and counts[0] < counts[1]:
                classes = classes[::-1]
            codes = {value: i for i, value in enumerate(classes)}
            np.savez_compressed(dest / "inputs.npz", x=x[dev], y=[codes[v] for v in y[dev]],
                                x_outer=x[test], train=train, proposal=prop, guard=guard, test_row=test)
            np.savez_compressed(dest / "targets.npz", test_row=test, y=[codes[v] for v in y[test]])
            time_ranges = {} if dates is None else {
                role: [str(dates[ids].min()), str(dates[ids].max())] for role, ids in roles.items()}
            layout = dict(dataset=name, fold=fold, id=spec["id"], source=sha, n=len(x), p=x.shape[1],
                          classes=classes.tolist(), features=columns, split=spec["split"],
                          time_ranges=time_ranges, missing=int(np.isnan(x).sum()),
                          duplicated_inputs=int(len(x) - len(np.unique(hashes))),
                          overlap_test_rows=int(np.isin(hashes[test], hashes[dev]).sum()),
                          inputs_sha256=digest(dest / "inputs.npz"),
                          targets_sha256=digest(dest / "targets.npz"),
                          **{role: ids.tolist() for role, ids in roles.items()})
            store(dest / "layout.json", layout)
            status[key] = {"feasible": True, "layout": digest(dest / "layout.json")}
        print(name, x.shape, len(outer), "planned folds", flush=True)
    store(EVIDENCE / "fold-status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["acquire", "prepare"])
    {"acquire": acquire, "prepare": prepare}[parser.parse_args().stage]()
