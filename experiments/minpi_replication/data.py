"""Four prespecified UCI transfer sources with hidden outer labels."""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from .fetch import MANIFEST, RAW

SOURCES = ("posture", "bankruptcy", "turbine", "forecast")
SEEDS = tuple(range(42, 47))
PARTS = ("fit", "vote", "width", "test")


@dataclass
class Records:
    x: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    columns: tuple
    task: str


@lru_cache(maxsize=4)
def _records(name):
    conf = json.loads(MANIFEST.read_text())[name]
    raw = RAW / (name + ".zip")
    if sha256(raw.read_bytes()).hexdigest() != conf["sha256"]:
        raise ValueError(f"Unrecognized source: {raw}")
    with ZipFile(raw) as bundle:
        if name == "posture":
            data = pd.read_csv(bundle.open("allUsers.lcl.csv"), na_values=["?"])
            data = data.loc[data.Class != 0]
            # Source-ordered thinning within each person, independent of posture labels.
            rows = [indices[np.linspace(0, len(indices)-1, min(200,len(indices)), dtype=int)]
                    for indices in data.groupby("User", sort=True).indices.values()]
            data = data.iloc[np.concatenate(rows)].copy()
            y = data.pop("Class").to_numpy(int)-1
            group = data.pop("User").to_numpy(int)
        elif name == "bankruptcy":
            data = pd.read_csv(bundle.open("data.csv"))
            y = data.pop("Bankrupt?").to_numpy(int)
            data.columns = data.columns.str.strip()
            group = pd.util.hash_pandas_object(data, index=False).to_numpy()
        elif name == "turbine":
            frames = []
            for year in range(2011,2016):
                block = pd.read_csv(bundle.open(f"gt_{year}.csv")).iloc[::10].copy()
                block["year"] = year
                frames.append(block)
            data = pd.concat(frames, ignore_index=True)
            group = data.pop("year").to_numpy()
            y = data.pop("CO").to_numpy(float)
            data = data.drop(columns="NOX")
        elif name == "forecast":
            data = pd.read_csv(bundle.open("Bias_correction_ucl.csv"))
            data = data.dropna(subset=["Date","Next_Tmax"]).copy()
            group = pd.to_datetime(data.pop("Date")).dt.year.to_numpy()
            y = data.pop("Next_Tmax").to_numpy(float)
            data = data.drop(columns=["station", "Next_Tmin"])
        else:
            raise KeyError(name)
    task = "classification" if name in ("posture","bankruptcy") else "regression"
    x = data.to_numpy(dtype=np.float32)
    if np.isinf(x).any() or not np.isfinite(y).all() or not data.columns.is_unique:
        raise ValueError(f"Invalid records in {name}")
    return Records(x, y, group, tuple(data.columns), task)


def _indices(name, seed, groups):
    if seed not in SEEDS:
        raise ValueError("Registered seeds are 42–46")
    unique = np.unique(groups)
    if name in ("turbine", "forecast"):
        if len(unique) != 5 or np.any(np.diff(unique) != 1):
            raise ValueError("Expected five consecutive observation years")
        units = [unique[:2], unique[2:3], unique[3:4], unique[4:]]
    else:
        ordered = np.random.default_rng(seed).permutation(unique)
        cuts = np.round(len(unique)*np.array([.48,.64,.8])).astype(int)
        units = np.split(ordered, cuts)
    return {key:np.flatnonzero(np.isin(groups, units[i])) for i,key in enumerate(PARTS)}


def split(name, seed=42):
    records = _records(name)
    idx = _indices(name, seed, records.groups)
    y = records.target.copy()
    y[idx["test"]] = -1 if records.task=="classification" else np.nan
    return records.x.copy(), y, records.task, idx


def feature_names(name):
    return _records(name).columns


def group_ids(name):
    return _records(name).groups.copy()


def audit():
    for name in SOURCES:
        for seed in SEEDS:
            x,y,task,idx = split(name,seed)
            groups = group_ids(name)
            np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))),np.arange(len(y)))
            parts = [set(groups[ix]) for ix in idx.values()]
            assert all(parts) and sum(map(len,parts))==len(set.union(*parts))
            if task=="classification":
                assert np.all(y[idx["test"]]==-1)
                expected = set(range(5 if name=="posture" else 2))
                assert all(set(y[idx[key]])==expected for key in PARTS[:3])
            else:
                assert np.isnan(y[idx["test"]]).all()
            if seed==42:
                print(name,x.shape,[len(idx[p]) for p in PARTS],
                      "absent entries",int(np.isnan(x).sum()),"groups",[len(g) for g in parts])


if __name__ == "__main__":
    audit()
