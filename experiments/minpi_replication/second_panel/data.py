"""Five prospectively specified deployment tasks with masked outer labels."""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2] / "data/minpi_replication_r3"
MANIFEST = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
SOURCES = ("vowel", "hcv", "shoppers", "garment", "tetouan")
SEEDS = tuple(range(42, 47))
PARTS = ("fit", "vote", "width", "test")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, spec in MANIFEST.items():
        path = RAW / (name+".zip")
        if not path.exists():
            with urlopen(Request(spec["url"], headers={"User-Agent":"academic-replication/1"}), timeout=90) as req:
                content = req.read()
            if sha256(content).hexdigest() != spec["sha256"]:
                raise ValueError(f"Unexpected download for {name}")
            path.write_bytes(content)
        if sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Local archive failed digest check: {path}")


@dataclass
class Records:
    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    columns: tuple
    original_rows: np.ndarray
    task: str


@lru_cache(maxsize=5)
def _read(name):
    if name not in SOURCES:
        raise KeyError(name)
    path = RAW / (name+".zip")
    if sha256(path.read_bytes()).hexdigest() != MANIFEST[name]["sha256"]:
        raise ValueError(f"Unrecognized source {name}")
    task = "classification" if name in {"hcv","shoppers","vowel"} else "regression"
    with ZipFile(path) as ar:
        if name == "vowel":
            raw = pd.read_csv(ar.open("vowel-context.data"), sep=r"\s+", header=None)
            group = raw[1]
            if not np.array_equal(raw[0].values, (group.values >= 8).astype(int)):
                raise ValueError("Published speaker and outer split disagree")
            cols = raw.iloc[:,3:13].set_axis([f"lpc{j}" for j in range(10)], axis=1)
            y = raw[13]
        elif name == "hcv":
            raw = pd.read_csv(ar.open("hcvdat0.csv"))
            if not raw.iloc[:,0].is_unique:
                raise ValueError("Patient IDs should be unique")
            raw = raw.loc[~raw.Category.str.startswith("0s=")]
            y = (raw.Category != "0=Blood Donor").astype(int)
            cols = raw.loc[:, ["Age","ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]]
            group = pd.util.hash_pandas_object(cols,index=False)
        elif name == "shoppers":
            raw = pd.read_csv(ar.open("online_shoppers_intention.csv"))
            selected = np.random.default_rng(617).choice(len(raw), size=min(8192,len(raw)), replace=False)
            raw = raw.iloc[np.sort(selected)]
            y = raw.Revenue.astype(int)
            cols = raw.loc[:,["Administrative","Administrative_Duration","Informational",
                             "Informational_Duration","ProductRelated","ProductRelated_Duration",
                             "SpecialDay","Weekend"]]
            months = ("Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec")
            group = raw.Month.map({m:i+1 for i,m in enumerate(months)})
        elif name == "garment":
            raw = pd.read_csv(ar.open("garments_worker_productivity.csv"))
            dates = pd.to_datetime(raw.date, format="%m/%d/%Y")
            department = raw.department.str.strip().map({"sweing":0, "sewing":0, "finishing":1})
            if department.isna().any():
                raise ValueError("Unknown department")
            cols = raw.loc[:,["targeted_productivity","smv","no_of_style_change","no_of_workers"]].assign(
                weekday=dates.dt.dayofweek, month_day=dates.dt.day, department=department)
            group = dates.dt.normalize()
            y = raw.actual_productivity
        else:
            raw = pd.read_csv(ar.open("Tetuan City power consumption.csv"))
            dates = pd.to_datetime(raw.DateTime, format="%m/%d/%Y %H:%M")
            if not dates.is_unique:
                raise ValueError("Repeated source timestamp")
            raw = raw.loc[dates.dt.minute == 0]
            dates = dates.loc[raw.index]
            cols = raw.loc[:,["Temperature","Humidity","Wind Speed","general diffuse flows","diffuse flows"]].copy()
            for key, period, values in (("hour",24,dates.dt.hour),("weekday",7,dates.dt.dayofweek)):
                angle = 2*np.pi*values/period
                cols[key+"_sin"] = np.sin(angle)
                cols[key+"_cos"] = np.cos(angle)
            group = dates.dt.normalize()
            y = raw["Zone 1 Power Consumption"]
    if pd.isna(group).any() or not np.isfinite(y).all() or not cols.index.equals(y.index):
        raise ValueError(f"Invalid source records for {name}")
    return Records(cols.to_numpy(dtype=float), y.to_numpy(dtype=int if task=="classification" else float),
                   group.to_numpy(), tuple(cols.columns), cols.index.to_numpy(), task)


def _parts(name, seed, group):
    if seed not in SEEDS:
        raise ValueError("Unregistered seed")
    units = np.unique(group)
    rng = np.random.default_rng(seed)
    if name == "vowel":
        train = rng.permutation(units[units<8])
        pieces = [train[:4], train[4:6], train[6:], units[units>=8]]
    elif name == "shoppers":
        pieces = [units[units<=9], [10], [11], [12]]
    else:
        if name == "hcv":
            units = rng.permutation(units)
        pieces = np.split(units, np.rint(len(units)*np.array([.48,.64,.80])).astype(int))
    return {key:np.flatnonzero(np.isin(group,piece)) for key,piece in zip(PARTS,pieces,strict=True)}


def split(name,seed=42):
    rec = _read(name)
    idx = _parts(name,seed,rec.group)
    y = rec.y.copy()
    y[idx["test"]] = -1 if rec.task=="classification" else np.nan
    return rec.x.copy(), y, rec.task, idx


def truth(name,seed=42):
    """Test responses in exactly the indices returned by split; use after forecast lock."""
    rec = _read(name)
    return rec.y[_parts(name,seed,rec.group)["test"]].copy()


def feature_names(name):
    return _read(name).columns


def row_ids(name):
    return _read(name).original_rows.copy()


def group_ids(name):
    return _read(name).group.copy()


def audit():
    for name in SOURCES:
        for seed in SEEDS:
            x,y,task,idx = split(name,seed)
            groups = group_ids(name)
            partitions = [set(groups[idx[p]]) for p in PARTS]
            if sum(map(len,partitions)) != len(set.union(*partitions)):
                raise ValueError(f"Overlapping groups for {name}/{seed}")
            np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))),np.arange(len(y)))
            assert len(np.unique(row_ids(name))) == len(y)
            if task == "classification":
                assert (y[idx["test"]] == -1).all()
                expected = set(range(11 if name=="vowel" else 2))
                assert all(set(y[idx[part]])==expected for part in PARTS[:3])
            else:
                assert np.isnan(y[idx["test"]]).all()
            if seed==42:
                print(name, x.shape, [len(idx[p]) for p in PARTS], "units", list(map(len,partitions)))


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch",action="store_true")
    if p.parse_args().fetch:
        fetch()
    audit()
