"""Seven untouched application sources for supervised coalition-risk validation."""
import argparse
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.minpi_compression.benchmark import metrics
from experiments.minpi_rework.validation_panel import DATA, SEEDS, _canonical_groups


MANIFEST = Path(__file__).with_name("validation_predictive_sources.json")
SOURCES = json.loads(MANIFEST.read_bytes())
CLASSIFICATION = {name for name, source in SOURCES.items() if source["task"] == "classification"}
TEMPORAL = {"beijing", "seoul"}
PARTS = ("fit", "vote", "width", "test")
LIMIT = 2000


def fetch():
    DATA.mkdir(exist_ok=True, parents=True)
    for name, conf in SOURCES.items():
        path = DATA / conf["filename"]
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != conf["sha256"]:
                raise ValueError(f"Unexpected local archive for {name}")
            continue
        req = Request(conf["url"], headers={"User-Agent": "MinPI research replication"})
        with urlopen(req, timeout=90) as res:
            raw = res.read()
        if sha256(raw).hexdigest() != conf["sha256"]:
            raise ValueError(f"Changed remote source for {name}")
        with NamedTemporaryFile(delete=False, dir=DATA) as tmp:
            tmp.write(raw)
        Path(tmp.name).replace(path)


def _records(name, archive):
    if name == "votes":
        frame = pd.read_csv(archive.open("house-votes-84.data"), header=None)
        y = frame.pop(0).map({"democrat": 0, "republican": 1})
        x = frame.apply(lambda col: col.map({"n": -1, "?": 0, "y": 1}))
    elif name == "churn":
        frame = pd.read_csv(archive.open("Customer Churn.csv"))
        y = frame.pop("Churn")
        x = frame.drop(columns=["Age Group", "Status", "Customer Value"])
    elif name == "liver":
        frame = pd.read_csv(archive.open("Indian Liver Patient Dataset (ILPD).csv"), header=None).dropna()
        code = frame.pop(10)
        assert set(code.unique()) == {1, 2}
        y = code.eq(1)
        x = frame.copy()
        x[1] = x[1].map({"Male": 1, "Female": 0})
    elif name == "german":
        frame = pd.read_csv(archive.open("SouthGermanCredit.asc"), sep=r"\s+")
        code = frame.pop("kredit")
        assert set(code.unique()) == {0, 1}
        y = code.eq(0)
        x = frame.drop(columns=["laufkont", "moral", "verw", "sparkont", "famges",
                                "buerge", "weitkred", "wohn"])
    elif name == "sgemm":
        frame = pd.read_csv(archive.open("sgemm_product.csv"))
        timings = [f"Run{i} (ms)" for i in range(1, 5)]
        assert (frame[timings] > 0).all().all()
        y = np.log(frame[timings].median(axis=1, skipna=False))
        x = frame.drop(columns=timings)
    else:
        raise KeyError(name)
    group = _canonical_groups(x, None)
    if len(x) > LIMIT:
        units, counts = np.unique(group, return_counts=True)
        order = np.random.default_rng(71117).permutation(len(units))
        chosen = units[order][counts[order].cumsum() <= LIMIT]
        keep = np.flatnonzero(np.isin(group, chosen))
        x, y, group = x.iloc[keep], y.iloc[keep], group[keep]
    return x, y, group, None


def _weather(name, archive):
    if name == "seoul":
        f = pd.read_csv(archive.open("SeoulBikeData.csv"), encoding="latin1")
        f.index = pd.to_datetime(f.Date, format="%d/%m/%Y") + pd.to_timedelta(f.Hour, unit="h")
        assert f.columns[0:3].tolist() == ["Date", "Rented Bike Count", "Hour"]
        assert len(f.columns[3:11]) == 8 and f.columns[11] == "Seasons"
        x = f.loc[f.Hour.isin((0, 6, 12, 18)), f.columns[3:11]].copy()
        lag = pd.Timedelta(hours=1)
        target_time = x.index + lag
        x["target_hour"], x["weekday"], x["month"] = target_time.hour, target_time.dayofweek, target_time.month
        target = f["Rented Bike Count"]
    else:
        member = "PRSA2017_Data_20130301-20170228.zip"
        station = "PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
        with ZipFile(BytesIO(archive.read(member))) as nested:
            f = pd.read_csv(nested.open(station))
        f.index = pd.to_datetime(f[["year", "month", "day", "hour"]])
        x = f.loc[f.hour.eq(12), ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].copy()
        compass = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
        direction = f.loc[x.index, "wd"].map({val: i * np.pi / 8 for i, val in enumerate(compass)})
        x["wind_sin"], x["wind_cos"] = np.sin(direction), np.cos(direction)
        lag = pd.Timedelta(hours=24)
        target = f["PM2.5"]

    assert f.index.is_unique and f.index.is_monotonic_increasing
    y = pd.Series(target.reindex(x.index + lag).to_numpy(), index=x.index)
    keep = x.notna().all(axis=1) & y.notna()
    x, y = x.loc[keep], y.loc[keep]
    dates = x.index.to_numpy(dtype="datetime64[D]").astype("int64")
    response_dates = (x.index + lag).to_numpy(dtype="datetime64[D]").astype("int64")
    match = _canonical_groups(x, None)
    unique_days_per_vector = pd.Series(dates).groupby(match).transform("nunique").to_numpy()
    keep = unique_days_per_vector == 1
    x, y, dates, response_dates = x.loc[keep], y.loc[keep], dates[keep], response_dates[keep]
    unique = np.unique(dates)
    cuts = unique[[int(len(unique) * frac) for frac in (.48, .64, .8)]]
    slot = np.searchsorted(cuts, dates, side="right")
    keep = np.searchsorted(cuts, response_dates, side="right") == slot
    x, y, dates, slot = x.loc[keep], y.loc[keep], dates[keep], slot[keep]
    for boundary in range(3):
        assert response_dates[keep][slot == boundary].max() < dates[slot == boundary + 1].min()
    return x, y, dates, slot


@lru_cache
def _prepared(name):
    conf = SOURCES[name]
    raw = (DATA / conf["filename"]).read_bytes()
    if sha256(raw).hexdigest() != conf["sha256"]:
        raise ValueError(f"Unregistered archive: {name}")
    with ZipFile(BytesIO(raw)) as archive:
        x, y, group, slot = _weather(name, archive) if name in TEMPORAL else _records(name, archive)
    y = y.to_numpy(float if conf["task"] == "regression" else int)
    x = x.to_numpy(float)
    assert len(y) == len(x) == len(group) and len(x) <= LIMIT
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y, conf["task"], group, slot


def _data(name):
    return _prepared(name)[:4]


def group_ids(name):
    return _data(name)[3].copy()


def _partition(name, seed):
    if seed not in SEEDS:
        raise ValueError(f"Unregistered seed {seed}")
    x, y, task, group, slot = _prepared(name)
    if slot is not None:
        idx = {k: np.flatnonzero(slot == i) for i, k in enumerate(PARTS)}
    else:
        units, first, inverse = np.unique(group, return_index=True, return_inverse=True)
        def partition(pool, portion, salt):
            strata = y[first[pool]] if task == "classification" else None
            return train_test_split(pool, test_size=portion, stratify=strata, random_state=seed + salt)
        dev, test = partition(np.arange(len(units)), .2, 0)
        joint, width = partition(dev, .2, 101)
        fit, vote = partition(joint, .25, 201)
        idx = {k: np.flatnonzero(np.isin(inverse, u)) for k, u in zip(PARTS, (fit, vote, width, test))}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(len(y)))
    if task == "classification":
        assert all(len(np.unique(y[rows])) == 2 for rows in idx.values())
    return x, y, task, idx


def split(name, seed):
    """Selection-only interface; every outer y is replaced with a sentinel."""
    x, y, task, idx = _partition(name, seed)
    masked = y.copy()
    masked[idx["test"]] = -1 if task == "classification" else np.nan
    return x.copy(), masked, task, idx


def score(name, seed, forecast_path, frozen_sha256):
    """After a seal, score only forecasts matching the registered SHA-256."""
    raw = Path(forecast_path).read_bytes()
    if sha256(raw).hexdigest() != frozen_sha256:
        raise ValueError("Wrong forecast digest")
    _, y, task, idx = _partition(name, seed)
    dev = np.concatenate([idx[k] for k in PARTS[:-1]])
    scale = max(float(np.std(y[dev])), 1e-12)
    with np.load(BytesIO(raw), allow_pickle=False) as preds:
        return {arm: metrics(y[idx["test"]], preds[arm], task, scale) for arm in preds.files}


def receipt():
    record = {}
    for name, conf in SOURCES.items():
        groups = group_ids(name)
        splits = {}
        for seed in SEEDS:
            x, masked, task, idx = split(name, seed)
            hidden = masked[idx["test"]]
            assert (hidden == -1).all() if task == "classification" else np.isnan(hidden).all()
            split_groups = [set(groups[rows]) for rows in idx.values()]
            assert sum(map(len, split_groups)) == len(set.union(*split_groups))
            splits[seed] = {k: {"n": len(v), "groups": len(set(groups[v])),
                                "sha256": sha256(v.astype("<i8").tobytes()).hexdigest()}
                            for k, v in idx.items()}
        record[name] = dict(rows=len(x), columns=x.shape[1], groups=len(set(groups)),
                            source_sha256=conf["sha256"], seeds=splits)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args()
    if args.fetch:
        fetch()
    audit = receipt()
    for name, v in audit.items():
        print(name, v["rows"], v["columns"], v["groups"])
    if args.write_receipt:
        args.write_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.write_receipt.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
