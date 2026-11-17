"""Five application sources with an outer-label-masked selection interface."""
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


MANIFEST = Path(__file__).with_name("validation_exchange_sources.json")
SOURCES = json.loads(MANIFEST.read_bytes())
CLASSIFICATION = {name for name, source in SOURCES.items() if source["task"] == "classification"}
GROUPED = {"ctg", "telemonitor"}
PARTS = ("fit", "vote", "width", "test")


def fetch():
    """Install only the registered source bytes into the ignored data directory."""
    DATA.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        path = DATA / source["filename"]
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"Changed local source: {name}")
            continue
        req = Request(source["url"], headers={"User-Agent": "MinPI research replication"})
        with urlopen(req, timeout=90) as response:
            raw = response.read()
        if sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Changed source at registered URL: {name}")
        with NamedTemporaryFile(dir=DATA, delete=False) as tmp:
            tmp.write(raw)
        Path(tmp.name).replace(path)


def _ctg(archive):
    import xlrd

    book = xlrd.open_workbook(file_contents=archive.read("CTG.xls"))
    data, raw = book.sheet_by_name("Data"), book.sheet_by_name("Raw Data")
    columns = data.row_values(1, 10, 31)
    stop = 2 + 2126
    assert columns[:4] == ["LB", "AC", "FM", "UC"] and columns[-2:] == ["Variance", "Tendency"]
    assert raw.row_values(0, 0, 5) == ["FileName", "Date", "SegFile", "b", "e"]
    assert data.cell_value(1, 45) == "NSP"
    x = pd.DataFrame([data.row_values(i, 10, 31) for i in range(2, stop)], columns=columns)
    y = np.asarray(data.col_values(45, 2, stop), dtype=int) - 1
    # Segment boundaries and consensus labels independently verify the cross-sheet alignment.
    for data_col, raw_col in [(0, 3), (1, 4), (45, 39)]:
        np.testing.assert_array_equal(data.col_values(data_col, 2, stop), raw.col_values(raw_col, 2, stop))
    ids = np.asarray([s.strip().casefold() for s in raw.col_values(0, 2, stop)])
    assert len(np.unique(raw.col_values(2, 2, stop))) == len(y) and np.all(ids != "")
    return x, y, ids


def _taiwan(archive):
    raw = archive.read("default of credit card clients.xls")
    f = pd.read_excel(BytesIO(raw), header=1, engine="xlrd")
    assert len(f) == f.ID.nunique() == 30000
    y = f.pop("default payment next month").to_numpy(int)
    x = f.drop(columns=["ID", "SEX", "EDUCATION", "MARRIAGE"])
    return x, y, None


def _telemonitor(archive):
    f = pd.read_csv(archive.open("parkinsons_updrs.data"))
    x = f.drop(columns=["subject#", "test_time", "total_UPDRS", "motor_UPDRS"])
    assert f["subject#"].nunique() == 42
    return x, f.motor_UPDRS.to_numpy(float), f["subject#"]


def _studentmath(archive):
    with ZipFile(BytesIO(archive.read("student.zip"))) as inner:
        f = pd.read_csv(inner.open("student-mat.csv"), sep=";")
    x = f.drop(columns=["G1", "G2", "G3", "absences", "Mjob", "Fjob", "reason", "guardian"])
    positive = {"school": "MS", "sex": "M", "address": "U", "famsize": "GT3", "Pstatus": "T"}
    for column in x.select_dtypes(exclude=np.number):
        levels = set(x[column].unique())
        high = positive.get(column, "yes")
        assert len(levels) == 2 and high in levels
        x[column] = x[column].eq(high).astype(int)
    return x, f.G3.to_numpy(float), None


def _tecator(raw):
    lines = raw.decode("ascii").splitlines()
    values = np.loadtxt(lines[-240 * 25:])
    assert values.shape == (240 * 25, 5)
    table = values.reshape(240, 125)[:215]
    # Last columns are moisture, fat, protein. Use the original raw wavelength grid.
    return pd.DataFrame(table[:, :100]), table[:, 123].copy(), None


@lru_cache
def _data(name):
    source = SOURCES[name]
    raw = (DATA / source["filename"]).read_bytes()
    if sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"Unregistered data: {name}")
    if name == "tecator":
        x, y, ids = _tecator(raw)
    else:
        readers = {"ctg": _ctg, "taiwan": _taiwan, "telemonitor": _telemonitor, "studentmath": _studentmath}
        with ZipFile(BytesIO(raw)) as archive:
            x, y, ids = readers[name](archive)
    groups = _canonical_groups(x, ids)
    if name == "taiwan":
        unique, counts = np.unique(groups, return_counts=True)
        order = np.random.default_rng(71117).permutation(len(unique))
        chosen = unique[order][np.cumsum(counts[order]) <= 4000]
        keep = np.flatnonzero(np.isin(groups, chosen))
        x, y, groups = x.iloc[keep], y[keep], groups[keep]
    if name == "tecator":
        # Preserve every official test specimen and purge matching development spectra.
        keep = np.r_[~np.isin(groups[:172], groups[172:]), np.ones(43, dtype=bool)]
        x, y, groups = x.loc[keep], y[keep], groups[keep]
    features = x.to_numpy(float)
    assert features.shape == (len(y), source["columns"])
    assert np.isfinite(features).all() and np.isfinite(y).all()
    return features, y, source["task"], groups


def _partition(name, seed):
    if seed not in SEEDS:
        raise ValueError(f"Unregistered seed: {seed}")
    x, y, task, groups = _data(name)
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    keys = np.arange(len(first))

    def partition(pool, fraction, salt):
        strata = y[first[pool]] if name == "taiwan" else None
        return train_test_split(pool, test_size=fraction, random_state=seed + salt, stratify=strata)

    if name == "tecator":
        # Exact published test by specimen position, including when group codes differ from row ids.
        dev, test = np.unique(inverse[:-43]), np.unique(inverse[-43:])
        assert not np.intersect1d(dev, test).size
    else:
        dev, test = partition(keys, .2, 0)
    joint, width = partition(dev, .2, 101)
    fit, vote = partition(joint, .25, 201)
    idx = {k: np.flatnonzero(np.isin(inverse, v)) for k, v in zip(PARTS, (fit, vote, width, test))}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(len(y)))
    if any(len(v) == 0 for v in idx.values()):
        raise ValueError(f"Empty partition: {name}/{seed}")
    if task == "classification":
        if any(len(np.unique(y[v])) != len(np.unique(y)) for v in idx.values()):
            raise ValueError(f"Missing class in {name}/{seed}")
    return x, y, task, idx


def split(name, seed):
    """Return all X, masked y, task and disjoint fit/vote/width/test row positions."""
    x, y, task, idx = _partition(name, seed)
    masked = y.copy()
    masked[idx["test"]] = -1 if task == "classification" else np.nan
    return x.copy(), masked, task, idx


def group_ids(name):
    """Return split-unit codes without opening any label to the caller."""
    return _data(name)[3].copy()


def score(name, seed, forecast_path, frozen_sha256):
    """Score previously sealed outer forecasts after release; use split() for training."""
    raw = Path(forecast_path).read_bytes()
    if sha256(raw).hexdigest() != frozen_sha256:
        raise ValueError(f"Wrong forecast digest: {name}/{seed}")
    _, y, task, idx = _partition(name, seed)
    dev = np.concatenate([idx[k] for k in PARTS[:-1]])
    scale = max(float(np.std(y[dev])), 1e-12)
    with np.load(BytesIO(raw), allow_pickle=False) as predictions:
        return {arm: metrics(y[idx["test"]], predictions[arm], task, scale) for arm in predictions.files}


def receipt():
    records = {}
    for name in SOURCES:
        groups = group_ids(name)
        seeds = {}
        for seed in SEEDS:
            x, y, task, idx = split(name, seed)
            outer = y[idx["test"]]
            assert ((outer == -1).all() if task == "classification" else np.isnan(outer).all())
            split_groups = [set(groups[idx[k]]) for k in PARTS]
            assert sum(map(len, split_groups)) == len(set.union(*split_groups))
            seeds[seed] = {
                key: {"n": len(rows), "groups": len(set(groups[rows])),
                      "sha256": sha256(rows.astype("<i8").tobytes()).hexdigest()}
                for key, rows in idx.items()
            }
        records[name] = {"rows": len(x), "columns": x.shape[1], "groups": len(set(groups)),
                         "source_sha256": SOURCES[name]["sha256"], "seeds": seeds}
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args()
    if args.fetch:
        fetch()
    audit = receipt()
    for name, record in audit.items():
        print(name, record["rows"], record["columns"], record["groups"])
    if args.write_receipt:
        args.write_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.write_receipt.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
