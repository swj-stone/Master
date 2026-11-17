"""Fresh UCI panel. The selection interface masks every outer target."""
import argparse
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

from experiments.minpi_compression.benchmark import metrics, ROOT


MANIFEST = Path(__file__).with_name("validation_sources.json")
SOURCES = json.loads(MANIFEST.read_text(encoding="utf8"))
CLASSIFICATION = {"anuran", "mice", "retina", "coimbra"}
STRESS = ("dense_cls", "dense_reg", "proxy_cls", "proxy_reg")
DATA = ROOT / "data" / "minpi_rework"
SEEDS = tuple(range(42, 47))


def fetch():
    """Download source bytes once, verifying the pinned digest before installation."""
    DATA.mkdir(exist_ok=True, parents=True)
    for filename, suffix, expected in SOURCES.values():
        path = DATA / filename
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Unexpected digest for existing {path}")
            continue
        request = Request("https://archive.ics.uci.edu/" + suffix,
                          headers={"User-Agent": "MinPI research replication"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=50) as response:
                    raw = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                sleep(2 ** attempt)
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Changed source bytes for {filename}")
        temporary = path.with_suffix(".part")
        temporary.write_bytes(raw)
        temporary.replace(path)


def _stress(name):
    """Matched numeric diagnostics. Dense controls and proxy cases have 48 columns."""
    rng = np.random.default_rng(90817)
    n, p = 1600, 48
    if name.startswith("dense"):
        x = rng.standard_normal((n, p))
        signal = x @ (np.tile([1., -1.], p // 2) / np.sqrt(p))
    else:
        factors = rng.standard_normal((n, 8))
        x = np.repeat(factors, 6, axis=1) + .12 * rng.standard_normal((n, p))
        signal = factors @ (np.tile([1., -1.], 4) / np.sqrt(8))
    y = signal + .5 * rng.standard_normal(n)
    cls = name.endswith("cls")
    return x, (y > 0).astype(int) if cls else y, "classification" if cls else "regression", np.arange(n)


def _canonical_groups(frame, identities):
    """Join rows sharing either their declared identity or an identical feature vector."""
    feature_group = frame.groupby(list(frame.columns), sort=False, dropna=False).ngroup().to_numpy()
    if identities is None:
        return feature_group
    person_group = pd.factorize(identities)[0]
    if np.any(person_group < 0):
        raise ValueError("Missing grouping identifier")
    offset = person_group.max() + 1
    n = offset + feature_group.max() + 1
    graph = coo_matrix((np.ones(len(frame), dtype=bool), (person_group, offset + feature_group)), shape=(n, n))
    _, components = connected_components(graph, directed=False)
    return components[person_group]


@lru_cache
def _data(name):
    if name in STRESS:
        return _stress(name)
    filename, _, digest = SOURCES[name]
    path = DATA / filename
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected raw data for {name}")
    ids = None
    if name == "mice":
        f = pd.read_csv(path)
        ids = f.MouseID.str.rsplit("_", n=1).str[0]
        y = f.Genotype.eq("Ts65Dn").to_numpy(int)
        x = f.filter(regex="_N$")
    elif name == "appliances":
        f = pd.read_csv(path)
        f["date"] = pd.to_datetime(f["date"], format="%Y-%m-%d%H:%M:%S")
        f = f.sort_values("date")
        f = f.loc[f.date.dt.minute == 0]
        ids = f.date.dt.normalize()
        y = f.Appliances.to_numpy(float)
        x = f.drop(columns=["date", "Appliances", "lights"])
    elif name == "coimbra":
        x = pd.read_csv(path)
        y = x.pop("Classification").to_numpy(int) - 1
    else:
        with ZipFile(path) as archive:
            if name == "anuran":
                f = pd.read_csv(archive.open("Frogs_MFCCs.csv"))
                x = f.filter(regex="^MFCCs_")
                y = f.Family.eq("Leptodactylidae").to_numpy(int)
                ids = f.RecordID
            elif name == "retina":
                rows = loadarff(StringIO(archive.read("messidor_features.arff").decode()))[0]
                f = pd.DataFrame(rows)
                y = f.pop("Class").to_numpy(int)
                x = f.drop(columns=["18"])  # precomputed AM/FM classifier output
            elif name == "grid":
                f = pd.read_csv(archive.open("Data_for_UCI_named.csv"))
                y = f.stab.to_numpy(float)
                x = f.drop(columns=["stab", "stabf"])
            elif name == "music":
                f = pd.read_csv(archive.open("Geographical Original of Music/default_features_1059_tracks.txt"), header=None)
                x = f.iloc[:, :68]
                y = f.iloc[:, 68].to_numpy(float)
                ids = pd.MultiIndex.from_frame(f.iloc[:, 68:70])  # capital location
            elif name == "superconductor":
                x = pd.read_csv(archive.open("train.csv"))
                material = pd.read_csv(archive.open("unique_m.csv"), usecols=["material", "critical_temp"])
                y = x.pop("critical_temp").to_numpy(float)
                np.testing.assert_array_equal(y, material.critical_temp)
                ids = material.material

    group = _canonical_groups(x, ids)
    if name == "appliances":
        # Sequence by first appearance in increasing date order. Duplicate vectors
        # on different dates would invalidate this chronological partition.
        if len(np.unique(group)) != ids.nunique():
            raise ValueError("Duplicate predictor vectors span separate dates")
        group = pd.factorize(ids, sort=True)[0]
    if name in {"grid", "superconductor"}:
        # Group-preserving, label-blind sample with bounded training cost.
        unique, counts = np.unique(group, return_counts=True)
        order = np.random.default_rng(71117).permutation(len(unique))
        chosen = unique[order][np.cumsum(counts[order]) <= 4000]
        keep = np.flatnonzero(np.isin(group, chosen))
        x, y, group = x.iloc[keep], y[keep], group[keep]
    return x.to_numpy(float), y, "classification" if name in CLASSIFICATION else "regression", np.asarray(group)


def _partition(name, seed):
    x, y, task, group = _data(name)
    unique, representative, inverse = np.unique(group, return_index=True, return_inverse=True)
    if name == "appliances":
        boundaries = [int(len(unique) * fraction) for fraction in (.48, .64, .8)]
        fit, vote, width, test = np.split(np.arange(len(unique)), boundaries)
    else:
        labels = y[representative] if task == "classification" else None
        def part(pool, fraction, salt):
            strata = None if labels is None else labels[pool]
            return train_test_split(pool, test_size=fraction, stratify=strata, random_state=seed + salt)
        dev, test = part(np.arange(len(unique)), .2, 0)
        base, width = part(dev, .2, 101)
        fit, vote = part(base, .25, 201)
    idx = {key: np.flatnonzero(np.isin(inverse, pool))
           for key, pool in dict(fit=fit, vote=vote, width=width, test=test).items()}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(len(y)))
    if task == "classification" and any(np.unique(y[indices]).size != 2 for indices in idx.values()):
        raise ValueError(f"Unsupported class distribution for {name}/{seed}")
    return x, y, task, idx


def split(name, seed):
    """Return X, y, task, partitions for any selector. Outer y is a sentinel."""
    x, y, task, idx = _partition(name, seed)
    blinded = y.copy()
    blinded[idx["test"]] = -1 if task == "classification" else np.nan
    return x.copy(), blinded, task, idx


def score(name, seed, forecast_path, frozen_sha256):
    """Score all methods in a previously locked .npz of outer predictions."""
    path = Path(forecast_path)
    if sha256(path.read_bytes()).hexdigest() != frozen_sha256:
        raise ValueError(f"Forecast digest mismatch for {name}/{seed}")
    _, y, task, idx = _partition(name, seed)
    dev = np.r_[idx["fit"], idx["vote"], idx["width"]]
    scale = max(float(np.std(y[dev])), 1e-12)
    with np.load(path, allow_pickle=False) as archive:
        return {key: metrics(y[idx["test"]], archive[key], task, scale) for key in archive.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    for name in [*SOURCES, *STRESS]:
        for seed in SEEDS:
            x, y, task, idx = split(name, seed)
            if seed == SEEDS[0]:
                print(name, task, x.shape, len(np.unique(_data(name)[3])),
                      {k: len(v) for k, v in idx.items()})
