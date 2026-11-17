"""Load the four UCI archives declared in chain_validation_protocol.md."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.minpi_compression.sources import DATA


SOURCES = {
    "landsat": (146, "statlog+landsat+satellite", "7c54e0e11c872a1b0b647da370d596dcb06746159cce4121d92ccd70b7d7ce3c"),
    "biodeg": (254, "qsar+biodegradation", "b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac"),
    "aquatic": (505, "qsar+aquatic+toxicity", "09e45d1b2b66c5bc871a64b87ad40e452623e1c2dc375e517e4769379a9db9cd"),
    "power": (294, "combined+cycle+power+plant", "cc7b2a4977c0a44e8221c91d9a7e5746b3c68186cff7e5c61c70af6432b98c7a"),
}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for source, (number, slug, expected) in SOURCES.items():
        path = DATA / f"{source}.zip"
        if not path.exists():
            with urlopen(f"https://archive.ics.uci.edu/static/public/{number}/{slug}.zip", timeout=60) as response:
                raw = response.read()
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f"Unexpected UCI file for {source}")
            path.write_bytes(raw)
        print(source, load(source)[0].shape, expected)


@lru_cache
def load(name):
    raw = (DATA / f"{name}.zip").read_bytes()
    if sha256(raw).hexdigest() != SOURCES[name][2]:
        raise ValueError(f"Archive hash mismatch for {name}")
    official = None
    with ZipFile(BytesIO(raw)) as archive:
        if name == "landsat":
            dev, test = [pd.read_csv(archive.open(member), sep=r"\s+", header=None)
                         for member in ("sat.trn", "sat.tst")]
            shared = pd.MultiIndex.from_frame(dev.iloc[:, :-1]).isin(pd.MultiIndex.from_frame(test.iloc[:, :-1]))
            frame = pd.concat([dev[~shared], test], ignore_index=True)
            official = np.arange((~shared).sum(), len(frame))
        elif name == "power":
            with ZipFile(BytesIO(archive.read("CCPP/Folds5x2_pp.xlsx"))) as book:
                sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
                rows = [[c.findtext("{*}v") for c in row.findall("{*}c")]
                        for row in sheet.findall(".//{*}row")[1:]]
            frame = pd.DataFrame(rows, dtype=float)
            assert frame.shape == (9568, 5)
        else:
            member = "biodeg.csv" if name == "biodeg" else "qsar_aquatic_toxicity.csv"
            frame = pd.read_csv(archive.open(member), sep=";", header=None)
    x = frame.iloc[:, :-1].to_numpy(float)
    y = frame.iloc[:, -1].to_numpy()
    domain = "classification" if name in ("landsat", "biodeg") else "regression"
    if domain == "classification":
        _, y = np.unique(y, return_inverse=True)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    _, representative, groups = np.unique(x, axis=0, return_inverse=True, return_index=True)
    return x, y, domain, groups, representative, official


def split(name, seed):
    x, y, domain, groups, representatives, official = load(name)
    ids = np.arange(len(representatives))

    def partition(pool, fraction, salt):
        stratify = y[representatives[pool]] if domain == "classification" else None
        return train_test_split(pool, test_size=fraction, random_state=seed + salt, stratify=stratify)

    if official is None:
        development, test = partition(ids, .20, 0)
    else:
        test = np.unique(groups[official])
        development = np.setdiff1d(ids, test)
    base, width = partition(development, .20, 101)
    fit, vote = partition(base, .25, 201)
    idx = {k: np.flatnonzero(np.isin(groups, a)) for k, a in
           dict(fit=fit, vote=vote, width=width, test=test).items()}
    assert np.array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(len(x)))
    if official is not None:
        np.testing.assert_array_equal(idx["test"], official)
    return x, y, domain, idx


if __name__ == "__main__":
    fetch()
