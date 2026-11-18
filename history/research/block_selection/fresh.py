"""Held-out UCI cases for the adaptive block rule; all preprocessing is per fold."""
from functools import partial
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from research.feature_game_eval import panels
from research.pair_selection.benchmark import Case, standard_score


RAW = Path(__file__).resolve().parents[2] / "data/raw/block-external"
ARCHIVES = {
    "raisin": ("https://archive.ics.uci.edu/static/public/850/raisin.zip",
               "5d516c040923fd154ecf85e31b6e1b5a096724fa0e68d588c33fb5a2f0c3e5db"),
    "yacht": ("https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip",
              "aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac"),
}


def read_data(name):
    raw = (RAW / (name + ".zip")).read_bytes()
    if sha256(raw).hexdigest() != ARCHIVES[name][1]:
        raise ValueError(f"Unexpected UCI archive for {name}")
    with ZipFile(BytesIO(raw)) as outer:
        if name == "raisin":
            with ZipFile(BytesIO(outer.read("Raisin_Dataset.zip"))) as inner:
                arff = inner.read("Raisin_Dataset/Raisin_Dataset.arff").decode("utf-8")
            frame = pd.read_csv(StringIO(arff.split("@data", 1)[1]), header=None)
            assert set(frame.iloc[:, -1]) == {"Kecimen", "Besni"}
            X = np.ascontiguousarray(frame.iloc[:, :-1], dtype=np.float64)
            y = frame.iloc[:, -1].eq("Besni").to_numpy(dtype=int)
            assert X.shape == (900, 7)
            group_key = X
        else:
            data = np.loadtxt(BytesIO(outer.read("yacht_hydrodynamics.data")))
            assert data.shape == (308, 7)
            X, y = data[:, :-1].copy(), data[:, -1].copy()
            group_key = X[:, :5]  # Five geometry descriptors identify each hull.
    assert np.isfinite(X).all() and np.isfinite(y).all()
    groups, ids = np.unique(group_key, axis=0, return_inverse=True)
    if name == "yacht":
        assert len(groups) == 22
    return X, y, ids


def fresh_cases():
    result = []
    for source_id, name in enumerate(ARCHIVES):
        X, y, groups = read_data(name)
        task, metric = ("classification", "auc") if name == "raisin" else ("regression", "r2")
        splitter = StratifiedGroupKFold if name == "raisin" else GroupKFold
        splitter = splitter(n_splits=3, shuffle=True, random_state=32119)
        tests = []
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
            assert not np.intersect1d(groups[train], groups[test]).size
            tests.extend(test)
            train = np.random.default_rng([32119, 6 + source_id, fold]).permutation(train)
            seed = 32011 + fold
            d = SimpleNamespace(name="uci_" + name, task=task, seed=seed, fold=fold,
                                X=X[train], y=y[train], Xt=X[test], truth=y[test])
            result.append(Case(("fresh", d.name, fold, seed), d.X, d.y, d.Xt, d.truth,
                               task, False, partial(panels.fit, d),
                               partial(standard_score, d.truth, metric), None))
        np.testing.assert_array_equal(np.sort(tests), np.arange(len(y)))
    return result
