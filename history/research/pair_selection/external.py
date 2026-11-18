"""Three disjoint folds per UCI source, grouping identical predictor rows."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


RAW = Path(__file__).resolve().parents[2] / "data/raw/pair-external"
SPLIT_SEED = 32119
SOURCES = (
    ("uci_magic_gamma", "classification", "magic.zip", "magic04.data",
     "252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d"),
    ("uci_htru2", "classification", "htru2.zip", "HTRU_2.csv",
     "ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379"),
    ("uci_concrete", "regression", "concrete.zip", "Concrete_Data.xls",
     "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7"),
    ("uci_red_wine_quality", "regression", "wine.zip", "winequality-red.csv",
     "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4"),
)


def external_panels():
    for source_id, (name, task, file, member, fingerprint) in enumerate(SOURCES):
        archive = (RAW / file).read_bytes()
        if sha256(archive).hexdigest() != fingerprint:
            raise ValueError(f"Unexpected UCI archive for {name}")
        with ZipFile(BytesIO(archive)) as zipped:
            content = BytesIO(zipped.read(member))
        if file == "concrete.zip":
            frame = pd.read_excel(content, engine="xlrd")
        elif file == "wine.zip":
            frame = pd.read_csv(content, sep=";")
        else:
            frame = pd.read_csv(content, header=None)
        X = np.ascontiguousarray(frame.iloc[:, :-1], dtype=np.float64)
        target = frame.iloc[:, -1]
        y = target.eq("g").to_numpy(dtype=int) if file == "magic.zip" else target.to_numpy()
        assert np.isfinite(X).all() and np.isfinite(y).all()
        _, group = np.unique(X, axis=0, return_inverse=True)
        splitter = StratifiedGroupKFold if task == "classification" else GroupKFold
        folds = splitter(n_splits=3, shuffle=True, random_state=SPLIT_SEED)
        for fold, (train, test) in enumerate(folds.split(X, y, group), start=1):
            assert not np.intersect1d(group[train], group[test]).size
            # MAGIC orders records by class. Randomize training order before sampling rows.
            seed = 32011 + fold
            train = np.random.default_rng([SPLIT_SEED, source_id, fold]).permutation(train)
            yield SimpleNamespace(name=name, task=task, fold=fold, seed=seed,
                                  X=X[train], y=y[train], Xt=X[test], truth=y[test])
