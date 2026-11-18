"""Isolate pair seeding and subsequent conditional additions on archived low-dimensional inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.feature_game_eval import panels as bench
from research.lowdim.method import fast_support, pair_support
from research.pair_checks.reference import Refits, mechanism_supports, standardize
from research.pair_selection.benchmark import cases


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEY = ["cohort", "dataset", "fold", "seed"]


def run():
    development = pd.read_csv(ROOT / "research/lowdim/peer_evidence/cases.csv").set_index(KEY)
    index = pd.read_csv(ROOT / "research/lowdim/evidence/all-cases.csv", keep_default_na=False)
    index = index.loc[index.p <= 32].set_index(KEY)
    assert index.index.is_unique and len(index) == 56 and len(development) == 31
    rows = []
    for case in cases(set(index.index)):
        key = case.key
        if key not in index.index:
            continue
        X, y, task = case.X, case.y, case.task
        if case.impute:
            X = bench._impute_validation(X, X[:0])[0]
        seed = key[-1]
        ref = Refits(*standardize(X, y, task, seed))
        old, info = fast_support(X, y, task=task, seed=seed)
        pair, audit = pair_support(X, y, task=task, seed=seed, audit=True)
        with np.load(case.archive) as saved:
            np.testing.assert_array_equal(old, saved["new_cols"])
        assert 1 <= len(pair) == len(old) == index.loc[key, "k"] <= X.shape[1]
        np.testing.assert_array_equal(info["rank"], audit["rank"])
        paths = mechanism_supports(ref, old, info["rank"])
        if len(old) < X.shape[1]:
            np.testing.assert_array_equal(audit["selection_order"], paths["pair_forward"])
        np.testing.assert_array_equal(pair, np.sort(paths["pair_forward"]))

        # All four supports are fixed before scoring the first test prediction.
        observed = {}
        for arm, path in paths.items():
            columns = np.sort(path)
            ids = tuple(columns)
            if ids not in observed:
                observed[ids] = case.score(*case.predict(columns))
            if arm == "old":
                np.testing.assert_allclose(observed[ids], index.loc[key, "score_new"], atol=1e-12, rtol=0)
            rows.append(dict(zip(KEY, key)) | {
                "metric": index.loc[key, "metric"] if key not in development.index else development.loc[key, "metric"],
                "p": X.shape[1], "k": len(old),
                "arm": arm, "columns": " ".join(map(str, columns)),
                "path": "" if arm == "old" else " ".join(map(str, path)),
                "train_utility": ref.utility(columns), "score": observed[ids],
            })
    results = pd.DataFrame(rows).sort_values(KEY + ["arm"])
    assert len(results) == 4 * len(index) and not results.duplicated(KEY + ["arm"]).any()
    results.to_csv(HERE / "archive-all.csv", index=False)
    subset = results.set_index(KEY).index.isin(development.index)
    assert subset.sum() == 4 * len(development)
    results.loc[subset].to_csv(HERE / "cases.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
