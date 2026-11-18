"""Check the nonidentical class decision seen after exact training-constant removal."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.minpi_constant_transfer import panels
from experiments.minpi_study import P
from minppi_mofs.stable_minpi import _impute_validation


def compare(run):
    rows = []
    for name, fold, seed, X, y, Xt, truth, task, split in panels():
        if name != "internet-advertisements-256":
            continue
        with np.load(run/f"{name}-{fold}.npz") as archive:
            cols = archive["exchange_cols"]
            previous = {m: archive[m+"_pred"] for m in ("full","exchange")}
        removed = np.setdiff1d(np.arange(X.shape[1]), cols)
        assert np.all(np.isfinite(X[:, removed])) and np.all(X[:, removed] == X[0, removed])
        for dtype, tol, cap in ((np.float32, P["ridge_tol"], P["ridge_max_iter"]),
                                (np.float64, 1e-10, 3000)):
            scores, predictions, iterations = {}, {}, {}
            for m in previous:
                a, b = (X, Xt) if m == "full" else (X[:, cols], Xt[:, cols])
                a, b = _impute_validation(a.astype(dtype, copy=False), b.astype(dtype, copy=False))
                cls = RidgeClassifier(alpha=P["linear_alpha"], solver="lsqr", tol=tol, max_iter=cap)
                model = make_pipeline(StandardScaler(), cls).fit(a, y)
                if m == "full":
                    removed_coef = float(np.abs(cls.coef_.reshape(-1, X.shape[1])[:, removed]).max())
                scores[m] = model.decision_function(b)
                predictions[m] = cls.classes_[(scores[m] > 0).astype(int)]
                iterations[m] = int(np.asarray(cls.n_iter_).max())
                if dtype == np.float32:
                    np.testing.assert_array_equal(predictions[m], previous[m])
            changed = predictions["full"] != predictions["exchange"]
            rows.append(dict(dataset=name, fold=fold, seed=seed, dtype=dtype.__name__, tol=tol,
                             full_iterations=iterations["full"], reduced_iterations=iterations["exchange"],
                             max_removed_coef=removed_coef, changed_labels=int(changed.sum()),
                             max_score_change=float(np.max(np.abs(scores["full"]-scores["exchange"]))),
                             max_full_margin_changed=float(np.abs(scores["full"][changed]).max()) if changed.any() else 0.))
    assert len(rows) == 6
    pd.DataFrame(rows).to_csv(run/"zero-vote-numerics.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True, type=Path)
    with threadpool_limits(limits=1):
        compare(cli.parse_args().run)
