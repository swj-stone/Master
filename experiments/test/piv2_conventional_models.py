"""Conventional model-based PIv2 controls with a common training-row budget."""

from hashlib import sha256
from math import ceil, sqrt

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROW_LIMIT = 4096
FORESTS = {
    "random_forest": (RandomForestClassifier, RandomForestRegressor),
    "extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor),
}


def training_rows(y, regression, seed):
    rows = np.arange(len(y))
    if len(rows) > ROW_LIMIT:
        rows, _ = train_test_split(rows, train_size=ROW_LIMIT, random_state=seed,
                                   stratify=None if regression else y)
        rows.sort()
    return rows


def choose_model(backend, train, yt, policy, *, original_width):
    """Return selected names in input order, fitted estimator count and diagnostics."""
    width = train.shape[1]
    y = np.asarray(yt)
    if (not 1 <= width <= original_width or not train.columns.is_unique or
            y.ndim != 1 or len(y) != len(train) or not len(train)):
        raise ValueError("Expected aligned training data and a valid original feature count")
    task = policy["task"]
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError(f"Unknown prediction task {task}")
    regression = task == "regression"
    seed = policy["seed"]
    rows = training_rows(y, regression, seed)
    x = train.iloc[rows].to_numpy(dtype=np.float64)
    y = y[rows]
    k = min(width, ceil(sqrt(original_width)))
    fits = 1
    diagnostics = dict(ranking_sample_rows=len(rows), row_limit=ROW_LIMIT,
                       sample_rows_sha256=sha256(rows.astype("<i8").tobytes()).hexdigest())

    with threadpool_limits(limits=1):
        if backend in FORESTS:
            classifier, regressor = FORESTS[backend]
            model = (regressor if regression else classifier)(n_estimators=100, random_state=seed, n_jobs=1)
            model.fit(x, y)
            importance = model.feature_importances_
            support = np.sort(np.argsort(-importance, kind="stable")[:k])
            diagnostics.update(k=k, scores=importance.tolist(), trees=len(model.estimators_))
        elif backend == "rfe_linear":
            x = StandardScaler().fit_transform(x)
            estimator = (Ridge if regression else RidgeClassifier)(alpha=1., solver="cholesky")
            step = max(1, width // 5)
            if width == 1:
                estimator.fit(x, y)
                ranks = np.array([1])
            else:
                rfe = RFE(estimator, n_features_to_select=k, step=step).fit(x, y)
                ranks = rfe.ranking_
            support = np.flatnonzero(ranks == 1)
            # One ridge fit per removal step and one final fit with the selected columns.
            fits = ceil((width-k) / step) + 1
            diagnostics.update(k=k, elimination_step=step, rankings=ranks.tolist())
        elif backend == "elastic_net":
            x = StandardScaler().fit_transform(x)
            if regression:
                y = (y - y.mean()) / max(y.std(), 1e-12)
                model = ElasticNet(alpha=.03, l1_ratio=.5, tol=1e-4, max_iter=2500)
            else:
                model = LogisticRegression(C=.1, l1_ratio=.5, solver="saga", random_state=seed,
                                           tol=1e-3, max_iter=2500)
            model.fit(x, y)
            scores = np.max(np.abs(np.atleast_2d(model.coef_)), axis=0)
            support = np.flatnonzero(scores > 0.)
            # A zeroed-out model gets a single reproducibly chosen input column.
            fallback = not support.size
            if fallback:
                support = np.array([int(np.argmax(scores))])
            iterations = int(np.max(model.n_iter_))
            diagnostics.update(scores=scores.tolist(), fallback_top1=fallback, iterations=iterations,
                               iteration_limit=model.max_iter, hit_iteration_limit=iterations >= model.max_iter)
        else:
            raise ValueError(f"Unknown conventional model selector {backend}")
    selected = train.columns[support].tolist()
    diagnostics.update(selected=selected, selector_fits=fits)
    return selected, fits, diagnostics
