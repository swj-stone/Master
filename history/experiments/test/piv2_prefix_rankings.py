"""Feature paths for the validation-selected prefix-length experiment."""

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.test.piv2_conventional_filters import FILTERS, choose_filter
from experiments.test.piv2_conventional_models import FORESTS, choose_model, training_rows
from experiments.test.piv2_extra_selectors import masked_banzhaf_importance
from experiments.test.piv2_selectors import _univariate
from experiments.test.run_sampled_permutation_seed42 import predict_with
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance
from minppi_mofs.modeling import make_model


METHODS = ("f_score", "mutual_info", "tree_shap", "banzhaf", "spearman", "mrmr", "chi2",
           "variance", "random_forest", "extra_trees", "rfe_linear")


def _ridge_elimination_path(train, yt, policy):
    """Reproduce RFE support for every target k by recording within-step coefficient order."""
    regression = policy["task"] == "regression"
    y = np.asarray(yt)
    rows = training_rows(y, regression, policy["seed"])
    x = StandardScaler().fit_transform(train.iloc[rows].to_numpy(dtype=np.float64))
    step = max(1, train.shape[1] // 5)
    active = np.arange(train.shape[1])
    removed = []
    fits = 0
    with threadpool_limits(limits=1):
        while len(active) > 1:
            estimator = (Ridge if regression else RidgeClassifier)(alpha=1., solver="cholesky")
            estimator.fit(x[:, active], y[rows])
            fits += 1
            importance = np.sum(np.atleast_2d(estimator.coef_)**2, axis=0)
            weakest = np.argsort(importance, kind="stable")[:min(step, len(active)-1)]
            removed.extend(active[weakest].tolist())
            active = np.delete(active, weakest)
    order = active.tolist() + removed[::-1]
    return order, fits, None, dict(elimination_step=step, ranking_sample_rows=len(rows))


def rank_features(backend, train, yt, val, yv, policy, metric, progress, *, original_columns):
    """Return descending indices, selector fits, optional endpoint predictions and diagnostics."""
    if (backend not in METHODS or train.empty or not train.columns.is_unique
            or not train.columns.equals(val.columns) or not set(train.columns).issubset(original_columns)):
        raise ValueError("Invalid prefix backend or unaligned features")
    if backend == "rfe_linear":
        return _ridge_elimination_path(train, yt, policy)
    if backend in FILTERS:
        _, details = choose_filter(backend, train, yt, policy, original_width=len(original_columns), full_order=True)
        order = train.columns.get_indexer(details["choice_order"]).tolist()
        details.pop("selected")
        details.pop("choice_order")
        details.pop("k")
        return order, 0, None, details

    full_prediction = None
    if backend in ("f_score", "mutual_info"):
        scores, samples = _univariate(backend, train, yt, policy)
        details = dict(ranking_sample_rows=samples)
        fits = 0
    elif backend in FORESTS:
        _, fits, details = choose_model(backend, train, yt, policy, original_width=len(original_columns))
        scores = details["scores"]
        details.pop("selected")
        details.pop("k")
    else:
        safe = [f"f{i}" for i in range(train.shape[1])]
        if backend == "tree_shap":
            safe_train = train.set_axis(safe, axis=1)
            model = make_model(**policy)
            model.fit(safe_train, yt)
            scores = fitted_tree_shap_importance(model, safe_train, seed=policy["seed"], max_rows=1024)
            details = dict(ranking_sample_rows=min(len(train), 1024))
        else:
            model, scores, details = masked_banzhaf_importance(train, yt, val, yv, policy, metric, progress,
                                                             original_columns)
        full_prediction = predict_with(model, val.set_axis(safe, axis=1), policy)
        fits = 1
    scores = np.asarray(scores, dtype=float).copy()
    scores[np.isnan(scores)] = -np.inf
    return np.argsort(-scores, kind="stable").tolist(), fits, full_prediction, details
