"""Two additional seed-42 controls: sparse linear selection and sampled Banzhaf ranking."""

from math import ceil, sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from experiments.test.run_sampled_permutation_seed42 import batched_utility, predict_with
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.piv1 import _sample_rows


def _linear_support(train, y, policy):
    regression = policy["task"] == "regression"
    rows = np.arange(len(train))
    if len(rows) > 8192:
        rows, _ = train_test_split(rows, train_size=8192, random_state=policy["seed"],
                                   stratify=None if regression else y)
        rows.sort()
    x = StandardScaler().fit_transform(train.iloc[rows])
    targets = np.asarray(y)[rows]
    if regression:
        targets = (targets - targets.mean()) / max(targets.std(), 1e-12)
        model = Lasso(alpha=.03, max_iter=2000).fit(x, targets)
        importance = np.abs(model.coef_)
    else:
        model = OneVsRestClassifier(LogisticRegression(
            l1_ratio=1, C=.1, solver="liblinear", max_iter=2000, random_state=policy["seed"]
        )).fit(x, targets)
        importance = np.max([np.abs(est.coef_[0]) for est in model.estimators_], axis=0)
    indices = np.flatnonzero(importance > 0)
    if not len(indices):
        indices = [int(np.argmax(importance))]
    return train.columns[indices].tolist(), dict(ranking_sample_rows=len(rows), coefficients=importance.tolist(),
                                                selector_fits=1)


def masked_banzhaf_importance(train, yt, val, yv, policy, metric, progress, original_columns,
                              *, model=None, max_batch_elements=None):
    # Complementary coalition pairs give exactly 32 present/absent observations per column.
    # Generate masks before projecting so both arms use identical common random numbers.
    row_seed, mask_seed, background_seed = np.random.SeedSequence(policy["seed"]).spawn(3)
    rows = _sample_rows(yv, metric, np.random.default_rng(row_seed))
    background_rng = np.random.default_rng(background_seed)
    background = np.sort(background_rng.choice(len(train), min(1024, len(train)), replace=False))
    fill = np.median(train.iloc[background].to_numpy(), axis=0)
    index = pd.Index(original_columns).get_indexer(train.columns)
    if np.any(index < 0):
        raise ValueError("Unknown original feature in Banzhaf backend")
    first = np.random.default_rng(mask_seed).integers(0, 2, size=(32, len(original_columns))).astype(bool)
    masks = np.concatenate([first[:, index], ~first[:, index]], axis=0)
    columns = [f"f{i}" for i in range(train.shape[1])]
    if model is None:
        model = make_model(**policy)
        model.fit(train.set_axis(columns, axis=1), yt)
    else:
        columns = model.get_booster().feature_names or columns
    x = val.iloc[rows].to_numpy()
    y = np.asarray(yv)[rows]
    scores = []
    batch_size = 8
    if max_batch_elements is not None:
        batch_size = max(1, min(batch_size, max_batch_elements // x.size))
    for start in range(0, len(masks), batch_size):
        batch = masks[start:start+batch_size]
        data = np.where(batch[:, None, :], x[None, :, :], fill)
        prediction = predict_with(model, pd.DataFrame(data.reshape(-1, len(columns)), columns=columns), policy)
        scores.extend(batched_utility(y, prediction.reshape(len(batch), len(rows), *prediction.shape[1:]), metric))
        progress("masked_banzhaf", checked=start+len(batch), candidates=len(masks))
    importance = (2*masks.astype(float)-1).T @ np.asarray(scores) / 32
    diagnostics = dict(masks=len(masks), sample_rows=len(rows), importance=importance.tolist())
    return model, importance, diagnostics


def _masked_banzhaf(train, yt, val, yv, policy, metric, progress, original_columns):
    model, importance, diagnostics = masked_banzhaf_importance(train, yt, val, yv, policy, metric, progress,
                                                             original_columns)
    columns = [f"f{i}" for i in range(train.shape[1])]
    k = min(len(columns), ceil(sqrt(len(original_columns))))
    indices = np.sort(np.argsort(-importance, kind="stable")[:k])
    selected = train.columns[indices].tolist()
    diagnostics["k"] = k
    if len(selected) == len(columns):
        return selected, predict_with(model, val.set_axis(columns, axis=1), policy), 1, diagnostics
    return selected, fit_predict(train[selected], yt, val[selected], **policy), 2, diagnostics


def select_extra(backend, train, yt, val, yv, policy, metric, progress, *, original_width, original_columns):
    if backend == "banzhaf":
        return _masked_banzhaf(train, yt, val, yv, policy, metric, progress, original_columns)
    if backend != "l1":
        raise ValueError(f"Unknown extra backend {backend}")
    selected, diagnostics = _linear_support(train, yt, policy)
    return selected, fit_predict(train[selected], yt, val[selected], **policy), 2, diagnostics
