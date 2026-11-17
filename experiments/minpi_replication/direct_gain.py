"""Native exhaustive XGBoost gain path matching the archived PIv0 control."""
from time import perf_counter
import json

import numpy as np

from minppi_mofs.modeling import make_model
from minppi_mofs.utility import compute_utility


MODEL_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=.1, n_jobs=1)


def fit_native(x, y, task, seed, device="cuda"):
    """Fit a native float32 XGBoost learner, preserving missing measurements."""
    classes = np.unique(y) if task != "regression" else ()
    model_task = "regression" if task == "regression" else ("binary" if len(classes) == 2 else "multiclass")
    model = make_model(task=model_task, model_params=MODEL_PARAMS, backend=device, seed=int(seed))
    model.fit(np.ascontiguousarray(x, dtype=np.float32), y)
    if device == "cuda":
        configured = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not configured.startswith("cuda"):
            raise RuntimeError(f"XGBoost used {configured} despite a CUDA request")
    return model


def predict_native(model, x, task):
    array = np.ascontiguousarray(x, dtype=np.float32)
    return model.predict(array) if task == "regression" else model.predict_proba(array)


def validation_utility(y, predictions, task):
    metric = "r2" if task == "regression" else "ovr_auc"
    return float(compute_utility(y, predictions, metric=metric))


def gain_order(model, width):
    scores = model.get_booster().get_score(importance_type="gain")
    return sorted(range(width), key=lambda j: (scores.get(f"f{j}", 0.), j), reverse=True)


def run_direct_gain(x, y, task, idx, seed, device="cuda"):
    """Return JSON-safe validation/path record and predictions with no test-label access."""
    started = perf_counter()
    x = np.asarray(x, dtype=np.float32)
    train = np.sort(idx["fit"])
    valid = np.sort(np.concatenate([idx["vote"], idx["width"]]))
    train_x, valid_x = x[train], x[valid]
    train_y, valid_y = y[train], y[valid]
    width = x.shape[1]
    full = fit_native(train_x, train_y, task, seed, device)
    order = gain_order(full, width)
    full_utility = validation_utility(valid_y, predict_native(full, valid_x, task), task)
    curve = []
    best = -np.inf
    best_width = 0
    for k in range(1, width+1):
        utility = full_utility
        if k != width:
            cols = sorted(order[:k])
            model = fit_native(train_x[:, cols], train_y, task, seed, device)
            utility = validation_utility(valid_y, predict_native(model, valid_x[:, cols], task), task)
        if not np.isfinite(utility):
            raise ValueError(f"Nonfinite validation score at width {k}")
        curve.append(float(utility))
        if utility > best + 1e-14:
            best, best_width = utility, k
    near_width = next(k for k, u in enumerate(curve, 1) if u >= max(curve) - .005 - 1e-14)
    support = sorted(order[:best_width])
    dev = np.sort(np.concatenate([train, valid]))
    final = fit_native(x[np.ix_(dev, support)], y[dev], task, seed, device)
    predicted = predict_native(final, x[np.ix_(idx["test"], support)], task)
    elapsed = perf_counter()-started
    return {
        "method":"direct_gain", "retained":best_width, "support":support,
        "validation_utility":best, "full_validation_utility":full_utility,
        "width_curve":curve, "gain_order":order, "near_width":near_width,
        "near_support":sorted(order[:near_width]), "near_validation_utility":curve[near_width-1],
        "fits":width+1, "worker_s":elapsed,
    }, predicted
