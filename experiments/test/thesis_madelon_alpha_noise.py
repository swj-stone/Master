"""Training-only diagnostics for Madelon's sampled +1 features at K=2."""

import hashlib
import time

import numpy as np
from sklearn.model_selection import train_test_split

from minppi_mofs.modeling import make_model
from minppi_mofs.utility import compute_utility


PERMUTATION_REPEATS = 5


def check_positive_noise(train, yt, positive_columns, *, policy, margin, publish=None):
    """Screen full-model permutation importance, then verify weak columns by refitting.

    Positive columns from different alpha values may be passed as a union. Each
    result depends on the original column position and on the complete feature
    matrix, so sharing the diagnostic does not change a column's result.
    """
    started = time.monotonic()
    features = list(train.columns)
    requested = set(positive_columns)
    if not requested.issubset(features) or len(set(features)) != len(features):
        raise ValueError("Expected unique training columns containing all requested features")
    if policy["task"] != "binary" or policy["utility"] != "roc_auc":
        raise ValueError("This Madelon diagnostic requires a binary ROC-AUC policy")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Expected a positive finite validation margin")
    positions = [i for i, name in enumerate(features) if name in requested]
    seed = policy["seed"]
    tolerance = 0.1 * margin
    result = dict(
        positive_columns=[features[i] for i in positions],
        stable_positive=[], suspected_noise=[], inconclusive=[], rows=[],
        fit_count=0, prediction_count=0, fit_s=0., prediction_s=0.,
        seed=seed, permutation_repeats=PERMUTATION_REPEATS,
        practical_auc_change=tolerance, reference="full_feature_model",
        screening_rule="all five baseline-minus-permuted AUCs > 0 and mean drop >= practical_auc_change",
        refit_rule="among other columns, flag if dropped-minus-full confirmation AUC >= -practical_auc_change",
        scope="Diagnostic only. All +1 columns remain protected. The selector used the whole training partition.",
        split=None, permutation_baseline_auc=None, confirmation_baseline_auc=None,
    )

    def progress(stage, completed, feature=None):
        result["elapsed_s"] = time.monotonic() - started
        if publish is not None:
            publish(dict(stage=stage, feature=feature, completed=completed, total=len(positions),
                         fit_count=result["fit_count"], prediction_count=result["prediction_count"],
                         elapsed_s=result["elapsed_s"]))

    if not positions:
        progress("complete", 0)
        return result

    target = np.asarray(yt)
    if (target.shape != (len(train),) or not np.array_equal(np.unique(target), [0, 1])):
        raise ValueError("Expected aligned binary labels with both classes present")
    fit_rows, held_rows = train_test_split(
        np.arange(len(target)), test_size=.4, random_state=seed, stratify=target)
    screen_rows, confirm_rows = train_test_split(
        held_rows, test_size=.5, random_state=seed, stratify=target[held_rows])
    fit_rows, screen_rows, confirm_rows = map(np.sort, (fit_rows, screen_rows, confirm_rows))
    result["split"] = {
        role: dict(n_rows=len(rows), index_sha256=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest())
        for role, rows in (("fit", fit_rows), ("permutation", screen_rows), ("refit_confirmation", confirm_rows))
    }
    values = train.to_numpy(dtype=np.float32)
    X_fit, X_screen, X_confirm = values[fit_rows], values[screen_rows], values[confirm_rows]
    model_policy = {key: policy[key] for key in ("task", "model_params", "backend", "seed")}

    def fit(x):
        model = make_model(**model_policy)
        t = time.monotonic()
        model.fit(x, target[fit_rows])
        result["fit_s"] += time.monotonic() - t
        result["fit_count"] += 1
        return model

    def score(model, x, y):
        t = time.monotonic()
        prediction = model.predict_proba(x)
        result["prediction_s"] += time.monotonic() - t
        result["prediction_count"] += 1
        return float(compute_utility(y, prediction, metric="roc_auc"))

    progress("full_model_fit", 0)
    model = fit(X_fit)
    baseline = score(model, X_screen, target[screen_rows])
    result["permutation_baseline_auc"] = baseline
    for completed, position in enumerate(positions):
        name = features[position]
        progress("permutations", completed, name)
        shuffled = X_screen.copy()
        aucs = []
        for repeat in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, position, repeat]))
            shuffled[:, position] = rng.permutation(X_screen[:, position])
            aucs.append(score(model, shuffled, target[screen_rows]))
        drops = [baseline - value for value in aucs]
        mean_drop = float(np.mean(drops))
        row = dict(feature=name, permuted_auc=aucs, mean_permuted_auc=float(np.mean(aucs)),
                   auc_drop=drops, mean_auc_drop=mean_drop, positive_drop_count=sum(d > 0 for d in drops),
                   reduced_confirmation_auc=None, refit_delta_auc=None)
        if min(drops) > 0 and mean_drop >= tolerance:
            label = "stable_positive"
        else:
            progress("drop_column_refit", completed, name)
            if result["confirmation_baseline_auc"] is None:
                result["confirmation_baseline_auc"] = score(model, X_confirm, target[confirm_rows])
            if len(features) == 1:
                # An intercept-only prediction gives tied ranks for all observations.
                reduced = .5
            else:
                keep = np.arange(len(features)) != position
                reduced = score(fit(X_fit[:, keep]), X_confirm[:, keep], target[confirm_rows])
            delta = reduced - result["confirmation_baseline_auc"]
            row.update(reduced_confirmation_auc=reduced, refit_delta_auc=delta)
            label = "suspected_noise" if delta >= -tolerance else "inconclusive"
        row["diagnosis"] = label
        result[label].append(name)
        result["rows"].append(row)
    progress("complete", len(positions))
    return result
