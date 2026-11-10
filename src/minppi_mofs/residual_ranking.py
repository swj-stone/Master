"""Rank sampled-zero recovery candidates using training-only prediction residuals."""

import hashlib
from numbers import Integral

import numpy as np
from sklearn.model_selection import train_test_split

from minppi_mofs.modeling import _validated_frame, _validated_target, make_model
from minppi_mofs.removal_selection import _feature_order, _gains, _supplemental_features


def _residuals(y, prediction, task, n_classes):
    prediction = np.asarray(prediction, dtype=float)
    expected = (len(y),) if task == "regression" else (len(y), n_classes)
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise ValueError("Invalid holdout prediction shape or nonfinite values")
    if task == "regression":
        return (y - prediction)[:, None]
    if task == "binary":
        return (y - prediction[:, 1])[:, None]
    residual = np.eye(n_classes)[y] - prediction
    residual[:, np.bincount(y, minlength=n_classes) == 0] = 0.
    return residual


def _recovery_correlations(values, residual):
    """Root-mean-square Pearson correlation across output residuals."""
    def normalize(columns):
        columns = np.asarray(columns, dtype=float)
        scale = np.max(np.abs(columns), axis=0)
        centered = columns / np.where(scale > 0, scale, 1.)
        centered -= centered.mean(axis=0)
        norm = np.linalg.norm(centered, axis=0)
        return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)

    correlations = normalize(values).T @ normalize(residual)
    return np.minimum(np.sqrt(np.mean(correlations**2, axis=1)), 1.)


def rank_residual_recovery(train, yt, features, incumbent_mask, sample_minima, *,
                           task, model_params, backend, seed, utility=None):
    """Fit the frozen incumbent on 75% of training rows and score dropped zero columns on the rest.

    Return original-column recovery scores, retained-column XGBoost gains and
    positional split hashes. The frozen training frame is the only feature
    data this function accepts.
    """
    features = _feature_order(features)
    frame = _validated_frame(train, name="frozen training features")
    if tuple(frame.columns) != features:
        raise ValueError("Feature order must match training columns")
    labels = _validated_target(yt, row_count=len(frame), task=task)
    if isinstance(incumbent_mask, bool) or not isinstance(incumbent_mask, Integral) or not 0 < incumbent_mask < 1 << len(features):
        raise ValueError("Invalid retained mask")
    if sample_minima is None:
        raise ValueError("Sample minima are required")
    witnessed = {f for f in features if sample_minima.get(f) == -1}
    zeros = _supplemental_features(features, witnessed, sample_minima)
    retained = [f for i, f in enumerate(features) if int(incumbent_mask) & (1 << i)]
    recover = [f for f in features if f in zeros and f not in retained]
    scores = dict.fromkeys(recover, 0.)
    gains = dict.fromkeys(retained, 0.)
    metadata = {"fit_count": 0, "stage": "training_residual_ranking", "split": None}
    if not (witnessed | zeros).intersection(retained) or (len(retained) == 1 and not recover):
        metadata["skip_reason"] = "no nonempty swap or deletion"
        return scores, gains, metadata

    row_indices = np.arange(len(frame))
    stratified = task != "regression"
    fit, hold = (np.sort(part) for part in train_test_split(
        row_indices, test_size=.25, random_state=42, stratify=labels if stratified else None))
    metadata["split"] = dict(seed=42, index_space="zero-based positions in frozen training CSV",
                             stratified=stratified, sizes=[len(fit), len(hold)],
                             index_sha256=[hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest() for idx in (fit, hold)])
    model = make_model(task=task, model_params=model_params, backend=backend, seed=seed)
    safe_names = [f"f{i}" for i in range(len(retained))]
    selected = frame.loc[:, retained].set_axis(safe_names, axis=1)
    model.fit(selected.iloc[fit], labels[fit])
    model_gains = _gains(safe_names, model.get_booster().get_score(importance_type="gain"))
    gains = {f: model_gains[name] for f, name in zip(retained, safe_names, strict=True)}
    metadata["fit_count"] = 1
    if recover:
        predict = model.predict if task == "regression" else model.predict_proba
        prediction = predict(selected.iloc[hold])
        n_classes = len(np.unique(labels)) if stratified else 0
        residual = _residuals(labels[hold], prediction, task, n_classes)
        measured = _recovery_correlations(frame.loc[:, recover].iloc[hold].to_numpy(), residual)
        scores = {f: float(v) for f, v in zip(recover, measured, strict=True)}
    return scores, gains, metadata
