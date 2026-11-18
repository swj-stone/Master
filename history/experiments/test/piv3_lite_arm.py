"""One timed seed-42 route for the full-width, lightweight PI comparison."""

import json
from pathlib import Path
import time

import numpy as np
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.test.piv2_conventional_models import choose_model
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, load_split
from experiments.test.run_sampled_permutation_seed42 import batched_utility
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import make_model
from minppi_mofs.piv1 import _sample_rows
from minppi_mofs.utility import compute_utility


SEED = 42
BACKENDS = ("gain", "pfi", "mi", "banzhaf", "extra_trees", "f_score")
METHODS = ("full", "pi", "pi_mean", *BACKENDS, *("pi_" + backend for backend in BACKENDS))


def _matrix(frame):
    return np.ascontiguousarray(frame.to_numpy(copy=False), dtype=np.float32)


def _predict(model, x, regression):
    return np.asarray((model.predict if regression else model.predict_proba)(x), dtype=float)


def _metrics(y, pred, utility, fraud):
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    if utility == "r2":
        return dict(r2=compute_utility(y, pred, metric=utility), mae=float(np.mean(np.abs(y - pred))))
    result = dict(auc=compute_utility(y, pred, metric=utility), accuracy=float(np.mean(y == pred.argmax(axis=1))))
    if fraud:
        result["ap"] = float(average_precision_score(y, pred[:, 1]))
    return result


def _balanced_rows(labels, limit, regression, seed):
    """Stratified evaluation sample with all available minority classes represented."""
    rng = np.random.default_rng(seed)
    if len(labels) <= limit:
        return np.arange(len(labels))
    if regression:
        return np.sort(rng.choice(len(labels), limit, replace=False))
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) > limit:
        raise ValueError("Evaluation row budget smaller than number of classes")
    budget, blocks = limit, []
    for j, index in enumerate(np.argsort(counts, kind="stable")):
        n = min(int(counts[index]), budget // (len(classes) - j))
        blocks.append(rng.choice(np.flatnonzero(labels == classes[index]), n, replace=False))
        budget -= n
    return np.sort(np.concatenate(blocks))


def _positive(names, scores, threshold=0., inclusive=False):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError("Invalid feature scores")
    selected = np.flatnonzero(scores >= threshold if inclusive else scores > threshold)
    if not len(selected):
        selected = [int(scores.argmax())]
    return [names[i] for i in selected]


def _gain(model, width):
    # All fit inputs are NumPy matrices. Booster keys are positional, even for a single feature.
    scores = np.zeros(width)
    for name, gain in model.get_booster().get_score(importance_type="gain").items():
        scores[int(name.removeprefix("f"))] = gain
    return scores


def _pfi(model, x, y, metric, original_indices, progress):
    rows = _balanced_rows(y, 128, metric == "r2", SEED)
    x, y = _matrix(x.iloc[rows]), y[rows]
    base = batched_utility(y, _predict(model, x, metric == "r2")[None], metric)[0]
    used = np.flatnonzero(_gain(model, x.shape[1]) > 0)
    repeats = 3
    batch_size = max(1, min(16, 4_000_000 // (x.size * repeats)))
    scores = np.zeros(x.shape[1])
    for start in range(0, len(used), batch_size):
        features = used[start:start + batch_size]
        batch = np.broadcast_to(x, (len(features), repeats, *x.shape)).copy()
        for j, feature in enumerate(features):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, original_indices[feature]]))
            for rep in range(repeats):
                batch[j, rep, :, feature] = x[rng.permutation(len(x)), feature]
        pred = _predict(model, batch.reshape(-1, x.shape[1]), metric == "r2")
        utilities = batched_utility(y, pred.reshape(len(features)*repeats, len(x), *pred.shape[1:]), metric)
        scores[features] = base - utilities.reshape(len(features), repeats).mean(axis=1)
        progress("permutation", checked=start + len(features), total=len(used))
    return scores, dict(ranking_rows=rows.tolist(), shuffled_features=len(used), repeats=repeats,
                        mean_drops=scores.tolist(), unpermuted_utility=float(base))


def _banzhaf(model, train, val, y, metric, support_indices, original_width, progress):
    # The masks, sampled validation rows, and training-median background match
    # masked_banzhaf_importance. Projection preserves paired random numbers.
    row_seed, mask_seed, background_seed = np.random.SeedSequence(SEED).spawn(3)
    rows = _sample_rows(y, metric, np.random.default_rng(row_seed))
    y, x = y[rows], _matrix(val.iloc[rows])
    bg = np.sort(np.random.default_rng(background_seed).choice(len(train), min(1024, len(train)), replace=False))
    fill = np.asarray(np.median(train.iloc[bg].to_numpy(), axis=0), dtype=x.dtype)
    first = np.random.default_rng(mask_seed).integers(0, 2, size=(32, original_width)).astype(bool)
    first = first[:, support_indices]
    masks = np.concatenate([first, ~first])
    utilities = []
    batch_size = max(1, min(8, 4_000_000 // x.size))
    for start in range(0, len(masks), batch_size):
        batch = masks[start:start + batch_size]
        masked = np.where(batch[:, None, :], x[None, :, :], fill)
        pred = _predict(model, masked.reshape(-1, x.shape[1]), metric == "r2")
        utilities.extend(batched_utility(y, pred.reshape(len(batch), len(x), *pred.shape[1:]), metric))
        progress("masked_banzhaf", checked=start + len(batch), total=len(masks))
    scores = (2*masks.astype(float) - 1).T @ np.asarray(utilities) / 32
    return scores, dict(ranking_rows=rows.tolist(), masks=len(masks), importance=scores.tolist())


def _mi(train, yt, names, regression):
    rows = _balanced_rows(yt, 1024, regression, SEED)
    x, y = train.iloc[rows], np.asarray(yt)[rows]
    rng = np.random.default_rng(811)
    controls = np.sort(rng.choice(train.shape[1], min(64, train.shape[1]), replace=False))
    estimate = mutual_info_regression if regression else mutual_info_classif
    # The null threshold uses identical, original coordinates on both sides.
    shuffled = estimate(_matrix(x.iloc[:, controls]), rng.permutation(y), random_state=SEED)
    threshold = float(np.quantile(shuffled, .95))
    scores = estimate(_matrix(x.loc[:, names]), y, random_state=SEED)
    return _positive(names, scores, threshold), dict(null_quantile=.95, threshold=threshold,
            null_features=controls.tolist(), ranking_sample_rows=len(y), importance=scores.tolist())


def _f_score(train, yt, names, regression):
    # Constants or undefined tests receive p=1 and still count toward multiplicity.
    frame = train if names == train.columns.tolist() else train.loc[:, names]
    scores, p = (f_regression if regression else f_classif)(frame, yt)
    scores = np.nan_to_num(scores, nan=0., posinf=np.finfo(float).max, neginf=0.)
    p = np.where(np.isfinite(p), p, 1.)
    ordered_p = np.sort(p)
    passing = np.flatnonzero(ordered_p <= .05 * np.arange(1, len(names)+1) / len(names))
    threshold = float(ordered_p[passing[-1]]) if len(passing) else None
    flags = p <= threshold if threshold is not None else np.arange(len(p)) == np.argmax(scores)
    return [c for c, keep in zip(names, flags, strict=True) if keep], dict(
        p_values=p.tolist(), threshold=threshold, q=.05, ranking_sample_rows=len(yt),
        fallback_top1=threshold is None)


def _failures(reference, candidate):
    failed = []
    for metric, baseline in reference.items():
        tolerance = .05 if metric == "ap" else .03
        loss = candidate[metric] - baseline if metric == "mae" else baseline - candidate[metric]
        if loss > tolerance * abs(baseline) + 1e-12:
            failed.append(metric)
    return failed


def run_arm(dataset, method, output):
    """Evaluate one independent route; caller owns the external 300-second watchdog."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    output = Path(output)
    started = time.perf_counter()
    fits = 0
    previous_tick = 0.
    previous_phase = None

    def progress(phase, **counts):
        nonlocal previous_tick, previous_phase
        now = time.perf_counter()
        if phase != previous_phase or now - previous_tick >= 1:
            atomic_write_json(output/"progress.json", dict(dataset=dataset, method=method, phase=phase,
                              elapsed_s=now-started, fits=fits, **counts))
            previous_phase, previous_tick = phase, now

    progress("loading")
    data = DEFAULT_ROOT / dataset
    meta = json.loads((data/"metadata.json").read_bytes())
    policy = {k: v for k, v in meta["policy"].items() if k != "utility"}
    utility, regression = meta["utility"], meta["task"] == "regression"
    fraud = dataset == "credit_card_fraud"
    train, yt = load_split(DEFAULT_ROOT, dataset, "train")
    val, yv = load_split(DEFAULT_ROOT, dataset, "val")
    columns = train.columns.tolist()
    if columns != meta["features"] or columns != val.columns.tolist():
        raise ValueError("Input column order differs from the frozen split")
    index = {c: i for i, c in enumerate(columns)}
    rank_rows, guard_rows = train_test_split(np.arange(len(val)), test_size=.5, random_state=SEED,
                                             stratify=None if regression else yv)
    rank_rows, guard_rows = np.sort(rank_rows), np.sort(guard_rows)
    guard = val.iloc[guard_rows]
    yg = np.asarray(yv)[guard_rows]

    def subset(frame, names):
        return frame if names == columns else frame.iloc[:, [index[c] for c in names]]

    def fit_training(names):
        nonlocal fits
        model = make_model(**policy)
        model.fit(_matrix(subset(train, names)), yt)
        fits += 1
        return model

    progress("reference")
    begin = time.perf_counter()
    full_model = fit_training(columns)
    full_pred = _predict(full_model, _matrix(guard), regression)
    reference = _metrics(yg, full_pred, utility, fraud)
    reference_s = time.perf_counter() - begin
    front, front_details, front_s = columns, {}, 0.
    if method == "pi" or method.startswith("pi_"):
        progress("voting")
        begin = time.perf_counter()
        from minppi_mofs.piv3_lite import screen
        front, front_details = screen(train, yt, task=policy["task"], seed=SEED,
                                      aggregate="mean" if method == "pi_mean" else "min")
        front_s = time.perf_counter() - begin
        members = set(front)
        if not front or front != [c for c in columns if c in members]:
            raise ValueError("PI support must retain a nonempty subset in input-column order")

    backend = method.removeprefix("pi_")
    rank_details, rank_fits, rank_s = {}, 0, 0.
    candidate = front
    support_model = full_model if front == columns else None
    if backend in BACKENDS:
        begin = time.perf_counter()
        fit_before = fits
        progress("ranking")
        # Match all hyperparameters between ordinary and PI-prefixed arms.
        with threadpool_limits(limits=1):
            if backend == "extra_trees":
                _, n_fits, rank_details = choose_model(backend, subset(train, front), yt, policy,
                                                       original_width=len(columns))
                fits += n_fits
                scores = np.asarray(rank_details["scores"])
                rank_details.pop("selected", None)
                rank_details.pop("k", None)
                candidate = _positive(front, scores, scores.mean(), inclusive=True)
                rank_details["threshold"] = float(scores.mean())
            elif backend == "mi":
                candidate, rank_details = _mi(train, yt, front, regression)
            elif backend == "f_score":
                candidate, rank_details = _f_score(train, yt, front, regression)
            else:
                if support_model is None:
                    support_model = fit_training(front)
                offsets = [index[c] for c in front]
                if backend == "gain":
                    scores = _gain(support_model, len(front))
                    rank_details = dict(importance=scores.tolist())
                else:
                    pool = subset(val.iloc[rank_rows], front)
                    labels = np.asarray(yv)[rank_rows]
                    if backend == "pfi":
                        scores, rank_details = _pfi(support_model, pool, labels, utility, offsets, progress)
                    else:
                        scores, rank_details = _banzhaf(support_model, subset(train, front), pool, labels,
                                                         utility, offsets, len(columns), progress)
                candidate = _positive(front, scores)
        rank_s = time.perf_counter() - begin
        rank_fits = fits - fit_before
    proposal_s = time.perf_counter() - started

    progress("quality_check")
    begin = time.perf_counter()
    if candidate == columns:
        candidate_pred, candidate_metrics = full_pred, reference
    else:
        model = support_model if candidate == front and support_model is not None else fit_training(candidate)
        candidate_pred = _predict(model, _matrix(subset(guard, candidate)), regression)
        candidate_metrics = _metrics(yg, candidate_pred, utility, fraud)
    failed = _failures(reference, candidate_metrics)
    selected = columns if failed else candidate
    selected_metrics = reference if failed else candidate_metrics
    guard_s = time.perf_counter() - begin
    frozen = dict(dataset=dataset, method=method, seed=SEED, front=front, candidate=candidate,
                  selected=selected, failed_metrics=failed, validation_full=reference,
                  validation_candidate=candidate_metrics, validation_selected=selected_metrics,
                  rank_rows_sha256=canonical_hash(rank_rows.tolist()),
                  guard_rows_sha256=canonical_hash(guard_rows.tolist()), policy=policy)
    atomic_write_json(output/"frozen.json", frozen)
    atomic_write_json(output/"front.json", front_details)
    atomic_write_json(output/"ranking.json", rank_details)
    begin = time.perf_counter()
    progress("final_refit")
    final = make_model(**policy)
    x_final = np.concatenate([_matrix(subset(train, selected)), _matrix(subset(val, selected))])
    final.fit(x_final, np.concatenate([yt, yv]))
    fits += 1
    del x_final
    atomic_write_json(output/"test_access.json", dict(frozen_hash=canonical_hash(frozen),
                      test_x_sha256=meta["sha256"]["test.npy"], test_y_sha256=meta["sha256"]["test_y.npy"]))
    test, ye = load_split(DEFAULT_ROOT, dataset, "test")
    prediction = _predict(final, _matrix(subset(test, selected)), regression)
    test_scores = _metrics(ye, prediction, utility, fraud)
    final_s = time.perf_counter() - begin
    np.savez_compressed(output/"predictions.npz", guard_rows=guard_rows, ranking_rows=rank_rows, y_guard=yg,
                        guard_full=full_pred, guard_candidate=candidate_pred,
                        y_test=ye, test_pred=prediction)
    result = dict(dataset=dataset, method=method, seed=SEED, width=len(columns), front_k=len(front),
                  candidate_k=len(candidate), k=len(selected), front=front, candidate=candidate, selected=selected,
                  fits=fits, front_fits=0, rank_fits=rank_fits, reference_s=reference_s,
                  front_s=front_s, rank_s=rank_s, selector_s=front_s+rank_s, proposal_s=proposal_s,
                  guard_s=guard_s, final_s=final_s, elapsed_s=time.perf_counter()-started,
                  fallback=bool(failed), failed_metrics=failed, validation_full=reference,
                  validation_candidate=candidate_metrics, validation_selected=selected_metrics,
                  test_scores=test_scores, frozen_hash=canonical_hash(frozen), status="completed")
    atomic_write_json(output/"result.json", result)
    progress("completed", retained=len(selected))
    return result
