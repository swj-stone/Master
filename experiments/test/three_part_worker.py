"""Run one independently timed route of the seed-42 paired three-part study."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.test.piv2_conventional_models import choose_model
from experiments.test.piv3_lite_arm import _failures, _gain, _matrix, _metrics, _mi, _pfi, _positive, _predict
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, load_split
from minppi_mofs import piv3_compact_tree, piv3_compact_witness
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import make_model


METHODS = ("full", "gain", "min_gain", "mean_gain", "pfi", "min_pfi", "mean_pfi",
           "nsga", "min_nsga", "mean_nsga", "mi", "extra_trees")
SEED = 42


def _pick_front(front, reference, primary):
    """Freeze the Pareto operating point using only ranking-side measurements."""
    eligible = [point for point in front if not _failures(reference, point["scores"])]
    if eligible:
        return min(eligible, key=lambda p: (p["k"], -p["scores"][primary], p["id"])), "smallest_feasible"

    def maximum_loss(point):
        worst = 0.
        for metric, base in reference.items():
            v = point["scores"][metric]
            loss = v-base if metric == "mae" else base-v
            if loss > 0:
                worst = max(worst, loss/abs(base) if base else float("inf"))
        return worst, point["k"], point["id"]

    return min(front, key=maximum_loss), "least_worst_relative_loss"


def trial(dataset, method, out):
    if method not in METHODS:
        raise ValueError(f"Unknown study route: {method}")
    clock = time.perf_counter
    started = clock()
    out = Path(out)
    fits = 0
    last_phase, last_tick = None, 0.

    def progress(phase, **extra):
        nonlocal last_phase, last_tick
        now = clock()
        if phase != last_phase or now-last_tick >= 1:
            atomic_write_json(out/"progress.json", dict(dataset=dataset, method=method,
                              phase=phase, fits=fits, elapsed_s=now-started, **extra))
            last_phase, last_tick = phase, now

    progress("loading")
    meta = json.loads((DEFAULT_ROOT/dataset/"metadata.json").read_bytes())
    policy = {k: v for k,v in meta["policy"].items() if k != "utility"}
    task, utility = meta["task"], meta["utility"]
    regression = task == "regression"
    fraud = dataset == "credit_card_fraud"
    train, yt = load_split(DEFAULT_ROOT,dataset,"train")
    val, yv = load_split(DEFAULT_ROOT,dataset,"val")
    columns = list(train.columns)
    if columns != meta["features"] or list(val.columns) != columns or not train.columns.is_unique:
        raise ValueError("Invalid feature schema")
    rank_rows, guard_rows = train_test_split(np.arange(len(val)), test_size=.5, random_state=SEED,
                                             stratify=None if regression else yv)
    rank_rows, guard_rows = np.sort(rank_rows), np.sort(guard_rows)
    ranking, yr = val.iloc[rank_rows], np.asarray(yv)[rank_rows]
    guard, yg = val.iloc[guard_rows], np.asarray(yv)[guard_rows]
    options = dict(task=task, utility=utility, seed=SEED)

    def subset(frame, names):
        return frame if names == columns else frame.loc[:,names]

    def fit(names):
        nonlocal fits
        model = make_model(**policy)
        model.fit(_matrix(subset(train,names)), yt)
        fits += 1
        return model

    def predict(model, frame, names):
        return _predict(model, _matrix(subset(frame,names)), regression)

    def score(y, pred):
        return _metrics(y, pred, utility, fraud)

    progress("full_reference")
    start = clock()
    full = fit(columns)
    full_guard = predict(full, guard, columns)
    reference = score(yg, full_guard)
    ref_ranking = score(yr, predict(full, ranking, columns))
    reference_s = clock()-start

    aggregate, _, backend = method.partition("_")
    if aggregate not in ("min", "mean"):
        aggregate, backend = "none", method
    support = columns
    front_s = 0.
    front_details = dict(aggregate="none",front_fits=0)
    if aggregate != "none":
        progress("prefilter")
        start = clock()
        support, front_details = piv3_compact_witness.select(
            full, train, yt, ranking, yr, aggregate=aggregate, **options)
        front_s = clock()-start
    support_set = set(support)
    if (not support or len(support_set) != len(support)
            or [name for name in columns if name in support_set] != support):
        raise ValueError("Invalid prefilter support")

    progress("backend")
    start = clock()
    fits_before = fits
    candidate = support
    support_model = full if support == columns else None
    candidate_model = None
    details = {}
    permutation_s = 0.
    if backend in ("gain","pfi"):
        if support_model is None:
            support_model = fit(support)
        block = subset(ranking,support)
        if backend == "gain":
            candidate, details = piv3_compact_tree.select(
                support_model,subset(train,support),yt,block,yr,aggregate="gain",**options)
        else:
            tick = clock()
            importance, details = _pfi(support_model,block,yr,utility,
                                       train.columns.get_indexer(support).tolist(),progress)
            permutation_s = clock()-tick
            candidate = _positive(support,importance)
    elif backend == "mi":
        candidate, details = _mi(train, yt, columns, regression)
    elif backend == "extra_trees":
        _, estimator_fits, details = choose_model(backend,train,yt,policy,original_width=len(columns))
        fits += estimator_fits
        values = np.asarray(details["scores"],dtype=float)
        details.pop("selected",None)
        details.pop("k",None)
        details["threshold"] = float(values.mean())
        candidate = _positive(columns,values,details["threshold"],inclusive=True)
    elif backend == "nsga":
        from experiments.test.three_part_nsga import search

        weights = _gain(full,len(columns))
        order = [columns[j] for j in np.argsort(-weights,kind="stable") if columns[j] in support_set]
        # Cache fitted boosters as compact bytes instead of keeping per-mask GPU training workspaces.
        # The eventual guard reuses the exact estimator selected on the ranking partition.
        snapshots = {}
        cache = {tuple(columns):ref_ranking}

        def evaluate(names):
            key = tuple(names)
            if key not in cache:
                model = fit(names)
                cache[key] = score(yr,predict(model,ranking,names))
                snapshots[key] = model.get_booster().save_raw(raw_format="ubj")
            return cache[key]

        details = search(support,columns,order,evaluate,progress,seed=SEED,population=8,budget=24)
        point, mode = _pick_front(details["front"],ref_ranking,"r2" if regression else "auc")
        candidate = list(point["names"])
        details.update(selected_id=point["id"],selection_mode=mode,reference_ranking=ref_ranking)
        if candidate != columns:
            candidate_model = make_model(**policy)
            candidate_model.load_model(snapshots[tuple(candidate)])
            # Fit-free reloads use the original execution backend for predictions.
            candidate_model.get_booster().set_param({"device":policy["backend"],"nthread":1})
        del snapshots
    elif backend != "full":
        raise ValueError(f"Unknown downstream selector: {backend}")

    candidate_set = set(candidate)
    if (not candidate or candidate != [c for c in support if c in candidate_set]
            or len(candidate) != len(candidate_set)):
        raise ValueError("Backend returned duplicate, unordered, or excluded columns")
    backend_s, backend_fits = clock()-start, fits-fits_before

    progress("guard")
    start = clock()
    before_guard = fits
    if candidate == columns:
        candidate_guard, guarded = full_guard, reference
    else:
        if candidate_model is None:
            candidate_model = support_model if candidate == support and support_model is not None else fit(candidate)
        candidate_guard = predict(candidate_model,guard,candidate)
        guarded = score(yg,candidate_guard)
    failed = _failures(reference,guarded)
    selected = columns if failed else candidate
    chosen_pred = full_guard if failed else candidate_guard
    selected_quality = reference if failed else guarded
    guard_s, guard_fits = clock()-start, fits-before_guard

    frozen = dict(dataset=dataset,method=method,seed=SEED,policy=policy,columns_sha256=canonical_hash(columns),
                  front=support,candidate=candidate,selected=selected,reference_ranking=ref_ranking,
                  validation_full=reference,validation_candidate=guarded,validation_selected=selected_quality,
                  failed_metrics=failed,rank_rows_sha256=canonical_hash(rank_rows.tolist()),
                  guard_rows_sha256=canonical_hash(guard_rows.tolist()))
    fingerprint = canonical_hash(frozen)
    atomic_write_json(out/"frozen.json",frozen)
    atomic_write_json(out/"front.json",front_details)
    atomic_write_json(out/"backend.json",details)
    start = clock()
    progress("final_refit")
    final = make_model(**policy)
    x = np.concatenate([_matrix(subset(frame,selected)) for frame in (train,val)])
    final.fit(x,np.concatenate([yt,yv]))
    fits += 1
    del x
    atomic_write_json(out/"test_access.json",dict(frozen_hash=fingerprint,
                      test_x_sha256=meta["sha256"]["test.npy"],test_y_sha256=meta["sha256"]["test_y.npy"]))
    test, ye = load_split(DEFAULT_ROOT,dataset,"test")
    if list(test.columns) != columns:
        raise ValueError("Held-out feature order changed")
    test_pred = predict(final,test,selected)
    test_scores = score(ye,test_pred)
    final_s = clock()-start
    np.savez_compressed(out/"predictions.npz",ranking_rows=rank_rows,guard_rows=guard_rows,
                        y_guard=yg,guard_full=full_guard,guard_candidate=candidate_guard,guard_selected=chosen_pred,
                        y_test=ye,test_pred=test_pred)
    result = dict(dataset=dataset,method=method,seed=SEED,aggregate=aggregate,backend=backend,
                  width=len(columns),front_k=len(support),candidate_k=len(candidate),k=len(selected),
                  selected=selected,front=support,candidate=candidate,fits=fits,fit_count=fits,front_fits=0,
                  backend_fits=backend_fits,guard_fits=guard_fits,front_s=front_s,backend_s=backend_s,
                  reference_s=reference_s,guard_s=guard_s,final_s=final_s,elapsed_s=clock()-started,
                  permutation_s=permutation_s,pfi_shuffled_features=details.get("shuffled_features"),
                  pfi_repeats=details.get("repeats"),candidate_is_full=(candidate==columns),
                  reference_ranking=ref_ranking,validation_full=reference,validation_candidate=guarded,
                  validation_selected=selected_quality,test_scores=test_scores,fallback=bool(failed),
                  failed_metrics=failed,frozen_hash=fingerprint,nsga_hv=details.get("hv"),
                  nsga_count=details.get("unique_evaluations"),
                  nsga_front_size=len(details["front"]) if backend=="nsga" else None,status="completed")
    atomic_write_json(out/"result.json",result)
    progress("completed",retained=len(selected))
    return result

