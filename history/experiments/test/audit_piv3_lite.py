"""Recompute the raw-width Lite benchmark metrics and paired comparisons without fitting models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.test.piv3_raw_splits import DEFAULT_ROOT, PROJECT
from experiments.test.run_piv3_lite_six import DATASETS, METHODS
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


def _read(path):
    return json.loads(path.read_bytes())


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(value, issue):
    if not value:
        raise ValueError(issue)


def _subset(names, universe):
    seen = set(names)
    return bool(seen) and names == [c for c in universe if c in seen]


def _score(y, pred, dataset):
    _require(np.isfinite(pred).all() and len(y) == len(pred), "Invalid predictions or prediction count")
    if dataset == "abalone":
        _require(y.shape == pred.shape, "Regression prediction shape")
        return dict(r2=max(0., float(r2_score(y, pred))), mae=float(mean_absolute_error(y, pred)))
    _require(pred.shape == (len(y), len(np.unique(y))), "Classification prediction shape")
    _require(np.all((pred >= 0) & (pred <= 1)) and np.allclose(pred.sum(axis=1), 1, rtol=0, atol=2e-6),
             "Invalid class probabilities")
    auc = (roc_auc_score(y, pred[:, 1]) if pred.shape[1] == 2 else
           roc_auc_score(y, pred, multi_class="ovr", average="macro"))
    output = dict(auc=float(auc), accuracy=float(accuracy_score(y, pred.argmax(axis=1))))
    if dataset == "credit_card_fraud":
        output["ap"] = float(average_precision_score(y, pred[:, 1]))
    return output


def _verify_scores(recorded, recomputed):
    _require(recorded.keys() == recomputed.keys(), "Scored metrics differ")
    for name in recorded:
        _require(math.isclose(recorded[name], recomputed[name], rel_tol=0., abs_tol=1e-12),
                 f"{name}: recorded {recorded[name]}, recomputed {recomputed[name]}")


def _fails(reference, candidate, tolerance, ap_tolerance):
    return sorted(m for m, ref in reference.items()
                  if ((candidate[m]-ref) if m == "mae" else (ref-candidate[m]))
                  > abs(ref)*(ap_tolerance if m == "ap" else tolerance)+1e-12)


def _vote(folder, method, width, front):
    diag = _read(folder/"front.json")
    if not method.startswith("pi"):
        _require(not diag and len(front) == width, "Unexpected frontend for direct selector")
        return None
    policy = "mean" if method == "pi_mean" else "min"
    _require(diag["aggregate"] == policy and diag["front_fits"] == 0 and diag["input_count"] == width,
             "Unexpected voting configuration")
    if diag["sample_rows"]:
        votes = np.asarray(diag["fold_tier_rates"])
        _require(votes.shape == (2, 3, width) and np.all(np.abs(votes) <= 1), "Vote dimensions/range")
        rates = votes.mean(axis=0)
        np.testing.assert_array_equal(rates, diag["tier_rates"])
        np.testing.assert_array_equal(rates.min(axis=0) if policy == "min" else rates.mean(axis=0),
                                      diag["aggregate_scores"])
        _require(diag["removed_count"] == width-len(front), "Frontend width")
    if diag["bypass"]:
        _require(len(front) == width, "Bypassed frontend dropped columns")
    return diag


def _route(folder, data, method, test_y, valid_y, ranking_rows, guard_rows, timer):
    dataset, cols = data["dataset"], data["features"]
    r, frozen, access = (_read(folder/n) for n in ("result.json","frozen.json","test_access.json"))
    _require(timer["result"] == r and r["status"] == "completed" and timer["kill_requested_s"] is None
             and timer["exit_code"] == 0, "Worker/watchdog disagreement")
    _require(timer["limit_s"] == 300 and 0 < r["elapsed_s"] <= 300 and
             r["elapsed_s"] <= timer["elapsed_s"]+1e-7, "Timing or watchdog deadline")
    _require((r["dataset"], r["method"], r["seed"], r["width"]) == (dataset, method, 42, len(cols)),
             "Dataset or route identity")
    _require(frozen["policy"] == {k:v for k,v in data["policy"].items() if k != "utility"},
             "Prediction policy")
    h = canonical_hash(frozen)
    _require(h == r["frozen_hash"] == access["frozen_hash"] and
             access["test_x_sha256"] == data["sha256"]["test.npy"] and
             access["test_y_sha256"] == data["sha256"]["test_y.npy"], "Test access provenance")
    for key in ("dataset","method","seed","front","candidate","selected","failed_metrics",
                "validation_full","validation_candidate","validation_selected"):
        _require(r[key] == frozen[key], f"{key} differs from frozen selection")
    _require(_subset(r["front"], cols) and _subset(r["candidate"],r["front"]) and
             _subset(r["selected"], cols), "Invalid feature names or column order")
    _require([len(r[k]) for k in ("front","candidate","selected")] == [r["front_k"],r["candidate_k"],r["k"]],
             "Feature count mismatch")
    backend = method.removeprefix("pi_")
    needs_model = backend in {"pfi", "banzhaf", "gain"}
    rank_fits = int(backend == "extra_trees" or (needs_model and r["front"] != cols))
    reuse_candidate = r["candidate"] == cols or (r["candidate"] == r["front"] and needs_model)
    _require(r["front_fits"] == 0 and r["rank_fits"] == rank_fits and
             r["fits"] == 2 + rank_fits + int(not reuse_candidate), "Total fit count")
    for key in ("front_s","rank_s","guard_s","reference_s","final_s"):
        _require(0 <= r[key] <= r["elapsed_s"], f"Stage timing {key}")
    _require(math.isclose(r["selector_s"], r["front_s"]+r["rank_s"], abs_tol=1e-8) and
             sum(r[k] for k in ("selector_s","guard_s","reference_s","final_s")) <= r["elapsed_s"]+1e-8,
             "Pipeline accounting")
    with np.load(folder/"predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["guard_rows"], guard_rows)
        np.testing.assert_array_equal(z["ranking_rows"], ranking_rows)
        _require(frozen["guard_rows_sha256"] == canonical_hash(guard_rows.tolist()) and
                 frozen["rank_rows_sha256"] == canonical_hash(ranking_rows.tolist()), "Split hashes")
        np.testing.assert_array_equal(z["y_guard"], valid_y[guard_rows])
        np.testing.assert_array_equal(z["y_test"], test_y)
        full = _score(z["y_guard"], z["guard_full"], dataset)
        candidate = _score(z["y_guard"], z["guard_candidate"], dataset)
        test = _score(test_y, z["test_pred"], dataset)
    failed = _fails(full,candidate,.03,.05)
    _require(sorted(r["failed_metrics"]) == failed and r["fallback"] == bool(failed) and
             r["selected"] == (cols if failed else r["candidate"]), "Gate or fallback")
    for key, expected in (("validation_full",full),("validation_candidate",candidate),
                          ("validation_selected",full if failed else candidate),("test_scores",test)):
        _verify_scores(r[key], expected)
    votes = _vote(folder,method,len(cols),r["front"])
    return r, votes


def audit(root, allow_pending=False):
    root = Path(root)
    protocol = _read(root/"protocol.json")
    _require(tuple(protocol["datasets"]) == DATASETS and tuple(protocol["methods"]) == METHODS and
             protocol["seconds_limit"] == 300 and protocol["seed"] == 42, "Protocol grid")
    for file, digest in protocol["source_sha256"].items():
        _require(_sha(PROJECT/file) == digest, f"Code differs from the run protocol: {file}")
    status = Counter()
    rows, pairs, ablations, bypass = [], [], [], []
    for d in DATASETS:
        source = DEFAULT_ROOT/d
        m = _read(source/"metadata.json")
        _require(protocol["data"][d] == dict(width=m["width"],hash=_sha(source/"metadata.json")) and
                 m["dataset"] == d and m["seed"] == 42, f"{d} input identity")
        for name, digest in m["sha256"].items():
            _require(_sha(source/name) == digest, f"Input hash failed for {d}/{name}")
        test = np.load(source/"test_y.npy",allow_pickle=False)
        val = np.load(source/"val_y.npy",allow_pickle=False)
        ranking, guard = train_test_split(np.arange(len(val)),test_size=.5,random_state=42,
                                          stratify=None if d == "abalone" else val)
        ranking, guard = np.sort(ranking), np.sort(guard)
        _require(np.intersect1d(ranking,guard).size == 0 and len(ranking)+len(guard) == len(val),
                 "Ranking and gate overlap or miss observations")
        methods, diagnostics = {}, {}
        for method in METHODS:
            folder = root/d/method
            timer_file = folder/"watchdog.json"
            if not timer_file.exists():
                _require(allow_pending, f"Missing terminal result for {d}/{method}")
                status["pending"] += 1
                continue
            timer = _read(timer_file)
            state = timer["status"]
            _require(state in {"timed_out","completed"} and timer["limit_s"] == 300, "Unexpected worker state")
            status[state] += 1
            if state == "timed_out":
                _require(timer["kill_requested_s"] is not None and timer["elapsed_s"] >= 299.9, "Missing deadline")
                rows.append(dict(dataset=d,method=method,status=state,wall_s=timer["elapsed_s"]))
                continue
            try:
                r, diagnostics[method] = _route(folder,m,method,test,val,ranking,guard,timer)
            except (ValueError, AssertionError, KeyError) as err:
                raise ValueError(f"{folder}: {err}") from err
            methods[method] = r
            rows.append(dict(dataset=d,method=method,status=state,guard_full=r["validation_full"],
                             guard_candidate=r["validation_candidate"],test=r["test_scores"],
                             wall_s=timer["elapsed_s"],active_s=r["elapsed_s"],k=r["k"],fallback=r["fallback"]))
        if "pi" in methods and "pi_mean" in methods:
            base, mean = methods["pi"], methods["pi_mean"]
            _require(diagnostics["pi"]["fold_tier_rates"] == diagnostics["pi_mean"]["fold_tier_rates"],
                     f"{d}: min/mean ablations use different votes")
            a,b = set(base["front"]),set(mean["front"])
            ablations.append(dict(dataset=d,min_k=len(a),mean_k=len(b),mean_only=len(b-a),min_only=len(a-b),
                                  jaccard=len(a & b)/len(a | b)))
        for method in METHODS:
            plain = method.removeprefix("pi_")
            if method == plain or plain not in methods or method not in methods or "full" not in methods:
                continue
            a,b,full = methods[method], methods[plain], methods["full"]
            pairs.append(dict(dataset=d,method=plain,active_time_ratio=a["elapsed_s"]/b["elapsed_s"],
                              fail_vs_paired=_fails(b["test_scores"],a["test_scores"],.05,.05),
                              fail_vs_full=_fails(full["test_scores"],a["test_scores"],.05,.05)))
            if diagnostics[method]["bypass"]:
                bypass.append(dict(dataset=d,method=plain,same_candidate=a["candidate"]==b["candidate"],
                                   same_selected=a["selected"]==b["selected"],
                                   score_deltas={metric:a["test_scores"][metric]-v for metric,v in b["test_scores"].items()}))
    return dict(status=dict(status),routes=rows,pairs=pairs,ablations=ablations,bypass=bypass)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-pending",action="store_true")
    args = parser.parse_args()
    result = audit(args.run,args.allow_pending)
    if args.output:
        atomic_write_json(args.output,result)
        print(json.dumps(result["status"]))
    else:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
