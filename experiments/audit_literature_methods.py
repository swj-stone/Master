"""Independently audit the recorded estimators, fit ledger and decisions without fitting a model."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.run_literature_benchmark import CONTROLS, DATASETS, DEFAULT, ESTIMATORS
from experiments.test.piv3_raw_splits import DEFAULT_ROOT


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        result = sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


def json_hash(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def check_ledger(folder, timer, width, max_workers=1):
    path = folder / "fits.jsonl"
    lines = path.read_bytes().splitlines() if path.exists() else []
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            require(i == len(lines)-1 and timer["status"] in ("timed_out","manually_stopped"), "Broken fit record")
    completed, started, in_flight = [], {}, set()
    for i, event in enumerate(records):
        fit_id = event["fit_id"]
        require(0 < event["k"] <= width, "Fit width exceeds input width")
        if i:
            require(event["elapsed_s"] >= records[i-1]["elapsed_s"], "Nonmonotone fit time")
        if event["event"] == "started":
            require(fit_id==len(started)+1 and len(in_flight)<max_workers,"Fit ID or concurrency")
            started[fit_id]=event
            in_flight.add(fit_id)
        else:
            require(event["event"]=="completed" and fit_id in in_flight, "Unpaired completion")
            begin=started[fit_id]
            require(all(event.get(k)==begin.get(k) for k in ("phase","k","prefix_sha256","methods")),
                    "Mismatched fit")
            completed.append(dict(event,fit_time_s=event["elapsed_s"]-begin["elapsed_s"]))
            in_flight.remove(fit_id)
    if records:
        require((records[0]["phase"], records[0]["k"]) == ("reference", width), "Cold full model")
    if timer["status"] in ("timed_out","manually_stopped"):
        if timer["status"] == "timed_out":
            require(timer["kill_requested_s"] >= 299.9, "Watchdog fired before budget")
        require(not timer["result"] and not (folder/"result.json").exists(), "Interrupted result published")
    else:
        require(len(records) == 2*len(completed), "Incomplete fit on a completed route")
        require(timer["kill_requested_s"] is None and timer["exit_code"] == 0, "Exit and timeout")
    return completed, len(in_flight)


def metrics(y, pred, task, fraud):
    require(np.isfinite(pred).all(), "Finite predictions")
    if task == "regression":
        require(pred.shape == y.shape, "Regression vector shape")
        return dict(r2=max(0., r2_score(y, pred)), mae=np.mean(np.abs(y-pred)))
    require(pred.shape == (len(y), len(np.unique(y))), "Probability matrix shape")
    np.testing.assert_allclose(pred.sum(axis=1), 1, atol=2e-6, rtol=0)
    require(np.all((pred>=0) & (pred<=1)), "Invalid class probability")
    auc = (roc_auc_score(y, pred[:,1]) if task == "binary" else
           roc_auc_score(y, pred, average="macro", multi_class="ovr"))
    scores = dict(auc=auc, accuracy=np.mean(y == pred.argmax(axis=1)))
    if fraud:
        scores["ap"] = average_precision_score(y, pred[:,1])
    return scores


def permutation(info, mode, method, columns, candidate, active):
    width, n = len(columns), len(active)
    evidence = np.asarray(info["sampled_marginals"])
    require(info["paths"] == 4 and evidence.shape == (4, width) and np.isfinite(evidence).all(),
            "Incomplete permutation marginals")
    require(set(info["active_indices"]) == active and len(info["active_indices"]) == n,
            "Permutation projection")
    require(info["implied_zero_marginals"] == 4*(width-n) and info["sampled_marginal_count"] == 4*n
            and info["all_covered"] and info["feature_count"] == width, "Permutation coverage")
    np.testing.assert_array_equal(evidence[:, sorted(set(range(width))-active)], 0)
    rng = np.random.default_rng(42)
    permutations = [rng.permutation(width) for _ in range(4)]
    serialized = np.asarray(permutations, dtype="<u4").tobytes()
    require(sha256(serialized).hexdigest() == info["permutation_sha256"], "Permutation hash")
    distinct = {0}
    for order in permutations:
        prefix = 0
        for i in order:
            if int(i) in active:
                prefix |= 1 << int(i)
                distinct.add(prefix)
    require(info["oracle_prefixes"] == info["oracle"]["actual_coalitions"] == len(distinct),
            "Distinct sampled prefixes")
    match(evidence.sum(axis=1), np.repeat(evidence[0].sum(), 4))
    scores = evidence.min(axis=0) if method == "minshap" else evidence.mean(axis=0)
    match(scores, info["scores"])
    if mode == "native" and method == "minshap":
        variance = np.asarray(info["variance_of_mean"])
        require(variance.shape == evidence.shape and (variance>=0).all() and info["alpha"] == .05,
                "MinShap significance inputs")
        worst = evidence.argmin(axis=0)
        np.testing.assert_array_equal(worst, info["worst_path"])
        threshold = np.sqrt(-2*np.log(.05)*variance[worst,np.arange(width)])
        match(threshold, info["thresholds"])
        support = [name for name, keep in zip(columns, scores>=threshold) if keep]
        require(support == candidate and info["loss_rows"] == info["oracle"]["utility_rows"],
                "Native MinShap support and validation sample")
        require(info["degenerate_zero_ties"] == np.sum((scores==0) & (threshold==0)), "Zero ties")
    return evidence


def banzhaf(info, width, active):
    oracle = info["oracle"]
    require((info["complementary_pairs"], info["sampled_masks"], info["features_covered"],
             info["zero_marginals_known"]) == (32,64,width,width-len(active)), "Banzhaf coverage")
    masks = np.random.default_rng(42).integers(2, size=(32,width)).astype(bool)
    cols = sorted(active)
    distinct = {np.packbits(m[cols]).tobytes() for row in masks for m in (row,~row)}
    require(oracle["coalition_calls"] == 64 and oracle["actual_coalitions"] == len(distinct),
            "Banzhaf unique coalition count")
    # All scores must arise from the same vector of 32 paired utility differences.
    design = (2.*masks[:,cols].T-1.)/32
    scores = np.asarray(info["scores"])[cols]
    differences = np.linalg.lstsq(design,scores,rcond=None)[0]
    match(design @ differences, scores)
    if np.linalg.matrix_rank(design) == 32:
        require(np.max(np.abs(differences)) <= 1+1e-8, "Infeasible Banzhaf utility differences")


def graph(info, width, active, train_rows):
    degree = np.asarray(info["graph_degree"])
    require(degree.shape == (width,) and np.all((degree>=0) & (degree<width)), "ShapG degrees")
    require(info["depth"] == 1 and info["max_neighbors"] == 3 and sum(degree) == 2*info["graph_edges"]
            and info["features_covered"] == width and info["construction_rows"] == train_rows
            and info["zero_marginals_known"] == width-len(active), "ShapG graph coverage")
    draws = marginals = 0
    for n in degree[sorted(active)]:
        count = 1 if n <= 3 else int(np.ceil(((n+.5)/3-.5)*(np.log(n)+np.euler_gamma)+.5))
        draws += count
        marginals += count * 2**min(int(n),3)
    require((draws, marginals) == (info["neighbor_draws"],info["local_marginals"]), "ShapG draw totals")
    return degree


def native_fit_schedule(method, width, completed):
    """Check actual successful fit widths against the seeded coalition schedule, even after a kill."""
    observed = [e["k"] for e in completed if e["phase"] == "coalition"]
    if not observed or method == "shapg":
        return
    full = (1 << width)-1
    seen = {0,full}
    expected = []
    rng = np.random.default_rng(42)
    if method == "banzhaf":
        for mask in rng.integers(2,size=(32,width)).astype(bool):
            row = int.from_bytes(np.packbits(mask,bitorder="little").tobytes(),"little")
            for key in (row,full^row):
                if key not in seen:
                    expected.append(key.bit_count())
                    seen.add(key)
    else:
        for _ in range(4):
            key = 0
            for k, i in enumerate(rng.permutation(width),1):
                key |= 1 << int(i)
                if key not in seen:
                    expected.append(k)
                    seen.add(key)
    require(observed == expected[:len(observed)], "Native coalition fit does not follow the seeded schedule")
    return len(expected)


def route(folder, timer, meta, rows, completed, pair_checks, models, enforce_guard):
    dataset, mode, method = folder.parts[-3:]
    columns, width = meta["features"], meta["width"]
    r = read(folder/"result.json")
    info = read(folder/"estimator.json")
    frozen = read(folder/"frozen.json")
    access = read(folder/"test_access.json")
    require((r["dataset"],r["mode"],r["method"],r["seed"],r["width"]) == (dataset,mode,method,42,width)
            and r == timer["result"] and r["fits"] == r["fits_started"] == len(completed), "Route identity")
    freeze = json_hash(frozen)
    require(access["frozen_hash"] == r["frozen_hash"] == freeze and all(
        access[k] == meta["sha256"][filename] for k, filename in
        [("test_x_sha256","test.npy"),("test_y_sha256","test_y.npy")]), "Test access freeze")
    candidate, delivered = r["candidate"], r["selected"]
    candidate_set = set(candidate)
    require(candidate == [c for c in columns if c in candidate_set] and delivered ==
            (columns if r["fallback"] else candidate) and r["candidate_k"] == len(candidate)
            and r["k"] == len(delivered), "Candidate or delivered feature list")
    phases = Counter(e["phase"] for e in completed)
    require(phases["reference"] == 1 and phases["final"] == bool(delivered) and
            phases["guard"] == (0 < len(candidate) < width) and phases["coalition"] == r["selection_fits"],
            "Fit stage decomposition")
    require(mode == "native" or r["selection_fits"] == 0, "Unexpected coalition refits")
    for phase, k in (("final", len(delivered)), ("guard", len(candidate))):
        require(all(e["k"] == k for e in completed if e["phase"] == phase), "Fit subset width")
    rank, guard, yr, yt, ntrain = rows
    policy = {k:v for k,v in meta["policy"].items() if k != "utility"}
    require(frozen["policy"] == policy and frozen["candidate"] == candidate and
            frozen["selected"] == delivered and json_hash(rank.tolist()) == frozen["rank_rows_sha256"]
            and json_hash(guard.tolist()) == frozen["guard_rows_sha256"], "Frozen inputs/policy/split")
    with np.load(folder/"predictions.npz", allow_pickle=False) as pred:
        for k, v in (("rank_rows",rank),("guard_rows",guard),("y_guard",yr[guard]),("y_test",yt)):
            np.testing.assert_array_equal(pred[k],v)
        for key, value in (("test_scores","test_pred"),("validation_full","guard_full"),
                           ("validation_candidate","guard_candidate"),("validation_selected","guard_selected")):
            recalc = metrics(yt if key == "test_scores" else yr[guard],pred[value],meta["task"],
                             dataset == "credit_card_fraud")
            require(recalc.keys() == r[key].keys(), "Reported metric schema")
            match(list(recalc.values()),[r[key][k] for k in recalc])
        np.testing.assert_array_equal(pred["guard_selected"],pred["guard_full" if r["fallback"] else "guard_candidate"])
        # Every independently fit reference uses the same policy. Matching delivered
        # subsets must produce the same held-out predictions across routes as well.
        for role, subset, values in (("guard",columns,pred["guard_full"]),("test",delivered,pred["test_pred"])):
            key = (dataset,role,tuple(subset))
            fingerprint = sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            require(models.setdefault(key,fingerprint) == fingerprint, "Unmatched replicated model predictions")
    failed = [k for k, b in r["validation_full"].items() if
              (r["validation_candidate"][k]-b)*(1 if k=="mae" else -1) > .05*abs(b)+1e-12]
    require(set(failed) == set(r["failed_metrics"]) and
            (enforce_guard and bool(failed)) == r["fallback"], "Guard decision")
    if "enforce_guard" in r or not enforce_guard:
        require(all(obj["enforce_guard"] == enforce_guard and obj["validation_loss_limit"] == .05
                    and obj["fallback"] == r["fallback"] for obj in (r,frozen)),
                "Guard configuration")
    require(frozen["failed_metrics"] == r["failed_metrics"], "Frozen diagnostic losses")
    if method == "full":
        require(candidate == columns and not r["fallback"], "Full feature support")
    if method == "pfi":
        scores = np.asarray(info["scores"])
        keep = np.flatnonzero(scores > 0).tolist() or [int(scores.argmax())]
        require(candidate == [columns[i] for i in keep] and info["repeats"] == 3, "PFI support")
    sizing = info.get("sizing", info if "masked_proposals" in info else None)
    if sizing is not None:
        order = sizing["order"]
        exclude = set(columns)-set(order[:sizing["k"]])
        require(candidate == [c for c in columns if c not in exclude], "Incorrect selected prefix")
        tested_passes = {q["k"] for q in sizing["masked_proposals"] if q["passes"]}
        prefix_k = sizing["anchor_k"] if method == "minpi" else sizing["k"]
        require(prefix_k == min(tested_passes | {sizing["exact_endpoint_width"]}), "Smallest observed prefix")
    if method == "minpi":
        live = set(info["anchor_order"][:info["anchor_k"]])
        require(live == set(info["anchor"]) and info["aggregate"] == "min", "minPI anchor")
        for witness in info["witnesses"]:
            votes = np.asarray(witness["votes"])
            require(votes.shape[0] == 3 and np.isin(votes,[-1,1]).all(), "minPI votes")
            nets = np.mean(votes,axis=1)
            match(nets, witness["net"])
            match(nets.min(), witness["score"])
            accepted = nets.min() > 0 and witness["cumulative_anchor_pass"]
            require(witness["accepted"] == accepted and set(witness["block"]) <= live, "minPI proposal")
            if accepted:
                live.difference_update(witness["block"])
        require(candidate_set == live, "minPI final support")
    if mode == "shared":
        return failed
    if info.get("support_only"):
        require((mode,method)==("native","minshap") and "scores" not in info,
                "Support certificate scope")
        counts = np.asarray(info["per_feature_marginals"])
        mins = np.asarray(info["observed_minimum"])
        thresholds = np.asarray(info["threshold_of_observed_minimum"])
        cert = np.asarray(info["negative_certificate_path"])
        require(all(v.shape==(width,) for v in [counts,mins,thresholds,cert])
                and np.isfinite(mins).all() and np.isfinite(thresholds).all() and (thresholds>=0).all()
                and info["alpha"] == .05 and info["paths"] == 4, "Support certificate inputs")
        require(np.all(np.where(cert<0,(cert==-1)&(counts==4)&(mins>=0),
                                (mins<0)&(cert<4)&(counts==cert+1))), "Invalid decision certificate")
        require(candidate==[c for c,m,t in zip(columns,mins,thresholds) if m>=t],
                "Certified candidate support")
        require(info["oracle_prefixes"]==info["oracle"]["actual_coalitions"]
                and info["sampled_marginal_count"]==sum(counts), "Certificate counters")
        return failed
    scores = np.asarray(info["scores"])
    require(scores.shape == (width,) and np.isfinite(scores).all(), "Score vector coverage")
    positions = {c:i for i,c in enumerate(columns)}
    active = set(range(width)) if mode == "native" else {positions[c] for c in sizing["order"]}
    require(mode == "native" or len(active) == r["effective_split_features"], "Frozen active projection")
    np.testing.assert_array_equal(scores[sorted(set(range(width))-active)], 0)
    if sizing is not None:
        order = [columns[i] for i in np.argsort(-scores,kind="stable") if i in active]
        require(order == sizing["order"], "Sorted score prefix")
    oracle = info["oracle"]
    require(all(r[k] == v for k,v in oracle.items()) and
            oracle["ranking_prediction_rows"] == oracle["actual_coalitions"]*oracle["utility_rows"]
            and oracle["utility_rows"] == (len(rank) if mode == "native" else min(256,len(rank))),
            "Oracle counters or sample size")
    if method in ("minshap","shapley"):
        evidence = permutation(info,mode,method,columns,candidate,active)
        if mode == "frozen":
            pair_checks.setdefault((dataset,"permutations"),[]).append(evidence)
        else:
            require(phases["coalition"] == info["oracle_prefixes"]-2, "Native permutation refits")
    elif method == "banzhaf":
        banzhaf(info,width,active)
    else:
        degree = graph(info,width,active,ntrain)
        pair_checks.setdefault((dataset,"graphs"),[]).append(degree)
    return failed


def audit(root, *, partial=False, stopped=None):
    proto = read(root/"protocol.json")
    enforce_guard = proto.get("enforce_guard",True)
    repo = Path(__file__).resolve().parents[1]
    require(all(digest(repo/path) == h for path,h in proto["source"].items()), "Pinned source change")
    results, pairs, models, missing = [], {}, {}, []
    for dataset in DATASETS:
        directory = DEFAULT_ROOT/dataset
        meta = read(directory/"metadata.json")
        require(digest(directory/"metadata.json") == proto["inputs"][dataset]["metadata_sha256"], "Raw metadata pin")
        if not partial or stopped:
            require(all(digest(directory/file) == h for file,h in meta["sha256"].items()), "Raw split pin")
        val, test, train = [np.load(directory/(role+"_y.npy"),mmap_mode="r") for role in ("val","test","train")]
        rank, guard = train_test_split(np.arange(len(val)),test_size=.5,random_state=42,
                                      stratify=None if meta["task"]=="regression" else val)
        rows = (np.sort(rank),np.sort(guard),val,test,len(train))
        for mode, methods in (("shared",CONTROLS),("frozen",ESTIMATORS),("native",ESTIMATORS)):
            for method in methods:
                folder = root/dataset/mode/method
                key = f"{dataset}/{mode}/{method}"
                if not (folder/"watchdog.json").is_file():
                    if key == stopped:
                        require(folder.is_dir() and not (folder/"test_access.json").exists(), "Manual stop artifacts")
                        timer = dict(status="manually_stopped", result={}, limit_s=300)
                    else:
                        require(partial and (not stopped or not folder.exists()), f"Missing route {folder}")
                        missing.append(key)
                        continue
                else:
                    timer = read(folder/"watchdog.json")
                try:
                    require(timer["status"] in ("completed","timed_out","manually_stopped") and timer["limit_s"] == 300, "Deadline")
                    workers=proto.get("native_workers",1) if mode=="native" and method in ("minshap","shapley") else 1
                    completed, open_fit = check_ledger(folder,timer,meta["width"],workers)
                    expected_fits = None
                    if mode == "native" and workers==1 and not (method=="minshap" and proto.get("minshap_support_only")):
                        expected_fits = native_fit_schedule(method,meta["width"],completed)
                    failed = None
                    if timer["status"] == "completed":
                        failed = route(folder,timer,meta,rows,completed,pairs,models,enforce_guard)
                    coalition_s = sum(e["fit_time_s"] for e in completed if e["phase"]=="coalition")
                    results.append(dict(dataset=dataset, mode=mode, method=method, status=timer["status"],
                                        completed_fits=len(completed), interrupted_fits=open_fit,
                                        native_coalition_fits_required=expected_fits,
                                        native_coalition_fits_completed=sum(e["phase"]=="coalition" for e in completed),
                                        native_completed_coalition_fit_seconds=coalition_s,
                                        validation_metrics_over_5_percent=failed))
                except Exception as exc:
                    raise ValueError(f"Audit failure at {folder}: {exc}") from exc
    checks = []
    for (dataset, name), arrays in pairs.items():
        if len(arrays) == 2:
            match(*arrays)
            checks.append(f"{dataset}/{name}")
    return dict(routes=results, checked_routes=len(results), matched_cross_route_evidence=checks,
                enforce_guard=enforce_guard,
                unstarted_or_pending_routes=missing, raw_split_hashes_checked=bool(stopped) or not partial,
                limits=["Banzhaf and ShapG per-coalition utilities were not retained. Their archived scores, "
                        "support, mathematical consistency and sampling counts can be checked; exact numeric "
                        "replay of their utility calls requires running the predictor again."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--stopped",help="Identify an explicitly interrupted route as dataset/mode/method")
    parser.add_argument("--output",type=Path)
    options = parser.parse_args()
    report = audit(options.run,partial=options.partial,stopped=options.stopped)
    if options.output:
        options.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"{report['checked_routes']} routes verified. "
          f"{len(report['matched_cross_route_evidence'])} paired route traces matched.")
