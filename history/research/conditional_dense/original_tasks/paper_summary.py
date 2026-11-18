"""Rebuild per-split scores and paired time ratios for the original thesis sources."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, ROOT, load_dataset, train
from research.conditional_dense.original_tasks.paper_splits import ARMS, SOURCES, f_support
from research.conditional_dense.original_tasks.regression import regression_support
from research.conditional_dense.vote_width.method import vote_support
from research.feature_game_eval.audit_controls import effect_size


def metrics(truth, pred, decision):
    if decision is None:
        return {"r2": r2_score(truth, pred), "mae": mean_absolute_error(truth, pred)}
    return {"auc": roc_auc_score(truth, decision), "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "accuracy": accuracy_score(truth, pred)}


def read_runs(first, second):
    group = ["source", "seed", "arm"]
    score_tables, timings, fingerprints = [], [], []
    for process, directory in enumerate((first, second), 1):
        protocol = json.loads((directory / "protocol.json").read_text())
        assert protocol["pass_id"] == process
        fingerprints.append(protocol["sha256"])
        trials = pd.read_csv(directory / "times.csv")
        units = len(SOURCES) * len(P["formal_seeds"])
        assert len(trials) == units * len(ARMS)**2
        assert not trials.duplicated(group + ["repeat"]).any()
        assert not trials.duplicated(["source", "seed", "repeat", "position"]).any()
        assert trials.groupby(group).position.nunique().eq(len(ARMS)).all()
        assert set(trials.arm) == set(ARMS) and trials.total_ms.gt(0).all()
        assert trials.select_ms.ge(0).all() and trials.select_ms.le(trials.total_ms).all()
        assert trials.groupby(group)[["p", "width"]].nunique().eq(1).all().all()
        times = trials.groupby(group, as_index=False).agg(p=("p","first"), width=("width","first"),
                        select_ms=("select_ms","median"), total_ms=("total_ms","median"))
        timings.append(times.assign(process=process))
        scores = pd.read_csv(directory / "scores.csv").set_index(group).sort_index()
        assert scores.index.is_unique and len(scores) == units * 8
        score_tables.append(scores)
    assert fingerprints[0] == fingerprints[1]
    for name, digest in fingerprints[0].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    pd.testing.assert_frame_equal(*score_tables)
    return score_tables[0], pd.concat(timings, ignore_index=True)


def boundary_violation(effects, columns):
    outside = np.setdiff1d(np.arange(len(effects)), columns)
    return max(0., float(effects[outside].max() - effects[columns].min())) if len(outside) else 0.


def check_split(source, seed, features, labels, task, score_rows, first, second, fixed_arrays):
    main = "auc" if task == "classification" else "r2"
    file = f"{source}-{seed}.npz"
    archived_dir = OUTPUT / "formal" / source / f"s{seed}"
    record = json.loads((archived_dir/"record.json").read_text())
    fold = {"source": source, "seed": seed, "task": task, "primary": main, "p": features.shape[1]}
    scores = {}
    with np.load(first/file) as a, np.load(second/file) as b, np.load(archived_dir/"predictions.npz") as old:
        assert set(a.files) == set(b.files)
        for field in a.files:
            np.testing.assert_array_equal(a[field], b[field])
        for key, original in (("train", "train_indices"), ("test", "test_indices"), ("truth","y_test")):
            np.testing.assert_array_equal(a[key], old[original])
        tr, te, truth = a["train"], a["test"], a["truth"]
        np.testing.assert_array_equal(np.sort(np.r_[tr,te]), np.arange(len(labels)))
        np.testing.assert_array_equal(labels[te], truth)
        x, y = features[tr], labels[tr]
        for arm in (*ARMS, "minpi_10pct"):
            cols = a[arm+"_cols"]
            width = len(cols)
            assert np.array_equal(cols, np.unique(cols)) and cols.min() >= 0 and cols.max() < x.shape[1]
            assert width == score_rows.loc[arm, "width"]
            scores[arm] = metrics(truth, a[arm+"_pred"], a.get(arm+"_decision"))
            for metric, value in scores[arm].items():
                np.testing.assert_allclose(score_rows.loc[arm, metric], value, rtol=0, atol=1e-12)
        fold["k"], fold["fixed_k"] = len(a["minpi_cols"]), len(a["minpi_10pct_cols"])
        k, fixed_k = fold["k"], fold["fixed_k"]
        assert fixed_k == (x.shape[1] + 9) // 10
        np.testing.assert_array_equal(a["full_cols"], np.arange(x.shape[1]))
        np.testing.assert_array_equal(a["random_cols"], np.sort(np.random.default_rng(seed).permutation(x.shape[1])[:k]))
        select = vote_support if task == "classification" else regression_support
        chosen, info = select(x, y, seed=seed)
        np.testing.assert_array_equal(a["minpi_cols"], chosen)
        np.testing.assert_array_equal(a["minpi_10pct_cols"], np.sort(np.argsort(-info["rank"],kind="stable")[:fixed_k]))
        for arm in ("exact", "f", "random", "full"):
            historical = record["arms"][arm]
            historic_label = "archived_" + arm
            scores[historic_label] = metrics(truth, old[arm+"_ridge_pred"], old.get(arm+"_ridge_score"))
            for metric, value in scores[historic_label].items():
                np.testing.assert_allclose(historical["backends"]["ridge"][metric], value, rtol=0, atol=1e-12)
                if arm != "full":
                    np.testing.assert_allclose(score_rows.loc[historic_label, metric], value, rtol=0, atol=1e-12)
            assert historical["k"] == (x.shape[1] if arm == "full" else fixed_k)
        effects = np.concatenate([effect_size(x[:,j:j+256], y, task) for j in range(0,x.shape[1],256)])
        expected = np.argsort(-effects, kind="stable")
        auto_f = a["f_cols"]
        assert len(auto_f) == k
        stable = f_support(x, y, task, fixed_k)
        prior = np.asarray(record["arms"]["f"]["selected"])
        f_audit = {"source": source, "seed": seed, "auto_k": k, "fixed_k": fixed_k,
                   "auto_f_diff_reference": len(np.setdiff1d(auto_f,expected[:k])),
                   "auto_f_boundary_gap": boundary_violation(effects, auto_f),
                   "fixed_f_diff_reference": len(np.setdiff1d(stable,expected[:fixed_k])),
                   "fixed_f_boundary_gap": boundary_violation(effects, stable),
                   "archived_f_columns_replaced": len(np.setdiff1d(prior, stable)),
                   "archived_f_boundary_gap": boundary_violation(effects, prior)}
        assert max(f_audit["auto_f_boundary_gap"],f_audit["fixed_f_boundary_gap"]) <= 1e-9
        if np.array_equal(stable, prior):
            pred, decision = old["f_ridge_pred"], old.get("f_ridge_score")
        else:
            xt = features[te]
            pred, decision = train("ridge", x[:, stable], y, xt[:, stable], task, seed)
        scores["stable_f_10pct"] = metrics(truth, pred, decision)
        prefix = f"{source}_{seed}_"
        fixed_arrays[prefix+"columns"] = stable
        fixed_arrays[prefix+"prediction"] = pred
        if decision is not None:
            fixed_arrays[prefix+"decision"] = decision
        f_audit["refit_minus_archived_primary"] = scores["stable_f_10pct"][main] - scores["archived_f"][main]
        fold["cross_process_arrays"] = len(a.files)
    for arm, values in scores.items():
        for metric, value in values.items():
            fold[f"{metric}_{arm}"] = value
        fold[f"score_{arm}"] = values[main]
    fold["full_minus_archive"] = fold["score_full"] - fold["score_archived_full"]
    assert abs(fold["full_minus_archive"]) <= 5e-5
    for comparator in ("archived_exact","archived_random","stable_f_10pct"):
        gap = fold["score_minpi_10pct"] - fold["score_"+comparator]
        fold["fixed_gap_"+comparator] = gap
        fold["fixed_win_"+comparator] = gap > 1e-12
    return fold, f_audit


def cost_summary(fold, times):
    by_process = times.pivot(index="process", columns="arm", values="total_ms")
    assert by_process.shape == (2,4)
    fold["removed"] = fold["k"] < fold["p"]
    for arm in ARMS:
        rows = times[times.arm==arm]
        assert rows.width.eq(fold["p"] if arm=="full" else fold["k"]).all() and rows.p.eq(fold["p"]).all()
        for row in rows.itertuples():
            fold[f"total_ms{row.process}_{arm}"] = row.total_ms
            fold[f"select_ms{row.process}_{arm}"] = row.select_ms
    for arm in ("full", "f", "random"):
        ratio = float((by_process.minpi / by_process[arm]).max())
        gap = fold["score_minpi"] - fold["score_"+arm]
        fold["ratio_"+arm], fold["gap_"+arm] = ratio, gap
        fold["faster_"+arm] = ratio < 1
        fold["win_"+arm] = gap > 1e-12
        fold["better_faster_"+arm] = ratio < 1 and gap > 1e-12
    fold["cut_faster"] = fold["removed"] and fold["faster_full"]
    fold["cut_faster_within_002"] = fold["cut_faster"] and fold["gap_full"] >= -.02 - 1e-12


def aggregate(table, keys):
    agg = dict(configurations=("seed","size"), k_min=("k","min"), k_max=("k","max"),
               removed=("removed","sum"), cut_faster=("cut_faster","sum"),
               cut_faster_within_002=("cut_faster_within_002","sum"))
    for col in table:
        if col.startswith(("score_","balanced_accuracy_","ratio_","gap_","fixed_gap_")):
            agg["median_"+col] = (col,"median")
        if col.startswith(("fixed_win_","win_","better_faster_","faster_")):
            agg[col] = (col, "sum")
    return table.groupby(keys).agg(**agg).reset_index()


def summarize(first, second, out):
    scores, medians = read_runs(first, second)
    folds, f_checks, fixed_arrays = [], [], {}
    for source in SOURCES:
        X,y,task,*_ = load_dataset(source)
        for seed in P["formal_seeds"]:
            fold, f_audit = check_split(source, seed, X, y, task, scores.loc[(source,seed)], first, second, fixed_arrays)
            cost_summary(fold, medians[(medians.source==source) & (medians.seed==seed)])
            folds.append(fold)
            f_checks.append(f_audit)
    cases, audit = pd.DataFrame(folds), pd.DataFrame(f_checks)
    assert len(cases)==30
    out.mkdir(parents=True,exist_ok=False)
    medians.to_csv(out/"process-medians.csv",index=False)
    cases.to_csv(out/"cases.csv",index=False)
    aggregate(cases,["source","task","primary","p","fixed_k"]).to_csv(out/"sources.csv",index=False)
    aggregate(cases,["task","primary"]).to_csv(out/"tasks.csv",index=False)
    audit.to_csv(out/"f-audit.csv",index=False)
    np.savez_compressed(out/"stable-f10.npz",**fixed_arrays)
    verification = dict(configurations=len(cases), timed_pipelines=len(medians)*len(ARMS),
        cross_process_arrays=int(cases.cross_process_arrays.sum()), archived_score_vectors=4*len(cases),
        current_score_vectors=5*len(cases), adaptive_F_checked=len(audit), stable_fixed_F_checked=len(audit),
        original_fixed_F_supports_changed=int((audit.archived_f_columns_replaced>0).sum()),
        archived_full_max_abs_score_difference=float(cases.full_minus_archive.abs().max()))
    (out/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps(verification))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("first","second","out"):
        parser.add_argument(argument,type=Path)
    options = parser.parse_args()
    with threadpool_limits(1):
        summarize(options.first,options.second,options.out)
