"""Development-only comparison with a training-held permutation screen and fixed ridge grid."""
import argparse
import json
import platform
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import ROOT, SEEDS, choose, panels, quality
from experiments.minpi_study import P, file_hash


METHODS = ("pfi", "rational", "grouped", "loss", "f", "full")
ALPHAS = (.1, 1., 10.)
BUDGETS = (10, 50)
FIT_CAP, ASSESS_CAP, PERMUTATIONS = 256, 128, 3


def ridge(task, alpha):
    cls = Ridge if task == "regression" else RidgeClassifier
    return make_pipeline(StandardScaler(), cls(alpha=alpha, solver="lsqr",
                         tol=P["ridge_tol"], max_iter=P["ridge_max_iter"]))


def linear_permutation_losses(model, x_check, target, orders, task, row_weights=None):
    # In standardized coordinates, changing one feature moves every output by
    # exactly its fitted coefficient times the change in that single input.
    z = model[0].transform(x_check)
    estimator = model[-1]
    if task == "classification":
        baseline = estimator.decision_function(z)
    else:
        baseline = estimator.predict(z)
    scores = np.zeros(z.shape[1])
    for order in orders:
        delta = z[order] - z
        if task == "regression":
            estimated = baseline[:, None] + delta * estimator.coef_
            errors = np.ascontiguousarray((estimated - target[:, None]).T)
            scores += np.square(errors).mean(axis=1)
        else:
            if baseline.ndim == 1:
                decisions = baseline[:, None] + delta * estimator.coef_.reshape(-1)
                indices = (decisions > 0).astype(int)
            else:
                decisions = baseline[:, None, :] + delta[:, :, None] * estimator.coef_.T
                indices = np.argmax(decisions, axis=2)
            incorrect = np.ascontiguousarray((estimator.classes_[indices] != target[:, None]).T)
            scores += (incorrect * row_weights).sum(axis=1)
    return scores / len(orders)


def pfi(X, y, groups, task, seed, backend="direct"):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if task == "classification":
        held = groups == unique.max()  # next available batch inside the outer training prefix
    else:
        selected = rng.permutation(unique)[:max(1, len(unique)//4)]
        held = np.isin(groups, selected)
    def subsample(mask, cap):
        rows = np.flatnonzero(mask)
        return rng.choice(rows, size=min(cap, len(rows)), replace=False)
    fit, assess = subsample(~held, FIT_CAP), subsample(held, ASSESS_CAP)
    if np.intersect1d(groups[fit], groups[assess]).size:
        raise AssertionError("PFI model fit and assessment use overlapping groups")
    model = ridge(task, 1.).fit(X[fit], y[fit])
    x_check = X[assess].copy()
    target = y[assess]
    row_weights = None
    if task == "classification":
        _, codes, counts = np.unique(target, return_inverse=True, return_counts=True)
        row_weights = 1 / (len(counts) * counts[codes])
        def loss(pred):
            return np.sum((pred != target) * row_weights)
    else:
        def loss(pred):
            return np.mean((target-pred)**2)
    base_loss = loss(model.predict(x_check))
    orders = [rng.permutation(len(assess)) for _ in range(PERMUTATIONS)]
    if backend == "linear":
        importance = linear_permutation_losses(model, x_check, target, orders, task, row_weights) - base_loss
    else:
        importance = np.empty(X.shape[1])
        for col in range(X.shape[1]):
            clean = x_check[:, col].copy()
            permuted_loss = 0.
            for order in orders:
                x_check[:, col] = clean[order]
                permuted_loss += loss(model.predict(x_check))
            x_check[:, col] = clean
            importance[col] = permuted_loss / PERMUTATIONS - base_loss
    return np.argsort(-importance, kind="stable"), dict(
        importance=importance, fit=fit, assess=assess, base_loss=base_loss, orders=orders)


def replay_fast(output, previous):
    """Pair complete raw-input screens and refits, with the archived direct PFI as oracle."""
    output.mkdir(parents=True, exist_ok=False)
    trials, audits = [], []
    methods = ("direct_pfi", "linear_pfi", "rational", "loss", "grouped", "f", "full")
    previous_protocol = json.loads((previous/"protocol.json").read_text(encoding="utf-8"))
    current_hash = file_hash(Path(__file__))
    for name, expected in previous_protocol["source_sha256"].items():
        if (ROOT/name).resolve() != Path(__file__).resolve() and file_hash(ROOT/name) != expected:
            raise RuntimeError(f"Source drift from direct PFI experiment: {name}")
    (output/"protocol.json").write_text(json.dumps(dict(
        code_sha256=current_hash, previous_protocol_sha256=file_hash(previous/"protocol.json"),
        timing_repeats=3, alpha=1, budgets=BUDGETS, threads=1, methods=methods,
        timing="Each screen starts from raw training input. One PFI ranking serves both budgets; charge its full time to each standalone call. Clock excludes checking the archived oracle.",
        arithmetic="Batch all singleton perturbations using linear ridge coefficients. Validate each feature score, exact supports, and outputs against direct pipeline evaluations."
    ), indent=2), encoding="utf-8")
    for panel in panels("development", ("gas","supercon")):
        X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
        groups = panel.groups[panel.tr]
        prefix = f"{panel.source}-{panel.fold}"
        with np.load(previous/f"{prefix}-split.npz") as split:
            np.testing.assert_array_equal(panel.tr, split["train"])
            np.testing.assert_array_equal(panel.te, split["test"])
        for seed in SEEDS:
            first = np.random.default_rng([seed,panel.fold,2087]).permutation(methods)
            with np.load(previous/f"{prefix}-{seed}.npz") as oracle:
                for repeat in range(3):
                    for position, method in enumerate(np.roll(first, repeat)):
                        is_pfi = method in ("linear_pfi", "direct_pfi")
                        if is_pfi:
                            start = perf_counter_ns()
                            backend = "linear" if method == "linear_pfi" else "direct"
                            ranking, info = pfi(X, y, groups, panel.task, seed, backend)
                            rank_ms = (perf_counter_ns()-start)/1e6
                            np.testing.assert_array_equal(oracle["pfi_fit"], panel.tr[info["fit"]])
                            np.testing.assert_array_equal(oracle["pfi_assess"], panel.tr[info["assess"]])
                            np.testing.assert_array_equal(oracle["pfi_orders"], info["orders"])
                            actual, expected = info["importance"], oracle["pfi_importance"]
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                            audits.append(dict(source=panel.source,fold=panel.fold,seed=seed,
                                method=method, repeat=repeat, width=X.shape[1],
                                max_abs_error=np.max(np.abs(actual-expected))))
                        for budget in ((100,) if method == "full" else BUDGETS):
                            count = ceil(X.shape[1]*budget/100)
                            begin = perf_counter_ns()
                            if is_pfi:
                                cols = np.sort(ranking[:count])
                                select_ms = rank_ms + (perf_counter_ns()-begin)/1e6
                            else:
                                cols = choose(method, X, y, panel.task, seed, count)
                                select_ms = (perf_counter_ns()-begin)/1e6
                            begin = perf_counter_ns()
                            xf, xe = (X, Xt) if method=="full" else (X[:,cols], Xt[:,cols])
                            pred = ridge(panel.task, 1.).fit(xf,y).predict(xe)
                            fit_ms = (perf_counter_ns()-begin)/1e6
                            key = "pfi" if is_pfi else method
                            np.testing.assert_array_equal(cols,oracle[f"{key}_{budget}_columns"])
                            np.testing.assert_array_equal(pred,oracle[f"{key}_{budget}_1_prediction"])
                            trials.append(dict(source=panel.source,fold=panel.fold,seed=seed,
                                repeat=repeat,position=position,method=method,budget=budget,
                                score=quality(panel,pred), select_ms=select_ms,fit_ms=fit_ms,
                                total_ms=select_ms+fit_ms))
        pd.DataFrame(trials).to_csv(output/"trials.csv", index=False)
        pd.DataFrame(audits).to_csv(output/"audit.csv", index=False)
        print(f"Verified linear PFI {prefix}", flush=True)
    data = pd.DataFrame(trials)
    cases = data.groupby(["source","fold","seed","budget","method"],as_index=False).agg(
        score=("score","first"),select_ms=("select_ms","median"),total_ms=("total_ms","median"))
    cases.groupby(["source","budget","method"],as_index=False).agg(
        score=("score","mean"),median_select_ms=("select_ms","median"),
        median_pipeline_ms=("total_ms","median")).to_csv(output/"summary.csv",index=False)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), ROOT/"experiments/compare_predictive_minpi.py",
            ROOT/"experiments/minpi_study.py", ROOT/"experiments/minpi_study_new_cohorts.py",
            ROOT/"configs/minpi-study.json"]
    code += [ROOT/f"src/minppi_mofs/{name}.py" for name in ("exact_minpi", "word_minpi",
             "integer_minpi", "grouped_minpi", "loss_minpi", "piv2", "piv3_compact_filter")]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in code}
    protocol = dict(source_sha256=hashes, methods=METHODS, budgets=BUDGETS, seeds=SEEDS,
                    downstream_alphas=ALPHAS, selector_alpha=1., fit_cap=FIT_CAP,
                    assessment_cap=ASSESS_CAP, permutations=PERMUTATIONS, threads=1,
                    pfi_loss=dict(gas="one minus balanced accuracy", supercon="MSE"),
                    data="development folds only; deployment cohort held out",
                    sharing="One PFI ranking supplies both widths; screening time charged in full to each standalone call. Other supports fixed for all downstream alphas.",
                    interpretation="Prespecified alpha sensitivity on identical held-out rows; no alpha is selected from these outcomes. Timing is descriptive elapsed wall time.",
                    python=platform.python_version(), sklearn=sklearn.__version__)
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    for panel in panels("development", ("gas", "supercon")):
        tr, te = panel.tr, panel.te
        X, y, Xt = panel.X[tr], panel.y[tr], panel.X[te]
        if np.intersect1d(panel.groups[tr], panel.groups[te]).size:
            raise AssertionError("Overlapping development panels")
        name = f"{panel.source}-{panel.fold}"
        np.savez_compressed(output/f"{name}-split.npz", train=tr, test=te, y=panel.y[te],
                            groups=panel.groups[te])
        for seed in SEEDS:
            rotation = np.random.default_rng([seed, panel.fold, 137])
            saved = {}
            for method in rotation.permutation(METHODS):
                if method == "pfi":
                    begin = perf_counter_ns()
                    rank, diagnostics = pfi(X, y, panel.groups[tr], panel.task, seed)
                    ranking_ms = (perf_counter_ns()-begin)/1e6
                    for key, value in diagnostics.items():
                        saved["pfi_"+key] = tr[value] if key in ("fit", "assess") else value
                for budget in ((100,) if method == "full" else BUDGETS):
                    count = ceil(X.shape[1] * budget / 100)
                    begin = perf_counter_ns()
                    if method == "pfi":
                        cols = np.sort(rank[:count])
                        select_ms = ranking_ms + (perf_counter_ns()-begin)/1e6
                    else:
                        cols = choose(method, X, y, panel.task, seed, count)
                        select_ms = (perf_counter_ns()-begin)/1e6
                    if len(cols) != count or not np.array_equal(np.unique(cols), cols):
                        raise AssertionError("Invalid support")
                    saved[f"{method}_{budget}_columns"] = cols
                    for alpha in rotation.permutation(ALPHAS):
                        begin = perf_counter_ns()
                        xf, xe = (X, Xt) if method == "full" else (X[:, cols], Xt[:, cols])
                        pred = ridge(panel.task, alpha).fit(xf, y).predict(xe)
                        fit_ms = (perf_counter_ns()-begin)/1e6
                        saved[f"{method}_{budget}_{alpha:g}_prediction"] = pred
                        rows.append(dict(source=panel.source, fold=panel.fold, seed=seed,
                                         method=method, budget=budget, width=count, alpha=alpha,
                                         score=quality(panel,pred), select_ms=select_ms,
                                         fit_ms=fit_ms, total_ms=select_ms+fit_ms))
            np.savez_compressed(output/f"{name}-{seed}.npz", **saved)
        pd.DataFrame(rows).to_csv(output/"trials.csv", index=False)
        print(f"Completed {name}", flush=True)
    trials = pd.DataFrame(rows)
    summary = trials.groupby(["source","budget","method","alpha"], as_index=False).agg(
        score=("score","mean"), median_select_ms=("select_ms","median"),
        median_pipeline_ms=("total_ms","median"), cases=("score","size"))
    summary.to_csv(output/"summary.csv", index=False)
    for path in code:
        if file_hash(path) != hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during experiment: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast-replay", type=Path, metavar="DIRECT_RUN",
                        help="Time and verify ridge-specific fast PFI against a saved direct run")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fast_replay:
            replay_fast(args.output, args.fast_replay)
        else:
            run(args.output)
