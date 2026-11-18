"""Check the selectors and Ridge predictions actually used in the cost and text studies.

Run from history with .\\thesis-python.ps1 -m research.baselines.verify_existing.
The command reads frozen results and prints a JSON report. It does not time methods.
"""
import hashlib
import json
import platform
import warnings
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.feature_selection import f_classif
from threadpoolctl import threadpool_limits

from experiments import compare_predictive_minpi as comparator
from experiments import document_vote_benchmark as documents
from experiments import fast_document_vote_benchmark as fast_text
from experiments import grouped_solver_cost_audit as dense_cost
from experiments import minpi_constant_transfer as sparse_cost
from experiments.shared_grouped_development import panels as dense_panels
from minppi_mofs.prefix_factor_minpi import factorized_grouped_top_from_training

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/minpi-study"
DENSE = ARCHIVE / "grouped-solver-cost-20261117a"
TEXT = ARCHIVE / "calibrated-same-width-dev-20261117a"
BLOCK = ARCHIVE / "conditional-block-fixed-sketch-20261117a"


def manual_anova(X, y):
    classes = np.unique(y)
    groups = [X[y == label] for label in classes]
    grand = X.mean(axis=0)
    between = sum(len(block) * (block.mean(axis=0) - grand)**2 for block in groups)
    within = sum(np.square(block - block.mean(axis=0)).sum(axis=0) for block in groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (between / (len(classes) - 1)) / (within / (len(y) - len(classes)))


def manual_sorted_f(X, y):
    scores = manual_anova(X, y)
    return np.argsort(-np.where(np.isnan(scores), -np.inf, scores), kind="stable")


def independent_ridge(X, y, Xt, solver):
    """Direct SVD-style augmented least squares, or scipy LSQR with the run's stopping rule."""
    means, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale == 0., 1., scale)
    train, test = (X - means) / scale, (Xt - means) / scale
    classes = np.unique(y)
    encoded = np.where(y[:, None] == classes, 1., -1.)
    if len(classes) == 2:
        encoded = encoded[:, 1:]
    offset = encoded.mean(axis=0)
    centered = encoded - offset
    if solver == "cholesky":
        augmented = np.vstack([train, np.eye(X.shape[1])])
        rhs = np.vstack([centered, np.zeros((X.shape[1], centered.shape[1]))])
        coef = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    else:
        coef = np.column_stack([lsqr(train, target, damp=1., atol=.001, btol=.001, iter_lim=250)[0]
                                for target in centered.T])
    decision = test @ coef + offset
    if len(classes) == 2:
        return decision.ravel(), classes[(decision.ravel() > 0).astype(int)]
    return decision, classes[decision.argmax(axis=1)]


def verify_fixtures():
    # NIST/SEMATECH e-Handbook 7.4.3.3 example: the printed statistic is rounded to 9.59.
    example = np.array([6.9,5.4,5.8,4.6,4., 8.3,6.8,7.8,9.2,6.5, 8.,10.5,8.1,6.9,9.3])
    labels = np.repeat([2, 9, 40], 5)
    X = np.column_stack([np.ones(15), example, example, labels, np.tile(np.arange(5), 3)])
    expected = manual_anova(X, labels)
    assert round(expected[1], 2) == 9.59
    for data in (X, sparse.csr_matrix(X)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(f_classif(data, labels)[0], expected, equal_nan=True)
        for width in range(1, X.shape[1] + 1):
            actual = comparator.choose("f", data, labels, "classification", 1129, width)
            np.testing.assert_array_equal(actual, np.sort([3, 1, 2, 4, 0][:width]))
    rng = np.random.default_rng(9)
    max_error = {}
    for n_class in (2, 4):
        y = 2 + 5 * np.repeat(np.arange(n_class), 16)
        train = rng.normal(size=(len(y), 8))
        test = rng.normal(size=(17, 8))
        train[:, 0], test[:, 0] = 1., 1.
        train[:, 3], test[:, 3] = train[:, 2], test[:, 2]
        panel = SimpleNamespace(X=train, y=y, Xt=test)
        for solver in dense_cost.SOLVERS:
            decision, labels_ref = independent_ridge(train, y, test, solver)
            pred, actual, _ = dense_cost.fit(panel, np.arange(train.shape[1]), solver)
            np.testing.assert_allclose(actual, decision, rtol=1e-9, atol=1e-9)
            np.testing.assert_array_equal(pred, labels_ref)
            max_error[f"{solver}_{n_class}classes"] = float(np.max(abs(actual-decision)))
        sparse_pred = sparse_cost.fit_predict(sparse.csr_matrix(train), y, sparse.csr_matrix(test),
                                               np.arange(train.shape[1]), "classification", 1129, False)
        np.testing.assert_array_equal(sparse_pred, independent_ridge(train, y, test, "lsqr")[1])
    # The pooled gain equals the decrease in penalized SSE vs the intercept-only fit.
    binary = rng.integers(2, size=(32, 8)).astype(float)
    binary[:, :2] = [0., 1.]
    y = np.where(np.arange(32) < 7, "minor", "major")
    Y = (y[:, None] == np.unique(y)).astype(float)
    Y -= Y.mean(axis=0)
    gains = []
    for raw in binary.T:
        column = raw - raw.mean()
        coeff = column @ Y / (column @ column + 1.)
        residual = Y - column[:, None] * coeff
        gains.append(np.sum(Y**2) - np.sum(residual**2) - coeff @ coeff)
    np.testing.assert_allclose(fast_text.pooled_binary_gain(sparse.csr_matrix(binary), y),
                               gains, atol=1e-12, rtol=1e-12)
    return {"nist_f":float(expected[1]), "ridge_max_abs_decision_error":max_error,
            "anova_all_widths_dense_and_csr":True, "pooled_gain_independent_objective":True}


def verify_dense():
    checked = []
    for panel in dense_panels():
        p = panel.X.shape[1]
        fixed = {"full":np.arange(p),
                 "F50":np.sort(manual_sorted_f(panel.X.astype(float), panel.y)[:ceil(p/2)]),
                 "random50":np.sort(np.random.default_rng(panel.seed).permutation(p)[:ceil(p/2)])}
        for rank in ("F", "representatives"):
            cols, info = factorized_grouped_top_from_training(panel.X, panel.y, random_state=panel.seed,
                                                               ranking=rank)
            fit, held = info["ranking_fit_rows"], info["calibration_rows"]
            assert not np.intersect1d(info["pilot_fit_rows"], held).size
            if rank == "F":
                ref_order = manual_sorted_f(panel.X[fit].astype(float), panel.y[fit])
                # Constant-column cancellation / near ties can perturb unused ordering.
                for width in info["candidate_widths"]:
                    np.testing.assert_array_equal(np.sort(ref_order[:width]), np.sort(info["ranking"][:width]))
            else:
                assert not info["voting"] and np.array_equal(info["ranking"], info["relevance_order"])
                ref_order = info["relevance_order"]
            cal = np.asarray([np.mean([np.mean(pred[panel.y[held] == lab] == lab)
                                      for lab in np.unique(panel.y[held])])
                              for pred in info["calibration_predictions"]])
            np.testing.assert_allclose(info["calibration_scores"], cal)
            winner = np.argmax(cal[:-1])
            count = info["candidate_widths"][winner] if cal[winner] >= cal[-1]-.02-1e-12 else p
            fixed[rank+"_bestcal"] = np.sort(ref_order[:count]) if count < p else np.arange(p)
            np.testing.assert_array_equal(cols, fixed[rank+"_bestcal"])
        archive_name = f"{panel.name}-1-{panel.seed}.npz"
        with np.load(DENSE / archive_name) as saved:
            for method, expected in fixed.items():
                actual = dense_cost.select(method, panel)
                np.testing.assert_array_equal(actual, expected)
                for solver in dense_cost.SOLVERS:
                    prefix = method+"_"+solver
                    np.testing.assert_array_equal(actual, saved[prefix+"_columns"])
                    pred, decision, _ = dense_cost.fit(panel, actual, solver)
                    np.testing.assert_array_equal(pred, saved[prefix+"_pred"])
                    np.testing.assert_array_equal(decision, saved[prefix+"_decision"])
        checked.append({"dataset":panel.name, "arms":len(fixed)*len(dense_cost.SOLVERS)})
    assert len(checked) == 2
    return checked


def verify_text():
    reports = []
    for X, y, Xt, _, meta in documents.panels():
        if meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp") or meta["fold"] != 1:
            continue
        assert sparse.isspmatrix_csr(X) and X.has_canonical_format and np.all(X.data == 1)
        p, seed = X.shape[1], meta["seed"]
        # Count positives by summing the actual values, independently of the
        # fast implementation which counts stored CSR indices.
        with np.load(BLOCK / f"{meta['dataset']}-1-{seed}.npz") as block:
            for scope, rows in (("all", np.arange(len(y))), ("proposal", block["global_proposal_rows"])):
                sample, labels = X[rows], y[rows]
                population = np.array([(labels == lab).sum() for lab in np.unique(labels)])
                successes = np.stack([np.asarray(sample[labels == lab].sum(axis=0)).ravel()
                                      for lab in np.unique(labels)])
                terms = successes.sum(axis=0)
                probabilities = successes / population[:, None]
                global_probability = terms / len(labels)
                between = np.sum(population[:, None] * (probabilities-global_probability)**2, axis=0)
                within = np.sum(population[:, None] * probabilities*(1-probabilities), axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    independent_f = between * (len(labels)-len(population)) / ((len(population)-1)*within)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    np.testing.assert_allclose(f_classif(sample, labels)[0], independent_f,
                                               rtol=1e-10, atol=1e-10, equal_nan=True)
                centered_counts = successes - population[:, None] * global_probability
                pooled = (centered_counts**2).sum(axis=0) / (terms*(1-global_probability) + 1.)
                np.testing.assert_allclose(fast_text.pooled_binary_gain(sample, labels), pooled,
                                           rtol=1e-12, atol=1e-12)
                for arm in ("F", "pooled"):
                    key = f"{arm}_{scope}"
                    cols = fast_text.select(arm+"_match", sample, labels, seed, len(block[key+"_cols"]))
                    np.testing.assert_array_equal(cols, block[key+"_cols"])
                    pred = sparse_cost.fit_predict(X, y, Xt, cols, "classification", seed, False)
                    np.testing.assert_array_equal(pred, block[key+"_pred"])
        with np.load(TEXT / (meta["dataset"]+"-arrays.npz")) as saved:
            for budget in ("a2", "b2"):
                width = len(saved[budget+"_minpi_support"])
                for arm, method in (("F","F_match"), ("pooled","pooled_match"), ("random_fixed","random_match")):
                    cols = fast_text.select(method, X, y, seed, width)
                    np.testing.assert_array_equal(cols, saved[budget+"_"+arm+"_support"])
                    pred = sparse_cost.fit_predict(X, y, Xt, cols, "classification", seed, False)
                    np.testing.assert_array_equal(pred, saved[budget+"_"+arm+"_prediction"])
                for draw in range(20):
                    permutation = np.random.default_rng([108391, 1, seed, draw]).permutation(p)
                    cols = saved[f"{budget}_random_{draw:02d}_support"]
                    np.testing.assert_array_equal(cols, np.sort(permutation[:width]))
        reports.append({"dataset":meta["dataset"], "two_budgets_matched_supports":46,
                        "two_budgets_named_control_predictions_replayed":6,
                        "conditional_block_F_and_pooled_full_and_proposal_arms":4})
    assert len(reports) == 4
    return reports


def main():
    with threadpool_limits(1):
        proof = {"environment":{"python":platform.python_version(), "numpy":np.__version__,
                                "scipy":scipy.__version__, "sklearn":sklearn.__version__},
                 "fixture_checks":verify_fixtures(), "dense_replay":verify_dense(), "text_replay":verify_text()}
    files = [Path(__file__), DENSE/"protocol.json", TEXT/"protocol.json",
             Path(comparator.__file__), Path(dense_cost.__file__), Path(documents.__file__),
             Path(fast_text.__file__), Path(sparse_cost.__file__)]
    proof["sha256"] = {path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in files}
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
