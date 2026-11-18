"""Development budgets using analytic prefix scaling of one fixed minPI direction."""
import numpy as np

from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr, fast_document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.utility_minpi import _utility_inputs


def _prefix_moments(X, target, order, mean, beta, weights, checkpoints=()):
    """Compute target/product moments for every prefix in sparse linear work."""
    checkpoints = set(checkpoints)
    columns = X.tocsc()
    classes = target.shape[1]
    q = np.zeros_like(target)
    offset = np.zeros(classes)
    first = np.zeros(classes)
    second, product = 0., 0.
    total_weight = float(weights.sum())
    target_sum = weights @ target
    base = float(np.sum(weights[:, None] * np.square(target)))
    cross, square = np.zeros(X.shape[1] + 1), np.zeros(X.shape[1] + 1)
    saved = {0: q.copy()} if 0 in checkpoints else {}
    for k, j in enumerate(order, 1):
        rows = columns.indices[columns.indptr[j]:columns.indptr[j + 1]]
        active_weights = weights[rows]
        mass = active_weights.sum()
        active_first = active_weights @ q[rows]
        coefficient = beta[j]
        second += float(2 * active_first @ coefficient + mass * (coefficient @ coefficient))
        first += mass * coefficient
        product += float((active_weights @ target[rows]) @ coefficient)
        q[rows] += coefficient
        offset -= mean[j] * coefficient
        cross[k] = product + offset @ target_sum
        square[k] = second + 2 * offset @ first + total_weight * (offset @ offset)
        if k in checkpoints:
            saved[k] = q + offset
    return base, cross, square, saved


def rescaled_calibrated_document_top_from_training(
    X, y, count=None, *, task="classification", random_state=1129,
    groups=None, ordered_groups=False, backend="auto", budget_rule="brier",
):
    """Keep minPI ranking, then calibrate prefix widths without multivariate refits.

    Fit a fixed marginal unit-ridge direction b on the 512 voting rows. For
    each prefix S, solve the scalar ridge problem analytically:
      eta_S = <Y_centered, X_centered,S b_S> /
              (||X_centered,S b_S||^2 + ||b_S||^2).
    Sparse moment updates compute all eta and calibration squared losses.

    A2 (brier) uses the shortest sqrt(p)..ceil(p/2) prefix with no empirical
    class-balanced squared-loss increase relative to the complete predictor.
    B2 (balanced_accuracy) considers only 16 fixed logarithmically spaced
    widths in that range and requires calibration BA >= full BA - 0.02.
    No feasible candidate returns all columns and is explicitly a failure.
    Calibration rows are disjoint from all voting/fitting/tie-breaking rows.

    Both rules preserve a restricted ray predictor, not the downstream
    refitted classifier. Choosing width on calibration rows is model selection;
    neither empirical criterion is a confidence or test-generalization bound.
    Groups and n<=512 remain explicitly outside this development prototype.
    """
    if budget_rule not in ("brier", "balanced_accuracy"):
        raise ValueError("budget_rule must be brier or balanced_accuracy")
    if groups is not None or ordered_groups:
        raise ValueError("Independent calibration for groups is not implemented")
    if task != "classification":
        raise ValueError("Document calibration requires classification labels")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    n, p = X.shape
    if n <= 512:
        raise ValueError("Independent calibration requires more than 512 training rows")
    if p < 2:
        raise ValueError("Development calibration requires at least two columns")
    classes, labels = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Need at least two classes")
    _, original = fast_document_vote_top_from_training(
        X, y, 1, task=task, random_state=random_state, backend=backend)
    fold = original["folds"][0]
    fit_rows = np.sort(np.concatenate((fold["fit_rows"], fold["held_rows"])))
    calibration_rows = np.setdiff1d(np.arange(n), fit_rows, assume_unique=True)
    fit = X[fit_rows]
    stats = _counts(fit, labels[fit_rows], len(classes))
    cal_counts = np.bincount(labels[calibration_rows], minlength=len(classes))
    if (stats.classes == 0).any() or (cal_counts == 0).any():
        raise ValueError("Every class must occur in both voting and calibration rows")
    scores = original["scores"]
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    marginal = _row_dot(stats.rhs, stats.rhs) / (stats.square + 1.)
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    mean, prior = stats.term / len(fit_rows), stats.classes / len(fit_rows)
    beta = stats.rhs / (stats.square + 1.)[:, None]
    target = np.eye(len(classes))[labels[fit_rows]] - prior
    _, fit_cross, fit_square, _ = _prefix_moments(
        fit, target, rank, mean, beta, np.ones(len(fit_rows)))
    penalty = np.r_[0., np.square(beta[rank]).sum(axis=1).cumsum()]
    denominator = fit_square + penalty
    eta = np.divide(fit_cross, denominator, out=np.zeros_like(fit_cross), where=denominator > 0.)
    floor, cap = min((p + 1) // 2, int(np.ceil(np.sqrt(p)))), (p + 1) // 2
    grid = np.unique(np.rint(np.geomspace(floor, cap, 16)).astype(int))
    cal_labels = labels[calibration_rows]
    weights = 1. / (len(classes) * cal_counts[cal_labels])
    target = np.eye(len(classes))[cal_labels] - prior
    base, cal_cross, cal_square, saved = _prefix_moments(
        X[calibration_rows], target, rank, mean, beta, weights, set(grid) | {p})
    losses = base - 2 * eta * cal_cross + eta**2 * cal_square

    def accuracy(k):
        prediction = np.argmax(prior + eta[k] * saved[k], axis=1)
        return float(weights @ (prediction == cal_labels))

    full_ba = accuracy(p)
    grid_ba = np.array([accuracy(int(k)) for k in grid])
    eligible = np.arange(floor, cap + 1)
    tolerance = 1e-12 * max(1., abs(losses[-1]))
    brier_pass = eligible[losses[eligible] <= losses[-1] + tolerance]
    ba_pass = grid[grid_ba >= full_ba - .02 - 1e-12]
    brier_budget = int(brier_pass[0]) if len(brier_pass) else p
    ba_budget = int(ba_pass[0]) if len(ba_pass) else p
    automatic = brier_budget if budget_rule == "brier" else ba_budget
    if count is None:
        count = automatic
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=original["zero_peer_scores"], one_peer_scores=original["one_peer_scores"],
        rank=rank, fit_rows=fit_rows, calibration_rows=calibration_rows, direction=beta,
        feature_mean=mean, prior=prior, eta_path=eta, prefix_losses=losses,
        full_teacher_loss=float(losses[-1]), selected_teacher_loss=float(losses[count]),
        empirical_best_count=int(np.argmin(losses[1:]) + 1), feasible=automatic < p,
        budget=int(count), floor=floor, cap=cap, folds=original["folds"], backend=original["backend"],
        budget_rule=budget_rule, brier_budget=brier_budget, ba_budget=ba_budget,
        brier_feasible=bool(len(brier_pass)), ba_feasible=bool(len(ba_pass)),
        full_teacher_ba=full_ba, ba_grid=grid, ba_grid_scores=grid_ba,
    )
