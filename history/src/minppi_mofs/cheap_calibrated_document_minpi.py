"""Development-only loss-calibrated budget for the original document minPI votes."""
import numpy as np

from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr, fast_document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.utility_minpi import _utility_inputs


def _fit_direction(X, labels, classes):
    """One diagonal ridge direction and the exact unit-ridge optimum on its ray."""
    stats = _counts(X, labels, classes)
    mean, prior = stats.term / X.shape[0], stats.classes / X.shape[0]
    diagonal = stats.rhs / (stats.square + 1.)[:, None]
    combined = X @ diagonal - mean @ diagonal
    target = np.eye(classes)[labels] - prior
    denominator = np.square(combined).sum() + np.square(diagonal).sum()
    eta = float(np.sum(target * combined) / denominator) if denominator > 0 else 0.
    return mean, prior, eta * diagonal, eta


def _prefix_losses(X, labels, order, mean, prior, beta):
    """All balanced squared-loss prefixes in O(nnz(X)*classes + p*classes).

    Residuals are q_i + offset. Only rows containing a newly added term alter
    q; centering changes the shared offset. Weighted first/second moments
    therefore give exact losses without a document-by-feature dense array.
    """
    classes = len(prior)
    counts = np.bincount(labels, minlength=classes)
    if (counts == 0).any():
        raise ValueError("Every class must occur in independent calibration rows")
    weights = 1. / (classes * counts[labels])
    q = np.eye(classes)[labels] - prior
    first = weights @ q
    second = float(np.sum(weights[:, None] * np.square(q)))
    offset = np.zeros(classes)
    losses = np.empty(X.shape[1] + 1)
    losses[0] = second
    columns = X.tocsc()
    for k, j in enumerate(order, 1):
        rows = columns.indices[columns.indptr[j]:columns.indptr[j + 1]]
        active_weights = weights[rows]
        mass = active_weights.sum()
        active_first = active_weights @ q[rows]
        coefficient = beta[j]
        second += float(-2 * active_first @ coefficient + mass * (coefficient @ coefficient))
        first -= mass * coefficient
        q[rows] -= coefficient
        offset += mean[j] * coefficient
        losses[k] = second + 2 * offset @ first + offset @ offset
    return losses


def cheap_calibrated_document_top_from_training(
    X, y, count=None, *, task="classification", random_state=1129,
    groups=None, ordered_groups=False, backend="auto",
):
    """Choose the shortest allowed prefix with no measured teacher-loss increase.

    Preserve the original two-direction zero/one-peer votes and their minimum.
    The 512 voting rows also fit a fixed joint linear predictor: its diagonal
    ridge coefficients share one exactly optimized global scaling. Remaining
    training rows calibrate the budget by class-balanced squared loss. Ties
    use only the voting rows' marginal strength, preventing calibration labels
    from entering the ranking. All prefix coefficients stay fixed; no prefix
    model is fitted.

    The fixed rule is the first k in [ceil(sqrt(p)), ceil(p/2)] whose calibration
    loss does not exceed the full fixed predictor's loss, up to numerical
    precision. No admissible prefix returns all columns and feasible=False.
    This is empirical teacher preservation, not a statistical confidence
    bound or a guarantee for the final downstream refitted classifier.

    This development prototype needs n>512 and at least two classes present
    in both voting and calibration rows. Grouped/ordered sampling is explicitly
    unsupported until a separate disjoint-group calibration design is fixed.
    """
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
    fit_stats = _counts(fit, labels[fit_rows], len(classes))
    if (fit_stats.classes == 0).any():
        raise ValueError("Every class must occur in voting rows")
    scores = original["scores"]
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    marginal = _row_dot(fit_stats.rhs, fit_stats.rhs) / (fit_stats.square + 1.)
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    mean, prior, beta, eta = _fit_direction(fit, labels[fit_rows], len(classes))
    losses = _prefix_losses(X[calibration_rows], labels[calibration_rows], rank, mean, prior, beta)
    floor, cap = min((p + 1) // 2, int(np.ceil(np.sqrt(p)))), (p + 1) // 2
    eligible = np.arange(floor, cap + 1)
    tolerance = 1e-12 * max(1., abs(losses[-1]))
    admissible = eligible[losses[eligible] <= losses[-1] + tolerance]
    feasible = bool(len(admissible))
    if count is None:
        count = int(admissible[0]) if feasible else p
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=original["zero_peer_scores"], one_peer_scores=original["one_peer_scores"],
        rank=rank, fit_rows=fit_rows, calibration_rows=calibration_rows, coefficient=beta,
        feature_mean=mean, prior=prior, eta=eta, prefix_losses=losses,
        full_teacher_loss=float(losses[-1]), selected_teacher_loss=float(losses[count]),
        empirical_best_count=int(np.argmin(losses[1:]) + 1), feasible=feasible,
        budget=int(count), floor=floor, cap=cap, folds=original["folds"], backend=original["backend"],
        budget_rule="shortest_prefix_no_empirical_balanced_teacher_loss_increase",
    )
