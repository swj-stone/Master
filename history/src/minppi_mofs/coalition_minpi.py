"""Greedy minPI for a cross-fitted additive predictor with signed loss-gain votes."""
import numpy as np

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.utility_minpi import _endpoint_gains, _fitted_halves, _utility_inputs


def _greedy_coalition_order(folds, count, tiers):
    """Compute conditional minima from held-row first and second moments."""
    folds = list(folds)
    p = folds[0][0].shape[1]
    sample_size = sum(len(check) for check, _, _ in folds)
    gains = np.zeros(p)
    overlap = np.zeros(p)
    for check, centered_target, coefficients in folds:
        empty, full = _endpoint_gains(check, centered_target, coefficients)
        gains += empty
        overlap += (empty - full) / 2

    selected = np.zeros(p, dtype=bool)
    order, worst_gains = [], []
    for step in range(count):
        available_peers = p - step - 1
        if tiers is None:
            smallest, largest = 0, count - step - 1
        else:
            smallest, largest = min(min(tiers), available_peers), min(max(tiers), available_peers)
        # Each available peer occurs in k / available_peers of the coalitions.
        # Positive total overlap lowers the gain as k grows; negative raises it.
        k = np.where(overlap >= 0, largest, smallest)
        worst = gains - (2 * k / max(1, available_peers)) * overlap
        worst[selected] = -np.inf
        winner = int(np.argmax(worst))
        selected[winner] = True
        order.append(winner)
        worst_gains.append(float(worst[winner] / sample_size))
        if step + 1 < count:
            # For effects d_j = x_j beta_j, sum_i <d_ij,d_iw> factorizes
            # into column and coefficient inner products. Only one row of
            # feature interactions is needed per greedy choice.
            cross = np.zeros(p)
            for check, _, coefficients in folds:
                cross += (check.T @ check[:, winner]) * (coefficients @ coefficients[winner])
            gains -= 2 * cross
            overlap -= cross
    return order, worst_gains


def coalition_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Select the largest worst-case mean loss reduction, conditioned on earlier choices.

    A held sample votes with the decrease in its squared prediction error when
    a candidate is added to the fixed selected set and k additional uniform
    peers. Candidate contributions use an opposite-fold ridge; missing
    coordinates take the corresponding fit-fold mean. Default k spans zero
    through the unfilled budget after selecting this candidate. Explicit k
    values are truncated to the number of available peers at each step.

    For squared loss the expected marginal is affine in k. Evaluating its
    endpoints and updating inner products after each selection exactly
    implements the finite-coalition expectation, up to floating-point error.
    """
    X, y = _utility_inputs(X, y, count, task)
    levels = None if tiers is None else _selection_tiers(tiers, X.shape[1])
    if count == X.shape[1]:
        return np.arange(count), {"selection_order": [], "worst_gains": [], "ballot_rows": 0}
    folds = list(_fitted_halves(X, y, task, random_state))
    order, worst_gains = _greedy_coalition_order(folds, count, levels)
    return np.sort(order), {"selection_order": order, "worst_gains": worst_gains,
                            "ballot_rows": sum(len(check) for check, _, _ in folds)}
