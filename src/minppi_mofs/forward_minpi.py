"""Forward ridge paths with anchored quota votes on nested candidate contexts."""

import numpy as np


def additions(fit_gram, fit_cross, vote_gram, vote_cross, prefix):
    """Validation squared-error improvements for all one-column ridge refits."""
    p = len(fit_gram)
    if prefix:
        sub = np.asarray(prefix)
        projections = np.linalg.solve(fit_gram[np.ix_(sub, sub)], fit_gram[sub])
        coefficients = np.linalg.solve(fit_gram[np.ix_(sub, sub)], fit_cross[sub])
        schur = fit_gram.diagonal() - np.einsum("ij,ij->j", fit_gram[sub], projections)
        cross = fit_cross - fit_gram[:, sub] @ coefficients
        residual = vote_cross - vote_gram[:, sub] @ coefficients
        residual = residual - projections.T @ residual[sub]
        norm = (vote_gram.diagonal() - 2*np.einsum("ij,ij->j", vote_gram[sub], projections)
                + np.einsum("ij,ij->j", projections, vote_gram[np.ix_(sub, sub)] @ projections))
    else:
        schur, cross = fit_gram.diagonal(), fit_cross
        norm, residual = vote_gram.diagonal(), vote_cross
    valid = np.ones(p, dtype=bool)
    valid[prefix] = False
    adjustment = np.zeros_like(cross)
    adjustment[valid] = cross[valid] / schur[valid, None]
    return np.sum(adjustment * (2*residual - norm[:, None]*adjustment), axis=1)


def rank_forward(x, y, z, truth, task, rule="min"):
    """Use previous and current prefixes as the three comparison coalitions."""
    if task == "classification":
        one_hot = np.eye(int(np.max(y))+1)
        y, truth = one_hot[y], one_hot[truth]
    else:
        y, truth = y[:, None], truth[:, None]
    center, target_center = x.mean(axis=0), y.mean(axis=0)
    x, z, y, truth = x-center, z-center, y-target_center, truth-target_center
    p = x.shape[1]
    gg, cross = x.T @ x + np.eye(p), x.T @ y
    vg, vc = z.T @ z / len(z), z.T @ truth / len(z)
    order, gains = [], []
    for k in range(p):
        if rule == "fit":
            now = additions(gg, cross, (gg-np.eye(p))/len(x), cross/len(x), order)
        else:
            now = additions(gg, cross, vg, vc, order)
        gains.append(now)
        panels = np.asarray([gains[j] for j in sorted({k//4, k//2, k})])
        if rule == "min":
            score = panels.min(axis=0)
        elif rule == "mean":
            score = panels.mean(axis=0)
        elif rule in ("current", "fit"):
            score = now.copy()
        else:
            raise ValueError(rule)
        score[order] = -np.inf
        order.append(int(np.argmax(score)))
    return np.array(order)
