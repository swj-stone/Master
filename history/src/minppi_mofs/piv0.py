"""PIv0: sequential relative-negative screening in a size-calibrated coalition game."""

import numpy as np
from scipy.special import betaincc, betaln, softmax
from scipy.stats import rankdata
from xgboost import DMatrix


PATHS = 96
TIE = 0.005
RISK = 0.05
RANK_TOL = 1e-12


class _PrefixUtility:
    """Score batches of additive raw margins with shape (prefix, output, sample)."""

    def __init__(self, metric, labels, outputs):
        self.metric = metric
        self.labels = np.asarray(labels)
        if metric == "r2" and outputs == 1:
            self.denominator = np.sum((self.labels - self.labels.mean()) ** 2)
            return
        if not ((metric == "roc_auc" and outputs == 1) or (metric == "ovr_auc" and outputs > 2)):
            raise ValueError(f"Unsupported frozen PIv0 utility/output pair: {metric}, {outputs}")
        classes = np.arange(2 if outputs == 1 else outputs)
        if not np.array_equal(np.unique(self.labels), classes):
            raise ValueError("Validation labels must cover the fitted classes for ROC AUC")
        targets = [1] if outputs == 1 else classes
        self.positives = [np.flatnonzero(self.labels == target) for target in targets]

    def __call__(self, margins):
        if self.metric == "r2":
            residual = margins[:, 0, :] - self.labels
            errors = np.einsum("ij,ij->i", residual, residual)
            if self.denominator == 0:
                return (errors == 0).astype(float)
            return np.maximum(0., 1. - errors / self.denominator)
        # Binary raw margins preserve ROC order. Macro OVR ranks class probabilities.
        scores = softmax(margins, axis=1) if self.metric == "ovr_auc" else margins
        ranks = rankdata(scores, axis=-1, method="average")
        auc = np.zeros(len(scores))
        for group, positive in enumerate(self.positives):
            n = len(positive)
            wins = ranks[:, group, :][:, positive].sum(axis=1) - n * (n + 1) / 2
            auc += wins / (n * (len(self.labels) - n))
        return auc / len(self.positives)


def _orders(generator_seed, width):
    generator = np.random.default_rng(generator_seed)
    return np.array([generator.permutation(width) for _ in range(PATHS)])


def negative_vote(model, valid, labels, *, metric, seed=42, progress=None):
    """Return each feature's first negative lock round and auditable vote counts.

    The bias is included in partial multiclass and regression predictions. For ROC
    AUC, omitting the constant binary bias reproduces the original Madelon game.
    Calibration and monitoring draw independent uniform orders on the full schema.
    A locked feature remains coalition context but receives no further votes.
    """
    matrix = DMatrix(valid, nthread=1)
    booster = model.get_booster()
    parts = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    bias = parts[:, :, -1].T.astype(float)
    phi = np.ascontiguousarray(parts[:, :, :-1].transpose(2, 1, 0), dtype=float)
    del parts
    width, outputs, samples = phi.shape
    if width < 3 or len(labels) != samples or not np.all(bias == bias[:, :1]):
        raise ValueError("Expected at least three features, aligned labels and a constant per-output bias")
    score = _PrefixUtility(metric, labels, outputs)
    full = phi.sum(axis=0)
    actual_margins = booster.predict(matrix, output_margin=True, strict_shape=True).T
    err = float(np.max(np.abs(full + bias - actual_margins)))
    if not np.allclose(full + bias, actual_margins, rtol=1e-4, atol=1e-4):
        raise ValueError(f"TreeSHAP full margins did not reconstruct the fitted model (max error {err})")
    actual_utility = float(score(actual_margins.astype(float)[None])[0])
    offset = 0. if metric == "roc_auc" else bias
    empty = float(score((np.zeros_like(full) + offset)[None])[0])
    full_score = float(score((full + offset)[None])[0])
    positions = np.arange(width)
    tiers = np.where(positions < width // 3, 0, np.where(positions < 2 * width // 3, 1, 2))

    def prefix_scores(order, edges):
        utilities = np.full(width + 1, np.nan)
        utilities[0], utilities[-1] = empty, full_score
        interior = edges[(edges > 0) & (edges < width)]
        if len(interior):
            cumulative = np.cumsum(phi[order], axis=0)
            # Bound sort/R² working memory on datasets with large class or row counts.
            batch_size = max(1, 350_000 // (outputs * samples))
            for start in range(0, len(interior), batch_size):
                endpoints = interior[start:start + batch_size]
                utilities[endpoints] = score(cumulative[endpoints - 1] + offset)
        return utilities

    calibration = _orders(seed + 100003, width)
    changes = np.empty((PATHS, width))
    samples_per_tier = np.zeros((3, width), dtype=int)
    for step, order in enumerate(calibration):
        changes[step] = np.diff(prefix_scores(order, np.arange(width + 1)))
        samples_per_tier[tiers, order] += 1
        if progress:
            progress("calibration", step + 1, PATHS)
    if (samples_per_tier == 0).any():
        raise ValueError("Calibration requires at least one observation per feature and tier")
    reference = changes.mean(axis=0)
    residuals = changes - reference
    votes = (residuals > TIE).astype(int) - (residuals < -TIE).astype(int)
    sums = np.zeros_like(samples_per_tier)
    for order, ballot in zip(calibration, votes, strict=True):
        sums[tiers, order] += ballot
    cal_min = (sums / samples_per_tier).min(axis=0)
    eligible = (cal_min[:, None] > cal_min[None, :] + RANK_TOL).sum(axis=0) > width / 2

    neg, pos = np.indices((PATHS + 1, PATHS + 1))
    evidence = (np.exp((neg + pos + 1) * np.log(2.) + betaln(neg + 1, pos + 1))
                * betaincc(neg + 1, pos + 1, .5))
    up = np.zeros_like(samples_per_tier)
    down = np.zeros_like(samples_per_tier)
    lock = np.zeros(width, dtype=int)
    counted_votes = scored_prefixes = 0
    for step, order in enumerate(_orders(seed, width), 1):
        active = np.flatnonzero(eligible[order] & (lock[order] == 0))
        if not len(active):
            break
        edges = np.union1d(active, active + 1)
        utilities = prefix_scores(order, edges)
        residual = utilities[active + 1] - utilities[active] - reference[active]
        up[tiers[active], order[active]] += residual > TIE
        down[tiers[active], order[active]] += residual < -TIE
        lock[eligible & (lock == 0) & (evidence[down, up].max(axis=0) >= 3 / RISK - 1e-10)] = step
        counted_votes += len(active)
        scored_prefixes += int(np.sum((edges > 0) & (edges < width)))
        if progress:
            progress("monitoring", step, PATHS)
    return dict(lock_round=lock.tolist(), calibration_min=cal_min.tolist(), eligible=eligible.tolist(),
                calibration_votes=width * PATHS, primary_votes=counted_votes,
                primary_ranked_prefixes=scored_prefixes, metric=metric, tie=TIE, paths=PATHS,
                empty_utility=empty, full_proxy_utility=full_score, model_margin_utility=actual_utility,
                full_margin_max_abs_error=err)
