"""Exact fixed-panel classification votes with deferred scoring and paired mask prediction."""
import numpy as np

from .piv3_compact_witness import _within


class LazyAccuracyBallot:
    """Provides the 3%-accuracy ballot used with the MinPI AUC/AP panel scorer.

    The predictor must act on each row independently of other batch rows. The
    full scorer must include the same weighted argmax accuracy as this first check.
    """

    def __init__(self, predict_proba, rows, background, labels, weights, anchor, scorer):
        self.predict_proba = predict_proba
        self.rows = rows
        self.background = background
        self.labels = labels
        self.weights = weights
        self.mass = weights.sum(axis=1)
        self.anchor = frozenset(anchor)
        self.scorer = scorer
        self.buffer = np.empty((2, *rows.shape), dtype=rows.dtype)
        self.accuracy = {}
        self.pending = {}
        self.scores = {}
        self.dispatches = 0
        self.accuracy_vetoes = 0

    def _load(self, requested):
        missing = list(dict.fromkeys(k for k in requested if k not in self.accuracy))
        if not missing:
            return
        for slot, visible in enumerate(missing):
            block = self.buffer[slot]
            block[:] = self.background
            columns = sorted(visible)
            block[:, columns] = self.rows[:, columns]
        batch = self.buffer[:len(missing)].reshape(-1, self.rows.shape[1])
        probabilities = self.predict_proba(batch)
        self.dispatches += 1
        for visible, prob in zip(missing, np.split(probabilities,len(missing)), strict=True):
            self.pending[visible] = prob
            self.accuracy[visible] = (self.weights @ (prob.argmax(axis=1) == self.labels)) / self.mass

    def _score(self, visible):
        if visible not in self.scores:
            self.scores[visible] = self.scorer(self.pending.pop(visible))
        return self.scores[visible]

    def __call__(self, support, extension):
        after, before = support | extension, self.anchor | extension
        self._load((after,before))
        reference = self.accuracy[before]
        if np.all(reference - self.accuracy[after] > .03 * np.abs(reference) + 1e-12):
            self.accuracy_vetoes += 1
            return (False,) * len(reference)
        return tuple(_within(a,b) for a,b in zip(self._score(after),self._score(before),strict=True))
