"""Inference-only, pickle-free artifact for a MinPI-restricted probability or regression forest."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


FORMAT = "minpi-restricted-sklearn-rf-1"
ARRAYS = ("medians", "roots", "left", "right", "feature", "threshold", "value")


@dataclass
class RestrictedRF:
    metadata: dict
    medians: np.ndarray
    roots: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray

    @property
    def names(self):
        return self.metadata["features"]

    @property
    def classes(self):
        return self.metadata["classes"]

    @classmethod
    def from_forest(cls, forest, support, names, classes, winner):
        """Persist only reachable nodes and selected feature medians, retaining frontier values."""
        cols = np.asarray(support, dtype=int)
        if (len(names) != forest.p or len(np.unique(cols)) != len(cols)
                or np.any((cols < 0) | (cols >= forest.p))):
            raise ValueError("Invalid original schema or support")
        index = {j: i for i, j in enumerate(cols)}
        roots, left, right, features, thresholds, values = [], [], [], [], [], []

        def copy_node(t, node):
            i = len(left)
            left.append(-1)
            right.append(-1)
            features.append(-1)
            thresholds.append(0.)
            values.append(forest.value[t, node])
            j = forest.feature[t, node]
            if forest.left[t, node] >= 0 and j in index:
                features[i] = index[j]
                thresholds[i] = forest.threshold[t, node]
                left[i] = copy_node(t, forest.left[t, node])
                right[i] = copy_node(t, forest.right[t, node])
            return i

        for t in range(len(forest.left)):
            roots.append(copy_node(t, 0))
        meta = dict(format=FORMAT, task="classification" if forest.classification else "regression",
                    features=[names[j] for j in cols], classes=list(classes), winner=winner)
        result = cls(meta, forest.medians[cols].copy(), np.asarray(roots, np.int32),
                     np.asarray(left, np.int32), np.asarray(right, np.int32),
                     np.asarray(features, np.int32), np.asarray(thresholds), np.asarray(values))
        result.validate()
        return result

    def validate(self):
        m = self.metadata
        if not isinstance(m, dict) or m.get("format") != FORMAT:
            raise ValueError("Unknown restricted forest format")
        if m.get("task") not in ("classification", "regression") or m.get("winner") not in ("mdi_min", "min_min"):
            raise ValueError("Invalid forest task or path")
        names, labels = m.get("features"), m.get("classes")
        if not isinstance(names, list) or not all(isinstance(s, str) and s for s in names) or len(set(names)) != len(names):
            raise ValueError("Invalid selected feature names")
        if not isinstance(labels, list) or not all(isinstance(s, str) and s for s in labels) or len(set(labels)) != len(labels):
            raise ValueError("Invalid class mapping")
        classification = m["task"] == "classification"
        if (classification and len(labels) < 2) or (not classification and labels):
            raise ValueError("Invalid number of classes")
        n = len(self.left)
        for arr in (self.roots, self.left, self.right, self.feature):
            if arr.ndim != 1 or arr.dtype.kind != "i":
                raise ValueError("Invalid node index buffer")
        for arr in (self.medians, self.threshold, self.value):
            if arr.dtype.kind != "f" or not np.isfinite(arr).all():
                raise ValueError("Invalid numeric forest buffer")
        if (not n or not len(self.roots) or self.roots[0] != 0 or self.roots[-1] >= n
                or np.any(self.roots[1:] <= self.roots[:-1])):
            raise ValueError("Invalid tree roots")
        if (self.medians.shape != (len(names),) or self.right.shape != (n,) or self.feature.shape != (n,)
                or self.threshold.shape != (n,) or self.value.shape != (n, max(1, len(labels)))):
            raise ValueError("Inconsistent forest dimensions")
        split = self.feature >= 0
        if ((self.feature < -1).any() or (self.feature[split] >= len(names)).any()
                or (self.left[~split] != -1).any() or (self.right[~split] != -1).any()):
            raise ValueError("Invalid split or leaf")
        nodes = np.flatnonzero(split)
        ends = np.r_[self.roots[1:], n][np.searchsorted(self.roots, nodes, side="right") - 1]
        for children in (self.left[split], self.right[split]):
            if np.any((children <= nodes) | (children >= ends)):
                raise ValueError("Out-of-tree child or non-forward edge")
        refs = np.r_[self.roots, self.left[split], self.right[split]]
        if not np.all(np.bincount(refs, minlength=n) == 1):
            raise ValueError("Nodes must be reachable exactly once")
        if classification and (np.any(self.value < 0.) or
                not np.allclose(self.value.sum(axis=1), 1. / len(self.roots), rtol=1e-10, atol=1e-12)):
            raise ValueError("Invalid class probabilities")

    def reachable_schema(self):
        """Drop input names that no surviving internal node can request."""
        used = np.unique(self.feature[self.left >= 0])
        if len(used) == len(self.names):
            return self
        remapped = self.feature.copy()
        live = self.left >= 0
        remapped[live] = np.searchsorted(used, self.feature[live])
        metadata = dict(self.metadata, features=[self.names[i] for i in used],
                        declared_features=len(self.names))
        result = RestrictedRF(metadata, self.medians[used], self.roots, self.left,
                              self.right, remapped, self.threshold, self.value)
        result.validate()
        return result

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            staged = Path(tmp.name)
        try:
            with staged.open("wb") as stream:
                np.savez_compressed(stream, metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)),
                                    **{name: getattr(self, name) for name in ARRAYS})
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != len(ARRAYS) + 1 or set(data.files) != {"metadata", *ARRAYS}:
                raise ValueError("Unexpected restricted forest members")
            meta = data["metadata"]
            if meta.shape != () or meta.dtype.kind != "U":
                raise ValueError("Invalid metadata encoding")
            result = cls(json.loads(meta.item()), *(data[name] for name in ARRAYS))
        result.validate()
        return result

    def predict(self, x, available=None):
        """Return class probabilities in classes order or one regression prediction per row."""
        raw = np.asarray(x, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.names) or np.isinf(raw).any():
            raise ValueError("Provide finite numeric inputs or NaN, in selected-schema order")
        if available is not None:
            available = np.asarray(available)
            if available.shape != raw.shape or available.dtype.kind != "b":
                raise ValueError("Availability must be a Boolean array matching the reduced input matrix")
        fixed = np.where(np.isnan(raw), self.medians, raw).astype(np.float32)
        result = np.zeros((len(raw), self.value.shape[1]))
        rows = np.arange(len(raw))
        for root in self.roots:
            node = np.full(len(raw), root, dtype=self.roots.dtype)
            active = rows
            while len(active):
                active = active[self.feature[node[active]] >= 0]
                j = self.feature[node[active]]
                if available is not None:
                    take = available[active, j]
                    active, j = active[take], j[take]
                old = node[active]
                node[active] = np.where(fixed[active, j] <= self.threshold[old], self.left[old], self.right[old])
            result += self.value[node]
        return result if self.classes else result[:, 0]
