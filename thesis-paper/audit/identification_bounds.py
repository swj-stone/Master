"""Enumerate finite couplings and check both metrics against a fitted scikit-learn stump."""

from collections import defaultdict
from itertools import product
from math import isclose

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def fit_stumps():
    labels = np.repeat([0, 1], 4)
    retained = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    training = np.column_stack((labels, retained))
    full = DecisionTreeClassifier(max_depth=1, random_state=0).fit(training, labels)
    reduced = DecisionTreeClassifier(max_depth=1, random_state=0).fit(training[:, 1:], labels)
    assert full.tree_.feature[0] == 0
    assert np.array_equal(reduced.predict_proba([[0], [1]])[:, 1], [0.25, 0.75])
    return full, reduced, np.median(training, axis=0)


def check():
    full, reduced, background = fit_stumps()
    assignments = 0
    tables = 0
    for n in range(2, 9):
        vectors = np.array(list(product((0, 1), repeat=n)))
        reduced_scores = reduced.predict_proba(vectors.reshape(-1, 1))[:, 1].reshape(vectors.shape)
        reduced_hard = reduced.predict(vectors.reshape(-1, 1)).reshape(vectors.shape)
        for positive_count in range(1, n):
            labels = np.arange(n) < positive_count
            p = positive_count / n
            # Check the full transcript for both the perfect and the constant anchor.
            for deleted in (labels, np.zeros(n)):
                deleted_matrix = np.broadcast_to(deleted, vectors.shape)
                for show_d, show_r in product((False, True), repeat=2):
                    inputs = np.column_stack((deleted_matrix.ravel(), vectors.ravel()))
                    inputs = np.where((show_d, show_r), inputs, background)
                    frozen = full.predict_proba(inputs)[:, 1].reshape(vectors.shape)
                    assert np.all(frozen == frozen[0])
                    if not np.any(deleted):
                        assert np.all(frozen == 0)

            representatives = defaultdict(dict)
            for row, retained in enumerate(vectors):
                high_score_count = retained.sum()
                overlap = retained[labels].sum()
                representatives[high_score_count][overlap] = row
            for high_score_count, overlaps in representatives.items():
                low = max(0, positive_count + high_score_count - n)
                high = min(positive_count, high_score_count)
                assert set(overlaps) == set(range(low, high + 1))
                q = high_score_count / n
                for overlap, row in overlaps.items():
                    u = overlap / n
                    auc = roc_auc_score(labels, reduced_scores[row])
                    accuracy = accuracy_score(labels, reduced_hard[row])
                    assert isclose(auc, 0.5 + (u - p*q) / (2*p*(1-p)), abs_tol=1e-12)
                    assert isclose(accuracy, 1 - p - q + 2*u, abs_tol=1e-12)
                    tables += 1
            assignments += len(vectors)
    print(f"Checked {assignments} assignments, {tables} attainable tables and both mask transcripts.")


if __name__ == "__main__":
    check()
