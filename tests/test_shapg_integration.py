"""
Tests for MI-correlation guidance and its deprecated compatibility aliases.

Covers:
- compute_mi_guidance_signal returns valid scores for all features
- get_correlation_neighbors preserves weighted-tree ordering
- build_mi_correlation_guidance returns both guidance components
- filter_top_by_mi count correctness
- rank_by_mi ordering
- legacy ShapG-labelled aliases emit deprecation warnings

Run::

    pytest tests/test_shapg_integration.py -v
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.shapg_integration import (
    build_guidance_from_data,
    build_mi_correlation_guidance,
    compute_mi_guidance_signal,
    filter_top_by_shapg,
    filter_top_by_mi,
    get_correlation_neighbors,
    get_correlated_neighbors,
    rank_by_shapg,
    rank_by_mi,
)
from minppi_mofs.baselines.shapg import build_feature_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic 10-feature dataset with known structure."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(10)})
    # Make f0 and f1 correlated (should show up as neighbours)
    X["f1"] = X["f0"] * 0.7 + rng.normal(0, 0.3, n)
    y = np.where(X["f0"] + 0.2 * X["f1"] > 0, 1, 0)
    return X, y


# ---------------------------------------------------------------------------
# compute_mi_guidance_signal
# ---------------------------------------------------------------------------


class TestComputeMiGuidanceSignal:
    """Mutual-information guidance score computation."""

    def test_returns_dict_with_all_features(self, toy_data) -> None:
        X, y = toy_data
        scores = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        assert isinstance(scores, dict)
        assert len(scores) == 10
        for col in X.columns:
            assert col in scores

    def test_scores_are_numeric(self, toy_data) -> None:
        X, y = toy_data
        scores = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        for val in scores.values():
            assert isinstance(val, (int, float))
            assert not np.isnan(val)

    def test_descending_order(self, toy_data) -> None:
        X, y = toy_data
        scores = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        items = list(scores.items())
        for i in range(len(items) - 1):
            assert items[i][1] >= items[i + 1][1], (
                f"Not descending: {items[i]} < {items[i + 1]}"
            )

    def test_seed_reproducibility(self, toy_data) -> None:
        X, y = toy_data
        s1 = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        s2 = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        for f in X.columns:
            assert s1[f] == pytest.approx(s2[f])

    def test_different_seed_gives_different(self, toy_data) -> None:
        """Different seeds may produce different scores (but similar ranking)."""
        X, y = toy_data
        s1 = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        s2 = compute_mi_guidance_signal(X, y, task="binary", seed=123)
        # At least the top feature should be among top-3 in both
        top_f1 = next(iter(s1))
        top3_f2 = list(s2.keys())[:3]
        assert top_f1 in top3_f2 or list(s1.keys())[0] in top3_f2


class TestExplicitTaskRouting:
    """MI estimator selection is controlled only by the explicit task."""

    def test_26_class_target_uses_classification_mi(self) -> None:
        rng = np.random.default_rng(60)
        X = pd.DataFrame(rng.normal(size=(52, 3)), columns=["a", "b", "c"])
        y = np.tile(np.arange(26), 2)

        with (
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_classif",
                return_value=np.array([0.3, 0.2, 0.1]),
            ) as classif,
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_regression",
            ) as regression,
        ):
            scores = compute_mi_guidance_signal(
                X, y, task="multiclass", seed=42,
            )

        classif.assert_called_once()
        regression.assert_not_called()
        assert list(scores) == ["a", "b", "c"]

    def test_regression_task_uses_regression_mi(self) -> None:
        rng = np.random.default_rng(61)
        X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
        y = np.linspace(0.0, 1.0, len(X))

        with (
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_classif",
            ) as classif,
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_regression",
                return_value=np.array([0.4, 0.1]),
            ) as regression,
        ):
            compute_mi_guidance_signal(X, y, task="regression", seed=42)

        regression.assert_called_once()
        classif.assert_not_called()

    def test_invalid_task_is_rejected_before_mi_call(self) -> None:
        rng = np.random.default_rng(62)
        X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["a", "b"])
        y = rng.integers(0, 2, len(X))

        with (
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_classif",
            ) as classif,
            mock.patch(
                "minppi_mofs.shapg_integration.mutual_info_regression",
            ) as regression,
            pytest.raises(ValueError, match="task"),
        ):
            compute_mi_guidance_signal(X, y, task="ranking", seed=42)

        classif.assert_not_called()
        regression.assert_not_called()


# ---------------------------------------------------------------------------
# get_correlated_neighbors
# ---------------------------------------------------------------------------


class TestGetCorrelationNeighbors:
    """Correlation-graph neighbour extraction."""

    def test_returns_dict_with_all_features(self, toy_data) -> None:
        X, _ = toy_data
        neighbors = get_correlation_neighbors(X, top_k=5, seed=42)
        assert len(neighbors) == 10
        for col in X.columns:
            assert col in neighbors

    def test_neighbors_are_lists(self, toy_data) -> None:
        X, _ = toy_data
        neighbors = get_correlation_neighbors(X, top_k=5, seed=42)
        for nbrs in neighbors.values():
            assert isinstance(nbrs, list)

    def test_no_self_neighbors(self, toy_data) -> None:
        X, _ = toy_data
        neighbors = get_correlation_neighbors(X, top_k=5, seed=42)
        for feat, nbrs in neighbors.items():
            assert feat not in nbrs, f"{feat} appears in its own neighbour list"

    def test_top_k_respected(self, toy_data) -> None:
        X, _ = toy_data
        for k in [2, 5, 10]:
            neighbors = get_correlation_neighbors(X, top_k=k, seed=42)
            for nbrs in neighbors.values():
                assert len(nbrs) <= k, f"Neighbour list exceeds top_k={k}"

    def test_correlated_features_are_neighbors(self, toy_data) -> None:
        """f0 and f1 are correlated → should be mutual neighbours."""
        X, _ = toy_data
        neighbors = get_correlation_neighbors(X, top_k=5, seed=42)
        # f0 and f1 are linearly correlated (f1 = 0.7*f0 + noise)
        assert "f1" in neighbors["f0"] or "f0" in neighbors["f1"], (
            "Correlated features f0/f1 should be neighbours"
        )

    def test_different_corr_methods(self, toy_data) -> None:
        X, _ = toy_data
        for method in ["pearson", "spearman"]:
            neighbors = get_correlation_neighbors(
                X, top_k=5, corr_method=method, seed=42
            )
            assert len(neighbors) == 10

    def test_canonical_neighbors_follow_descending_edge_weight(self, toy_data) -> None:
        X, _ = toy_data
        neighbors = get_correlation_neighbors(X, top_k=10, seed=42)
        graph = build_feature_graph(X, corr_method="pearson", seed=42)

        for feature in X.columns:
            expected = sorted(
                graph.neighbors(feature),
                key=lambda neighbor: graph.edges[feature, neighbor]["weight"],
                reverse=True,
            )
            assert neighbors[feature] == expected


# ---------------------------------------------------------------------------
# build_guidance_from_data
# ---------------------------------------------------------------------------


class TestBuildMiCorrelationGuidance:
    """Combined convenience wrapper."""

    def test_returns_both(self, toy_data) -> None:
        X, y = toy_data
        scores, neighbors = build_mi_correlation_guidance(
            X, y, task="binary", top_k=5, seed=42
        )
        assert isinstance(scores, dict)
        assert isinstance(neighbors, dict)
        assert len(scores) == 10
        assert len(neighbors) == 10

    def test_feature_sets_match(self, toy_data) -> None:
        X, y = toy_data
        scores, neighbors = build_mi_correlation_guidance(
            X, y, task="binary", top_k=5, seed=42
        )
        assert set(scores.keys()) == set(neighbors.keys()) == set(X.columns)


# ---------------------------------------------------------------------------
# rank_by_shapg
# ---------------------------------------------------------------------------


class TestRankByMi:
    """Utility: rank features by MI score."""

    def test_descending_order(self, toy_data) -> None:
        X, y = toy_data
        scores = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        ranked = rank_by_mi(scores)
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]

    def test_returns_tuples(self, toy_data) -> None:
        X, y = toy_data
        scores = compute_mi_guidance_signal(X, y, task="binary", seed=42)
        ranked = rank_by_mi(scores)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked)


# ---------------------------------------------------------------------------
# filter_top_by_shapg
# ---------------------------------------------------------------------------


class TestFilterTopByMi:
    """Top-k filter utility."""

    def test_returns_correct_count(self, toy_data) -> None:
        X, y = toy_data
        for k in [1, 3, 7, 10]:
            top = filter_top_by_mi(
                X, y, task="binary", top_k=k, seed=42,
            )
            assert len(top) == k

    def test_returns_feature_names(self, toy_data) -> None:
        X, y = toy_data
        top = filter_top_by_mi(X, y, task="binary", top_k=5, seed=42)
        assert all(f in X.columns for f in top)

    def test_top_k_exceeds_n_features(self, toy_data) -> None:
        """If top_k > n_features, return all features."""
        X, y = toy_data
        top = filter_top_by_mi(X, y, task="binary", top_k=100, seed=42)
        assert len(top) == 10  # only 10 features exist


class TestLegacyGuidanceAliases:
    """Old ShapG-labelled helper names remain warning compatibility shims."""

    def test_legacy_aliases_warn(self, toy_data) -> None:
        X, y = toy_data

        with pytest.warns(DeprecationWarning):
            get_correlated_neighbors(X, top_k=3, seed=42)
        with pytest.warns(DeprecationWarning):
            build_guidance_from_data(X, y, task="binary", top_k=3, seed=42)
        with pytest.warns(DeprecationWarning):
            rank_by_shapg({"a": 1.0})
        with pytest.warns(DeprecationWarning):
            filter_top_by_shapg(X, y, task="binary", top_k=3, seed=42)
