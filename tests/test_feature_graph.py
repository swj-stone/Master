"""
Correctness tests for the correlation-based feature graph.

The graph built by ``build_feature_graph`` / ``_create_minimal_edge_graph``
is a spanning tree over the feature correlation matrix.  These tests
verify structural invariants and determinism.

Run::

    python -m pytest tests/test_feature_graph.py -v
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from minppi_mofs.baselines.shapg import (
    _corr_matrix,
    _create_minimal_edge_graph,
    build_feature_graph,
)


# ═══════════════════════════════════════════════════════════════════════════
# Structural invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphSpanningTreeInvariants:

    def test_graph_is_connected(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(12)})
        G = build_feature_graph(X, corr_method="pearson")
        assert nx.is_connected(G), "Graph must be connected"

    def test_edge_count_equals_n_minus_1(self) -> None:
        """A strict spanning tree on n nodes has exactly n-1 edges."""
        rng = np.random.default_rng(43)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(8)})
        G = build_feature_graph(X, corr_method="pearson")
        n = X.shape[1]
        assert G.number_of_edges() == n - 1, (
            f"Graph has {G.number_of_edges()} edges with {n} nodes; expected {n - 1}"
        )

    def test_no_isolated_nodes(self) -> None:
        rng = np.random.default_rng(44)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 150) for i in range(20)})
        G = build_feature_graph(X, corr_method="pearson")
        for node in G.nodes():
            assert G.degree(node) >= 1, f"Node {node} is isolated"

    def test_single_node_graph_has_zero_edges(self) -> None:
        rng = np.random.default_rng(45)
        X = pd.DataFrame({"f0": rng.normal(0, 1, 100)})
        G = build_feature_graph(X)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_two_node_graph_has_one_edge(self) -> None:
        rng = np.random.default_rng(46)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
        })
        G = build_feature_graph(X)
        assert G.number_of_edges() == 1


# ═══════════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphDeterminism:

    def test_same_input_produces_same_graph(self) -> None:
        rng = np.random.default_rng(47)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(10)})
        G1 = build_feature_graph(X, corr_method="pearson")
        G2 = build_feature_graph(X, corr_method="pearson")
        assert set(G1.edges()) == set(G2.edges()), "Same data → same edges"

    def test_graph_nodes_match_feature_names(self) -> None:
        rng = np.random.default_rng(48)
        cols = ["alpha", "beta", "gamma", "delta"]
        X = pd.DataFrame({c: rng.normal(0, 1, 80) for c in cols})
        G = build_feature_graph(X)
        assert set(G.nodes()) == set(cols)


class TestGraphWeights:
    """Accepted MST edges retain their absolute correlation strength."""

    def test_networkx_weights_match_reduced_matrix(self) -> None:
        rng = np.random.default_rng(52)
        n = 160
        base = rng.normal(0, 1, n)
        X = pd.DataFrame({
            "strong": base,
            "strong_copy": 0.9 * base + rng.normal(0, 0.1, n),
            "medium": 0.5 * base + rng.normal(0, 0.7, n),
            "noise": rng.normal(0, 1, n),
        })
        W = _corr_matrix(X)
        adj, red = _create_minimal_edge_graph(W)
        G = build_feature_graph(X)

        expected_edges = {
            frozenset((left, right))
            for left in X.columns
            for right in X.columns
            if left < right and adj.loc[left, right] == 1
        }
        actual_edges = {frozenset((left, right)) for left, right in G.edges()}
        assert actual_edges == expected_edges
        for left, right, data in G.edges(data=True):
            assert data["weight"] == pytest.approx(red.loc[left, right])
        assert any(data["weight"] != 1.0 for _, _, data in G.edges(data=True))

    def test_zero_weight_tree_edges_are_preserved(self) -> None:
        X = pd.DataFrame({
            "a": np.ones(20),
            "b": np.ones(20),
            "c": np.ones(20),
        })

        G = build_feature_graph(X)

        assert nx.is_tree(G)
        assert G.number_of_edges() == 2
        assert all(data["weight"] == 0.0 for _, _, data in G.edges(data=True))


# ═══════════════════════════════════════════════════════════════════════════
# Edge semantics — strong correlations appear as edges
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeSemantics:

    def test_perfectly_correlated_pair_is_connected(self) -> None:
        rng = np.random.default_rng(49)
        n = 100
        x = rng.normal(0, 1, n)
        X = pd.DataFrame({
            "f0": x,
            "f0_copy": x,          # ρ = 1.0
            "noise": rng.normal(0, 1, n),
        })
        G = build_feature_graph(X, corr_method="pearson")
        assert G.has_edge("f0", "f0_copy") or (
            "f0" in nx.neighbors(G, "f0_copy")
        ), "Perfectly correlated pair must be connected"


# ═══════════════════════════════════════════════════════════════════════════
# Minimal-edge graph: only union-successful edges are in adjacency
# ═══════════════════════════════════════════════════════════════════════════


class TestMinimalEdgeGraphAdjacency:

    def test_adjacency_only_has_edges_from_union(self) -> None:
        """Every 1-entry in the adjacency matrix must correspond to an edge
        whose Union-Find union *might* have been successful.  We verify this
        indirectly: the adjacency produces a connected subgraph."""
        rng = np.random.default_rng(50)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(15)})
        W = _corr_matrix(X)
        adj, red = _create_minimal_edge_graph(W)

        # adj is a symmetric (0,1) matrix indexed by feature names
        assert adj.shape == (15, 15)
        assert (adj.values == adj.values.T).all(), "adjacency not symmetric"

        # Build a graph from the adjacency and check it's connected
        G = nx.Graph(adj)
        assert nx.is_connected(G), "adjacency-based graph not connected"
        assert nx.is_tree(G), "minimal-edge adjacency must not contain cycles"
        assert G.number_of_edges() == len(X.columns) - 1

        # Redundancy-weight matrix mirrors adjacency (edges are the same)
        edge_mask = adj.values > 0
        assert (red.values[edge_mask] > 0).all(), (
            "every adjacency edge must have a non-zero weight in red"
        )
        assert (red.values[~edge_mask] == 0).all(), (
            "non-edges must have zero weight in red"
        )

    def test_adjacency_binary_values_only(self) -> None:
        rng = np.random.default_rng(51)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 50) for i in range(10)})
        W = _corr_matrix(X)
        adj, _ = _create_minimal_edge_graph(W)
        unique_vals = set(adj.values.flatten())
        assert unique_vals <= {0, 1}, f"adjacency has non-binary values: {unique_vals}"
