import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

from minppi_mofs import reliability_document_minpi, robust_document_minpi
from minppi_mofs.support_selector import PivotalSupportSelector


@pytest.mark.parametrize("rule", [
    "budget_vote", "ridge_vote", "capped_vote", "band_vote",
    "calibrated_band", "allocation_vote", "document_vote", "robust_vote", "reliability_vote",
])
@pytest.mark.parametrize("fraction", ["auto", .375])
def test_public_minpi_rules_transform_unseen_rows_and_replay(rule, fraction):
    rng = np.random.default_rng(789)
    X = rng.normal(size=(180, 24))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    if rule in ("document_vote", "robust_vote", "reliability_vote"):
        X = sparse.csr_matrix(np.where(X > .3, 2, 0))
    training, held = X[:140], X[140:]
    selector = PivotalSupportSelector(rule=rule, fraction=fraction, random_state=517)
    selector.fit(training, y[:140])
    chosen = selector.get_support(indices=True)
    assert len(chosen) == (9 if isinstance(fraction, float) else selector.diagnostics_["budget"])
    transformed = selector.transform(held)
    if rule in ("document_vote", "robust_vote", "reliability_vote"):
        assert sparse.issparse(transformed)
        assert (transformed != held[:, chosen]).nnz == 0
    else:
        np.testing.assert_array_equal(transformed, held[:, chosen])
    np.testing.assert_array_equal(clone(selector).fit(training, y[:140]).get_support(indices=True), chosen)


DOCUMENT_REVISIONS = [
    ("robust_vote", robust_document_minpi, "robust_document_top_from_training"),
    ("reliability_vote", reliability_document_minpi, "reliability_document_top_from_training"),
]


@pytest.mark.parametrize("rule,module,function", DOCUMENT_REVISIONS)
@pytest.mark.parametrize("fraction", ["auto", .375])
@pytest.mark.parametrize("classes,ordered", [(2, False), (3, True)])
def test_revised_public_rules_match_core_and_keep_group_isolation(
    rule, module, function, fraction, classes, ordered,
):
    rng = np.random.default_rng(9721)
    X = sparse.csr_matrix(rng.binomial(1, .16, (744, 32)) * rng.integers(1, 6, (744, 32)))
    y = np.arange(X.shape[0]) % classes
    groups = np.arange(X.shape[0]) // 124
    options = dict(random_state=691, groups=groups, ordered_groups=ordered)
    expected, reference = getattr(module, function)(
        X, y, None if fraction == "auto" else 12, **options,
    )
    selector = PivotalSupportSelector(rule=rule, fraction=fraction, random_state=691).fit(
        X, y, groups=groups, ordered_groups=ordered,
    )
    np.testing.assert_array_equal(selector.get_support(indices=True), expected)
    for field in ("scores", "zero_peer_scores", "one_peer_scores"):
        np.testing.assert_array_equal(selector.diagnostics_[field], reference[field])
    for fold in selector.diagnostics_["folds"]:
        fitting, held = groups[fold["fit_rows"]], groups[fold["held_rows"]]
        assert not np.intersect1d(fitting, held).size
        if ordered:
            assert fitting.max() < held.min()
    unseen = X[:9].multiply(3).tocsr()
    assert (selector.transform(unseen) != unseen[:, expected]).nnz == 0


@pytest.mark.parametrize("rule,module,function", DOCUMENT_REVISIONS)
def test_revised_public_rules_work_without_prebuilt_library(monkeypatch, rule, module, function):
    def unavailable():
        raise OSError("Test environment has no prebuilt library")

    monkeypatch.setattr(module, "load_native", unavailable)
    X = sparse.csr_matrix(np.random.default_rng(971).binomial(1, .2, (85, 29)))
    y = np.arange(len(X.indptr)-1) % 2
    expected, reference = getattr(module, function)(X, y, random_state=119, backend="numpy")
    selector = PivotalSupportSelector(rule=rule, fraction="auto", random_state=119).fit(X, y)
    assert selector.diagnostics_["backend"] == "numpy"
    np.testing.assert_array_equal(selector.get_support(indices=True), expected)
    np.testing.assert_array_equal(selector.diagnostics_["scores"], reference["scores"])
