from math import ceil

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import load_iris

from minppi_mofs.conditional_minpi import conditional_top_from_votes
from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training
from minppi_mofs.coalition_minpi import coalition_top_from_training
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.grouped_minpi import grouped_top_from_training
from minppi_mofs.local_refit_minpi import local_refit_top_from_training
from minppi_mofs.loss_minpi import loss_top_from_training
from minppi_mofs.masked_loss_minpi import masked_loss_top_from_training
from minppi_mofs.moment_minpi import moment_top_from_training
from minppi_mofs.pair_vote_minpi import pair_vote_top_from_training
from minppi_mofs.proxy_minpi import proxy_top_from_training
from minppi_mofs.refit_minpi import refit_top_from_training
from minppi_mofs.restoration_minpi import restoration_top_from_training
from minppi_mofs.sketch_minpi import sketch_top_from_training
from minppi_mofs.support_selector import PivotalSupportSelector
from minppi_mofs.stable_native_minpi import stable_native_top_from_training
from minppi_mofs.tournament_minpi import tournament_top_from_training
from minppi_mofs.utility_minpi import utility_top_from_training


@pytest.mark.parametrize("rule", [
    "conditional", "grouped", "loss", "masked_loss", "utility", "coalition", "refit", "local_refit", "restoration"])
def test_configured_transformer_matches_core_and_transforms_new_rows(rule):
    X, y = load_iris(return_X_y=True)
    estimator = clone(PivotalSupportSelector(rule=rule, random_state=1129, fraction=.5,
                                             arithmetic="rational", tiers=(0, 1)))
    train = np.arange(len(y)) % 4 != 0
    estimator.fit(X[train], y[train])
    width = ceil(.5 * X.shape[1])
    if rule == "conditional":
        votes = np.sign(witness_margins(X[train], y[train], "classification", 1129))
        columns, _ = conditional_top_from_votes(votes, width, tiers=(0, 1))
    else:
        select = {"grouped": grouped_top_from_training, "loss": loss_top_from_training,
                  "masked_loss": masked_loss_top_from_training, "utility": utility_top_from_training,
                  "coalition": coalition_top_from_training, "refit": refit_top_from_training,
                  "local_refit": local_refit_top_from_training,
                  "restoration": restoration_top_from_training}[rule]
        columns, _ = select(X[train], y[train], width, random_state=1129, tiers=(0, 1))
    np.testing.assert_array_equal(estimator.get_support(indices=True), columns)
    np.testing.assert_array_equal(estimator.transform(X[~train]), X[~train][:, columns])


def test_restoration_receives_only_the_training_batch_metadata():
    X, y = load_iris(return_X_y=True)
    batch = np.arange(len(X)) % 4
    train = batch < 3
    selector = PivotalSupportSelector(rule="restoration", fraction=.5)
    Xt = selector.fit_transform(X[train], y[train], groups=batch[train], ordered_groups=True)
    info = selector.diagnostics_
    assert (batch[train][info["fit_rows"]] < 2).all()
    assert (batch[train][info["assessment_rows"]] == 2).all()
    np.testing.assert_array_equal(Xt, X[train][:, selector.get_support(indices=True)])


def test_older_estimator_with_no_rule_picks_legacy_support():
    X, y = load_iris(return_X_y=True)
    old = PivotalSupportSelector(random_state=31)
    del old.rule
    assert old.get_params()["rule"] == "static"
    expected = PivotalSupportSelector(rule="static", random_state=31).fit(X, y)
    np.testing.assert_array_equal(old.fit(X, y).get_support(), expected.get_support())


@pytest.mark.parametrize("rule,func", [
    ("moment", moment_top_from_training), ("pair_vote", pair_vote_top_from_training),
    ("sketch", sketch_top_from_training)])
def test_budget_transformer_uses_grouped_core_and_compresses_rounded_full_width(rule, func):
    X, y = load_iris(return_X_y=True)
    groups = np.arange(len(X)) % 5
    tr = groups < 4
    model = clone(PivotalSupportSelector(rule=rule, fraction=.99, random_state=72))
    model.fit(X[tr], y[tr], groups=groups[tr], ordered_groups=True)
    cols, _ = func(X[tr], y[tr], 3, random_state=72, groups=groups[tr], ordered_groups=True)
    np.testing.assert_array_equal(model.get_support(indices=True), cols)
    np.testing.assert_array_equal(model.transform(X[~tr]), X[~tr][:, cols])
    with pytest.raises(ValueError, match="Peer counts"):
        model.set_params(tiers=(0,)).fit(X[tr], y[tr])


@pytest.mark.parametrize("rule,func", [
    ("pilot", proxy_top_from_training), ("tournament", tournament_top_from_training),
    ("exchange_vote", exchange_vote_top_from_training), ("native", stable_native_top_from_training)])
def test_training_width_transformer_matches_algorithm(rule, func):
    X, y = load_iris(return_X_y=True)
    model = clone(PivotalSupportSelector(rule=rule, fraction="auto", random_state=17))
    expected, _ = func(X[:110], y[:110], random_state=17)
    model.fit(X[:110], y[:110])
    np.testing.assert_array_equal(model.transform(X[110:]), X[110:, expected])
    with pytest.raises(ValueError, match="Automatic width"):
        model.set_params(rule="moment").fit(X, y)


def test_sparse_exchange_pipeline_preserves_sparse_shape_on_unseen_rows():
    X, y = load_iris(return_X_y=True)
    X = sparse.csr_matrix(np.column_stack((X, np.zeros((len(X), 14)))))
    model = PivotalSupportSelector(rule="exchange_vote", fraction="auto").fit(X[:120], y[:120])
    np.testing.assert_array_equal(model.get_support(indices=True), np.arange(4))
    transformed = model.transform(X[120:])
    assert sparse.issparse(transformed)
    np.testing.assert_array_equal(transformed.toarray(), X[120:, :4].toarray())
