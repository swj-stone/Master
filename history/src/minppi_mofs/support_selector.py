"""Scikit-learn transformer for the support-only pivotal ranking."""
from math import ceil

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from minppi_mofs.allocation_vote_minpi import allocation_vote_top_from_training
from minppi_mofs.band_vote_minpi import band_vote_top_from_training
from minppi_mofs.budget_vote_minpi import budget_vote_top_from_training
from minppi_mofs.calibrated_band_minpi import calibrated_band_top_from_training
from minppi_mofs.capped_vote_minpi import capped_vote_top_from_training
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.coalition_minpi import coalition_top_from_training
from minppi_mofs.conditional_minpi import conditional_top_from_votes
from minppi_mofs.environment_minpi import environment_top_from_training
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.grouped_minpi import grouped_top_from_training
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training
from minppi_mofs.loss_minpi import loss_top_from_training
from minppi_mofs.masked_loss_minpi import masked_loss_top_from_training
from minppi_mofs.moment_minpi import moment_top_from_training
from minppi_mofs.pair_vote_minpi import pair_vote_top_from_training
from minppi_mofs.proxy_minpi import proxy_top_from_training
from minppi_mofs.refit_minpi import refit_top_from_training
from minppi_mofs.reliability_document_minpi import reliability_document_top_from_training
from minppi_mofs.restoration_minpi import restoration_top_from_training
from minppi_mofs.ridge_vote_minpi import ridge_vote_top_from_training
from minppi_mofs.robust_document_minpi import robust_document_top_from_training
from minppi_mofs.sketch_minpi import reduced_width, sketch_top_from_training
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training
from minppi_mofs.sparse_exchange_minpi import sparse_exchange_top_from_training
from minppi_mofs.stable_native_minpi import stable_native_top_from_training
from minppi_mofs.tournament_minpi import tournament_top_from_training
from minppi_mofs.utility_minpi import utility_top_from_training
from minppi_mofs.word_minpi import exact_top_from_votes


class PivotalSupportSelector(SelectorMixin, BaseEstimator):
    """Fit a pivotal rule on training rows and return its selected columns in input order.

    With rule="static", arithmetic="float" uses the double-precision scorer.
    Use arithmetic="rational" for the exact finite-game order. The conditional
    rule includes the selected votes in each subsequent majority and scores
    exact increments. The grouped rule screens redundant input columns before
    constructing witnesses and solving the resulting smaller voting game.
    The loss rule votes on whether a one-feature predictor improves loss on
    held-out training rows. The masked_loss rule measures the loss change by
    masking a coordinate of a joint predictor.

    The utility and coalition rules use signed squared-loss gains from a
    cross-fitted joint ridge and take the minimum over budget-reachable peer
    counts. The coalition rule updates gains after each selection. The refit
    rule refits ridge coefficients on nested selected supports and ranks by
    the worst held-row gain along that path. The local_refit rule compares
    only the current support and the same support before its last addition,
    incrementally updating the ridge system. The loss-gain rules evaluate a
    predictive game in floating point regardless of the static arithmetic.
    For either refit rule, explicit tiers are the block checkpoints, must
    include 0, and must be below the selected width.
    The restoration rule measures the gain from restoring a permuted column
    at each requested count of other corrupted columns. It accepts training
    groups in fit, with ordered_groups=True when larger group IDs are later.
    The environment rule ranks a fixed budget by the weakest normalized gain
    across held environments and two adjacent prefixes. For automatic width
    calibration use minppi_mofs.stable_minpi.StableMinPISelector.

    The moment and pair_vote rules score a column both alone and in the
    presence of its paired column. The sketch rule fits coefficients for the
    requested width under uniformly masked inputs. These three rules sample
    the supplied training rows, accept groups and ordered_groups, fix their
    peer counts from the budget, and drop at least one column whenever p > 1.
    Use fraction=.5 for the half-width moment experiment or fraction=.75 to
    reproduce the three-quarter-width comparisons.

    The tournament, pilot, native, signal_vote and exchange_vote rules accept fraction="auto" to
    choose the width from training data or a numeric fraction for a fixed
    budget. Pilot searches for correlation-confirmed proxy groups. Tournament
    compares paired columns. Exchange_vote audits a small number of proposed
    swaps and certifies training-constant columns when sufficiently prevalent.
    Exchange_vote also accepts SciPy sparse feature matrices. Native computes
    moment votes with the optional prebuilt kernel and rechecks ambiguous
    selection boundaries with the NumPy reference.
    Signal_vote measures a column's held-row loss gain both alone and with
    a strong peer chosen on fitting rows. Its automatic budget retains 99%
    of positive worst-context vote mass and keeps all columns if none is positive.

    Experimental adaptive rules include budget_vote (paired votes with a
    width floor based on their effective dimension), ridge_vote (joint ridge
    with held-loss prefix calibration), and capped_vote (cross-block peer
    votes with a half-width cap). Band_vote, calibrated_band and allocation_vote
    require neighboring columns to follow spectral acquisition order.
    Document_vote requires sparse nonnegative term counts and classification
    labels. It scores term presence with zero and one strong peer, using
    the prebuilt binary kernel when available and NumPy otherwise. Transform
    preserves the original values of the retained columns for all rules.
    Robust_vote repeats the document game on three training sketches and
    takes the median within each peer-count stratum before their minimum.
    It uses the same automatic vote-mass budget and sparse count inputs.
    Reliability_vote uses one sketch and takes the minimum over its two
    cross-fit directions within each peer-count stratum.
    """

    tiers = None  # Estimators saved before configurable peer counts used the default.
    rule = "static"  # Preserve the behaviour of estimators saved before rule selection.

    def __init__(self, *, fraction=.1, task="classification", random_state=42, arithmetic="float",
                 tiers=None, rule="static"):
        self.fraction = fraction
        self.task = task
        self.random_state = random_state
        self.arithmetic = arithmetic
        self.tiers = tiers
        self.rule = rule

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        if self.rule in ("exchange_vote", "document_vote", "robust_vote", "reliability_vote"):
            tags.input_tags.sparse = True
        if self.rule in ("document_vote", "robust_vote", "reliability_vote"):
            tags.input_tags.allow_nan = False
        return tags

    def fit(self, X, y, *, groups=None, ordered_groups=False):
        adaptive_rules = {"exchange_vote": sparse_exchange_top_from_training,
                          "document_vote": fast_document_vote_top_from_training,
                          "robust_vote": robust_document_top_from_training,
                          "reliability_vote": reliability_document_top_from_training,
                          "allocation_vote": allocation_vote_top_from_training,
                          "band_vote": band_vote_top_from_training,
                          "calibrated_band": calibrated_band_top_from_training,
                          "capped_vote": capped_vote_top_from_training,
                          "budget_vote": budget_vote_top_from_training,
                          "ridge_vote": ridge_vote_top_from_training,
                          "signal_vote": signal_vote_top_from_training,
                          "tournament": tournament_top_from_training, "pilot": proxy_top_from_training,
                          "native": stable_native_top_from_training}
        auto = isinstance(self.fraction, str) and self.fraction == "auto"
        if auto and self.rule not in adaptive_rules:
            raise ValueError(f"Automatic width requires one of {tuple(adaptive_rules)}")
        if not auto and not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0,1]")
        if self.arithmetic not in ("float", "rational"):
            raise ValueError("arithmetic must be 'float' or 'rational'")
        training_rules = {"grouped": grouped_top_from_training, "loss": loss_top_from_training,
                          "masked_loss": masked_loss_top_from_training, "utility": utility_top_from_training,
                          "coalition": coalition_top_from_training, "refit": refit_top_from_training,
                          "local_refit": local_refit_fast_top_from_training,
                          "environment": environment_top_from_training,
                          "restoration": restoration_top_from_training}
        budget_rules = {"moment": moment_top_from_training, "pair_vote": pair_vote_top_from_training,
                        "sketch": sketch_top_from_training, **adaptive_rules}
        rules = ("static", "conditional", *training_rules, *budget_rules)
        if self.rule not in rules:
            raise ValueError(f"Unknown pivotal rule. Choose from {rules}")
        grouped_rules = ("environment", "restoration")
        if self.rule not in (*grouped_rules, *budget_rules) and (groups is not None or ordered_groups):
            raise ValueError("Training groups require a rule that supports grouped sampling")
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("Feature names must be unique")
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite=True if self.rule in ("document_vote", "robust_vote", "reliability_vote") else "allow-nan",
                          accept_sparse="csr" if self.rule in ("exchange_vote", "document_vote", "robust_vote", "reliability_vote") else False)
        count = None if auto else ceil(self.fraction*self.n_features_in_)
        if self.rule in budget_rules:
            if self.tiers is not None:
                raise ValueError("Peer counts for this rule are fixed by the selection budget")
            if not auto:
                count = reduced_width(self.n_features_in_, self.fraction)
            support, self.diagnostics_ = budget_rules[self.rule](
                X, y, count, task=self.task, random_state=self.random_state,
                groups=groups, ordered_groups=ordered_groups)
        elif self.rule in training_rules:
            fit_params = {"groups": groups, "ordered_groups": ordered_groups} if self.rule in grouped_rules else {}
            support, self.diagnostics_ = training_rules[self.rule](
                X, y, count, task=self.task, random_state=self.random_state, tiers=self.tiers, **fit_params)
        else:
            votes = np.sign(witness_margins(X, y, self.task, self.random_state)).astype(np.int8)
            if self.rule == "conditional":
                scorer = conditional_top_from_votes
            else:
                scorer = exact_top_from_votes if self.arithmetic == "rational" else top_from_votes
            support, self.diagnostics_ = scorer(votes, count, tiers=self.tiers)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[support] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
