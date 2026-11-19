"""Replay the registered 387 input splits with their original estimator and metric."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score

from research.lowdim import extra_study, formal_study, survey_inputs, text_study
from research.lowdim.verify_combined import trace
from research.pair_selection import benchmark as prior

KEY = prior.KEY
INDEX = prior.INDEX
HASHES = ("train_sha", "target_sha", "test_sha", "truth_sha")
bench = prior.bench


@dataclass
class Case:
    key: tuple
    X: object
    y: np.ndarray
    Xt: object
    truth: np.ndarray
    task: str
    metric: str
    archive: Path
    impute: str
    predict: Callable
    score: Callable

    @property
    def seed(self):
        return self.key[-1]

    def selection_X(self):
        if self.impute == "training_means":
            return bench._impute_validation(self.X, self.X[:0])[0]
        if self.impute == "breadth_means":
            valid = np.isfinite(self.X)
            means = np.where(valid, self.X, 0.).sum(axis=0) / np.maximum(1, valid.sum(axis=0))
            return np.where(valid, self.X, means)
        return self.X

    def select_old(self):
        """The selector denoted new by the frozen 387-case lowdim archive."""
        if sparse.issparse(self.X):
            return text_study.choose("new", self.X, self.y, self.seed, None)
        return prior.method.fast_support(self.selection_X(), self.y, task=self.task, seed=self.seed)[0]

    def select_F(self, width):
        if sparse.issparse(self.X):
            return text_study.choose("F", self.X, self.y, self.seed, width)
        return prior.f_support(self.selection_X(), self.y, self.task, width)

    def select_random(self, width):
        return np.sort(np.random.default_rng(self.seed).permutation(self.X.shape[1])[:width])


def registry():
    rows = pd.read_csv(INDEX, keep_default_na=False).set_index(KEY, drop=False)
    if not rows.index.is_unique or len(rows) != 387:
        raise ValueError("Historical 387-case registry differs")
    return rows


def input_hashes(case):
    matrix_hash = text_study.sparse_digest if sparse.issparse(case.X) else bench.digest
    return (matrix_hash(case.X), bench.digest(case.y), matrix_hash(case.Xt), bench.digest(case.truth))


def standard(row, X, y, Xt, truth, task, impute, metric):
    return Case(row.name, X, y, Xt, truth, task, row.metric, trace(row),
                "training_means" if impute else "",
                partial(prior.restricted_train, X, y, Xt, task, row.seed, impute),
                partial(prior.standard_score, truth, metric))


def breadth_predict(cohort, data, cols):
    if cohort == "dense":
        return extra_study.acoustic_fit(data, cols, "cholesky")[:2]
    if cohort == "four_stage_ridge":
        return prior.four_predict(data, cols)
    if cohort == "spectra":
        return extra_study.prediction(data.X, data.y, data.Xt, cols, data.task, data.seed), None
    if cohort == "mechanism":
        return bench.fit(data, cols)
    raise ValueError(cohort)


def sparse_predict(X, y, Xt, seed, cols):
    return text_study.farm.fit_predict(X, y, Xt, cols, "classification", seed, False), None


def sparse_score(truth, pred, decision):
    return balanced_accuracy_score(truth, pred)


def load_cases(rows, wanted):
    cohorts = {key[0] for key in wanted}
    if cohorts & {"original", "extra"}:
        for cohort, d, truth, units, _ in bench.panels():
            key = (cohort, d.name, d.fold, d.seed)
            if key in wanted:
                yield Case(key, d.X, d.y, d.Xt, truth, d.task, rows.at[key, "metric"],
                           trace(rows.loc[key]), "training_means" if d.name in bench.deployment.ADDITIONAL else "",
                           partial(bench.fit, d), partial(prior.core_score, d, truth, units))

    for name in sorted({key[1] for key in wanted if key[0] == "formal"}):
        raw, y, task, _, order, _, _ = formal_study.load_dataset(name)
        for fold, seed in enumerate(formal_study.P["formal_seeds"], 1):
            key = ("formal", name, fold, seed)
            if key in wanted:
                X, b, Xt, truth, *_ = formal_study.inputs(name, seed, raw, y, order, task)
                yield standard(rows.loc[key], X, b, Xt, truth, task, False,
                               "r2" if task == "regression" else "auc")

    for cohort in extra_study.SIZES:
        if cohort not in cohorts:
            continue
        for d, _, _ in extra_study.cases(cohort):
            key = (cohort, d.name, d.fold, d.seed)
            if key in wanted:
                yield Case(key, d.X, d.y, d.Xt, d.gold, d.task, rows.at[key, "metric"], trace(rows.loc[key]),
                           "" if np.isfinite(d.X).all() else "breadth_means",
                           partial(breadth_predict, cohort, d), partial(prior.four_score, d))

    for cohort, loader in (("archive_supplement", survey_inputs.study_cases),
                           ("conference_fraud", survey_inputs.fraud_cases)):
        if cohort not in cohorts:
            continue
        for d in loader():
            key = (cohort, d.dataset, d.fold, d.seed)
            if key in wanted:
                yield standard(rows.loc[key], d.X, d.y, d.Xt, d.truth, d.task, d.impute, d.metric)

    if "text" in cohorts:
        for X, y, Xt, truth, meta, _ in text_study.cases():
            key = ("text", meta["dataset"], meta["fold"], meta["seed"])
            if key in wanted:
                yield Case(key, X, y, Xt, truth, "classification", rows.at[key, "metric"],
                           trace(rows.loc[key]), "", partial(sparse_predict, X, y, Xt, key[-1]),
                           partial(sparse_score, truth))


def cases(wanted_keys=None):
    rows = registry()
    wanted = set(rows.index) if wanted_keys is None else set(wanted_keys)
    unknown = wanted.difference(rows.index)
    if unknown:
        raise ValueError(f"Unregistered inputs: {unknown}")
    seen = set()
    for case in load_cases(rows, wanted):
        row = rows.loc[case.key]
        if case.key in seen or not case.archive.is_file():
            raise ValueError(f"Repeated or unarchived case {case.key}")
        if case.X.shape != (int(row.n_train), row.p) or case.Xt.shape != (int(row.n_test), row.p):
            raise ValueError(f"Archived shape mismatch {case.key}")
        if input_hashes(case) != tuple(row[list(HASHES)]):
            raise ValueError(f"Archived input fingerprints differ on {case.key}")
        seen.add(case.key)
        yield case
    if seen != wanted:
        raise ValueError(f"Missing cases: {wanted-seen}")
