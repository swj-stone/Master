"""Run the document selector on the historical panels with document-term columns."""
import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from experiments import signal_vote_benchmark as prior
from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import summarize_timings
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training


ARMS = ("document_vote", "varying_only", "F_match", "random_match", "F_50", "random_50", "full")


def selected(panel, arm, width):
    if arm == "full":
        return np.arange(panel.X.shape[1])
    # Each arm pays for reading the archived dense input and identifying its
    # varying word-presence columns. Downstream models still get original counts.
    seen = np.count_nonzero(panel.X, axis=0)
    varying = np.flatnonzero((seen > 0) & (seen < len(panel.y)))
    if arm == "varying_only":
        return varying
    if arm == "document_vote":
        X = sparse.csr_matrix(panel.X[:, varying])
        cols, _ = fast_document_vote_top_from_training(X, panel.y, random_state=panel.seed,
                                                      groups=panel.groups)
        return varying[cols]
    rule, budget = arm.split("_")
    count = width if budget == "match" else (len(varying) + 1) // 2
    if rule == "random":
        return varying[np.sort(np.random.default_rng(panel.seed).permutation(len(varying))[:count])]
    return varying[choose("f", panel.X[:, varying], panel.y, "classification", panel.seed, count)]


def cases():
    for suite, dataset in (("original", "cnae"), ("transfer", "dexter")):
        found = 0
        for panel in prior.panels(suite):
            if panel.name != dataset:
                continue
            assert panel.task == "classification" and np.isfinite(panel.X).all() and np.min(panel.X) >= 0
            yield suite, panel
            found += 1
            if found == 3:
                break
        assert found == 3


def run(output, verify):
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
    trials, checks = [], []
    for suite, panel in cases():
        key = prior.case_id(panel)
        path = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        cols, predictions, scores = {}, {}, {}
        width = len(selected(panel, "document_vote", None))
        for arm in ARMS:
            cols[arm] = selected(panel, arm, width)
            pred, decision = prior.fit(panel, cols[arm], suite)
            predictions[arm] = pred
            scores[arm] = prior.score(panel, pred, decision, suite)[0]
            checks.append(dict(**key, method=arm, support_sha256=prior.digest(cols[arm]),
                               prediction_sha256=prior.digest(pred), width=len(cols[arm]), score=scores[arm]))
        with np.load(prior.REFERENCE[suite] / path) as archive:
            np.testing.assert_array_equal(predictions["full"], archive["full_pred"])
        arrays = {**{k + "_cols": v for k, v in cols.items()},
                  **{k + "_pred": v for k, v in predictions.items()}}
        if verify:
            with np.load(output / path) as archive:
                for k, v in arrays.items():
                    np.testing.assert_array_equal(v, archive[k])
            continue
        np.savez_compressed(output / path, **arrays)
        order = np.random.default_rng(panel.seed).permutation(ARMS)
        for r in range(len(ARMS)):
            for position, arm in enumerate(np.roll(order, r)):
                a = perf_counter_ns()
                subset = selected(panel, arm, width)
                b = perf_counter_ns()
                pred, _ = prior.fit(panel, subset, suite)
                c = perf_counter_ns()
                np.testing.assert_array_equal(subset, cols[arm])
                np.testing.assert_array_equal(pred, predictions[arm])
                trials.append(dict(**key, method=arm, repeat=r, position=position, width=len(subset),
                                   p=panel.X.shape[1], score=scores[arm], select_ms=(b-a)/1e6,
                                   refit_ms=(c-b)/1e6, total_ms=(c-a)/1e6))
    if verify:
        pd.DataFrame(checks).to_csv(output / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == 6 * len(ARMS)**2
        summarize_timings(output, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
