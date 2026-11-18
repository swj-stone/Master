"""Secondary same-width loss-vote ablation with known correlated and independent signals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training


SEEDS = (74113, 74143, 74159, 74161, 74167)
SELECTOR_SEED = 1129
PROXIES = (12, 48)
TASKS = ("classification", "regression")
P = 128
N_TRAIN = 2000
N_TEST = 1200


def generated(task, q, seed):
    rng = np.random.default_rng([20261117, TASKS.index(task), q, seed])
    n = N_TRAIN + N_TEST
    latent = rng.standard_normal(n)
    X = rng.standard_normal((n, P))
    X[:, :q] = np.sqrt(.98) * latent[:, None] + np.sqrt(.02) * X[:, :q]
    y = 1.5 * latent + X[:, q:q+4] @ [.6, .5, .4, .3] + .35 * rng.standard_normal(n)
    if task == "classification":
        y = (y > 0).astype(int)
    order = rng.permutation(P)
    X = np.ascontiguousarray(X[:, order], dtype=np.float32)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:], order


def run_case(task, q, seed):
    X, y, Xt, yt, order = generated(task, q, seed)
    support, info = signal_vote_top_from_training(X, y, task=task, random_state=SELECTOR_SEED)
    k = len(support)
    selections = {
        "vote": support,
        "no_peer": np.sort(np.argsort(-info["zero_peer_scores"], kind="stable")[:k]),
        "F": choose("f", X, y, task, SELECTOR_SEED, k),
        "random": choose("random", X, y, task, SELECTOR_SEED, k),
        "full": np.arange(P),
    }
    rows = []
    for method, cols in selections.items():
        assert method == "full" or len(cols) == k
        pred = prediction(X, y, Xt, cols, task, SELECTOR_SEED)
        rows.append(dict(task=task, proxies_available=q, seed=seed, method=method, width=len(cols),
                         p=P, score=metric(yt, pred, task), proxies_kept=int((order[cols] < q).sum()),
                         distinct_kept=int(((order[cols] >= q) & (order[cols] < q+4)).sum()),
                         support_sha256=digest(cols), prediction_sha256=digest(pred)))
    return rows


def main(args):
    protocol = dict(DGP="iid standardized Gaussian features and independent errors; leading q columns "
                        "measure a latent standard normal with pairwise correlation 0.98. Next four are "
                        "independent active predictors. All remaining columns are nonconstant independent noise. "
                        "Shuffle all feature positions. y = 1.5*latent + 0.6*x1 + 0.5*x2 + 0.4*x3 + 0.3*x4 "
                        "+ N(0,0.35^2); dichotomize at zero for classification.",
                    purpose="Prespecified secondary mechanism test. Report the number of distinct signals "
                            "and proxies retained and paired test score differences at the actual vote width. "
                            "No wall-clock measurements or claims about real data.",
                    seeds=list(SEEDS), first_seed_components=[20261117, "task index", "proxy count"],
                    selector_seed=SELECTOR_SEED, proxy_counts=list(PROXIES), features=P,
                    n_train=N_TRAIN, n_test=N_TEST, tasks=list(TASKS), implementation_hashes={
                        str(p.relative_to(ROOT)): file_hash(p) for p in (
                            ROOT/"src/minppi_mofs/signal_vote_minpi.py", Path(__file__),
                            ROOT/"experiments/stable_minpi_stress.py", ROOT/"experiments/compare_predictive_minpi.py")})
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/"protocol.json", protocol)
    else:
        import json
        assert protocol == json.loads((args.output/"protocol.json").read_text())
    with threadpool_limits(1):
        rows = [row for task in TASKS for q in PROXIES for seed in SEEDS for row in run_case(task, q, seed)]
    cases = pd.DataFrame(rows)
    keys = ["task", "proxies_available", "seed"]
    cases = cases.merge(cases[cases.method == "full"][[*keys, "score"]].rename(columns={"score": "full_score"}),
                        on=keys, validate="many_to_one")
    cases["safe_compression"] = (cases.width < P) & (cases.score >= cases.full_score - .02)
    if args.verify:
        pd.testing.assert_frame_equal(cases, pd.read_csv(args.output/"cases.csv"), check_exact=False,
                                      rtol=1e-14, atol=1e-14)
        cases[[*keys, "method", "support_sha256", "prediction_sha256"]].to_csv(
            args.output/"fresh-process-audit.csv", index=False)
        return
    cases.to_csv(args.output/"cases.csv", index=False)
    compared = cases.merge(cases[cases.method == "vote"][[*keys, "score", "distinct_kept"]],
                           on=keys, suffixes=("", "_vote"), validate="many_to_one")
    compared["gap_vote_minus_comparator"] = compared.score_vote - compared.score
    compared["vote_better"] = compared.gap_vote_minus_comparator > 1e-12
    compared["vote_worse"] = compared.gap_vote_minus_comparator < -1e-12
    compared["vote_more_distinct"] = compared.distinct_kept_vote > compared.distinct_kept
    compared["all_distinct_kept"] = compared.distinct_kept == 4
    compared["at_least_one_proxy"] = compared.proxies_kept > 0
    compared.groupby(["task", "proxies_available", "method"]).agg(
        cases=("seed", "size"), median_width=("width", "median"), safe_compression=("safe_compression", "sum"),
        mean_distinct=("distinct_kept", "mean"), all_distinct=("all_distinct_kept", "sum"),
        has_proxy=("at_least_one_proxy", "sum"), vote_more_distinct=("vote_more_distinct", "sum"),
        vote_better=("vote_better", "sum"), vote_worse=("vote_worse", "sum"),
        median_score_delta=("gap_vote_minus_comparator", "median")).to_csv(args.output/"by-mechanism.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    main(parser.parse_args())
