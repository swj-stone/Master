"""Compare paired document votes with two marginal rankers at the recorded vote widths."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import CASE, panels
from experiments.minpi_constant_transfer import fit_predict
from experiments.stable_minpi_stress import ROOT, digest, file_hash, write_json
from minppi_mofs.document_vote_minpi import document_vote_top_from_training


METHODS = ("document_vote", "zero_peer", "pooled")
TOL = 1e-12


def pooled_gain(X, y):
    """Directly compute the all-row marginal unit-ridge fit gain via CSR/indicator products."""
    _, labels = np.unique(y, return_inverse=True)
    target = np.eye(labels.max() + 1)[labels]
    freq = np.asarray(X.sum(axis=0)).ravel()
    rhs = X.T @ target - freq[:, None] * target.mean(axis=0)
    norm = freq * (1 - freq / len(y))
    return np.einsum("ij,ij->i", rhs, rhs) / (norm + 1.)


def run(output, prior, verify):
    files = [prior/"protocol.json", prior/"panels.csv", prior/"random-draws.csv", prior/"cases.csv",
             ROOT/"src/minppi_mofs/document_vote_minpi.py", ROOT/"experiments/document_vote_benchmark.py",
             ROOT/"experiments/minpi_constant_transfer.py", Path(__file__)]
    state = {"files": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
             "methods": METHODS, "width": "Frozen document-vote support length on the same training rows",
             "zero_peer": "Same relative 12-decimal grid and all-training marginal tie break as document vote",
             "pooled": "All-training unit-ridge explained sum of squares divided by fit penalty-adjusted term norm",
             "timing": "Quality and support ablation only"}
    if verify:
        import json
        assert json.loads((output/"protocol.json").read_text()) == json.loads(json.dumps(state))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"protocol.json", state)
    archived_cases = pd.read_csv(prior/"cases.csv").set_index([*CASE, "method"])
    archived_info = pd.read_csv(prior/"panels.csv").set_index(CASE)
    random = pd.read_csv(prior/"random-draws.csv").set_index(CASE)
    cases, overlaps = [], []
    for X, y, Xt, gold, info in panels():
        key = {k: info[k] for k in CASE}
        ix = tuple(key.values())
        archived = archived_info.loc[ix]
        for field in info:
            if field.endswith("_sha256"):
                assert archived[field] == info[field]
        voted, diagnostics = document_vote_top_from_training(X, y, random_state=info["seed"])
        width = len(voted)
        index = np.arange(X.shape[1])
        marginal = pooled_gain(X, y)
        zero = diagnostics["zero_peer_scores"]
        scale = np.max(np.abs(zero))
        zero = np.round(zero / scale, 12) if scale else zero
        supports = dict(document_vote=voted, zero_peer=np.sort(np.lexsort((index, -marginal, -zero))[:width]),
                        pooled=np.sort(np.lexsort((index, -marginal))[:width]))
        # Confirm the independently calculated tie breaker also recreates the original support.
        paired = diagnostics["scores"]
        scale = np.max(np.abs(paired))
        paired = np.round(paired / scale, 12) if scale else paired
        np.testing.assert_array_equal(np.sort(np.lexsort((index, -marginal, -paired))[:width]), voted)
        draws = random.loc[[ix]]
        assert len(draws) == 20 and (draws.width == width).all()
        median = draws.score.median()
        full_score = archived_cases.loc[(*ix, "full"), "score"]
        trace = f"{info['dataset']}-{info['fold']}-{info['seed']}.npz"
        arrays, scores = {}, {}
        for method, cols in supports.items():
            pred = fit_predict(X, y, Xt, cols, "classification", info["seed"], False)
            score = balanced_accuracy_score(gold, pred)
            scores[method] = score
            arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
            cases.append(dict(**key, method=method, width=width, p=X.shape[1], score=score,
                              score_minus_full=score-full_score, score_minus_random_median=score-median,
                              random_median=median, wins_of_20=int((score > draws.score+TOL).sum()),
                              support_sha256=digest(cols), prediction_sha256=digest(pred)))
        with np.load(prior / trace) as source:
            for suffix in ("_cols", "_pred"):
                field = "document_vote" + suffix
                np.testing.assert_array_equal(arrays[field], source[field])
        for method, reference in combinations(METHODS, 2):
            shared = len(np.intersect1d(supports[method], supports[reference]))
            overlaps.append(dict(**key, method=method, reference=reference, width=width,
                                 shared=shared, intersection_fraction=shared/width,
                                 jaccard=shared/(2*width-shared), score_delta=scores[method]-scores[reference]))
        if verify:
            with np.load(output/trace) as source:
                for field, values in arrays.items():
                    np.testing.assert_array_equal(values, source[field])
        else:
            np.savez_compressed(output/trace, **arrays)
    cases, overlaps = pd.DataFrame(cases), pd.DataFrame(overlaps)
    assert len(cases) == len(overlaps) == 51
    if verify:
        pd.testing.assert_frame_equal(cases, pd.read_csv(output/"cases.csv"), rtol=0., atol=1e-12)
        pd.testing.assert_frame_equal(overlaps, pd.read_csv(output/"paired.csv"), rtol=0., atol=1e-12)
        cases.to_csv(output/"fresh-process-audit.csv", index=False)
        return
    cases.to_csv(output/"cases.csv", index=False)
    overlaps.to_csv(output/"paired.csv", index=False)
    overlaps.groupby(["dataset", "method", "reference"]).agg(
        cases=("width","size"), median_jaccard=("jaccard","median"),
        mean_score_delta=("score_delta","mean"), median_score_delta=("score_delta","median"),
        wins=("score_delta", lambda s: (s > TOL).sum()), ties=("score_delta", lambda s: (s.abs() <= TOL).sum()),
        losses=("score_delta", lambda s: (s < -TOL).sum())).to_csv(output/"by-source.csv")
    cases.groupby(["dataset","method"]).agg(
        cases=("width","size"), above_random_median=("score_minus_random_median", lambda s: (s > TOL).sum()),
        median_random_delta=("score_minus_random_median","median"), min_wins_of_20=("wins_of_20","min"),
        within_full=("score_minus_full", lambda s: (s >= -.02).sum())).to_csv(output/"against-random.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior", type=Path, default=ROOT/"results/minpi-study/document-vote-20261117b")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output.resolve(), args.prior.resolve(), args.verify)
