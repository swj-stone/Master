"""Same-width column quality for frozen A2/B2 development supports; no timing."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import ROOT, panels, digest, file_hash
from experiments.fast_document_vote_benchmark import select
from experiments.minpi_constant_transfer import fit_predict


def run(output, inputs, verify):
    protocol = dict(
        input_arrays={p.name: file_hash(p) for p in sorted(inputs.glob("*-arrays.npz"))},
        source_sha256=file_hash(Path(__file__)), random_draws=20,
        random_rule="SeedSequence-compatible [108391, fold, selector_seed, draw] permutation prefix",
        score_epsilon=1e-12, same_training_refit=True, formal_timing=False,
        interpretation="Development comparisons at frozen A2/B2 widths; full retention remains failure.",
    )
    if verify:
        assert json.loads((output / "protocol.json").read_text()) == protocol
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    records, summary = [], []
    for X, y, Xt, truth, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp"):
            continue
        filename = meta["dataset"] + "-arrays.npz"
        saved = np.load(inputs / filename)
        arrays = {}
        cache = {}
        for budget in ("a2", "b2"):
            candidate = saved[budget + "_support"]
            width = len(candidate)
            pred = fit_predict(X, y, Xt, candidate, "classification", meta["seed"], False)
            np.testing.assert_array_equal(pred, saved[budget + "_prediction"])
            candidate_score = balanced_accuracy_score(truth, pred)
            methods = [("minpi", candidate)]
            if width not in cache:
                controls = [("F", select("F_match", X, y, meta["seed"], width)),
                            ("pooled", select("pooled_match", X, y, meta["seed"], width)),
                            ("random_fixed", select("random_match", X, y, meta["seed"], width))]
                controls += [(f"random_{draw:02d}", np.sort(np.random.default_rng(
                    [108391, meta["fold"], meta["seed"], draw]).permutation(X.shape[1])[:width]))
                             for draw in range(20)]
                cache[width] = [(name, cols, fit_predict(X, y, Xt, cols, "classification", meta["seed"], False))
                                for name, cols in controls]
            entries = [("minpi", candidate, pred), *cache[width]]
            scores = {}
            for method, cols, prediction in entries:
                assert len(cols) == width
                score = balanced_accuracy_score(truth, prediction)
                scores[method] = score
                arrays[budget + "_" + method + "_support"] = cols
                arrays[budget + "_" + method + "_prediction"] = prediction
                records.append(dict(dataset=meta["dataset"], budget=budget, method=method,
                                    width=width, p=X.shape[1], score=score,
                                    support_sha256=digest(cols), prediction_sha256=digest(prediction)))
            draws = np.array([scores[f"random_{i:02d}"] for i in range(20)])
            summary.append(dict(dataset=meta["dataset"], budget=budget, width=width, p=X.shape[1],
                                reduced=width < X.shape[1], score=candidate_score,
                                F_score=scores["F"], F_delta=candidate_score-scores["F"],
                                pooled_score=scores["pooled"], pooled_delta=candidate_score-scores["pooled"],
                                random_fixed_score=scores["random_fixed"],
                                random_fixed_delta=candidate_score-scores["random_fixed"],
                                random_median=float(np.median(draws)), random_min=float(draws.min()),
                                random_max=float(draws.max()), random_wins=int((candidate_score > draws+1e-12).sum()),
                                random_ties=int((np.abs(candidate_score-draws) <= 1e-12).sum()),
                                random_losses=int((candidate_score < draws-1e-12).sum())))
        saved.close()
        if verify:
            with np.load(output / filename) as previous:
                assert set(previous.files) == set(arrays)
                for name, value in arrays.items():
                    np.testing.assert_array_equal(value, previous[name])
        else:
            np.savez_compressed(output / filename, **arrays)
    for name, rows in (("results", records), ("summary", summary)):
        frame = pd.DataFrame(rows)
        if verify:
            pd.testing.assert_frame_equal(frame, pd.read_csv(output / (name + ".csv")),
                                          check_exact=False, atol=1e-12, rtol=0)
        else:
            frame.to_csv(output / (name + ".csv"), index=False)
    assert len(records) == 192 and len(summary) == 8
    print("Verified" if verify else "Measured scores", len(records), "method configurations")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.inputs, args.verify)
