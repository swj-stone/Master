"""Check every stored Turkish-mail score using exact classwise hit counts."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.reliability_mail_external import panels
from experiments.stable_minpi_stress import file_hash, write_json


KEY = ["dataset", "fold", "seed"]


def run(source, output):
    frame = pd.read_csv(source / "cases.csv", float_precision="round_trip").set_index(KEY+["method"])
    class_rows, score_rows = [], []
    for _, _, _, truth, info in panels():
        key = tuple(info[name] for name in KEY)
        exact = {}
        with np.load(source / f"{key[0]}-{key[1]}-{key[2]}.npz") as saved:
            for method in frame.xs(key).index:
                pred = saved[method+"_pred"]
                terms = []
                for label in np.unique(truth):
                    count = int(np.sum(truth == label))
                    correct = int(np.sum((truth == label) & (pred == label)))
                    terms.append(Fraction(correct, count))
                    class_rows.append(dict(zip(KEY, key), method=method, label=int(label),
                                           class_count=count, class_correct=correct))
                exact[method] = sum(terms)/len(terms)
                np.testing.assert_allclose(float(exact[method]), frame.loc[(*key, method), "score"],
                                           rtol=0, atol=1e-14)
        for method, score in exact.items():
            gap = score-exact["full"]
            raw_gap = frame.loc[(*key, method), "gap"]
            assert (raw_gap >= -.02-1e-12) == (gap >= -Fraction(1, 50))
            score_rows.append(dict(zip(KEY, key), method=method, score_fraction=str(score),
                                   gap_fraction=str(gap), strict_gap=raw_gap,
                                   within_exact=gap >= -Fraction(1, 50),
                                   corrected=(raw_gap >= -.02) != (gap >= -Fraction(1, 50))))
        for baseline, score in exact.items():
            raw = frame.loc[(*key, "reliability"), "score"]-frame.loc[(*key, baseline), "score"]
            assert (raw >= -.02-1e-12) == (exact["reliability"]-score >= -Fraction(1, 50))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_rows).to_csv(output / "prediction-class-counts.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "prediction-exact-scores.csv", index=False)
    write_json(output / "prediction-audit.json", dict(
        source=str(source), audit_source_sha256=file_hash(Path(__file__)),
        raw_arrays_sha256={path.name: file_hash(path) for path in sorted(source.glob("*.npz"))},
        checked_cases=len(score_rows), all_scores_equal_exact_balanced_accuracy=True,
        all_guarded_comparisons_equal_fraction=True, corrected=sum(row["corrected"] for row in score_rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
