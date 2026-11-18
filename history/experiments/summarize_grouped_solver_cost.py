"""Independent paired summary of the two development-only solver-cost runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "selector", "solver"]


def read(path):
    raw = pd.read_csv(path/"trials.csv")
    grouped = raw.groupby(KEY)
    assert grouped.size().eq(5).all()
    assert grouped[["width", "p", "score"]].nunique().eq(1).all().all()
    assert not raw.duplicated(KEY+["repeat"]).any()
    assert grouped.position.nunique().eq(5).all()
    saved = pd.read_csv(path/"cases.csv").set_index(KEY).sort_index()
    for field in ("select_ms", "fit_ms", "predict_ms", "slice_ms", "total_ms"):
        np.testing.assert_allclose(grouped[field].median(), saved[field], rtol=0, atol=1e-9)
    return saved, len(raw)


def run(runs, output):
    (first, n_first), (second, n_second) = [read(path) for path in runs]
    pd.testing.assert_index_equal(first.index, second.index)
    for field in ("width", "p", "score"):
        np.testing.assert_array_equal(first[field], second[field])
    for path in runs[0].glob("*.npz"):
        with np.load(path) as a, np.load(runs[1]/path.name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=path.name+key)
    combined = first[["width", "p", "score"]].copy()
    for field in ("select_ms", "fit_ms", "predict_ms", "slice_ms", "total_ms"):
        combined[field] = np.maximum(first[field], second[field])
    records = []
    for key, row in combined.iterrows():
        dataset, selector, solver = key
        values = dict(zip(KEY, key)) | row.to_dict()
        for baseline_name, baseline_key in (("full_same_solver", (dataset, "full", solver)),
                                            ("full_cholesky", (dataset, "full", "cholesky"))):
            gap = row.score - combined.loc[baseline_key, "score"]
            ratio = max(frame.loc[key, "total_ms"] / frame.loc[baseline_key, "total_ms"]
                        for frame in (first, second))
            values[baseline_name+"_score_delta"] = gap
            values[baseline_name+"_time_ratio"] = ratio
            values[baseline_name+"_joint"] = bool(row.width < row.p and gap >= -.02-1e-12 and ratio < 1)
        # This is a retrospective descriptive frontier, not a rule for choosing
        # an arm from outer outcomes. Timing uses the worse median per arm.
        peers = combined.loc[dataset]
        dominates = ((peers.score >= row.score-1e-12) & (peers.total_ms <= row.total_ms)
                     & (peers.width <= row.width)
                     & ((peers.score > row.score+1e-12) | (peers.total_ms < row.total_ms)
                        | (peers.width < row.width)))
        values["dominated_by"] = ";".join(selector+"/"+solver for selector, solver in peers.index[dominates])
        records.append(values)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(output/"paired-costs.csv", index=False)
    # Score differences need no aggregation: deterministic supports/predictions
    # are identical across timing passes and the table was already checked.
    a, b = [pd.read_csv(path/"solver-score-differences.csv") for path in runs]
    pd.testing.assert_frame_equal(a, b)
    a.to_csv(output/"solver-score-differences.csv", index=False)
    (output/"audit.json").write_text(json.dumps(dict(
        cases=2, method_solver_outputs_per_pass=len(first), timed_calls=n_first+n_second,
        array_agreement=True, time="Maximum of two run medians; relative cost is maximum of paired run ratios",
        scope="Development-only solver diagnosis. Frontier computed descriptively after evaluation; never used to choose a support.",
        runs=[str(path) for path in runs]), indent=2), encoding="utf8")
    print(pd.DataFrame(records)[KEY+["width", "score", "select_ms", "fit_ms", "total_ms",
                                    "full_cholesky_score_delta", "full_cholesky_time_ratio",
                                    "full_cholesky_joint"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.runs, args.output)
