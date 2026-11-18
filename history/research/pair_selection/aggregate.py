"""Check completed timing rotations and join the two independent processes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
KEY = ["cohort", "dataset", "fold", "seed"]
METHOD = KEY + ["method"]


def table(run, suite, count):
    pairs = []
    for process in (1, 2):
        base = run / f"{suite}-pass{process}"
        info = json.loads((base / "complete.json").read_text())
        assert info["cases"] == count and info["process"] == process
        assert all(pool["num_threads"] == 1 for pool in info["threads"])
        trials = pd.read_csv(base / "times.csv", keep_default_na=False)
        assert len(trials) == count * 25
        assert not trials.duplicated(METHOD + ["repeat"]).any()
        assert not trials.duplicated(METHOD + ["position"]).any()
        assert not trials.duplicated(KEY + ["repeat", "position"]).any()
        assert (trials.select_ms >= 0).all() and (trials.total_ms >= trials.select_ms).all()
        medians = trials.groupby(METHOD)[["select_ms", "total_ms"]].median().add_suffix(str(process))
        quality = pd.read_csv(base / "quality.csv", keep_default_na=False).set_index(METHOD).sort_index()
        assert len(quality) == count * 5 and quality.index.is_unique
        if process == 1:
            reference = quality
        else:
            pd.testing.assert_frame_equal(reference, quality)
        pairs.append(medians)
    df = reference.join(pairs, how="inner").reset_index()
    inputs = pd.read_csv(base / "inputs.csv", keep_default_na=False)
    return df.merge(inputs, on=KEY, how="inner", validate="many_to_one")


def save(run, out):
    frames = [table(run, side, n) for side, n in [("archive", 161), ("external", 12)]]
    methods = pd.concat(frames, ignore_index=True).sort_values(METHOD)
    assert len(methods) == 173 * 5
    out.mkdir(parents=True, exist_ok=True)
    methods.to_csv(out / "method-cases.csv", index=False)
    fields = ["score", "cols", "k", "select_ms1", "select_ms2", "total_ms1", "total_ms2"]
    cases = methods.pivot(index=KEY, columns="method", values=fields)
    cases.columns = ["_".join(pair) for pair in cases.columns]
    metadata = [*KEY, "task", "metric", "p", "n_train", "n_test", "train_sha", "target_sha", "test_sha", "truth_sha"]
    cases = methods[metadata].drop_duplicates().set_index(KEY).join(cases)
    for c in cases.columns:
        if c.startswith(("score_", "k_", "select_ms", "total_ms")):
            cases[c] = pd.to_numeric(cases[c])
    for arm in ("current", "F", "random", "full"):
        cases["gap_" + arm] = cases.score_pair - cases["score_" + arm]
        cases["ratio_" + arm] = np.maximum(cases.total_ms1_pair / cases["total_ms1_" + arm],
                                           cases.total_ms2_pair / cases["total_ms2_" + arm])
    assert cases.k_pair.eq(cases.k_current).all() and cases.k_pair.eq(cases.k_F).all()
    assert cases.k_pair.eq(cases.k_random).all() and cases.k_full.eq(cases.p).all()
    cases.to_csv(out / "paired-cases.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    save(args.run, args.out)
