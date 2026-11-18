"""Recalculate compression, quality, robustness margins and same-width controls from raw runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]


def audit(run):
    raw = pd.read_csv(run/"trials.csv")
    case_key = [*KEY, "method"]
    assert not raw.duplicated([*case_key, "repeat"]).any()
    assert raw.groupby(case_key).repeat.size().isin([5, 7]).all()
    assert raw.groupby(case_key)[["width", "p", "score"]].nunique().eq(1).all().all()
    assert raw.total_ms.gt(0).all() and raw.select_ms.ge(0).all()
    elapsed = raw.groupby(case_key, as_index=False).agg(
        width=("width","first"), p=("p","first"), score=("score","first"),
        select_ms=("select_ms","median"), total_ms=("total_ms","median"))
    base = elapsed.loc[elapsed.method=="full", [*KEY,"score","total_ms"]]
    elapsed = elapsed.merge(base, on=KEY, suffixes=("","_full"), validate="many_to_one")
    elapsed["gap"] = elapsed.score - elapsed.score_full
    elapsed["ratio"] = elapsed.total_ms / elapsed.total_ms_full
    elapsed["keep"] = elapsed.width / elapsed.p
    elapsed["reduced"] = elapsed.width < elapsed.p
    elapsed["one_removed"] = elapsed.width == elapsed.p - 1
    elapsed["quality"] = elapsed.gap >= -.02
    safe = elapsed.quality & elapsed.reduced
    elapsed["joint"] = safe & (elapsed.ratio < 1)
    elapsed["joint_5pct"] = safe & (elapsed.ratio <= .95)
    elapsed["joint_10pct"] = safe & (elapsed.ratio <= .9)
    elapsed["joint_no_loss"] = elapsed.reduced & (elapsed.gap >= 0) & (elapsed.ratio < 1)
    saved = pd.read_csv(run/"cases.csv")
    aligned = elapsed.merge(saved, on=case_key, validate="one_to_one", suffixes=("", "_saved"))
    assert len(aligned) == len(elapsed) == len(saved)
    for field in ("joint","joint_5pct","joint_10pct","width","p","reduced"):
        np.testing.assert_array_equal(aligned[field], aligned[field+"_saved"])
    np.testing.assert_array_equal(aligned.quality, aligned.within)
    assert not elapsed.loc[~elapsed.reduced, "joint"].any()
    # Check the free width handed to every adaptive random/F baseline.
    names = set(elapsed.method)
    for stem in ("pilot","tournament","exchange","native","stable"):
        candidate = next((m for m in (stem+"_auto",stem) if m in names), None)
        if candidate is None:
            continue
        controls = (["F","random"] if candidate=="exchange" and "F" in names else
                    ["F_"+stem, "random_"+stem])
        widths = elapsed[elapsed.method==candidate].set_index(KEY).width.sort_index()
        for m in controls:
            if m in names:
                pd.testing.assert_series_equal(widths, elapsed[elapsed.method==m].set_index(KEY).width.sort_index())
    if "random_75" in names:
        fixed = elapsed[elapsed.method == "random_75"].set_index(KEY).width.sort_index()
        for method in names:
            if method.endswith("_75"):
                actual = elapsed[elapsed.method == method].set_index(KEY).width.sort_index()
                pd.testing.assert_series_equal(actual, fixed)
    if "sparse_exchange" in names:
        sparse_width = elapsed[elapsed.method == "sparse_exchange"].set_index(KEY).width.sort_index()
        random_width = elapsed[elapsed.method == "random"].set_index(KEY).width.sort_index()
        pd.testing.assert_series_equal(sparse_width, random_width)
    replay = pd.read_csv(run/"fresh-process-audit.csv").set_index(case_key).sort_index()
    expected = elapsed.set_index(case_key).sort_index()
    pd.testing.assert_index_equal(replay.index, expected.index)
    for field in ("support_sha256", "prediction_sha256"):
        assert replay[field].str.fullmatch("[0-9a-f]{64}").all()
    for field in ("width", "score"):
        if field in replay:
            np.testing.assert_allclose(replay[field], expected[field], rtol=0, atol=1e-12)
    elapsed["run"] = run.name
    return elapsed


def main(args):
    cases = pd.concat([audit(run) for run in args.runs], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=False)
    cases.to_csv(args.output/"cases.csv", index=False)
    metrics = dict(configs=("width","size"), median_keep=("keep","median"), min_keep=("keep","min"),
                   max_keep=("keep","max"), median_ratio=("ratio","median"),
                   median_select_ms=("select_ms","median"), worst_gap=("gap","min"),
                   **{f:(f,"sum") for f in ("reduced","one_removed","quality","joint","joint_5pct",
                                             "joint_10pct","joint_no_loss")})
    for keys, name in ((["run","method"],"summary"), (["run","dataset","method"],"by-source")):
        cases.groupby(keys).agg(**metrics).to_csv(args.output/(name+".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    main(parser.parse_args())
