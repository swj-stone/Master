"""Evaluate frozen adaptive votes on published probe data and unmodified digit measurements."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_transfer as transfer
from experiments import signal_vote_benchmark as bench
from experiments.fast_minpi_five_source import summarize_timings
from experiments.minpi_study_external import load_challenge
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json


ARMS = ("signal_auto", "no_peer", "F_signal", "random_signal", "full")
DATA = ROOT / "data/raw"
ASSETS = {
    "gisette.csv": "02b8d81c00ed14e65310e9c57ec8ec34deaed237eb04117ba545c41ceef4ed6a",
    "uci-multiple-features.zip": "898a50a7637f1ed5a8cee2493aaa0e7f4d52795c412f59b0969ec7c3046ee7bd",
}


def published_sources():
    for name in ("arcene", "dexter"):
        (X, y), (Xt, truth) = load_challenge(name)
        yield name, 1, X, y, Xt, truth, {"split": "published train / validation"}
    for name in ("gisette", "multiple_features"):
        if name == "gisette":
            table = pd.read_csv(DATA / "gisette.csv", dtype=np.float32)
            y = table.pop("label").to_numpy(dtype=int)
            X = table.to_numpy()
        else:
            with ZipFile(DATA / "uci-multiple-features.zip") as archive:
                blocks = []
                for group in ("fac", "fou", "kar", "mor", "pix", "zer"):
                    with archive.open("mfeat-" + group) as src:
                        blocks.append(np.loadtxt(src, dtype=np.float32))
            X = np.column_stack(blocks)
            assert X.shape == (2000, 649)
            y = np.repeat(np.arange(10), 200)
        splitter = StratifiedKFold(3, shuffle=True, random_state=96731)
        # Keep the predeclared fold assignments. Exclude *every* copy of a
        # repeated image descriptor, so no label determines which copy wins.
        unique = (~pd.Series([digest(row) for row in X]).duplicated(keep=False).to_numpy()
                  if name == "multiple_features" else np.ones(len(y), dtype=bool))
        for fold, (train, test) in enumerate(splitter.split(X, y), 1):
            if name == "gisette":
                pool, test = train_test_split(test, test_size=512, stratify=y[test], random_state=731 + fold)
                train, _ = train_test_split(pool, train_size=1536, stratify=y[pool], random_state=291 + fold)
            train, test = train[unique[train]], test[unique[test]]
            assert not np.intersect1d(train, test).size
            yield name, fold, X[train], y[train], X[test], y[test], {
                "train": train, "test": test, "excluded_repeated_source_rows": int((~unique).sum())}


def variable_space(X, Xt, seed):
    variable = np.flatnonzero(np.ptp(X, axis=0) > 0)
    order = np.random.default_rng(seed).permutation(variable)
    return np.ascontiguousarray(X[:, order]), np.ascontiguousarray(Xt[:, order]), order


def select(method, panel, widths):
    width = widths.get("signal")
    return bench.select(method, panel, width, ordered=False)[0]


def run(output, verify):
    for name, expected in ASSETS.items():
        assert file_hash(DATA / name) == expected
    protocol = {"predeclared": file_hash(ROOT / "docs/minpi-study/signal-vote-source-protocol.md"),
                "duplicate_rule": "Multiple Features: exclude all copies of every repeated raw feature vector "
                                  "without reading labels and retain the original fold assignment.",
                "sources": ASSETS, "arms": list(ARMS), "repeats": 5, "random_draws": 32,
                "code": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (
                    Path(__file__), Path(bench.__file__), Path(transfer.__file__),
                    ROOT / "src/minppi_mofs/signal_vote_minpi.py",
                    ROOT / "experiments/minpi_study_external.py")}}
    if verify:
        assert json.loads((output / "protocol.json").read_text()) == protocol
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
    results, audits, metadata, random_results, splits = [], [], [], [], {}
    for index, (name, fold, X, y, Xt, truth, partition) in enumerate(published_sources()):
        original_p = X.shape[1]
        seed = 723101 + index
        constant_times = []
        for _ in range(1 if verify else 5):
            start = perf_counter_ns()
            active_train, active_test, ids = variable_space(X, Xt, seed)
            constant_times.append((perf_counter_ns() - start) / 1e6)
        assert np.ptp(active_train, axis=0).min() > 0
        # Check exact feature vectors across the defined training and test sets.
        train_rows = {digest(row) for row in active_train}
        overlap = sum(digest(row) in train_rows for row in active_test)
        assert overlap == 0
        panel = transfer.Panel(name, "published", name, "classification", fold, seed, active_train, y,
                               active_test, truth, {})
        key = bench.case_id(panel)
        splits[f"{name}-{fold}"] = {"rows": partition, "column_ids": ids}
        trace = output / f"{name}-{fold}-{seed}.npz"
        rows, audit = transfer.run_case(panel, [10109, fold, seed], 5, trace, verify, ARMS,
                                        select, {"signal_auto": "signal"})
        results.extend(rows)
        audits.extend(audit)
        with np.load(trace) as stored:
            width = len(stored["signal_auto_cols"])
            assert len(stored["no_peer_cols"]) == width
        metadata.append(dict(**key, p=active_train.shape[1], original_p=original_p, width=width, n_train=len(y),
                             n_test=len(truth), across_split_duplicate_rows=overlap,
                             common_preprocess_ms=float(np.median(constant_times))))
        for draw in range(32):
            rng = np.random.default_rng([748219, index, draw])
            cols = np.sort(rng.permutation(active_train.shape[1])[:width])
            pred = prediction(active_train, y, active_test, cols, panel.task, seed)
            random_results.append(dict(**key, draw=draw, width=width, score=metric(truth, pred, panel.task),
                                       support_sha256=digest(cols), prediction_sha256=digest(pred)))
        print(name, fold, "verified" if verify else "measured", flush=True)
    random_frame = pd.DataFrame(random_results)
    if verify:
        saved = pd.read_csv(output / "random-draws.csv")
        pd.testing.assert_frame_equal(random_frame, saved, rtol=0, atol=1e-14)
        assert json.loads((output / "splits.json").read_text()) == json.loads(json.dumps(splits, default=lambda x:x.tolist()))
        pd.DataFrame(audits).to_csv(output / "fresh-process-audit.csv", index=False)
        return
    assert len(metadata) == 8 and len(results) == 200 and len(random_frame) == 256
    write_json(output / "splits.json", splits)
    random_frame.to_csv(output / "random-draws.csv", index=False)
    cases = summarize_timings(output, results)
    panels = pd.DataFrame(metadata)
    panels.to_csv(output / "panels.csv", index=False)
    vote = cases[cases.method == "signal_auto"].merge(panels, on=[*bench.CASE, "width", "p"], validate="one_to_one")
    vote["ratio_including_preprocess"] = ((vote.total_ms + vote.common_preprocess_ms) /
                                          (vote.full_total_ms + vote.common_preprocess_ms))
    votes = random_frame.merge(vote[[*bench.CASE, "score"]], on=bench.CASE, suffixes=("_draw", "_vote"))
    votes["win"] = votes.score_vote > votes.score_draw + 1e-12
    stats = votes.groupby(bench.CASE, as_index=False).agg(wins=("win","sum"), median_random=("score_draw","median"))
    vote = vote.merge(stats, on=bench.CASE, validate="one_to_one")
    vote["strict"] = vote.joint & (vote.wins >= 24)
    vote.to_csv(output / "vote-comparisons.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
