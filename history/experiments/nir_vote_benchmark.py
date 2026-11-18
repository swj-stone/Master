"""Nine-way same-run comparison on publisher-split, naturally ordered near-infrared spectra."""
import argparse
import json
import platform
import re
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json
from minppi_mofs.band_vote_minpi import band_center_from_training, band_vote_top_from_training
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training
from minppi_mofs.native_pivotal_minpi import library_path


DATA = ROOT / "data/raw"
MANIFEST = DATA / "nirbench-manifest.csv"
IDS = (1, 5, 13, 15, 19, 22, 30)
SEEDS = (1129, 2027, 3469)
ARMS = ("band_vote", "band_center", "grid_match", "F_match", "random_match", "F_50", "random_50",
        "signal_vote", "full")
CASE = ["dataset", "fold", "seed"]
REPEATS = 9


def wavelengths(frame, target, expected):
    """Use numeric wavelength headers and reject any disagreement with the publisher's width."""
    values = {}
    for name in frame.columns.drop(target):
        match = re.fullmatch(r"(?:X|wl_|nm_|wavelength_)?([0-9]+(?:[.,][0-9]+)?)(?:\s*nm)?", name, re.I)
        if match:
            values[name] = float(match[1].replace(",", "."))
    if len(values) != expected or len(set(values.values())) != expected:
        raise ValueError(f"Expected {expected} unique ordered spectral channels; found {len(values)} in "
                         f"{list(frame.columns[:6])} ... {list(frame.columns[-3:])}")
    return sorted(values, key=values.get)


def sources():
    manifest = pd.read_csv(MANIFEST)
    ids = manifest.dataset.str.extract(r"^(\d+)-", expand=False).astype(int)
    for source_id in IDS:
        records = manifest[ids == source_id].set_index("split")
        assert records.index.is_unique and set(records.index) == {"train", "test"}
        splits, input_rows, kept_ids = [], [], []
        train_columns = None
        for part in ("train", "test"):
            record = records.loc[part]
            path = DATA / "nirbench" / record.dataset / Path(record.path).name
            if path.stat().st_size != int(record.bytes) or file_hash(path) != record.sha256:
                raise ValueError(f"Publisher hash mismatch: {path}")
            df = pd.read_csv(path)
            assert len(df) == int(record.rows)
            cols = wavelengths(df, record.target_column, int(record.spectral_features))
            if train_columns is None:
                train_columns = cols
            if train_columns != cols:
                raise ValueError("Train and test wavelength coordinates disagree.")
            selected = df[[*cols, record.target_column]].to_numpy(dtype=np.float64)
            valid = np.isfinite(selected).all(axis=1)
            clean = selected[valid]
            clean[clean == 0.] = 0.  # Canonicalize signed zero for label-blind duplicate detection.
            splits.append((np.ascontiguousarray(clean[:, :-1]), np.ascontiguousarray(clean[:, -1])))
            input_rows.append(len(df))
            kept_ids.append(np.flatnonzero(valid))
        (X, y), (Xt, yt) = splits
        # The publisher supplies outer partitions. Decide cross-split duplicate removal from spectra alone.
        train_hash = {digest(row) for row in X}
        test_hash = [digest(row) for row in Xt]
        new_test = np.asarray([h not in train_hash for h in test_hash])
        assert new_test.any()
        info = dict(dataset=record.dataset, dataset_id=source_id, n_train=len(y), n_test=int(new_test.sum()),
                    p=X.shape[1], original_train=input_rows[0], original_test=input_rows[1],
                    drop_missing_train=input_rows[0]-len(y), drop_missing_test=input_rows[1]-len(yt),
                    drop_cross_test=int((~new_test).sum()), repeated_train=len(y)-len(train_hash),
                    repeated_test=len(test_hash)-len(set(test_hash)),
                    constant_channels=int(np.count_nonzero(np.ptp(X,axis=0)==0.)),
                    eligible=len(y) >= 512 and X.shape[1] >= 100,
                    train_ids_sha256=digest(kept_ids[0]), test_ids_sha256=digest(kept_ids[1][new_test]),
                    wavelengths=json.dumps(train_columns), target=record.target_column)
        yield X, y, Xt[new_test], yt[new_test], info


def select(method, X, y, seed, width):
    if method == "band_vote":
        return band_vote_top_from_training(X, y, task="regression", random_state=seed)
    if method == "band_center":
        return band_center_from_training(X, y, task="regression", random_state=seed)
    if method == "signal_vote":
        return signal_vote_top_from_training(X, y, task="regression", random_state=seed)
    if method == "grid_match":
        return np.rint(np.linspace(0, X.shape[1] - 1, width)).astype(np.intp), {}
    if method == "full":
        return np.arange(X.shape[1]), {}
    rule, budget = method.split("_")
    size = width if budget == "match" else (X.shape[1] + 1) // 2
    return choose(rule.lower(), X, y, "regression", seed, size), {}


def one_case(X, y, Xt, yt, source, seed, output, verify):
    key = dict(dataset=source["dataset"], fold=1, seed=seed)
    trace = output / f"{source['dataset']}-{seed}.npz"
    arrays, scores, trials, audit = {}, {}, [], []
    width = None
    for method in ARMS:
        cols, info = select(method, X, y, seed, width)
        check_columns(cols, X.shape[1])
        if method == "band_vote":
            width = len(cols)
            arrays["starts"] = np.asarray(info["band_starts"])
            arrays["stops"] = np.asarray(info["band_stops"])
            assert len(cols) == len(arrays["starts"])
            assert np.all((cols >= arrays["starts"]) & (cols < arrays["stops"]))
        if method == "band_center":
            np.testing.assert_array_equal(info["band_starts"], arrays["starts"])
            np.testing.assert_array_equal(info["band_stops"], arrays["stops"])
        if method.endswith("_match") or method == "band_center":
            assert len(cols) == width
        pred = prediction(X, y, Xt, cols, "regression", seed)
        scores[method] = metric(yt, pred, "regression")
        arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
        audit.append(dict(**key, method=method, width=len(cols), score=scores[method], support_sha256=digest(cols),
                          prediction_sha256=digest(pred)))
    if verify:
        with np.load(trace) as previous:
            assert set(previous.files) == set(arrays)
            for k, value in arrays.items():
                np.testing.assert_array_equal(value, previous[k], err_msg=k)
    else:
        order = np.random.default_rng([931017, source["dataset_id"], seed]).permutation(ARMS)
        for r in range(REPEATS):
            for position, method in enumerate(np.roll(order, r)):
                t0 = perf_counter_ns()
                cols, _ = select(method, X, y, seed, width)
                t1 = perf_counter_ns()
                pred = prediction(X, y, Xt, cols, "regression", seed)
                t2 = perf_counter_ns()
                np.testing.assert_array_equal(cols, arrays[method+"_cols"])
                np.testing.assert_array_equal(pred, arrays[method+"_pred"])
                trials.append(dict(**key, method=method, repeat=r, position=position, width=len(cols),
                                   p=X.shape[1], score=scores[method], select_ms=(t1-t0)/1e6,
                                   refit_ms=(t2-t1)/1e6, total_ms=(t2-t0)/1e6))
        np.savez_compressed(trace, **arrays)
    return trials, audit, width, scores["band_vote"]


def random_draws(X, y, Xt, yt, source, seed, width, vote):
    rows = []
    for draw in range(20):
        rng = np.random.default_rng([369479, source["dataset_id"], seed, draw])
        cols = np.sort(rng.permutation(X.shape[1])[:width])
        pred = prediction(X, y, Xt, cols, "regression", seed)
        score = metric(yt, pred, "regression")
        rows.append(dict(dataset=source["dataset"], fold=1, seed=seed, draw=draw, width=width, score=score,
                         vote_score=vote, vote_wins=vote > score + 1e-12, support_sha256=digest(cols),
                         prediction_sha256=digest(pred)))
    return rows


def finish(output, trials, metadata, draws):
    cases = summarize_timings(output, trials)
    info = pd.DataFrame(metadata)
    info.to_csv(output/"panels.csv", index=False)
    cc = cases.merge(info[["dataset", "eligible"]], on="dataset", validate="many_to_one")
    spec = dict(configs=("score","size"), reduced=("reduced","sum"), quality=("within","sum"),
                joint=("joint","sum"), joint5=("joint_5pct","sum"), joint10=("joint_10pct","sum"),
                median_ratio=("ratio","median"), width=("width","median"))
    cc.groupby(["eligible","method"]).agg(**spec).to_csv(output/"by-class.csv")
    vote = cc[cc.method == "band_vote"].set_index(CASE)
    comparisons = []
    for name, control in cc.groupby("method"):
        grp = control.set_index(CASE)
        delta = vote.score - grp.score
        result = vote[["eligible","width","joint","score","full_score","total_ms"]].copy()
        result["baseline"] = name
        result["delta_score"] = delta
        result["width_ratio"] = vote.width / grp.width
        result["time_ratio"] = vote.total_ms / grp.total_ms
        result["baseline_total_ms"] = grp.total_ms
        result["baseline_score"] = grp.score
        result["baseline_width"] = grp.width
        result["beats_score"] = delta > 1e-12
        result["noninferior_score"] = delta >= -.02
        result["faster"] = result.time_ratio < 1
        result["smaller_or_equal"] = vote.width <= grp.width
        result["within_10pct_width"] = vote.width <= 1.1 * grp.width
        result["joint_vs_baseline"] = (vote.joint & result.noninferior_score & result.faster &
                                       result.smaller_or_equal)
        result["joint_vs_baseline_10pct"] = (vote.joint & result.noninferior_score & result.faster &
                                             result.within_10pct_width)
        comparisons.append(result)
    comparison = pd.concat(comparisons)
    comparison.to_csv(output/"paired-baselines.csv")
    draws.to_csv(output/"random-draws.csv", index=False)
    results = vote.join(draws.groupby(CASE).agg(random_wins=("vote_wins","sum"), random_median=("score","median")))
    results["above_random_median"] = results.score > results.random_median + 1e-12
    results["strong_joint"] = results.joint & (results.random_wins >= 15)
    for flag in ("joint_vs_baseline", "joint_vs_baseline_10pct"):
        table = comparison.reset_index().pivot(index=CASE, columns="baseline", values=flag)
        results["F50_and_random50_"+flag] = table["F_50"] & table["random_50"]
    results.to_csv(output/"vote-comparisons.csv")


def run(out, verify):
    fingerprint = [Path(__file__), ROOT/"docs/minpi-study/next-vote-protocol.md",
                   ROOT/"src/minppi_mofs/band_vote_minpi.py", ROOT/"src/minppi_mofs/signal_vote_minpi.py",
                   ROOT/"src/minppi_mofs/budget_vote_minpi.py", ROOT/"src/minppi_mofs/native_pivotal_minpi.py",
                   ROOT/"experiments/fast_minpi_five_source.py", ROOT/"experiments/stable_minpi_stress.py",
                   ROOT/"experiments/minpi_study.py", ROOT/"experiments/compare_predictive_minpi.py",
                   ROOT/"configs/minpi-study.json", MANIFEST]
    protocol = dict(ids=list(IDS), seeds=list(SEEDS), arms=list(ARMS), repeats=REPEATS, environment=dict(
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                    native_library_sha256=file_hash(library_path()) if library_path().is_file() else None,
                    sha256={str(p.relative_to(ROOT)): file_hash(p) for p in fingerprint})
    if verify:
        assert protocol == json.loads((out/"protocol.json").read_text(encoding="utf8"))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"protocol.json", protocol)
    trials, audit, metadata, random = [], [], [], []
    for X, y, Xt, yt, info in sources():
        metadata.append(info)
        for seed in SEEDS:
            rows, checked, width, score = one_case(X, y, Xt, yt, info, seed, out, verify)
            trials.extend(rows)
            audit.extend(checked)
            random.extend(random_draws(X, y, Xt, yt, info, seed, width, score))
        print("Verified" if verify else "Measured", info["dataset"], flush=True)
    assert len(metadata) == 7 and len(audit) == 189 and len(random) == 420
    draws = pd.DataFrame(random)
    if verify:
        pd.testing.assert_frame_equal(draws, pd.read_csv(out/"random-draws.csv"), rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(pd.DataFrame(metadata), pd.read_csv(out/"panels.csv"))
        pd.DataFrame(audit).to_csv(out/"fresh-process-audit.csv", index=False)
        draws.to_csv(out/"fresh-process-random-draws.csv", index=False)
    else:
        assert len(trials) == REPEATS * len(audit)
        finish(out, trials, metadata, draws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
