"""Frozen evaluation of vote-dependent allocation of spectral channels across local bands."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import calibrated_nir_vote_benchmark as previous
from experiments import nir_vote_benchmark as nir
from experiments.adaptive_minpi_transfer import Panel, run_case
from experiments.fast_minpi_five_source import summarize_timings
from experiments.stable_minpi_stress import ROOT, file_hash, metric, write_json
from minppi_mofs.allocation_vote_minpi import allocation_vote_top_from_training


ARMS = ("allocation", "band_vote", "grid_match", "F_match", "random_match", "F_50",
        "random_50", "signal_vote", "full")
PRIOR_IDS = set(nir.IDS) | set(previous.EXTRA_IDS) | {101}


def additional_season(source_id):
    manifest = pd.read_csv(nir.MANIFEST)
    rows = manifest[manifest.dataset.str.startswith(f"{source_id}-")].set_index("split")
    arrays, columns = [], None
    for split in ("train", "test"):
        row = rows.loc[split]
        path = nir.DATA / "nirbench" / row.dataset / Path(row.path).name
        assert path.stat().st_size == int(row.bytes) and file_hash(path) == row.sha256
        frame = pd.read_csv(path)
        names = nir.wavelengths(frame, row.target_column, row.spectral_features)
        assert len(frame) == row.rows and (columns is None or columns == names)
        columns = names
        x = frame[names].to_numpy(dtype=np.float64)
        x[x == 0] = 0.
        arrays.extend([x, frame[row.target_column].to_numpy(dtype=np.float64)])
    return previous.clean_split(row.dataset, source_id, *arrays, row.target_column)


def sources():
    yield from previous.all_sources()
    for source_id in (14, 24):
        yield additional_season(source_id)


def select(method, panel, widths):
    if method == "allocation":
        return allocation_vote_top_from_training(panel.X, panel.y, task=panel.task, random_state=panel.seed)[0]
    return nir.select(method, panel.X, panel.y, panel.seed, widths.get("match"))[0]


def comparisons(out, trials, draws, metadata):
    cases = summarize_timings(out, trials)
    metadata.to_csv(out/"panels.csv", index=False)
    draws.to_csv(out/"random-draws.csv", index=False)
    cases = cases.merge(metadata[["dataset", "eligible", "validation_set"]], on="dataset", validate="many_to_one")
    vote = cases[cases.method == "allocation"].set_index(nir.CASE)
    paired = []
    for arm, frame in cases.groupby("method"):
        control = frame.set_index(nir.CASE)
        r = vote[["eligible", "validation_set", "score", "width", "total_ms", "joint"]].copy()
        r["baseline"] = arm
        r["score_difference"] = vote.score - control.score
        r["width_ratio"] = vote.width / control.width
        r["time_ratio"] = vote.total_ms / control.total_ms
        suitable = vote.joint & (r.score_difference >= -.02) & (r.time_ratio < 1)
        r["joint_vs_baseline"] = suitable & (r.width_ratio <= 1)
        r["joint_vs_baseline_10pct"] = suitable & (r.width_ratio <= 1.1)
        paired.append(r)
    paired = pd.concat(paired).reset_index()
    paired.to_csv(out/"paired-baselines.csv", index=False)
    comparison = vote.join(draws.groupby(nir.CASE).agg(random_wins=("vote_wins","sum"),
                                                        random_median=("score","median")))
    for flag in ("joint_vs_baseline", "joint_vs_baseline_10pct"):
        table = paired.pivot(index=nir.CASE, columns="baseline", values=flag)
        comparison["F50_and_random50_"+flag] = table.F_50 & table.random_50
    comparison["strong_joint"] = comparison.joint & (comparison.random_wins >= 15)
    comparison.to_csv(out/"vote-comparisons.csv")


def run(out, verify):
    files = (Path(__file__), Path(nir.__file__), Path(previous.__file__), nir.MANIFEST,
             ROOT/"src/minppi_mofs/allocation_vote_minpi.py", ROOT/"src/minppi_mofs/band_vote_minpi.py",
             ROOT/"src/minppi_mofs/calibrated_band_minpi.py", ROOT/"src/minppi_mofs/masked_loss_minpi.py",
             ROOT/"src/minppi_mofs/budget_vote_minpi.py", ROOT/"src/minppi_mofs/signal_vote_minpi.py",
             ROOT/"src/minppi_mofs/native_pivotal_minpi.py", ROOT/"experiments/adaptive_minpi_transfer.py",
             ROOT/"experiments/minpi_study.py", ROOT/"experiments/stable_minpi_stress.py",
             ROOT/"experiments/compare_predictive_minpi.py", ROOT/"configs/minpi-study.json",
             ROOT/"docs/minpi-study/allocation-vote-protocol.md")
    protocol = {"sha256":{p.relative_to(ROOT).as_posix():file_hash(p) for p in files},
                "native_sha256":file_hash(nir.library_path()) if nir.library_path().is_file() else None,
                "methods":list(ARMS), "seeds":list(nir.SEEDS), "original_ids":list(nir.IDS),
                "calibration_transfer":list(previous.EXTRA_IDS) + [101], "new_ids":[14,24],
                "publisher_soil_sha256":previous.SOIL_SHA}
    if verify:
        assert protocol == json.loads((out/"protocol.json").read_text(encoding="utf8"))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"protocol.json", protocol)
    trials, audits, draws, meta = [], [], [], []
    for X, y, Xt, yt, info in sources():
        source_id = info["dataset_id"]
        stage = "first seven" if source_id in nir.IDS else ("previous confirmation" if source_id in PRIOR_IDS
                                                           else "new allocation confirmation")
        meta.append(dict(info, validation_set=stage))
        for seed in nir.SEEDS:
            panel = Panel(info["dataset"], "spectra", "wavelength", "regression", 1, seed, X, y, Xt, yt, {})
            trace = out/f"{panel.name}-{seed}.npz"
            rows, audit = run_case(panel, [15811, source_id, seed], len(ARMS), trace, verify,
                                   ARMS, select, {"allocation":"match"})
            trials.extend(rows)
            audits.extend(audit)
            with np.load(trace) as saved:
                width = len(saved["allocation_cols"])
                assert all(len(saved[arm+"_cols"]) == width for arm in ("grid_match","F_match","random_match"))
                score = metric(yt, saved["allocation_pred"], panel.task)
            draws.extend(nir.random_draws(X, y, Xt, yt, info, seed, width, score))
        print("Verified" if verify else "Measured", info["dataset"], flush=True)
    assert len(meta) == len(set(protocol["original_ids"] + protocol["calibration_transfer"] + protocol["new_ids"]))
    n = len(meta) * len(nir.SEEDS)
    assert len(draws) == n * 20
    draws, meta = pd.DataFrame(draws), pd.DataFrame(meta)
    if verify:
        assert len(audits) == n * len(ARMS)
        pd.testing.assert_frame_equal(draws, pd.read_csv(out/"random-draws.csv"), rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(meta, pd.read_csv(out/"panels.csv"), check_dtype=False)
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == n * len(ARMS)**2
        comparisons(out, trials, draws, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
