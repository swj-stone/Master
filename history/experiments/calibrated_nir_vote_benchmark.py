"""Paired end-to-end test of training-calibrated spectral voting, including new harvests and soil."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import nir_vote_benchmark as nir
from experiments.adaptive_minpi_transfer import Panel, run_case
from experiments.fast_minpi_five_source import summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, write_json
from minppi_mofs.calibrated_band_minpi import calibrated_band_top_from_training


ARMS = ("calibrated", "band_vote", "grid_match", "F_match", "random_match",
        "F_50", "random_50", "signal_vote", "full")
EXTRA_IDS = (16, 23)
SOIL_SHA = "b4476a144b981ab7ac4882e8e3462386bf5ca98b1fb393a2df7c4c9c37e92149"


def clean_split(name, source_id, X, y, Xt, yt, target):
    train = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = np.ascontiguousarray(X[train]), y[train]
    seen = {digest(row) for row in X}
    valid = np.isfinite(yt) & np.isfinite(Xt).all(axis=1)
    test = valid & np.array([digest(row) not in seen for row in Xt])
    info = dict(dataset=name, dataset_id=source_id, p=X.shape[1], n_train=len(y), n_test=int(test.sum()),
                drop_cross_test=int(valid.sum() - test.sum()), target=target, eligible=len(y) >= 512)
    assert X.shape[1] >= 100 and test.any()
    return X, y, np.ascontiguousarray(Xt[test]), yt[test], info


def all_sources():
    yield from nir.sources()
    records = pd.read_csv(nir.MANIFEST)
    for source_id in EXTRA_IDS:
        task = records[records.dataset.str.startswith(str(source_id) + "-")].set_index("split")
        data, columns = [], None
        for part in ("train", "test"):
            row = task.loc[part]
            path = nir.DATA / "nirbench" / row.dataset / Path(row.path).name
            assert file_hash(path) == row.sha256 and path.stat().st_size == row.bytes
            df = pd.read_csv(path)
            labels = nir.wavelengths(df, row.target_column, row.spectral_features)
            assert len(df) == row.rows and (columns is None or labels == columns)
            columns = labels
            data.extend([df[labels].to_numpy(dtype=float), df[row.target_column].to_numpy(dtype=float)])
        yield clean_split(row.dataset, source_id, *data, row.target_column)
    soil = nir.DATA / "nirsoil.json"
    assert file_hash(soil) == SOIL_SHA
    rows = json.loads(soil.read_text(encoding="utf8"))
    X = np.array([row["spc"] for row in rows], dtype=float)
    y = np.array([row.get("Ciso") for row in rows], dtype=float)
    train = np.array([row["train"] == 1 for row in rows])
    assert X.shape == (825, 700) and train.sum() == 618
    yield clean_split("NIRsoil", 101, X[train], y[train], X[~train], y[~train], "Ciso")


def select(method, panel, widths):
    if method == "calibrated":
        return calibrated_band_top_from_training(panel.X, panel.y, task="regression",
                                                 random_state=panel.seed)[0]
    return nir.select(method, panel.X, panel.y, panel.seed, widths.get("match"))[0]


def run(out, verify):
    core_files = [Path(__file__), Path(nir.__file__), ROOT / "docs/minpi-study/calibrated-band-protocol.md",
                  ROOT / "src/minppi_mofs/calibrated_band_minpi.py", ROOT / "src/minppi_mofs/band_vote_minpi.py",
                  ROOT / "experiments/adaptive_minpi_transfer.py", nir.MANIFEST]
    state = dict(methods=list(ARMS), seeds=list(nir.SEEDS), soil_sha256=SOIL_SHA, extra=list(EXTRA_IDS),
                 source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in core_files})
    manifest = out / "protocol.json"
    if verify:
        assert state == json.loads(manifest.read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(manifest, state)
    trials, audits, draws, sources = [], [], [], []
    for X, y, Xt, yt, info in all_sources():
        sources.append(info)
        for seed in nir.SEEDS:
            panel = Panel(info["dataset"], "spectra", "ordered wavelength", "regression", 1, seed, X, y, Xt, yt, {})
            trace = out / f"{panel.name}-{seed}.npz"
            records, audited = run_case(panel, [94651, info["dataset_id"], seed], len(ARMS), trace,
                                        verify, ARMS, select, {"calibrated": "match"})
            trials.extend(records)
            audits.extend(audited)
            with np.load(trace) as stored:
                width = len(stored["calibrated_cols"])
                assert width == len(stored["grid_match_cols"])
                score = metric(yt, stored["calibrated_pred"], "regression")
            draws.extend(nir.random_draws(X, y, Xt, yt, info, seed, width, score))
        print(info["dataset"], "verified" if verify else "measured", flush=True)
    random_frame = pd.DataFrame(draws)
    assert len(draws) == 600
    if verify:
        pd.testing.assert_frame_equal(random_frame, pd.read_csv(out / "random-draws.csv"), rtol=0, atol=1e-12)
        assert len(audits) == 270
        pd.DataFrame(audits).to_csv(out / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == 2430
        summarize_timings(out, trials)
        pd.DataFrame(sources).to_csv(out / "panels.csv", index=False)
        random_frame.to_csv(out / "random-draws.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
