"""Four-fold development-only attribution of one fixed-width minPI block swap."""
import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as benchmark
from experiments.fast_document_vote_benchmark import pooled_binary_gain
from experiments.stable_minpi_stress import file_hash, write_json
from minppi_mofs.conditional_block_document_minpi import (
    _partition_rows, conditional_block_document_top_from_training,
)
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training, library_path


ROOT = benchmark.ROOT
DEVELOPMENT = {"sms", "amazon_cells", "imdb", "yelp"}


def run(output):
    files = [Path(__file__), Path(benchmark.__file__), library_path(),
             ROOT / "docs/minpi-study/conditional-block-development-protocol.md",
             ROOT / "docs/superpowers/plans/2026-11-17-minpi-combination-search.md"]
    files.extend(ROOT / "src/minppi_mofs" / name for name in (
        "conditional_block_document_minpi.py", "fast_document_vote_minpi.py", "document_vote_minpi.py",
        "document_vote_kernel.c", "signal_vote_minpi.py", "sketch_minpi.py", "utility_minpi.py"))
    files.extend(ROOT / "experiments" / name for name in (
        "minpi_constant_transfer.py", "fast_document_vote_benchmark.py", "compare_predictive_minpi.py"))
    hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", dict(
        source_sha256=hashes, role="development only: four pre-existing fold-1 configurations",
        fixed_width="old full-training document support size; conditional-on-k isolation only",
        acceptance=["global", "local"], ba_margin=.02, exchange_budget=8,
        max_calibration=256, max_pilot=512, teacher_max_iter=8,
        timing="single-call nonformal diagnostics, not evidence of acceleration",
        raw_assets=benchmark.ASSETS,
    ))
    results, diagnostics = [], []
    for X, y, Xt, truth, meta in benchmark.panels():
        if meta["fold"] != 1 or meta["dataset"] not in DEVELOPMENT:
            continue
        seed = meta["seed"]
        start = perf_counter_ns()
        old, _ = fast_document_vote_top_from_training(X, y, random_state=seed)
        width_ms = (perf_counter_ns()-start)/1e6
        count = len(old)
        proposal, _ = _partition_rows(y, seed)
        selections = {"old_all": (old, width_ms, {}), "full": (np.arange(X.shape[1]), 0., {})}
        for rule in ("global", "local"):
            start = perf_counter_ns()
            cols, info = conditional_block_document_top_from_training(X, y, count, random_state=seed,
                                                                      acceptance=rule)
            selections[rule] = cols, (perf_counter_ns()-start)/1e6, info
        first, second = selections["global"][2], selections["local"][2]
        for field in ("initial_support", "proposed_support", "teacher_coefficients",
                      "replacement_coefficients", "calibration_endpoints"):
            np.testing.assert_array_equal(first[field], second[field])
        start = perf_counter_ns()
        initial, _ = fast_document_vote_top_from_training(X[proposal], y[proposal], count, random_state=seed)
        selections["proposal_only"] = initial, (perf_counter_ns()-start)/1e6, {}
        np.testing.assert_array_equal(initial, first["initial_support"])
        for scope, rows in (("proposal", proposal), ("all", np.arange(len(y)))):
            for rule in ("F", "pooled"):
                start = perf_counter_ns()
                if rule == "F":
                    cols = benchmark.select("F_match", X[rows], y[rows], seed, count)
                else:
                    gain = pooled_binary_gain(X[rows], y[rows])
                    cols = np.sort(np.argsort(-gain, kind="stable")[:count])
                selections[rule+"_"+scope] = cols, (perf_counter_ns()-start)/1e6, {}
        start = perf_counter_ns()
        cols = benchmark.select("random_match", X, y, seed, count)
        selections["random"] = cols, (perf_counter_ns()-start)/1e6, {}
        archive = {}
        for method, (cols, elapsed, info) in selections.items():
            start = perf_counter_ns()
            pred = benchmark.fit_predict(X, y, Xt, cols, "classification", seed, False)
            refit_ms = (perf_counter_ns()-start)/1e6
            row = dict(dataset=meta["dataset"], fold=1, seed=seed, method=method, width=len(cols), p=X.shape[1],
                       score=benchmark.balanced_accuracy_score(truth, pred),
                       select_ms_nonformal=elapsed, refit_ms_nonformal=refit_ms,
                       total_ms_nonformal=elapsed+refit_ms,
                       old_full_width_cost_ms=width_ms,
                       total_with_width_ms_nonformal=elapsed+refit_ms+(width_ms if method in ("global", "local") else 0.))
            results.append(row)
            archive[method+"_cols"], archive[method+"_pred"] = cols, pred
            if info:
                np.testing.assert_allclose(info["old_teacher_loss"]-info["new_teacher_loss"],
                                           info["calibration_endpoints"][-1], rtol=0, atol=1e-12)
                diagnostics.append(dict(dataset=meta["dataset"], method=method,
                    proposed_swaps=len(info["incoming"]), accepted_swaps=len(info["incoming"]) if info["accepted"] else 0,
                    accepted=info["accepted"], loss_accepted=info["loss_accepted"], ba_gate=info["ba_gate"],
                    proposal_rows=len(info["proposal_rows"]), pilot_rows=len(info["pilot_rows"]),
                    calibration_rows=len(info["calibration_rows"]), common_columns=len(info["common"]),
                    global_gain=info["global_gain"], local_gain=info["local_gain"],
                    old_teacher_ba=info["old_teacher_ba"], new_teacher_ba=info["new_teacher_ba"],
                    old_teacher_loss=info["old_teacher_loss"], new_teacher_loss=info["new_teacher_loss"]))
                archive.update({method+"_"+key: value for key, value in info.items() if isinstance(value, np.ndarray)})
        with np.load(ROOT / "results/minpi-study/fast-document-20261117a" /
                     f"{meta['dataset']}-1-{seed}.npz") as saved:
            np.testing.assert_array_equal(archive["old_all_cols"], saved["fast_vote_cols"])
            np.testing.assert_array_equal(archive["old_all_pred"], saved["fast_vote_pred"])
            np.testing.assert_array_equal(archive["full_pred"], saved["full_pred"])
        np.savez_compressed(output / f"{meta['dataset']}-1-{seed}.npz", **archive)
        print(meta["dataset"], "old_k", count, "accepted", first["accepted"], second["accepted"], flush=True)
    assert len(diagnostics) == 8 and len(results) == 40
    assert hashes == {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}
    pd.DataFrame(results).to_csv(output / "cases.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output)
