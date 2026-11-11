"""Compare two feature spaces on Madelon seed 42, with one timed K=2 run per arm."""

import argparse
import hashlib
import multiprocessing
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from experiments.run_thesis_sampling import MSK, ROOT, _read, _save, supervise, write_summary


LIMIT_S = 900
WIDTHS = (100, 500)
RUNS = ROOT / "results" / "test" / "thesis_sampling"


def _worker(width, directory, alpha):
    with (directory / "worker.log").open("w", encoding="utf-8", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                from experiments.test.thesis_madelon_alpha_worker import run
                run(width, directory, alpha=alpha)
            except Exception:
                traceback.print_exc(file=stream)
                raise SystemExit(1)


def _heartbeat(progress, elapsed, remaining):
    sampling_eta = progress.get("sampling_eta_s")
    eta = "pending" if sampling_eta is None else f"~{sampling_eta:.0f}s"
    return (f"phase={progress.get('phase', 'starting')} | round={progress.get('round', 0)}/2"
            f" | draws={progress.get('sampled_draws', 0)} | certified={progress.get('witness_count', 0)}"
            f" | sample fits={progress.get('sample_fits', 0)} | elapsed={elapsed:.0f}s"
            f" | sampling ETA={eta} | deadline in {remaining:.0f}s")


def _row(directory, width, outcome, *, alpha=.99):
    finished = outcome["status"] == "completed"
    data = outcome["result"] if finished else outcome["progress"]
    new = data.get("new_negative_by_round", {})
    rounds = [len(new[str(k)]) if str(k) in new else None for k in (1, 2)]
    if finished:
        if data["alpha"] != alpha or data["width"] != width:
            raise ValueError("Worker returned a result for different experiment settings")
        if None in rounds or sum(rounds) != len(set(new["1"]) | set(new["2"])):
            raise ValueError(f"Invalid round-specific witness counts for width {width}")
        if data["total_fits"] < data["sample_fits"] or not 1 <= data["retained_count"] <= width:
            raise ValueError(f"Invalid model fit or subset counts for width {width}")
        if not 0 <= data["retained_outside_top100_count"] <= min(width-100, data["retained_count"]):
            raise ValueError(f"Invalid count outside frozen Top-100 for width {width}")
        if not all(0 <= data[k] <= 1 for k in ("full_test_auc", "selected_test_auc")):
            raise ValueError(f"Invalid test AUC for width {width}")
    else:
        for k in (1, 2):
            checkpoint = _read(directory / f"k-{k}" / "sampling.json")
            if checkpoint.get("new_negative_features") is not None:
                rounds[k-1] = len(checkpoint["new_negative_features"])
    return dict(width=width, seed=42, alpha=alpha, k=2, status=outcome["status"], phase=data.get("phase"),
                wall_s=round(outcome["elapsed_s"], 3), limit_s=LIMIT_S,
                new_negative_k1=rounds[0], new_negative_k2=rounds[1],
                sampled_draws=data.get("sampled_draws"), sample_fits=data.get("sample_fits"),
                total_fits=data.get("total_fits"), positive_count=data.get("positive_count"),
                noise_fits=data.get("noise_fits"), retained_count=data.get("retained_count") if finished else None,
                retained_outside_top100=data.get("retained_outside_top100_count") if finished else None,
                full_test_auc=data.get("full_test_auc") if finished else None,
                selected_test_auc=data.get("selected_test_auc") if finished else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, choices=(.95, .99), default=.99)
    parser.add_argument("--widths", type=int, choices=WIDTHS, nargs="+", default=WIDTHS)
    args = parser.parse_args(argv)
    if len(set(args.widths)) != len(args.widths):
        parser.error("Each feature width may run only once.")
    # Check both arms' common dependencies before either arm consumes its trial.
    from experiments.test.thesis_madelon_alpha_worker import run
    if not callable(run):
        raise TypeError("Missing Madelon worker")
    alpha_tag = f"{args.alpha:.2f}".replace(".", "")
    tag = datetime.now(MSK).strftime(f"madelon-s42-alpha{alpha_tag}-k2-%Y%m%d-%H%M%S-%f")
    root = RUNS / tag
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    files = [Path(__file__), *(source / f"thesis_madelon_alpha_{part}.py" for part in ("data", "noise", "worker")),
             *(ROOT / "experiments" / name for name in ("thesis_sampling_worker.py", "thesis_nested_k_worker.py",
                                                          "run_thesis_sampling.py")),
             *(ROOT / "src/minppi_mofs" / name for name in ("guarded_update.py", "sign_aware_removal.py",
                                                            "minimum_sampling.py", "removal_selection.py"))]
    hashes = {str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    _save(root / "protocol.json", dict(dataset="madelon", seed=42, alpha=args.alpha, k=2, widths=args.widths,
          arms_per_width=1, limit_s_per_width=LIMIT_S, predictor_rounds=100, feature_selection="guarded",
          positive_check="diagnostic_only", source_sha256=hashes))

    def log(message):
        line = f"{datetime.now(MSK):%Y-%m-%d %H:%M:%S} MSK | {message}"
        print(line, flush=True)
        with (root / "run.log").open("a", encoding="utf-8") as file:
            print(line, file=file)

    log(f"Madelon seed=42 alpha={args.alpha} K=2 | limit={LIMIT_S}s per arm | output={root}")
    rows = []
    for width in args.widths:
        directory = root / f"width-{width}"
        directory.mkdir()
        process = multiprocessing.get_context("spawn").Process(target=_worker, args=(width, directory, args.alpha))
        log(f"width={width} | starting")
        try:
            outcome = supervise(process, directory, LIMIT_S, lambda s: log(f"width={width} | {s}"),
                                heartbeat_formatter=_heartbeat)
        finally:
            process.close()
        row = _row(directory, width, outcome, alpha=args.alpha)
        rows.append(row)
        write_summary(root, rows)
        log(f"width={width} | {row['status']} | wall={row['wall_s']}s | selected AUC={row['selected_test_auc']}")
    log(f"summary={root / 'summary.csv'}")
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
