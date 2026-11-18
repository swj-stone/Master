"""Recover a looser plain selection from the measured trajectory of its strict run."""

import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

from minppi_mofs.experiment_protocol import atomic_write_json


def _read(path):
    return json.loads(path.read_bytes())


def materialize(root, dataset, method, pct, out):
    """Publish a backend + timer, or return None if the first crossing was not saved."""
    if pct not in (5, 8):
        return None
    base = Path(root) / "stages" / dataset
    source = base / "backend" / "plain" / method / "tol-03"
    checkpoint_path = source / f"pass-tol-{pct:02d}.json"
    if not checkpoint_path.is_file():
        return None
    out = Path(out)
    timer_path = out / "timer.json"
    if timer_path.exists():
        return _read(timer_path)
    started = time.monotonic()
    checkpoint = _read(checkpoint_path)
    ref = _read(base / "reference" / "stage.json")
    k = checkpoint["k"]
    if (checkpoint["pct"] != pct or ref["dataset"] != dataset or ref["stage"] != "reference"
            or type(k) is not int or not 1 <= k <= len(ref["names"])):
        raise ValueError(f"{checkpoint_path}: invalid source identity or prefix width")
    names = checkpoint["names"]
    aux = checkpoint["selector_aux_fits"]
    ranking, checks = [], []
    fits = aux
    last_s = 0.0
    with (source / "prefix_trace.jsonl").open(encoding="utf-8") as handle:
        # A timeout may leave a partial line after k. Only the committed prefix matters.
        for i in range(1, k + 1):
            item = json.loads(handle.readline())
            ranking.append(item.pop("added"))
            fits += int(not item["reused"])
            scores = item["scores"]
            ratios = []
            for j, (base_score, candidate) in enumerate(zip(ref["scores"], scores, strict=True)):
                loss = candidate - base_score if dataset == "abalone" and j == 1 else base_score - candidate
                budget = base_score * (pct/100)
                ratios.append(loss / budget if budget > 0 else (2.0 if loss > 0 else 0.0))
            passed = max(ratios) <= 1 + 1e-12
            elapsed = item["elapsed_s"]
            if (item["k"] != i or fits != item["fits"] or not math.isfinite(elapsed) or elapsed < last_s
                    or len(scores) != 2 or not all(math.isfinite(s) for s in scores) or passed != (i == k)):
                raise ValueError(f"{checkpoint_path}: invalid measured trace at k={i}")
            item["accepted"] = passed
            checks.append(item)
            last_s = elapsed
    selected = set(ranking)
    if (len(selected) != k or names != [name for name in ref["names"] if name in selected]
            or len(names) != k or checkpoint["predictions"] != f"validation-k{k}.npz"
            or (checkpoint["fits"], checkpoint["elapsed_s"], checkpoint["scores"]) !=
               (fits, last_s, checks[-1]["scores"])):
        raise ValueError(f"{checkpoint_path}: crossing disagrees with measured rank/check ledger")
    origin = source.relative_to(root).as_posix()
    record = dict(dataset=dataset, stage="backend", method=method, arm="plain", pct=pct, k=k,
                  names=names, scores=checkpoint["scores"], fits=fits, checked=k, checks=checks,
                  prefixes=checks, ranking=ranking, ranking_prefix=ranking, selector_aux_fits=aux,
                  support_k=len(ref["names"]), shared_from=origin, checkpoint_s=last_s)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temp:
        archive = Path(temp) / "predictions.npz"
        shutil.copyfile(source / checkpoint["predictions"], archive)
        archive.replace(out / archive.name)
    record["elapsed_s"] = last_s + time.monotonic() - started
    atomic_write_json(out / "stage.json", record)
    atomic_write_json(out / "progress.json", dict(phase="completed", fits=fits, checked=k))
    materialization_s = time.monotonic() - started
    clock = dict(status="completed", elapsed_s=last_s + materialization_s, shared_from=origin,
                 checkpoint_s=last_s, materialization_s=materialization_s)
    atomic_write_json(timer_path, clock)
    return clock
