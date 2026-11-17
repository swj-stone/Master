"""Independent all-mask validation of the locked supervised-loss forecasts."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import validation_predictive as panel


def all_mask_risks(increment, incidence, centered_target, supports):
    """Enumerate predictions at every mask using subset sums, without Gram or hypergeometric weights."""
    n, paths = increment.shape
    p = incidence.shape[1]
    mask = np.arange(1 << p)
    sizes = np.fromiter((int(i).bit_count() for i in mask), dtype=int, count=len(mask))
    path_bits = incidence @ (1 << np.arange(p))
    if len(path_bits) != paths or p > 18:
        raise ValueError("Wrong path basis or impractical enumeration")
    sums = {key: np.zeros(p + 1) for key in supports}
    gather = {key: mask & sum(1 << int(j) for j in key) for key in supports}
    for start in range(0, n, 8):
        stop = min(n, start + 8)
        rows = increment[start:stop].tocoo()
        pred = np.zeros((stop - start, len(mask)))
        np.add.at(pred, (rows.row, path_bits[rows.col]), rows.data)
        for bit in range(p):
            step = 1 << bit
            pairs = pred.reshape(len(pred), -1, 2 * step)
            pairs[:, :, step:] += pairs[:, :, :step]
        loss = (pred - centered_target[start:stop, None]) ** 2
        full = loss.sum(axis=0)
        for key, selector in gather.items():
            sums[key] += np.bincount(sizes, weights=full[selector] - full, minlength=p + 1)
    norm = np.dot(centered_target, centered_target)
    scale = (p / np.arange(1, p + 1)) / (np.bincount(sizes)[1:] * norm)
    return {key: value[1:] * scale for key, value in sums.items()}


def discrepancy(archive, scores, name, seed):
    key = f"{name}-{seed}"
    def raw(member):
        return archive.read(f"{key}/{member}")
    receipt = json.loads(raw("selection.json"))
    lock = json.loads(archive.read("lock.json"))
    assert sha256(raw("selection.json")).hexdigest() == lock["receipts"][key]
    for file, digest in receipt["files"].items():
        assert sha256(raw(file)).hexdigest() == digest
    x, y, task, idx = panel._partition(name, seed)
    assert receipt["idx"] == {a: b.tolist() for a, b in idx.items()}
    test = y[idx["test"]]
    offset = np.array(receipt["empty"])
    target = np.eye(len(offset))[test] if task == "classification" else test[:, None]
    centered = (target - offset).ravel()
    by_arm = {arm: tuple(sorted(info["support"])) for arm, info in receipt["records"].items()}
    with np.load(BytesIO(raw("forecasts.npz"))) as archived:
        errs = []
        for arm in by_arm:
            pred = archived[arm]
            row = scores.loc[(name, seed, arm)]
            if task == "classification":
                aucs = []
                for c in range(pred.shape[1]):
                    diff = pred[test == c, c][:, None] - pred[test != c, c]
                    aucs.append(((diff > 0).sum() + .5 * (diff == 0).sum()) / diff.size)
                metrics = [abs(row.accuracy - accuracy_score(test, pred.argmax(axis=1))),
                           abs(row.auc - np.mean(aucs))]
            else:
                metrics = [abs(row.r2 - r2_score(test, pred)), abs(row.mae - mean_absolute_error(test, pred))]
            errs.extend(metrics)
    risk_error = np.nan
    if seed == 42:
        a = sparse.load_npz(BytesIO(raw("increments.npz"))).tocsr()
        incidence = sparse.load_npz(BytesIO(raw("incidence.npz"))).tocsr()
        curves = all_mask_risks(a, incidence, centered, set(by_arm.values()))
        error = []
        for arm, support in by_arm.items():
            row = scores.loc[(name, seed, arm)]
            r = curves[support]
            error.extend([abs(row.outer_worst - r.max()), abs(row.outer_intact - r[-1]),
                          abs(row.outer_mean - r.mean()), abs(row.outer_positive - max(r.max(), 0))])
        risk_error = max(error)
    return dict(source=name, seed=seed, arms=len(by_arm), clean_error=max(errs), risk_error=risk_error)


def check(bundle, evidence):
    expect = (evidence / "forecast-sha256.txt").read_text().split()[0]
    assert sha256(bundle.read_bytes()).hexdigest() == expect
    scores = pd.read_csv(evidence / "scores.csv").set_index(["dataset", "seed", "method"])
    rows = []
    with ZipFile(bundle) as archive:
        for name in panel.SOURCES:
            for seed in panel.SEEDS:
                rows.append(discrepancy(archive, scores, name, seed))
    audit = pd.DataFrame(rows)
    assert len(audit) == 35 and audit.risk_error.notna().sum() == 7
    assert audit.clean_error.max() < 1e-10 and audit.risk_error.max() < 1e-10
    audit.to_csv(evidence / "independent-mask-audit.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        check(args.bundle, args.evidence)
