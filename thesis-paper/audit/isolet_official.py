"""Verify the official speaker-group split from the UCI ZIP and all four fitted arms."""

import argparse
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd

from verify import HERE, MAIN, ROOT, RUNS, audit_arm, check, read, sha

ARCHIVE_HASH = "fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053"
DEFAULT_ARCHIVE = ROOT / "tmp/isolet-uci-official-20261115.zip"
CAMPAIGN = RUNS / "context-vote-isolet-official-20261115"
NAMES = ("isolet1+2+3+4.data.Z", "isolet5.data.Z")


def official_arrays(archive):
    check(sha(archive) == ARCHIVE_HASH, archive)
    decompressor = shutil.which("uncompress") or shutil.which("gzip")
    check(decompressor, "Install a compress-compatible uncompress or gzip.")
    flag = "-c" if Path(decompressor).stem.lower() == "uncompress" else "-dc"
    arrays = []
    with ZipFile(archive) as bundle:
        for n in NAMES:
            payload = subprocess.check_output(
                [decompressor, flag], input=bundle.read(n)
            )
            arrays.append(
                pd.read_csv(
                    BytesIO(payload), header=None, skipinitialspace=True
                ).to_numpy()
            )
    training, test = arrays
    check(
        training.shape == (6238, 618) and test.shape == (1559, 618),
        [a.shape for a in arrays],
    )
    # 30 speakers, each asked for two examples of every letter; two of the
    # first four groups' expected 6240 records are missing.
    cut_candidates = []
    for cut in range(4678, 4681):
        y = training[cut:, -1].astype(int) - 1
        if max(np.bincount(y, minlength=26)) <= 60:
            cut_candidates.append(cut)
    check(cut_candidates == [4680], cut_candidates)
    matrix = np.concatenate(arrays)
    original = matrix[:, -1].astype(int)
    # The local CSV uses lexicographical factorization of the numeric letter IDs.
    classes = sorted(set(map(str, original)))
    labels = np.array([classes.index(str(y)) for y in original], dtype=int)
    names = [f"f{i}" for i in range(1, 618)]
    local = pd.read_csv(ROOT / "data/raw/isolet.csv")
    check(list(local.columns) == names + ["letter"], "feature names")
    np.testing.assert_array_equal(local[names], matrix[:, :-1])
    np.testing.assert_array_equal(local.letter, labels)
    return (
        names,
        matrix[:, :-1],
        labels,
        dict(
            train=np.arange(4680),
            val=np.arange(4680, len(training)),
            test=np.arange(len(training), len(matrix)),
        ),
    )


def verify(archive):
    names, x, y, indices = official_arrays(archive)
    input_dir = CAMPAIGN / "inputs/isolet"
    proto = read(CAMPAIGN / "protocol.json")
    meta = read(input_dir / "metadata.json")
    check(
        sha(archive) == proto["zip_sha256"]
        and sha(ROOT / "data/raw/isolet.csv") == proto["local_csv_sha256"],
        archive,
    )
    check(
        sha(input_dir / "metadata.json") == proto["metadata_sha256"]
        and sha(MAIN / "protocol.json") == proto["primary_protocol_sha256"],
        CAMPAIGN,
    )
    check(all(sha(ROOT / k) == v for k, v in proto["source"].items()), CAMPAIGN)
    check(meta["features"] == names and meta["width"] == 617, input_dir)
    training = x[indices["train"]]
    mean = training.mean(axis=0)
    scale = training.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    for part, rows in indices.items():
        np.testing.assert_array_equal(
            np.load(input_dir / (part + "_indices.npy")), rows
        )
        np.testing.assert_array_equal(np.load(input_dir / (part + "_y.npy")), y[rows])
        np.testing.assert_allclose(
            np.load(input_dir / (part + ".npy")),
            (x[rows] - mean) / scale,
            rtol=1e-11,
            atol=1e-12,
        )
    frame = pd.read_csv(CAMPAIGN / "per-seed.csv")
    check(
        len(frame) == 4
        and set(frame.arm) == {"min", "gain", "matched_gain", "full"}
        and all(frame.seed == 42)
        and all(frame.status == "completed")
        and frame.wall_s.sum() < 3600,
        CAMPAIGN,
    )
    records = {}
    supports = {}
    correct = {}
    for row in frame.itertuples():
        folder = CAMPAIGN / "runs/seed-42/isolet" / row.arm
        check(row.run_folder == folder.relative_to(CAMPAIGN).as_posix(), folder)
        records[row.arm], supports[row.arm] = audit_arm(
            folder, row, row.arm == "min", input_dir
        )
        with np.load(folder / "predictions.npz") as pred:
            correct[row.arm] = pred["test_pred"].argmax(axis=1) == pred["y_test"]
    donor = supports["min"]
    k = len(donor["support"])
    check(
        supports["gain"]["support"] == set(donor["estimator"]["anchor_order"]), CAMPAIGN
    )
    check(
        supports["matched_gain"]["support"] == set(donor["estimator"]["order"][:k]),
        CAMPAIGN,
    )
    check(supports["full"]["support"] == set(names), CAMPAIGN)
    check(
        supports["matched_gain"]["estimator"]["donor_hash"] == donor["frozen_hash"],
        CAMPAIGN,
    )
    for arm, record in records.items():
        record["utterances_only_min_correct"] = int(
            (correct["min"] & ~correct[arm]).sum()
        )
        record["utterances_only_comparator_correct"] = int(
            (correct[arm] & ~correct["min"]).sum()
        )
    out = pd.DataFrame(list(records.values()))
    out.to_csv(HERE / "isolet-official.csv", index=False)
    print(
        out[["arm", "width", "k", "auc", "accuracy", "selector_s", "wall_s"]].to_string(
            index=False
        )
    )
    print(
        "Verified official file membership and exact 7797 x 617 features; total cold seconds",
        frame.wall_s.sum(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    verify(parser.parse_args().archive)
