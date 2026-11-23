"""Verify source-to-panel provenance offline and list the original data to retain."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_splits():
    from research.text_peer import bench
    from research.lowdim.text_study import sparse_digest

    metadata = {r["key"]: r for r in json.loads((bench.CACHE / "panels.json").read_text())}
    X, y, Xt, truth, source = bench.mail_vote_transfer.prepare()
    mail = [(X, y, Xt, truth, dict(source, seed=1129))]
    seen = set()
    for sequence in (bench.original.panels(), bench.refined_document_external.panels(),
                     bench.reliability_mail_external.panels(), mail, bench.cornell()):
        for X, y, Xt, truth, meta in sequence:
            key = f"{meta['dataset']}-{meta['fold']}"
            if key in seen:
                raise ValueError(f"Repeated split {key}")
            seen.add(key)
            for suffix, matrix in (("-X", X), ("-Xt", Xt)):
                saved = sparse.load_npz(bench.CACHE / f"{key}{suffix}.npz")
                if sparse_digest(matrix) != sparse_digest(saved):
                    raise ValueError(f"Text matrix mismatch on {key}{suffix}")
            with np.load(bench.CACHE / f"{key}-labels.npz") as labels:
                np.testing.assert_array_equal(y, labels["y"])
                np.testing.assert_array_equal(truth, labels["gold"])
            for name, value in meta.items():
                if name == "seed" or name.endswith("sha256"):
                    if metadata[key][name] != value:
                        raise ValueError(f"Text metadata mismatch on {key}, {name}")
    if seen != metadata.keys() or len(seen) != 30:
        raise ValueError("Incomplete regeneration of cached text splits")
    return len(seen)


def csr_keys(matrix):
    indices = matrix.indices.astype(np.int32, copy=False)
    for first, last in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True):
        row_hash = hashlib.sha256(indices[first:last].tobytes())
        row_hash.update(matrix.data[first:last].tobytes())
        yield row_hash.digest()


def real_sim():
    from research.fast_single import transfer
    from research.wide_blocks import panels

    base = HISTORY / "data/raw/full-scale-trials"
    source = base / "real-sim.bz2"
    meta = json.loads((transfer.REAL / "manifest.json").read_text())
    if digest(source) != meta["source_sha256"]:
        raise ValueError("Wrong LIBSVM source file")
    raw, labels = load_svmlight_file(str(source))
    labels = (labels > 0).astype(np.int64)
    case = transfer.case_for("real_sim", meta["group_seed"])
    if list(panels.input_hashes(case)) != meta["input_hashes"]:
        raise ValueError("Frozen Real-Sim input hashes mismatch")
    train_keys = set(csr_keys(case.X))
    test_keys = set(csr_keys(case.Xt))
    if train_keys & test_keys or len(train_keys | test_keys) != meta["unique_groups"]:
        raise ValueError("Unexpected Real-Sim group partition")
    mask = np.asarray([row in train_keys for row in csr_keys(raw)])
    # Row order and response recoding are verified before preserving the split.
    for take, expected_matrix, expected_y in ((mask, case.X, case.y), (~mask, case.Xt, case.truth)):
        actual = raw[take]
        np.testing.assert_array_equal(actual.indptr, expected_matrix.indptr)
        np.testing.assert_array_equal(actual.indices, expected_matrix.indices)
        np.testing.assert_array_equal(actual.data, expected_matrix.data)
        np.testing.assert_array_equal(labels[take], expected_y)
    return meta["unique_groups"]


def mhealth(scratch):
    from experiments import minpi_study_device as device
    fresh = scratch / "mhealth.npz"
    device.extract(device.DATA / "mhealth.zip", fresh)
    with np.load(fresh) as actual, np.load(device.DATA / "mhealth-windows.npz") as old:
        if set(actual.files) != set(old.files):
            raise ValueError("MHEALTH field list differs")
        for field in actual.files:
            np.testing.assert_array_equal(actual[field], old[field])
        return len(old["ids"])


def tablet(scratch):
    from experiments import prepare_nir_tablets as prep
    from shutil import copyfile
    original = prep.ROOT
    prep.ROOT = scratch
    try:
        for name in ("nirbench-manifest.csv", "nirbench-tablets.zip"):
            copyfile(original / name, scratch / name)
        prep.main()
        files = list((scratch / "nirbench").rglob("*.csv"))
        for path in files:
            if digest(path) != digest(original / path.relative_to(scratch)):
                raise ValueError(f"Tablet mismatch on {path.name}")
        if len(files) != 2:
            raise ValueError("Incomplete tablet split")
        return len(files)
    finally:
        prep.ROOT = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    paths = {}
    def retain(phase, names):
        for name in names:
            path = HISTORY / name
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.setdefault(path, set()).add(phase)

    retain("text-cache", [
        *[f"data/raw/text-{source}.zip" for source in ("sms", "youtube", "sentences")],
        *[f"data/raw/20030228_{name}.tar.bz2" for name in ("easy_ham", "hard_ham", "spam", "easy_ham_2", "spam_2")],
        "data/raw/turkish-sms-original.csv", "data/raw/turkish-mail-v01.zip",
        "data/raw/text-peer/review_polarity.tar.gz", "data/raw/text-peer/rotten_imdb.tar.gz",
        "research/text_peer/bench.py", "research/text_peer/protocol.md",
        "docs/minpi-study/document-vote-protocol.md", "docs/minpi-study/mail-vote-protocol.md",
        "docs/minpi-study/refined-document-protocol.md", "docs/minpi-study/reliability-mail-external-protocol.md"])
    retain("mhealth-windows", ["data/revision-cohorts/mhealth.zip", "experiments/minpi_study_device.py",
                              "docs/minpi-study/device-protocol.md"])
    retain("nir-tablets", ["data/raw/nirbench-tablets.zip", "experiments/prepare_nir_tablets.py",
                          "experiments/fetch_nir_vote_data.py", "data/raw/nirbench/publisher_tablet_split.txt"])
    retain("real-sim", ["data/raw/full-scale-trials/real-sim.bz2", "data/raw/full-scale-trials/prepared/real_sim/manifest.json"])
    retain("historical-protocol", ["research/lowdim/" + protocol + "-protocol.md" for protocol in
                                   ("core", "formal", "comparison", "fraud", "peer", "survey", "text")])
    retain("historical-protocol", ["research/wide_blocks/protocol.md", "research/wide_blocks/reproduce.md",
                                  "research/four_stage/protocol.md"])
    summary = dict(text_splits=text_splits(), real_sim_disjoint_groups=real_sim())
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        summary.update(mhealth_windows=mhealth(scratch), tablet_splits=tablet(scratch))
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(HISTORY) and not path.is_relative_to(HERE):
                paths.setdefault(path, set()).add("upstream-import")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "inputs-origins.csv").open("w", encoding="utf8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "bytes", "sha256", "used_by"))
        for path in sorted(paths):
            writer.writerow((Path(os.path.relpath(path,HISTORY)).as_posix(), path.stat().st_size, digest(path),
                             "|".join(sorted(paths[path]))))
    (args.out / "inputs-origins-verified.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
