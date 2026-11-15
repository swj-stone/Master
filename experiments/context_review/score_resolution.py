"""Rebuild frozen probability vectors and feature-type IDs for the matched fingerprint experiment."""
from io import BytesIO
from hashlib import sha256
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import transcript_information as info
from experiments.context_review import transcript_permutations as tp
from experiments.context_study.data import DATA, OUT, ROOT, digest


DESIGN = ROOT / "configs/context_review/score_resolution.json"
DEST = ROOT / "thesis-paper/evidence/score_resolution"


def text_digest(path):
    # CSV replay on Linux and Windows uses platform-specific line endings.
    return sha256(path.read_text(encoding="utf8").encode("utf8")).hexdigest()


def point_digest(path):
    columns = ["observer","dataset","fold","proposal","disposition"]
    recorded = pd.read_csv(path, usecols=columns)
    point = recorded[recorded.observer=="outer_point"][columns].sort_values(columns)
    assert len(point)==704
    return sha256(point.to_csv(index=False,lineterminator="\n").encode("ascii")).hexdigest()


def feature_types(raw):
    # Compare observed feature values before model-side float32 conversion and
    # imputation. Identical rows share the same token even when they contain NaNs.
    raw = np.asarray(raw, dtype=np.float64).copy()
    raw[raw == 0] = 0.0
    raw[np.isnan(raw)] = np.nan
    return tp.partition(raw.view(np.uint64))


def collect():
    manifest = tp.rules.read(tp.DEST / "manifest.json")
    previous = tp.DEST / "row-partitions.zip"
    assert digest(previous) == manifest["archive"]
    DEST.mkdir(parents=True, exist_ok=True)
    tmp = DEST / "unmasked.tmp"
    dest = DEST / "unmasked.zip"
    with ZipFile(previous) as partitions, ZipFile(tmp, "w") as out, threadpool_limits(limits=1):
        for key, meta in sorted(manifest["folds"].items()):
            ds, fold = meta["dataset"], meta["fold"]
            table = DATA / ds / f"{ds}.tsv.gz"
            assert digest(table) == meta["source"]
            selection = tp.rules.read(OUT / key / "selection.json")
            frame = pd.read_csv(table, sep="\t")
            y = frame.pop("target").map({v: i for i, v in enumerate(selection["class_order"])}).to_numpy()
            train, outer = selection["train"], selection["test"]
            types = feature_types(frame.iloc[outer].to_numpy())
            x = frame.to_numpy(dtype=np.float32)
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            xt = imp.fit_transform(x[train])
            p = tp.fit(xt, y[train], 20421+fold).predict_proba(imp.transform(x[outer]))
            with np.load(BytesIO(partitions.read(key + ".npz"))) as data:
                np.testing.assert_array_equal(data["row_id"], outer)
                np.testing.assert_array_equal(data["initial"], tp.partition(p))
            buffer = BytesIO()
            np.savez_compressed(buffer, probabilities=p, types=types)
            out.writestr(ZipInfo(key+".npz"), buffer.getvalue(), compress_type=ZIP_DEFLATED)
    tmp.replace(dest)
    receipt = dict(design=text_digest(DESIGN), source=manifest["archive"], predictions=digest(dest),
                   point_certificates=point_digest(info.DEST/"proposals.csv"))
    (DEST/"inputs.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf8")
    print("Verified the complete outer-row score partition on all 42 folds.")


if __name__ == "__main__":
    collect()
