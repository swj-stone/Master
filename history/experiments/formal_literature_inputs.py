"""Export the existing official preprocessed CSVs, unmodified, into the worker's NPY format."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.icomp2026_stages import _label_summary, _label_summaries_match
from experiments.test.piv3_raw_splits import ROLES, _sha
from experiments.thesis_prepared_catalog import FORMAL_DATASETS, FORMAL_SEEDS, audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


def _metadata(source,row,catalog):
    dataset,seed=row["dataset_id"],int(row["seed"])
    split_path=source/Path(row["train_relpath"]).parent/"split_metadata.json"
    split=json.loads(split_path.read_bytes())
    features=json.loads(row["selected_features_json"])
    transform=json.loads(row["transform_metadata_json"])
    if (split["dataset_id"]!=dataset or split["seed"]!=seed
            or split["task"]!=row["task"] or split["target"]!=row["target"]
            or any(split[k]!=v for k,v in transform.items())
            or features!=split["selected_features"] or not 0<len(features)<=100):
        raise ValueError(f"{split_path}: seed, task, preprocessing or feature order mismatch")
    indices={}
    for role in ROLES:
        ids=split[f"{role}_indices"]
        digest=row[f"{role}_index_sha256"]
        if (len(ids)!=int(row[f"{role}_row_count"]) or any(type(i) is not int for i in ids)
                or canonical_hash(ids)!=digest or split[f"{role}_index_sha256"]!=digest):
            raise ValueError(f"{dataset}/s{seed}: archived {role} rows or order drift")
        indices[role]=np.asarray(ids,dtype=np.int64)
    all_ids=np.concatenate(list(indices.values()))
    raw_n=int(row["raw_row_count"])
    if not np.array_equal(np.sort(all_ids),np.arange(raw_n)):
        raise ValueError(f"{dataset}/s{seed}: overlapping or missing archived rows")
    source_spec=catalog["dataset_settings"][dataset]
    if row["raw_file_sha256"]!=source_spec["raw_sha256"]:
        raise ValueError(f"{dataset}: provenance of the declared raw source differs")
    shared=catalog["shared"]
    policy=dict(task=row["task"],utility=row["utility"],seed=seed,
                backend=shared["backend"],model_params=shared["model_params"])
    meta=dict(dataset=dataset,seed=seed,source=source.name,
              input_type="archived_preprocessed_csv", archive=str(source),
              archive_partitions={r:row[f"{r}_relpath"] for r in ROLES},
              archive_split_metadata_sha256=_sha(split_path),raw_sha256=row["raw_file_sha256"],
              raw_width=int(row["raw_feature_count"]),raw_rows=raw_n,target=row["target"],
              task=row["task"],utility=row["utility"],target_encoding=split["target_encoding"],
              features=features,width=len(features),feature_order_sha256=canonical_hash(features),
              policy=policy,protocol_hash=row["protocol_hash"],manifest_sha256=catalog["shards"][source.name]["manifest_sha256"],
              campaign_plan_sha256=catalog["campaign_plan_sha256"], catalog_sha256=catalog["sha256"],
              index_sha256={r:row[f"{r}_index_sha256"] for r in ROLES},
              partition_sha256={r:row[f"{r}_file_sha256"] for r in ROLES},
              partitions={r:dict(rows=len(indices[r])) for r in ROLES},dtype="float64")
    return meta,indices


def _export(to,source,row,meta,indices):
    import hashlib

    hashes={}
    for role in ROLES:
        archived=source/meta["archive_partitions"][role]
        payload=archived.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=meta["partition_sha256"][role]:
            raise ValueError(f"{archived}: prepared CSV differs from its archive manifest")
        frame=pd.read_csv(BytesIO(payload))
        del payload
        if list(frame.columns)!=[*meta["features"],meta["target"]] or len(frame)!=len(indices[role]):
            raise ValueError(f"{archived}: prepared CSV rows or ordered feature schema changed")
        labels=frame.pop(meta["target"]).to_numpy()
        x=frame.to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(labels).all():
            raise ValueError(f"{archived}: non-finite data")
        if meta["task"]=="regression":
            labels=labels.astype(np.float64)
        else:
            if not np.array_equal(labels,np.floor(labels)):
                raise ValueError(f"{archived}: invalid archived class labels")
            labels=labels.astype(np.int64)
        expected=json.loads(row[f"{role}_label_summary_json"])
        if not _label_summaries_match(_label_summary(labels,meta["task"]),expected,task=meta["task"]):
            raise ValueError(f"{archived}: label summary mismatch")
        for name,array in ((role,x),(f"{role}_y",labels),(f"{role}_indices",indices[role])):
            file=to/f"{name}.npy"
            np.save(file,array,allow_pickle=False)
            saved=np.load(file,allow_pickle=False,mmap_mode="r")
            if saved.dtype!=array.dtype or not np.array_equal(saved,array):
                raise ValueError(f"{file}: binary export differs from the archived values")
            del saved
            hashes[file.name]=_sha(file)
    atomic_write_json(to/"metadata.json",dict(meta,sha256=hashes))


def _prepare_one(root,source,row,catalog):
    meta,indices=_metadata(source,row,catalog)
    parent=root/"inputs"/f"seed-{meta['seed']}"
    parent.mkdir(parents=True,exist_ok=True)
    dest=parent/meta["dataset"]
    if not dest.exists():
        with TemporaryDirectory(dir=parent,prefix=f".{dest.name}-") as tmp:
            staged=Path(tmp)/dest.name
            staged.mkdir()
            _export(staged,source,row,meta,indices)
            try:
                staged.rename(dest)
            except FileExistsError:
                pass
    else:
        for role,relative in meta["archive_partitions"].items():
            if _sha(source/relative)!=meta["partition_sha256"][role]:
                raise ValueError(f"Archived CSV changed: {source/relative}")
    filename=dest/"metadata.json"
    cached=json.loads(filename.read_bytes())
    hashes=cached.pop("sha256")
    expected_files={f"{role}{ext}.npy" for role in ROLES for ext in ("","_y","_indices")}
    if cached!=meta or set(hashes)!=expected_files or any(_sha(dest/f)!=h for f,h in hashes.items()):
        raise ValueError(f"{dest}: input hashes or provenance changed")
    return dict(dataset=meta["dataset"],seed=meta["seed"],width=meta["width"],task=meta["task"],
                policy=meta["policy"],archive=meta["archive"],source=meta["source"],
                manifest_sha256=meta["manifest_sha256"],protocol_hash=meta["protocol_hash"],
                catalog_sha256=meta["catalog_sha256"],partition_sha256=meta["partition_sha256"],
                feature_order_sha256=meta["feature_order_sha256"],metadata_sha256=_sha(filename))


def prepare_inputs(run_root, *, seeds=FORMAL_SEEDS, datasets=FORMAL_DATASETS):
    """Return audited provenance keyed by decimal seed string, then dataset ID."""
    jobs,catalog=audit_prepared_catalog()
    rows={}
    for source in set(jobs.values()):
        with (source/"data_manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                rows[source,row["dataset_id"],int(row["seed"])]=row
    root=Path(run_root)
    result={}
    for seed in seeds:
        result[str(seed)]={}
        for dataset in datasets:
            source=jobs[dataset,seed]
            result[str(seed)][dataset]=_prepare_one(root,source,rows[source,dataset,seed],catalog)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    result=prepare_inputs(args.run)
    print(f"Verified {sum(map(len,result.values()))} preprocessed datasets across seeds {', '.join(result)}")
    print("Widths:",{d:sorted({result[s][d]["width"] for s in result}) for d in FORMAL_DATASETS})
