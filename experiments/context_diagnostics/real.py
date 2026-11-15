"""Audit restoration identities on fixed proposals from the six archived tree tasks."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.context_diagnostics.design import OUTPUT, ROOT, config, digest, families, novel_masks
from minppi_mofs.experiment_protocol import atomic_write_json
from minppi_mofs.modeling import make_model
from minppi_mofs.piv3_compact_witness import _sample


ARCHIVES = ROOT / "results/thesis_benchmark"
SOURCE = ARCHIVES / "context-vote-20261115"
PREPARED = ARCHIVES / "baselines-eight-preprocessed-five-seeds/inputs"


def proposals(diag, index, limit):
    anchor = frozenset(index[name] for name in diag["anchor_order"])
    live = anchor
    history = {}
    for step in diag["trace"]:
        if len(live) != step["before"]:
            raise ValueError("Trace differs from original active support")
        subset = live.difference(index[n] for n in step["block"])
        if not subset or subset == live:
            raise ValueError("Invalid proposed subset")
        status = step["votes"]
        if history.setdefault(subset, status) != status:
            raise ValueError("Repeated identical mask produced different ballots")
        if step["accepted"]:
            live = subset
    items = list(history.items())
    ids = np.linspace(0, len(items)-1, min(len(items), limit), dtype=int)
    return anchor, [items[i] for i in ids]


def read_array(input_dir, name, meta):
    path = input_dir / name
    if digest(path) != meta["sha256"][name]:
        raise ValueError(f"Archived array hash mismatch: {path}")
    return np.load(path, allow_pickle=False)


def one(dataset, seed, out):
    study = config()["archived_trees"]
    with (SOURCE/"per-seed-effective.csv").open(encoding="utf-8-sig", newline="") as stream:
        found = [row for row in csv.DictReader(stream)
                 if row["arm"] == "min" and row["dataset"] == dataset and int(row["seed"]) == seed]
    if len(found) != 1:
        raise ValueError(f"Expected one original run {dataset}/{seed}")
    folder = SOURCE / found[0]["run_folder"]
    inp = PREPARED / f"seed-{seed}" / dataset
    meta = json.loads((inp/"metadata.json").read_bytes())
    diag = json.loads((folder/"estimator.json").read_bytes())
    index = {name: i for i, name in enumerate(meta["features"])}
    anchor, candidates = proposals(diag, index, study["candidates_per_split"])
    ordered_tail = [index[name] for name in diag["order"] if index[name] not in anchor]
    dataset_id = study["datasets"].index(dataset)
    rng = lambda code: np.random.default_rng(np.random.SeedSequence([seed, dataset_id, code]))
    rules = families(ordered_tail, len(anchor), rng(study["family_rng_stream"]))
    held = novel_masks(ordered_tail, len(anchor), {m for f in rules.values() for m in f},
                       rng(study["heldout_rng_stream"]), study["masks_per_stratum"])
    extensions = list(dict.fromkeys([m for family in rules.values() for m in family] + held))
    supports = [anchor, *(s for s, _ in candidates)]
    masks = list(dict.fromkeys(s | e for s in supports for e in extensions))
    mask_ids = {m: i for i, m in enumerate(masks)}
    lookup = [[mask_ids[s | e] for e in extensions] for s in supports]
    train = np.ascontiguousarray(read_array(inp, "train.npy", meta), dtype=np.float32)
    yt = read_array(inp, "train_y.npy", meta)
    val = np.ascontiguousarray(read_array(inp, "val.npy", meta), dtype=np.float32)
    yv = read_array(inp, "val_y.npy", meta)
    policy = {k: v for k, v in meta["policy"].items() if k != "utility"}
    task = meta["task"]
    rank, guard = train_test_split(np.arange(len(yv)), test_size=.5, random_state=seed,
                                  stratify=None if task == "regression" else yv)
    rank, guard = np.sort(rank), np.sort(guard)
    rank_idx, rank_w = _sample(yv[rank], task, np.random.SeedSequence([seed, 659]))
    guard_idx, guard_w = _sample(yv[guard], task, np.random.SeedSequence([seed, study["guard_rng_stream"]]))
    if rank_idx.tolist() != diag["witness_rows"] or not np.array_equal(rank_w, diag["panel_weights"]):
        raise ValueError("Ranking panels differ from archived voting run")
    fill_idx = np.random.default_rng(seed).choice(len(train), min(512,len(train)), replace=False)
    fill = np.nanmedian(train[fill_idx], axis=0)
    started = time.perf_counter()
    model = make_model(**policy).fit(train, yt)
    # Native training follows the exact archived device. Batch mask inference on host.
    model.set_params(device="cpu")
    predict = model.predict if task == "regression" else model.predict_proba
    with np.load(folder/"predictions.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["guard_rows"], guard) or not np.array_equal(stored["rank_rows"], rank):
            raise ValueError("Reserved split differs from archived experiment")
        full_diff = float(np.max(np.abs(stored["guard_full"] - predict(val[guard]))))
        if full_diff > 2e-5:
            raise ValueError(f"Reconstructed predictor differs on original full guard: {full_diff}")
    rows = np.r_[rank[rank_idx], guard[guard_idx]]
    matrix = val[rows]
    out.mkdir(parents=True, exist_ok=True)
    preds = []
    for start in range(0, len(masks), 8):
        batch = masks[start:start+8]
        cube = np.broadcast_to(fill, (len(batch), *matrix.shape)).copy()
        for i, mask in enumerate(batch):
            cols = sorted(mask)
            cube[i][:, cols] = matrix[:, cols]
        p = predict(cube.reshape(-1, matrix.shape[1]))
        preds.append(p.reshape(len(batch), len(rows), *p.shape[1:]))
    np.savez_compressed(out/"predictions.npz", predictions=np.concatenate(preds),
                        rank_rows=rank[rank_idx], guard_rows=guard[guard_idx],
                        rank_y=yv[rank[rank_idx]], guard_y=yv[guard[guard_idx]],
                        rank_w=rank_w, guard_w=guard_w)
    evidence = dict(dataset=dataset,seed=seed,task=task,utility=meta["utility"],
                    rare=bool(task=="binary" and np.unique(yv[rank],return_counts=True)[1].min()<=.05*len(rank)),
                    anchor=sorted(anchor),tail=ordered_tail,candidates=[sorted(s) for s,_ in candidates],
                    original_votes=[votes for _,votes in candidates],rules={r:[extensions.index(m) for m in f]
                    for r,f in rules.items()},novel=[extensions.index(m) for m in held],
                    extensions=[sorted(m) for m in extensions],masks=[sorted(m) for m in masks],lookup=lookup,
                    source={p.relative_to(ROOT).as_posix(): digest(p) for p in
                    [inp/"metadata.json",folder/"estimator.json",folder/"predictions.npz"]},
                    predictions_sha256=digest(out/"predictions.npz"),full_guard_max_difference=full_diff,
                    elapsed_s=time.perf_counter()-started)
    atomic_write_json(out/"receipt.json",evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT/"trees")
    parser.add_argument("--one", nargs=2, metavar=("DATASET","SEED"))
    args=parser.parse_args()
    schema=config()["archived_trees"]
    if args.one:
        ds,raw=args.one
        with threadpool_limits(limits=1):
            one(ds,int(raw),args.run/f"seed-{raw}"/ds)
        return
    args.run.mkdir(parents=True,exist_ok=True)
    src=[Path(__file__), Path(__file__).with_name("design.py"),
         ROOT/"configs/context_diagnostics/design.json", ROOT/"src/minppi_mofs/context_design.py",
         SOURCE/"per-seed-effective.csv"]
    pins={p.relative_to(ROOT).as_posix():digest(p) for p in src}
    protocol=args.run/"pins.json"
    if protocol.exists() and json.loads(protocol.read_bytes())!=pins:
        raise ValueError("Input/source hashes differ from the started study")
    atomic_write_json(protocol,pins)
    env={**os.environ, "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}
    jobs=[(d,s) for s in schema["seeds"] for d in schema["datasets"]]
    with (args.run/"run.log").open("a",encoding="utf-8",buffering=1) as f:
        for i,(dataset,seed) in enumerate(jobs,1):
            folder=args.run/f"seed-{seed}"/dataset
            receipt=folder/"receipt.json"
            if receipt.is_file():
                if json.loads(receipt.read_bytes())["predictions_sha256"]!=digest(folder/"predictions.npz"):
                    raise ValueError(f"Corrupt evidence: {folder}")
                continue
            start=time.monotonic()
            def log(line):
                print(line,flush=True)
                print(line,file=f,flush=True)
            log(f"{i}/{len(jobs)} starting {dataset}/{seed}")
            subprocess.run([sys.executable,"-m","experiments.context_diagnostics.real","--run",str(args.run),
                            "--one",dataset,str(seed)],cwd=ROOT,env=env,check=True,
                           timeout=schema["watchdog_seconds_per_dataset_seed"])
            log(f"{i}/{len(jobs)} completed {dataset}/{seed} in {time.monotonic()-start:.2f}s")


if __name__ == "__main__":
    main()
