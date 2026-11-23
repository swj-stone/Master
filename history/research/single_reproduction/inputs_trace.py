"""Trace real inputs and compare paper predictions from the thesis checkout."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
CHECKOUT = HISTORY.parent
ARCHIVE = HISTORY / "results/minpi-study"


class Reads:
    def __init__(self):
        self.phase = "imports"
        self.paths = {}

    def event(self, event, args):
        if self.phase is None:
            return
        if event == "open":
            path, mode, flags = args
            if flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_RDWR):
                return
        elif event == "ctypes.dlopen":
            path = args[0]
        else:
            return
        if isinstance(path, (str, bytes)):
            # Normalizing on export keeps file-open hooks inexpensive.
            self.paths.setdefault(os.fsdecode(path), set()).add(self.phase)

    def export(self, out):
        self.phase = None
        for module in list(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if isinstance(path, str):
                self.paths.setdefault(path, set()).add("imports")
        files, package_data = {}, {}
        for name, phases in self.paths.items():
            path = Path(name).resolve()
            if path.suffix == ".pyc":
                try:
                    path = Path(importlib.util.source_from_cache(str(path)))
                except ValueError:
                    continue
            if not path.is_file():
                continue
            if path.is_relative_to(CHECKOUT):
                if (path.is_relative_to(HISTORY / "tmp/matplotlib")
                        or path.is_relative_to(HISTORY / "tmp/numba") or path.is_relative_to(HERE)):
                    continue
                files.setdefault(path,set()).update(phases)
            elif path.is_relative_to(Path(sys.prefix)):
                if path.suffix in {".csv", ".gz", ".arff", ".asc", ".npz", ".npy"}:
                    package_data.setdefault(path,set()).update(phases)
            elif path.suffix in {".csv", ".gz", ".arff", ".npz", ".npy"}:
                raise ValueError(f"An input outside the checkout and environment needs an explicit retention rule: {path}")
        out.mkdir(parents=True, exist_ok=True)
        for name, paths, base in (("inputs.csv",files,HISTORY), ("inputs-environment.csv",package_data,Path(sys.prefix))):
            with (out/name).open("w", encoding="utf8", newline="") as handle:
                writer=csv.writer(handle)
                writer.writerow(("path","bytes","sha256","used_by"))
                for path in sorted(paths):
                    with path.open("rb") as source:
                        digest=hashlib.file_digest(source,"sha256").hexdigest()
                    writer.writerow((Path(os.path.relpath(path,base)).as_posix(),path.stat().st_size,
                                     digest,"|".join(sorted(paths[path]))))
        return len(files), len(package_data)


def same_stats(frozen, quality, aliases):
    import numpy as np

    for label, physical in aliases.items():
        for field,value in quality[physical].items():
            if field == "score":
                np.testing.assert_allclose(value,frozen[f"{field}_{label}"],rtol=0,atol=2e-12)
            elif value != frozen[f"{field}_{label}"]:
                raise ValueError(f"Frozen {field} mismatch on {label}, case {frozen['case_id']}")


def replay(reads):
    import numpy as np
    import pandas as pd
    from research.fast_single import ablation, extension, runner, selector, transfer
    from research.lowdim import ablate_peers
    from research.wide_blocks import panels

    reads.phase = "protocol"
    runner.protocol(3,selector)
    # transfer.run and extension.run also fingerprint these files before dispatching.
    for path in (HISTORY/"configs/minpi-study.json",HISTORY/"experiments/minpi_constant_transfer.py",
                 HISTORY/"research/pair_selection/external.py",HISTORY/"research/block_selection/fresh.py"):
        path.read_bytes()
    reads.phase = "main405"
    load = dict(keep_default_na=False,float_precision="round_trip")
    evidence = pd.read_csv(HISTORY/"research/fast_single/evidence-extension/all-405.csv", **load).set_index(panels.KEY)
    original = panels.registry()
    paper = pd.read_csv(runner.PAPER,**load).set_index(panels.KEY)
    wide = pd.read_csv(HISTORY/"research/fast_single/evidence/ablation-cases.csv",**load).set_index("case_id")
    main_counts = {}
    ablations = 0
    for case in panels.cases():
        frozen = evidence.loc[case.key]
        if panels.input_hashes(case) != tuple(frozen[h] for h in panels.HASHES):
            raise ValueError(f"Input mismatch on {case.key}")
        _, q, _, aliases=runner.warmup(case,original.loc[case.key],paper.loc[case.key],selector.select_fast)
        same_stats(frozen,q,aliases)
        main_counts[case.key[0]]=main_counts.get(case.key[0],0)+1
        if frozen.p > 32:
            flat=frozen.copy()
            for field,value in zip(panels.KEY,case.key):
                flat[field]=value
            reads.phase="wide-ablation"
            observed=ablation.evaluate(case,flat)
            expected=wide.loc[frozen.case_id]
            for arm in ("fast","marginal","unadjusted"):
                for field in ("k","score","cols_sha","pred_sha","decision_sha"):
                    key=f"{field}_{arm}"
                    if field=="score":
                        np.testing.assert_allclose(observed[key],expected[key],rtol=0,atol=2e-12)
                    elif observed[key]!=expected[key]:
                        raise ValueError(f"Ablation mismatch on {case.key}, {key}")
            ablations+=1
            reads.phase="main405"
        if sum(main_counts.values()) % 50 == 0:
            print("verified historical cases",sum(main_counts.values()),flush=True)

    reads.phase="extension18"
    extra=extension.registry()
    quality=pd.concat([pd.read_csv(extension.ARCHIVE/f"{suite}-pass1"/"quality.csv",**load)
                       for suite in extension.SUITES]).set_index([*panels.KEY,"method"])
    extras=0
    with tempfile.TemporaryDirectory() as scratch:
        for raw in (*extension.external_cases(),*extension.fresh_cases()):
            key=raw.key
            old=extra.loc[key]
            case=panels.Case(key,raw.X,raw.y,raw.Xt,raw.truth,raw.task,old.metric,Path(scratch)/"controls.npz",
                             "training_means" if raw.impute else "",raw.predict,raw.score)
            if panels.input_hashes(case)!=tuple(evidence.loc[key][list(panels.HASHES)]):
                raise ValueError(f"Extension input mismatch {key}")
            old,ref=extension.references(case,old,quality.loc[key])
            _,q,_,aliases=runner.warmup(case,old,ref,selector.select_fast)
            same_stats(evidence.loc[key],q,aliases)
            extras+=1

    reads.phase="transfer12"
    for i,(name,seed) in enumerate(transfer.REGISTRY):
        frozen=json.loads((ARCHIVE/"fast-single-transfer-20261121b"/f"input-{i:03}.json").read_text(encoding="utf8"))
        case=transfer.case_for(name,seed)
        if panels.input_hashes(case)!=tuple(frozen[h] for h in panels.HASHES):
            raise ValueError(f"Transfer input mismatch {name}/{seed}")
        aliases,_,q,_=transfer.matched_reference(case,selector.select_fast)
        same_stats(frozen,q,aliases)

    reads.phase="figure2"
    with tempfile.TemporaryDirectory() as scratch:
        destination=Path(scratch)/"peer-subsets"
        ablate_peers.run(ARCHIVE/"lowdim-core-20261118a-pass1",ARCHIVE/"lowdim-formal-20261118a-pass1",destination)
        ab_archive=ARCHIVE/"lowdim-peer-ablation-20261118a"
        for fresh in destination.glob("*.npz"):
            with np.load(fresh) as new, np.load(ab_archive/fresh.name) as old:
                if set(new.files)!=set(old.files):
                    raise ValueError(f"Missing Figure 2 array {fresh.name}")
                for key in new.files:
                    np.testing.assert_array_equal(new[key],old[key])
        draws=json.loads((destination/"verification.json").read_text(encoding="utf8"))
    if sum(main_counts.values())!=387 or extras!=18 or ablations!=331 or len(transfer.REGISTRY)!=12 or draws["cases"]!=31:
        raise ValueError("Incomplete paper experiment")
    return dict(main_cases=main_counts,extension_cases=extras,wide_ablation_cases=ablations,
                supplementary_cases=len(transfer.REGISTRY),subset_checks=draws)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path,default=HERE)
    args=cli.parse_args()
    reads=Reads()
    sys.addaudithook(reads.event)
    from threadpoolctl import threadpool_limits
    # Import numerical backends before applying threadpool_limits.
    import numpy
    import sklearn
    with threadpool_limits(1):
        summary=replay(reads)
    local,env=reads.export(args.out)
    summary.update(workspace_files=local,environment_data_files=env)
    (args.out/"inputs-verified.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
