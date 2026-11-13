"""Repeat the three supplemental PFI pathways sequentially with the same total 300s per pathway."""
import argparse
import os
from pathlib import Path
from statistics import median

from experiments.test.repeat_piv3_lite_times import observation,run_one,THREADS
from experiments.test.run_compact_witness import DEFAULT,METHODS,read,parent,worker
from experiments.test.report_piv3_lite_six import csv_table
from minppi_mofs.experiment_protocol import atomic_write_json


TIMED=("witness_min_pfi","tree_mean_pfi","compact_gain_pfi")


def run(root):
    with parent.exclusive_run(root):
        protocol=read(root/"protocol.json")
        for name,digest in protocol["source_sha256"].items():
            if parent._sha(parent.ROOT/name)!=digest:
                raise ValueError(f"Changing source {name} during timing")
        for dataset in parent.DATASETS:
            parent.prepare(parent.DEFAULT_ROOT,dataset)
            if parent._sha(parent.DEFAULT_ROOT/dataset/"metadata.json")!=protocol["data"][dataset]["hash"]:
                raise ValueError("Changed dataset cache")
        spec=dict(methods=list(TIMED),seed=42,protocol_sha256=parent._sha(root/"protocol.json"),
                  runner_sha256=parent._sha(Path(__file__)),total_route_budget_s=300.)
        path=root/"supp-timing-protocol.json"
        if path.exists() and read(path)!=spec:
            raise ValueError("Timing repeat source or configuration changed")
        atomic_write_json(path,spec)
        series={(d,m):[observation(root/d/m,dataset=d,method=m)] for d in parent.DATASETS for m in TIMED}
        for rep in (2,3):
            datasets=parent.DATASETS if rep==3 else tuple(reversed(parent.DATASETS))
            for d in datasets:
                i=parent.DATASETS.index(d)
                original=METHODS[i:]+METHODS[:i]
                order=[m for m in original if m in TIMED]
                shift=rep-1
                for m in order[shift:]+order[:shift]:
                    outcomes=series[d,m]
                    r=run_one(
                        root,root,f"repeat-{rep}",d,m,outcomes,target=worker)
                    outcomes.append(r)
                    print(f"{rep} {d}/{m} {r['status']} cold={r['cold']:.2f}s",flush=True)
        rows=[]
        for (d,m),records in series.items():
            ok=all(a["status"]=="completed" for a in records)
            row=dict(dataset=d,method=m,complete=ok,total_cold_s=sum(a["cold"] for a in records))
            if row["total_cold_s"]>300:
                raise ValueError("Cumulative cold time exceeded 300 seconds")
            for i,obs in enumerate(records,1):
                for k in ("cold","active","front","status"):
                    row[f"r{i}_{k}"]=obs.get(k)
            if ok:
                primary, *extra = [a["result"] for a in records]
                for key in ("selected","fits","front_k","initial_k","fallback","accepted_route","frozen_hash"):
                    row["same_"+key]=all(r[key]==primary[key] for r in extra)
                for key in ("test_scores","validation_full","validation_selected"):
                    row["max_diff_"+key]=max(abs(primary[key][s]-r[key][s]) for r in extra for s in primary[key])
                row["reproducible"]=(all(v for k,v in row.items() if k.startswith("same_")) and
                                     all(v<=1e-12 for k,v in row.items() if k.startswith("max_diff_")))
                for field in ("active","cold","front"):
                    row[f"median_{field}_s"]=median(x[field] for x in records)
            rows.append(row)
        csv_table(root/"supp-timing.csv",rows)
        print(f"{sum(r.get('reproducible',False) for r in rows)} of {len(rows)} reproducible triplicates")
        return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    args=parser.parse_args()
    for name in THREADS:
        os.environ[name]="1"
    run(args.run.resolve())

