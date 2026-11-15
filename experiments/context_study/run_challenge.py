"""Run each frozen NIPS challenge split in a separate process with a hard wall-clock limit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from threadpoolctl import threadpool_limits

from experiments.context_study import nips_challenge as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.ROOT/"results/context_study/nips2003-guarded")
    parser.add_argument("--worker", nargs=2, metavar=("DATASET","SEED"), help=argparse.SUPPRESS)
    args=parser.parse_args()
    config=json.loads(study.CONFIG.read_bytes())
    if args.worker:
        name, seed_text=args.worker
        seed=int(seed_text)
        width=config["datasets"][name][1]
        if seed not in config["seeds"]:
            raise ValueError("Unexpected inner training split")
        x,y,xv,_=study.load_train_inputs(name,width)
        with threadpool_limits(limits=1):
            study.freeze_split(x,xv,y,name,seed,config["methods"],args.out/name/str(seed))
        return
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/"guard.json"
    bindings={"hard_seconds":3600,"script":study.digest(Path(__file__)),
              "runner":study.digest(Path(study.__file__)),"design":study.digest(study.CONFIG),
              "dependencies":{n:study.digest(study.ROOT/n) for n in [
                  "src/minppi_mofs/context_design.py","src/minppi_mofs/context_vote.py",
                  "src/minppi_mofs/piv3_compact_witness.py","experiments/test/minpi_convergence_metrics.py"]},
              "archives":{n:study.digest(study.RAW/f"{n}.zip") for n in config["datasets"]}}
    if lock.exists() and json.loads(lock.read_bytes())!=bindings:
        raise ValueError("Use a new output directory for changed inputs")
    if not lock.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty directory for a fresh guarded study")
    study.write_json(lock,bindings)
    for name in config["datasets"]:
        for seed in config["seeds"]:
            dst=args.out/name/str(seed)
            if (dst/"frozen.json").is_file() and (dst/"predictions.npz").is_file():
                continue
            subprocess.run([sys.executable,"-m","experiments.context_study.run_challenge","--out",
                            str(args.out),"--worker",name,str(seed)],timeout=3600,check=True)
            print(f"{name} seed {seed}: complete",flush=True)
    # The original scoring pass reads published validation labels once all supports are frozen.
    subprocess.run([sys.executable,"-m","experiments.context_study.nips_challenge","--out",str(args.out)],
                   timeout=3600,check=True)


if __name__=="__main__":
    main()
