"""Recreate paper tables from the original clocks, or run fresh paired deployments."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
ARCHIVE = HISTORY / "results/minpi-study"
PUBLISHED = HISTORY / "research/fast_single"
RUNS = {"main": "fast-single-20261121b", "extension": "fast-single-extension-20261121b",
        "transfer": "fast-single-transfer-20261121b"}


def invoke(module, *args):
    # Use this root even when an extracted copy shares an environment with a checkout.
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, (HISTORY/"src", HISTORY))),
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    subprocess.run([sys.executable, "-B", "-m", module, *map(str, args)],
                   cwd=HISTORY, check=True, env=env)


def verify_tables(out, published_clocks):
    import pandas as pd
    pairs = [(out/"merged/all-405.csv", PUBLISHED/"evidence-extension/all-405.csv"),
             (out/"transfer.csv", PUBLISHED/"evidence/transfer.csv")]
    for generated, reference in pairs:
        def read(path):
            return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
        actual, expected = read(generated), read(reference)
        if list(actual.columns) != list(expected.columns):
            raise ValueError(f"Columns differ in {generated}")
        if not published_clocks:
            stable = [c for c in actual.columns if not (
                "ms" in c or "ratio" in c or c=="three_way_002")]
            actual, expected = actual[stable], expected[stable]
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        print(f"{generated.name}: verified {len(actual)} rows.", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("archived","fresh"))
    cli.add_argument("--out", type=Path, help="New output directory. Defaults to a unique run ID.")
    args = cli.parse_args()
    out = (args.out or ARCHIVE/"single-paper-replays"/
           (datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid4().hex[:8])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Output:", out, flush=True)
    archived = args.mode=="archived"
    runs = {name: ARCHIVE/value if archived else out/name for name,value in RUNS.items()}
    if not archived:
        invoke("research.lowdim.build_native")
        for name, mod in (("main","runner"),("extension","extension"),("transfer","transfer")):
            for proc in (1,2):
                invoke("research.fast_single."+mod, runs[name], "--process", proc)
    invoke("research.fast_single.summarize", runs["main"], "--out", out/"main-tables")
    invoke("research.fast_single.extension", runs["extension"], "--summarize",
           "--main", out/"main-tables/paired-cases.csv", "--out", out/"merged")
    invoke("research.fast_single.transfer", runs["transfer"], "--summarize", "--out", out/"transfer.csv")
    verify_tables(out, archived)
    if not archived:
        invoke("research.fast_single.ablation", out/"main-tables/paired-cases.csv", "--out", out/"ablation")
        invoke("research.lowdim.ablate_peers", ARCHIVE/"lowdim-core-20261118a-pass1",
               ARCHIVE/"lowdim-formal-20261118a-pass1",out/"subsets")
        import pandas as pd
        for fresh, recorded in [(out/"ablation/ablation-cases.csv",PUBLISHED/"evidence/ablation-cases.csv"),
                                (out/"subsets/cases.csv", HISTORY/"research/lowdim/peer_evidence/cases.csv")]:
            opts=dict(keep_default_na=False, float_precision="round_trip")
            pd.testing.assert_frame_equal(pd.read_csv(fresh,**opts),pd.read_csv(recorded,**opts),
                                          check_exact=False,rtol=1e-13,atol=1e-13)
        invoke("research.single_reproduction.paper_evidence","--peer-run",out/"subsets",
               "--write-subsets",out/"subsets/abalone-subset-scores.csv")
    print("Validated output:", out, flush=True)


if __name__=="__main__":
    main()
