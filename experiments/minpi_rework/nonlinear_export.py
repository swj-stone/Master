"""Package development-only rank, pruning and two-proposal observations."""
import json
from shutil import copyfile

import pandas as pd

from experiments.minpi_compression import benchmark
from experiments.minpi_rework.nonlinear_pilot import ROOT


def export():
    dest = benchmark.ROOT / "thesis-paper/evidence/minpi_rework/nonlinear"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(ROOT.glob("pilot-*/*.json")):
        run = json.loads(path.read_bytes())
        for method, arm in run["arms"].items():
            k = arm["k"]
            rows.append(dict(game=run.get("game", "triplet"), dataset=run["dataset"],
                             seed=run["seed"], method=method, k=k,
                             rank_family_s=run["rank_s"], **arm["trace"][str(k)]))
    pd.DataFrame(rows).to_csv(dest/"rank-pilot.csv", index=False)
    for filename in ("guard-refit-inputs.csv", "guard-screen.csv", "guard-refit-screen.csv",
                     "guard-two-proposals.csv"):
        copyfile(ROOT/filename, dest/filename)
    # Each guard obtains its smallest accepted width from the direct Gain trace.
    all_inputs = pd.read_csv(dest/"guard-refit-inputs.csv")
    assert len(all_inputs.groupby(["dataset", "seed"])) == 30
    for (name, seed), group in all_inputs.groupby(["dataset", "seed"]):
        assert group.loc[group.eligible, "k"].min() == group.original.iloc[0], (name, seed)


if __name__ == "__main__":
    export()
