"""Export the development trials and check matched eager/lazy selection receipts."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.minpi_compression import benchmark as b

TRIALS = {
    "path-uci-dev": 8, "path-six-dev": 6,
    "path-gcv-uci": 18, "path-gcv-six": 6,
    "path-dense-uci": 18, "path-dense-six": 6,
    "path-dense-uci-five": 90, "path-dense-six-five": 30,
    "path-simple-uci-five": 90, "path-zero-uci-five": 90, "path-zero-six-five": 30,
}


def export(destination):
    runs = b.ROOT/"results/minpi_rework"
    frames = []
    for study, expected in TRIALS.items():
        directory = runs/study
        frame = pd.read_csv(directory/"scores.csv")
        assert len(frame[["dataset","seed"]].drop_duplicates()) == expected, study
        assert not frame.duplicated(["dataset","seed","method"]).any(), study
        for (name,seed), rows in frame.groupby(["dataset","seed"]):
            folder = directory/f"{name}-{seed}"
            records = json.loads((folder/"selection.json").read_bytes())
            with np.load(folder/"predictions.npz") as outputs:
                assert set(outputs.files) == set(rows.method)
                for record in rows.itertuples():
                    repetitions = records[record.method]
                    assert all(r["k"] == record.k for r in repetitions)
                    assert all(r["order"] == repetitions[0]["order"] for r in repetitions)
                    if "stages" in repetitions[0]:
                        first = repetitions[0]["stages"][0]
                        mask = (frame.dataset == name) & (frame.seed == seed) & (frame.method == record.method)
                        for k in ("negative","zero","positive"):
                            frame.loc[mask,"first_"+k] = first[k]
                if "signed_eager_gcv" in records:
                    a,c = records["signed_gcv"][0], records["signed_eager_gcv"][0]
                    assert a["order"] == c["order"] and a["k"] == c["k"]
                    assert a["deletion_edges"] + a["skipped_edges"] == c["deletion_edges"]
                    np.testing.assert_array_equal(outputs["signed_gcv"],outputs["signed_eager_gcv"])
        frames.append(frame.assign(study=study))
    all_rows = pd.concat(frames,ignore_index=True)
    destination.parent.mkdir(exist_ok=True,parents=True)
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as stream:
        stream.write(all_rows.to_csv(index=False,lineterminator="\n").encode("utf8"))
    print(f"Exported {len(all_rows)} rows from {len(TRIALS)} studies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    export(parser.parse_args().destination)
