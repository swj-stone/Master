"""Audit cold-start permutation selection and pair it with the two frozen PIv1 benchmark arms."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_thesis_sampling import MSK
from experiments.test.run_piv1_pilot import DATASETS, OUTPUT
from experiments.test.run_sampled_permutation_seed42 import metrics
from experiments.thesis_prepared_catalog import FORMAL_SEEDS
from minppi_mofs.experiment_protocol import canonical_hash


def read(path):
    return json.loads(path.read_bytes())


def paired_rows(formal, runs):
    archived = pd.read_csv(formal / "final_results_by_seed.csv").set_index(["dataset", "seed"])
    benchmark = pd.read_csv(formal / "paired_comparison.csv").set_index(["dataset", "seed"])
    expected = pd.MultiIndex.from_product([DATASETS, FORMAL_SEEDS])
    assert archived.index.is_unique and benchmark.index.is_unique
    assert set(archived.index) == set(benchmark.index) == set(expected)
    catalog = read(formal / "protocol.json")["catalog_hash"]
    records = []
    for run in runs:
        protocol = read(run / "protocol.json")
        seed = protocol["seed"]
        assert protocol["catalog_sha256"] == catalog and protocol["repeats"] == 5
        for dataset in DATASETS:
            arm = run / dataset
            result = read(arm / "result.json")
            frozen = read(arm / "frozen.json")
            clock = read(arm / "watchdog.json")
            events = read(arm / "test_access.json")["events"]
            index = dataset, seed
            methods = formal / f"seed-{seed}" / dataset
            ranking = read(methods / "PIv1" / "ranking.json")
            assert clock["status"] == "completed" and clock["result"] == result
            assert clock["elapsed_s"] < clock["limit_s"] == 900
            assert canonical_hash(frozen) == result["frozen_hash"]
            assert len(events) == 1 and events[0]["job_id"] == result["frozen_hash"]
            assert events[0]["reason"] == "final_removal_evaluation" and events[0]["decision"] == "allowed"
            assert (result["dataset"], result["seed"]) == index
            assert frozen["policy"] == ranking["policy"] and frozen["features"] == ranking["features"]
            assert frozen["partition_sha256"] == ranking["partition_sha256"]
            assert frozen["rows"] == ranking["votes"]["row_indices"]
            drops = np.asarray(frozen["drops"])
            assert drops.shape == (len(frozen["features"]), 5)
            means = drops.mean(axis=1)
            chosen = [name for name, drop in zip(frozen["features"], means, strict=True) if drop > 0]
            assert frozen["selected"] == (chosen or [frozen["features"][int(means.argmax())]])
            assert len(frozen["selected"]) == result["retained"]
            assert result["fits"] == 2 + int(result["retained"] < result["width"])
            a, b = archived.loc[index], benchmark.loc[index]
            assert abs(result["full_validation_utility"] - a.full_validation_primary) < 1e-12
            primary = "r2" if result["metric"] == "r2" else "auc"
            secondary = result["secondary_metric"]
            row = dict(dataset=dataset, seed=seed, primary=primary, secondary=secondary)
            with np.load(arm / "predictions.npz") as pred:
                for split, labels, predictions in (("test", "y", "selected"), ("validation", "validation_y", "validation")):
                    calculated = metrics(pred[labels], pred[predictions], result["metric"])
                    np.testing.assert_allclose(calculated, [result[f"{split}_utility"], result[f"{split}_secondary"]],
                                               rtol=0, atol=1e-12)
                for method, folder in (("piv1", "PIv1"), ("gain", "direct_gain")):
                    with np.load(methods / folder / "predictions.npz") as historical:
                        np.testing.assert_array_equal(pred["y"], historical["y"])
                        observed = metrics(historical["y"], historical["selected"], result["metric"])
                        np.testing.assert_allclose(observed, [a[f"{method}_test_{primary}"], a[f"{method}_test_{secondary}"]],
                                                   rtol=0, atol=1e-12)
                    for split in ("test", "validation"):
                        row[f"{method}_{split}_primary"] = a[f"{method}_{split}_{primary}"]
                        row[f"{method}_{split}_secondary"] = a[f"{method}_{split}_{secondary}"]
                    for key in ("retained", "fits", "wall_s"):
                        row[f"{method}_{key}"] = b[f"{method}_{key}"]
            for split in ("test", "validation"):
                row[f"pfi_{split}_primary"] = result[f"{split}_utility"]
                row[f"pfi_{split}_secondary"] = result[f"{split}_secondary"]
            row.update(pfi_retained=result["retained"], pfi_fits=result["fits"], pfi_wall_s=clock["elapsed_s"])
            records.append(row)
    pairs = pd.DataFrame(records)
    assert len(pairs) == len(expected) and set(zip(pairs.dataset, pairs.seed)) == set(expected)
    return pairs.sort_values(["dataset", "seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal", type=Path)
    parser.add_argument("runs", type=Path, nargs=len(FORMAL_SEEDS))
    args = parser.parse_args()
    pairs = paired_rows(args.formal, args.runs)
    folder = OUTPUT / datetime.now(MSK).strftime("sampled-permutation-comparison-%Y%m%d-%H%M%S-%f")
    folder.mkdir()
    pairs.to_csv(folder / "paired.csv", index=False)
    means = pairs.drop(columns="seed").groupby(["dataset", "primary", "secondary"]).mean()
    means.to_csv(folder / "by_dataset.csv")
    (folder / "inputs.json").write_text(json.dumps({
        "formal": args.formal.name, "runs": [run.name for run in args.runs],
        "timing": "single cold start per arm and split; historical PIv1 and gain, new PFI",
    }, indent=2), encoding="utf-8")
    print(folder)
    print(means.filter(regex="wall_s|test_primary|retained").round(5).to_string())
    print(pairs.filter(regex="wall_s|fits").sum().round(3).to_string())
    delta = pairs.piv1_test_primary - pairs.pfi_test_primary
    print("PIv1 vs PFI: wins / ties / losses", (delta > 1e-14).sum(), (abs(delta) <= 1e-14).sum(),
          (delta < -1e-14).sum(), "PIv1 faster", (pairs.piv1_wall_s < pairs.pfi_wall_s).sum())


if __name__ == "__main__":
    main()
