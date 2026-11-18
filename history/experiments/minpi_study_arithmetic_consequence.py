"""Evaluate the archived Gas tie substitution and audit two native-sum bounds."""
import argparse
import csv
from hashlib import sha256
from math import comb, lcm
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, train
from experiments.minpi_study_gas_replay import H, SOURCE, TIERS, VOTE_HASH
from experiments.minpi_study_new_cohorts import batch_scores, load_cohort
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes


def gas_substitution(output):
    with np.load(ROOT / SOURCE, allow_pickle=False) as archive:
        votes = archive["train_votes"]
    assert sha256(votes.tobytes()).hexdigest() == VOTE_HASH
    exact, _ = exact_top_from_votes(votes, H, tiers=TIERS)
    floating, _ = top_from_votes(votes, H, tiers=TIERS)
    np.testing.assert_array_equal(np.setdiff1d(exact, floating), [45])
    np.testing.assert_array_equal(np.setdiff1d(floating, exact), [115])
    x, y, batch, fit, test, task = load_cohort("gas")
    with np.load(ROOT / "results/minpi-study/new-cohorts/gas-split.npz") as split:
        for key, array in (("train", fit), ("test", test), ("y", y), ("groups", batch)):
            np.testing.assert_array_equal(array, split[key])
    predictions = pd.DataFrame(dict(row=test, batch=batch[test], target=y[test]))
    scores = {}
    for method, cols in (("exact", exact), ("float", floating)):
        predictions[method], _ = train("ridge", x[fit][:, cols], y[fit], x[test][:, cols], task, 1129)
        scores[method] = batch_scores(y[test], predictions[method].to_numpy(), batch[test])
    predictions.to_csv(output / "gas-predictions.csv", index=False)
    pd.DataFrame(dict(exact=exact, float=floating)).to_csv(output / "gas-supports.csv", index=False)
    accuracy = pd.DataFrame(scores).rename_axis("batch")
    accuracy.loc["equal_batch_mean"] = accuracy.mean()
    accuracy["exact_minus_float"] = accuracy["exact"] - accuracy["float"]
    accuracy.to_csv(output / "gas-accuracy.csv")


def native_bounds(output):
    rows = []
    with (ROOT / "results/minpi-study/budget-comparison/targets.csv").open(encoding="utf-8") as stream:
        for panel in csv.DictReader(stream):
            if panel["budget_pct"] != "10":
                continue
            w, p = int(panel["w"]), int(panel["p"])
            den = [comb(p - 1, int(k)) for k in panel["tiers"].split()]
            common = w * lcm(*den)
            separate = w * max(den)
            assert (common < 2**63) == (panel["native"] == "True")
            rows.append(dict(source=panel["source"], seed=panel["seed"], w=w, p=p,
                             tiers=panel["tiers"], common_bound=common, layer_bound=separate,
                             common_fits=common < 2**63, layers_fit=separate < 2**63))
    assert len(rows) == 42
    pd.DataFrame(rows).to_csv(output / "native-bounds.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(1):
        native_bounds(output)
        gas_substitution(output)
