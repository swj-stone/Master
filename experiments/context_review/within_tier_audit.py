"""Independently reconstruct the highest-tier lotteries from stored candidate outcomes."""
from fractions import Fraction
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import forest_predict, neighborhood_replicates
from experiments.context_review import transport_predict as primary
from experiments.context_review.within_tier import DEST
from minppi_mofs.piv3_compact_witness import _within


def verify():
    saved = pd.read_csv(DEST / "policies.csv", dtype={"nominees": str})
    saved = saved.set_index(["pipeline", "panel", "dataset", "fold", "draw", "method"])
    original_points = bg.read(DEST / "gain-points.json")["scores"]
    source_totals = {}
    checked = 0
    for pipeline, directory in (("gain", forest_predict.DEST), ("anova", neighborhood_replicates.DEST)):
        candidates = pd.read_csv(directory / "candidates.csv")
        if pipeline == "gain":
            candidates = candidates.query("voter=='forest' and receiver=='forest'")
        candidates = candidates.set_index(["panel", "dataset", "fold", "index"])
        lock = bg.read(directory / "prediction-lock.json")
        with ZipFile(directory / "predictions.zip") as pack:
            for key, rec in lock["folds"].items():
                source = (pipeline, rec["panel"], rec["dataset"])
                fold = (*source, rec["fold"])
                names = rec["metrics"]
                supports = rec["supports"]
                with np.load(BytesIO(pack.read(key + ".npz"))) as a:
                    tensor = a["tensor"]
                    points = original_points[key] if pipeline == "gain" else a["points"]
                vote, full = {}, {}
                for i in range(1, len(supports)):
                    context_votes, context_thirds = [], []
                    for reference_panels, candidate_panels in zip(tensor[0], tensor[i], strict=True):
                        margins, approve = [], []
                        for ref, alt in zip(reference_panels, candidate_panels, strict=True):
                            anchor = dict(zip(names, ref, strict=True))
                            measured = dict(zip(names, alt, strict=True))
                            margins.append(primary.panel_margin(measured, anchor))
                            approve.append(_within(measured, anchor))
                        context_votes.append(sum(approve))
                        context_thirds.append(sorted(margins, reverse=True)[2])
                    vote[i], full[i] = min(context_votes), min(context_thirds)
                slates = rec.get("slates", [list(range(len(supports)))])
                for j, slate in enumerate(slates, 1):
                    top = [i for i in slate[1:] if vote[i] == max(vote[k] for k in slate[1:])] or [0]
                    expected = {"uniform": top, "first": top[:1],
                                "point": [max(top, key=lambda i: points[i - 1]) if top != [0] else 0],
                                "full": [max(top, key=full.__getitem__) if top != [0] else 0]}
                    for rule, ids in expected.items():
                        row = saved.loc[(*fold, j, rule)]
                        assert list(map(int, row.nominees.split())) == ids
                        u, breach = Fraction(), Fraction()
                        for i in ids:
                            if i:
                                observation = candidates.loc[(*fold[1:], i)]
                                if observation.guard:
                                    if observation.outer:
                                        u += Fraction(len(supports[0]) - len(supports[i]), len(supports[0]))
                                    else:
                                        breach += 1
                        exact = (u / len(ids), breach / len(ids))
                        np.testing.assert_allclose(row[["u", "breach"]].to_numpy(float),
                                                   np.asarray(exact, float), atol=1e-14, rtol=0)
                        group = source_totals.setdefault((*source, rule), {})
                        group.setdefault(rec["fold"], []).append(exact)
                        checked += 1
    expected_tasks = pd.read_csv(DEST / "tasks.csv").set_index(["pipeline", "panel", "dataset", "method"])
    for group, folds in source_totals.items():
        values = []
        for metric in (0, 1):
            task_sum = sum(sum(v[metric] for v in draws) / len(draws) for draws in folds.values())
            values.append(float(task_sum / len(folds)))
        np.testing.assert_allclose(values, expected_tasks.loc[group, ["u", "breach"]], atol=1e-14, rtol=0)
    assert checked == len(saved) == 736 and len(source_totals) == len(expected_tasks) == 128
    # Three historical forest arms were evaluated before this diagnostic. Recover
    # their complete conditional policy metrics, including randomized AUC/accuracy.
    old = pd.read_csv(forest_predict.DEST / "policies.csv").query("voter=='forest' and receiver=='forest'").copy()
    old["method"] = old.method.map({"vote": "first", "top_uniform": "uniform", "vote_slack": "full"})
    old = old.dropna(subset=["method"]).set_index(["panel", "dataset", "fold", "method"]).sort_index()
    new = saved.reset_index().query("pipeline=='gain'").set_index(old.index.names).loc[old.index]
    metrics = ["u", "breach", "auc", "accuracy"]
    np.testing.assert_allclose(old[metrics], new[metrics], atol=1e-12, rtol=0)
    print(f"Verified {checked} within-tier decisions, {len(expected_tasks)} source-level expectations "
          f"and {len(old)} previously archived forest policies.")


if __name__ == "__main__":
    verify()
