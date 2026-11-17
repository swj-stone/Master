"""Sensitivity to whether failures target all supplied columns or a forest's split features."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark
from experiments.minpi_rework import predictive_reachable as reach
from experiments.minpi_rework import validation_predictive as small
from experiments.minpi_rework import wide_veto_data as wide_data
from experiments.minpi_rework import wide_veto_run as wide
from experiments.minpi_rework import quota_coalition_rf as forest
from experiments.minpi_rework.predictive_guard import margin_path
from experiments.minpi_rework.predictive_mask_study import centered
from experiments.minpi_rework.quota_compress_tree import Game
from minppi_mofs.availability_minpi import probabilities
from minppi_mofs.predictive_quota import PredictiveGame
from minppi_mofs.size_extrema import rank_arrays


OUT = reach.OUT.parent / "predictive-universe"
PROTOCOL = Path(__file__).with_name("predictive_universe_protocol.md")
FROZEN = "fc9f4f899bab900dd72ff681bb60eeb286801b03abdf82b95c219e009d4177c8"


def sample_game(model, x, y, task, active):
    raw = model.game(x)
    raw = Game(raw.a, raw.incidence[:, active].tocsr())
    return raw, PredictiveGame.from_paths(raw.a, raw.incidence, centered(model, y, task))


def attained(game, support, active, original_width):
    coefficient = game.coefficients(np.isin(active, support))
    result = {}
    for law, n in (("declared", original_width), ("active", len(active))):
        w = probabilities(n, len(coefficient)-1)[1:]
        loss = (w @ coefficient) * n / np.arange(1, n+1)
        i = int(loss.argmax())
        result[law] = dict(worst=float(loss[i]/game.null), worst_size=i+1,
                           first=float(loss[0]/game.null), intact=float(loss[-1]/game.null),
                           fail=bool(loss[i] > .02*game.null+1e-12*max(1., game.null)))
    np.testing.assert_allclose([result["declared"][key] for key in ("first", "intact")],
                               [result["active"][key] for key in ("first", "intact")], atol=2e-12)
    assert result["declared"]["worst"] <= result["active"]["worst"] + 2e-10
    return result


def choose(model, cal, game, labels, task, scale, order):
    n = len(order)
    quota = (game.path_curves(order)[:, 1:] * n/np.arange(1, n+1)).max(axis=1)
    allowed = np.flatnonzero(quota <= .02*game.null + 1e-12*max(1., game.null))
    pred = margin_path(model, cal, order)
    def metrics(i):
        return benchmark.metrics(labels, pred[i] if task == "classification" else pred[i, :, 0],
                                 task, scale)
    full = metrics(0)
    for depth in allowed[::-1]:
        if benchmark.eligible(metrics(depth), full, task):
            return np.sort(order[depth:])
    raise AssertionError("Full forest must pass both checks.")


def original(key, locked):
    record = reach.RUN / key / "selection.json"
    assert benchmark.digest(record) == locked[key]
    return json.loads(record.read_bytes())


def collect():
    assert benchmark.digest(PROTOCOL) == FROZEN
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((reach.OUT/"lock.json").read_bytes())["receipts"]
    records = {}
    for key in sorted(lock):
        old = original(key, lock)
        panel, name, seed = old["panel"], old["name"], old["seed"]
        source = wide_data if panel == "wide" else small
        x, y, task, ind = source.split(name, seed)
        test_y = y[ind["test"]]
        assert (test_y == -1).all() if task == "classification" else np.isnan(test_y).all()
        model = forest.fit(x[ind["fit"]], y[ind["fit"]], task, seed)
        active = np.unique(model.feature[model.feature >= 0])
        np.testing.assert_array_equal(active, old["records"]["full"]["canonical"])
        supports = {arm: old["records"][label]["canonical"] for arm, label in
                    (("original_mdi", "mdi_original"), ("original_portfolio", "portfolio_min_original"))}
        rows = ind["width"]
        cal, cg = sample_game(model, x[rows], y[rows], task, active)
        selected = {}
        if panel == "wide":
            rv = ind["vote"]
            _, vg = sample_game(model, x[rv], y[rv], task, active)
            order = rank_arrays(*wide.args_for(vg))[0]
            mdi = np.argsort(model.model.feature_importances_[active], kind="stable")
            scale = max(float(np.std(y[ind["fit"]])), 1e-12)
            choices = [active[choose(model, cal, cg, y[rows], task, scale, v)] for v in (mdi, order)]
            for arm, columns in zip(("active_mdi", "active_min", "active_portfolio"),
                                    [*choices, min(choices, key=len)]):
                selected[arm] = columns.tolist()
                supports[arm] = reach.structural(model.sets, model.p, columns).tolist()
        acceptance = {arm: attained(cg, ids, active, model.p) for arm, ids in supports.items()}
        for arm in ("original_mdi", "original_portfolio"):
            label = "mdi_original" if arm == "original_mdi" else "portfolio_min_original"
            np.testing.assert_allclose(acceptance[arm]["declared"]["worst"],
                                       old["records"][label]["risk"], atol=2e-9)
        records[key] = dict(panel=panel, name=name, seed=seed, p=model.p, m=len(active),
                            supports=supports, active_nominal=selected, acceptance=acceptance)
        print("selected", key, flush=True)
    assert len(records) == 55
    receipt = OUT / "supports.json"
    receipt.write_text(json.dumps(records, separators=(",", ":")), encoding="utf8")
    (OUT/"lock.json").write_text(json.dumps(dict(protocol=FROZEN, code=benchmark.digest(Path(__file__)),
        parents=benchmark.digest(reach.OUT/"lock.json"), supports=benchmark.digest(receipt)), indent=2),
        encoding="utf8")


def score():
    lock = json.loads((OUT/"lock.json").read_bytes())
    receipt = OUT / "supports.json"
    assert (lock["protocol"], lock["supports"], lock["code"]) == (
        benchmark.digest(PROTOCOL), benchmark.digest(receipt), benchmark.digest(Path(__file__)))
    rows = []
    for key, rec in json.loads(receipt.read_bytes()).items():
        panel, name, seed = rec["panel"], rec["name"], rec["seed"]
        x, y, task, ind = (wide_data.partition if panel == "wide" else small._partition)(name, seed)
        fit, test = ind["fit"], ind["test"]
        model = forest.fit(x[fit], y[fit], task, seed)
        active = np.unique(model.feature[model.feature >= 0])
        assert (len(active), model.p) == (rec["m"], rec["p"])
        _, game = sample_game(model, x[test], y[test], task, active)
        scale = max(np.std(y[np.r_[fit, ind["vote"], ind["width"]]]), 1e-12)
        for arm, cols in rec["supports"].items():
            metrics = benchmark.metrics(y[test], model.predict(x[np.ix_(test, cols)], cols), task, scale)
            stages = {"acceptance": rec["acceptance"][arm], "outer": attained(game, cols, active, model.p)}
            for stage, laws in stages.items():
                for law, result in laws.items():
                    rows.append(dict(panel=panel, dataset=name, seed=seed, p=model.p, m=len(active),
                                     arm=arm, reachable=len(cols), stage=stage, law=law, **result,
                                     **(metrics if stage == "outer" else {})))
        print("scored", key, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"scores.csv", index=False)
    frame.groupby(["panel", "dataset", "arm", "stage", "law"]).mean(numeric_only=True).to_csv(
        OUT/"source-means.csv")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=["collect", "score"])
    with threadpool_limits(1):
        (collect if cli.parse_args().stage == "collect" else score)()
