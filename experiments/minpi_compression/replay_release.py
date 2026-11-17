"""Read the bundled forecasts and independently verify the recorded scores without retraining selectors."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_compression import chain_validation as chain
from experiments.minpi_compression import legacy_gain_receiver_audit, legacy_gain_replay, result_audit, witness_guard
from experiments.minpi_compression import witness_transfer_sources


def verified_payload(archive, key, expected, prediction_key):
    raw = archive.read(key + "/selection.json")
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Changed selection receipt for {key}")
    state = json.loads(raw)
    payload = archive.read(key + "/predictions.npz")
    if sha256(payload).hexdigest() != state[prediction_key]:
        raise ValueError(f"Changed forecasts for {key}")
    return state, BytesIO(payload)


def compare(values, expected):
    for field, value in values.items():
        np.testing.assert_allclose(value, expected[field], rtol=1e-9, atol=1e-10, err_msg=field)


def replay_ridge_panel(folder, methods, split, expected_splits):
    table = pd.read_csv(folder / "scores.csv").set_index(["dataset", "seed", "method"])
    with ZipFile(folder / "forecasts.zip") as primary:
        lock = json.loads(primary.read("lock.json"))
        assert len(lock) == expected_splits and len(table) == expected_splits * len(methods)
        for key, receipt in lock.items():
            state, payload = verified_payload(primary, key, receipt, "predictions")
            name, seed = state["dataset"], state["seed"]
            x, y, task, idx = split(name, seed)
            for part in idx:
                np.testing.assert_array_equal(idx[part], state["splits"][part])
            dev = np.r_[idx["fit"], idx["vote"], idx["width"]]
            with np.load(payload) as predictions:
                for method in methods:
                    runs = [r for r in state["rows"] if r["method"] == method]
                    row = runs[0]
                    prediction = predictions[method]
                    chain.verify_predictions(x, y, task, idx, row, prediction)
                    metrics = result_audit.outcomes(y[idx["test"]], prediction, task, y[dev].std())
                    metrics.update({field: np.median([r[field] for r in runs]) for field in
                                    ["rank_s", "selection_s", "deployment_s"]}, k=row["k"])
                    compare(metrics, table.loc[name, seed, method])
    return lock


def replay_chain():
    folder = chain.EVIDENCE
    lock = replay_ridge_panel(folder, chain.METHODS, chain.data.split, 20)
    control = pd.read_csv(folder / "matched-control.csv").set_index(["dataset", "seed"])
    with ZipFile(folder / "matched-forecasts.zip") as random:
        matched = json.loads(random.read("lock.json"))["jobs"]
        assert set(matched) == set(lock) and len(control) == 20
        for key, receipt in lock.items():
            name, seed_text = key.rsplit("-",1)
            seed = int(seed_text)
            x, y, task, idx = chain.data.split(name,seed)
            entry = matched[key]
            assert entry["primary"] == receipt
            raw = random.read(key + ".npz")
            assert sha256(raw).hexdigest() == entry["prediction"]
            with np.load(BytesIO(raw)) as data:
                pred = data["prediction"]
            chain.verify_predictions(x, y, task, idx, entry["random"], pred)
            dev = np.r_[idx["fit"], idx["vote"], idx["width"]]
            metrics = result_audit.outcomes(y[idx["test"]], pred, task, y[dev].std())
            metrics["k_random"] = entry["random"]["k"]
            compare(metrics, control.loc[name, seed])


def replay_witness_transfer():
    folder = b.EVIDENCE / "witness-transfer"
    lock = replay_ridge_panel(folder, witness_guard.METHODS, witness_transfer_sources.split, 30)
    opportunities = pd.read_csv(folder / "opportunity.csv").set_index(["dataset", "seed"])
    summary = pd.read_csv(folder / "opportunity-summary.csv").set_index(["task", "dataset"])
    table = pd.read_csv(folder / "scores.csv").set_index(["dataset", "seed", "method"])
    bounds = []
    with ZipFile(folder / "forecasts.zip") as archive:
        for key, receipt in lock.items():
            state, payload = verified_payload(archive, key, receipt, "predictions")
            name, seed, domain = state["dataset"], state["seed"], state["task"]
            records = {r["method"]: r for r in state["rows"] if r["repeat"] == 0}
            witness = records["witness"]
            prefix = set(records["coefficient"]["order"][:records["coefficient"]["k"]])
            assert prefix == set(witness["prefix"]) and set(witness["negative"]) <= prefix
            h, k = len(set(witness["negative"])), len(prefix)
            eligible = 0 < h < k
            floor = (k - h) / k if eligible else 1.
            compare(dict(inside_negatives=h, prefix_width=k, best_ratio=floor, eligible=int(eligible)),
                    opportunities.loc[name, seed])
            assert records["witness"]["k"] / k >= floor
            bounds.append(dict(dataset=name, task=domain, floor=floor, eligible=eligible, h=h))
            if name not in ("musk1", "fires"):
                continue
            _, y, _, idx = witness_transfer_sources.split(name, seed)
            test = idx["test"]
            with np.load(payload) as predictions:
                for method in witness_guard.METHODS:
                    pred = predictions[method]
                    if name == "musk1":
                        groups = witness_transfer_sources.load(name)[3][test]
                        scores, labels = [], []
                        for group in np.unique(groups):
                            mask = groups == group
                            assert len(np.unique(y[test][mask])) == 1
                            labels.append(y[test][mask][0])
                            scores.append(np.max(pred[mask, 1] - pred[mask, 0]))
                        score = np.array(scores, dtype=float)
                        metrics = result_audit.outcomes(np.asarray(labels), np.column_stack([-score, score]),
                                                        "classification", 1.)
                        check = dict(molecule_auc=metrics["auc"], molecule_accuracy=metrics["accuracy"])
                    else:
                        metrics = result_audit.outcomes(np.expm1(y[test]), np.maximum(np.expm1(pred), 0),
                                                        "regression", 1.)
                        check = dict(hectares_mae=metrics["mae"], hectares_r2=metrics["r2"])
                    compare(check, table.loc[name, seed, method])
    frame = pd.DataFrame(bounds)
    for (task, name), row in summary.iterrows():
        relevant = frame[frame.task == task] if name == "ALL" else frame[frame.dataset == name]
        per_source = relevant.groupby("dataset")["floor"].apply(lambda r: np.log(r).mean())
        floor = np.exp(per_source.mean())
        compare(dict(best_width_ratio=floor, max_saving=1-floor, splits=len(relevant),
                     eligible_splits=relevant.eligible.sum(), inside_negatives=relevant.h.sum()), row)


def replay_wide():
    folder = b.EVIDENCE / "wide-transfer"
    table = pd.read_csv(folder / "scores.csv").set_index(["dataset", "fold", "method"])
    with ZipFile(folder / "forecasts.zip") as pack:
        lock = json.loads(pack.read("lock.json"))
        assert len(lock) == 16 and len(table) == 96
        for key, receipt in lock.items():
            name, fold = key.split("-", 1)
            state, payload = verified_payload(pack, key, receipt, "predictions")
            meta = json.loads(pack.read(key + "/layout.json"))
            targets = pack.read(key + "/targets.npz")
            assert sha256(targets).hexdigest() == meta["targets_sha256"]
            assert state["inputs_sha256"] == meta["inputs_sha256"]
            with np.load(BytesIO(targets)) as target, np.load(payload) as pred:
                np.testing.assert_array_equal(target["test_row"], pred["test_row"])
                for method in b.METHODS:
                    runs = [r for r in state["rows"] if r["method"] == method]
                    values = result_audit.outcomes(target["y"], pred[method], "classification", 1.)
                    values.update({key: np.median([r[key] for r in runs]) for key in
                                   ["rank_s", "selection_s", "deployment_s"]}, k=runs[0]["k"])
                    compare(values, table.loc[name, fold, method])


def replay_receivers():
    folder = b.EVIDENCE / "legacy-gain"
    table = pd.read_csv(folder / "xgb-receiver/scores.csv").set_index(["part", "dataset", "seed", "method"])
    with ZipFile(folder / "xgb-receiver/forecasts.zip") as strong, ZipFile(folder / "forecasts.zip") as ridge, \
            np.load(folder / "novel-input-rows.npz") as masks:
        lock = json.loads(strong.read("lock.json"))
        parent = json.loads(ridge.read("lock.json"))
        assert len(lock) == len(parent) == 30
        for key, receipt in lock.items():
            state, payload = verified_payload(strong, key, receipt, "forecast")
            assert parent[key] == state["parent"]
            base, _ = verified_payload(ridge, key, parent[key], "forecast")
            y = np.load(BytesIO(ridge.read(key + "/targets.npy")))
            with np.load(payload) as pred:
                assert len(pred.files) == 8
                for method in pred:
                    row = state["records"][method]
                    for part, mask in (("primary", slice(None)), ("novel", masks[key])):
                        # The boosted receiver stores float32 scores. Use sklearn's AUC implementation
                        # independently of the historical scorer's double-precision rank sums.
                        values = b.metrics(y[mask], pred[method][mask], state["task"], base["scale"])
                        values.update({field: val for field, val in row.items() if field.endswith("_s")}, k=row["k"])
                        compare(values, table.loc[part, base["dataset"], base["seed"], method])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        with ZipFile(b.EVIDENCE / "confirmation/forecasts.zip") as archive:
            frame, _, strict = result_audit.verify(archive)
            assert len(frame) == len(strict) == 240
        replay_chain()
        replay_ridge_panel(witness_guard.EVIDENCE, witness_guard.METHODS, witness_guard.split, 60)
        replay_witness_transfer()
        replay_wide()
        legacy_gain_replay.verify()
        replay_receivers()
        historic = b.ROOT / "packages/piv1-original-forecasts.zip"
        legacy_gain_receiver_audit.check(historic if historic.is_file() else None)
    print("Replayed 480 primary and sensitivity outcomes, 140 independent chain outcomes, "
          "300 retrospective and 150 fresh guarded-deletion outcomes, 96 wider outcomes and "
          "120 original ridge outcomes. Checked predictions for all eight "
          "boosted receiver arms on the 30 original splits and their unseen-input subsets.")
