"""Independently re-score held-out predictions and verify every terminal raw-width arm (read only)."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from experiments.test.piv3_raw_splits import DEFAULT_ROOT
from experiments.test.report_piv3_raw_six import ARMS, NAMES, TOLERANCES, WIDTHS
from experiments.test.run_sampled_permutation_seed42 import metrics
from minppi_mofs.experiment_protocol import canonical_hash


def _read(path):
    return json.loads(path.read_bytes())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_scores(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    _require(actual.shape == expected.shape == (2,) and np.isfinite(actual).all() and
             np.allclose(actual, expected, atol=1e-12, rtol=0.), f"Scores differ: {actual} vs {expected}")


def _within(scores, reference, tolerance, regression):
    ref = np.asarray(reference)
    loss = ref - np.asarray(scores)
    if regression:
        loss[1] *= -1
    return bool(np.all(loss <= tolerance * np.abs(ref) + 1e-12))


def _ordered(names, superset):
    members = set(names)
    return bool(names) and names == [name for name in superset if name in members]


def _digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit(run, inputs=None, require_complete=True):
    run = Path(run)
    if not run.is_dir():
        raise FileNotFoundError(run)
    protocol_path = run/"protocol.json"
    protocol = _read(protocol_path) if protocol_path.exists() else None
    if inputs is None:
        inputs = protocol["data"] if protocol else DEFAULT_ROOT
    inputs = Path(inputs)
    outcomes = set((run/"arms").rglob("outcome.json"))
    statuses = Counter()

    @lru_cache(maxsize=None)
    def dataset_info(dataset):
        root = inputs/dataset
        metadata = _read(root/"metadata.json")
        if protocol:
            _require(_digest(root/"metadata.json") == protocol["meta_sha256"][dataset], "Run-input provenance")
        _require(metadata["dataset"] == dataset and metadata["seed"] == 42 and
                 metadata["width"] == WIDTHS[dataset] == len(set(metadata["features"])), "Input identity")
        return metadata

    @lru_cache(maxsize=None)
    def labels_for(dataset, part):
        meta = dataset_info(dataset)
        label_path = inputs/dataset/f"{part}_y.npy"
        _require(_digest(label_path) == meta["sha256"][label_path.name], f"Corrupt staged {part} labels")
        return np.load(label_path, allow_pickle=False)

    @lru_cache(maxsize=None)
    def stage(path):
        return _read(path/"stage.json")

    @lru_cache(maxsize=None)
    def prediction_scores(folder, dataset, part):
        record = stage(folder)
        meta, labels = dataset_info(dataset), labels_for(dataset, part)
        _require(record["dataset"] == dataset, "Prediction stage identity")
        if part == "test":
            _require(record["stage"] in ("test_selected","test_full"), "Expected test stage")
            event = _read(folder/"test_access.json")
            source = Path(event["frozen_source"])
            frozen = _read(source)
            _require(canonical_hash(frozen) == record["selection_hash"] == event["selection_hash"] and
                     event["test_x_sha256"] == meta["sha256"]["test.npy"] and
                     event["test_y_sha256"] == meta["sha256"]["test_y.npy"], "Wrong test prediction provenance")
            declared = frozen["names"] if record["stage"] == "test_selected" else frozen["features"]
            _require(declared == record["names"], "Test model feature names differ from frozen selection")
        with np.load(folder/"predictions.npz", allow_pickle=False) as archive:
            pred = archive["predictions"]
            np.testing.assert_array_equal(archive["y"], labels)
            _require(np.isfinite(pred).all(), "Nonfinite test predictions")
            if meta["task"] == "regression":
                _require(pred.shape == labels.shape, "Invalid regression prediction shape")
            else:
                _require(pred.shape == (len(labels), len(np.unique(labels))) and np.min(pred) >= 0. and
                         np.max(pred) <= 1. and np.allclose(pred.sum(axis=1),1.,rtol=0.,atol=2e-6),
                         "Invalid test probabilities")
            scored = metrics(labels, pred, meta["utility"])
        _same_scores(record["scores"], scored)
        return scored

    @lru_cache(maxsize=None)
    def front_support(folder, dataset, tol, reference):
        meta = dataset_info(dataset)
        front, vote = stage(folder), _read(folder/"voting.json")
        trials, steps = front["trials"], vote["steps"]
        _require(front["dataset"] == dataset and front["stage"] == "front" and
                 front["pct"] == round(100*tol) and front["search_seed"] == 42, "Front identity")
        _require(len(trials) == len(steps) == front["fits"] <= 12 and
                 vote["features"] == meta["features"], "Voting candidate ledger")
        rates = np.asarray(vote["tier_rates"])
        _require(rates.shape == (3, meta["width"]), "Invalid number of coalition strata")
        np.testing.assert_array_equal(rates.min(axis=0), vote["min_vote"])
        labels = labels_for(dataset,"val")
        winner, winning_scores = meta["features"], reference
        with np.load(folder/"validation_trials.npz",allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["y"], labels)
            _require(len(archive.files) == len(steps)+1, "Trial prediction ledger")
            for i,(trial, step) in enumerate(zip(trials, steps, strict=True),1):
                names = trial["names"]
                scores = metrics(labels, archive[f"trial_{i}"], meta["utility"])
                _same_scores(trial["scores"], scores)
                _require(step["proposed"] == names and len(names) == step["k"] and
                         _ordered(names, meta["features"]), "Voting proposal support")
                _require(_within(scores, reference, tol, meta["task"] == "regression") == step["accepted"],
                         "Voting validation acceptance")
                if i == vote["winner_id"]:
                    _require(step["accepted"], "Winner failed validation")
                    winner, winning_scores = names, trial["scores"]
        _require(0 <= vote["winner_id"] <= len(trials) and front["names"] == winner, "PIv3 winner")
        _same_scores(front["scores"], winning_scores)
        return front["names"]

    def inspect(data, dataset, method, arm, tol):
        regression = dataset == "abalone"
        width = WIDTHS[dataset]
        _require((data["dataset"], data["method"], data["arm"], data["tolerance"], data["width"]) ==
                 (dataset, method, arm, tol, width), "Arm identity")
        seconds = data["wall_s"]
        _require(np.isfinite(seconds) and seconds > 0., "Invalid runtime")
        folders = {k: run/v for k,v in data["stages"].items()}
        for folder in folders.values():
            _require(folder.resolve().is_relative_to((run/"stages"/dataset).resolve()),
                     "Stage outside expected dataset tree")
        expected = ["reference"] + (["front"] if arm == "piv3" else []) + ["backend","test_full","test_selected"]
        if data["status"] == "timeout":
            _require(data.get("test_scores") is None and data.get("test_pass") is None,
                     "Timeout contains selected-model test result")
            order = [name for name in expected if name in folders]
            _require(bool(order) and set(order) == set(expected[:len(folders)]) == set(folders),
                     "Broken timeout stage chain")
            _require(abs(seconds-900) < 1e-6 and data["phase"].split("/")[0] == order[-1], "Invalid timeout stage")
            spent = 0.
            for step in order:
                timer = _read(folders[step]/"timer.json")
                if step == order[-1]:
                    _require(timer["status"] == "timeout" or timer["elapsed_s"] > 900 - spent,
                             "Timeout was not triggered by watchdog or cumulative budget")
                else:
                    _require(timer["status"] == "completed", "Unfinished stage before watchdog")
                    spent += timer["elapsed_s"]
            return
        _require(data["status"] == "completed" and seconds <= 900.000001, "Completion or watchdog status")
        _require(type(data["fits"]) is int and data["fits"] >= 1, "Invalid fit count")
        meta = dataset_info(dataset)
        _require(set(folders) == set(expected), "Missing stage from completed route")
        timers = {key: _read(folder/"timer.json") for key, folder in folders.items()}
        _require(all(t["status"] == "completed" for t in timers.values()) and
                 abs(sum(t["elapsed_s"] for t in timers.values())-seconds) <= 1e-6,
                 "Stage measurements disagree with reported wall time")
        _require(abs(data["front_s"] - (timers["front"]["elapsed_s"] if arm == "piv3" else 0.)) < 1e-6,
                 "Reported preprocessing time disagrees")
        reference = stage(folders["reference"])
        names = meta["features"]
        _require(names == reference["names"], "Wrong full model columns")
        _same_scores(data["reference_scores"], prediction_scores(folders["reference"], dataset, "val"))
        support = names
        if arm == "piv3":
            support = front_support(folders["front"], dataset, tol, tuple(reference["scores"]))
        _require(data["front_k"] == len(support) and 1 <= data["k"] <= data["front_k"] <= width,
                 "Invalid front or final support size")
        backend = stage(folders["backend"])
        _require((backend["dataset"], backend["stage"], backend["arm"], backend["method"], backend["pct"]) ==
                 (dataset, "backend", arm, method, round(100*tol)), "Backend identity")
        timer = timers["backend"]
        if "shared_from" in timer:
            source = run/timer["shared_from"]
            pct = round(100*tol)
            _require(arm == "plain" and pct in (5,8) and
                     source == folders["backend"].parent/"tol-03" and backend["shared_from"] == timer["shared_from"],
                     "Unexpected shared backend source")
            cross = _read(source/f"pass-tol-{pct:02d}.json")
            _require((cross["pct"], cross["names"], cross["scores"], cross["fits"]) ==
                     (pct, backend["names"], backend["scores"], backend["fits"]) and
                     cross["elapsed_s"] == timer["checkpoint_s"] and timer["materialization_s"] >= 0. and
                     abs(timer["elapsed_s"] - timer["checkpoint_s"] - timer["materialization_s"]) < 1e-8 and
                     _digest(source/cross["predictions"]) == _digest(folders["backend"]/"predictions.npz"),
                     "Cross-tolerance timing/selection/validation artifact does not match its source")
        keep = backend["names"]
        _require(data["k"] == backend["k"] == len(keep) and _ordered(keep, support), "Invalid selected support")
        _same_scores(data["validation_scores"], prediction_scores(folders["backend"], dataset, "val"))
        _require(_within(backend["scores"], reference["scores"], tol, regression), "Failed validation floors")
        prefix = backend["ranking_prefix"]
        checks = backend["checks"]
        _require(len(prefix) == len(set(prefix)) and set(prefix) <= set(support), "Invalid ranked prefix")
        _require(bool(checks) and [c["k"] for c in checks] == list(range(1, len(keep)+1)) and
                 len(prefix) >= len(keep) and set(keep) == set(prefix[:len(keep)]), "Prefix search/selection")
        for i, c in enumerate(checks):
            accepted = _within(c["scores"], reference["scores"], tol, regression)
            _require(c["accepted"] == accepted == (i == len(checks)-1), "Wrong first passing prefix")
        _same_scores(checks[-1]["scores"], backend["scores"])
        _require(backend["support_k"] == len(support) and
                 backend["fits"] == backend["selector_aux_fits"] + sum(not c["reused"] for c in checks),
                 "Prefix training ledger")
        full_test = prediction_scores(folders["test_full"], dataset, "test")
        _same_scores(full_test, data["test_reference_scores"])
        final = folders["test_selected"]
        _require(keep == stage(final)["names"], "Selected support and final model disagree")
        _same_scores(prediction_scores(final, dataset, "test"), data["test_scores"])
        _require(data["fits"] == sum(stage(f)["fits"] for f in set(folders.values())), "Total fit count")
        _require(data["test_pass"] is _within(data["test_scores"], full_test, tol, regression),
                 "Wrong held-out pass flag")

    for dataset, method, arm, tol in product(WIDTHS, NAMES, ARMS, TOLERANCES):
        path = run/"arms"/dataset/method/arm/f"tol-{round(100*tol):02}"/"outcome.json"
        if path not in outcomes:
            statuses["pending"] += 1
            continue
        outcomes.remove(path)
        data = _read(path)
        try:
            inspect(data, dataset, method, arm, tol)
        except (KeyError, ValueError, AssertionError, OSError) as e:
            raise ValueError(f"{path}: {e}") from e
        statuses[data["status"]] += 1
    _require(not outcomes, f"Unexpected outcome files: {outcomes}")
    _require(sum(statuses.values()) == 396, "Expected exactly 396 arm identities")
    if require_complete:
        _require(statuses["pending"] == 0, f"Missing {statuses['pending']} outcomes")
    return dict(statuses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--inputs", type=Path, help="Override protocol input directory")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.inputs, not args.allow_pending), sort_keys=True))
