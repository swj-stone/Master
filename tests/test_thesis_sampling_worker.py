import csv
import hashlib
import json

import pandas as pd
import yaml

from experiments import thesis_sampling_worker as worker


def test_cutoff_after_base_leaves_unvisited_feature_unscored(monkeypatch):
    monkeypatch.setattr(worker, "sample_random_round", lambda *args: iter([(0, 2, 3)]))
    now = [0.0]
    queries = []

    def evaluate(mask):
        queries.append(mask)
        now[0] = 12.0
        return 0.9

    report = worker.sample(("a", "b"), 1, 10, 0.5, evaluate, lambda **_: None, clock=lambda: now[0])
    assert queries == [2]
    assert report["coverage"] == report["marginals"] == 0
    assert report["sample_minima"] == {"a": None, "b": None}
    assert report["partial_pair"]["after_mask"] == "0x3"


def test_skips_losing_augmentation_but_keeps_first_negative_witness(monkeypatch):
    monkeypatch.setattr(worker, "sample_random_round", lambda *args: iter([(0, 0, 1), (1, 1, 3)]))
    queries = []

    def evaluate(mask):
        queries.append(mask)
        return {0: 0.0, 1: 1.0, 3: 0.0}[mask]

    report = worker.sample(("a", "b"), 2, 30, 0.5, evaluate, lambda **_: None, clock=lambda: 0)
    assert report["sample_minima"] == {"a": 1, "b": -1}
    assert report["witnesses"]["b"]["before_utility"] == 1.0
    assert report["draws_per_feature"] == {"a": 2, "b": 1}
    assert report["skipped_after"] == 0

    zero_queries = []
    report = worker.sample(("a", "b"), 2, 30, 0.5, lambda m: zero_queries.append(m) or 0.0,
                           lambda **_: None, clock=lambda: 0)
    assert report["skipped_after"] == 2
    assert zero_queries == [0, 1, 1, 3, 0, 1]
    assert report["sample_minima"] == {"a": 0, "b": 0}
    assert report["sampled_draws"] == report["marginals"] == 4
    assert report["observed_marginals"] == 2


def test_sampling_eta_respects_the_budget_and_an_unfinished_pair(monkeypatch):
    monkeypatch.setattr(worker, "sample_random_round", lambda *_: iter([(0, 0, 1), (1, 1, 3)]))
    now = [0.0]
    snapshots = []

    def evaluate(_):
        now[0] += 3
        return 0.8

    report = worker.sample(("a", "b"), 3, 8, 0.5, evaluate, lambda **s: snapshots.append(s),
                           clock=lambda: now[0])
    first = next(s for s in snapshots if s["pair_state"] == "complete")
    assert first["sampling_eta_s"] == 2  # Six seconds elapsed and five of six draws remain.
    assert first["sampling_deadline_utc"] == report["sampling_deadline_utc"]
    assert snapshots[0]["sampling_eta_s"] is None
    assert report["stop"] == "sample_time_limit" and report["sampling_eta_s"] == 0
    assert report["sampled_draws"] == report["observed_marginals"] == 1
    assert report["partial_pair"]["feature"] == "b"


def test_gate_veto_freezes_selection_and_refits_once_on_train_plus_validation(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(worker, "SOURCE", root)
    output = tmp_path / "madelon"
    row = dict(dataset_id="madelon", seed=42, task="binary", utility="roc_auc", target="y",
               protocol_hash="a" * 64, selected_features_json=json.dumps(["a", "b", "c"]))
    for part, size in (("train", 4), ("val", 2), ("test", 2)):
        frame = pd.DataFrame({"a": range(size), "b": range(size), "c": range(size), "y": [0, 1] * (size // 2)})
        path = root / f"{part}.csv"
        frame.to_csv(path, index=False)
        row.update({f"{part}_relpath": path.name, f"{part}_row_count": size,
                    f"{part}_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (root / "data_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    protocol = dict(model_params_by_dataset={"madelon": {}}, backend="cpu", alpha_default=0.95,
                    removal=dict(classification_noninferiority_margin=0.005, regression_noninferiority_margin=0.01))
    (root / "resolved_protocol.yaml").write_text(yaml.safe_dump(protocol), encoding="utf-8")
    monkeypatch.setattr(worker, "sample_random_round", lambda *_: iter([(0, 0, 1), (1, 1, 3), (2, 0, 4)]))
    held_out_calls = []

    def score(train, y_train, test, y_test, selected, **kwargs):
        if len(train) == 6:
            decision = json.loads((output / "progress.json").read_text())
            assert decision["removed_features"] == [] and decision["proposed_features"] == ["b"]
            held_out_calls.append(selected)
            return 0.7
        return {(): 0.5, ("a",): 0.9, ("a", "b"): 0.4, ("c",): 0.5,
                ("a", "c"): 0.6, ("a", "b", "c"): 0.8}[selected]

    monkeypatch.setattr(worker, "subset_utility", score)
    worker.run("madelon", output, 1, 20.0)
    result = json.loads((output / "result.json").read_bytes())
    assert result["gate_passed"] is False and result["test_delta"] == 0.0
    assert result["coverage"] == result["marginals"] == 3
    assert result["validation_fits"] == 5 and result["test_fits"] == 1
    assert held_out_calls == [("a", "b", "c")]
    assert json.loads((output / "test_access.json").read_bytes())["events"][0]["decision"] == "allowed"
