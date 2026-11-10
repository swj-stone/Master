import csv
import hashlib
import json

import pandas as pd
import pytest
import yaml
import numpy as np

from experiments import thesis_sampling_worker as worker


def test_validation_halves_score_one_prediction_vector_and_reproduce_indices():
    target = np.array([0, 1] * 12)
    probability = np.linspace(.08, .94, len(target))
    prediction = np.column_stack([1-probability, probability])
    folds, info = worker.validation_halves(target, "binary")
    assert sorted([*folds[0], *folds[1]]) == list(range(len(target)))
    assert [sum(target[h]) for h in folds] == [6, 6]
    assert worker.validation_halves(target, "binary")[1] == info
    scores = worker.prediction_scores(target, prediction, folds, "roc_auc")
    assert scores[0] == worker.compute_utility(target, prediction, metric="roc_auc")
    assert scores[1:] == tuple(worker.compute_utility(target[h], prediction[h], metric="roc_auc") for h in folds)


def test_structured_full_model_reuses_fit_for_gain_and_never_calls_treeshap(monkeypatch):
    from minppi_mofs.baselines import shap_top_k

    class Model:
        fits = 0

        def fit(self, X, y):
            self.fits += 1
            assert list(X.columns) == ["f0", "f1"]

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

        def get_booster(self):
            return self

        def get_score(self, *, importance_type):
            assert importance_type == "gain"
            return {"f0": 0.75}

    model = Model()
    monkeypatch.setattr(worker, "make_model", lambda **_: model)
    monkeypatch.setattr(shap_top_k, "fitted_tree_shap_importance", lambda *_args, **_kw: pytest.fail("unexpected TreeSHAP"))
    frame = pd.DataFrame({"x": range(4), "y": range(4)})
    y = np.array([0, 1, 0, 1])
    halves = (np.array([0, 1]), np.array([2, 3]))
    scores, gain, shap = worker.full_validation_and_gain(frame, y, frame, y, halves, include_shap=False,
                                                        task="binary", seed=42, model_params={}, backend="cpu", utility="roc_auc")
    assert model.fits == 1 and scores == (0.5, 0.5, 0.5)
    assert gain == {"x": 0.75, "y": 0.0} and shap is None


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


def test_unlimited_sampling_completes_all_rounds_or_certifies_the_minimum(monkeypatch):
    monkeypatch.setattr(worker, "sample_random_round", lambda *_: iter([(0, 0, 1), (1, 1, 3)]))
    now = 0.
    snapshots = []

    def evaluate(mask):
        nonlocal now
        now += 1000.
        return {0: 0., 1: 1., 3: 0.}[mask]

    report = worker.sample(("a", "b"), 5, None, .5, evaluate, lambda **s: snapshots.append(s),
                           clock=lambda: now)
    first = next(s for s in snapshots if s["pair_state"] == "complete")
    assert first["sampling_eta_s"] > 900 and np.isfinite(first["sampling_eta_s"])
    assert first["sampling_deadline_utc"] is None
    assert report["sample_elapsed_s"] > 900 and report["sampling_eta_s"] == 0
    assert report["stop"] == "round_limit" and report["completed_rounds"] == 5
    assert report["sample_minima"] == {"a": 1, "b": -1}
    assert report["draws_per_feature"] == {"a": 5, "b": 1}
    assert report["planned_trials"] == report["resolved_trials"] == 10
    assert report["partial_pair"] is None


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
    for part, size, start in (("train", 8, 0), ("val", 4, 100), ("test", 2, 200)):
        values = range(start, start+size)
        frame = pd.DataFrame({"a": values, "b": values, "c": values, "y": [0, 1] * (size // 2)})
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
        if len(train) == 12:
            decision = json.loads((output / "progress.json").read_text())
            assert decision["removed_features"] == [] and decision["proposed_features"] == ["b"]
            held_out_calls.append(selected)
            return 0.7
        return {(): 0.5, ("a",): 0.9, ("a", "b"): 0.4, ("c",): 0.5,
                ("a", "c"): 0.6, ("a", "b", "c"): 0.8}[selected]

    monkeypatch.setattr(worker, "subset_utility", score)
    worker.run("madelon", output, 1, 20.0, selection_policy="witness_only")
    result = json.loads((output / "result.json").read_bytes())
    assert result["gate_passed"] is False and result["test_delta"] == 0.0
    assert result["coverage"] == result["marginals"] == 3
    assert result["validation_fits"] == 5 and result["test_fits"] == 1
    assert held_out_calls == [("a", "b", "c")]
    assert json.loads((output / "test_access.json").read_bytes())["events"][0]["decision"] == "allowed"

    # Previous deletion and test outcomes must not reach the new selector. Only the
    # witnessed validation evidence is shared, and the wide policy evaluates test anew.
    source = output / "result.json"
    result.update(removed_features=["a", "b", "c"], full_test_utility=1000, retained_test_utility=-999)
    source.write_text(json.dumps(result), encoding="utf-8")
    cold_sampler = worker.sample
    monkeypatch.setattr(worker, "sample", lambda *_args, **_kwargs: pytest.fail("must reuse sampling"))
    rankings = ({"a": 10., "b": 0., "c": 9.}, {"a": 0.01, "b": 0.9, "c": 0.03})
    monkeypatch.setattr(worker, "full_validation_and_gain",
                        lambda *_args, include_shap=True, **_kwargs: ((0.8,)*3, rankings[0], rankings[1] if include_shap else None))
    fresh = tmp_path / "new-policy"
    new_tests = []

    def score_again(train, y_train, test, y_test, selected, **kwargs):
        if len(train) == 12:
            frozen = json.loads((fresh / "progress.json").read_bytes())
            assert frozen["phase"].startswith("test_")
            if frozen["selection_policy"] == "sign_aware":
                assert set(frozen["method_masks"]) == {"sign_pooled", "sign_worst", "matched_gain", "matched_treeshap"}
                assert frozen["selected_method"] == "sign_worst"
            if "capture_prediction" in kwargs:
                kwargs["capture_prediction"](np.tile([.31, .69], (len(y_test), 1)))
            new_tests.append(selected)
            return 0.72 if len(selected) == 3 else 0.71
        if len(train) == 6:
            assert len(test) == 2 and set(train.a) | set(test.a) == set(range(8))
            assert set(train.a).isdisjoint(test.a) and set(y_test) == {0, 1}
            return 0.8 if len(selected) == 3 else 0.5
        return {(): 0.5, ("a",): 0.9, ("a", "b"): 0.4, ("c",): 0.5,
                ("a", "c"): 0.6, ("a", "b", "c"): 0.8, ("b",): 0.3,
                ("b", "c"): 0.81}[selected]

    monkeypatch.setattr(worker, "subset_utility", score_again)
    validation_calls = []
    def triple(train, yt, val, yv, selected, halves, **kwargs):
        assert len(halves) == 2
        validation_calls.append(selected)
        return (score_again(train, yt, val, yv, selected, **kwargs),) * 3
    monkeypatch.setattr(worker, "subset_validation_scores", triple)
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="wide")
    wide = json.loads((fresh / "result.json").read_bytes())
    assert wide["full_test_utility"] == 0.72 and new_tests
    assert wide["sample_fits"] == 0 and wide["inherited_sample_fits"] == result["sample_fits"]
    assert wide["witness_count"] == 1 and wide["sampling_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert wide["selection"]["nomination"] and wide["selection"]["trials"]
    assert set(wide["removed_features"]) <= set(wide["proposed_features"])
    assert "a" not in wide["proposed_features"] and wide["sample_minima"]["a"] == 1
    assert wide["raw_structured_pool"]["raw_pool_size"] == wide["structured_proposed_count"]
    assert wide["ranking_fits"] == 1 and wide["extra_ranking_fits"] == 0
    assert wide["selected_gate_passed"] and wide["sampling_reused"]
    assert len(new_tests) == len(set(new_tests)) and len(validation_calls) == len(set(validation_calls))
    assert set(wide["method_results"]) >= {"gain_only", "treeshap_only", "witness_proxy_gain", "reserve_witness",
                                           "witness_zero_shap"}
    assert wide["selected_method"] not in {"gain_only", "treeshap_only"}

    fresh = tmp_path / "structured-default"
    new_tests.clear()
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source)
    structured = json.loads((fresh / "result.json").read_bytes())
    assert structured["selection_policy"] == "structured"
    assert set(structured["method_results"]) == {"witness_proxy_gain"}
    assert structured["treeshap_train_rows"] == 0 and set(structured["training_rankings"]) == {"gain"}
    expected = wide["method_results"]["witness_proxy_gain"]
    for key in ("mask", "scores", "test_utility", "removed_features", "pooled_only_mask"):
        assert structured["method_results"]["witness_proxy_gain"][key] == expected[key]
    wide_trials = {t["mask"]: (t["scores"], t["accepted"]) for t in wide["selection_trials"]
                   if any(s.startswith("witness_proxy_gain:") for s in t["sources"])}
    assert {t["mask"]: (t["scores"], t["accepted"]) for t in structured["selection_trials"]} == wide_trials
    assert structured["external_candidate_features"] == []
    assert len(new_tests) == len({7, expected["mask"]}) == structured["test_fits"]

    fresh = tmp_path / "sign-aware-replay"
    new_tests.clear()
    validation_calls.clear()
    audited_load = worker.AuditedSplitAccess.load_test
    frozen_at_access = []

    def load_after_freeze(access, **kwargs):
        snapshot = json.loads((fresh / "progress.json").read_bytes())
        if snapshot["selection_policy"] == "sign_aware":
            assert snapshot["phase"] == "test_full"
            frozen_at_access.append(snapshot["method_masks"])
        if snapshot["selection_policy"] == "confirmed":
            assert snapshot["phase"] == "test_full" and set(snapshot["method_masks"]) == {"confirmed"}
            assert snapshot["confirmation_fits"] > 0 and snapshot["confirmation"]["selected_source"] == "full_set"
        if snapshot["selection_policy"] == "guarded":
            assert snapshot["method_masks"] == {"guarded": snapshot["guarded_update"]["selected_mask"]}
            assert snapshot["phase"] == "test_full" and "confirmation" not in snapshot
        if snapshot["selection_policy"] == "local_refine":
            assert set(snapshot["method_masks"]) == {"local_robust", "local_pooled", "local_drop", "validation_frontier",
                                                     "guarded_incumbent"}
            assert snapshot["guarded_update"]["selected_mask"] == snapshot["method_masks"]["guarded_incumbent"]
            assert snapshot["phase"] == "test_full" and snapshot["selected_method"] == "local_robust"
            assert "local_refinement" in snapshot
        if snapshot["selection_policy"] == "fold_consensus":
            assert set(snapshot["method_masks"]) == {"fold_consensus", "guarded_incumbent"}
            assert snapshot["method_masks"]["guarded_incumbent"] == snapshot["guarded_update"]["selected_mask"]
            assert snapshot["selected_method"] == "fold_consensus" and snapshot["phase"] == "test_full"
            assert "fold_consensus" in snapshot
        return audited_load(access, **kwargs)

    monkeypatch.setattr(worker.AuditedSplitAccess, "load_test", load_after_freeze)
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="sign_aware")
    signed = json.loads((fresh / "result.json").read_bytes())
    assert frozen_at_access == [signed["method_masks"]]
    assert signed["sampling_reused"] and signed["new_sampling_s"] == 0
    assert signed["treeshap_train_rows"] == 8 and signed["training_rankings"]["mean_abs_treeshap"] == rankings[1]
    assert signed["sample_sign_groups"] == {"negative": ["b"], "zero": ["c"], "positive": ["a"], "unseen": []}
    tried = {f for trial in signed["selection_trials"] if trial["is_sign_candidate"] for f in trial["removed_features"]}
    assert tried == set(signed["proposed_features"]) == {"b", "c"}
    assert signed["proposed_count"] == len(tried) and signed["sampled_eligible_count"] == 2
    assert signed["raw_structured_pool"] is None and signed["gate"]["scope"] == "largest_examined_fused_prefix"
    assert signed["gate"]["raw_pool_scores"] == signed["selection"]["structured_pool"]["scores"]
    assert signed["test_fits"] == len(new_tests) == len({7, *signed["method_masks"].values()})
    assert len(validation_calls) == len(set(validation_calls))

    fresh = tmp_path / "confirmed-replay"
    new_tests.clear()
    validation_calls.clear()
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="confirmed")
    checked = json.loads((fresh / "result.json").read_bytes())
    assert checked["method_masks"] == {"confirmed": 7} and checked["removed_features"] == []
    confirmation = checked["confirmation"]
    masks = {"structured": structured["method_masks"]["witness_proxy_gain"], "sign_worst": signed["method_masks"]["sign_worst"]}
    assert confirmation["candidate_masks"] == masks
    assert checked["confirmation_fits"] == len({7, *masks.values()})
    assert checked["confirmation"]["split"]["sizes"] == [6, 2]
    assert checked["test_fits"] == 1 and len(new_tests) == 1 and not checked["external_candidate_features"]
    assert len(validation_calls) == len(set(validation_calls))

    fresh = tmp_path / "guarded-replay"
    new_tests.clear()
    validation_calls.clear()
    monkeypatch.setattr(worker, "confirm_on_training", lambda *_a, **_kw: pytest.fail("unexpected confirmation"))
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="guarded")
    guarded = json.loads((fresh / "result.json").read_bytes())
    assert guarded["guarded_update"]["candidate_masks"] == masks
    assert guarded["guarded_update"]["selected_source"] in {"structured", "sign_worst"}
    assert guarded["test_fits"] == len(new_tests) == len({7, *guarded["method_masks"].values()})
    assert guarded["external_candidate_features"] == [] and len(validation_calls) == len(set(validation_calls))

    fresh = tmp_path / "local-replay"
    new_tests.clear()
    validation_calls.clear()
    from minppi_mofs import residual_ranking

    def training_rank(train, yt, features, mask, minima, **policy):
        assert set(train.a) == set(range(8)) and set(yt) == {0, 1}
        assert mask == guarded["method_masks"]["guarded"] and policy["seed"] == 42
        assert not (fresh / "test_access.json").exists()
        kept = {f: 1. for i, f in enumerate(features) if mask & (1 << i)}
        return {f: 1. for f in features if f not in kept and minima[f] == 0}, kept, {"fit_count": 1}

    monkeypatch.setattr(residual_ranking, "rank_residual_recovery", training_rank)
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="local_refine")
    local = json.loads((fresh / "result.json").read_bytes())
    assert local["guarded_update"] == guarded["guarded_update"]
    assert local["sampling_reused"] and local["extra_ranking_fits"] == 1
    assert set(local["proposed_features"]) <= {"b", "c"}
    assert local["test_fits"] == len(new_tests) == len(set(new_tests)) == len({7, *local["method_masks"].values()})
    assert len(validation_calls) == len(set(validation_calls))
    prediction_file = fresh / local["paired_test_predictions"]
    assert hashlib.sha256(prediction_file.read_bytes()).hexdigest() == local["paired_test_predictions_sha256"]
    with np.load(prediction_file) as paired:
        assert set(paired.files) == {"y", "full_set", *local["method_masks"]}
        assert paired["y"].tolist() == [0, 1]
        np.testing.assert_array_equal(paired["local_robust"], np.tile([.31, .69], (2, 1)))

    fresh = tmp_path / "local-short-budget"
    with monkeypatch.context() as short:
        short.setenv("MINPPI_DATASET_DEADLINE", "0")
        worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="local_refine")
    stopped = json.loads((fresh / "result.json").read_bytes())
    assert stopped["local_refinement"]["budget_stopped"] and stopped["extra_ranking_fits"] == 0
    assert set(stopped["method_masks"].values()) == {guarded["guarded_update"]["selected_mask"]}

    fresh = tmp_path / "consensus-replay"
    new_tests.clear()
    validation_calls.clear()
    worker.run("madelon", fresh, 1, 20., reuse_sampling=source, selection_policy="fold_consensus")
    consensus = json.loads((fresh / "result.json").read_bytes())
    assert consensus["guarded_update"] == guarded["guarded_update"]
    assert consensus["fold_consensus"]["evaluated_count"] <= 5
    assert 0 <= consensus["extra_ranking_fits"] == consensus["fold_consensus"]["ranking_fits"] <= 2
    assert set(consensus["proposed_features"]) <= {"b", "c"}
    assert consensus["test_fits"] == len(new_tests) == len({7, *consensus["method_masks"].values()})
    assert len(validation_calls) == len(set(validation_calls))
    with np.load(fresh / consensus["paired_test_predictions"]) as paired:
        assert set(paired.files) == {"y", "full_set", "fold_consensus", "guarded_incumbent"}

    for key, field in (("endpoint", "witness"), ("split", "val hash"), ("count", "draw counts"), ("alpha", "alpha")):
        bad = json.loads(json.dumps(result))
        if key == "endpoint":
            bad["witnesses"]["b"]["after_utility"] = 0.79
        elif key == "split":
            bad["partition_sha256"]["val"] = "0" * 64
        elif key == "count":
            bad["draws_per_feature"]["b"] = 2
        else:
            bad["alpha"] = 0.8
        source.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            worker.run("madelon", tmp_path / key, 1, 20., reuse_sampling=source)
        assert not (tmp_path / key / "test_access.json").exists()

    source.with_name("watchdog.json").write_text(json.dumps({"status": "completed", "result": result}))
    bad = json.loads(json.dumps(result))
    bad["sample_minima"]["c"] = 1
    source.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="watchdog sampling evidence: sample_minima"):
        worker.run("madelon", tmp_path / "tampered-neutral", 1, 20., reuse_sampling=source)

    fresh = tmp_path / "cold-wide"
    monkeypatch.setattr(worker, "sample", cold_sampler)
    worker.run("madelon", fresh, 1, 20., selection_policy="wide")
    cold = json.loads((fresh / "result.json").read_bytes())
    assert not cold["sampling_reused"] and cold["inherited_sample_fits"] == 0 and cold["sample_fits"] > 0
    assert cold["sample_minima"] == wide["sample_minima"] and cold["method_masks"] == wide["method_masks"]

    fresh = tmp_path / "cold-sign-aware"
    worker.run("madelon", fresh, 1, 20., selection_policy="sign_aware")
    cold_signed = json.loads((fresh / "result.json").read_bytes())
    assert not cold_signed["sampling_reused"] and cold_signed["sample_fits"] > 0
    assert cold_signed["method_masks"] == signed["method_masks"] and cold_signed["method_results"] == signed["method_results"]
    assert frozen_at_access == [signed["method_masks"], cold_signed["method_masks"]]

    fresh = tmp_path / "cold-guarded"
    worker.run("madelon", fresh, 1, 20., selection_policy="guarded")
    cold_guarded = json.loads((fresh / "result.json").read_bytes())
    assert cold_guarded["sample_fits"] > 0 and not cold_guarded["sampling_reused"]
    assert cold_guarded["guarded_update"] == guarded["guarded_update"]
    assert cold_guarded["method_results"] == guarded["method_results"]

    fresh = tmp_path / "local-cold"
    worker.run("madelon", fresh, 1, 20., selection_policy="local_refine")
    cold_local = json.loads((fresh / "result.json").read_bytes())
    assert cold_local["sample_fits"] > 0 and not cold_local["sampling_reused"]
    assert cold_local["method_masks"] == local["method_masks"]
    assert cold_local["method_results"] == local["method_results"]

    fresh = tmp_path / "consensus-cold"
    worker.run("madelon", fresh, 1, 20., selection_policy="fold_consensus")
    cold_consensus = json.loads((fresh / "result.json").read_bytes())
    assert cold_consensus["sample_fits"] > 0 and not cold_consensus["sampling_reused"]
    assert cold_consensus["method_masks"] == consensus["method_masks"]
    assert cold_consensus["method_results"] == consensus["method_results"]
