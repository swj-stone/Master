"""The formal worker derives each mask and recipe on its own frozen seed."""

import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import thesis_sampling_worker as worker


def test_two_prepared_seeds_get_fresh_masks_recipes_and_audited_holdouts(tmp_path, monkeypatch):
    source = tmp_path / "frozen"
    source.mkdir()
    features = ["x", "x_copy", "noise"]
    seeds = (123, 456)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        row = dict(dataset_id="toy", seed=seed, task="regression", utility="r2", target="y",
                   selected_features_json=json.dumps(features), protocol_hash="c" * 64)
        for part, size in (("train", 55), ("val", 32), ("test", 22)):
            signal = rng.normal(size=size)
            frame = pd.DataFrame({"x": signal, "x_copy": signal, "noise": rng.normal(size=size), "y": signal ** 2})
            path = source / f"{seed}-{part}.csv"
            frame.to_csv(path, index=False)
            row.update({f"{part}_row_count": size, f"{part}_relpath": path.name,
                        f"{part}_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append(row)
    with (source / "data_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (source / "resolved_protocol.yaml").write_text(yaml.safe_dump(dict(
        model_params_by_dataset={"toy": dict(n_estimators=4, n_jobs=1, max_depth=2)}, backend="cpu",
        alpha_default=.95, removal={"classification_noninferiority_margin": .005, "regression_noninferiority_margin": .01}
    )))
    masks = []
    original_schedule = worker.sample_random_round
    def capture_round(n, rng):
        trials = list(original_schedule(n, rng))
        masks.append(trials)
        return iter(trials)
    monkeypatch.setattr(worker, "sample_random_round", capture_round)

    frozen = []
    original_test = worker.AuditedSplitAccess.load_test
    def audit_before_test(access, **kwargs):
        progress = json.loads((output / "progress.json").read_bytes())
        assert progress["phase"] == "test_access" and set(progress["predictor_decisions"]) == {"full", "guarded"}
        assert set(progress["method_masks"]) == {f"{r}_{m}" for r in ("full", "guarded") for m in ("original", "blend", "chosen")}
        assert progress["predictor_fits_done"] == len(set(progress["method_masks"].values()))
        frozen.append(progress["job_id"])
        return original_test(access, **kwargs)
    monkeypatch.setattr(worker.AuditedSplitAccess, "load_test", audit_before_test)
    for seed, row in zip(seeds, rows, strict=True):
        output = tmp_path / f"out-{seed}"
        worker.run("toy", output, 1, 10, selection_policy="guarded_rounds", seed=seed, source=source)
        r = json.loads((output / "result.json").read_bytes())
        methods = r["method_test_utilities"]
        assert r["seed"] == r["policy"]["seed"] == r["treeshap_seed"] == r["validation_halves"]["seed"] == seed
        assert r["sampling_reused"] is False and r["source_run_id"] == source.name
        assert r["selected_method"] == "guarded_chosen"
        assert r["parameters"]["blend_checkpoints"] == [6, 8]
        assert len(json.loads((output / "test_access.json").read_bytes())["events"]) == 1
        assert r["partition_sha256"] == {p: row[f"{p}_file_sha256"] for p in ("train","val","test")}
        assert r["validation_halves"] == worker.validation_halves(np.empty(32), "regression", seed=seed)[1]
        assert r["test_fits"] == len({r["method_masks"]["full_original"], r["method_masks"]["guarded_original"]})
        assert r["full_test_utility"] == methods["full_chosen"]
        assert r["retained_test_utility"] == methods["guarded_chosen"]
        assert r["test_delta"] == pytest.approx(methods["guarded_chosen"]-methods["full_chosen"])
        assert r["feature_test_delta"] == pytest.approx(methods["guarded_original"]-methods["full_original"])
        assert r["predictor_test_deltas"]["guarded"] == pytest.approx(methods["guarded_chosen"]-methods["guarded_original"])
        assert r["overall_test_delta_vs_original_full"] == pytest.approx(methods["guarded_chosen"]-methods["full_original"])
        assert masks[-1] == list(original_schedule(len(features), np.random.default_rng(seed)))
        with np.load(output / r["paired_test_predictions"]) as predictions:
            for role in ("full", "guarded"):
                recipe = "blend" if r["predictor_decisions"][role]["adopt_blend"] else "original"
                np.testing.assert_array_equal(predictions[f"{role}_chosen"], predictions[f"{role}_{recipe}"])
    assert len(set(frozen)) == 2
    assert masks[0] != masks[1]


def test_formal_worker_rejects_reused_seed42_sampling(tmp_path):
    with pytest.raises(ValueError, match="cold"):
        worker.run("toy", tmp_path, 5, 10, selection_policy="guarded_rounds", reuse_sampling="seed42.json")
