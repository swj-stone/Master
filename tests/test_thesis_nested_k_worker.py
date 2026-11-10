"""A three-round trajectory must reproduce the separately stopped formal workers."""

import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import thesis_nested_k_worker as nested
from experiments import thesis_sampling_worker as original


def prepared_source(root, seed):
    root.mkdir()
    rng = np.random.default_rng(seed)
    cols = ["signal", "proxy", "noise", "other"]
    row = dict(dataset_id="fixture", seed=seed, target="y", utility="r2", task="regression",
               selected_features_json=json.dumps(cols), protocol_hash="7" * 64)
    for part, n in (("train",100), ("val",40), ("test",36)):
        x = rng.normal(size=n)
        frame = pd.DataFrame(dict(signal=x, proxy=x*1.01, noise=rng.normal(size=n), other=rng.normal(size=n),
                                  y=x + .05*rng.normal(size=n)))
        p = root / f"{part}.csv"
        frame.to_csv(p, index=False)
        row.update({f"{part}_relpath": p.name, f"{part}_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    f"{part}_row_count": n})
    with (root / "data_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    protocol = dict(backend="cpu", model_params_by_dataset={"fixture": {"n_estimators":6, "max_depth":2, "n_jobs":1}},
                    alpha_default=.95, removal={"classification_noninferiority_margin":.005,
                                               "regression_noninferiority_margin":.01})
    (root/"resolved_protocol.yaml").write_text(yaml.safe_dump(protocol))
    return root


def read(path):
    return json.loads(path.read_bytes())


def test_one_schedule_three_prefixes_freeze_before_test_and_match_independent_workers(tmp_path, monkeypatch):
    seed = 123
    source = prepared_source(tmp_path/"source", seed)
    for k in (1,2,3):
        original.run("fixture", tmp_path/f"independent-{k}", k, None,
                     selection_policy="guarded_rounds", seed=seed, source=source)
    out = tmp_path/"nested"
    actual_rounds = []
    original_round = original.sample_random_round
    def tracked_round(n, rng):
        actual_rounds.append(n)
        return original_round(n, rng)
    monkeypatch.setattr(original, "sample_random_round", tracked_round)
    accessed = []
    load_test = nested.AuditedSplitAccess.load_test
    def check_frozen(self, **kwargs):
        assert len(actual_rounds) == 3
        assert read(out/"progress.json")["frozen_ks"] == [1,2,3]
        assert all(read(out/f"k-{k}/validation.json")["phase"] == "validation_frozen" for k in (1,2,3))
        assert all(not (out/f"k-{k}/result.json").exists() for k in (1,2,3))
        accessed.append(1)
        return load_test(self, **kwargs)
    monkeypatch.setattr(nested.AuditedSplitAccess, "load_test", check_frozen)
    nested.run("fixture", out, source=source, seed=seed)
    top = read(out/"result.json")
    assert accessed == [1] and len(read(out/"test_access.json")["events"]) == 1
    assert top["dataset"]=="fixture" and top["seed"]==seed
    full = (1 << 4)-1
    masks = {full}
    fits = spent = 0
    for k in (1,2,3):
        r = read(out/f"k-{k}/result.json")
        assert top["checkpoints"][str(k)] == r
        ref = read(tmp_path/f"independent-{k}/result.json")
        for name in ("sample_minima", "draws_per_feature", "witnesses", "sampled_draws",
                     "planned_trials", "resolved_trials", "skipped_after", "removed_features", "removed_count",
                     "method_masks", "predictor_decisions", "method_test_utilities"):
            assert r[name] == ref[name], (k, name)
        assert r["planned_trials"]==r["resolved_trials"]==4*k
        assert r["incremental_sample_fits"] == r["sample_fits"] - fits
        assert r["incremental_sample_s"] == pytest.approx(r["sample_elapsed_s"]-spent)
        assert r["sample_fits"]+r["sampling_cache_hits_from_selection"] == ref["sample_fits"]
        assert r["selected_validation_utility"] == ref["method_results"]["guarded_chosen"]["scores"][0]
        fits,spent = r["sample_fits"],r["sample_elapsed_s"]
        masks.add(r["selected_mask"])
    assert top["test_fits"] == top["predictor_validation_fits"] == len(masks)
