import csv
import hashlib

import pytest

from experiments import run_thesis_benchmark as batch


def fake_catalog(root):
    seeds = (42, 123, 456, 789, 1024)
    jobs = {("toy", seed): root / "shard" for seed in seeds}
    values = b"original"
    digest = hashlib.sha256(values).hexdigest()
    for path in (root / batch.CAMPAIGN / "campaign_plan.json", root / "shard/data_manifest.csv",
                 root / "shard/resolved_protocol.yaml", root / "shard/metadata_complete.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(values)
    return (jobs, dict(status="PASS", jobs=[{"dataset":"toy","seed":s} for s in seeds], seeds=seeds, sha256="unit",
        campaign_plan_sha256=digest, shards={"shard": dict(manifest_sha256=digest, protocol_file_sha256=digest,
        metadata_complete_sha256=digest)}, shared=dict(backend="cpu", alpha=.95, preprocessing={},
        model_params={"n_estimators":4}, classification_margin=.005, regression_margin=.01, checkpoints=[4,6,8])))


def test_plan_validates_inputs_without_creating_a_training_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)
    output = tmp_path / "runs"
    code = batch.main(["--plan", "--prepared-root", str(tmp_path), "--output-root", str(output)])
    assert code == 0 and not output.exists()
    plan = capsys.readouterr().out
    assert "5 jobs" in plan and "seed=1024" in plan and "crimedata" not in plan and "config audit=PASS" in plan
    with pytest.raises(SystemExit):
        batch.main(["--plan", "--datasets", "crimedata", "--prepared-root", str(tmp_path), "--output-root", str(output)])
    assert not output.exists()


def test_job_timeouts_preserve_partial_state_and_following_seeds_run(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)
    attempts = []

    def supervise(process, directory, seconds, log, sample_seconds, *, heartbeat_formatter):
        seed = int(directory.parent.name.removeprefix("seed-"))
        attempts.append(seed)
        assert seconds == 900 and sample_seconds == 780
        progress = dict(phase="predictor_validation", predictor_fits_done=1, predictor_fits_planned=2)
        log(heartbeat_formatter(progress, 13.4, 886.6))
        if seed == 42:
            return dict(status="timed_out", elapsed_s=900.01, progress=progress, result={})
        result = dict(dataset="toy", seed=seed, phase="complete", n_features=4, removed_count=2, retained_count=2,
                      policy={"utility": "roc_auc"}, predictor_decisions={"full": {"adopt_blend": False},
                      "guarded": {"adopt_blend": True}}, method_test_utilities={"full_original": .8, "full_chosen": .8,
                      "guarded_original": .82, "guarded_chosen": .83})
        return dict(status="completed", elapsed_s=21.37, progress=progress, result=result)

    monkeypatch.setattr(batch, "supervise", supervise)
    output = tmp_path / "runs"
    code = batch.main(["--prepared-root", str(tmp_path), "--output-root", str(output),
                       "--datasets", "toy", "--seeds", "42", "123"])
    run, = output.iterdir()
    assert code == 1 and attempts == [42, 123]
    with (run / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        first, second = csv.DictReader(stream)
    assert first["status"] == "timed_out" and first["phase"] == "predictor_validation"
    assert first["guarded_chosen_test"] == ""
    assert second["seed"] == "123" and second["wall_s"] == "21.37"
    for key, expected in (("selection_delta", .02), ("predictor_delta", .01), ("net_delta", .03),
                          ("matched_full_delta", .03), ("compression_fraction", .5)):
        assert float(second[key]) == pytest.approx(expected)
    log = (run / "run.log").read_text(encoding="utf-8")
    assert "seed=42" in log and "seed=123" in log and "phase=predictor_validation" in log
    assert "1/2" in log and "wall=21.370s" in log


def test_run_stops_before_next_seed_if_audited_settings_change(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)
    calls = []

    def change_config(process, directory, seconds, log, sample_seconds, *, heartbeat_formatter):
        calls.append(directory)
        (tmp_path / "shard/resolved_protocol.yaml").write_text("changed", encoding="utf-8")
        return dict(status="error", elapsed_s=1.0, progress={"phase":"loading"}, result={})

    monkeypatch.setattr(batch, "supervise", change_config)
    output = tmp_path / "runs"
    assert batch.main(["--prepared-root", str(tmp_path), "--output-root", str(output)]) == 1
    assert len(calls) == 1
    run, = output.iterdir()
    log = (run / "run.log").read_text(encoding="utf-8")
    assert "Stopped before dataset=toy seed=123" in log and "resolved_protocol.yaml" in log
    assert not (run / "seed-123").exists()
