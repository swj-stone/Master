import csv
import hashlib
import json

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
    assert "mode=formal" in plan and "time limits=none" in plan
    assert "upper bound" not in plan and "900s" not in plan and "780s" not in plan
    with pytest.raises(SystemExit):
        batch.main(["--plan", "--datasets", "crimedata", "--prepared-root", str(tmp_path), "--output-root", str(output)])
    for flag in ("--sample-seconds", "--dataset-seconds"):
        with pytest.raises(SystemExit):
            batch.main(["--plan", flag, "60", "--prepared-root", str(tmp_path)])
    assert not output.exists()


@pytest.mark.parametrize("test_mode", [False, True])
def test_entries_use_separate_default_output_roots_and_reject_cross_writes(tmp_path, monkeypatch, test_mode):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)
    output = batch.TEST_OUTPUT if test_mode else batch.FORMAL_OUTPUT
    opposite = batch.FORMAL_OUTPUT if test_mode else batch.TEST_OUTPUT

    def check_root(jobs, root, rounds, sample_seconds, limit, config, pins):
        assert root == output and rounds == 5
        assert config["experiment"]["mode"] == ("test" if test_mode else "formal")
        assert (sample_seconds, limit) == ((780, 900) if test_mode else (None, None))
        return 0

    monkeypatch.setattr(batch, "_run", check_root)
    assert batch.main(["--prepared-root", str(tmp_path)], test_mode=test_mode) == 0
    with pytest.raises(SystemExit):
        batch.main(["--output-root", str(opposite), "--prepared-root", str(tmp_path)], test_mode=test_mode)


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
                       "--datasets", "toy", "--seeds", "42", "123"], test_mode=True)
    run, = output.iterdir()
    assert code == 1 and attempts == [42, 123]
    with (run / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        first, second = csv.DictReader(stream)
    assert first["status"] == "timed_out" and first["phase"] == "predictor_validation"
    assert run.name.startswith("test-") and first["scope"] == second["scope"] == "test"
    assert json.loads((run / "protocol.json").read_bytes())["mode"] == "test"
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


def finished_test_stage():
    return dict(dataset="toy", seed=42, selected_method="guarded_chosen",
                method_test_utilities={f"{role}_{recipe}": .8 for role in ("full", "guarded")
                                       for recipe in ("original", "blend", "chosen")},
                predictor_decisions={role: {"adopt_blend": True} for role in ("full", "guarded")})


@pytest.mark.parametrize("planned,resolved,stop,finished_rounds,witness_count,sample_limit,expected", [
    (20, 20, "round_limit", 5, 1, None, "completed"),
    (20, 20, "all_witnessed", 2, 4, None, "completed"),
    (20, 14, "sample_time_limit", 3, 1, None, "error"),
    (20, 14, "round_limit", 5, 1, None, "error"),
    (12, 12, "round_limit", 3, 1, None, "error"),
    (20, 20, "round_limit", 3, 1, None, "error"),
    (20, 20, "all_witnessed", 2, 1, None, "error"),
    (20, 20, "round_limit", 5, 1, 780, "error"),
])
def test_formal_worker_has_no_deadlines_and_requires_the_entire_plan(
        tmp_path, monkeypatch, planned, resolved, stop, finished_rounds, witness_count, sample_limit, expected):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)

    def simulate(process, directory, seconds, log, sample_seconds, *, heartbeat_formatter):
        assert seconds is sample_seconds is process._args[-1] is None
        progress = dict(phase="sampling", round=2, sampling_eta_s=1500., resolved_trials=8, planned_trials=20)
        heartbeat = heartbeat_formatter(progress, 1800., None)
        assert "elapsed=00:30:00" in heartbeat and "sampling ETA=~00:25:00" in heartbeat
        assert "deadline" not in heartbeat and "resolved=8/20" in heartbeat
        log(heartbeat)
        result = dict(**finished_test_stage(), phase="complete", n_features=4, planned_trials=planned, resolved_trials=resolved,
                      completed_rounds=finished_rounds, witness_count=witness_count, sampling_stop=stop,
                      run_mode="formal" if sample_limit is None else "test", sample_seconds=sample_limit)
        return dict(status="completed", elapsed_s=1801., progress=progress, result=result)

    monkeypatch.setattr(batch, "supervise", simulate)
    output = tmp_path / "formal"
    status = batch.main(["--prepared-root", str(tmp_path), "--output-root", str(output), "--seeds", "42"])
    assert status == int(expected != "completed")
    run, = output.iterdir()
    protocol = json.loads((run / "protocol.json").read_bytes())
    assert protocol["mode"] == "formal" and protocol["rounds"] == 5
    assert protocol["sample_seconds"] is protocol["dataset_seconds"] is None
    audit = json.loads((run / "config_audit.json").read_bytes())
    assert audit["experiment"]["mode"] == "formal"
    with (run / "summary.csv").open(encoding="utf-8-sig") as stream:
        row, = csv.DictReader(stream)
    assert row["scope"] == "formal" and row["status"] == expected
    assert row["wall_s"] == "1801.0"
    if expected == "error":
        assert row["error"] and json.loads((run / "seed-42/toy/watchdog.json").read_bytes())["status"] == "error"


@pytest.mark.parametrize("incomplete", ["missing_score", "nonfinite_score", "missing_recipe"])
def test_formal_result_requires_all_frozen_predictor_outcomes(tmp_path, monkeypatch, incomplete):
    monkeypatch.setattr(batch, "audit_prepared_catalog", fake_catalog)
    result = dict(**finished_test_stage(), n_features=4, completed_rounds=5, run_mode="formal", phase="complete",
                  planned_trials=20, resolved_trials=20, sampling_stop="round_limit", sample_seconds=None)
    if incomplete == "missing_score":
        del result["method_test_utilities"]["guarded_blend"]
    elif incomplete == "nonfinite_score":
        result["method_test_utilities"]["full_chosen"] = float("inf")
    else:
        del result["predictor_decisions"]["full"]
    def fake_run(*args, **kwargs):
        return dict(status="completed", result=result, elapsed_s=6100., progress={"phase": "predictor_test"})
    monkeypatch.setattr(batch, "supervise", fake_run)
    output = tmp_path / "runs"
    assert batch.main(["--prepared-root", str(tmp_path), "--output-root", str(output), "--seeds", "42"]) == 1
    run, = output.iterdir()
    with (run / "summary.csv").open(encoding="utf-8-sig") as stream:
        row, = csv.DictReader(stream)
    assert row["status"] == "error" and row["guarded_chosen_test"] == ""
