import json

from experiments.reconcile_context_vote import invalid_group


def test_entire_group_is_invalid_on_late_receipt_even_if_worker_succeeded(tmp_path):
    for name, elapsed, limit in (("first", 4, 3600), ("second", 3605, 3596)):
        directory=tmp_path/name
        directory.mkdir()
        (directory/"watchdog.json").write_text(json.dumps(
            dict(status="completed",elapsed_s=elapsed,limit_s=limit)),encoding="utf-8")
    reasons=invalid_group(tmp_path,["first","second"])
    assert any("second" in item for item in reasons)
    assert any("group wall" in item for item in reasons)


def test_group_requires_all_planned_arms_and_strictly_positive_time_left(tmp_path):
    folder=tmp_path/"arm"
    folder.mkdir()
    path=folder/"watchdog.json"
    path.write_text(json.dumps(dict(status="completed",elapsed_s=5,limit_s=6)))
    assert invalid_group(tmp_path,["arm"])==[]
    assert invalid_group(tmp_path,["arm","missing"])==["missing: missing completion"]
    path.write_text(json.dumps(dict(status="budget_exhausted",elapsed_s=0)))
    assert invalid_group(tmp_path,["arm"])==["arm: budget_exhausted"]
