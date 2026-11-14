import csv
import json

import pytest

from experiments import report_formal_literature as formal


def _read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_report(root, monkeypatch, routes):
    jobs = [dict(seed=42, dataset="isolet", mode=mode, method=method) for mode, method, _ in routes]
    protocol = dict(jobs=jobs, inputs={"42": {"isolet": {}}})
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    results = {(mode, method): fields for mode, method, fields in routes}

    def read_route(root, job, pins):
        row = {**job, "width": 100, **results[(job["mode"], job["method"])]}
        features = [{**job, "feature": f"feature_{i}", "source_position": i, "selected_position": i}
                    for i in range(1, row.get("k", 0)+1)]
        return row, features

    monkeypatch.setattr(formal, "_route", read_route)
    formal.report(root, partial=True)
    return (root / "results.md").read_text(encoding="utf-8")


def _finished(**changes):
    return dict(status="completed", k=1, deleted=99, fits=3, fits_started=3, selector_s=1.25,
                wall_s=5.0, auc=.95, accuracy=.81, **changes)


def test_shared_methods_report_prediction_work_and_do_not_invent_full_scores(tmp_path, monkeypatch):
    counts = dict(sample_prediction_rows=1200, sample_prediction_calls=13,
                  selection_prediction_rows=1600, selection_prediction_calls=15)
    routes = [("shared", "sampling_shap", _finished(**counts, sampling_policy="paired draws")),
              ("shared", "minpi", _finished(sample_prediction_rows=0, sample_prediction_calls=0,
                                           selection_prediction_rows=400, selection_prediction_calls=2))]
    text = _run_report(tmp_path, monkeypatch, routes)
    assert text.count("|SamplingSHAP|") == 1
    assert "## 共享参照" in text and "排列 Shapley" not in text
    rows = _read_rows(tmp_path / "per-seed.csv")
    assert [{k: int(r[k]) for k in counts} for r in rows][0] == counts
    assert rows[0]["sampling_policy"] == "paired draws"
    assert rows[1]["sample_prediction_rows"] == "0"
    assert all(r["full_auc"] == "" for r in rows)
    groups = _read_rows(tmp_path / "five-seed-summary.csv")
    assert float(groups[0]["selection_prediction_rows_mean"]) == 1600
    assert groups[0]["sample_prediction_rows_sd"] == ""


@pytest.mark.parametrize("shap_method", ["shapley", "sampling_shap"])
def test_full_protocol_splits_modes_and_keeps_unmeasured_work_blank(tmp_path, monkeypatch, shap_method):
    pending = dict(status="not_started")
    routes = [("native", "minshap", pending), ("frozen", "minshap", pending)]
    if shap_method == "sampling_shap":
        routes.append(("shared", shap_method, pending))
    else:
        routes.extend((mode, shap_method, pending) for mode in ("native", "frozen"))
    text = _run_report(tmp_path, monkeypatch, routes)
    label = "SamplingSHAP" if shap_method == "sampling_shap" else "排列 Shapley"
    assert text.count("|"+label+"|") == 2
    rows = _read_rows(tmp_path / "per-seed.csv")
    groups = _read_rows(tmp_path / "five-seed-summary.csv")
    assert len(rows) == len(routes)
    assert all(r["fits"] == "" and r["wall_s"] == "" for r in rows)
    assert all(g["n_completed"] == "0" and g["k_mean"] == "" for g in groups)
    if shap_method == "sampling_shap":
        assert all(g["sample_prediction_rows_mean"] == "" for g in groups)
    else:
        assert "sample_prediction_rows" not in rows[0]
