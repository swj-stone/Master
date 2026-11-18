from experiments.run_formal_literature import DATASETS, FORMAL_SEEDS, protocol


def test_formal_protocol_has_one_samplingshap_job_per_partition():
    jobs = protocol({})["jobs"]
    routes = {(j["seed"], j["dataset"], j["mode"], j["method"]) for j in jobs}
    assert len(routes) == len(jobs) == 11 * len(FORMAL_SEEDS) * len(DATASETS)
    assert all(j["method"] != "shapley" for j in jobs)
    assert {r[:3] for r in routes if r[3] == "sampling_shap"} == {
        (seed, dataset, "shared") for seed in FORMAL_SEEDS for dataset in DATASETS
    }


def test_isolet_pilot_runs_only_the_requested_two_methods():
    spec = protocol({}, seeds=(42,), datasets=("isolet",), methods=("minpi", "sampling_shap"))
    assert spec["jobs"] == [
        dict(seed=42, dataset="isolet", mode="shared", method=method)
        for method in ("minpi", "sampling_shap")
    ]
    assert spec["enforce_guard"] is False and spec["limit_s"] is None
