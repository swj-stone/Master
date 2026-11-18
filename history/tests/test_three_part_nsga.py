import json
from itertools import combinations

import numpy as np
import pytest

from experiments.test.three_part_nsga import search


def run(cols, support=None, order=None, **kwargs):
    support = cols if support is None else support
    callbacks=[]
    def evaluate(names):
        assert set(names) <= set(support)
        callbacks.append(tuple(names))
        return {"auc": .5+.3*("f2" in names)+.1*("f7" in names), "accuracy":.6}
    details = search(support, cols, order if order is not None else cols[::-1], evaluate, None, **kwargs)
    assert len(callbacks)==len(set(callbacks))==details["unique_evaluations"]
    assert json.loads(json.dumps(details,allow_nan=False)) == details
    return details


def test_tiny_space_exhausts_and_uses_full_width_denominator():
    cols = [f"f{i}" for i in range(9)]
    support = ["f7","f3","f2"]
    result = run(cols,support=support,order=["f2"])
    assert result["exhaustive"] and result["unique_evaluations"] == 7
    assert {tuple(r["names"]) for r in result["evaluations"]} == {
        x for k in range(1,4) for x in combinations(sorted(support),k)}
    assert {tuple(r["names"]) for r in result["front"]} == {("f2",), ("f2","f7")}
    # Two disjoint rectangles from the front in normalized original coordinates.
    assert result["hv"] == pytest.approx((1.1-1/9)*(1.1-.2)+(1.1-2/9)*(.2-.1))
    assert all(r["objectives"][0]==r["k"]/9 for r in result["evaluations"])


@pytest.mark.parametrize("width,budget", [(5,24),(7,10),(5000,24),(32,1)])
def test_evolution_uses_distinct_budget_and_reports_true_archive(width,budget):
    cols=[f"f{i}" for i in range(width)]
    a=run(cols,budget=budget)
    assert a==run(cols,budget=budget)
    assert not a["exhaustive"]
    assert a["unique_evaluations"]==budget
    assert a["evaluations"][0]["names"]==cols
    if budget>1:
        assert {"f"+str(width-1)} == set(a["evaluations"][1]["names"])
    assert np.all(np.diff(a["hv_history"]) >= -1e-12)
    ps=np.asarray([r["objectives"] for r in a["evaluations"]])
    ids={i for i,v in enumerate(ps) if not np.any(np.all(ps<=v,axis=1)&np.any(ps<v,axis=1))}
    assert ids=={r["id"] for r in a["front"]}


def test_clipped_regression_with_tied_objectives_and_progress():
    messages=[]
    def progress(phase,**count):
        messages.append((phase,count["checked"],count["k"]))
    a=search(["b","a"],["a","b","c"],["b"],lambda _: {"r2":-.5,"mae":2.},progress)
    assert [r["names"] for r in a["front"]]==[["a"],["b"]]
    assert all(r["objectives"][1]==1. for r in a["evaluations"])
    assert messages==[("nsga_search",0,2),("nsga_search",1,1),("nsga_search",2,1)]


def test_invalid_inputs_and_nonfinite_metric_fail_closed():
    with pytest.raises(ValueError):
        run(["f1"],support=["unknown"])
    with pytest.raises(ValueError):
        search(["a"],["a"],["a"],lambda _: {"r2":.2,"mae":float("nan")}, None)

