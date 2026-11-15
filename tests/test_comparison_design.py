from fractions import Fraction

import pytest

from experiments.context_review.plan_comparison import assess


def returned(cols, p=1):
    return dict(columns=cols, probability=p)


def test_lotteries_cancel_on_common_supports_before_bounding():
    anchor = ["a", "b", "c"]
    fold = dict(source="example", fold=0, anchor=anchor,
                left=[returned(["a", "b"], "1/6"), returned(["b", "a"], "1/6"),
                      returned(["a"], "2/3")],
                right=[returned(["a", "b"], "1/3"), returned(anchor, "2/3")])
    bounds = assess([fold], Fraction(400, 9))
    assert bounds["lower_pp"] == 0
    assert bounds["upper_pp"] == float(Fraction(400, 9))
    assert bounds["target_ruled_out"]


def test_panel_weights_sources_before_folds_and_rejects_incomplete_lotteries():
    a = ["x", "y"]
    def fold(source, number, left):
        return dict(source=source, fold=number, anchor=a, left=[returned(left)], right=[returned(a)])
    rows = [fold("one", 1, ["x"]), fold("one", 2, a), fold("two", 1, a)]
    assert assess(rows)["upper_pp"] == 12.5
    rows[0]["left"][0]["probability"] = "1/2"
    with pytest.raises(ValueError, match="sum to one"):
        assess(rows)
