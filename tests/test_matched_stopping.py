from itertools import product

from experiments.context_review.matched_stopping import choose


def test_stopping_preserves_first_maximum_of_coordinatewise_minima():
    # A candidate can attain its minimum vote and its minimum margin on
    # different restorations. Taking the lexicographic min of contextual
    # pairs would discard a genuine margin and can change the winner.
    values = ((0, -2), (1, -1), (2, -3), (4, 1), (4, 2))
    for entries in product(values, repeat=4):
        slates = (entries[:2], entries[2:])
        for method, fields in (("vote", (0,)), ("margin", (1,)), ("vote_margin", (0, 1))):
            exact = [tuple(min(pair[i] for pair in slate) for i in fields) for slate in slates]
            expected = max(range(2), key=exact.__getitem__) + 1

            def read(candidate, context):
                return tuple(slates[candidate-1][context][i] for i in fields)

            assert choose((1, 2), (0, 1), read, method, True) == expected
            assert choose((1, 2), (0, 1), read, method, False) == expected


def test_empty_slate_and_known_maximum_skip_work():
    queried = []

    def read(candidate, context):
        queried.append((candidate, context))
        return (4,)

    assert choose([], (0,), read, "vote", True) == 0
    assert not queried
    assert choose([8, 2, 4], (0, 1), read, "vote", True) == 8
    assert queried == [(8, 0), (8, 1)]
