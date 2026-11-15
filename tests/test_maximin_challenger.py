from itertools import product

from minppi_mofs.maximin_challenger import nominate


def test_every_two_context_three_deletion_table_preserves_argmax_and_weakest_tie():
    order = (7, 3, 2)
    for scores in product((0,2,4),repeat=6):
        def ballot(candidate, context):
            removed = next(i for i,f in enumerate(order) if f not in candidate)
            return [j<scores[2*removed+context] for j in range(4)]
        full, record = nominate(order, (0,1), ballot, prune=False)
        fast, reduced = nominate(order, (0,1), ballot)
        target = max(range(3),key=lambda i:(min(scores[2*i:2*i+2]),i))
        assert fast==full==frozenset(order)-{order[target]}
        assert reduced["min_yes"]==record["min_yes"] and reduced["visits"]<=6
