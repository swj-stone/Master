"""Compact split-Gain baseline using frozen-model prefix sizing."""
from minppi_mofs import piv3_compact_tree
from minppi_mofs.baselines._selection_context import Selection


def select(context):
    columns, evidence = piv3_compact_tree.select(
        context.full_model, context.train, context.yt, context.ranking, context.yr,
        task=context.task, utility=context.utility, seed=context.seed, aggregate="gain")
    return Selection(selected=columns, diagnostics=evidence)
