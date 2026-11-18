"""Latest minPI: a compact Gain anchor followed by three-context witnessed block deletion."""
from minppi_mofs import piv3_compact_witness
from minppi_mofs.baselines._selection_context import Selection


def select(context):
    columns, evidence = piv3_compact_witness.select(
        context.full_model, context.train, context.yt, context.ranking, context.yr,
        task=context.task, utility=context.utility, seed=context.seed, aggregate="min")
    return Selection(selected=columns, diagnostics=evidence)
