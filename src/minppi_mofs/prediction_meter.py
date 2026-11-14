"""Count rows submitted to a fitted predictor during feature selection."""


class PredictionMeter:
    """Forward model calls and count completed predictions for each caller-defined phase."""

    def __init__(self, model, phase="estimate"):
        self._model = model
        self._counts = {}
        self.set_phase(phase)

    def set_phase(self, phase):
        self._phase = phase
        self._counts.setdefault(phase, {"calls": 0, "rows": 0})

    @property
    def counts(self):
        return {phase: dict(count) for phase, count in self._counts.items()}

    def __getattr__(self, name):
        # Feature ranking can inspect the same fitted booster and column metadata.
        return getattr(self._model, name)

    def _call(self, function, x, *args, **kwargs):
        output = function(x, *args, **kwargs)
        submitted_rows = x.shape[0] if hasattr(x, "shape") else len(x)
        count = self._counts[self._phase]
        count["calls"] += 1
        count["rows"] += int(submitted_rows)
        return output

    def predict(self, x, *args, **kwargs):
        return self._call(self._model.predict, x, *args, **kwargs)

    def predict_proba(self, x, *args, **kwargs):
        return self._call(self._model.predict_proba, x, *args, **kwargs)


def minpi_sample_counts(info, measured=None):
    """Split selector predictions into Gain prefix checks and cached minPI votes.

    The unmasked ranking reference is computed once. The three bootstrap draws
    per voting mask reuse a single prediction on the sampled ranking rows.
    """
    prefix_calls = 1 + len(info["masked_proposals"])
    voting_calls = info["witness_predictions"]
    voting_rows = len(info["witness_rows"]) if voting_calls else 0
    breakdown = {
        "prefix": {"calls": prefix_calls, "rows": prefix_calls * info["ranking_rows"]},
        "voting": {"calls": voting_calls, "rows": voting_calls * voting_rows},
    }
    if measured is not None:
        total = {key: sum(phase[key] for phase in breakdown.values()) for key in ("calls", "rows")}
        if total != measured:
            raise ValueError(f"minPI prediction diagnostics disagree with measured forwards: {total} vs {measured}")
    return breakdown
