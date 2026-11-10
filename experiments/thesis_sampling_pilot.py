"""Historical import and command path for the trial in experiments.test."""

import sys

from experiments.test import thesis_sampling_pilot as trial

if __name__ == "__main__":
    raise SystemExit(trial.main())

sys.modules[__name__] = trial
