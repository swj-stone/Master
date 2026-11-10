"""Historical import and command path for the trial in experiments.test."""

import sys

from experiments.test import compare_thesis_sign_aware as trial

if __name__ == "__main__":
    raise SystemExit(trial.main())

sys.modules[__name__] = trial
