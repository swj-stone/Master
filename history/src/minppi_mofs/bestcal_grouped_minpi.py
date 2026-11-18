"""Quality-first decision on the unchanged dimension-scaled pilot curve."""
import numpy as np

from minppi_mofs.scaled_pilot_grouped_minpi import scaled_pilot_grouped_top_from_training


def best_calibrated_support(p, info):
    """Choose maximum calibration BA; exact ties prefer the shorter prefix.

    Retain the requirement of at most .02 (or the given tolerance) loss versus
    full calibration. A full fallback never counts as a successful reduction.
    """
    if "candidate_widths" not in info or not len(info["candidate_widths"]):
        return np.arange(p), dict(info, reduced=False)
    widths, scores = np.asarray(info["candidate_widths"]), np.asarray(info["calibration_scores"])
    assert np.all(np.diff(widths) > 0) and widths[-1] < p
    winner = int(np.argmax(scores[:-1]))
    okay = scores[winner] >= scores[-1] - float(info["tolerance"]) - 1e-12
    count = int(widths[winner]) if okay else p
    columns = np.sort(info["ranking"][:count]) if okay else np.arange(p)
    return columns, dict(info, reduced=bool(okay),
                         reason="best_calibration_score" if okay else "no_acceptable_prefix",
                         selected_calibration_score=float(scores[winner] if okay else scores[-1]))


def bestcal_grouped_top_from_training(X, y, **kwargs):
    """Pay the entire shared pilot calibration cost, then apply quality priority."""
    _, info = scaled_pilot_grouped_top_from_training(X, y, **kwargs)
    return best_calibrated_support(X.shape[1], info)
