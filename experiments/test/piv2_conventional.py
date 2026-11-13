"""Run the extra paired selectors with the common full-training validation predictor."""

from minppi_mofs.modeling import fit_predict


def select(backend, train, yt, val, yv, policy, metric, progress, *, original_width, original_columns):
    from experiments.test.piv2_conventional_filters import FILTERS

    progress("conventional_selection", checked=0, candidates=1)
    if backend in FILTERS:
        from experiments.test.piv2_conventional_filters import choose_filter
        selected, diag = choose_filter(backend, train, yt, policy, original_width=original_width)
        fits = 0
    else:
        from experiments.test.piv2_conventional_models import choose_model
        selected, fits, diag = choose_model(backend, train, yt, policy, original_width=original_width)
    diag["selector_fits"] = fits
    progress("selected_validation", fits=fits)
    predictions = fit_predict(train[selected], yt, val[selected], **policy)
    return selected, predictions, fits + 1, diag
