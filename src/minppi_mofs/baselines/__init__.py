"""Feature selectors for the thesis literature benchmark."""
from importlib import import_module


SELECTOR_MODULES = {
    "minpi": "minppi_mofs.minpi_selector",
    "gain": "minppi_mofs.baselines.gain",
    "pfi": "minppi_mofs.baselines.pfi",
    "sampling_shap": "minppi_mofs.baselines.sampling_shap",
    "minshap": "minppi_mofs.baselines.min_shap",
    "banzhaf": "minppi_mofs.baselines.banzhaf_fs",
    "shapg": "minppi_mofs.baselines.shapg",
}


def get_selector(name):
    """Return the named method's select(context) entry point."""
    try:
        module = SELECTOR_MODULES[name]
    except KeyError as error:
        raise ValueError(f"Unknown selector {name!r}; available: {', '.join(SELECTOR_MODULES)}") from error
    return import_module(module).select
