import numpy as np

from experiments.adaptive_minpi_transfer import generated, redundant_table, select


def test_prespecified_stratum_checks_geometry_without_outcomes():
    rng = np.random.default_rng(91)
    independent = rng.normal(size=(500, 24))
    redundant = np.column_stack((independent[:, :12], -independent[:, :12]))
    assert redundant_table(redundant) == (True, 1.)
    assert redundant_table(independent)[0] is False
    assert redundant_table(redundant[:20])[0] is False
    assert redundant_table(np.ones((500, 24))) == (False, 0.)


def test_independent_generated_panels_and_count_matched_controls():
    panel = generated("proxy", "classification", 48, 2)
    other = generated("proxy", "classification", 48, 3)
    assert redundant_table(panel.X)[0]
    assert not np.array_equal(panel.X, other.X)
    assert panel.X.shape == (1600, 48) and panel.Xt.shape == (800, 48)
    widths = {"pilot": 40, "tournament": 19}
    for base in ("random", "F"):
        small = select(base + "_tournament", panel, widths)
        large = select(base + "_pilot", panel, widths)
        assert len(small) == 19 and len(large) == 40
        assert set(small).issubset(set(large))
