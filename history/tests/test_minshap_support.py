from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from minppi_mofs.baselines._permutation import permutation_scores
from minppi_mofs.baselines.min_shap import select


@pytest.mark.parametrize("seed", [0, 7, 42, 83])
@pytest.mark.parametrize("workers", [1,2])
def test_negative_certificates_preserve_native_support_and_coalitions(seed,workers):
    columns = [f"f{i}" for i in range(16)]
    rng = np.random.default_rng(seed)
    row_effects = rng.normal(0, .8, (16, 30)) + np.linspace(-2, 2, 16)[:,None]
    row_effects[7:9] = 0
    calls = Counter()
    def losses(mask):
        calls[mask] += 1
        assert tuple(sorted(mask)) == mask
        interaction = 2.5 if 0 in mask and 9 in mask else 0.
        return 100-np.sum(row_effects[list(mask)],axis=0)-interaction

    progress = lambda *args, **kw: None
    _,expected,ref = permutation_scores("minshap","native",columns,losses,progress,seed=seed)
    full_masks = set(calls)
    calls.clear()
    context = SimpleNamespace(method="minshap",mode="native",columns=columns,yr=np.zeros(30),
                              seed=seed,losses=losses,progress=progress,minshap_support_only=True,
                              native_workers=workers,native_losses_batch=lambda masks:[losses(m) for m in masks])
    result = select(context)
    assert result.selected == expected
    assert result.scores is None
    assert set(calls) < full_masks and max(calls.values()) == 1
    diag = result.diagnostics
    assert len(calls) == diag["oracle_prefixes"] and diag["sampled_marginal_count"] < 4*16
    assert diag["permutation_sha256"] == ref["permutation_sha256"]
    assert all((v >= 0 and n == 4) or (v < 0 and cert >= 0) for v,n,cert in
               zip(diag["observed_minimum"],diag["per_feature_marginals"],diag["negative_certificate_path"]))
    assert columns[7] in expected and columns[8] in expected  # original >= convention on zero/zero
