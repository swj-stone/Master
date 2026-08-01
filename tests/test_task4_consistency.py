"""Repository-level consistency checks for the canonical Task 4 API."""

from __future__ import annotations

import minppi_mofs


def test_package_exports_canonical_guidance_api() -> None:
    canonical_names = (
        "estimate_minppi_mi_correlation_guided",
        "build_mi_correlation_guidance",
        "get_correlation_neighbors",
        "rank_by_mi",
        "filter_top_by_mi",
    )

    assert all(hasattr(minppi_mofs, name) for name in canonical_names)
