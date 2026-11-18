"""Small exact-enumeration checks for a proposed conditional block minPI game.

This is an algebraic diagnostic, not a fitted selector or benchmark candidate.
"""
from itertools import combinations
import json

import numpy as np


def endpoints(target, common, removed, replacement):
    """Rows x outputs; common has one extra final axis for peer contributions."""
    difference = replacement-removed
    first = 2*np.sum(target*difference)+np.sum(removed**2)-np.sum(replacement**2)
    last = first-2*np.sum(common.sum(axis=-1)*difference)
    return np.array([first, last])


def enumerated(target, common, removed, replacement):
    result = []
    for size in range(common.shape[-1]+1):
        values = []
        for subset in combinations(range(common.shape[-1]), size):
            prediction = common[:, :, subset].sum(axis=-1)
            values.append(np.sum((target-prediction-removed)**2)-
                          np.sum((target-prediction-replacement)**2))
        result.append(np.mean(values))
    return np.array(result)


def main():
    rng = np.random.default_rng(481107)
    checked = 0
    for rows in (1, 5):
        for outputs in (1, 3):
            for peers in range(6):
                target = rng.normal(size=(rows, outputs))
                common = rng.normal(size=(rows, outputs, peers))
                removed = rng.normal(size=target.shape)
                replacement = rng.normal(size=target.shape)
                direct = enumerated(target, common, removed, replacement)
                bounds = endpoints(target, common, removed, replacement)
                analytic = np.linspace(*bounds, peers+1) if peers else bounds[:1]
                np.testing.assert_allclose(direct, analytic, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(direct.min(), bounds.min(), rtol=1e-12, atol=1e-12)
                checked += 1
    # Two separately good swaps can conflict when applied together.
    individual_gain = 1-(1-1.1)**2
    simultaneous_gain = 1-(1-2.2)**2
    assert individual_gain > 0 and simultaneous_gain < 0
    # A weaker singleton is complementary to an already retained feature.
    target = np.ones((2, 1))
    common = np.array([[[1.]], [[0.]]])
    removed, replacement = np.array([[1.], [0.]]), np.array([[0.], [.9]])
    complement = endpoints(target, common, removed, replacement)
    np.testing.assert_allclose(complement, [-.01, 1.99], atol=1e-14)
    # Preserving one duplicate's frozen half coefficient changes predictions;
    # transferring the removed coefficient to its exact duplicate preserves them.
    duplicated = np.ones((3, 1))
    frozen_delete_loss = np.sum((duplicated-.5*duplicated)**2)
    transfer_loss = np.sum((duplicated-duplicated)**2)
    assert frozen_delete_loss == .75 and transfer_loss == 0
    print(json.dumps(dict(
        finite_games_checked=checked,
        maximum_peers=5,
        all_peer_count_expectations_equal_endpoint_interpolation=True,
        individually_good_swap_gain=individual_gain,
        joint_bad_swap_gain=simultaneous_gain,
        complementary_single_swap_endpoints=complement.tolist(),
        duplicate_frozen_deletion_loss=frozen_delete_loss,
        duplicate_coefficient_transfer_loss=transfer_loss,
    ), indent=2))


if __name__ == "__main__":
    main()
