from __future__ import annotations

import numpy as np

from scripts.build_strain_semantics import classical_mds


def test_classical_mds_reconstructs_euclidean_distances():
    points = np.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    embedded, eigenvalues = classical_mds(distances, dimensions=2)
    reconstructed = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
    assert len(eigenvalues) == 2
    assert np.allclose(reconstructed, distances, atol=1e-8)


def test_classical_mds_is_deterministic_and_pads_dimensions():
    distances = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    first, _ = classical_mds(distances, dimensions=4)
    second, _ = classical_mds(distances, dimensions=4)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 1:], 0.0)
