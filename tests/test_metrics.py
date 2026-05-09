import math

import pytest

from drift.metrics import (
    centroid,
    cosine,
    histogram,
    intra_set_avg_cosine,
    kl_divergence,
    psi,
)


def test_identical_vectors():
    assert math.isclose(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_orthogonal_vectors():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_negated_vectors():
    assert math.isclose(cosine([1.0, 2.0], [-1.0, -2.0]), -1.0)


def test_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_centroid_averages_vectors():
    result = centroid([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert result == [3.0, 4.0]


def test_centroid_rejects_empty():
    with pytest.raises(ValueError):
        centroid([])


def test_intra_set_avg_cosine_single_vector_is_one():
    assert intra_set_avg_cosine([[1.0, 2.0, 3.0]]) == 1.0


def test_intra_set_avg_cosine_identical_pair_is_one():
    assert math.isclose(intra_set_avg_cosine([[1.0, 0.0], [1.0, 0.0]]), 1.0)


def test_intra_set_avg_cosine_orthogonal_pair_is_zero():
    assert math.isclose(intra_set_avg_cosine([[1.0, 0.0], [0.0, 1.0]]), 0.0, abs_tol=1e-9)


def test_histogram_normalizes_to_one():
    h = histogram([0.05, 0.15, 0.55, 0.95], bins=10)
    assert len(h) == 10
    assert math.isclose(sum(h), 1.0)


def test_histogram_empty_input_returns_zeros():
    assert histogram([], bins=5) == [0.0] * 5


def test_psi_identical_distributions_is_zero():
    p = [0.1, 0.2, 0.3, 0.2, 0.2]
    assert math.isclose(psi(p, p), 0.0, abs_tol=1e-9)


def test_psi_signals_shift():
    p = [0.0, 0.0, 0.0, 0.0, 1.0]  # all probability mass in last bin
    q = [1.0, 0.0, 0.0, 0.0, 0.0]  # all in first bin
    # PSI for a complete shift between two well-separated distributions is large.
    assert psi(p, q) > 1.0


def test_psi_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [0.3, 0.3, 0.4])


def test_kl_divergence_identical_is_zero():
    p = [0.25, 0.25, 0.25, 0.25]
    assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1e-9)


def test_kl_divergence_asymmetric():
    # KL(p||q) is not symmetric in general.
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) != kl_divergence(q, p)
