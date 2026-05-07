import math

import pytest

from drift.metrics import centroid, cosine, intra_set_avg_cosine


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
