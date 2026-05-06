import math

from drift.metrics import cosine


def test_identical_vectors():
    assert math.isclose(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_orthogonal_vectors():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_negated_vectors():
    assert math.isclose(cosine([1.0, 2.0], [-1.0, -2.0]), -1.0)


def test_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
