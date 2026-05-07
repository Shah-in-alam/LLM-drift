from collections.abc import Sequence

import numpy as np


def cosine(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector is all zeros."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Mean vector across a non-empty list of equal-length vectors."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("centroid expects a non-empty 2D set of vectors")
    return [float(x) for x in arr.mean(axis=0)]


def intra_set_avg_cosine(vectors: Sequence[Sequence[float]]) -> float:
    """Average pairwise cosine across a set of vectors. Useful as a noise floor.

    Returns 1.0 for a single-vector set (no pairs to average — perfectly stable).
    """
    n = len(vectors)
    if n <= 1:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine(vectors[i], vectors[j]))
    return float(np.mean(sims))
