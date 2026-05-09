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


_PROB_EPSILON = 1e-6  # floor for zero-bin probabilities; standard PSI/KL convention


def histogram(
    values: Sequence[float],
    *,
    bins: int = 10,
    range_: tuple[float, float] = (0.0, 1.0),
) -> list[float]:
    """Normalized histogram (probabilities sum to 1). Empty input → uniform zeros."""
    if len(values) == 0:
        return [0.0] * bins
    counts, _ = np.histogram(values, bins=bins, range=range_)
    total = counts.sum()
    if total == 0:
        return [0.0] * bins
    return [float(c) for c in counts / total]


def psi(p: Sequence[float], q: Sequence[float]) -> float:
    """Population Stability Index between two same-shape probability distributions.

    PSI ≈ 0 → distributions match.
    PSI < 0.10 → no significant shift.
    PSI 0.10–0.25 → moderate shift.
    PSI > 0.25 → major shift.
    """
    pa = np.asarray(p, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    if pa.shape != qa.shape:
        raise ValueError("psi: distributions must have the same shape")
    pa = np.where(pa <= 0, _PROB_EPSILON, pa)
    qa = np.where(qa <= 0, _PROB_EPSILON, qa)
    return float(np.sum((pa - qa) * np.log(pa / qa)))


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q). Both inputs treated as probability distributions of the same shape."""
    pa = np.asarray(p, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    if pa.shape != qa.shape:
        raise ValueError("kl_divergence: distributions must have the same shape")
    pa = np.where(pa <= 0, _PROB_EPSILON, pa)
    qa = np.where(qa <= 0, _PROB_EPSILON, qa)
    return float(np.sum(pa * np.log(pa / qa)))
