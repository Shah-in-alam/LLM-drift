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
