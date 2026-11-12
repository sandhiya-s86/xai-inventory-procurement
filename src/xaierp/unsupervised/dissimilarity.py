"""Measuring dissimilarity (Unit 3).

Clustering never sees the data directly -- it only ever sees the dissimilarity
between pairs of points. The metric is therefore not a detail, it IS the model
assumption, and swapping it changes which SKUs look alike:

    Euclidean    absolute difference in every profile dimension.
                 Two SKUs are close only if they sell at a similar VOLUME.
    Manhattan    same idea, but a single wild dimension cannot dominate the
                 sum, so it is the more robust choice on skewed retail data.
    Cosine       compares the DIRECTION of the profile vector, ignoring its
                 length. A slow seller and a fast seller with the same weekly
                 shape are close. Volume information is thrown away.
    Correlation  cosine applied after removing each SKU's own mean, i.e. pure
                 SHAPE similarity with both level and scale removed.

Scale matters as much as the metric. The raw profile mixes units/day (0-400)
with a zero-rate (0-1), so an un-standardised Euclidean distance is almost
entirely the demand columns. `standardize` puts every column on the same
footing before any distance is computed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# --------------------------------------------------------------- scaling


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every column: z = (x - mu) / sigma.

    Returns ``(Z, mean, std)`` so the *training* statistics can be reused on
    any later data. Constant columns have sigma = 0; they are divided by 1
    instead of 0, which maps them to all-zeros -- a column carrying no
    information contributes no distance, which is the correct behaviour.
    """
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=0)
    safe = np.where(std > 0, std, 1.0)
    return (X - mean) / safe, mean, std


# ------------------------------------------------------------- distances


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance, sqrt(sum (a-b)^2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 / cityblock distance, sum |a-b|."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b)))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance, max |a-b|."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(angle). Zero vectors are defined to be maximally dissimilar.

    A zero vector has no direction, so the cosine is undefined rather than 0.
    Returning 1.0 keeps the function total and never produces a NaN that would
    silently poison a whole distance matrix.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Pearson r, i.e. cosine distance after centring each vector."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return cosine_distance(a - a.mean(), b - b.mean())


DISSIMILARITIES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_distance,
    "correlation": correlation_distance,
}


# ---------------------------------------------------------------- matrix


def dissimilarity_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Full symmetric n x n dissimilarity matrix with a zero diagonal.

    Euclidean and Manhattan are computed with vectorised broadcasting because
    the pairwise loop is the slow part of hierarchical clustering; the rest
    fall back to the elementwise functions above.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if metric not in DISSIMILARITIES:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(DISSIMILARITIES)}"
        )

    if metric == "euclidean":
        sq = np.sum(X**2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        # Floating point can make a squared distance a tiny negative number.
        D = np.sqrt(np.maximum(d2, 0.0))
    elif metric == "manhattan":
        D = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    else:
        f = DISSIMILARITIES[metric]
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = f(X[i], X[j])

    np.fill_diagonal(D, 0.0)
    # Enforce exact symmetry -- downstream linkage code relies on it.
    return (D + D.T) / 2.0


def condensed(D: np.ndarray) -> np.ndarray:
    """Upper triangle of a square distance matrix, in SciPy's condensed order."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def metric_comparison(X: np.ndarray, metrics: list[str] | None = None) -> list[dict]:
    """Summarise what each metric does to the same data.

    Reported so the metric choice is defended with numbers rather than
    asserted: the Spearman agreement against Euclidean shows how much the
    ranking of "which SKUs are similar" actually changes.
    """
    from scipy.stats import spearmanr

    metrics = metrics or ["euclidean", "manhattan", "chebyshev", "cosine", "correlation"]
    base = condensed(dissimilarity_matrix(X, "euclidean"))

    rows = []
    for m in metrics:
        d = condensed(dissimilarity_matrix(X, m))
        rho = 1.0 if m == "euclidean" else float(spearmanr(base, d).statistic)
        rows.append(
            {
                "metric": m,
                "mean_distance": float(d.mean()),
                "std_distance": float(d.std()),
                "min_distance": float(d.min()),
                "max_distance": float(d.max()),
                "spearman_vs_euclidean": rho,
            }
        )
    return rows
