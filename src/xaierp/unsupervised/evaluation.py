"""Evaluating the output of clustering methods (Unit 3).

Clustering is unsupervised, so there is no accuracy to report. The question
"is this clustering any good?" has to be answered either INTERNALLY -- from
the geometry of the result alone -- or RELATIVELY, by asking whether two
methods agree.

Internal indices implemented here:

    silhouette          per point: (b - a) / max(a, b), where a is the mean
                        distance to its own cluster and b the mean distance
                        to the nearest OTHER cluster. Ranges -1..1.
                        ~1 = well inside its cluster, ~0 = on the boundary,
                        negative = probably in the wrong cluster. HIGHER better.
    Davies-Bouldin      mean over clusters of the worst-case ratio
                        (spread_i + spread_j) / separation_ij. LOWER better.
    Calinski-Harabasz   between-cluster dispersion over within-cluster
                        dispersion, scaled by the degrees of freedom --
                        essentially an F-statistic. HIGHER better.

Relative index:

    Adjusted Rand Index agreement between two labellings, corrected for the
                        agreement you would expect by chance. 1 = identical
                        partitions, 0 = chance, negative = worse than chance.
                        Label PERMUTATIONS do not affect it, which is what
                        makes it usable at all: cluster 0 from K-means and
                        cluster 2 from Ward may be the same group.

Caveat worth stating up front: silhouette and Calinski-Harabasz both reward
compact spherical clusters, which is exactly what K-means produces by
construction. Using them to decide that K-means "beat" a density- or
mixture-based method is close to circular. That is why the comparison table
in scripts/run_unsupervised.py reports ARI alongside them.
"""

from __future__ import annotations

import numpy as np

from .dissimilarity import dissimilarity_matrix


def silhouette_samples(
    X: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Silhouette coefficient for every point.

    A point that is the only member of its cluster gets s = 0 by convention:
    a mean within-cluster distance over zero other members is undefined, and
    0 ("on the boundary") is the neutral answer.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = dissimilarity_matrix(X, metric)
    n = len(labels)
    uniq = np.unique(labels)

    if len(uniq) < 2:
        raise ValueError("Silhouette needs at least 2 clusters.")

    sil = np.zeros(n, dtype=float)
    for i in range(n):
        own = labels == i * 0 + labels[i]
        n_own = int(own.sum())
        if n_own <= 1:
            sil[i] = 0.0
            continue

        # a: mean distance to the other members of its own cluster.
        a = float((D[i, own].sum()) / (n_own - 1))

        # b: smallest mean distance to any other cluster.
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, float(D[i, other].mean()))

        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return sil


def silhouette_score(
    X: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> float:
    """Mean silhouette across all points."""
    return float(np.mean(silhouette_samples(X, labels, metric)))


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index -- lower is better."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2:
        raise ValueError("Davies-Bouldin needs at least 2 clusters.")

    centroids = np.array([X[labels == c].mean(axis=0) for c in uniq])
    # S_i: mean distance from the members of cluster i to its centroid.
    spread = np.array(
        [
            float(np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1)))
            for i, c in enumerate(uniq)
        ]
    )

    total = 0.0
    for i in range(k):
        worst = 0.0
        for j in range(k):
            if i == j:
                continue
            sep = float(np.linalg.norm(centroids[i] - centroids[j]))
            if sep > 0:
                worst = max(worst, (spread[i] + spread[j]) / sep)
        total += worst
    return float(total / k)


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz (variance-ratio) index -- higher is better."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2 or k >= n:
        raise ValueError("Calinski-Harabasz needs 2 <= n_clusters < n_samples.")

    overall = X.mean(axis=0)
    between = within = 0.0
    for c in uniq:
        members = X[labels == c]
        centroid = members.mean(axis=0)
        between += len(members) * float(np.sum((centroid - overall) ** 2))
        within += float(np.sum((members - centroid) ** 2))

    if within == 0:
        return float("inf")
    return float((between / (k - 1)) / (within / (n - k)))


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Chance-corrected agreement between two partitions of the same points.

    Built from the contingency table of the two labellings. The "adjusted"
    part subtracts the agreement expected if the two partitions were drawn
    independently with the same cluster sizes -- without it, splitting the
    data into many tiny clusters inflates the raw Rand index towards 1.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("Both labellings must cover the same points.")

    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)), dtype=float)
    np.add.at(table, (ia, ib), 1)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(float(len(a)))

    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:          # both partitions trivial
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, name: str = "", metric: str = "euclidean"
) -> dict:
    """One row of the clustering comparison table."""
    labels = np.asarray(labels)
    sizes = np.bincount(labels - labels.min(), minlength=len(np.unique(labels)))
    return {
        "method": name,
        "n_clusters": int(len(np.unique(labels))),
        "silhouette": silhouette_score(X, labels, metric),
        "davies_bouldin": davies_bouldin(X, labels),
        "calinski_harabasz": calinski_harabasz(X, labels),
        "largest_cluster_share": float(sizes.max() / len(labels)),
        "smallest_cluster_size": int(sizes.min()),
    }
