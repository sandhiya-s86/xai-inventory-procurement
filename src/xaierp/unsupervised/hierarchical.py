"""Hierarchical (agglomerative) clustering with a dendrogram (Unit 3).

Agglomerative = bottom-up. Start with every SKU as its own cluster, then
repeatedly merge the two closest clusters until one remains. The sequence of
merges IS the model: a dendrogram whose height at each join is the distance
at which those two groups became one. Unlike K-means you do not have to
commit to k in advance -- you cut the tree afterwards, at whatever height
the problem calls for.

"Closest" needs a definition for *clusters*, not just points, and the choice
changes the answer completely:

    single     distance between the two NEAREST members.
               Finds long stringy shapes; prone to "chaining", where two
               real groups get bridged by a line of intermediate points.
    complete   distance between the two FURTHEST members.
               Produces compact, roughly equal-diameter clusters.
    average    mean distance over all cross-pairs (UPGMA). The compromise.
    ward       merge the pair that increases within-cluster variance least.
               Closest in spirit to K-means, and usually the best default on
               standardised numeric profiles like ours.

All four are implemented through one Lance-Williams update, which gives the
distance from a newly merged cluster to every other cluster as a weighted
combination of the three distances that already existed -- so the full
distance matrix never has to be recomputed after a merge.

The output follows SciPy's linkage format, an (n-1, 4) array whose rows are
``[idx_a, idx_b, distance, n_observations]`` with cluster n+i denoting the
cluster created at step i. Matching that layout means SciPy's `dendrogram`
can plot our tree directly, and the test-suite can compare our linkage
against `scipy.cluster.hierarchy.linkage` row for row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dissimilarity import dissimilarity_matrix

LINKAGES = ("single", "complete", "average", "ward")


def _lance_williams(
    method: str,
    d_ik: np.ndarray,
    d_jk: np.ndarray,
    d_ij: float,
    n_i: int,
    n_j: int,
    n_k: np.ndarray,
) -> np.ndarray:
    """Distance from the merged cluster (i+j) to every other cluster k."""
    if method == "single":
        return np.minimum(d_ik, d_jk)
    if method == "complete":
        return np.maximum(d_ik, d_jk)
    if method == "average":
        return (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
    if method == "ward":
        # Ward operates on SQUARED Euclidean distances internally; the inputs
        # and outputs here are ordinary distances, hence the square/sqrt.
        total = n_i + n_j + n_k
        num = (
            (n_i + n_k) * d_ik**2
            + (n_j + n_k) * d_jk**2
            - n_k * (d_ij**2)
        )
        return np.sqrt(np.maximum(num / total, 0.0))
    raise ValueError(f"Unknown linkage {method!r}. Available: {LINKAGES}")


def linkage(X: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    """Build the (n-1, 4) linkage matrix by repeated nearest-pair merging.

    Ward's update is only valid for Euclidean distances -- the variance
    argument it is derived from has no meaning under, say, cosine -- so that
    combination is rejected rather than silently producing a number.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage is only defined for the Euclidean metric.")
    if n < 2:
        raise ValueError("Need at least 2 observations to build a hierarchy.")

    D = dissimilarity_matrix(X, metric).copy()
    np.fill_diagonal(D, np.inf)          # a cluster is never its own nearest

    active = np.ones(n, dtype=bool)      # which slots are still live clusters
    sizes = np.ones(n, dtype=int)
    cluster_id = np.arange(n)            # slot -> current linkage cluster id

    Z = np.zeros((n - 1, 4), dtype=float)

    for step in range(n - 1):
        # Nearest pair among the live clusters.
        masked = np.where(active[:, None] & active[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))
        dist = float(masked[i, j])

        n_i, n_j = sizes[i], sizes[j]
        a, b = cluster_id[i], cluster_id[j]
        # SciPy orders each row with the smaller cluster id first.
        Z[step] = [min(a, b), max(a, b), dist, n_i + n_j]

        # Update distances from the merged cluster (stored in slot i) to all
        # other live clusters, then retire slot j.
        others = active.copy()
        others[i] = others[j] = False
        if others.any():
            D[i, others] = D[others, i] = _lance_williams(
                method, D[i, others], D[j, others], dist, n_i, n_j, sizes[others]
            )

        sizes[i] = n_i + n_j
        cluster_id[i] = n + step
        active[j] = False
        D[j, :] = D[:, j] = np.inf
        D[i, i] = np.inf

    return Z


def fcluster(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut the dendrogram to leave exactly `n_clusters` groups.

    Replays the first ``n - n_clusters`` merges with a union-find structure:
    the merges are already sorted by height, so stopping early is the same as
    cutting the tree horizontally. Labels are renumbered 0..k-1 in order of
    first appearance so they are comparable with the other methods.
    """
    n = Z.shape[0] + 1
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")

    parent = list(range(2 * n - 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for step in range(n - n_clusters):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        new = n + step
        parent[find(a)] = new
        parent[find(b)] = new

    roots = [find(i) for i in range(n)]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([order[r] for r in roots], dtype=int)


def cophenetic_distances(Z: np.ndarray) -> np.ndarray:
    """Cophenetic distance: the merge height at which each pair first joins.

    This is the distance matrix the dendrogram *implies*. Comparing it with
    the original distances (below) measures how much structure the tree threw
    away in order to be a tree.
    """
    n = Z.shape[0] + 1
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    C = np.zeros((n, n), dtype=float)

    for step in range(n - 1):
        a, b, h = int(Z[step, 0]), int(Z[step, 1]), float(Z[step, 2])
        ma, mb = members[a], members[b]
        for p in ma:
            for q in mb:
                C[p, q] = C[q, p] = h
        members[n + step] = ma + mb
    return C


def cophenetic_correlation(Z: np.ndarray, D: np.ndarray) -> float:
    """Pearson r between the original and cophenetic distances.

    Closer to 1 means the dendrogram is a faithful summary of the real
    pairwise distances. It is the standard way to compare linkage methods on
    the same data without reference to any particular cut.
    """
    C = cophenetic_distances(Z)
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.corrcoef(D[iu], C[iu])[0, 1])


@dataclass
class AgglomerativeClustering:
    """scikit-learn-shaped wrapper around `linkage` + `fcluster`."""

    n_clusters: int
    linkage_method: str = "ward"
    metric: str = "euclidean"

    linkage_matrix_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    cophenetic_corr_: float = float("nan")

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        X = np.asarray(X, dtype=float)
        self.linkage_matrix_ = linkage(X, self.linkage_method, self.metric)
        self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters)
        self.cophenetic_corr_ = cophenetic_correlation(
            self.linkage_matrix_, dissimilarity_matrix(X, self.metric)
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


def compare_linkages(
    X: np.ndarray, n_clusters: int, metric: str = "euclidean"
) -> list[dict]:
    """Fit every linkage method and report how different the answers are.

    Ward is excluded for non-Euclidean metrics, where it is undefined.
    """
    from .evaluation import adjusted_rand_index, silhouette_score

    X = np.asarray(X, dtype=float)
    D = dissimilarity_matrix(X, metric)
    methods = [m for m in LINKAGES if not (m == "ward" and metric != "euclidean")]

    rows, labels_by_method = [], {}
    for m in methods:
        Z = linkage(X, m, metric)
        labels = fcluster(Z, n_clusters)
        labels_by_method[m] = labels
        sizes = np.bincount(labels, minlength=n_clusters)
        rows.append(
            {
                "linkage": m,
                "metric": metric,
                "n_clusters": n_clusters,
                "cophenetic_corr": cophenetic_correlation(Z, D),
                "silhouette": silhouette_score(X, labels, metric=metric),
                "largest_cluster_share": float(sizes.max() / len(labels)),
                "smallest_cluster_size": int(sizes.min()),
                "merge_height_max": float(Z[:, 2].max()),
            }
        )

    # Agreement with Ward (or the first method available) as the reference.
    ref = "ward" if "ward" in labels_by_method else methods[0]
    for row in rows:
        row["ari_vs_" + ref] = adjusted_rand_index(
            labels_by_method[ref], labels_by_method[row["linkage"]]
        )
    return rows
