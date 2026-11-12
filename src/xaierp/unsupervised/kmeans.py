"""K-means clustering (Unit 3), implemented from the algorithm in the notes.

    1. choose k initial centroids
    2. assign every point to the nearest centroid (Euclidean)
    3. recompute each centroid as the mean of the points assigned to it
    4. repeat 2-3 until no point changes cluster

Two properties are worth stating because the tests check them:

* Each of steps 2 and 3 is the exact minimiser of the inertia
  J = sum_i ||x_i - mu_{c(i)}||^2 holding the other fixed, so **inertia can
  never increase between iterations**. That is why the algorithm terminates.
* It only converges to a LOCAL minimum. Different initialisations give
  different answers, which is why `n_init` restarts are run and the best
  (lowest-inertia) one is kept, and why k-means++ seeding is used instead of
  picking k points uniformly at random.

K-means is a HARD assignment method: every SKU belongs to exactly one segment
with no measure of how confidently. That limitation is the entire motivation
for the Gaussian mixture in gmm.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, k) matrix of squared Euclidean distances from each point to each centroid."""
    return np.maximum(
        np.sum(X**2, axis=1)[:, None]
        + np.sum(C**2, axis=1)[None, :]
        - 2.0 * (X @ C.T),
        0.0,
    )


def kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids out on purpose.

    The first centre is uniform; every later centre is drawn with probability
    proportional to its squared distance from the nearest centre already
    chosen. Picking k points uniformly tends to land several of them inside
    the same dense blob, which wastes a cluster and is the usual cause of a
    bad local minimum.
    """
    n = X.shape[0]
    centres = np.empty((k, X.shape[1]), dtype=float)
    centres[0] = X[rng.integers(n)]

    closest = _pairwise_sq_dist(X, centres[:1]).ravel()
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:  # all points identical -- any choice is as good
            centres[i] = X[rng.integers(n)]
        else:
            centres[i] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, _pairwise_sq_dist(X, centres[i : i + 1]).ravel())
    return centres


@dataclass
class KMeans:
    """Lloyd's algorithm with k-means++ seeding and multiple restarts."""

    n_clusters: int
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-6
    random_state: int = 42

    # Fitted state.
    cluster_centers_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    inertia_: float = float("nan")
    n_iter_: int = 0
    inertia_history_: list[float] | None = None

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        C = kmeans_plusplus(X, self.n_clusters, rng)
        labels = np.full(X.shape[0], -1, dtype=int)
        history: list[float] = []

        for it in range(1, self.max_iter + 1):
            d2 = _pairwise_sq_dist(X, C)
            new_labels = np.argmin(d2, axis=1)
            inertia = float(d2[np.arange(X.shape[0]), new_labels].sum())
            history.append(inertia)

            converged = np.array_equal(new_labels, labels)
            labels = new_labels

            # Recompute centroids. An empty cluster is re-seeded onto the
            # point currently worst-served by its own centroid; leaving it
            # empty would silently return fewer than k clusters.
            new_C = np.empty_like(C)
            for j in range(self.n_clusters):
                mask = labels == j
                if mask.any():
                    new_C[j] = X[mask].mean(axis=0)
                else:
                    worst = int(np.argmax(d2[np.arange(X.shape[0]), labels]))
                    new_C[j] = X[worst]

            shift = float(np.sqrt(np.sum((new_C - C) ** 2, axis=1)).max())
            C = new_C
            if converged or shift <= self.tol:
                break

        d2 = _pairwise_sq_dist(X, C)
        labels = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(X.shape[0]), labels].sum())
        history.append(inertia)
        return C, labels, inertia, it, history

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds n_samples={X.shape[0]}"
            )

        best = None
        for r in range(self.n_init):
            rng = np.random.default_rng(self.random_state + r)
            out = self._single_run(X, rng)
            if best is None or out[2] < best[2]:
                best = out

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_, \
            self.inertia_history_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() first.")
        return np.argmin(
            _pairwise_sq_dist(np.asarray(X, dtype=float), self.cluster_centers_),
            axis=1,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


# ----------------------------------------------------------- elbow method


def elbow_curve(
    X: np.ndarray,
    k_values: range | list[int],
    n_init: int = 10,
    random_state: int = 42,
) -> list[dict]:
    """Inertia (and its rate of change) for each candidate k.

    Inertia always falls as k rises -- at k = n it reaches zero -- so the
    smallest inertia is never the answer. The elbow is the k after which the
    *improvement* stops being worth the extra cluster.
    """
    rows = []
    prev = None
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)
        drop = float("nan") if prev is None else prev - km.inertia_
        pct = float("nan") if prev is None else 100.0 * (prev - km.inertia_) / prev
        rows.append(
            {
                "k": int(k),
                "inertia": km.inertia_,
                "inertia_drop": drop,
                "inertia_drop_pct": pct,
                "n_iter": km.n_iter_,
            }
        )
        prev = km.inertia_
    return rows


def knee_point(k_values: list[int], inertias: list[float]) -> int:
    """Pick the elbow as the point furthest from the chord joining the ends.

    Reading an elbow off a chart by eye is not reproducible. This is the
    standard geometric rule: draw the straight line from (k_min, J_min) to
    (k_max, J_max) and take the k whose curve point lies furthest below it.
    """
    k = np.asarray(k_values, dtype=float)
    j = np.asarray(inertias, dtype=float)
    if len(k) < 3:
        return int(k[0])

    # Normalise both axes so the result does not depend on units.
    kn = (k - k.min()) / (k.max() - k.min())
    jn = (j - j.min()) / (j.max() - j.min()) if j.max() > j.min() else np.zeros_like(j)

    p0 = np.array([kn[0], jn[0]])
    p1 = np.array([kn[-1], jn[-1]])
    line = p1 - p0
    line = line / np.linalg.norm(line)

    dists = []
    for x, y in zip(kn, jn):
        v = np.array([x, y]) - p0
        proj = np.dot(v, line) * line
        dists.append(float(np.linalg.norm(v - proj)))
    return int(k[int(np.argmax(dists))])
