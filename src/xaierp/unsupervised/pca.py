"""Principal Component Analysis and choosing the number of latent dimensions.

PCA finds the orthogonal directions along which the data varies most. The
first component is the direction of maximum variance, the second is the
maximum-variance direction orthogonal to the first, and so on.

Computed via the SVD of the centred data rather than by eigendecomposing the
covariance matrix. Both are mathematically equivalent -- if X = U S V^T then
the columns of V are the eigenvectors of X^T X and the eigenvalues are
s_i^2 / (n-1) -- but forming X^T X squares the condition number and loses
precision on near-collinear columns, which the SKU profile matrix has plenty
of (rolling_mean_7 and rolling_mean_28 are highly correlated by construction).

Sign convention: an eigenvector is only defined up to sign, so every library
picks a rule. This one matches scikit-learn's: flip each component so that
its largest-magnitude loading is positive. Without a rule, two runs can
return components that differ by a sign and look like a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    """PCA by SVD of the centred data matrix."""

    n_components: int | None = None

    # Fitted state.
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None          # (n_components, n_features)
    explained_variance_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    singular_values_: np.ndarray | None = None
    n_samples_: int = 0

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.n_samples_ = n

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Deterministic sign: largest-magnitude entry of each component is +ve.
        signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        signs[signs == 0] = 1.0
        Vt = Vt * signs[:, None]

        total_components = min(n, d)
        k = total_components if self.n_components is None else int(self.n_components)
        if k > total_components:
            raise ValueError(
                f"n_components={k} exceeds min(n_samples, n_features)={total_components}"
            )

        var = (S**2) / (n - 1)
        total_var = var.sum()

        self.singular_values_ = S[:k]
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / total_var
        # Kept in full so a scree plot can show the whole spectrum even when
        # only the first k components are retained.
        self._all_explained_variance = var
        self._all_ratio = var / total_var
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto the principal axes: the latent scores."""
        if self.components_ is None:
            raise RuntimeError("Call fit() first.")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back from latent scores to the original space (lossy unless k = d)."""
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error after the round trip."""
        X = np.asarray(X, dtype=float)
        return float(np.mean((X - self.inverse_transform(self.transform(X))) ** 2))


def scree_table(X: np.ndarray) -> list[dict]:
    """Eigenvalue, explained variance and cumulative variance per component.

    The scree plot is this table's `eigenvalue` column drawn in order; the
    "elbow" where it levels off is the usual visual stopping rule.
    """
    p = PCA().fit(X)
    ratio = p._all_ratio
    var = p._all_explained_variance
    cum = np.cumsum(ratio)
    return [
        {
            "component": i + 1,
            "eigenvalue": float(var[i]),
            "explained_variance_ratio": float(ratio[i]),
            "cumulative_variance_ratio": float(cum[i]),
        }
        for i in range(len(var))
    ]


def choose_n_components(
    X: np.ndarray, variance_threshold: float = 0.90
) -> dict[str, int | float]:
    """Three published rules for choosing the number of latent dimensions.

    They disagree on purpose -- reporting all three is more honest than
    quoting whichever one gives the number you wanted:

    kaiser
        Keep components with eigenvalue > 1. The rule only means anything on
        STANDARDISED data, where total variance equals the number of features
        so an average component has eigenvalue exactly 1; a component below
        that explains less than one original variable's worth and has not
        earned its place. (Known to over-retain on wide matrices.)
    variance
        Smallest k reaching `variance_threshold` of total variance. Direct
        and tunable, but the threshold is a choice, not a finding.
    elbow
        Largest drop between consecutive eigenvalues -- the scree elbow, made
        reproducible instead of eyeballed.
    """
    rows = scree_table(X)
    eig = np.array([r["eigenvalue"] for r in rows])
    cum = np.array([r["cumulative_variance_ratio"] for r in rows])

    kaiser = int(np.sum(eig > 1.0))
    variance = int(np.argmax(cum >= variance_threshold) + 1)

    if len(eig) >= 3:
        drops = eig[:-1] - eig[1:]
        elbow = int(np.argmax(drops) + 1)
    else:
        elbow = len(eig)

    return {
        "kaiser": max(1, kaiser),
        "variance": max(1, variance),
        "elbow": max(1, elbow),
        "variance_threshold": float(variance_threshold),
        "n_features": int(X.shape[1]),
    }
