"""Mixtures of Gaussians and the EM algorithm (Unit 3).

K-means assigns each SKU to exactly one segment. A Gaussian mixture is the
SOFT version of the same idea: every SKU gets a *responsibility* r_nk, the
posterior probability that component k generated it, and those responsibilities
sum to 1 across components. A SKU sitting between "steady seller" and "spiky
seasonal" is reported as 0.55 / 0.45 instead of being forced into one box --
and for an inventory decision that ambiguity is information, not noise.

The model
---------
    p(x) = sum_k  pi_k  N(x | mu_k, Sigma_k)         sum_k pi_k = 1

Fitted by Expectation-Maximization:

    E-step   r_nk = pi_k N(x_n | mu_k, Sigma_k) / sum_j pi_j N(x_n | mu_j, Sigma_j)
    M-step   N_k   = sum_n r_nk
             pi_k  = N_k / N
             mu_k  = (1/N_k) sum_n r_nk x_n
             Sig_k = (1/N_k) sum_n r_nk (x_n - mu_k)(x_n - mu_k)^T

The guarantee that matters: **each full EM iteration cannot decrease the
observed-data log-likelihood**. `fit` records the log-likelihood at every
iteration and tests/test_unit3_unsupervised.py asserts the sequence is
non-decreasing -- if a future edit breaks the maths, that test fails rather
than the model quietly getting worse.

Numerical care
--------------
Densities are computed in the log domain via the Cholesky factor, and
responsibilities are normalised with log-sum-exp. Doing it naively underflows
to 0/0 on the first iteration in anything above a handful of dimensions.
Each covariance also gets `reg_covar` added to its diagonal: without it a
component that captures a single point collapses to zero variance and the
likelihood diverges to +infinity, which is a real failure mode of ML
estimation for mixtures, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .kmeans import KMeans


def _log_gaussian(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(x | mu, Sigma) for every row of X, via the Cholesky factor.

    Using Cholesky instead of inverting Sigma gives both the solve and the
    log-determinant in one factorisation, and fails loudly (LinAlgError) if
    the covariance has stopped being positive definite.
    """
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    diff = X - mu
    # solve L z = diff^T  =>  z^T z  is the Mahalanobis distance
    z = np.linalg.solve(L, diff.T)
    maha = np.sum(z**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


@dataclass
class GaussianMixtureEM:
    """Full-covariance Gaussian mixture fitted by EM."""

    n_components: int
    max_iter: int = 200
    tol: float = 1e-6
    reg_covar: float = 1e-6
    random_state: int = 42
    init: str = "kmeans"          # "kmeans" | "random"

    # Fitted state.
    weights_: np.ndarray | None = None
    means_: np.ndarray | None = None
    covariances_: np.ndarray | None = None
    responsibilities_: np.ndarray | None = None
    loglik_history_: list[float] = field(default_factory=list)
    n_iter_: int = 0
    converged_: bool = False

    # ------------------------------------------------------------- init

    def _initialise(self, X: np.ndarray) -> None:
        n, d = X.shape
        K = self.n_components
        rng = np.random.default_rng(self.random_state)

        if self.init == "kmeans":
            # Seeding from K-means is standard: EM is only guaranteed to find
            # a local optimum, and a hard-clustering warm start reliably lands
            # in a better one than random means do.
            km = KMeans(n_clusters=K, n_init=5, random_state=self.random_state).fit(X)
            labels = km.labels_
            self.means_ = km.cluster_centers_.copy()
        else:
            labels = rng.integers(0, K, size=n)
            self.means_ = X[rng.choice(n, size=K, replace=False)].copy()

        self.weights_ = np.full(K, 1.0 / K)
        self.covariances_ = np.empty((K, d, d))
        global_cov = np.cov(X.T, bias=True) + self.reg_covar * np.eye(d)
        for k in range(K):
            mask = labels == k
            if mask.sum() > d:
                self.covariances_[k] = (
                    np.cov(X[mask].T, bias=True) + self.reg_covar * np.eye(d)
                )
            else:
                # Too few points to estimate a covariance -- fall back to the
                # global one rather than producing a singular matrix.
                self.covariances_[k] = global_cov
            self.weights_[k] = max(mask.mean(), 1.0 / n)
        self.weights_ /= self.weights_.sum()

    # ---------------------------------------------------------- E and M

    def _log_weighted_densities(self, X: np.ndarray) -> np.ndarray:
        """(n, K) matrix of log( pi_k N(x_n | mu_k, Sigma_k) )."""
        K = self.n_components
        out = np.empty((X.shape[0], K))
        for k in range(K):
            out[:, k] = np.log(self.weights_[k]) + _log_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
        return out

    def e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Responsibilities r_nk and the observed-data log-likelihood."""
        log_w = self._log_weighted_densities(X)
        log_norm = logsumexp(log_w, axis=1)           # log p(x_n)
        log_resp = log_w - log_norm[:, None]
        return np.exp(log_resp), float(log_norm.sum())

    def m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Closed-form maximisation given the responsibilities."""
        n, d = X.shape
        # Nk can be ~0 for a dying component; the floor stops a divide-by-zero
        # turning the means into NaN and taking the whole fit down with it.
        Nk = np.maximum(resp.sum(axis=0), 1e-10)

        self.weights_ = Nk / n
        self.means_ = (resp.T @ X) / Nk[:, None]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (resp[:, k, None] * diff).T @ diff / Nk[k]
            self.covariances_[k] = cov + self.reg_covar * np.eye(d)

    # ------------------------------------------------------------- fit

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        X = np.asarray(X, dtype=float)
        self._initialise(X)
        self.loglik_history_ = []

        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            resp, ll = self.e_step(X)
            self.loglik_history_.append(ll)
            self.m_step(X, resp)
            self.n_iter_ = it

            # Convergence on the RELATIVE change, so the threshold means the
            # same thing whether the log-likelihood is -50 or -500,000.
            if np.isfinite(prev) and abs(ll - prev) <= self.tol * abs(prev):
                self.converged_ = True
                break
            prev = ll

        # One last E-step so the stored responsibilities match the final
        # parameters rather than the ones from before the last M-step.
        self.responsibilities_, final_ll = self.e_step(X)
        self.loglik_history_.append(final_ll)
        return self

    # ------------------------------------------------------- inference

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.e_step(np.asarray(X, dtype=float))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard labels -- the argmax component. Throws the soft part away."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Total observed-data log-likelihood."""
        return self.e_step(np.asarray(X, dtype=float))[1]

    # --------------------------------------------------- model selection

    def n_parameters(self) -> int:
        """Free parameters: means + full covariances + mixing weights.

        A d x d covariance is symmetric, so it has d(d+1)/2 free entries, not
        d^2. The weights lose one degree of freedom to the sum-to-one
        constraint.
        """
        K, d = self.n_components, self.means_.shape[1]
        return int(K * d + K * d * (d + 1) / 2 + (K - 1))

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion -- LOWER is better."""
        n = np.asarray(X).shape[0]
        return float(-2.0 * self.score(X) + self.n_parameters() * np.log(n))

    def aic(self, X: np.ndarray) -> float:
        """Akaike information criterion -- LOWER is better."""
        return float(-2.0 * self.score(X) + 2.0 * self.n_parameters())


# ------------------------------------------------ the alternative view of EM


def em_lower_bound(
    X: np.ndarray, model: "GaussianMixtureEM", q: np.ndarray
) -> dict[str, float]:
    """The variational (lower-bound) view of EM from the Unit 3 notes.

    Instead of "compute responsibilities, then update parameters", EM can be
    read as coordinate ascent on a single objective -- the free energy

        F(q, theta) = E_q[ log p(x, z | theta) ] + H(q)
                    = log p(x | theta) - KL( q(z) || p(z | x, theta) )

    Read that second line carefully, because it is the whole point:

    * F is a LOWER BOUND on log p(x | theta) for any q, since KL >= 0.
    * The E-step maximises F over q. The maximiser is exactly
      q(z) = p(z | x, theta), the responsibilities -- at which point KL = 0
      and **the bound is tight**: F equals the log-likelihood exactly.
    * The M-step maximises F over theta with q held fixed.

    Because each step increases F and the E-step makes F equal to the
    log-likelihood, the log-likelihood cannot decrease. That is the proof of
    the monotonicity property, and this function makes it checkable: pass the
    true responsibilities and `gap` comes back 0; pass anything else and the
    bound is strictly below the log-likelihood by the KL divergence.

    Returns the three quantities with ``log_likelihood = lower_bound + kl``.
    """
    X = np.asarray(X, dtype=float)
    q = np.asarray(q, dtype=float)

    log_joint = model._log_weighted_densities(X)      # log p(x_n, z_n=k)
    log_marginal = logsumexp(log_joint, axis=1)       # log p(x_n)

    # Clip only inside the logs. 0 * log 0 is defined as 0 in the entropy and
    # in the expectation, which is what the clip plus the q-weighting gives.
    log_q = np.log(np.clip(q, 1e-300, None))

    expected_complete = float(np.sum(q * log_joint))
    entropy = float(-np.sum(q * log_q))
    lower_bound = expected_complete + entropy
    log_likelihood = float(log_marginal.sum())
    kl = float(np.sum(q * (log_q - (log_joint - log_marginal[:, None]))))

    return {
        "expected_complete_loglik": expected_complete,
        "entropy_q": entropy,
        "lower_bound": lower_bound,
        "log_likelihood": log_likelihood,
        "kl_divergence": kl,
        "gap": log_likelihood - lower_bound,
    }


def select_n_components(
    X: np.ndarray,
    k_values: range | list[int],
    random_state: int = 42,
    **kwargs,
) -> list[dict]:
    """Fit a mixture for each K and score it by BIC and AIC.

    Unlike inertia in K-means, BIC does not fall monotonically with K: the
    penalty term grows as K adds parameters, so BIC has a genuine minimum and
    can be used to CHOOSE K rather than merely to look at.
    """
    rows = []
    for K in k_values:
        gm = GaussianMixtureEM(n_components=K, random_state=random_state, **kwargs)
        gm.fit(X)
        resp = gm.responsibilities_
        # Mean max-responsibility: 1.0 means the soft model has collapsed to a
        # hard one, ~1/K means the components overlap almost completely.
        rows.append(
            {
                "n_components": int(K),
                "log_likelihood": gm.score(X),
                "n_parameters": gm.n_parameters(),
                "BIC": gm.bic(X),
                "AIC": gm.aic(X),
                "n_iter": gm.n_iter_,
                "converged": bool(gm.converged_),
                "mean_max_responsibility": float(resp.max(axis=1).mean()),
            }
        )
    return rows
