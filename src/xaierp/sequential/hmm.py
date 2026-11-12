"""Hidden Markov Models: forward-backward, scaling factors, Baum-Welch (Unit 4).

A Markov chain over demand STATES assumes the thing you care about is the
thing you can see. An HMM says the opposite: what drives demand is a hidden
REGIME -- normal trading, a promotion, a supply disruption, the Christmas
peak -- and daily demand is a noisy emission from it. Nobody records the
regime, but it is inferable, and it is what should set safety stock.

    z_t     hidden regime, a first-order Markov chain
    x_t     observed demand, drawn from p(x_t | z_t) alone

    p(z_1:T, x_1:T) = p(z_1) PROD p(z_t|z_{t-1}) PROD p(x_t|z_t)

The three canonical problems, and where each is solved:

    Evaluation   p(x_1:T | lambda)          -> forward        (here)
    Decoding     argmax_z p(z_1:T | x_1:T)  -> Viterbi        (viterbi.py)
    Learning     estimate lambda            -> Baum-Welch/EM  (here)

Scaling factors
---------------
The forward recursion multiplies a probability at every time step. For
T = 600 that product is around 10^-400 and underflows float64 to exactly 0,
which silently turns the likelihood into -inf and every posterior into NaN.

The fix from the notes: normalise alpha at each step and keep the normaliser.

    c_t = SUM_i alpha~_t(i)        alpha_t(i) = alpha~_t(i) / c_t
    log p(x_1:T) = SUM_t log c_t

The scaled alpha now sums to 1 at every t -- it is the filtered posterior
p(z_t | x_1:t) -- and the likelihood is recovered as a SUM of logs, which
cannot underflow. `demonstrate_underflow` runs the unscaled recursion
alongside the scaled one and reports the exact T at which the naive version
dies.

Everything is parameterised by an emission matrix ``B_t`` of shape (T, N)
holding p(x_t | z_t = j). Discrete and Gaussian emissions differ only in how
that matrix is built, so forward, backward, Baum-Welch, Viterbi and
sum-product are all written once and shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

EPS = 1e-300


# ----------------------------------------------------------- emissions


def emission_discrete(obs: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(T, N) emission matrix for categorical observations."""
    return np.asarray(B, dtype=float)[:, np.asarray(obs, dtype=int)].T


def emission_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(T, N) emission matrix for univariate Gaussian emissions per state."""
    x = np.asarray(x, dtype=float)[:, None]
    mu = np.asarray(mu, dtype=float)[None, :]
    var = np.maximum(np.asarray(var, dtype=float)[None, :], 1e-12)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


# ------------------------------------------------- scaled forward-backward


def forward_scaled(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward recursion.

    Returns ``(alpha, c, loglik)`` where alpha is (T, N) with every row
    summing to 1 -- i.e. alpha[t] is the FILTERED posterior p(z_t | x_1:t) --
    c is the (T,) vector of scaling constants, and
    ``loglik = sum(log c) = log p(x_1:T)``.
    """
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    Bt = np.asarray(Bt, dtype=float)
    T, N = Bt.shape

    alpha = np.zeros((T, N))
    c = np.zeros(T)

    # t = 1:  alpha~(i) = pi_i b_i(x_1)
    a = pi * Bt[0]
    c[0] = a.sum()
    if c[0] <= 0:
        # Impossible first observation under every state. Bail out with a
        # uniform belief and -inf likelihood rather than dividing by zero.
        alpha[0] = 1.0 / N
        c[0] = EPS
    else:
        alpha[0] = a / c[0]

    # t > 1:  alpha~(j) = b_j(x_t) SUM_i alpha_{t-1}(i) a_ij
    for t in range(1, T):
        a = Bt[t] * (alpha[t - 1] @ A)
        c[t] = a.sum()
        if c[t] <= 0:
            alpha[t] = 1.0 / N
            c[t] = EPS
        else:
            alpha[t] = a / c[t]

    return alpha, c, float(np.sum(np.log(c)))


def backward_scaled(A: np.ndarray, Bt: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward recursion, using the SAME constants as the forward pass.

    beta_T(i) = 1;  beta_t(i) = [SUM_j a_ij b_j(x_{t+1}) beta_{t+1}(j)] / c_{t+1}

    Reusing c (rather than re-normalising independently) is what makes the
    product alpha_t(i) * beta_t(i) proportional to the smoothed posterior.
    Mixing conventions here is the classic HMM implementation bug -- the
    notes' own warning -- so the constants are passed in explicitly.
    """
    A = np.asarray(A, dtype=float)
    Bt = np.asarray(Bt, dtype=float)
    T, N = Bt.shape

    beta = np.zeros((T, N))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (Bt[t + 1] * beta[t + 1])) / max(c[t + 1], EPS)
    return beta


def gamma_xi(
    alpha: np.ndarray, beta: np.ndarray, A: np.ndarray, Bt: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed state posteriors and pairwise posteriors.

        gamma_t(i)   = p(z_t = i | x_1:T)        shape (T, N)
        xi_t(i, j)   = p(z_t = i, z_{t+1} = j | x_1:T)   shape (T-1, N, N)

    gamma is the E-step's expected state occupancy and xi the expected
    transition usage; Baum-Welch does nothing more than normalise them.

    Note gamma uses the WHOLE sequence, past and future -- it is SMOOTHING.
    alpha alone is FILTERING, which uses only x_1:t. The two differ, and the
    difference is reported in scripts/run_sequential.py.
    """
    T, N = alpha.shape

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), EPS)

    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        xi[t] = (
            alpha[t][:, None]
            * A
            * (Bt[t + 1] * beta[t + 1])[None, :]
            / max(c[t + 1], EPS)
        )
        s = xi[t].sum()
        if s > 0:
            xi[t] /= s
    return gamma, xi


# ------------------------------------------------------- brute force refs


def brute_force_log_likelihood(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> float:
    """log p(x_1:T) by explicitly summing over all N^T state paths.

    Exponential and useless in production -- which is the point. The tests
    check the O(T N^2) forward recursion against this on small cases, so the
    efficient algorithm is shown to compute the SAME number, not merely a
    plausible one.
    """
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    Bt = np.asarray(Bt, dtype=float)
    T, N = Bt.shape

    total = 0.0
    for path in product(range(N), repeat=T):
        p = pi[path[0]] * Bt[0, path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * Bt[t, path[t]]
        total += p
    return float(np.log(total)) if total > 0 else float("-inf")


def brute_force_posteriors(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> np.ndarray:
    """gamma by enumeration: p(z_t = i | x_1:T) summed over all paths."""
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    Bt = np.asarray(Bt, dtype=float)
    T, N = Bt.shape

    gamma = np.zeros((T, N))
    total = 0.0
    for path in product(range(N), repeat=T):
        p = pi[path[0]] * Bt[0, path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * Bt[t, path[t]]
        total += p
        for t in range(T):
            gamma[t, path[t]] += p
    return gamma / total if total > 0 else gamma


def unscaled_forward(pi: np.ndarray, A: np.ndarray, Bt: np.ndarray) -> float:
    """The naive recursion with no scaling -- the one that underflows."""
    a = np.asarray(pi, dtype=float) * Bt[0]
    for t in range(1, Bt.shape[0]):
        a = Bt[t] * (a @ np.asarray(A, dtype=float))
    total = a.sum()
    return float(np.log(total)) if total > 0 else float("-inf")


def demonstrate_underflow(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray, step: int = 10
) -> list[dict]:
    """Run both recursions on growing prefixes and record where naive dies.

    Produces the evidence for the scaling-factors section of the report: the
    scaled log-likelihood stays finite and decreases smoothly, while the
    unscaled one hits exactly 0 probability and drops to -inf at a specific,
    reportable T.
    """
    T = Bt.shape[0]
    rows = []
    for t in range(step, T + 1, step):
        scaled = forward_scaled(pi, A, Bt[:t])[2]
        naive = unscaled_forward(pi, A, Bt[:t])
        rows.append(
            {
                "T": t,
                "loglik_scaled": scaled,
                "loglik_unscaled": naive,
                "unscaled_underflowed": bool(not np.isfinite(naive)),
            }
        )
    return rows


# ------------------------------------------------------------- the models


@dataclass
class _BaseHMM:
    n_states: int
    max_iter: int = 100
    tol: float = 1e-8
    # EM near a symmetric initialisation moves very slowly for a few steps.
    # Without a floor on the iteration count a relative tolerance mistakes
    # that plateau for convergence and returns the saddle point itself.
    min_iter: int = 10
    n_init: int = 5
    random_state: int = 42

    pi: np.ndarray | None = None
    A: np.ndarray | None = None
    loglik_history_: list[float] = field(default_factory=list)
    n_iter_: int = 0
    converged_: bool = False

    def _emissions(self, X: np.ndarray) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError

    def _update_emissions(self, X, gamma) -> None:            # pragma: no cover
        raise NotImplementedError

    def _get_params(self) -> tuple:                           # pragma: no cover
        raise NotImplementedError

    def _set_params(self, params: tuple) -> None:             # pragma: no cover
        raise NotImplementedError

    def _reset_params(self) -> None:                          # pragma: no cover
        raise NotImplementedError

    def log_likelihood(self, X: np.ndarray) -> float:
        """Evaluation: log p(x_1:T | lambda), via the scaled forward pass."""
        return forward_scaled(self.pi, self.A, self._emissions(X))[2]

    def filter(self, X: np.ndarray) -> np.ndarray:
        """p(z_t | x_1:t) -- past and present only."""
        return forward_scaled(self.pi, self.A, self._emissions(X))[0]

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """p(z_t | x_1:T) -- uses the whole sequence."""
        Bt = self._emissions(X)
        alpha, c, _ = forward_scaled(self.pi, self.A, Bt)
        beta = backward_scaled(self.A, Bt, c)
        return gamma_xi(alpha, beta, self.A, Bt, c)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """MAP state path by Viterbi decoding."""
        from .viterbi import viterbi_log

        return viterbi_log(self.pi, self.A, self._emissions(X))[0]

    def _single_fit(self, X: np.ndarray, seed: int) -> tuple[float, list[float], int, bool]:
        """One Baum-Welch run from one random start."""
        self._reset_params()
        self._init_params(X, seed)
        history: list[float] = []
        prev = -np.inf
        converged = False
        it = 0

        for it in range(1, self.max_iter + 1):
            Bt = self._emissions(X)

            # ---- E-step
            alpha, c, ll = forward_scaled(self.pi, self.A, Bt)
            beta = backward_scaled(self.A, Bt, c)
            gamma, xi = gamma_xi(alpha, beta, self.A, Bt, c)
            history.append(ll)

            # ---- M-step
            self.pi = gamma[0] / gamma[0].sum()
            denom = np.maximum(gamma[:-1].sum(axis=0), EPS)
            self.A = xi.sum(axis=0) / denom[:, None]
            self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), EPS)
            self._update_emissions(X, gamma)

            if (
                it >= self.min_iter
                and np.isfinite(prev)
                and abs(ll - prev) <= self.tol * abs(prev)
            ):
                converged = True
                break
            prev = ll

        history.append(self.log_likelihood(X))
        return history[-1], history, it, converged

    def fit(self, X: np.ndarray) -> "_BaseHMM":
        """Baum-Welch: EM for HMMs.

        E-step  forward-backward gives gamma (expected occupancy) and
                xi (expected transitions).
        M-step  normalise those expected counts:
                    pi_i  = gamma_1(i)
                    a_ij  = SUM_t xi_t(i,j) / SUM_t gamma_t(i)
                plus the emission update, which is the only part that differs
                between the discrete and Gaussian subclasses.

        Expected (fractional) counts, not hard counts -- that is what makes it
        EM rather than counting a Viterbi path.

        Like every EM algorithm this is monotonic in the likelihood but finds
        only a LOCAL optimum, and for HMMs the local optima are genuinely bad:
        a symmetric starting point is a saddle of the objective from which the
        states never differentiate, and EM crawls along it slowly enough to
        look converged. So `n_init` independent random restarts are run and
        the highest-likelihood one kept -- the same defence `n_init` provides
        in K-means, for the same reason.
        """
        X = np.asarray(X)

        best = None
        for r in range(self.n_init):
            score, history, n_iter, converged = self._single_fit(
                X, self.random_state + 1000 * r
            )
            params = self._get_params()
            if best is None or score > best[0]:
                best = (score, history, n_iter, converged, params)

        _, self.loglik_history_, self.n_iter_, self.converged_, params = best
        self._set_params(params)
        return self


@dataclass
class DiscreteHMM(_BaseHMM):
    """HMM with categorical emissions over `n_symbols` observation symbols."""

    n_symbols: int = 2
    B: np.ndarray | None = None

    def _init_params(self, X: np.ndarray, seed: int | None = None) -> None:
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        N, M = self.n_states, self.n_symbols
        if self.pi is None:
            self.pi = np.full(N, 1.0 / N)
        if self.A is None:
            # Dirichlet(1) draws, i.e. uniform over the simplex. A near-uniform
            # A is a SADDLE POINT of the EM objective -- the states are
            # exchangeable there and nothing breaks the symmetry -- so the
            # starting rows must be genuinely asymmetric, not uniform+epsilon.
            # The diagonal is then boosted because real regimes persist.
            A = rng.dirichlet(np.ones(N), size=N)
            A += np.eye(N) * rng.uniform(0.5, 2.0)
            self.A = A / A.sum(axis=1, keepdims=True)
        if self.B is None:
            # alpha < 1 concentrates mass on a few symbols per state, which
            # separates the emission distributions from the first iteration.
            self.B = rng.dirichlet(np.full(M, 0.5), size=N)

    def _get_params(self) -> tuple:
        return (self.pi.copy(), self.A.copy(), self.B.copy())

    def _set_params(self, params: tuple) -> None:
        self.pi, self.A, self.B = params

    def _reset_params(self) -> None:
        self.pi = self.A = self.B = None

    def _emissions(self, X: np.ndarray) -> np.ndarray:
        return emission_discrete(X, self.B)

    def _update_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        X = np.asarray(X, dtype=int)
        B = np.zeros((self.n_states, self.n_symbols))
        for k in range(self.n_symbols):
            B[:, k] = gamma[X == k].sum(axis=0)
        self.B = B / np.maximum(B.sum(axis=1, keepdims=True), EPS)


@dataclass
class GaussianHMM(_BaseHMM):
    """HMM with one univariate Gaussian emission per hidden state.

    This is the model actually used on demand: x_t is log1p(units sold) and
    each hidden regime has its own mean level and variance. The log transform
    matters -- raw retail demand is strongly right-skewed and a Gaussian fits
    it badly, while log1p is roughly symmetric and keeps zeros finite.
    """

    means_: np.ndarray | None = None
    variances_: np.ndarray | None = None
    min_variance: float = 1e-4

    def _init_params(self, X: np.ndarray, seed: int | None = None) -> None:
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        N = self.n_states
        if self.pi is None:
            self.pi = np.full(N, 1.0 / N)
        if self.A is None:
            # Persistent initialisation: regimes are expected to LAST, so the
            # diagonal is weighted. Starting uniform tends to converge to a
            # solution that flips state every few days, which is not a regime.
            A = np.full((N, N), 0.1 / max(N - 1, 1))
            np.fill_diagonal(A, rng.uniform(0.80, 0.97, size=N))
            A = A + rng.uniform(0, 0.05, (N, N))
            self.A = A / A.sum(axis=1, keepdims=True)
        if self.means_ is None:
            # Spread the initial means across the observed quantiles so the
            # states start out separated rather than on top of each other,
            # jittered so that each restart explores a different basin.
            qs = np.sort(rng.uniform(0.05, 0.95, size=N))
            self.means_ = np.quantile(X, qs)
        if self.variances_ is None:
            self.variances_ = np.full(N, max(float(X.var()), self.min_variance))

    def _get_params(self) -> tuple:
        return (self.pi.copy(), self.A.copy(), self.means_.copy(),
                self.variances_.copy())

    def _set_params(self, params: tuple) -> None:
        self.pi, self.A, self.means_, self.variances_ = params

    def _reset_params(self) -> None:
        self.pi = self.A = self.means_ = self.variances_ = None

    def _emissions(self, X: np.ndarray) -> np.ndarray:
        return emission_gaussian(X, self.means_, self.variances_)

    def _update_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        w = np.maximum(gamma.sum(axis=0), EPS)
        self.means_ = (gamma * X[:, None]).sum(axis=0) / w
        diff = X[:, None] - self.means_[None, :]
        var = (gamma * diff**2).sum(axis=0) / w
        # The floor stops a state that has captured a single point from
        # collapsing to zero variance and sending the likelihood to +inf.
        self.variances_ = np.maximum(var, self.min_variance)

    def state_order(self) -> np.ndarray:
        """Indices that sort the states by mean, low to high.

        EM returns states in arbitrary order, so "state 0" means nothing
        across runs. Sorting by mean makes "state 0 = quietest regime" stable
        and the dashboard labels reproducible.
        """
        return np.argsort(self.means_)

    def summary(self, labels: list[str] | None = None) -> list[dict]:
        order = self.state_order()
        stat = _stationary(self.A)
        rows = []
        for rank, i in enumerate(order):
            rows.append(
                {
                    "state": int(i),
                    "rank_by_mean": rank,
                    "label": labels[rank] if labels and rank < len(labels) else f"R{rank}",
                    "mean_log1p_demand": float(self.means_[i]),
                    "mean_demand_units": float(np.expm1(self.means_[i])),
                    "std_log1p": float(np.sqrt(self.variances_[i])),
                    "self_transition": float(self.A[i, i]),
                    "expected_duration_days": float(1.0 / max(1.0 - self.A[i, i], 1e-9)),
                    "stationary_prob": float(stat[i]),
                }
            )
        return rows


def _stationary(A: np.ndarray) -> np.ndarray:
    from .markov import stationary_distribution

    return stationary_distribution(A)
