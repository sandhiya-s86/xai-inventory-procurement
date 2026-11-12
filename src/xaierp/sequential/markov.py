"""Markov models for the demand-state process (Unit 4).

The first-order Markov assumption compresses the whole history into the
current state:

    p(s_t | s_1:t-1) = p(s_t | s_{t-1})

Applied here, daily demand for a SKU is bucketed into a small set of states
(ZERO / LOW / NORMAL / HIGH) and the transition matrix A is estimated by
counting. That immediately answers questions the point forecast cannot:
"given today was a spike, how likely is tomorrow also a spike?" -- i.e.
whether high-demand days CLUSTER, which is what determines whether a single
safety-stock number is adequate or whether runs of spikes will blow through it.

Key objects:

    A[i, j] = p(S_t = j | S_{t-1} = i), every ROW sums to 1
    A^k     = k-step transition probabilities
    pi*     = stationary distribution, pi* = pi* A, the long-run occupancy

Estimation is by maximum likelihood, which for a fully observed chain is just
row-normalised transition counts: a_ij = N_ij / sum_k N_ik. Laplace smoothing
is available because a zero count is a strong claim -- it makes any path
using that transition strictly impossible, and "never seen in 420 training
days" is not the same as "cannot happen".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_STATE_NAMES = ("ZERO", "LOW", "NORMAL", "HIGH")


def discretize_demand(
    y: np.ndarray,
    n_states: int = 4,
    cutpoints: np.ndarray | None = None,
    zero_state: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bucket a continuous demand series into ordered discrete states.

    Two regimes, chosen by whether the series actually contains zeros:

    * **Zero-inflated** (a single SKU's series). Zero gets a state of its own
      and the remaining n-1 states are quantile buckets of the POSITIVE
      values. Retail demand is 31.7% zeros in this dataset, and "sold
      nothing" is a categorically different event from "sold a little";
      merging them would hide exactly the intermittency Unit 3 segments on.
    * **No zeros** (the business-wide aggregate, which never has a zero day).
      A ZERO state would be structurally empty, leaving an unestimable row in
      the transition matrix, so all n states are quantile buckets instead.

    Returns ``(states, cutpoints, names)``. `cutpoints` are the INTERIOR
    boundaries only -- pass them back in to apply a rule learned on training
    data to a later period without re-fitting, which is what keeps the state
    definition leakage-free.
    """
    y = np.asarray(y, dtype=float)
    if zero_state is None:
        zero_state = bool(np.any(y <= 0))

    n_buckets = n_states - 1 if zero_state else n_states

    if cutpoints is None:
        pool = y[y > 0] if zero_state else y
        if len(pool) == 0 or n_buckets < 2:
            cutpoints = np.array([])
        else:
            qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
            cutpoints = np.unique(np.quantile(pool, qs))
    cutpoints = np.asarray(cutpoints, dtype=float)

    # Bucket index among the positive values: 0 .. len(cutpoints)
    bucket = np.searchsorted(cutpoints, y, side="right")

    if zero_state:
        # Shift the positive buckets up by one so state 0 is reserved for zero
        # and never collides with the lowest positive bucket.
        states = bucket + 1
        states[y <= 0] = 0
        n_actual = len(cutpoints) + 2
        names = (
            list(DEFAULT_STATE_NAMES)
            if n_actual == len(DEFAULT_STATE_NAMES)
            else ["ZERO"] + [f"Q{i}" for i in range(1, n_actual)]
        )
    else:
        states = bucket
        n_actual = len(cutpoints) + 1
        base = ["LOW", "MEDIUM", "HIGH", "PEAK"]
        names = (
            base[:n_actual]
            if n_actual <= len(base)
            else [f"Q{i + 1}" for i in range(n_actual)]
        )

    return states.astype(int), cutpoints, names


def transition_counts(states: np.ndarray, n_states: int | None = None) -> np.ndarray:
    """N[i, j] = number of times state i was followed by state j."""
    states = np.asarray(states, dtype=int)
    n = int(states.max()) + 1 if n_states is None else n_states
    N = np.zeros((n, n), dtype=float)
    np.add.at(N, (states[:-1], states[1:]), 1.0)
    return N


def transition_matrix(
    states: np.ndarray, n_states: int | None = None, alpha: float = 0.0
) -> np.ndarray:
    """Maximum-likelihood transition matrix, optionally Laplace-smoothed.

    a_ij = (N_ij + alpha) / sum_k (N_ik + alpha). With alpha = 0 this is the
    exact MLE. A state never observed as a source has no data at all; its row
    is set to uniform rather than left as zeros, because a row of zeros is not
    a probability distribution and would break A^k and the stationary solve.
    """
    N = transition_counts(states, n_states) + alpha
    rows = N.sum(axis=1, keepdims=True)
    n = N.shape[0]
    A = np.where(rows > 0, N / np.where(rows > 0, rows, 1.0), 1.0 / n)
    return A


def k_step(A: np.ndarray, k: int) -> np.ndarray:
    """A^k -- the k-step transition probabilities.

    Uses `np.linalg.matrix_power`, which is repeated squaring: O(log k)
    matrix products instead of k of them.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    return np.linalg.matrix_power(np.asarray(A, dtype=float), k)


def stationary_distribution(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Solve pi = pi A subject to sum(pi) = 1.

    pi is the LEFT eigenvector of A for eigenvalue 1. Rather than relying on
    eigen-decomposition (which can hand back a complex vector for a
    near-degenerate chain), this solves the linear system directly: take
    (A^T - I) pi = 0 and replace the last, redundant equation with the
    normalisation sum(pi) = 1, which makes the system non-singular and the
    answer real by construction.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    M = A.T - np.eye(n)
    M[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:        # pragma: no cover - reducible chain
        vals, vecs = np.linalg.eig(A.T)
        pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])

    pi = np.real(pi)
    pi[np.abs(pi) < tol] = 0.0
    pi = np.clip(pi, 0.0, None)
    total = pi.sum()
    return pi / total if total > 0 else np.full(n, 1.0 / n)


def path_logprob(states: np.ndarray, A: np.ndarray, pi: np.ndarray) -> float:
    """log p(s_1:T) = log pi_{s1} + SUM log a_{s_{t-1}, s_t}.

    In logs, not probabilities. A 600-day path multiplies 600 numbers below 1
    and underflows float64 long before the end -- the same problem the scaling
    factors in hmm.py exist to solve. A zero transition probability gives
    -inf, correctly marking the path as impossible.
    """
    states = np.asarray(states, dtype=int)
    with np.errstate(divide="ignore"):
        lp = float(np.log(pi[states[0]]))
        lp += float(np.sum(np.log(A[states[:-1], states[1:]])))
    return lp


def classify_states(A: np.ndarray, tol: float = 1e-12) -> list[dict]:
    """Label each state absorbing / recurrent / transient.

    * absorbing: a_ii = 1, the chain can never leave.
    * recurrent: reachable from everything it can reach -- once left, return
      is certain. Determined from the reachability closure of A.
    * transient: everything else; may eventually be left forever.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    # Reachability by repeated boolean multiplication (Warshall closure).
    reach = (A > tol)
    closure = reach.copy()
    for _ in range(n):
        new = closure | (closure @ reach)
        if np.array_equal(new, closure):
            break
        closure = new
    np.fill_diagonal(closure, True)

    rows = []
    for i in range(n):
        absorbing = bool(abs(A[i, i] - 1.0) <= tol)
        # i is recurrent iff everything i reaches can get back to i.
        recurrent = absorbing or all(
            closure[j, i] for j in range(n) if closure[i, j]
        )
        rows.append(
            {
                "state": i,
                "self_transition": float(A[i, i]),
                "absorbing": absorbing,
                "recurrent": bool(recurrent),
                "transient": bool(not recurrent),
                # Mean time to leave, for a geometric holding time.
                "expected_holding_time": (
                    float("inf") if absorbing else float(1.0 / (1.0 - A[i, i]))
                    if A[i, i] < 1.0 else float("inf")
                ),
            }
        )
    return rows


@dataclass
class MarkovChain:
    """A first-order Markov chain fitted by counting transitions."""

    n_states: int
    alpha: float = 0.0
    state_names: list[str] | None = None

    A: np.ndarray | None = None
    pi: np.ndarray | None = None
    counts_: np.ndarray | None = None

    def fit(self, states: np.ndarray) -> "MarkovChain":
        states = np.asarray(states, dtype=int)
        self.counts_ = transition_counts(states, self.n_states)
        self.A = transition_matrix(states, self.n_states, self.alpha)
        # Initial distribution from the empirical state frequencies: with a
        # single long sequence there is only one observed s_1, which is far
        # too little to estimate pi from.
        counts = np.bincount(states, minlength=self.n_states).astype(float)
        self.pi = counts / counts.sum()
        return self

    def predict_proba(self, state: int, k: int = 1) -> np.ndarray:
        """Distribution over the state k steps after being in `state`."""
        return k_step(self.A, k)[int(state)]

    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.A)

    def logprob(self, states: np.ndarray) -> float:
        return path_logprob(states, self.A, self.pi)

    def simulate(self, n: int, start: int | None = None,
                 seed: int = 42) -> np.ndarray:
        """Sample a path of length n from the fitted chain."""
        rng = np.random.default_rng(seed)
        out = np.empty(n, dtype=int)
        out[0] = rng.choice(self.n_states, p=self.pi) if start is None else int(start)
        for t in range(1, n):
            out[t] = rng.choice(self.n_states, p=self.A[out[t - 1]])
        return out

    def summary(self) -> list[dict]:
        """Per-state table: occupancy, persistence and long-run share."""
        stat = self.stationary()
        names = self.state_names or [f"S{i}" for i in range(self.n_states)]
        cls = classify_states(self.A)
        return [
            {
                "state": i,
                "state_name": names[i],
                "empirical_freq": float(self.pi[i]),
                "stationary_prob": float(stat[i]),
                "self_transition": float(self.A[i, i]),
                "expected_run_length": cls[i]["expected_holding_time"],
                "recurrent": cls[i]["recurrent"],
            }
            for i in range(self.n_states)
        ]
