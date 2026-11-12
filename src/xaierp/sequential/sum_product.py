"""Sum-product message passing on a factor graph (Unit 4).

Sum-product is the general algorithm; forward-backward is what it *becomes*
on a chain. Rather than asserting that, this module implements the general
message-passing rules and then shows numerically that the messages equal
alpha and beta and the beliefs equal gamma, to machine precision.

The factor graph for an HMM chain
---------------------------------
Variables z_1..z_T; factors

    f_1(z_1)           = pi_{z_1} b_{z_1}(x_1)
    f_t(z_{t-1}, z_t)  = a_{z_{t-1}, z_t} b_{z_t}(x_t)      t = 2..T

so that PROD_t f_t = p(z_1:T, x_1:T), which is the HMM factorisation.

The two message rules
---------------------
    variable -> factor   m_{z->f}(z) = PROD_{h in N(z)\\{f}} m_{h->z}(z)
    factor -> variable   m_{f->z}(z) = SUM_{other vars} f(.) PROD m_{y->f}(y)

A variable multiplies everything it has heard from elsewhere; a factor
multiplies in its own local function and sums out the variables it is not
addressing. A belief is the product of all incoming messages, normalised:

    b(z) PROPTO PROD_{f in N(z)} m_{f->z}(z)

On a TREE (and a chain is a tree) one inward and one outward sweep gives
EXACT marginals. On a loopy graph the same updates iterated give loopy belief
propagation, which is approximate and may not converge at all.

The semiring view
-----------------
Replace SUM with MAX and the identical code computes max-marginals instead of
marginals -- that is max-product, and tracing back its argmaxes is Viterbi.
`chain_max_product` below is literally `chain_sum_product` with one operator
swapped, which is the cleanest demonstration of the point that
sum-product = marginalisation and max-product = MAP.

Messages are normalised at every step for exactly the reason the scaling
section of hmm.py exists: the raw products underflow on long chains.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-300


def _normalise(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale a message to sum to 1, returning the constant removed."""
    s = float(v.sum())
    if s <= 0:
        n = len(v)
        return np.full(n, 1.0 / n), EPS
    return v / s, s


def chain_sum_product(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Exact marginals on an HMM chain by sum-product message passing.

    Returns the forward messages, backward messages, single-variable beliefs
    (= gamma), pairwise beliefs (= xi) and the log partition function
    log Z = log p(x_1:T).
    """
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    Bt = np.asarray(Bt, dtype=float)
    T, N = Bt.shape

    # ---- forward sweep: m_fwd[t] is the message arriving at z_t from the left
    m_fwd = np.zeros((T, N))
    scales = np.zeros(T)

    m_fwd[0], scales[0] = _normalise(pi * Bt[0])
    for t in range(1, T):
        # factor-to-variable: sum out z_{t-1} against f_t(z_{t-1}, z_t)
        msg = (m_fwd[t - 1] @ A) * Bt[t]
        m_fwd[t], scales[t] = _normalise(msg)

    # ---- backward sweep: message arriving at z_t from the right
    m_bwd = np.zeros((T, N))
    m_bwd[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        msg = A @ (Bt[t + 1] * m_bwd[t + 1])
        m_bwd[t] = msg / max(scales[t + 1], EPS)

    # ---- beliefs: product of incoming messages, normalised
    belief = m_fwd * m_bwd
    belief /= np.maximum(belief.sum(axis=1, keepdims=True), EPS)

    pair = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        pair[t] = (
            m_fwd[t][:, None] * A * (Bt[t + 1] * m_bwd[t + 1])[None, :]
            / max(scales[t + 1], EPS)
        )
        s = pair[t].sum()
        if s > 0:
            pair[t] /= s

    return {
        "forward_messages": m_fwd,
        "backward_messages": m_bwd,
        "beliefs": belief,
        "pairwise_beliefs": pair,
        "log_partition": float(np.sum(np.log(np.maximum(scales, EPS)))),
        "scales": scales,
    }


def chain_max_product(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Max-product on the same chain: SUM replaced by MAX.

    Returns the MAP path and its log-probability. Identical in structure to
    `chain_sum_product` -- the only changes are `max` in place of the
    summation and the backpointers needed to recover the argmax path.

    Computed in the log domain. Max-product has no scaling trick available
    (max does not distribute over a normalising constant the way sum does in
    the forward recursion), but log turns the products into sums exactly.
    """
    with np.errstate(divide="ignore"):
        log_pi = np.log(np.asarray(pi, dtype=float))
        log_A = np.log(np.asarray(A, dtype=float))
        log_B = np.log(np.asarray(Bt, dtype=float))
    T, N = log_B.shape

    msg = np.zeros((T, N))
    back = np.zeros((T, N), dtype=int)

    msg[0] = log_pi + log_B[0]
    for t in range(1, T):
        scores = msg[t - 1][:, None] + log_A      # (prev, next)
        back[t] = np.argmax(scores, axis=0)
        msg[t] = np.max(scores, axis=0) + log_B[t]

    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(msg[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return {
        "messages": msg,
        "backpointers": back,
        "map_path": path,
        "map_logprob": float(msg[T - 1, path[T - 1]]),
    }


def verify_equals_forward_backward(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> dict[str, float | bool]:
    """Check sum-product against forward-backward, and max-product vs Viterbi.

    This is the claim from the notes -- "forward-backward is sum-product on an
    HMM chain; Viterbi is a max-product analogue" -- turned into a number.
    Every difference below should be at machine-epsilon scale.
    """
    from .hmm import backward_scaled, forward_scaled, gamma_xi
    from .viterbi import viterbi_log

    sp = chain_sum_product(pi, A, Bt)
    mp = chain_max_product(pi, A, Bt)

    alpha, c, ll = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, xi = gamma_xi(alpha, beta, A, Bt, c)

    v_path, v_lp, _, _ = viterbi_log(pi, A, Bt)

    return {
        "max_abs_diff_forward_message_vs_alpha": float(
            np.max(np.abs(sp["forward_messages"] - alpha))
        ),
        "max_abs_diff_backward_message_vs_beta": float(
            np.max(np.abs(sp["backward_messages"] - beta))
        ),
        "max_abs_diff_belief_vs_gamma": float(np.max(np.abs(sp["beliefs"] - gamma))),
        "max_abs_diff_pairwise_vs_xi": float(
            np.max(np.abs(sp["pairwise_beliefs"] - xi))
        ),
        "abs_diff_logpartition_vs_loglik": float(abs(sp["log_partition"] - ll)),
        "max_product_path_matches_viterbi": bool(np.array_equal(mp["map_path"], v_path)),
        "abs_diff_map_logprob": float(abs(mp["map_logprob"] - v_lp)),
    }
