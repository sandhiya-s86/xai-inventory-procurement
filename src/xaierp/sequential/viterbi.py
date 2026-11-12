"""Viterbi decoding: the single most probable hidden-state path (Unit 4).

Viterbi is the forward algorithm with ONE operator changed:

    forward   alpha_t(j) = b_j(x_t) SUM_i alpha_{t-1}(i) a_ij
    Viterbi   delta_t(j) = b_j(x_t) MAX_i delta_{t-1}(i) a_ij

Forward sums over every path into state j, giving the total likelihood;
Viterbi keeps only the best one, giving the probability of the best partial
path. Backpointers psi_t(j) record WHICH predecessor won, and the MAP path is
recovered by following them back from the best final state.

Everything runs in the log domain. The products that underflow in the forward
recursion underflow here too, but max commutes with log (unlike sum), so logs
are an exact reformulation rather than an approximation -- no scaling
constants are needed at all.

Viterbi vs posterior decoding -- a real distinction
---------------------------------------------------
Posterior decoding takes argmax_i gamma_t(i) independently at each t. It
maximises the number of individually-correct states, but the resulting
sequence can be one the model considers IMPOSSIBLE: if a_ij = 0, posterior
decoding will still happily emit ...i, j... because it never looks at
transitions. Viterbi maximises the probability of the path AS A WHOLE and so
is always a legal sequence.

For the regime labelling here that matters: a jump from "calm" straight to
"disruption" without passing through the intermediate regime is an artefact
if the transition matrix says it cannot happen. The two decoders are run
against each other in scripts/run_sequential.py and their disagreement rate
reported.
"""

from __future__ import annotations

from itertools import product

import numpy as np

NEG_INF = -np.inf


def _safe_log(x: np.ndarray) -> np.ndarray:
    """log with 0 -> -inf, without the divide-by-zero warning."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, dtype=float))


def viterbi_log(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """MAP state path in the log domain.

    Parameters mirror the forward recursion: `Bt` is the (T, N) matrix of
    emission probabilities p(x_t | z_t = j).

    Returns ``(path, logprob, delta, psi)`` where `path` is the (T,) MAP
    state sequence, `logprob` is log p(path, x_1:T), and `delta`/`psi` are the
    (T, N) score and backpointer tables.
    """
    log_pi = _safe_log(pi)
    log_A = _safe_log(A)
    log_B = _safe_log(Bt)
    T, N = log_B.shape

    delta = np.full((T, N), NEG_INF)
    psi = np.zeros((T, N), dtype=int)

    # delta_1(j) = log pi_j + log b_j(x_1)
    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        # scores[i, j] = delta_{t-1}(i) + log a_ij
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        # The emission term is added AFTER the max -- it does not depend on
        # the predecessor i, so it cannot change which i wins. Folding it in
        # before the max would be wasted work; forgetting it entirely is the
        # classic Viterbi bug the notes warn about.
        delta[t] = np.max(scores, axis=0) + log_B[t]

    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    logprob = float(delta[T - 1, path[T - 1]])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, logprob, delta, psi


def posterior_decode(gamma: np.ndarray) -> np.ndarray:
    """argmax_i gamma_t(i) at each t, independently. NOT guaranteed legal."""
    return np.argmax(np.asarray(gamma, dtype=float), axis=1)


def path_logprob(
    path: np.ndarray, pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> float:
    """log p(z_1:T = path, x_1:T) for any given path.

    Used to check whether a posterior-decoded path is even possible: a path
    using a zero-probability transition scores -inf.
    """
    path = np.asarray(path, dtype=int)
    log_pi, log_A, log_B = _safe_log(pi), _safe_log(A), _safe_log(Bt)
    lp = log_pi[path[0]] + log_B[0, path[0]]
    for t in range(1, len(path)):
        lp += log_A[path[t - 1], path[t]] + log_B[t, path[t]]
    return float(lp)


def brute_force_best_path(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray
) -> tuple[np.ndarray, float]:
    """The MAP path found by scoring all N^T sequences. Reference only."""
    T, N = np.asarray(Bt).shape
    best, best_lp = None, NEG_INF
    for cand in product(range(N), repeat=T):
        lp = path_logprob(np.array(cand), pi, A, Bt)
        if lp > best_lp:
            best, best_lp = cand, lp
    return np.array(best, dtype=int), float(best_lp)


def compare_decoders(
    pi: np.ndarray, A: np.ndarray, Bt: np.ndarray, gamma: np.ndarray
) -> dict:
    """Quantify how far Viterbi and posterior decoding actually diverge."""
    v_path, v_lp, _, _ = viterbi_log(pi, A, Bt)
    p_path = posterior_decode(gamma)
    p_lp = path_logprob(p_path, pi, A, Bt)

    disagree = int(np.sum(v_path != p_path))
    return {
        "n_timesteps": int(len(v_path)),
        "n_disagreements": disagree,
        "disagreement_rate": float(disagree / len(v_path)),
        "viterbi_logprob": v_lp,
        "posterior_path_logprob": p_lp,
        # Non-negative by construction: no path can beat the MAP path.
        "logprob_advantage_viterbi": float(v_lp - p_lp),
        "posterior_path_is_impossible": bool(not np.isfinite(p_lp)),
        "viterbi_n_switches": int(np.sum(v_path[1:] != v_path[:-1])),
        "posterior_n_switches": int(np.sum(p_path[1:] != p_path[:-1])),
    }
