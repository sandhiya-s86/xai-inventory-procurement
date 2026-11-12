"""Unit 4 — sequential data, Markov models, HMMs and linear dynamical systems.

    sequential.py   sequential data: chain rule, autocorrelation,
                    stationarity, rolling-origin evaluation
    markov.py       Markov models: transition matrices, k-step prediction,
                    stationary distributions, state classification
    hmm.py          HMMs: the three canonical problems, scaled forward-backward,
                    Baum-Welch (EM), scaling factors
    viterbi.py      Viterbi MAP decoding vs posterior decoding
    sum_product.py  sum-product message passing on a factor graph;
                    forward-backward recovered as a special case
    lds.py          linear dynamical systems: Kalman filter and RTS smoother

The through-line of the unit is that one factorisation

    p(z, x) = p(z1) PROD p(z_t | z_{t-1}) PROD p(x_t | z_t)

supports every algorithm here, and that each algorithm is a different
operation on the same chain: sum-product gives marginals, max-product gives
the MAP path, EM learns the parameters, and replacing the discrete state with
a Gaussian one turns the HMM into a Kalman filter.

tests/test_unit4_sequential.py pins the implementations against brute-force
enumeration over all N^T state paths on small cases, so "efficient" is shown
to mean "same answer, less work" rather than "different answer, faster".
"""

from __future__ import annotations

from .hmm import (
    DiscreteHMM,
    GaussianHMM,
    backward_scaled,
    brute_force_log_likelihood,
    brute_force_posteriors,
    forward_scaled,
    gamma_xi,
)
from .lds import KalmanFilter, LocalLevelModel, kalman_filter, rts_smoother
from .markov import (
    MarkovChain,
    classify_states,
    discretize_demand,
    stationary_distribution,
    transition_matrix,
)
from .sequential import (
    autocorrelation,
    chain_rule_logprob,
    lag_matrix,
    rolling_origin_splits,
    stationarity_report,
)
from .sum_product import chain_max_product, chain_sum_product
from .viterbi import brute_force_best_path, posterior_decode, viterbi_log

__all__ = [
    "autocorrelation",
    "lag_matrix",
    "chain_rule_logprob",
    "stationarity_report",
    "rolling_origin_splits",
    "MarkovChain",
    "discretize_demand",
    "transition_matrix",
    "stationary_distribution",
    "classify_states",
    "DiscreteHMM",
    "GaussianHMM",
    "forward_scaled",
    "backward_scaled",
    "gamma_xi",
    "brute_force_log_likelihood",
    "brute_force_posteriors",
    "viterbi_log",
    "posterior_decode",
    "brute_force_best_path",
    "chain_sum_product",
    "chain_max_product",
    "kalman_filter",
    "rts_smoother",
    "KalmanFilter",
    "LocalLevelModel",
]
