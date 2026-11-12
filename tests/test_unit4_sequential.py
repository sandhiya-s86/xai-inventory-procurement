"""Unit 4 -- the sequential algorithms compute what they claim to.

The strongest tests here compare the efficient algorithms against BRUTE-FORCE
ENUMERATION over all N^T hidden-state paths on deliberately tiny problems.
That is the only way to show that dynamic programming is doing the same
arithmetic faster rather than different arithmetic quickly:

    forward recursion   vs  summing the joint over every path
    forward-backward    vs  marginalising every path per time step
    Viterbi             vs  scoring every path and taking the best

Alongside those sit the structural guarantees the lectures state -- rows of A
summing to 1, pi* = pi* A, EM monotonicity, smoothed variance never exceeding
filtered variance, and sum-product reducing exactly to forward-backward.
"""

from __future__ import annotations

import numpy as np
import pytest

from xaierp.sequential.hmm import (
    DiscreteHMM,
    GaussianHMM,
    backward_scaled,
    brute_force_log_likelihood,
    brute_force_posteriors,
    demonstrate_underflow,
    emission_discrete,
    emission_gaussian,
    forward_scaled,
    gamma_xi,
    unscaled_forward,
)
from xaierp.sequential.lds import (
    KalmanFilter,
    LocalLevelModel,
    kalman_filter,
    rts_smoother,
)
from xaierp.sequential.markov import (
    MarkovChain,
    classify_states,
    discretize_demand,
    k_step,
    path_logprob,
    stationary_distribution,
    transition_counts,
    transition_matrix,
)
from xaierp.sequential.sequential import (
    autocorrelation,
    chain_rule_logprob,
    lag_matrix,
    rolling_origin_splits,
    stationarity_report,
)
from xaierp.sequential.sum_product import (
    chain_max_product,
    chain_sum_product,
    verify_equals_forward_backward,
)
from xaierp.sequential.viterbi import (
    brute_force_best_path,
    compare_decoders,
    path_logprob as viterbi_path_logprob,
    posterior_decode,
    viterbi_log,
)

SEED = 0


@pytest.fixture(scope="module")
def tiny_hmm():
    """A 3-state, 4-symbol, 7-step HMM: 3^7 = 2187 paths, enumerable."""
    rng = np.random.default_rng(SEED)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    B = rng.dirichlet(np.ones(4), size=3)
    obs = rng.integers(0, 4, 7)
    return pi, A, B, obs, emission_discrete(obs, B)


# ========================================================== sequential data


def test_autocorrelation_is_one_at_lag_zero():
    rng = np.random.default_rng(SEED)
    rows = autocorrelation(rng.normal(size=200), max_lag=10)
    assert rows[0]["lag"] == 0
    assert rows[0]["acf"] == pytest.approx(1.0)


def test_autocorrelation_detects_a_planted_weekly_cycle():
    t = np.arange(400)
    y = np.sin(2 * np.pi * t / 7) + 0.05 * np.random.default_rng(SEED).normal(size=400)
    rows = autocorrelation(y, max_lag=21)
    # Lag 7 and 14 are in phase with the cycle; lag 3 and 4 are near antiphase.
    assert rows[7]["acf"] > 0.8
    assert rows[14]["acf"] > 0.8
    assert rows[3]["acf"] < 0.0


def test_autocorrelation_of_white_noise_is_mostly_insignificant():
    rng = np.random.default_rng(SEED)
    rows = autocorrelation(rng.normal(size=2000), max_lag=30)
    n_sig = sum(r["significant"] for r in rows)
    assert n_sig <= 4, "white noise should rarely breach the 95% band"


def test_autocorrelation_is_bounded():
    rng = np.random.default_rng(SEED)
    rows = autocorrelation(rng.normal(size=300), max_lag=20)
    assert all(abs(r["acf"]) <= 1.0 + 1e-9 for r in rows)


def test_autocorrelation_of_a_constant_series_is_defined():
    rows = autocorrelation(np.full(50, 7.0), max_lag=5)
    assert np.isfinite([r["acf"] for r in rows]).all()


def test_lag_matrix_alignment_is_correct():
    y = np.arange(10, dtype=float)
    X, Y = lag_matrix(y, 3)
    assert X.shape == (7, 3)
    assert np.array_equal(Y, y[3:])
    # Column 0 is the most recent lag, so row i starts with y[t-1].
    assert np.array_equal(X[:, 0], y[2:9])
    assert np.array_equal(X[:, 1], y[1:8])
    assert np.array_equal(X[:, 2], y[0:7])


def test_lag_matrix_rejects_too_many_lags():
    with pytest.raises(ValueError, match="must be <"):
        lag_matrix(np.arange(5, dtype=float), 5)


def test_chain_rule_shows_dependence_beats_independence():
    """An AR model must score a dependent series above an i.i.d. model."""
    rng = np.random.default_rng(SEED)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = 0.85 * y[t - 1] + rng.normal(0, 0.4)
    out = chain_rule_logprob(y, y.mean(), y.std(), n_lags=2)
    assert out["gain_per_obs"] > 0
    assert out["autoregressive"] / (out["n"] - 2) > out["independent"] / out["n"]


def test_chain_rule_shows_no_gain_on_independent_data():
    rng = np.random.default_rng(SEED)
    y = rng.normal(size=800)
    out = chain_rule_logprob(y, y.mean(), y.std(), n_lags=2)
    # Fitting lags to noise buys essentially nothing.
    assert out["gain_per_obs"] < 0.02


def test_stationarity_report_flags_a_random_walk_as_non_stationary():
    rng = np.random.default_rng(SEED)
    walk = np.cumsum(rng.normal(size=400))
    out = stationarity_report(walk)
    assert out["adf_stationary"] is False


def test_stationarity_report_flags_white_noise_as_stationary():
    rng = np.random.default_rng(SEED)
    out = stationarity_report(rng.normal(size=500))
    assert out["adf_stationary"] is True


def test_rolling_origin_splits_never_leak_the_future():
    """The whole point: no training index may come after a test index."""
    for train, test in rolling_origin_splits(200, n_splits=5, min_train=100,
                                             horizon=7):
        assert len(train) and len(test)
        assert train.max() < test.min(), "training data must precede test data"


def test_rolling_origin_training_windows_expand():
    splits = rolling_origin_splits(200, n_splits=4, min_train=100, horizon=5)
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1


def test_rolling_origin_rejects_an_impossible_request():
    with pytest.raises(ValueError, match="min_train \\+ horizon"):
        rolling_origin_splits(50, n_splits=3, min_train=48, horizon=10)


# ============================================================ Markov models


def test_transition_matrix_rows_are_distributions():
    rng = np.random.default_rng(SEED)
    states = rng.integers(0, 4, 500)
    A = transition_matrix(states, 4)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert (A >= 0).all()


def test_transition_counts_are_correct_for_a_known_path():
    states = np.array([0, 1, 1, 0, 2, 2, 2])
    N = transition_counts(states, 3)
    assert N[0, 1] == 1 and N[1, 1] == 1 and N[1, 0] == 1
    assert N[0, 2] == 1 and N[2, 2] == 2
    assert N.sum() == len(states) - 1


def test_transition_matrix_is_the_mle_from_counts():
    states = np.array([0, 1, 1, 0, 1, 1, 0, 1])
    A = transition_matrix(states, 2)
    N = transition_counts(states, 2)
    assert np.allclose(A, N / N.sum(axis=1, keepdims=True))


def test_transition_matrix_recovers_the_truth_from_a_long_chain():
    A_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    rng = np.random.default_rng(SEED)
    s = [0]
    for _ in range(60000):
        s.append(rng.choice(3, p=A_true[s[-1]]))
    assert np.max(np.abs(transition_matrix(np.array(s), 3) - A_true)) < 0.02


def test_unvisited_source_state_gets_a_uniform_row():
    # State 2 never appears, so its row has no data. A row of zeros is not a
    # distribution and would break A^k and the stationary solve.
    A = transition_matrix(np.array([0, 1, 0, 1, 0]), 3)
    assert np.allclose(A[2], 1.0 / 3.0)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_laplace_smoothing_removes_impossible_transitions():
    states = np.array([0, 1, 0, 1, 0, 1])
    hard = transition_matrix(states, 2, alpha=0.0)
    soft = transition_matrix(states, 2, alpha=1.0)
    assert hard[0, 0] == 0.0           # never observed -> declared impossible
    assert soft[0, 0] > 0.0            # smoothed -> merely unlikely


def test_k_step_matches_repeated_multiplication():
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert np.allclose(k_step(A, 1), A)
    assert np.allclose(k_step(A, 2), A @ A)
    assert np.allclose(k_step(A, 5), np.linalg.multi_dot([A] * 5))


def test_k_step_zero_is_the_identity():
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert np.allclose(k_step(A, 0), np.eye(2))


def test_k_step_rows_remain_distributions():
    A = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    for k in (1, 3, 10, 50):
        assert np.allclose(k_step(A, k).sum(axis=1), 1.0)


def test_stationary_distribution_is_a_fixed_point():
    """pi* = pi* A is the defining property."""
    A = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    pi = stationary_distribution(A)
    assert pi.sum() == pytest.approx(1.0)
    assert (pi >= 0).all()
    assert np.allclose(pi @ A, pi, atol=1e-10)


def test_powers_of_A_converge_to_the_stationary_distribution():
    A = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    pi = stationary_distribution(A)
    assert np.allclose(k_step(A, 200), np.tile(pi, (3, 1)), atol=1e-8)


def test_stationary_distribution_of_a_known_two_state_chain():
    # For [[1-a, a], [b, 1-b]] the stationary distribution is [b, a]/(a+b).
    a, b = 0.25, 0.5
    A = np.array([[1 - a, a], [b, 1 - b]])
    assert np.allclose(stationary_distribution(A),
                       np.array([b, a]) / (a + b))


def test_path_logprob_matches_the_hand_computed_product():
    A = np.array([[0.6, 0.4], [0.3, 0.7]])
    pi = np.array([0.5, 0.5])
    path = np.array([0, 1, 1, 0])
    expected = np.log(0.5) + np.log(0.4) + np.log(0.7) + np.log(0.3)
    assert path_logprob(path, A, pi) == pytest.approx(expected)


def test_path_logprob_is_negative_infinity_for_an_impossible_path():
    A = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert path_logprob(np.array([0, 1]), A, np.array([1.0, 0.0])) == -np.inf


def test_classify_states_identifies_absorbing_and_transient_states():
    A = np.array([[1.0, 0.0, 0.0], [0.3, 0.4, 0.3], [0.0, 0.2, 0.8]])
    rows = classify_states(A)
    assert rows[0]["absorbing"] is True
    assert rows[0]["recurrent"] is True
    # States 1 and 2 can leak into the absorbing state and never return.
    assert rows[1]["transient"] is True
    assert rows[2]["transient"] is True


def test_all_states_recurrent_in_an_irreducible_chain():
    A = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert all(r["recurrent"] for r in classify_states(A))


def test_expected_holding_time_matches_the_geometric_mean():
    A = np.array([[0.8, 0.2], [0.5, 0.5]])
    rows = classify_states(A)
    assert rows[0]["expected_holding_time"] == pytest.approx(1 / 0.2)


def test_discretize_gives_a_zero_state_only_when_zeros_exist():
    zeroful = np.array([0.0, 0, 5, 10, 20, 0, 30, 40])
    states, cuts, names = discretize_demand(zeroful, n_states=4)
    assert names[0] == "ZERO"
    assert (states[zeroful == 0] == 0).all()
    # A positive value must never be labelled ZERO.
    assert (states[zeroful > 0] > 0).all()

    positive = np.array([1.0, 5, 10, 20, 30, 40, 50, 60])
    states2, _, names2 = discretize_demand(positive, n_states=4)
    assert "ZERO" not in names2
    assert len(np.unique(states2)) == 4


def test_discretize_states_are_monotone_in_demand():
    y = np.array([0.0, 1, 2, 5, 9, 14, 30, 100])
    states, _, _ = discretize_demand(y, n_states=4)
    assert list(states) == sorted(states), "higher demand must mean a higher state"


def test_discretize_reuses_supplied_cutpoints():
    """Reusing train cut-points on later data is what keeps states leakage-free."""
    train = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    _, cuts, _ = discretize_demand(train, n_states=4)
    later = np.array([2.5, 100.0])
    s1, _, _ = discretize_demand(later, n_states=4, cutpoints=cuts, zero_state=False)
    s2, _, _ = discretize_demand(later, n_states=4, cutpoints=cuts, zero_state=False)
    assert np.array_equal(s1, s2)
    assert s1[1] == len(cuts), "a huge value lands in the top bucket"


def test_markov_chain_fit_and_summary_are_consistent():
    rng = np.random.default_rng(SEED)
    A_true = np.array([[0.9, 0.1], [0.2, 0.8]])
    s = [0]
    for _ in range(5000):
        s.append(rng.choice(2, p=A_true[s[-1]]))
    mc = MarkovChain(n_states=2, state_names=["A", "B"]).fit(np.array(s))
    assert np.allclose(mc.A.sum(axis=1), 1.0)
    assert mc.pi.sum() == pytest.approx(1.0)
    rows = mc.summary()
    assert len(rows) == 2
    assert sum(r["stationary_prob"] for r in rows) == pytest.approx(1.0)
    assert np.max(np.abs(mc.A - A_true)) < 0.05


# ====================================== HMM: forward-backward vs brute force


def test_scaled_forward_matches_brute_force_enumeration(tiny_hmm):
    """O(T N^2) recursion vs summing over all 2187 paths."""
    pi, A, B, obs, Bt = tiny_hmm
    _, _, ll = forward_scaled(pi, A, Bt)
    assert ll == pytest.approx(brute_force_log_likelihood(pi, A, Bt), abs=1e-10)


def test_scaled_alpha_is_the_filtered_posterior(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    alpha, _, _ = forward_scaled(pi, A, Bt)
    assert np.allclose(alpha.sum(axis=1), 1.0)
    assert (alpha >= 0).all()


def test_log_likelihood_equals_the_sum_of_log_scaling_factors(tiny_hmm):
    """The identity that makes scaling work: log p(x) = sum_t log c_t."""
    pi, A, B, obs, Bt = tiny_hmm
    _, c, ll = forward_scaled(pi, A, Bt)
    assert ll == pytest.approx(float(np.sum(np.log(c))))


def test_scaled_and_unscaled_agree_while_the_naive_one_still_works(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    _, _, scaled = forward_scaled(pi, A, Bt)
    assert unscaled_forward(pi, A, Bt) == pytest.approx(scaled, abs=1e-10)


def test_unscaled_forward_underflows_on_a_long_sequence(tiny_hmm):
    """Scaling is not decoration: the naive recursion really does die."""
    pi, A, B, obs, Bt = tiny_hmm
    long_Bt = np.tile(Bt, (120, 1))        # T = 840
    rows = demonstrate_underflow(pi, A, long_Bt, step=120)
    assert any(r["unscaled_underflowed"] for r in rows), "expected an underflow"
    # The scaled recursion stays finite throughout.
    assert all(np.isfinite(r["loglik_scaled"]) for r in rows)


def test_forward_backward_gamma_matches_brute_force(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, _ = gamma_xi(alpha, beta, A, Bt, c)
    assert np.allclose(gamma, brute_force_posteriors(pi, A, Bt), atol=1e-10)


def test_gamma_and_xi_are_normalised(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, xi = gamma_xi(alpha, beta, A, Bt, c)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    assert (gamma >= 0).all() and (xi >= 0).all()


def test_xi_marginalises_to_gamma(tiny_hmm):
    """sum_j xi_t(i,j) must equal gamma_t(i) -- they are the same posterior."""
    pi, A, B, obs, Bt = tiny_hmm
    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, xi = gamma_xi(alpha, beta, A, Bt, c)
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)


def test_backward_variable_starts_at_one(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    _, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    assert np.allclose(beta[-1], 1.0)


def test_filtering_and_smoothing_agree_at_the_final_step(tiny_hmm):
    """At t = T there is no future left, so the two must coincide."""
    pi, A, B, obs, Bt = tiny_hmm
    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, _ = gamma_xi(alpha, beta, A, Bt, c)
    assert np.allclose(gamma[-1], alpha[-1], atol=1e-10)


def test_gaussian_emissions_match_the_normal_density():
    from scipy.stats import norm

    x = np.array([0.0, 1.0, 2.5])
    mu = np.array([0.0, 2.0])
    var = np.array([1.0, 0.25])
    Bt = emission_gaussian(x, mu, var)
    for t in range(3):
        for j in range(2):
            assert Bt[t, j] == pytest.approx(
                norm.pdf(x[t], mu[j], np.sqrt(var[j]))
            )


# ====================================================== Baum-Welch (EM)


def test_baum_welch_log_likelihood_never_decreases():
    rng = np.random.default_rng(SEED)
    obs = rng.integers(0, 3, 400)
    hmm = DiscreteHMM(n_states=2, n_symbols=3, random_state=SEED,
                      max_iter=60, n_init=2).fit(obs)
    h = hmm.loglik_history_
    assert len(h) >= 2
    for a, b in zip(h, h[1:]):
        assert b >= a - 1e-7, f"Baum-Welch log-likelihood fell from {a} to {b}"


def test_baum_welch_recovers_planted_parameters():
    """The real test of a learning algorithm: can it find a known answer?"""
    rng = np.random.default_rng(1)
    pi_t = np.array([0.6, 0.4])
    A_t = np.array([[0.9, 0.1], [0.2, 0.8]])
    B_t = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    z = [rng.choice(2, p=pi_t)]
    for _ in range(1999):
        z.append(rng.choice(2, p=A_t[z[-1]]))
    z = np.array(z)
    x = np.array([rng.choice(3, p=B_t[s]) for s in z])

    hmm = DiscreteHMM(n_states=2, n_symbols=3, random_state=SEED,
                      max_iter=300).fit(x)
    # EM labels states arbitrarily, so align them before comparing.
    order = np.argsort(hmm.B[:, 0])[::-1]
    assert np.max(np.abs(hmm.A[np.ix_(order, order)] - A_t)) < 0.08
    assert np.max(np.abs(hmm.B[order] - B_t)) < 0.08


def test_baum_welch_beats_a_single_restart_on_likelihood():
    """n_init exists because a symmetric start is a saddle point."""
    rng = np.random.default_rng(2)
    pi_t = np.array([0.5, 0.5])
    A_t = np.array([[0.95, 0.05], [0.1, 0.9]])
    B_t = np.array([[0.8, 0.15, 0.05], [0.05, 0.25, 0.7]])
    z = [0]
    for _ in range(999):
        z.append(rng.choice(2, p=A_t[z[-1]]))
    x = np.array([rng.choice(3, p=B_t[s]) for s in z])

    one = DiscreteHMM(n_states=2, n_symbols=3, random_state=SEED,
                      n_init=1, max_iter=300).fit(x)
    many = DiscreteHMM(n_states=2, n_symbols=3, random_state=SEED,
                       n_init=8, max_iter=300).fit(x)
    assert many.log_likelihood(x) >= one.log_likelihood(x) - 1e-9


def test_baum_welch_keeps_A_a_valid_transition_matrix():
    rng = np.random.default_rng(SEED)
    obs = rng.integers(0, 3, 300)
    hmm = DiscreteHMM(n_states=3, n_symbols=3, random_state=SEED,
                      max_iter=40, n_init=2).fit(obs)
    assert np.allclose(hmm.A.sum(axis=1), 1.0)
    assert np.allclose(hmm.B.sum(axis=1), 1.0)
    assert hmm.pi.sum() == pytest.approx(1.0)


def test_gaussian_hmm_separates_planted_regimes():
    rng = np.random.default_rng(SEED)
    y = np.concatenate([
        rng.normal(1.0, 0.3, 200),
        rng.normal(4.0, 0.4, 200),
        rng.normal(1.0, 0.3, 200),
    ])
    hmm = GaussianHMM(n_states=2, random_state=SEED, max_iter=200).fit(y)
    means = np.sort(hmm.means_)
    assert means[0] == pytest.approx(1.0, abs=0.25)
    assert means[1] == pytest.approx(4.0, abs=0.25)


def test_gaussian_hmm_log_likelihood_never_decreases():
    rng = np.random.default_rng(SEED)
    y = np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 1, 150)])
    hmm = GaussianHMM(n_states=2, random_state=SEED, max_iter=100).fit(y)
    h = hmm.loglik_history_
    for a, b in zip(h, h[1:]):
        assert b >= a - 1e-7


def test_gaussian_hmm_variances_respect_the_floor():
    """Without a floor a state capturing one point sends the likelihood to +inf."""
    rng = np.random.default_rng(SEED)
    y = np.concatenate([np.full(50, 2.0), rng.normal(5, 1, 100)])
    hmm = GaussianHMM(n_states=3, random_state=SEED, max_iter=100).fit(y)
    assert (hmm.variances_ >= hmm.min_variance - 1e-12).all()
    assert np.isfinite(hmm.log_likelihood(y))


def test_gaussian_hmm_state_order_sorts_by_mean():
    rng = np.random.default_rng(SEED)
    y = np.concatenate([rng.normal(0, .5, 100), rng.normal(5, .5, 100)])
    hmm = GaussianHMM(n_states=2, random_state=SEED, max_iter=100).fit(y)
    order = hmm.state_order()
    assert hmm.means_[order[0]] <= hmm.means_[order[1]]


def test_hmm_filter_and_smooth_are_normalised():
    rng = np.random.default_rng(SEED)
    y = np.concatenate([rng.normal(0, 1, 80), rng.normal(4, 1, 80)])
    hmm = GaussianHMM(n_states=2, random_state=SEED, max_iter=60).fit(y)
    assert np.allclose(hmm.filter(y).sum(axis=1), 1.0)
    assert np.allclose(hmm.smooth(y).sum(axis=1), 1.0)


# ==================================================================== Viterbi


def test_viterbi_matches_brute_force_path_and_score(tiny_hmm):
    """The MAP path by dynamic programming vs scoring all 2187 paths."""
    pi, A, B, obs, Bt = tiny_hmm
    path, lp, _, _ = viterbi_log(pi, A, Bt)
    bf_path, bf_lp = brute_force_best_path(pi, A, Bt)
    assert np.array_equal(path, bf_path)
    assert lp == pytest.approx(bf_lp, abs=1e-10)


def test_viterbi_score_equals_its_own_path_probability(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    path, lp, _, _ = viterbi_log(pi, A, Bt)
    assert viterbi_path_logprob(path, pi, A, Bt) == pytest.approx(lp, abs=1e-10)


def test_no_path_beats_the_viterbi_path(tiny_hmm):
    """The defining optimality property, checked against random competitors."""
    pi, A, B, obs, Bt = tiny_hmm
    _, lp, _, _ = viterbi_log(pi, A, Bt)
    rng = np.random.default_rng(SEED)
    for _ in range(300):
        cand = rng.integers(0, 3, Bt.shape[0])
        assert viterbi_path_logprob(cand, pi, A, Bt) <= lp + 1e-10


def test_viterbi_never_scores_below_posterior_decoding(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, _ = gamma_xi(alpha, beta, A, Bt, c)
    out = compare_decoders(pi, A, Bt, gamma)
    assert out["logprob_advantage_viterbi"] >= -1e-10


def test_posterior_decoding_can_produce_an_impossible_path():
    """The concrete failure mode: posterior decoding ignores transitions.

    A forbids 0 -> 1 (a_01 = 0). Taking argmax of gamma independently at each
    step still yields [0, 1, 0, 0, 2], because t=1 most likely sits in state 0
    and t=2 most likely sits in state 1 -- and nothing in that per-step
    comparison ever consults A. The resulting sequence has probability
    exactly zero.

    Viterbi, which maximises the probability of the path as a whole, returns
    [2, 1, 0, 0, 2]: it gives up the single most likely state at t=1 in order
    to produce a sequence that is actually possible.

    This configuration was found by searching random sparse HMMs for a case
    where the two decoders genuinely diverge, rather than assumed.
    """
    pi = np.array([0.30, 0.30, 0.40])
    A = np.array([
        [0.90, 0.00, 0.10],    # 0 -> 1 is forbidden
        [0.50, 0.30, 0.20],
        [0.60, 0.40, 0.00],    # 2 -> 2 is forbidden
    ])
    B = np.array([
        [0.95, 0.03, 0.02],
        [0.13, 0.85, 0.02],
        [0.17, 0.54, 0.29],
    ])
    assert np.allclose(A.sum(axis=1), 1.0)
    Bt = emission_discrete(np.array([0, 1, 0, 0, 1]), B)

    alpha, c, _ = forward_scaled(pi, A, Bt)
    beta = backward_scaled(A, Bt, c)
    gamma, _ = gamma_xi(alpha, beta, A, Bt, c)

    p_path = posterior_decode(gamma)
    v_path, v_lp, _, _ = viterbi_log(pi, A, Bt)

    assert list(p_path) == [0, 1, 0, 0, 2]
    # It uses the banned transition, so its probability is zero.
    assert viterbi_path_logprob(p_path, pi, A, Bt) == -np.inf
    # Viterbi returns a legal path instead.
    assert list(v_path) == [2, 1, 0, 0, 2]
    assert np.isfinite(v_lp)
    assert not np.array_equal(p_path, v_path)


def test_viterbi_backpointers_reconstruct_the_path(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    path, _, delta, psi = viterbi_log(pi, A, Bt)
    T = Bt.shape[0]
    assert path[-1] == int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        assert path[t] == psi[t + 1, path[t + 1]]


def test_viterbi_handles_a_deterministic_chain():
    # A forced 0 -> 1 -> 0 -> 1 alternation: only one path is possible.
    pi = np.array([1.0, 0.0])
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    Bt = np.full((5, 2), 0.5)
    path, lp, _, _ = viterbi_log(pi, A, Bt)
    assert list(path) == [0, 1, 0, 1, 0]
    assert np.isfinite(lp)


# =============================================================== sum-product


def test_sum_product_equals_forward_backward(tiny_hmm):
    """'Forward-backward is sum-product on a chain', verified numerically."""
    pi, A, B, obs, Bt = tiny_hmm
    out = verify_equals_forward_backward(pi, A, Bt)
    assert out["max_abs_diff_forward_message_vs_alpha"] < 1e-12
    assert out["max_abs_diff_backward_message_vs_beta"] < 1e-12
    assert out["max_abs_diff_belief_vs_gamma"] < 1e-12
    assert out["max_abs_diff_pairwise_vs_xi"] < 1e-12
    assert out["abs_diff_logpartition_vs_loglik"] < 1e-10


def test_max_product_equals_viterbi(tiny_hmm):
    """'Viterbi is the max-product analogue', verified numerically."""
    pi, A, B, obs, Bt = tiny_hmm
    out = verify_equals_forward_backward(pi, A, Bt)
    assert out["max_product_path_matches_viterbi"] is True
    assert out["abs_diff_map_logprob"] < 1e-10


def test_sum_product_beliefs_match_brute_force(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    sp = chain_sum_product(pi, A, Bt)
    assert np.allclose(sp["beliefs"], brute_force_posteriors(pi, A, Bt), atol=1e-10)


def test_sum_product_log_partition_matches_brute_force(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    sp = chain_sum_product(pi, A, Bt)
    assert sp["log_partition"] == pytest.approx(
        brute_force_log_likelihood(pi, A, Bt), abs=1e-10
    )


def test_sum_product_beliefs_are_normalised(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    sp = chain_sum_product(pi, A, Bt)
    assert np.allclose(sp["beliefs"].sum(axis=1), 1.0)
    assert np.allclose(sp["pairwise_beliefs"].sum(axis=(1, 2)), 1.0)


def test_max_product_matches_brute_force_path(tiny_hmm):
    pi, A, B, obs, Bt = tiny_hmm
    mp = chain_max_product(pi, A, Bt)
    bf_path, bf_lp = brute_force_best_path(pi, A, Bt)
    assert np.array_equal(mp["map_path"], bf_path)
    assert mp["map_logprob"] == pytest.approx(bf_lp, abs=1e-10)


# ================================================ linear dynamical systems


def test_kalman_filter_matches_a_hand_computed_first_step():
    """One scalar update, worked out by hand from the textbook formulas."""
    y = np.array([[2.0]])
    A = C = np.array([[1.0]])
    Q, R = np.array([[0.5]]), np.array([[1.0]])
    mu0, P0 = np.array([0.0]), np.array([[2.0]])

    out = kalman_filter(y, A, C, Q, R, mu0, P0)
    # t = 0 uses the prior directly: S = P0 + R = 3, K = P0/S = 2/3.
    K = 2.0 / 3.0
    assert out["kalman_gains"][0, 0, 0] == pytest.approx(K)
    assert out["innovations"][0, 0] == pytest.approx(2.0)
    assert out["filtered_means"][0, 0] == pytest.approx(0.0 + K * 2.0)
    assert out["filtered_covs"][0, 0, 0] == pytest.approx((1 - K) * 2.0)


def test_kalman_recovers_a_hidden_level_better_than_the_raw_observations():
    rng = np.random.default_rng(SEED)
    q, r, T = 0.05, 1.0, 400
    level = np.cumsum(rng.normal(0, np.sqrt(q), T)) + 5.0
    obs = level + rng.normal(0, np.sqrt(r), T)

    kf = KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                      Q=np.array([[q]]), R=np.array([[r]]),
                      mu0=np.array([obs[0]]), P0=np.array([[1.0]]))
    out = kf.smooth(obs[:, None])
    rmse_raw = float(np.sqrt(np.mean((obs - level) ** 2)))
    rmse_filt = float(np.sqrt(np.mean((out["filtered_means"].ravel() - level) ** 2)))
    rmse_smooth = float(np.sqrt(np.mean((out["smoothed_means"].ravel() - level) ** 2)))

    assert rmse_filt < rmse_raw, "filtering should beat the raw observations"
    assert rmse_smooth < rmse_filt, "smoothing should beat filtering"


def test_smoothed_variance_never_exceeds_filtered_variance():
    """Smoothing conditions on strictly more data, so it cannot be less certain."""
    rng = np.random.default_rng(SEED)
    obs = np.cumsum(rng.normal(size=200))[:, None]
    kf = KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                      Q=np.array([[0.1]]), R=np.array([[1.0]]),
                      mu0=np.array([0.0]), P0=np.array([[1.0]]))
    out = kf.smooth(obs)
    assert np.all(out["smoothed_covs"][:, 0, 0] <= out["filtered_covs"][:, 0, 0] + 1e-9)


def test_kalman_covariances_stay_symmetric_and_positive():
    rng = np.random.default_rng(SEED)
    obs = rng.normal(size=100)[:, None]
    kf = KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                      Q=np.array([[0.1]]), R=np.array([[1.0]]),
                      mu0=np.array([0.0]), P0=np.array([[1.0]]))
    out = kf.smooth(obs)
    for P in list(out["filtered_covs"]) + list(out["smoothed_covs"]):
        assert np.allclose(P, P.T)
        assert P[0, 0] > 0


def test_rts_smoother_agrees_with_the_filter_at_the_final_step():
    rng = np.random.default_rng(SEED)
    obs = rng.normal(size=80)[:, None]
    A, Q = np.array([[1.0]]), np.array([[0.2]])
    f = kalman_filter(obs, A, np.array([[1.0]]), Q, np.array([[1.0]]),
                      np.array([0.0]), np.array([[1.0]]))
    s = rts_smoother(f, A, Q)
    assert np.allclose(s["smoothed_means"][-1], f["filtered_means"][-1])
    assert np.allclose(s["smoothed_covs"][-1], f["filtered_covs"][-1])


def test_kalman_gain_responds_to_the_noise_ratio():
    """Small R = trust the sensor (high gain); small Q = trust the model."""
    obs = np.zeros((50, 1))
    def gain(q, r):
        kf = KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                          Q=np.array([[q]]), R=np.array([[r]]),
                          mu0=np.array([0.0]), P0=np.array([[1.0]]))
        return float(kf.filter(obs)["kalman_gains"][-1, 0, 0])

    assert gain(1.0, 0.01) > gain(1.0, 100.0)
    assert gain(10.0, 1.0) > gain(0.001, 1.0)


def test_local_level_model_recovers_the_noise_ratio():
    rng = np.random.default_rng(SEED)
    q, r, T = 0.05, 1.0, 500
    level = np.cumsum(rng.normal(0, np.sqrt(q), T)) + 5.0
    obs = level + rng.normal(0, np.sqrt(r), T)

    m = LocalLevelModel().fit(obs)
    # The grid is coarse (log-spaced), so an order-of-magnitude check is the
    # honest assertion here, not an exact one.
    assert 0.2 < m.signal_to_noise / (q / r) < 5.0
    assert 0.0 < m.steady_state_gain < 1.0


def test_local_level_steady_state_gain_solves_the_riccati_equation():
    m = LocalLevelModel(q=0.2, r=1.0)
    K = m.steady_state_gain
    # p = K r / (1 - K) must satisfy p = p r/(p + r) + q.
    p = K * m.r / (1 - K)
    assert p == pytest.approx(p * m.r / (p + m.r) + m.q, rel=1e-9)


def test_local_level_forecast_is_flat_with_a_growing_band():
    """A random walk has no trend to extrapolate, but uncertainty compounds."""
    rng = np.random.default_rng(SEED)
    obs = np.cumsum(rng.normal(0, 0.3, 200)) + 10.0
    m = LocalLevelModel().fit(obs)
    mu, var = m.forecast(12)
    assert np.allclose(mu, mu[0]), "point forecast must be flat"
    assert np.all(np.diff(var) > 0), "variance must widen with the horizon"


def test_local_level_one_step_forecast_does_not_peek():
    """predicted_means[t] must use x_1:t-1 only -- the leakage rule."""
    rng = np.random.default_rng(SEED)
    obs = np.cumsum(rng.normal(0, 0.3, 120)) + 10.0
    m = LocalLevelModel().fit(obs)
    pred = m.one_step_forecast()
    # Refitting the filter on a truncated series must reproduce the same
    # prediction for the last retained step.
    kf = KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                      Q=np.array([[m.q]]), R=np.array([[m.r]]),
                      mu0=np.array([m.mu0]), P0=np.array([[m.p0]]))
    short = kf.filter(obs[:60, None])
    assert short["predicted_means"][59, 0] == pytest.approx(pred[59], rel=1e-9)


def test_kalman_loglikelihood_is_maximised_near_the_true_parameters():
    rng = np.random.default_rng(SEED)
    q, r, T = 0.1, 1.0, 400
    level = np.cumsum(rng.normal(0, np.sqrt(q), T))
    obs = (level + rng.normal(0, np.sqrt(r), T))[:, None]

    def ll(qq, rr):
        return KalmanFilter(A=np.array([[1.0]]), C=np.array([[1.0]]),
                            Q=np.array([[qq]]), R=np.array([[rr]]),
                            mu0=np.array([0.0]),
                            P0=np.array([[1e4]])).log_likelihood(obs)

    best = ll(q, r)
    assert best > ll(q * 50, r)
    assert best > ll(q / 50, r)
    assert best > ll(q, r * 50)
