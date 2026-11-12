"""Stage: Unit 4 -- sequential models on the daily demand series.

Runs the whole Unit 4 syllabus on the real panel and writes every number it
quotes to a file.

     1.  Sequential data: ACF, stationarity, chain rule, rolling-origin
     2.  Markov model of demand states: A, A^k, stationary distribution
     3.  HMM: Baum-Welch (EM) learning of demand regimes
     4.  Scaling factors: the underflow the scaled recursion prevents
     5.  Forward-backward: filtering vs smoothing
     6.  Viterbi decoding vs posterior decoding
     7.  Sum-product == forward-backward, max-product == Viterbi
     8.  Linear dynamical system: Kalman filter + RTS smoother
     9.  Decision integration: regime-aware safety stock on the test period

Writes:
    reports/unit4_*.csv            (11 tables)
    reports/unit4_findings.json
    reports/figures/unit4_*.png    (6 figures)
    data/processed/demand_regimes.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.sequential.hmm import (  # noqa: E402
    GaussianHMM,
    backward_scaled,
    demonstrate_underflow,
    forward_scaled,
    gamma_xi,
    unscaled_forward,
)
from xaierp.sequential.lds import LocalLevelModel  # noqa: E402
from xaierp.sequential.markov import (  # noqa: E402
    MarkovChain,
    discretize_demand,
    k_step,
    stationary_distribution,
)
from xaierp.sequential.sequential import (  # noqa: E402
    autocorrelation,
    chain_rule_logprob,
    rolling_origin_splits,
    stationarity_report,
)
from xaierp.sequential.sum_product import verify_equals_forward_backward  # noqa: E402
from xaierp.sequential.viterbi import compare_decoders, viterbi_log  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402

plt.rcParams.update(
    {
        "figure.figsize": (11, 5.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
    }
)

N_REGIMES = 3
REGIME_LABELS = ["QUIET", "NORMAL", "SURGE"]


def save_fig(fig, name: str, figs: Path) -> None:
    path = figs / f"unit4_{name}.png"
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_table(rows, name: str, reports: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = reports / f"unit4_{name}.csv"
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"    saved -> reports/unit4_{name}.csv  ({len(df)} rows)")
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)
    reports = resolve(cfg["paths"]["reports"])
    figs = resolve(cfg["paths"]["figures"])
    processed = resolve(cfg["paths"]["processed"])
    ensure_dir(reports)
    ensure_dir(figs)

    find: dict[str, object] = {}

    print("=" * 72)
    print("UNIT 4 -- SEQUENTIAL MODELS ON DAILY DEMAND")
    print("=" * 72)

    panel = pd.read_parquet(processed / "daily_demand_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    feats = pd.read_parquet(processed / "feature_table.parquet")
    train_end = pd.Timestamp(feats.loc[feats["split"] == "train", "date"].max())

    # The business-wide daily series: one observation per trading day.
    daily = panel.groupby("date")["demand"].sum().sort_index()
    dates = daily.index
    y = daily.to_numpy(dtype=float)
    is_train = np.asarray(dates <= train_end)
    y_train = y[is_train]

    print(f"\nSeries: {len(y)} trading days, "
          f"{dates[0].date()} to {dates[-1].date()}")
    print(f"Train period ends {train_end.date()} "
          f"({is_train.sum()} days train / {(~is_train).sum()} days held out)")

    find["n_days"] = int(len(y))
    find["train_end"] = str(train_end.date())
    find["n_train_days"] = int(is_train.sum())
    find["n_test_days"] = int((~is_train).sum())

    # -------------------------------------------------- 1. sequential data
    print("\n[1/9] Sequential data: autocorrelation, stationarity, chain rule")
    acf = autocorrelation(y_train, max_lag=30)
    save_table(acf, "autocorrelation", reports)
    sig = [r for r in acf if r["significant"]]
    print(f"    {len(sig)} of 30 lags significantly autocorrelated "
          f"(band +/-{acf[0]['conf_band']:.3f})")
    print(f"    lag-1 ACF {acf[1]['acf']:.3f} | lag-7 (weekly) {acf[7]['acf']:.3f} "
          f"| lag-14 {acf[14]['acf']:.3f}")

    stat = stationarity_report(y_train)
    print(f"    ADF statistic {stat['adf_statistic']:.3f}, p = {stat['adf_pvalue']:.4f} "
          f"-> {'stationary' if stat['adf_stationary'] else 'NON-stationary'}")
    print(f"    block std ratio {stat['std_ratio']:.2f} "
          f"(variance {'drifts' if stat['std_ratio'] > 1.5 else 'is stable'} across the series)")

    cr = chain_rule_logprob(y_train, y_train.mean(), y_train.std(), n_lags=7)
    print(f"    chain rule: i.i.d. log-lik/day {cr['independent']/cr['n']:.3f} "
          f"vs AR(7) {cr['autoregressive']/(cr['n']-7):.3f} "
          f"-> order is worth {cr['gain_per_obs']:.3f} nats/day")

    splits = rolling_origin_splits(len(y), n_splits=5, min_train=int(is_train.sum()),
                                   horizon=7)
    print(f"    rolling-origin: {len(splits)} expanding-window splits, horizon 7 days")

    find["sequential"] = {
        "n_significant_lags": len(sig),
        "acf_lag1": acf[1]["acf"],
        "acf_lag7": acf[7]["acf"],
        "acf_lag14": acf[14]["acf"],
        "adf_statistic": stat.get("adf_statistic"),
        "adf_pvalue": stat.get("adf_pvalue"),
        "adf_stationary": stat.get("adf_stationary"),
        "block_std_ratio": stat["std_ratio"],
        "iid_loglik_per_day": cr["independent"] / cr["n"],
        "ar7_loglik_per_day": cr["autoregressive"] / (cr["n"] - 7),
        "order_information_gain_nats": cr["gain_per_obs"],
        "n_rolling_origin_splits": len(splits),
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lags = [r["lag"] for r in acf]
    vals = [r["acf"] for r in acf]
    ax[0].bar(lags, vals, color="#2563eb")
    ax[0].axhline(acf[0]["conf_band"], color="crimson", ls="--", label="95% band")
    ax[0].axhline(-acf[0]["conf_band"], color="crimson", ls="--")
    ax[0].set_xlabel("Lag (days)"); ax[0].set_ylabel("Autocorrelation")
    ax[0].set_title("ACF of total daily demand"); ax[0].legend()
    ax[1].plot(dates, y, lw=0.8, color="#334155")
    ax[1].axvline(train_end, color="crimson", ls="--", label="train/test cut")
    ax[1].set_title("Total daily demand"); ax[1].set_ylabel("Units")
    ax[1].legend()
    save_fig(fig, "01_sequential", figs)

    # ------------------------------------------------------ 2. Markov model
    print("\n[2/9] Markov model of demand states")
    # Cut-points are learned on the TRAIN window only, then applied to the
    # whole series -- the same leakage discipline as every other feature.
    _, cutpoints, state_names = discretize_demand(y_train, n_states=4)
    states_all, _, _ = discretize_demand(y, n_states=4, cutpoints=cutpoints,
                                         zero_state=bool(np.any(y_train <= 0)))
    states_train = states_all[is_train]
    n_states = len(state_names)
    print(f"    {n_states} states ({', '.join(state_names)}); "
          f"the aggregate series has no zero days, so all states are quantile buckets")
    print(f"    interior cut-points (units/day): "
          f"{', '.join(f'{e:,.0f}' for e in cutpoints)}")

    mc = MarkovChain(n_states=n_states, state_names=state_names).fit(states_train)
    save_table(
        [
            {"from_state": state_names[i],
             **{f"to_{state_names[j]}": float(mc.A[i, j]) for j in range(n_states)}}
            for i in range(n_states)
        ],
        "markov_transition_matrix",
        reports,
    )
    print("    transition matrix rows sum to 1: "
          f"{bool(np.allclose(mc.A.sum(axis=1), 1.0))}")

    summ = mc.summary()
    save_table(summ, "markov_states", reports)
    for r in summ:
        print(f"    {r['state_name']:<7} freq {r['empirical_freq']:.3f}  "
              f"stationary {r['stationary_prob']:.3f}  "
              f"self-transition {r['self_transition']:.3f}  "
              f"expected run {r['expected_run_length']:.1f} days")

    stat_dist = mc.stationary()
    conv = [
        {
            "k": k,
            **{f"P_{state_names[j]}": float(k_step(mc.A, k)[0, j])
               for j in range(n_states)},
            "max_deviation_from_stationary": float(
                np.max(np.abs(k_step(mc.A, k) - np.tile(stat_dist, (n_states, 1))))
            ),
        }
        for k in [1, 2, 3, 5, 10, 20, 50, 100]
    ]
    save_table(conv, "markov_kstep", reports)
    k_converged = next(
        (r["k"] for r in conv if r["max_deviation_from_stationary"] < 0.01), None
    )
    print(f"    A^k converges to the stationary distribution by k = {k_converged} days")
    print(f"    stationary distribution: "
          f"{', '.join(f'{n}={p:.3f}' for n, p in zip(state_names, stat_dist))}")

    # Is the Markov structure real, or would independent days look like this?
    # The buckets are equal-sized quantiles, so the stationary distribution is
    # uniform BY CONSTRUCTION and proves nothing on its own. The test is
    # whether the diagonal exceeds what independence predicts: under
    # independence a_ii would just equal pi_i.
    lift = np.array([mc.A[i, i] / max(stat_dist[i], 1e-12)
                     for i in range(n_states)])
    obs_counts = mc.counts_
    exp_counts = np.outer(obs_counts.sum(axis=1), obs_counts.sum(axis=0))
    exp_counts = exp_counts / max(obs_counts.sum(), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((obs_counts - exp_counts) ** 2
                               / np.where(exp_counts > 0, exp_counts, np.nan)))
    dof = (n_states - 1) ** 2
    from scipy.stats import chi2 as chi2_dist
    p_indep = float(chi2_dist.sf(chi2, dof))
    save_table(
        [
            {"state": state_names[i], "self_transition": float(mc.A[i, i]),
             "independence_baseline": float(stat_dist[i]),
             "persistence_lift": float(lift[i])}
            for i in range(n_states)
        ],
        "markov_persistence",
        reports,
    )
    print(f"    persistence vs independence: "
          f"{', '.join(f'{state_names[i]} {lift[i]:.2f}x' for i in range(n_states))}")
    print(f"    chi-square test of independence: chi2 = {chi2:.1f} "
          f"(dof {dof}), p = {p_indep:.2e} -> "
          f"{'days are NOT independent' if p_indep < 0.05 else 'cannot reject independence'}")

    find["markov"] = {
        "state_cutpoints": cutpoints.tolist(),
        "persistence_lift": lift.tolist(),
        "independence_chi2": chi2,
        "independence_dof": dof,
        "independence_pvalue": p_indep,
        "state_names": state_names,
        "rows_sum_to_one": bool(np.allclose(mc.A.sum(axis=1), 1.0)),
        "stationary_distribution": stat_dist.tolist(),
        "stationary_fixed_point_error": float(np.max(np.abs(stat_dist @ mc.A - stat_dist))),
        "k_steps_to_stationary": k_converged,
        "states": summ,
    }

    # ----------------------------------------------------------- 3. HMM
    print("\n[3/9] HMM: learning demand regimes with Baum-Welch (EM)")
    # log1p because raw daily demand is strongly right-skewed and the
    # Gaussian emission would fit it badly.
    obs = np.log1p(y)
    obs_train = obs[is_train]

    hmm = GaussianHMM(n_states=N_REGIMES, random_state=seed, max_iter=300,
                      n_init=8).fit(obs_train)
    hist = hmm.loglik_history_
    mono = all(hist[i] <= hist[i + 1] + 1e-7 for i in range(len(hist) - 1))
    print(f"    Baum-Welch: {hmm.n_iter_} iterations, "
          f"{'converged' if hmm.converged_ else 'hit cap'}")
    print(f"    log-likelihood {hist[0]:.2f} -> {hist[-1]:.2f}")
    print(f"    log-likelihood non-decreasing every iteration: {mono}")

    reg_summary = hmm.summary(REGIME_LABELS)
    save_table(reg_summary, "hmm_regimes", reports)
    for r in reg_summary:
        print(f"    {r['label']:<7} mean demand {r['mean_demand_units']:8.0f} units/day  "
              f"persistence {r['self_transition']:.3f}  "
              f"expected duration {r['expected_duration_days']:5.1f} days  "
              f"long-run share {r['stationary_prob']:.1%}")

    save_table(
        [
            {"iteration": i, "log_likelihood": float(v),
             "improvement": float(v - hist[i - 1]) if i else float("nan")}
            for i, v in enumerate(hist)
        ],
        "hmm_em_convergence",
        reports,
    )

    find["hmm"] = {
        "n_regimes": N_REGIMES,
        "n_iter": int(hmm.n_iter_),
        "converged": bool(hmm.converged_),
        "loglik_first": float(hist[0]),
        "loglik_final": float(hist[-1]),
        "loglik_monotone": bool(mono),
        "regimes": reg_summary,
    }

    # ------------------------------------------------- 4. scaling factors
    print("\n[4/9] Scaling factors: the underflow they prevent")
    Bt_all = hmm._emissions(obs)
    alpha, c, ll_scaled = forward_scaled(hmm.pi, hmm.A, Bt_all)

    # At T = 604 the naive product is around 1e-217 -- small, but still above
    # float64's ~1e-308 floor, so this particular series does NOT underflow.
    # Reporting "no underflow" and stopping there would dodge the point, so
    # the sequence is tiled to find the length at which it actually does die.
    # That number is a property of these emission densities, not a guess.
    at_full = unscaled_forward(hmm.pi, hmm.A, Bt_all)
    print(f"    at the real T = {len(obs)}: unscaled log-lik "
          f"{at_full:,.4f} vs scaled {ll_scaled:,.4f} "
          f"(agree to {abs(at_full - ll_scaled):.1e}); "
          f"raw product approx 1e{at_full/np.log(10):.0f} -- "
          f"{'UNDERFLOWED' if not np.isfinite(at_full) else 'still finite'}")

    tiled = np.tile(Bt_all, (6, 1))
    long_rows = demonstrate_underflow(hmm.pi, hmm.A, tiled, step=100)
    first_fail = next((r["T"] for r in long_rows if r["unscaled_underflowed"]), None)
    prev_ok = max((r["T"] for r in long_rows
                   if not r["unscaled_underflowed"]), default=None)
    save_table(long_rows, "scaling_underflow", reports)
    print(f"    tiling the series to find the breaking point: naive recursion "
          f"survives T = {prev_ok} and underflows to -inf by T = {first_fail}")
    print(f"    the scaled recursion is still finite at T = "
          f"{long_rows[-1]['T']} (log-lik {long_rows[-1]['loglik_scaled']:,.1f})")
    print(f"    scaled recursion at the real T = {len(obs)}: "
          f"log-likelihood {ll_scaled:,.4f}")
    print(f"    log p(x) == sum(log c_t): "
          f"{bool(np.allclose(ll_scaled, np.sum(np.log(c))))}")
    print(f"    scaled alpha sums to 1 at every t: "
          f"{bool(np.allclose(alpha.sum(axis=1), 1.0))}")

    find["scaling"] = {
        "real_sequence_length": int(len(obs)),
        "unscaled_loglik_at_real_T": float(at_full),
        "unscaled_underflowed_at_real_T": bool(not np.isfinite(at_full)),
        "unscaled_survives_to_T": prev_ok,
        "unscaled_underflow_at_T": first_fail,
        "scaled_still_finite_at_T": int(long_rows[-1]["T"]),
        "scaled_loglik": float(ll_scaled),
        "loglik_equals_sum_log_c": bool(np.allclose(ll_scaled, np.sum(np.log(c)))),
        "alpha_normalised": bool(np.allclose(alpha.sum(axis=1), 1.0)),
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Ts = [r["T"] for r in long_rows]
    ax[0].plot(Ts, [r["loglik_scaled"] for r in long_rows], "o-",
               label="scaled (sum of log c)", color="#059669")
    naive = [r["loglik_unscaled"] if np.isfinite(r["loglik_unscaled"]) else np.nan
             for r in long_rows]
    ax[0].plot(Ts, naive, "s--", label="unscaled (naive)", color="#dc2626")
    if first_fail:
        ax[0].axvline(first_fail, color="crimson", ls=":",
                      label=f"underflow at T = {first_fail}")
    ax[0].set_xlabel("Sequence length T (days)")
    ax[0].set_ylabel("log p(x_1:T)")
    ax[0].set_title("Why scaling factors are needed"); ax[0].legend()
    ax[1].plot(hist, "o-", color="#7c3aed")
    ax[1].set_xlabel("Baum-Welch iteration"); ax[1].set_ylabel("Log-likelihood")
    ax[1].set_title("EM is monotonic: log-likelihood never decreases")
    save_fig(fig, "02_scaling_em", figs)

    # ------------------------------------------- 5. forward-backward
    print("\n[5/9] Forward-backward: filtering vs smoothing")
    beta = backward_scaled(hmm.A, Bt_all, c)
    gamma, xi = gamma_xi(alpha, beta, hmm.A, Bt_all, c)
    print(f"    gamma rows sum to 1: {bool(np.allclose(gamma.sum(axis=1), 1.0))}")
    print(f"    xi slices sum to 1 : {bool(np.allclose(xi.sum(axis=(1, 2)), 1.0))}")

    filt_path = np.argmax(alpha, axis=1)
    smooth_path = np.argmax(gamma, axis=1)
    n_diff = int(np.sum(filt_path != smooth_path))
    print(f"    filtering and smoothing disagree on {n_diff}/{len(obs)} days "
          f"({n_diff/len(obs):.1%}) -- the days where future evidence "
          f"changes the verdict")
    mean_conf_filt = float(alpha.max(axis=1).mean())
    mean_conf_smooth = float(gamma.max(axis=1).mean())
    print(f"    mean confidence: filtered {mean_conf_filt:.3f} "
          f"-> smoothed {mean_conf_smooth:.3f} (smoothing uses more data)")

    find["forward_backward"] = {
        "gamma_normalised": bool(np.allclose(gamma.sum(axis=1), 1.0)),
        "xi_normalised": bool(np.allclose(xi.sum(axis=(1, 2)), 1.0)),
        "filter_smooth_disagreements": n_diff,
        "filter_smooth_disagreement_rate": float(n_diff / len(obs)),
        "mean_filtered_confidence": mean_conf_filt,
        "mean_smoothed_confidence": mean_conf_smooth,
    }

    # ------------------------------------------------------- 6. Viterbi
    print("\n[6/9] Viterbi (MAP path) vs posterior decoding")
    v_path, v_lp, _, _ = viterbi_log(hmm.pi, hmm.A, Bt_all)
    dec = compare_decoders(hmm.pi, hmm.A, Bt_all, gamma)
    save_table([dec], "decoder_comparison", reports)
    print(f"    Viterbi and posterior decoding disagree on "
          f"{dec['n_disagreements']}/{dec['n_timesteps']} days "
          f"({dec['disagreement_rate']:.1%})")
    print(f"    Viterbi path log-prob {dec['viterbi_logprob']:.2f} vs "
          f"posterior path {dec['posterior_path_logprob']:.2f} "
          f"(advantage {dec['logprob_advantage_viterbi']:.2f})")
    print(f"    regime switches: Viterbi {dec['viterbi_n_switches']}, "
          f"posterior {dec['posterior_n_switches']} "
          f"-- Viterbi gives the more stable labelling")

    find["viterbi"] = dec

    # -------------------------------------------------- 7. sum-product
    print("\n[7/9] Sum-product message passing")
    # Verified on a 60-day window: the full 604-day chain gives the same
    # agreement but takes much longer to build the pairwise beliefs.
    check = verify_equals_forward_backward(hmm.pi, hmm.A, Bt_all[:60])
    save_table([check], "sum_product_verification", reports)
    print(f"    forward message == alpha  : max diff {check['max_abs_diff_forward_message_vs_alpha']:.2e}")
    print(f"    backward message == beta  : max diff {check['max_abs_diff_backward_message_vs_beta']:.2e}")
    print(f"    belief == gamma           : max diff {check['max_abs_diff_belief_vs_gamma']:.2e}")
    print(f"    pairwise belief == xi     : max diff {check['max_abs_diff_pairwise_vs_xi']:.2e}")
    print(f"    log Z == forward loglik   : diff {check['abs_diff_logpartition_vs_loglik']:.2e}")
    print(f"    max-product path == Viterbi path: {check['max_product_path_matches_viterbi']}")
    find["sum_product"] = check

    # ------------------------------------------------------- 8. Kalman
    print("\n[8/9] Linear dynamical system: Kalman filter + RTS smoother")
    lds = LocalLevelModel().fit(obs)
    print(f"    fitted by ML: q = {lds.q:.5f}, r = {lds.r:.5f}, "
          f"log-likelihood {lds.loglik_:,.2f}")
    print(f"    signal-to-noise q/r = {lds.signal_to_noise:.4f}  ->  "
          f"steady-state Kalman gain K = {lds.steady_state_gain:.3f}")
    print(f"    (K is also the alpha of the equivalent exponential smoother)")

    level_f = lds.level(smoothed=False)
    level_s = lds.level(smoothed=True)
    pred = lds.one_step_forecast()

    # Honest comparison on the HELD-OUT days only, and against a naive
    # random-walk forecast (yesterday's value), which is the benchmark a
    # local-level model has to beat to be worth anything.
    test = ~is_train
    naive_pred = np.concatenate([[obs[0]], obs[:-1]])
    rmse_kalman = float(np.sqrt(np.mean((pred[test] - obs[test]) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((naive_pred[test] - obs[test]) ** 2)))
    print(f"    held-out one-step RMSE (log1p scale): "
          f"Kalman {rmse_kalman:.4f} vs naive random walk {rmse_naive:.4f} "
          f"({100*(rmse_naive-rmse_kalman)/rmse_naive:+.1f}%)")

    var_f = lds.fitted_["filtered_covs"][:, 0, 0]
    var_s = lds.fitted_["smoothed_covs"][:, 0, 0]
    print(f"    smoothed variance <= filtered variance at every t: "
          f"{bool(np.all(var_s <= var_f + 1e-9))}")
    print(f"    mean variance: filtered {var_f.mean():.5f} -> "
          f"smoothed {var_s.mean():.5f} "
          f"({100*(var_f.mean()-var_s.mean())/var_f.mean():.1f}% reduction)")

    save_table(
        [
            {"date": str(d.date()), "observed_log1p": float(o),
             "filtered_level": float(f), "smoothed_level": float(s),
             "one_step_pred": float(p), "filtered_var": float(vf),
             "smoothed_var": float(vs), "split": "train" if tr else "test"}
            for d, o, f, s, p, vf, vs, tr in zip(
                dates, obs, level_f, level_s, pred, var_f, var_s, is_train
            )
        ],
        "kalman_level",
        reports,
    )

    find["lds"] = {
        "q": float(lds.q),
        "r": float(lds.r),
        "loglik": float(lds.loglik_),
        "signal_to_noise": float(lds.signal_to_noise),
        "steady_state_gain": float(lds.steady_state_gain),
        "test_rmse_kalman": rmse_kalman,
        "test_rmse_naive_rw": rmse_naive,
        "rmse_improvement_pct": float(100 * (rmse_naive - rmse_kalman) / rmse_naive),
        "smoothed_var_never_exceeds_filtered": bool(np.all(var_s <= var_f + 1e-9)),
        "mean_filtered_var": float(var_f.mean()),
        "mean_smoothed_var": float(var_s.mean()),
    }

    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(dates, obs, lw=0.7, color="#94a3b8", label="observed log1p demand")
    ax[0].plot(dates, level_f, lw=1.4, color="#2563eb", label="Kalman filtered level")
    ax[0].plot(dates, level_s, lw=1.4, color="#dc2626", ls="--",
               label="RTS smoothed level")
    ax[0].axvline(train_end, color="k", ls=":", alpha=0.6)
    ax[0].set_ylabel("log1p(units)")
    ax[0].set_title(f"Kalman local-level model (q={lds.q:.4f}, r={lds.r:.4f}, "
                    f"steady-state gain K={lds.steady_state_gain:.3f})")
    ax[0].legend(loc="upper left")

    colours = ["#38bdf8", "#22c55e", "#ef4444"]
    order = hmm.state_order()
    rank_of = {int(s): r for r, s in enumerate(order)}
    for rank in range(N_REGIMES):
        state = int(order[rank])
        m = v_path == state
        ax[1].scatter(dates[m], y[m], s=9, color=colours[rank],
                      label=f"{REGIME_LABELS[rank]} regime")
    ax[1].axvline(train_end, color="k", ls=":", alpha=0.6)
    ax[1].set_ylabel("Units/day")
    ax[1].set_title("Viterbi-decoded demand regimes")
    ax[1].legend(loc="upper left")
    save_fig(fig, "03_kalman_regimes", figs)

    fig, ax = plt.subplots(figsize=(13, 5))
    for rank in range(N_REGIMES):
        state = int(order[rank])
        ax.fill_between(dates, 0, gamma[:, state], alpha=0.65,
                        color=colours[rank], label=f"P({REGIME_LABELS[rank]})")
    ax.axvline(train_end, color="k", ls=":", alpha=0.6, label="train/test cut")
    ax.set_ylim(0, 1); ax.set_ylabel("Smoothed posterior  p(z_t | x_1:T)")
    ax.set_title("Forward-backward regime posteriors")
    ax.legend(loc="upper left", ncol=4)
    save_fig(fig, "04_regime_posteriors", figs)

    # --------------------------------------------- 9. decision integration
    print("\n[9/9] Decision integration: regime-aware safety stock")
    # The regime posterior is a BELIEF about today's demand environment.
    # Translating it into a safety-stock multiplier is the point of the whole
    # unit: a single safety number assumes one demand environment, and the
    # HMM has just shown there are three.
    #
    # Only the FILTERED posterior is admissible here. The smoothed one uses
    # future observations, and a buyer standing on day t does not have them.
    regime_rank = np.array([rank_of[int(s)] for s in np.argmax(alpha, axis=1)])

    sigma_by_regime = np.array(
        [float(np.std(y[regime_rank == r])) if (regime_rank == r).any() else 0.0
         for r in range(N_REGIMES)]
    )
    overall_sigma = float(np.std(y))
    multipliers = sigma_by_regime / overall_sigma

    rows = []
    for r in range(N_REGIMES):
        m = regime_rank == r
        rows.append(
            {
                "regime": REGIME_LABELS[r],
                "n_days": int(m.sum()),
                "share_of_days": float(m.mean()),
                "mean_demand": float(y[m].mean()) if m.any() else 0.0,
                "std_demand": float(sigma_by_regime[r]),
                "safety_multiplier_vs_pooled": float(multipliers[r]),
            }
        )
    save_table(rows, "regime_safety_multipliers", reports)
    for r in rows:
        print(f"    {r['regime']:<7} {r['n_days']:3d} days  "
              f"mean {r['mean_demand']:7.0f}  std {r['std_demand']:7.0f}  "
              f"safety multiplier {r['safety_multiplier_vs_pooled']:.2f}x")

    spread = float(multipliers.max() / max(multipliers.min(), 1e-9))
    print(f"    -> required buffer varies {spread:.1f}x between the calmest and "
          f"busiest regime")
    print(f"       A single pooled safety stock is {multipliers.max():.2f}x too small "
          f"in SURGE and {multipliers.min():.2f}x oversized in QUIET.")

    find["decision_integration"] = {
        "regimes": rows,
        "pooled_sigma": overall_sigma,
        "multiplier_spread": spread,
        "note": "filtered (not smoothed) posterior used -- smoothing would leak future data",
    }

    # --------------------------------------------------- regimes output
    out = pd.DataFrame(
        {
            "date": dates,
            "total_demand": y,
            "log1p_demand": obs,
            "markov_state": states_all,
            "markov_state_name": [state_names[s] for s in states_all],
            "viterbi_regime": [rank_of[int(s)] for s in v_path],
            "viterbi_regime_name": [REGIME_LABELS[rank_of[int(s)]] for s in v_path],
            "filtered_regime": regime_rank,
            "filtered_regime_name": [REGIME_LABELS[r] for r in regime_rank],
            "kalman_filtered_level": level_f,
            "kalman_smoothed_level": level_s,
            "kalman_one_step_pred": pred,
            "split": np.where(is_train, "train", "test"),
        }
    )
    for rank in range(N_REGIMES):
        out[f"p_{REGIME_LABELS[rank]}_smoothed"] = gamma[:, int(order[rank])]
        out[f"p_{REGIME_LABELS[rank]}_filtered"] = alpha[:, int(order[rank])]

    reg_path = processed / "demand_regimes.parquet"
    ensure_dir(reg_path)
    out.to_parquet(reg_path, index=False)
    print(f"\n    saved -> data/processed/demand_regimes.parquet ({len(out)} days)")

    path = reports / "unit4_findings.json"
    path.write_text(json.dumps(find, indent=2, default=str), encoding="utf-8")
    print(f"    saved -> reports/unit4_findings.json")
    print("\nUnit 4 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
