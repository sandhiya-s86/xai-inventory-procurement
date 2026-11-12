"""Sequential data: the groundwork the rest of Unit 4 is built on.

The defining property is that ORDER CARRIES INFORMATION. The exact joint
distribution of a sequence always factorises by the chain rule

    p(x_1:T) = p(x_1) PROD_{t=2..T} p(x_t | x_1:t-1)

and every sequential model in this unit is a *simplification* of the
conditioning history on the right:

    finite window (AR)   condition on the last L observations
    Markov chain         condition on the current state only
    HMM / LDS            condition on a hidden state that summarises the past

This project's existing feature table is the finite-window case already:
lag_1, lag_7, lag_14, lag_28 are a length-28 window, and the chronological
split in features/builder.py is the rolling-origin discipline below applied
once. What Unit 4 adds is the state-space view, where the thing being carried
forward is not raw history but a belief about an unobserved variable.

The leakage warning in the notes -- "random shuffling can leak future
information into training" -- is already enforced project-wide and tested in
tests/test_no_leakage.py. `rolling_origin_splits` generalises it from one
fixed cut to the repeated-refit evaluation used for the sequential models.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(y: np.ndarray, max_lag: int = 30) -> list[dict]:
    """ACF from lag 0 to `max_lag`, with approximate 95% significance bands.

    rho(k) = Cov(x_t, x_t-k) / Var(x_t), using the standard biased estimator
    (divide by n, not n-k). The biased form is what every ACF plot uses: it
    damps the long lags where few pairs are available, giving a positive
    semi-definite sequence instead of a wildly noisy tail.

    The band is +/- 1.96/sqrt(n), the white-noise null. A spike outside it is
    real dependence at that lag; for daily retail the lag-7 spike is the
    weekly trading cycle.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    yc = y - y.mean()
    denom = float(np.sum(yc**2))
    band = 1.96 / np.sqrt(n)

    rows = []
    for k in range(max_lag + 1):
        if denom == 0:
            rho = 1.0 if k == 0 else 0.0
        elif k == 0:
            rho = 1.0
        else:
            rho = float(np.sum(yc[k:] * yc[:-k]) / denom)
        rows.append(
            {
                "lag": k,
                "acf": rho,
                "significant": bool(abs(rho) > band and k > 0),
                "conf_band": float(band),
            }
        )
    return rows


def lag_matrix(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix for a finite-memory predictor  y_t ~ y_{t-1} .. y_{t-L}.

    Row i of X holds [y_{t-1}, y_{t-2}, ..., y_{t-L}] for t = L + i, and
    ``Y[i] = y_t``. Column 0 is the MOST RECENT lag. The first L observations
    cannot be predicted (no history), so they appear only as inputs.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n_lags >= n:
        raise ValueError(f"n_lags={n_lags} must be < len(y)={n}")

    X = np.column_stack([y[n_lags - k - 1 : n - k - 1] for k in range(n_lags)])
    return X, y[n_lags:]


def chain_rule_logprob(y: np.ndarray, mu: float, sigma: float,
                       n_lags: int = 0, coefs: np.ndarray | None = None) -> dict:
    """Sequence log-likelihood under two different conditioning assumptions.

    Makes the chain rule concrete. ``independent`` scores the data as if the
    observations were i.i.d. N(mu, sigma^2) -- order ignored entirely.
    ``autoregressive`` scores each point given its previous `n_lags` values.

    If the series really is dependent, the AR log-likelihood is HIGHER: the
    history removes uncertainty that the i.i.d. model has to pay for. The
    difference is a direct measurement of how much information the ordering
    carries, which is the claim the unit opens with.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    def gauss_ll(resid, s):
        s = max(float(s), 1e-12)
        return float(
            -0.5 * len(resid) * np.log(2 * np.pi * s**2)
            - 0.5 * np.sum(resid**2) / s**2
        )

    out = {"independent": gauss_ll(y - mu, sigma), "n": n}

    if n_lags > 0:
        X, Y = lag_matrix(y, n_lags)
        Xd = np.column_stack([np.ones(len(X)), X])
        if coefs is None:
            coefs, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
        resid = Y - Xd @ coefs
        out["autoregressive"] = gauss_ll(resid, resid.std(ddof=0))
        out["n_lags"] = n_lags
        out["coefficients"] = np.asarray(coefs).tolist()
        # Per-observation so the two are comparable despite different n.
        out["gain_per_obs"] = out["autoregressive"] / len(Y) - out["independent"] / n
    return out


def stationarity_report(y: np.ndarray, n_blocks: int = 4) -> dict:
    """Is the series stationary? Augmented Dickey-Fuller plus a block check.

    Stationarity means the distributional properties do not drift over time.
    Two independent readings are reported because they fail differently:

    * ADF tests for a unit root. p < 0.05 rejects "has a unit root", i.e.
      evidence FOR stationarity. It is blind to variance drift.
    * Splitting the series into blocks and comparing their means and standard
      deviations catches exactly that drift. `std_ratio` far from 1 means the
      variance is changing even when ADF is happy.

    This matters downstream: the Kalman local-level model in lds.py is a
    random walk precisely BECAUSE the demand level is non-stationary, so
    confirming non-stationarity is what justifies that model choice.
    """
    y = np.asarray(y, dtype=float)
    out: dict = {"n": len(y), "mean": float(y.mean()), "std": float(y.std(ddof=0))}

    try:
        from statsmodels.tsa.stattools import adfuller

        # result_object=False pins the legacy tuple return. statsmodels will
        # flip the default in 0.16; asking for the old shape explicitly keeps
        # the unpacking below correct across both versions.
        stat, pval, lags, nobs, crit, _ = adfuller(
            y, autolag="AIC", result_object=False
        )
        out.update(
            adf_statistic=float(stat),
            adf_pvalue=float(pval),
            adf_lags=int(lags),
            adf_crit_5pct=float(crit["5%"]),
            adf_stationary=bool(pval < 0.05),
        )
    except Exception as exc:                      # pragma: no cover
        out["adf_error"] = str(exc)

    blocks = np.array_split(y, n_blocks)
    means = np.array([float(b.mean()) for b in blocks])
    stds = np.array([float(b.std(ddof=0)) for b in blocks])
    out.update(
        block_means=means.tolist(),
        block_stds=stds.tolist(),
        mean_drift=float(means.max() - means.min()),
        # >1.5 or <0.67 is a practical flag for heteroscedasticity.
        std_ratio=float(stds.max() / stds.min()) if stds.min() > 0 else float("inf"),
    )
    return out


def rolling_origin_splits(
    n: int, n_splits: int = 5, min_train: int | None = None, horizon: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window (rolling-origin) train/test index pairs.

    The sequential replacement for k-fold cross-validation. Each split trains
    on everything up to an origin and tests on the `horizon` steps after it;
    the origin then moves forward. Training sets only ever grow, and no test
    index is ever earlier than a training index.

    K-fold would put day t+1 in train and day t in test, which both leaks the
    answer and scores the model on a task it will never face in production.
    """
    if min_train is None:
        min_train = max(horizon, n // 2)
    if min_train + horizon > n:
        raise ValueError(
            f"Need min_train + horizon <= n; got {min_train} + {horizon} > {n}"
        )

    last_origin = n - horizon
    origins = np.linspace(min_train, last_origin, n_splits, dtype=int)
    # np.linspace can repeat an origin when the range is short; keep the
    # distinct ones rather than silently scoring the same split twice.
    origins = np.unique(origins)

    return [
        (np.arange(0, o), np.arange(o, min(o + horizon, n)))
        for o in origins
    ]
