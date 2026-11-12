"""Linear Dynamical Systems: the Kalman filter and RTS smoother (Unit 4).

An LDS is the HMM with a CONTINUOUS hidden state:

    z_t = A z_{t-1} + w_t     w_t ~ N(0, Q)    state / dynamics
    x_t = C z_t     + v_t     v_t ~ N(0, R)    observation

Where an HMM's hidden variable is one of N discrete regimes, an LDS's is a
real vector -- here, the true underlying demand LEVEL, which drifts
continuously and is never observed directly because every day's sales are
that level plus a large amount of transaction noise.

Inference is the Kalman filter, which is exact for linear-Gaussian models:

    predict   mu_t^-  = A mu_{t-1}
              P_t^-   = A P_{t-1} A^T + Q
    update    K_t     = P_t^- C^T (C P_t^- C^T + R)^-1
              mu_t    = mu_t^- + K_t (x_t - C mu_t^-)
              P_t     = (I - K_t C) P_t^-

The Kalman gain K is the whole story. It weighs model against measurement:
small R (trustworthy sensor) drives K up and the estimate follows the data;
small Q (smooth dynamics) drives K down and the estimate follows the model.
The innovation x_t - C mu_t^- is the part of today's observation the model
did not anticipate -- the only part that carries new information.

Filtering vs smoothing, again
-----------------------------
The filter gives p(z_t | x_1:t), using only data up to t. The RTS smoother
runs backwards afterwards to give p(z_t | x_1:T), using the whole series. For
a FORECAST only the filter is admissible -- the smoother peeks at the future,
and using it to produce features would be the same leakage the project bans
everywhere else. The smoother is for retrospective analysis: "what was the
true demand level last March?" Both are provided, and the distinction is
enforced in how scripts/run_sequential.py uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def kalman_filter(
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
) -> dict:
    """Forward Kalman recursion.

    Returns filtered means/covariances, the one-step-ahead predictions
    (which are what a forecast actually uses), the innovations, and the
    log-likelihood of the observed sequence in innovation form.

    `y` is (T, d_obs); scalars should be passed as shape (T, 1).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T
    A, C, Q, R = map(lambda m: np.atleast_2d(np.asarray(m, dtype=float)), (A, C, Q, R))
    mu0 = np.asarray(mu0, dtype=float).ravel()
    P0 = np.atleast_2d(np.asarray(P0, dtype=float))

    T, d_obs = y.shape
    d_state = A.shape[0]

    mu_filt = np.zeros((T, d_state))
    P_filt = np.zeros((T, d_state, d_state))
    mu_pred = np.zeros((T, d_state))
    P_pred = np.zeros((T, d_state, d_state))
    innov = np.zeros((T, d_obs))
    S_all = np.zeros((T, d_obs, d_obs))
    K_all = np.zeros((T, d_state, d_obs))

    mu, P = mu0, P0
    loglik = 0.0
    I = np.eye(d_state)

    for t in range(T):
        # ---- predict
        if t == 0:
            m_pred, p_pred = mu, P
        else:
            m_pred = A @ mu
            p_pred = A @ P @ A.T + Q
        mu_pred[t], P_pred[t] = m_pred, p_pred

        # ---- update
        resid = y[t] - C @ m_pred                 # innovation
        S = C @ p_pred @ C.T + R                  # innovation covariance
        S_inv = np.linalg.inv(S)
        K = p_pred @ C.T @ S_inv                  # Kalman gain

        mu = m_pred + K @ resid
        # Joseph-free simple form; symmetrised below to resist drift.
        P = (I - K @ C) @ p_pred
        P = (P + P.T) / 2.0

        mu_filt[t], P_filt[t] = mu, P
        innov[t], S_all[t], K_all[t] = resid, S, K

        # Innovation-form log-likelihood: each observation scored against its
        # own one-step-ahead predictive distribution N(C mu^-, S).
        sign, logdet = np.linalg.slogdet(S)
        loglik += -0.5 * (
            d_obs * np.log(2 * np.pi) + logdet + float(resid @ S_inv @ resid)
        )

    return {
        "filtered_means": mu_filt,
        "filtered_covs": P_filt,
        "predicted_means": mu_pred,
        "predicted_covs": P_pred,
        "innovations": innov,
        "innovation_covs": S_all,
        "kalman_gains": K_all,
        "loglik": float(loglik),
    }


def rts_smoother(filt: dict, A: np.ndarray, Q: np.ndarray) -> dict:
    """Rauch-Tung-Striebel backward smoothing pass.

        J_t    = P_t A^T (P_{t+1}^-)^-1
        mu_t^s = mu_t + J_t (mu_{t+1}^s - mu_{t+1}^-)
        P_t^s  = P_t + J_t (P_{t+1}^s - P_{t+1}^-) J_t^T

    Because it conditions on strictly more data, the smoothed covariance is
    never larger than the filtered one -- a property the test-suite checks.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))

    mu_f, P_f = filt["filtered_means"], filt["filtered_covs"]
    mu_p, P_p = filt["predicted_means"], filt["predicted_covs"]
    T = mu_f.shape[0]

    mu_s = mu_f.copy()
    P_s = P_f.copy()
    J_all = np.zeros_like(P_f)

    for t in range(T - 2, -1, -1):
        J = P_f[t] @ A.T @ np.linalg.inv(P_p[t + 1])
        mu_s[t] = mu_f[t] + J @ (mu_s[t + 1] - mu_p[t + 1])
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_p[t + 1]) @ J.T
        P_s[t] = (P_s[t] + P_s[t].T) / 2.0
        J_all[t] = J

    return {"smoothed_means": mu_s, "smoothed_covs": P_s, "gains": J_all}


@dataclass
class KalmanFilter:
    """Object wrapper over the two recursions above."""

    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    mu0: np.ndarray
    P0: np.ndarray

    def filter(self, y: np.ndarray) -> dict:
        return kalman_filter(y, self.A, self.C, self.Q, self.R, self.mu0, self.P0)

    def smooth(self, y: np.ndarray) -> dict:
        f = self.filter(y)
        s = rts_smoother(f, self.A, self.Q)
        return {**f, **s}

    def log_likelihood(self, y: np.ndarray) -> float:
        return self.filter(y)["loglik"]


@dataclass
class LocalLevelModel:
    """The scalar local-level (random-walk-plus-noise) model, for demand.

        level_t = level_{t-1} + w_t      w ~ N(0, q)
        x_t     = level_t     + v_t      v ~ N(0, r)

    A = C = 1, so the only parameters are the two noise variances. Their
    RATIO q/r (the signal-to-noise ratio) is what matters: it alone sets the
    steady-state Kalman gain, and therefore how fast the estimate reacts.

    * q/r large  -> the level genuinely moves; track the data closely.
    * q/r small  -> the level is stable; heavily smooth the noise away.

    This is the right model for a demand level precisely BECAUSE the series
    is non-stationary (see `stationarity_report`): a random walk has no fixed
    mean to revert to. It is also exactly equivalent to simple exponential
    smoothing with alpha = the steady-state gain -- so fitting q and r by
    maximum likelihood is a principled way of choosing a smoothing constant
    instead of picking 0.3 because it looked reasonable.
    """

    q: float = 1.0
    r: float = 1.0
    mu0: float = 0.0
    p0: float = 1e4          # diffuse prior: the initial level is unknown

    loglik_: float = float("nan")
    fitted_: dict = field(default_factory=dict)

    def _kf(self) -> KalmanFilter:
        return KalmanFilter(
            A=np.array([[1.0]]),
            C=np.array([[1.0]]),
            Q=np.array([[self.q]]),
            R=np.array([[self.r]]),
            mu0=np.array([self.mu0]),
            P0=np.array([[self.p0]]),
        )

    def fit(self, y: np.ndarray, n_grid: int = 25) -> "LocalLevelModel":
        """Maximum-likelihood fit of q and r over a log-spaced grid.

        Only the ratio is identifiable in terms of the gain, but the absolute
        scale sets the predictive variance, so both are fitted. A grid is used
        rather than a gradient optimiser because the surface is 2-D, smooth,
        and cheap to evaluate -- and a grid cannot diverge or land on a
        negative variance the way an unconstrained optimiser can.
        """
        y = np.asarray(y, dtype=float).ravel()
        var = max(float(y.var()), 1e-8)

        self.mu0 = float(y[0])
        grid = np.logspace(-4, 1, n_grid) * var

        best = (-np.inf, var, var)
        for q in grid:
            for r in grid:
                self.q, self.r = float(q), float(r)
                ll = self._kf().log_likelihood(y[:, None])
                if ll > best[0]:
                    best = (ll, float(q), float(r))

        self.loglik_, self.q, self.r = best
        self.fitted_ = self._kf().smooth(y[:, None])
        return self

    @property
    def signal_to_noise(self) -> float:
        return float(self.q / self.r) if self.r > 0 else float("inf")

    @property
    def steady_state_gain(self) -> float:
        """Steady-state Kalman gain K, solved from the scalar Riccati equation.

        With p = steady-state prior variance:  p = (p r)/(p + r) + q, giving
        p^2 - q p - q r = 0, so p = (q + sqrt(q^2 + 4qr)) / 2 and K = p/(p+r).
        K is also the alpha of the equivalent exponential smoother.
        """
        q, r = self.q, self.r
        p = (q + np.sqrt(q**2 + 4 * q * r)) / 2.0
        return float(p / (p + r))

    def level(self, smoothed: bool = False) -> np.ndarray:
        """The estimated demand level. `smoothed=True` uses future data."""
        key = "smoothed_means" if smoothed else "filtered_means"
        return self.fitted_[key].ravel()

    def one_step_forecast(self) -> np.ndarray:
        """One-step-ahead predictions -- leakage-free, usable as a feature.

        predicted_means[t] is formed from x_1:t-1 only, which is exactly the
        timing convention the rest of the project enforces.
        """
        return self.fitted_["predicted_means"].ravel()

    def forecast(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Forecast h steps past the end of the data.

        For a random walk the point forecast is FLAT at the last level -- the
        model's honest statement that it knows of no trend. The variance,
        however, grows linearly at q per step, so the interval fans out. A
        flat forecast with an expanding band is the correct output here, not
        a failure to find a pattern.
        """
        mu = float(self.fitted_["filtered_means"][-1, 0])
        p = float(self.fitted_["filtered_covs"][-1, 0, 0])
        means = np.full(h, mu)
        variances = p + self.q * np.arange(1, h + 1) + self.r
        return means, variances
