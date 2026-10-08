"""Lag and rolling features.

The single most important rule in this file: a feature computed for date t may
only use observations from dates strictly before t.

pandas' .rolling() includes the current row by default, so every rolling
statistic here is built on a .shift(1) series. Forgetting that shift is the
classic way to leak the target into the features, and tests/test_no_leakage.py
exists specifically to catch its reintroduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEY = "stock_code"
VALUE_COL = "demand"


def add_lag_features(
    df: pd.DataFrame, lags: list[int], value_col: str = VALUE_COL
) -> pd.DataFrame:
    """demand observed `lag` trading days before the decision date."""
    out = df.copy()
    g = out.groupby(GROUP_KEY, sort=False)[value_col]
    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, windows: list[int], value_col: str = VALUE_COL
) -> pd.DataFrame:
    """Rolling mean / std / min / max over the window ENDING YESTERDAY."""
    out = df.copy()
    shifted = out.groupby(GROUP_KEY, sort=False)[value_col].shift(1)
    out["_prev"] = shifted
    g = out.groupby(GROUP_KEY, sort=False)["_prev"]

    for w in windows:
        roll = g.rolling(window=w, min_periods=w)
        out[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"rolling_std_{w}"] = roll.std().reset_index(level=0, drop=True)
        out[f"rolling_max_{w}"] = roll.max().reset_index(level=0, drop=True)
        out[f"rolling_min_{w}"] = roll.min().reset_index(level=0, drop=True)
        # Share of recent days with no sales -- captures intermittency, which
        # is what makes this an interesting forecasting problem.
        out[f"zero_rate_{w}"] = (
            g.rolling(window=w, min_periods=w)
            .apply(lambda a: float((a == 0).mean()), raw=True)
            .reset_index(level=0, drop=True)
        )

    out = out.drop(columns=["_prev"])
    return out


def add_trend_features(df: pd.DataFrame, short: int = 7, long: int = 28) -> pd.DataFrame:
    """Momentum: is recent demand running above or below the longer baseline?

    This is the feature that most often drives the 'demand is rising' sentence
    in the generated explanations, so it is computed explicitly rather than
    left for the model to infer.
    """
    out = df.copy()
    s, l = f"rolling_mean_{short}", f"rolling_mean_{long}"
    if s in out.columns and l in out.columns:
        out["demand_momentum"] = out[s] - out[l]
        # Guard the ratio: a SKU with zero 28-day mean would divide by zero.
        out["demand_momentum_ratio"] = out[s] / out[l].replace(0.0, np.nan)
        out["demand_momentum_ratio"] = (
            out["demand_momentum_ratio"].astype("float64").fillna(1.0)
        )
    # Coefficient of variation: volatility relative to level, which is what
    # safety-stock sizing actually depends on.
    if f"rolling_std_{long}" in out.columns and l in out.columns:
        out["demand_cv_28"] = (
            out[f"rolling_std_{long}"] / out[l].replace(0.0, np.nan)
        ).astype("float64").fillna(0.0)
    return out


def add_returns_features(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Rolling returned units, again lagged by one day."""
    out = df.copy()
    if "returned_units" not in out.columns:
        return out
    shifted = out.groupby(GROUP_KEY, sort=False)["returned_units"].shift(1)
    out["_prev_ret"] = shifted
    out[f"returns_mean_{window}"] = (
        out.groupby(GROUP_KEY, sort=False)["_prev_ret"]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out = out.drop(columns=["_prev_ret"])
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Yesterday's price and its deviation from the SKU's own baseline.

    Online Retail II has no promotion flag. Price movement is the closest
    honest proxy: a unit price below the SKU's trailing norm behaves like a
    discount. This is documented as a proxy, not claimed as a promotion field.
    """
    out = df.copy()
    g = out.groupby(GROUP_KEY, sort=False)["avg_price"]
    out["price_lag_1"] = g.shift(1)
    baseline = (
        out.groupby(GROUP_KEY, sort=False)["avg_price"]
        .shift(1)
        .groupby(out[GROUP_KEY])
        .rolling(window=28, min_periods=28)
        .median()
        .reset_index(level=0, drop=True)
    )
    out["price_baseline_28"] = baseline
    out["price_discount_pct"] = (
        (out["price_baseline_28"] - out["price_lag_1"])
        / out["price_baseline_28"].replace(0.0, np.nan)
    ).astype("float64").fillna(0.0) * 100.0
    out["is_price_cut"] = (out["price_discount_pct"] > 5.0).astype("int8")
    return out
