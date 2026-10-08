"""Assemble the modelling table: features + target + chronological split.

Timing convention (stated once, enforced everywhere)
----------------------------------------------------
A row keyed (sku, t) represents a decision taken at the START of trading day t.

    features : may only use demand observed on t-1 and earlier
    target   : future_7d_demand = demand on days t, t+1, ... t+6

So demand on day t is part of the LABEL, never part of the input. Every lag
and rolling statistic in features/lags.py is built on a .shift(1) series to
make that true, and tests/test_no_leakage.py verifies it empirically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import ensure_dir, load_config, resolve
from .calendar import add_calendar_features
from .lags import (
    add_lag_features,
    add_price_features,
    add_returns_features,
    add_rolling_features,
    add_trend_features,
)

GROUP_KEY = "stock_code"

# Columns that exist in the table but must never be fed to a model: they are
# identifiers, the label itself, or raw same-day observations.
NON_FEATURE_COLS = {
    "stock_code",
    "date",
    "demand",            # same-day actual -- part of the label window
    "revenue",           # same-day actual
    "n_invoices",        # same-day actual
    "avg_price",         # same-day actual (lagged version is a feature)
    "returned_units",    # same-day actual
    "description",
    "future_7d_demand",
    "split",
    # Inventory/label columns added downstream by the simulator.
    "beginning_stock",
    "ending_stock",
    "arrivals",
    "unmet_demand",
    "reorder_flag",
    "stockout_risk",
    "lead_time_demand_actual",
    "consumed",
    "on_order",
    "orders_placed",
    "supplier_id",
    "supplier_capacity",
    "is_simulated_supply",
    "source_sheet",
}

FEATURE_GROUPS = {
    "calendar": [
        "day_of_week", "day_of_month", "day_of_year", "week_of_year",
        "month", "quarter", "year", "is_weekend", "is_month_start",
        "is_month_end", "is_quarter_end", "is_bank_holiday",
        "days_to_bank_holiday", "is_q4_peak",
    ],
    "lag": ["lag_1", "lag_7", "lag_14", "lag_28"],
    "rolling": [
        "rolling_mean_7", "rolling_std_7", "rolling_max_7", "rolling_min_7",
        "zero_rate_7",
        "rolling_mean_14", "rolling_std_14", "rolling_max_14",
        "rolling_min_14", "zero_rate_14",
        "rolling_mean_28", "rolling_std_28", "rolling_max_28",
        "rolling_min_28", "zero_rate_28",
    ],
    "trend": ["demand_momentum", "demand_momentum_ratio", "demand_cv_28"],
    "price": ["price_lag_1", "price_baseline_28", "price_discount_pct", "is_price_cut"],
    "returns": ["returns_mean_28"],
    "sku": ["sku_rank", "median_price"],
}

# Inventory/supply features. Every one of these is knowable at the START of
# day t by a buyer looking at the ERP screen, so they are legitimate inputs
# to the reorder classifier. They are SIMULATED values (see sim/), not real
# measurements, which is a data-provenance caveat rather than a leakage one.
INVENTORY_FEATURES = [
    "current_stock",
    "inventory_position",
    "days_of_cover",
    "stock_vs_leadtime_demand",
    "lead_time_days",
    "moq",
    "pack_size",
    "unit_cost",
    "supplier_reliability",
]


def regression_features(df: pd.DataFrame) -> list[str]:
    """Inputs for demand forecasting.

    Stock level is deliberately excluded. How much is on the shelf does not
    change how much customers want -- including it would model the baseline
    replenishment policy rather than demand, and would stop the forecast
    being reusable under a different policy.
    """
    allowed = {c for g, cols in FEATURE_GROUPS.items() for c in cols}
    return sorted(c for c in df.columns if c in allowed)


def classification_features(df: pd.DataFrame) -> list[str]:
    """Inputs for the reorder / stockout-risk classifier.

    Demand signal PLUS the current inventory and supply position, because
    'do we need to reorder?' is a question about both.
    """
    allowed = {c for g, cols in FEATURE_GROUPS.items() for c in cols}
    allowed |= set(INVENTORY_FEATURES)
    return sorted(c for c in df.columns if c in allowed)


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """future_7d_demand = demand summed over days t .. t+horizon-1.

    Implemented as a reversed rolling sum so the window is forward-looking in
    exactly the intended way, with min_periods=horizon so partial windows at
    the end of each SKU's series become NaN and get dropped rather than
    silently under-counted.
    """
    out = df.copy()
    rev = (
        out.iloc[::-1]
        .groupby(GROUP_KEY, sort=False)["demand"]
        .rolling(window=horizon, min_periods=horizon)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["future_7d_demand"] = rev.iloc[::-1].values
    return out


def build_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel -> fully-featured modelling table, rows with incomplete history dropped."""
    cfg = load_config()
    fcfg = cfg["features"]
    horizon = cfg["target"]["horizon_days"]

    df = panel.sort_values([GROUP_KEY, "date"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_lag_features(df, fcfg["lags"])
    df = add_rolling_features(df, fcfg["rolling_windows"])
    df = add_trend_features(df)
    df = add_returns_features(df)
    df = add_price_features(df)
    df = add_target(df, horizon)

    n_before = len(df)

    # Drop rows whose history window is not fully observed. This is why
    # min_periods equals the window everywhere above: a half-filled 28-day
    # mean is a different statistic, not a usable approximation.
    feat_cols = feature_columns(df)
    df = df.dropna(subset=feat_cols)
    n_after_features = len(df)

    # Drop rows whose forward window runs past the end of the data.
    df = df.dropna(subset=["future_7d_demand"])
    n_after_target = len(df)

    df = df.reset_index(drop=True)
    df.attrs["rows_dropped_incomplete_history"] = n_before - n_after_features
    df.attrs["rows_dropped_incomplete_target"] = n_after_features - n_after_target
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: every numeric column that is not an id, label or same-day actual."""
    cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)


def chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row train / val / test by DATE, never at random.

    A random split would put day t+1 in train and day t in test for the same
    SKU. Because consecutive days share almost all of their rolling-window
    history, that leaks the answer and inflates every score. The split is
    therefore made on two global date cut-offs so that all three sets are
    strictly ordered in time, with no SKU straddling a boundary.
    """
    cfg = load_config()["split"]
    dates = np.sort(df["date"].unique())
    n = len(dates)
    i_train = int(np.floor(n * cfg["train_frac"]))
    i_val = int(np.floor(n * (cfg["train_frac"] + cfg["val_frac"])))

    train_end = dates[i_train - 1]
    val_end = dates[i_val - 1]

    out = df.copy()
    out["split"] = np.where(
        out["date"] <= train_end,
        "train",
        np.where(out["date"] <= val_end, "val", "test"),
    )
    out.attrs["train_end"] = str(pd.Timestamp(train_end).date())
    out.attrs["val_end"] = str(pd.Timestamp(val_end).date())
    return out


def save_features(df: pd.DataFrame) -> Path:
    cfg = load_config()
    out = resolve(cfg["paths"]["processed"]) / "feature_table.parquet"
    ensure_dir(out)
    df.to_parquet(out, index=False)
    return out


def load_features() -> pd.DataFrame:
    cfg = load_config()
    path = resolve(cfg["paths"]["processed"]) / "feature_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found at {path}.\nRun: python scripts/build_features.py"
        )
    return pd.read_parquet(path)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convenience: {'train': ..., 'val': ..., 'test': ...}."""
    return {name: df.loc[df["split"] == name].copy() for name in ("train", "val", "test")}


def xy(df: pd.DataFrame, features: list[str], target: str, dtype=np.float32):
    """Feature matrix and target vector as plain NumPy arrays.

    float32 by default. It halves the memory of every design matrix, which
    matters on a constrained machine, and the extra precision of float64 is
    irrelevant for features that are counts and prices. The target stays
    float64 so that error metrics accumulate cleanly.
    """
    X = df[features].to_numpy(dtype=dtype)
    y = df[target].to_numpy(dtype=np.float64)
    return X, y
