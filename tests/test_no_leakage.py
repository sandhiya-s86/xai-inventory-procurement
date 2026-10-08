"""The leakage tests.

If any of these fail, every performance number in the project is worthless,
so they are deliberately blunt and check the property empirically rather than
trusting that the shift(1) calls in features/lags.py are still there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaierp.features.builder import (
    NON_FEATURE_COLS,
    add_target,
    classification_features,
    regression_features,
)
from xaierp.features.lags import add_lag_features, add_rolling_features


def _toy_panel(n: int = 60) -> pd.DataFrame:
    """One SKU with a strictly increasing demand series.

    Monotone demand makes leakage unmissable: if a 'past' feature ever equals
    or exceeds today's value, it must have seen the present or the future.
    """
    return pd.DataFrame(
        {
            "stock_code": ["A"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "demand": np.arange(1, n + 1, dtype=float),
        }
    )


def test_lag_features_only_use_the_past():
    df = add_lag_features(_toy_panel(), [1, 7])
    ok = df.dropna(subset=["lag_1", "lag_7"])
    assert (ok["lag_1"] < ok["demand"]).all(), "lag_1 reached today or later"
    assert (ok["lag_7"] < ok["demand"]).all(), "lag_7 reached today or later"
    # Exact expected values on a monotone series.
    assert np.allclose(ok["lag_1"], ok["demand"] - 1)
    assert np.allclose(ok["lag_7"], ok["demand"] - 7)


def test_rolling_window_excludes_today():
    df = add_rolling_features(_toy_panel(), [7])
    ok = df.dropna(subset=["rolling_mean_7"])
    # The window covers t-7..t-1, so its max is yesterday's value.
    assert (ok["rolling_max_7"] == ok["demand"] - 1).all(), (
        "rolling_max_7 includes the current day -- the .shift(1) is missing"
    )
    assert (ok["rolling_mean_7"] < ok["demand"]).all()


def test_rolling_requires_a_full_window():
    """A partially-filled window is a different statistic, not an estimate."""
    df = add_rolling_features(_toy_panel(20), [7])
    # Rows 0..6 cannot have a complete 7-day history ending yesterday.
    assert df["rolling_mean_7"].iloc[:7].isna().all()
    assert df["rolling_mean_7"].iloc[7:].notna().all()


def test_target_is_forward_looking_and_correct():
    df = add_target(_toy_panel(30), horizon=7)
    ok = df.dropna(subset=["future_7d_demand"])
    # On 1,2,3,... the sum of days t..t+6 starting at value v is 7v + 21.
    expected = 7 * ok["demand"] + 21
    assert np.allclose(ok["future_7d_demand"], expected)
    # The tail cannot have a full forward window.
    assert df["future_7d_demand"].iloc[-6:].isna().all()


def test_same_day_actuals_are_not_features(features):
    """demand / revenue / n_invoices for day t must never be model inputs."""
    for name in ("demand", "revenue", "n_invoices", "avg_price", "returned_units"):
        assert name in NON_FEATURE_COLS
        assert name not in regression_features(features)
        assert name not in classification_features(features)


def test_label_columns_are_not_features(features):
    for name in ("future_7d_demand", "reorder_flag", "stockout_risk",
                 "lead_time_demand_actual", "unmet_demand", "ending_stock"):
        assert name not in regression_features(features)
        assert name not in classification_features(features)


def test_no_feature_correlates_suspiciously_with_the_target(features):
    """An empirical backstop against a leak nobody anticipated.

    A feature that is nearly a linear image of the target is the signature of
    accidental leakage. Genuine demand history tops out well below this.
    """
    feats = regression_features(features)
    sample = features.sample(min(20_000, len(features)), random_state=0)
    y = sample["future_7d_demand"]
    offenders = {}
    for c in feats:
        s = sample[c]
        if s.std() == 0:
            continue
        r = abs(np.corrcoef(s, y)[0, 1])
        if r > 0.97:
            offenders[c] = float(r)
    assert not offenders, f"Suspiciously target-like features: {offenders}"


def test_splits_do_not_overlap_in_time(features):
    """Chronological split: train < val < test, with no date in two splits."""
    g = features.groupby("split")["date"].agg(["min", "max"])
    assert g.loc["train", "max"] < g.loc["val", "min"]
    assert g.loc["val", "max"] < g.loc["test", "min"]

    dates = features.groupby("split")["date"].apply(lambda s: set(s.unique()))
    assert not (dates["train"] & dates["val"])
    assert not (dates["val"] & dates["test"])
    assert not (dates["train"] & dates["test"])
