"""Feature calculations are arithmetically correct."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaierp.features.calendar import UK_BANK_HOLIDAYS, add_calendar_features
from xaierp.features.lags import (
    add_price_features,
    add_rolling_features,
    add_trend_features,
)


def _panel(values, start="2024-01-01"):
    n = len(values)
    return pd.DataFrame(
        {
            "stock_code": ["A"] * n,
            "date": pd.date_range(start, periods=n, freq="D"),
            "demand": np.asarray(values, dtype=float),
        }
    )


def test_calendar_fields_are_correct():
    df = add_calendar_features(_panel([1] * 10, start="2024-01-01"))
    # 2024-01-01 was a Monday.
    assert df["day_of_week"].iloc[0] == 0
    assert df["is_weekend"].iloc[0] == 0
    assert df["is_weekend"].iloc[5] == 1          # Saturday
    assert df["is_month_start"].iloc[0] == 1
    assert df["month"].iloc[0] == 1
    assert df["quarter"].iloc[0] == 1


def test_bank_holiday_flag_matches_the_published_list():
    df = add_calendar_features(
        pd.DataFrame({"date": UK_BANK_HOLIDAYS[:5], "stock_code": "A", "demand": 1.0})
    )
    assert df["is_bank_holiday"].all()
    assert (df["days_to_bank_holiday"] == 0).all()


def test_q4_peak_flag():
    df = add_calendar_features(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2011-10-15", "2011-11-15", "2011-12-15", "2011-01-15"]
                ),
                "stock_code": "A",
                "demand": 1.0,
            }
        )
    )
    assert list(df["is_q4_peak"]) == [0, 1, 1, 0]


def test_rolling_mean_matches_a_hand_calculation():
    vals = [10, 20, 30, 40, 50, 60, 70, 80]
    df = add_rolling_features(_panel(vals), [3])
    # At index 3 the window covers indices 0..2 -> mean of 10,20,30 = 20.
    assert df["rolling_mean_3"].iloc[3] == pytest.approx(20.0)
    assert df["rolling_max_3"].iloc[3] == pytest.approx(30.0)
    assert df["rolling_min_3"].iloc[3] == pytest.approx(10.0)
    # Index 0,1,2 cannot have a full 3-day history ending yesterday.
    assert df["rolling_mean_3"].iloc[:3].isna().all()


def test_zero_rate_counts_days_without_sales():
    df = add_rolling_features(_panel([0, 0, 5, 7, 9, 11]), [3])
    # At index 3 the window is indices 0..2 = [0, 0, 5] -> two of three are zero.
    assert df["zero_rate_3"].iloc[3] == pytest.approx(2 / 3)


def test_momentum_compares_short_against_long_window():
    rng = np.random.default_rng(0)
    rising = np.concatenate([rng.normal(10, 1, 40), rng.normal(60, 1, 40)])
    df = add_rolling_features(_panel(rising), [7, 28])
    df = add_trend_features(df, short=7, long=28)
    tail = df.dropna(subset=["demand_momentum"]).iloc[-1]
    assert tail["demand_momentum"] > 0, "rising demand must give positive momentum"
    assert tail["demand_momentum_ratio"] > 1


def test_momentum_ratio_never_divides_by_zero():
    df = add_rolling_features(_panel([0.0] * 40), [7, 28])
    df = add_trend_features(df, short=7, long=28)
    assert np.isfinite(df["demand_momentum_ratio"]).all()
    assert np.isfinite(df["demand_cv_28"]).all()
    # With no demand at all the ratio falls back to the neutral value 1.
    assert (df["demand_momentum_ratio"].dropna() == 1.0).all()


def test_price_discount_is_measured_against_the_trailing_norm():
    n = 40
    price = np.full(n, 10.0)
    price[-1] = 7.0                       # a 30% cut on the final day
    df = pd.DataFrame(
        {
            "stock_code": ["A"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "demand": 5.0,
            "avg_price": price,
        }
    )
    out = add_price_features(df)
    # The cut shows up the day AFTER it happens, because the feature is lagged.
    assert out["price_discount_pct"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert out["is_price_cut"].iloc[-1] == 0


def test_price_cut_is_detected_one_day_later():
    n = 41
    price = np.full(n, 10.0)
    price[-2] = 7.0
    df = pd.DataFrame(
        {
            "stock_code": ["A"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "demand": 5.0,
            "avg_price": price,
        }
    )
    out = add_price_features(df)
    assert out["price_discount_pct"].iloc[-1] == pytest.approx(30.0, abs=1e-6)
    assert out["is_price_cut"].iloc[-1] == 1


def test_features_are_computed_per_sku_independently():
    """SKU B's history must never bleed into SKU A's features."""
    df = pd.DataFrame(
        {
            "stock_code": ["A"] * 10 + ["B"] * 10,
            "date": list(pd.date_range("2024-01-01", periods=10)) * 2,
            "demand": [1.0] * 10 + [1000.0] * 10,
        }
    )
    out = add_rolling_features(df, [3])
    a = out[out["stock_code"] == "A"].dropna(subset=["rolling_mean_3"])
    b = out[out["stock_code"] == "B"].dropna(subset=["rolling_mean_3"])
    assert (a["rolling_mean_3"] == 1.0).all()
    assert (b["rolling_mean_3"] == 1000.0).all()


def test_real_feature_table_column_groups(features):
    from xaierp.features.builder import classification_features, regression_features

    reg = regression_features(features)
    clf = classification_features(features)
    # Classification sees everything regression sees, plus inventory state.
    assert set(reg).issubset(set(clf))
    assert "current_stock" in clf and "current_stock" not in reg
    assert "lead_time_days" in clf and "lead_time_days" not in reg
    # Core demand features must be present.
    for c in ("rolling_mean_7", "rolling_mean_28", "lag_1", "lag_7", "zero_rate_28"):
        assert c in reg
