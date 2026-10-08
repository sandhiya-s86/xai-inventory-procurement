"""Data quality guarantees on the processed artefacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xaierp.utils.config import load_config, resolve


def test_panel_has_no_duplicate_sku_days(panel):
    assert not panel.duplicated(subset=["stock_code", "date"]).any()


def test_panel_demand_is_clean(panel):
    assert panel["demand"].notna().all()
    assert (panel["demand"] >= 0).all(), "negative demand survived cleaning"
    assert np.isfinite(panel["demand"]).all()


def test_panel_is_a_dense_rectangle(panel):
    """Every modelled SKU must span every trading day -- no ragged series."""
    n_sku = panel["stock_code"].nunique()
    n_day = panel["date"].nunique()
    assert len(panel) == n_sku * n_day
    assert (panel.groupby("stock_code")["date"].nunique() == n_day).all()


def test_prices_are_carried_not_zero_filled(panel):
    """An unsold item still has a price; zero would corrupt the cost basis."""
    assert panel["avg_price"].notna().all()
    assert (panel["avg_price"] > 0).all()


def test_non_trading_days_were_excluded(panel):
    """This retailer does not trade Saturdays; the calendar must reflect that."""
    dates = pd.Series(panel["date"].unique())
    saturdays = (dates.dt.dayofweek == 5).sum()
    span = pd.date_range(dates.min(), dates.max(), freq="D")
    expected_saturdays = (span.dayofweek == 5).sum()
    assert saturdays < expected_saturdays * 0.1, (
        f"{saturdays} Saturdays kept out of {expected_saturdays} in span -- "
        "closed days appear to have been zero-filled"
    )


def test_quality_report_exists_and_is_consistent():
    path = resolve(load_config()["paths"]["reports"]) / "data_quality_report.json"
    if not path.exists():
        pytest.skip("quality report not built -- run scripts/prepare_data.py")
    rep = json.loads(path.read_text(encoding="utf-8"))

    assert rep["raw"]["n_rows"] == 1_067_371, (
        "row count differs from the documented size of UCI Online Retail II"
    )
    assert rep["clean"]["n_rows"] <= rep["raw"]["n_rows"]

    # The cleaning log must account for every removed row.
    steps = rep["cleaning"]["steps"]
    assert steps[0]["rows_before"] == rep["cleaning"]["rows_in"]
    for a, b in zip(steps, steps[1:]):
        assert a["rows_after"] == b["rows_before"], "cleaning log has a gap"
    assert steps[-1]["rows_after"] == rep["cleaning"]["rows_out"]

    for s in steps:
        assert s["rows_removed"] == s["rows_before"] - s["rows_after"]
        assert s["reason"], f"step {s['step']} has no documented reason"


def test_features_have_no_missing_values(features):
    from xaierp.features.builder import classification_features, regression_features

    for col in regression_features(features) + classification_features(features):
        assert features[col].notna().all(), f"{col} contains NaN"
        assert np.isfinite(features[col]).all(), f"{col} contains inf"


def test_target_is_present_and_non_negative(features):
    assert features["future_7d_demand"].notna().all()
    assert (features["future_7d_demand"] >= 0).all()


def test_classification_label_is_binary(features):
    assert set(features["reorder_flag"].unique()) <= {0.0, 1.0}
    rate = features["reorder_flag"].mean()
    assert 0.05 < rate < 0.95, (
        f"reorder rate {rate:.2%} is degenerate -- the classification task "
        "would be trivial"
    )


def test_split_proportions_are_roughly_as_configured(features, cfg):
    counts = features["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(cfg["split"]["train_frac"], abs=0.08)
    assert counts["val"] == pytest.approx(cfg["split"]["val_frac"], abs=0.05)
    assert counts["test"] == pytest.approx(cfg["split"]["test_frac"], abs=0.05)
