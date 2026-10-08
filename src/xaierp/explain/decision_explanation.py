"""Turn numbers into an account of WHY a particular order was recommended.

Two layers, deliberately kept separate:

  1. Why this DEMAND FORECAST?   -- SHAP attributions over demand features.
  2. Why this ORDER QUANTITY?    -- the exact arithmetic decomposition from
                                    policy/reorder.py, which sums to the
                                    recommended quantity.

Layer 2 is the one that matters for a buyer, and it is the one most XAI
demos skip. "SHAP says rolling_mean_28 is important" does not tell anyone
why the system asked for 300 units. The component breakdown does.

Every sentence produced here is generated from a real computed value. There
are no canned explanation templates that fire without checking the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..policy.reorder import ReorderDecision

# Human-readable names. Anything not listed falls back to the raw column
# name rather than inventing a description.
FEATURE_LABELS = {
    "rolling_mean_7": "average daily demand over the last 7 days",
    "rolling_mean_14": "average daily demand over the last 14 days",
    "rolling_mean_28": "average daily demand over the last 28 days",
    "rolling_std_7": "demand volatility over the last 7 days",
    "rolling_std_28": "demand volatility over the last 28 days",
    "rolling_max_7": "peak daily demand in the last 7 days",
    "rolling_max_28": "peak daily demand in the last 28 days",
    "rolling_min_7": "lowest daily demand in the last 7 days",
    "rolling_min_28": "lowest daily demand in the last 28 days",
    "lag_1": "demand on the previous trading day",
    "lag_7": "demand 7 trading days ago",
    "lag_14": "demand 14 trading days ago",
    "lag_28": "demand 28 trading days ago",
    "zero_rate_7": "share of the last 7 days with no sales",
    "zero_rate_28": "share of the last 28 days with no sales",
    "demand_momentum": "recent demand minus the longer-run baseline",
    "demand_momentum_ratio": "recent demand as a ratio of the baseline",
    "demand_cv_28": "demand variability relative to its own level",
    "price_lag_1": "yesterday's unit price",
    "price_discount_pct": "price discount versus the 28-day norm",
    "is_price_cut": "whether the item is currently discounted",
    "returns_mean_28": "average returned units over the last 28 days",
    "day_of_week": "day of the week",
    "month": "month of the year",
    "week_of_year": "week of the year",
    "is_q4_peak": "whether this falls in the November-December peak",
    "is_bank_holiday": "whether this is a UK bank holiday",
    "days_to_bank_holiday": "days until the next bank holiday",
    "current_stock": "stock currently available",
    "inventory_position": "stock on hand plus stock already on order",
    "days_of_cover": "days of cover at current demand",
    "stock_vs_leadtime_demand": "stock relative to expected lead-time demand",
    "lead_time_days": "supplier lead time",
    "moq": "supplier minimum order quantity",
    "pack_size": "supplier pack size",
    "unit_cost": "unit purchase cost",
    "sku_rank": "how large this SKU is by revenue",
    "median_price": "the item's typical selling price",
}


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:.3f}"


def explain_decision(
    sku: str,
    description: str,
    date: Any,
    decision: ReorderDecision,
    forecast_7d: float,
    reorder_probability: float | None,
    shap_top: dict[str, list[dict]] | None,
    mean_daily_demand: float | None = None,
) -> dict[str, Any]:
    """Assemble a complete, checkable explanation of one recommendation."""
    if not decision.reconciles():
        raise AssertionError(
            "Decision components do not sum to the recommended quantity; "
            "the explanation would be wrong."
        )

    # --- why this forecast -------------------------------------------
    demand_reasons: list[str] = []
    if shap_top:
        for item in shap_top.get("positive", [])[:3]:
            demand_reasons.append(
                f"{label_for(item['feature'])} is {_fmt(item['feature_value'])}, "
                f"pushing the forecast up by {_fmt(item['shap_value'])} units"
            )
        for item in shap_top.get("negative", [])[:2]:
            demand_reasons.append(
                f"{label_for(item['feature'])} is {_fmt(item['feature_value'])}, "
                f"pulling the forecast down by {_fmt(abs(item['shap_value']))} units"
            )

    # --- why this quantity -------------------------------------------
    qty_reasons: list[str] = []
    c = decision.components()
    if c["expected demand over lead time"] > 0:
        qty_reasons.append(
            f"Expected demand over the {decision.lead_time_days}-day lead time "
            f"plus the review period is {_fmt(decision.expected_demand)} units."
        )
    if c["safety stock"] > 0:
        qty_reasons.append(
            f"A safety buffer of {_fmt(decision.safety_stock)} units covers "
            f"forecast error at the target service level."
        )
    qty_reasons.append(
        f"{_fmt(decision.inventory_position)} units are already available or "
        f"on order, which reduces the requirement."
    )
    if c["clip at zero (never order negative)"] > 0:
        qty_reasons.append(
            "Stock already exceeds the requirement, so the raw calculation was "
            "negative and has been clipped to zero."
        )
    if c["supplier minimum order quantity"] > 0:
        qty_reasons.append(
            f"The supplier's minimum order quantity of {decision.moq} units "
            f"raises the order by {_fmt(c['supplier minimum order quantity'])} units."
        )
    if c["pack-size rounding"] > 0:
        qty_reasons.append(
            f"Rounding up to whole packs of {decision.pack_size} adds "
            f"{_fmt(c['pack-size rounding'])} units."
        )
    if c["supplier capacity cap"] < -1e-9:
        qty_reasons.append(
            f"The supplier can ship at most {decision.supplier_capacity} units, "
            f"which cuts the order by {_fmt(abs(c['supplier capacity cap']))} units."
        )
    if decision.coverage_gap > 0.5:
        qty_reasons.append(
            f"WARNING: capacity leaves a shortfall of "
            f"{_fmt(decision.coverage_gap)} units against the requirement."
        )

    risk = "UNKNOWN"
    if reorder_probability is not None:
        if reorder_probability >= 0.70:
            risk = "HIGH"
        elif reorder_probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

    return {
        "sku": sku,
        "description": description,
        "date": str(date),
        "predicted_7d_demand": round(float(forecast_7d), 1),
        "current_stock": round(decision.inventory_position, 1),
        "lead_time_days": decision.lead_time_days,
        "stockout_risk": risk,
        "reorder_probability": (
            round(float(reorder_probability), 4) if reorder_probability is not None else None
        ),
        "reorder": decision.reorder,
        "recommended_qty": decision.final_qty,
        "order_value": round(decision.order_value, 2),
        "days_of_cover_after": (
            round(decision.days_of_cover_after, 1)
            if np.isfinite(decision.days_of_cover_after)
            else None
        ),
        "demand_reasons": demand_reasons,
        "quantity_reasons": qty_reasons,
        "decomposition": {k: round(v, 2) for k, v in c.items()},
        "decomposition_total": decision.final_qty,
        "reconciles": decision.reconciles(),
    }


def render_explanation(exp: dict[str, Any]) -> str:
    """Format an explanation as the plain-text block shown in the demo."""
    lines = [
        f"Product            : {exp['sku']}  {exp['description'][:48]}",
        f"Decision date      : {exp['date'][:10]}",
        f"Current stock      : {exp['current_stock']:,.0f} units",
        f"Predicted 7-day demand : {exp['predicted_7d_demand']:,.0f} units",
        f"Supplier lead time : {exp['lead_time_days']} days",
        f"Stockout risk      : {exp['stockout_risk']}"
        + (
            f"  (p = {exp['reorder_probability']:.2f})"
            if exp["reorder_probability"] is not None
            else ""
        ),
        f"Recommendation     : {'REORDER' if exp['reorder'] else 'DO NOT REORDER'}",
        f"Recommended order  : {exp['recommended_qty']:,} units",
        f"Order value        : {exp['order_value']:,.2f}  (SIMULATED cost basis)",
        "",
        "Why is demand forecast at this level?",
    ]
    if exp["demand_reasons"]:
        for i, r in enumerate(exp["demand_reasons"], 1):
            lines.append(f"  {i}. {r.capitalize()}")
    else:
        lines.append("  (no attribution available for this row)")

    lines += ["", "Why this order quantity?"]
    for i, r in enumerate(exp["quantity_reasons"], 1):
        lines.append(f"  {i}. {r}")

    lines += ["", "Quantity decomposition (sums to the recommendation):"]
    for k, v in exp["decomposition"].items():
        lines.append(f"  {v:>+12,.2f}   {k}")
    lines.append(f"  {'-' * 12}")
    lines.append(f"  {exp['decomposition_total']:>12,}   FINAL ORDER QUANTITY")
    return "\n".join(lines)
