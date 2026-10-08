"""Counterfactual ("what if") analysis on a single procurement decision.

This is what makes the system decision SUPPORT rather than a decision oracle:
a buyer can ask "what if I negotiate the lead time down to 4 days?" and see
the recommendation move, instead of being handed one number to accept.

Each scenario re-runs the real policy function with one input changed. No
scenario result is interpolated or approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..policy.reorder import ReorderDecision, decide_reorder, safety_stock_from_quantiles


@dataclass
class Scenario:
    name: str
    lead_time_days: int | None = None
    current_stock: float | None = None
    service_level_quantile_gap: float | None = None
    moq: int | None = None
    demand_multiplier: float | None = None


def run_scenario(
    base: dict[str, Any],
    scenario: Scenario,
) -> ReorderDecision:
    """Re-decide with one factor overridden.

    `base` carries the unmodified inputs:
        demand_7d, q50, q95, current_stock, lead_time_days, review_period,
        moq, pack_size, supplier_capacity, unit_cost, mean_daily_demand
    """
    lead = scenario.lead_time_days if scenario.lead_time_days is not None else base["lead_time_days"]
    stock = scenario.current_stock if scenario.current_stock is not None else base["current_stock"]
    moq = scenario.moq if scenario.moq is not None else base["moq"]

    horizon = base["horizon"]
    review = base["review_period"]
    scale = (lead + review) / horizon

    demand_7d = base["demand_7d"]
    if scenario.demand_multiplier is not None:
        demand_7d = demand_7d * scenario.demand_multiplier

    gap = (
        scenario.service_level_quantile_gap
        if scenario.service_level_quantile_gap is not None
        else max(0.0, base["q95"] - base["q50"])
    )

    return decide_reorder(
        expected_demand_leadtime=max(0.0, demand_7d) * scale,
        safety_stock=safety_stock_from_quantiles(gap, 0.0, scale=scale),
        inventory_position=stock,
        lead_time_days=int(lead),
        moq=int(moq),
        pack_size=int(base["pack_size"]),
        supplier_capacity=int(base["supplier_capacity"]),
        unit_cost=float(base["unit_cost"]),
        mean_daily_demand=base.get("mean_daily_demand"),
    )


def default_scenarios(base: dict[str, Any]) -> list[Scenario]:
    """A standard panel of counterfactuals around the current situation.

    Stock scenarios are expressed as multiples of expected lead-time demand
    rather than as multiples of the CURRENT stock. Scaling the current value
    is degenerate exactly when the question matters most: for a stocked-out
    item, 0 x 0.5 and 0 x 1.5 are both 0, so the panel would show no movement
    on the row a buyer most needs to interrogate.
    """
    lead = int(base["lead_time_days"])
    review = base["review_period"]
    horizon = base["horizon"]
    # Expected demand over the cover window, on the same scale as the policy.
    cover_demand = max(0.0, base["demand_7d"]) * (lead + review) / horizon

    return [
        Scenario("Current situation"),
        Scenario(f"Lead time halved ({lead} -> {max(1, lead // 2)} days)",
                 lead_time_days=max(1, lead // 2)),
        Scenario(f"Lead time doubled ({lead} -> {lead * 2} days)",
                 lead_time_days=lead * 2),
        Scenario("Stock fully depleted (0 units)", current_stock=0.0),
        Scenario("Stock = 50% of lead-time demand",
                 current_stock=round(0.5 * cover_demand, 1)),
        Scenario("Stock = 150% of lead-time demand",
                 current_stock=round(1.5 * cover_demand, 1)),
        Scenario("No safety buffer (service level dropped)",
                 service_level_quantile_gap=0.0),
        Scenario("Demand surge +30%", demand_multiplier=1.3),
        Scenario("Supplier raises MOQ to 288", moq=288),
    ]


def run_whatif_table(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the default panel and return one summary row per scenario."""
    rows = []
    baseline_qty = None
    for sc in default_scenarios(base):
        d = run_scenario(base, sc)
        if baseline_qty is None:
            baseline_qty = d.final_qty
        rows.append(
            {
                "scenario": sc.name,
                "lead_time_days": d.lead_time_days,
                "inventory_position": round(d.inventory_position, 1),
                "expected_demand": round(d.expected_demand, 1),
                "safety_stock": round(d.safety_stock, 1),
                "moq": d.moq,
                "recommended_qty": d.final_qty,
                "change_vs_current": d.final_qty - baseline_qty,
                "order_value": round(d.order_value, 2),
                "reorder": d.reorder,
            }
        )
    return rows
