"""Replay the test period under different replenishment policies and cost them.

Every policy sees the SAME real demand, the SAME simulated supplier terms and
the SAME opening stock. The only thing that varies is how the order quantity
is decided. That makes the cost difference attributable to the decision rule
rather than to luck.

The policy ladder (each rung adds one idea):

    P0  moving average, no buffer        -- what a spreadsheet does today
    P1  classical reorder point          -- textbook z*sigma*sqrt(L) safety stock
    P2  ML point forecast, no buffer     -- does a better forecast alone help?
    P3  ML forecast + safety stock       -- forecast plus classical buffer
    P4  ML forecast + quantile buffer    -- buffer learned from the demand tail,
                                            plus MOQ / pack / capacity constraints

Costs are SIMULATED (sim/supply_config.yaml): holding, shortage and ordering
costs do not exist in the source dataset. Comparisons between policies are
therefore meaningful; the absolute currency amounts are not a claim about the
real retailer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.config import load_supply_config


@dataclass
class PolicySpec:
    key: str
    label: str
    forecast: str       # which demand estimate drives the order
    buffer: str         # how safety stock is sized
    constraints: bool   # whether MOQ / pack / capacity are enforced
    description: str


POLICIES: dict[str, PolicySpec] = {
    "P0": PolicySpec(
        "P0", "P0: Moving average, no safety stock",
        forecast="rolling_mean_7", buffer="none", constraints=False,
        description="Order up to the 7-day moving average extended over the "
                    "lead time. No buffer, no supplier constraints. This is "
                    "the 'what the business does today' baseline.",
    ),
    "P1": PolicySpec(
        "P1", "P1: Classical reorder point (z*sigma*sqrt(L))",
        forecast="rolling_mean_28", buffer="classical", constraints=True,
        description="Textbook reorder policy: 28-day mean demand over the "
                    "lead time plus a Normal-theory safety stock at the "
                    "target service level, with supplier constraints applied.",
    ),
    "P2": PolicySpec(
        "P2", "P2: ML point forecast, no safety stock",
        forecast="ml_point", buffer="none", constraints=True,
        description="Replaces the moving average with the best ML forecast "
                    "but adds no buffer. Isolates the value of forecast "
                    "accuracy on its own.",
    ),
    "P3": PolicySpec(
        "P3", "P3: ML forecast + classical safety stock",
        forecast="ml_point", buffer="classical", constraints=True,
        description="ML forecast plus the same Normal-theory buffer as P1.",
    ),
    "P4": PolicySpec(
        "P4", "P4: ML forecast + quantile safety stock + constraints",
        forecast="ml_point", buffer="quantile", constraints=True,
        description="ML forecast with a buffer taken from the model's own "
                    "predicted upper quantile, so the buffer reflects the "
                    "learned demand tail rather than a Normal assumption. "
                    "This is the policy the dashboard recommends.",
    ),
}


def _daily_holding_cost(unit_cost: float, annual_rate: float) -> float:
    return unit_cost * annual_rate / 365.0


def simulate_policy(
    df: pd.DataFrame,
    spec: PolicySpec,
    service_level: float = 0.95,
    review_period: int = 7,
    horizon: int = 7,
) -> dict[str, float]:
    """Replay one policy over the test period for all SKUs.

    `df` must be the test-set prediction table: one row per SKU-day, sorted,
    carrying real `demand`, the ML forecasts, and the simulated supplier terms.
    """
    scfg = load_supply_config()
    costs = scfg["costs"]
    holding_rate = costs["holding_cost_rate_annual"]
    shortage_mult = costs["shortage_cost_multiplier"]
    order_cost = costs["ordering_cost_per_order"]
    z = float(norm.ppf(service_level))

    total_demand = 0.0
    total_unmet = 0.0
    total_holding = 0.0
    total_purchase = 0.0
    total_shortage = 0.0
    total_ordering = 0.0
    n_orders = 0
    n_stockout_days = 0
    n_days = 0
    ending_levels: list[float] = []

    for sku, grp in df.groupby("stock_code", sort=True):
        grp = grp.sort_values("date")
        n = len(grp)
        if n == 0:
            continue

        demand = grp["demand"].to_numpy(dtype=float)
        lead = int(grp["lead_time_days"].iloc[0])
        moq = int(grp["moq"].iloc[0])
        pack = max(1, int(grp["pack_size"].iloc[0]))
        unit_cost = float(grp["unit_cost"].iloc[0])
        capacity = (
            float(grp["supplier_capacity"].iloc[0])
            if "supplier_capacity" in grp.columns
            else math.inf
        )
        hold_per_unit_day = _daily_holding_cost(unit_cost, holding_rate)

        # Forecast of demand over the LEAD TIME. The ML target is a 7-day
        # total, so it is rescaled to the lead time by lead/horizon.
        if spec.forecast == "ml_point":
            f7 = grp["pred_best_regression"].to_numpy(dtype=float)
            fc = np.clip(f7, 0, None) * (lead + review_period) / horizon
        else:
            daily = grp[spec.forecast].to_numpy(dtype=float)
            fc = np.clip(daily, 0, None) * (lead + review_period)

        # Safety stock.
        if spec.buffer == "classical":
            sigma = np.nan_to_num(grp["rolling_std_28"].to_numpy(dtype=float))
            safety = z * sigma * math.sqrt(lead + review_period)
        elif spec.buffer == "quantile":
            q_hi = np.clip(grp["pred_q95"].to_numpy(dtype=float), 0, None)
            q_md = np.clip(grp["pred_q50"].to_numpy(dtype=float), 0, None)
            safety = np.maximum(0.0, q_hi - q_md) * (lead + review_period) / horizon
        else:
            safety = np.zeros(n)

        # Opening stock is the simulated position on the first test day, so
        # every policy starts from an identical inventory.
        stock = float(grp["current_stock"].iloc[0])
        pipeline = np.zeros(n + lead + 1)

        for t in range(n):
            arrivals = float(pipeline[t])
            available = stock + arrivals

            d = demand[t]
            consumed = min(d, available)
            unmet = d - consumed
            stock = available - consumed

            total_demand += d
            total_unmet += unmet
            n_days += 1
            if unmet > 0:
                n_stockout_days += 1
            total_holding += stock * hold_per_unit_day
            total_shortage += unmet * unit_cost * shortage_mult
            ending_levels.append(stock)

            # Weekly review.
            if t % review_period == 0:
                on_order = float(pipeline[t + 1 :].sum())
                position = stock + on_order
                need = max(0.0, fc[t] + safety[t] - position)
                qty = need
                if spec.constraints and qty > 0:
                    qty = max(qty, moq)
                    qty = math.ceil(qty / pack) * pack
                    qty = min(qty, capacity)
                qty = float(int(round(qty)))
                if qty > 0:
                    pipeline[min(t + lead, n + lead)] += qty
                    total_purchase += qty * unit_cost
                    total_ordering += order_cost
                    n_orders += 1

    fill_rate = 1.0 - (total_unmet / total_demand) if total_demand else float("nan")
    total_cost = total_holding + total_purchase + total_shortage + total_ordering
    mean_inv = float(np.mean(ending_levels)) if ending_levels else float("nan")
    turnover = (total_demand / mean_inv) if mean_inv else float("nan")

    return {
        "policy": spec.key,
        "policy_label": spec.label,
        "forecast_source": spec.forecast,
        "safety_stock_method": spec.buffer,
        "supplier_constraints_applied": spec.constraints,
        "total_demand_units": round(total_demand, 1),
        "unmet_demand_units": round(total_unmet, 1),
        "fill_rate": round(fill_rate, 6),
        "service_level_days": round(1.0 - n_stockout_days / n_days, 6) if n_days else float("nan"),
        "stockout_days": int(n_stockout_days),
        "n_orders": int(n_orders),
        "holding_cost": round(total_holding, 2),
        "purchase_cost": round(total_purchase, 2),
        "shortage_cost": round(total_shortage, 2),
        "ordering_cost": round(total_ordering, 2),
        "total_cost": round(total_cost, 2),
        "mean_inventory_units": round(mean_inv, 2),
        "inventory_turnover": round(turnover, 3),
    }


def run_policy_ladder(
    df: pd.DataFrame, service_level: float = 0.95
) -> pd.DataFrame:
    """Run every rung of the ladder and return one row per policy."""
    rows = []
    for key in ["P0", "P1", "P2", "P3", "P4"]:
        spec = POLICIES[key]
        print(f"    {spec.label} ...", end="", flush=True)
        res = simulate_policy(df, spec, service_level=service_level)
        print(
            f" fill={res['fill_rate']:.3%}  cost={res['total_cost']:,.0f}  "
            f"orders={res['n_orders']}"
        )
        rows.append(res)
    out = pd.DataFrame(rows)

    base = out.loc[out["policy"] == "P0", "total_cost"].iloc[0]
    out["cost_vs_P0_pct"] = ((out["total_cost"] - base) / base * 100).round(2)
    base_fill = out.loc[out["policy"] == "P0", "fill_rate"].iloc[0]
    out["fill_rate_vs_P0_pp"] = ((out["fill_rate"] - base_fill) * 100).round(3)
    return out
