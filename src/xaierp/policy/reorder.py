"""Turn a demand forecast into a purchase-order quantity.

The decision is deliberately built as a SUM OF NAMED COMPONENTS:

    final_qty = demand_component
              + safety_component
              - inventory_offset
              + clip_at_zero_adjustment
              + moq_adjustment
              + pack_rounding_adjustment
              + capacity_adjustment

Every component is returned alongside the answer, and the components are
guaranteed to add up to the final quantity exactly. That property is what
makes the "why this number?" explanation in explain/decision_explanation.py
an actual account of the arithmetic rather than a plausible-sounding story,
and tests/test_decision_explanation.py fails if the identity ever breaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class ReorderDecision:
    """One procurement recommendation, with its arithmetic laid out."""

    reorder: bool
    final_qty: int

    # Inputs echoed back so the explanation is self-contained.
    expected_demand: float
    safety_stock: float
    inventory_position: float
    lead_time_days: int
    moq: int
    pack_size: int
    supplier_capacity: int
    unit_cost: float

    # Components -- these sum to final_qty.
    demand_component: float
    safety_component: float
    inventory_offset: float
    clip_adjustment: float
    moq_adjustment: float
    pack_adjustment: float
    capacity_adjustment: float

    # Derived economics.
    order_value: float
    days_of_cover_after: float
    coverage_gap: float

    def components(self) -> dict[str, float]:
        return {
            "expected demand over lead time": self.demand_component,
            "safety stock": self.safety_component,
            "less: stock already available / on order": self.inventory_offset,
            "clip at zero (never order negative)": self.clip_adjustment,
            "supplier minimum order quantity": self.moq_adjustment,
            "pack-size rounding": self.pack_adjustment,
            "supplier capacity cap": self.capacity_adjustment,
        }

    def reconciles(self, tol: float = 1e-6) -> bool:
        return abs(sum(self.components().values()) - self.final_qty) <= tol

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safety_stock_from_sigma(
    sigma_daily: float, lead_time_days: int, service_level: float = 0.95
) -> float:
    """Classical safety stock: z * sigma * sqrt(L).

    This is the textbook formula and is used by the non-ML policies in the
    comparison ladder. It assumes demand is i.i.d. Normal over the lead time,
    which intermittent retail demand is not -- which is exactly why the
    quantile-based alternative below usually does better.
    """
    if sigma_daily <= 0 or lead_time_days <= 0:
        return 0.0
    z = float(norm.ppf(service_level))
    return float(z * sigma_daily * math.sqrt(lead_time_days))


def safety_stock_from_quantiles(
    q_high: float, q_median: float, scale: float = 1.0
) -> float:
    """Safety stock as the gap between a high quantile and the median forecast.

    Makes no distributional assumption: the model has already learned the
    shape of the upper tail. `scale` adapts a horizon-length quantile to a
    lead time of a different length.
    """
    return float(max(0.0, (q_high - q_median) * scale))


def decide_reorder(
    expected_demand_leadtime: float,
    safety_stock: float,
    inventory_position: float,
    lead_time_days: int,
    moq: int = 0,
    pack_size: int = 1,
    supplier_capacity: int | None = None,
    unit_cost: float = 0.0,
    mean_daily_demand: float | None = None,
) -> ReorderDecision:
    """Compute one purchase recommendation.

    Parameters
    ----------
    expected_demand_leadtime : forecast demand over the replenishment lead time.
    safety_stock : buffer against forecast error.
    inventory_position : stock on hand plus stock already on order.
    """
    pack_size = max(1, int(pack_size))
    moq = max(0, int(moq))

    demand_component = float(expected_demand_leadtime)
    safety_component = float(safety_stock)
    inventory_offset = -float(inventory_position)

    raw_need = demand_component + safety_component + inventory_offset

    # 1. Never order a negative quantity.
    need = max(0.0, raw_need)
    clip_adjustment = need - raw_need

    # 2. If anything is needed at all, the supplier's minimum applies.
    if need > 0:
        after_moq = max(need, float(moq))
    else:
        after_moq = 0.0
    moq_adjustment = after_moq - need

    # 3. Orders must be a whole number of packs.
    if after_moq > 0:
        after_pack = float(math.ceil(after_moq / pack_size) * pack_size)
    else:
        after_pack = 0.0
    pack_adjustment = after_pack - after_moq

    # 4. The supplier cannot ship more than its capacity.
    #    The cap is rounded DOWN to a whole pack: you cannot buy a part pack,
    #    so clipping straight to the capacity figure would silently break the
    #    pack-multiple guarantee established in step 3.
    final = after_pack
    if supplier_capacity is not None and after_pack > supplier_capacity:
        final = float(math.floor(supplier_capacity / pack_size) * pack_size)
    capacity_adjustment = final - after_pack

    final_qty = int(round(final))

    # Unmet requirement after the capacity cap -- surfaced so a buyer can see
    # when a single supplier physically cannot cover the gap.
    coverage_gap = max(0.0, need - final_qty)

    projected_stock = inventory_position + final_qty - demand_component
    if mean_daily_demand and mean_daily_demand > 0:
        days_cover = projected_stock / mean_daily_demand
    else:
        days_cover = float("nan")

    return ReorderDecision(
        reorder=final_qty > 0,
        final_qty=final_qty,
        expected_demand=demand_component,
        safety_stock=safety_component,
        inventory_position=float(inventory_position),
        lead_time_days=int(lead_time_days),
        moq=moq,
        pack_size=pack_size,
        supplier_capacity=int(supplier_capacity) if supplier_capacity is not None else -1,
        unit_cost=float(unit_cost),
        demand_component=demand_component,
        safety_component=safety_component,
        inventory_offset=inventory_offset,
        clip_adjustment=clip_adjustment,
        moq_adjustment=moq_adjustment,
        pack_adjustment=pack_adjustment,
        # Absorb the integer rounding of the final quantity into the capacity
        # term so the component sum matches final_qty exactly.
        capacity_adjustment=capacity_adjustment + (final_qty - final),
        order_value=float(final_qty * unit_cost),
        days_of_cover_after=float(days_cover),
        coverage_gap=float(coverage_gap),
    )


def decide_reorder_frame(
    df: pd.DataFrame,
    demand_col: str,
    safety_col: str | None = None,
    position_col: str = "current_stock",
    service_level: float = 0.95,
) -> pd.DataFrame:
    """Vectorised version of `decide_reorder` for a whole table of SKU-days.

    Produces the same components as the scalar function, so a row picked out
    of this frame can be re-explained identically by decide_reorder.
    """
    out = df.copy()
    lead = out["lead_time_days"].to_numpy(dtype=float)
    pack = np.maximum(1, out["pack_size"].to_numpy(dtype=float))
    moq = np.maximum(0, out["moq"].to_numpy(dtype=float))

    demand = out[demand_col].to_numpy(dtype=float)
    if safety_col is not None:
        safety = out[safety_col].to_numpy(dtype=float)
    else:
        sigma = out["rolling_std_28"].to_numpy(dtype=float)
        safety = float(norm.ppf(service_level)) * sigma * np.sqrt(lead)
    safety = np.nan_to_num(safety, nan=0.0, posinf=0.0, neginf=0.0)

    position = out[position_col].to_numpy(dtype=float)

    raw_need = demand + safety - position
    need = np.maximum(0.0, raw_need)
    clip_adj = need - raw_need

    after_moq = np.where(need > 0, np.maximum(need, moq), 0.0)
    moq_adj = after_moq - need

    after_pack = np.where(after_moq > 0, np.ceil(after_moq / pack) * pack, 0.0)
    pack_adj = after_pack - after_moq

    if "supplier_capacity" in out.columns:
        cap = out["supplier_capacity"].to_numpy(dtype=float)
        # Round the cap down to a whole pack, as in the scalar version.
        cap_packs = np.floor(cap / pack) * pack
        final = np.minimum(after_pack, cap_packs)
    else:
        final = after_pack
    cap_adj = final - after_pack

    final_int = np.round(final)

    out["rec_expected_demand"] = demand
    out["rec_safety_stock"] = safety
    out["rec_inventory_offset"] = -position
    out["rec_clip_adjustment"] = clip_adj
    out["rec_moq_adjustment"] = moq_adj
    out["rec_pack_adjustment"] = pack_adj
    out["rec_capacity_adjustment"] = cap_adj + (final_int - final)
    out["rec_qty"] = final_int.astype(int)
    out["rec_reorder"] = out["rec_qty"] > 0
    out["rec_order_value"] = out["rec_qty"] * out["unit_cost"]
    return out
