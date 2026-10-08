"""Reconstruct a SIMULATED daily inventory position for every modelled SKU.

===========================================================================
THE STOCK NUMBERS PRODUCED HERE ARE SYNTHETIC.
===========================================================================
Online Retail II records what was sold, never what was on the shelf. To ask
"should we reorder?" we need a stock level, so one is simulated.

How it is simulated matters for honesty:

  * Demand is REAL -- taken straight from the cleaned daily panel.
  * The replenishment policy driving the trajectory is a deliberately plain
    periodic-review rule using only a TRAILING mean of past demand. It never
    looks forward. This is the "P0 / existing practice" baseline that the ML
    system is later compared against; if it were already optimal there would
    be nothing to improve.
  * Because the baseline policy is naive, it stocks out sometimes. Those
    stockouts are what create the classification labels.

Conservation identity, asserted in tests/test_inventory_conservation.py:

    ending_stock = beginning_stock + arrivals - consumed
    consumed     = min(demand, beginning_stock + arrivals)
    unmet_demand = demand - consumed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xaierp.utils.config import ensure_dir, load_config, load_supply_config, resolve  # noqa: E402

REVIEW_PERIOD_DAYS = 7  # the baseline buyer reviews stock weekly


def simulate_sku(
    demand: np.ndarray,
    lead_time: int,
    initial_stock: float,
    warmup: int = 28,
    review_period: int = REVIEW_PERIOD_DAYS,
    safety_factor: float = 0.5,
) -> dict[str, np.ndarray]:
    """Run the baseline inventory trajectory for one SKU.

    Parameters
    ----------
    demand : real daily demand for this SKU, in trading-day order.
    lead_time : SIMULATED supplier lead time, in trading days.
    initial_stock : SIMULATED opening position.
    warmup : days of history the naive forecast needs before it will order.
    safety_factor : multiples of the trailing std held as buffer. Kept small
        on purpose -- this is the weak baseline, not a tuned policy.

    Returns arrays aligned to `demand`, one entry per day.
    """
    n = len(demand)
    beginning = np.zeros(n)
    arrivals = np.zeros(n)
    consumed = np.zeros(n)
    ending = np.zeros(n)
    unmet = np.zeros(n)
    on_order = np.zeros(n)
    orders_placed = np.zeros(n)

    pipeline = np.zeros(n + lead_time + 1)  # arrivals scheduled by day index
    stock = float(initial_stock)

    for t in range(n):
        beginning[t] = stock
        arr = float(pipeline[t])
        arrivals[t] = arr
        available = stock + arr

        d = float(demand[t])
        c = min(d, available)
        consumed[t] = c
        unmet[t] = d - c
        stock = available - c
        ending[t] = stock
        on_order[t] = float(pipeline[t + 1 :].sum())

        # Weekly review, using only demand strictly before today.
        if t >= warmup and t % review_period == 0:
            hist = demand[max(0, t - warmup) : t]
            mu = float(hist.mean())
            sd = float(hist.std(ddof=0))
            cover_days = lead_time + review_period
            target = mu * cover_days + safety_factor * sd * np.sqrt(cover_days)
            position = stock + float(pipeline[t + 1 :].sum())
            qty = max(0.0, target - position)
            if qty > 0:
                orders_placed[t] = qty
                pipeline[min(t + lead_time, n + lead_time)] += qty

    return {
        "beginning_stock": beginning,
        "arrivals": arrivals,
        "consumed": consumed,
        "ending_stock": ending,
        "unmet_demand": unmet,
        "on_order": on_order,
        "orders_placed": orders_placed,
    }


def build_inventory_state(panel: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Simulate stock for every SKU and attach the classification labels."""
    scfg = load_supply_config()
    rng = np.random.default_rng(scfg["seed"] + 1)

    preferred = (
        suppliers.loc[suppliers["is_preferred"]]
        .set_index("stock_code")[
            ["lead_time_days", "moq", "pack_size", "unit_cost", "capacity_units",
             "supplier_id", "reliability"]
        ]
    )

    inv_cfg = scfg["inventory"]
    cover_spec = inv_cfg["initial_days_of_cover"]

    panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)
    out_parts = []

    for sku, grp in panel.groupby("stock_code", sort=True):
        grp = grp.sort_values("date")
        demand = grp["demand"].to_numpy(dtype=float)
        if sku not in preferred.index:
            continue
        info = preferred.loc[sku]
        lead_time = int(info["lead_time_days"])

        early_mean = float(demand[: min(28, len(demand))].mean())
        days_cover = float(rng.uniform(cover_spec["low"], cover_spec["high"]))
        initial_stock = max(1.0, early_mean * days_cover)

        sim = simulate_sku(demand, lead_time, initial_stock)

        part = grp.copy()
        for k, v in sim.items():
            part[k] = v
        part["lead_time_days"] = lead_time
        part["moq"] = int(info["moq"])
        part["pack_size"] = int(info["pack_size"])
        part["unit_cost"] = float(info["unit_cost"])
        part["supplier_capacity"] = int(info["capacity_units"])
        part["supplier_id"] = info["supplier_id"]
        part["supplier_reliability"] = float(info["reliability"])

        # ---- classification label -------------------------------------
        # At the start of day t the buyer can see beginning_stock and the
        # supplier's lead time. The question is whether demand over the next
        # `lead_time` days will exceed what is available before any new order
        # could land. The ANSWER uses future demand -- that is legitimate,
        # it is the label. The INPUTS do not.
        L = lead_time
        fwd = (
            pd.Series(demand[::-1])
            .rolling(window=L, min_periods=L)
            .sum()
            .iloc[::-1]
            .to_numpy()
        )
        part["lead_time_demand_actual"] = fwd
        available_now = sim["beginning_stock"] + sim["arrivals"]
        tol = float(inv_cfg["stockout_risk_tolerance"])
        part["reorder_flag"] = (
            fwd > available_now * (1.0 + tol)
        ).astype("float64")
        part.loc[np.isnan(fwd), "reorder_flag"] = np.nan
        part["stockout_risk"] = part["reorder_flag"]

        # Decision-time inventory features (all knowable at the start of t).
        part["current_stock"] = sim["beginning_stock"] + sim["arrivals"]
        part["inventory_position"] = part["current_stock"] + sim["on_order"]
        trailing_mean = (
            pd.Series(demand).shift(1).rolling(28, min_periods=28).mean().to_numpy()
        )
        part["days_of_cover"] = np.where(
            trailing_mean > 0, part["current_stock"] / trailing_mean, np.nan
        )
        part["stock_vs_leadtime_demand"] = np.where(
            trailing_mean > 0,
            part["current_stock"] / (trailing_mean * L),
            np.nan,
        )

        out_parts.append(part)

    result = pd.concat(out_parts, ignore_index=True)
    result["is_simulated_supply"] = True
    return result


def main() -> int:
    cfg = load_config()
    panel_path = resolve(cfg["paths"]["processed"]) / "daily_demand_panel.parquet"
    sup_path = resolve(cfg["paths"]["simulated"]) / "suppliers.parquet"
    for p, hint in [
        (panel_path, "scripts/prepare_data.py"),
        (sup_path, "sim/generate_supply.py"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run {hint} first.")

    panel = pd.read_parquet(panel_path)
    suppliers = pd.read_parquet(sup_path)

    state = build_inventory_state(panel, suppliers)

    out = resolve(cfg["paths"]["simulated"]) / "inventory_state.parquet"
    ensure_dir(out)
    state.to_parquet(out, index=False)

    lab = state["reorder_flag"].dropna()
    service = 1.0 - (state["unmet_demand"].sum() / state["demand"].sum())
    print(f"SIMULATED inventory state: {len(state):,} SKU-days")
    print(f"  baseline fill rate      : {service:.3%}")
    print(f"  days with unmet demand  : {(state['unmet_demand'] > 0).mean():.2%}")
    print(f"  reorder_flag positives  : {lab.mean():.2%} of {len(lab):,} labelled rows")
    print(f"  mean ending stock       : {state['ending_stock'].mean():.1f} units")
    print(f"  saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
