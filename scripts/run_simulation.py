"""Stage 4: policy ladder, ablation and procurement recommendations.

Writes:
    reports/policy_comparison.csv
    reports/ablation_study.csv
    data/results/procurement_recommendations.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.policy.reorder import decide_reorder_frame  # noqa: E402
from xaierp.simulate.procurement import POLICIES, run_policy_ladder, simulate_policy  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, load_supply_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    service_level = cfg["quantiles"]["service_level"]

    pred_path = resolve(cfg["paths"]["forecasts"]) / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} missing. Run scripts/train_models.py first."
        )
    df = pd.read_parquet(pred_path)

    # Supplier capacity is needed for the constrained policies.
    sup = pd.read_parquet(resolve(cfg["paths"]["simulated"]) / "suppliers.parquet")
    cap = (
        sup.loc[sup["is_preferred"]]
        .set_index("stock_code")["capacity_units"]
    )
    df["supplier_capacity"] = df["stock_code"].map(cap)

    print("=" * 70)
    print("POLICY LADDER  (test period "
          f"{df['date'].min().date()} -> {df['date'].max().date()}, "
          f"{df['stock_code'].nunique()} SKUs)")
    print("=" * 70)
    ladder = run_policy_ladder(df, service_level=service_level)
    out = resolve(cfg["paths"]["reports"]) / "policy_comparison.csv"
    ensure_dir(out)
    ladder.to_csv(out, index=False)
    print(f"  saved -> {out.relative_to(ROOT)}")

    # ---------------------------------------------------------------
    # Ablation: add one capability at a time, measure the effect.
    # Rungs A..E deliberately mirror the ladder so the two tables agree.
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    ablation_map = {
        "A: historical average only": "P0",
        "B: + classical safety stock": "P1",
        "C: ML point forecast": "P2",
        "D: ML forecast + safety stock": "P3",
        "E: ML forecast + quantile buffer + supplier constraints": "P4",
    }
    rows = []
    prev_cost = None
    for label, pkey in ablation_map.items():
        r = ladder.loc[ladder["policy"] == pkey].iloc[0].to_dict()
        delta = None if prev_cost is None else round(r["total_cost"] - prev_cost, 2)
        rows.append(
            {
                "stage": label,
                "policy": pkey,
                "fill_rate": r["fill_rate"],
                "stockout_days": r["stockout_days"],
                "mean_inventory_units": r["mean_inventory_units"],
                "total_cost": r["total_cost"],
                "cost_change_vs_previous_stage": delta,
                "cost_vs_P0_pct": r["cost_vs_P0_pct"],
            }
        )
        prev_cost = r["total_cost"]
        print(f"  {label:<58} fill={r['fill_rate']:.3%}  cost={r['total_cost']:,.0f}")
    abl = pd.DataFrame(rows)
    out = resolve(cfg["paths"]["reports"]) / "ablation_study.csv"
    abl.to_csv(out, index=False)
    print(f"  saved -> {out.relative_to(ROOT)}")

    # ---------------------------------------------------------------
    # Per-SKU-day procurement recommendations under the winning policy.
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PROCUREMENT RECOMMENDATIONS (policy P4)")
    print("=" * 70)
    horizon = cfg["target"]["horizon_days"]
    rec = df.copy()
    lead = rec["lead_time_days"].to_numpy(dtype=float)
    scale = (lead + 7.0) / horizon
    rec["demand_over_leadtime"] = np.clip(rec["pred_best_regression"], 0, None) * scale
    rec["quantile_safety_stock"] = (
        np.maximum(0.0, np.clip(rec["pred_q95"], 0, None) - np.clip(rec["pred_q50"], 0, None))
        * scale
    )
    rec = decide_reorder_frame(
        rec,
        demand_col="demand_over_leadtime",
        safety_col="quantile_safety_stock",
        position_col="inventory_position",
        service_level=service_level,
    )

    # The component identity must hold for every row.
    comp_cols = [
        "rec_expected_demand", "rec_safety_stock", "rec_inventory_offset",
        "rec_clip_adjustment", "rec_moq_adjustment", "rec_pack_adjustment",
        "rec_capacity_adjustment",
    ]
    resid = (rec[comp_cols].sum(axis=1) - rec["rec_qty"]).abs()
    max_resid = float(resid.max())
    print(f"  decomposition max residual: {max_resid:.10f}  "
          f"({'OK' if max_resid < 1e-6 else 'FAILED'})")
    if max_resid >= 1e-6:
        raise AssertionError(
            f"Reorder decomposition does not reconcile (max residual {max_resid})."
        )

    print(f"  rows                : {len(rec):,}")
    print(f"  recommended reorders: {rec['rec_reorder'].mean():.2%}")
    print(f"  mean order qty      : {rec.loc[rec['rec_reorder'], 'rec_qty'].mean():.1f} units")
    print(f"  total order value   : {rec['rec_order_value'].sum():,.0f} (SIMULATED cost basis)")

    out = resolve(cfg["paths"]["results"]) / "procurement_recommendations.parquet"
    ensure_dir(out)
    rec.to_parquet(out, index=False)
    print(f"  saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
