"""Stage 2: daily panel + simulated inventory -> modelling table with splits.

Run from the repository root:
    .venv\\Scripts\\python scripts\\build_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.features.builder import (  # noqa: E402
    INVENTORY_FEATURES,
    build_feature_table,
    chronological_split,
    classification_features,
    regression_features,
    save_features,
)
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402

# Simulated columns carried through to the modelling table.
INVENTORY_CARRY = INVENTORY_FEATURES + [
    "beginning_stock", "ending_stock", "arrivals", "consumed",
    "unmet_demand", "on_order", "orders_placed",
    "reorder_flag", "stockout_risk", "lead_time_demand_actual",
    "supplier_id", "supplier_capacity",
]


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])

    panel_path = resolve(cfg["paths"]["processed"]) / "daily_demand_panel.parquet"
    inv_path = resolve(cfg["paths"]["simulated"]) / "inventory_state.parquet"
    for p, hint in [
        (panel_path, "scripts/prepare_data.py"),
        (inv_path, "sim/inventory_state.py"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run {hint} first.")

    print("[1/4] Building demand features and 7-day target...")
    panel = pd.read_parquet(panel_path)
    feats = build_feature_table(panel)
    print(
        f"      {len(feats):,} usable rows "
        f"(dropped {feats.attrs['rows_dropped_incomplete_history']:,} for incomplete "
        f"history, {feats.attrs['rows_dropped_incomplete_target']:,} for incomplete "
        f"forward window)"
    )

    print("[2/4] Attaching SIMULATED inventory state...")
    inv = pd.read_parquet(inv_path)
    carry = ["stock_code", "date"] + [c for c in INVENTORY_CARRY if c in inv.columns]
    feats = feats.merge(inv[carry], on=["stock_code", "date"], how="left")
    before = len(feats)
    feats = feats.dropna(subset=["reorder_flag"] + INVENTORY_FEATURES)
    print(f"      {len(feats):,} rows with a complete inventory position "
          f"(dropped {before - len(feats):,})")

    print("[3/4] Chronological split...")
    feats = chronological_split(feats)
    counts = feats["split"].value_counts()
    dates = feats.groupby("split")["date"].agg(["min", "max"])
    for name in ("train", "val", "test"):
        row = dates.loc[name]
        print(
            f"      {name:<5} {counts[name]:>7,} rows  "
            f"{row['min'].date()} -> {row['max'].date()}"
        )

    reg_feats = regression_features(feats)
    clf_feats = classification_features(feats)
    print(f"      regression features    : {len(reg_feats)}")
    print(f"      classification features: {len(clf_feats)}")
    pos = feats["reorder_flag"].mean()
    print(f"      reorder_flag positive rate: {pos:.2%}")

    print("[4/4] Saving...")
    path = save_features(feats)
    print(f"      saved -> {path.relative_to(ROOT)}")

    meta = {
        "n_rows": int(len(feats)),
        "n_skus": int(feats["stock_code"].nunique()),
        "date_min": str(feats["date"].min().date()),
        "date_max": str(feats["date"].max().date()),
        "train_end": feats.attrs.get("train_end"),
        "val_end": feats.attrs.get("val_end"),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "regression_features": reg_feats,
        "classification_features": clf_feats,
        "target_regression": cfg["target"]["regression_target"],
        "target_classification": "reorder_flag",
        "reorder_positive_rate": float(pos),
        "horizon_days": int(cfg["target"]["horizon_days"]),
    }
    meta_path = resolve(cfg["paths"]["reports"]) / "feature_metadata.json"
    ensure_dir(meta_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"      saved -> {meta_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
