"""Generate the SIMULATED supplier master.

===========================================================================
EVERY NUMBER THIS MODULE PRODUCES IS SYNTHETIC.
===========================================================================
UCI Online Retail II is a sales log. It contains no suppliers, lead times,
minimum order quantities, capacities or costs. Those are required to make a
procurement decision, so they are generated here from the distributions
declared in sim/supply_config.yaml, under a fixed seed.

The generator is deterministic: the same seed and config always produce the
same supplier table (verified by tests/test_simulation.py). It is reproducible,
but it is not real, and no conclusion in this project may be stated as a
finding about the actual retailer's supply chain.
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


def _draw(rng: np.random.Generator, spec: dict, size: int) -> np.ndarray:
    """Draw `size` values from a distribution spec in the YAML config."""
    dist = spec["distribution"]
    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size)
    if dist == "gamma":
        return rng.gamma(spec["shape"], spec["scale"], size)
    raise ValueError(f"Unsupported distribution in supply_config.yaml: {dist!r}")


def generate_suppliers(sku_master: pd.DataFrame) -> pd.DataFrame:
    """One row per (SKU, supplier). All columns SIMULATED."""
    scfg = load_supply_config()
    rng = np.random.default_rng(scfg["seed"])

    n_per = scfg["suppliers"]["per_sku"]
    skus = sku_master["stock_code"].tolist()
    n = len(skus) * n_per

    sup = scfg["suppliers"]
    lt_spec = sup["lead_time"]
    lead_time = _draw(rng, lt_spec, n)
    lead_time = np.clip(np.round(lead_time), lt_spec["min_days"], lt_spec["max_days"])

    lead_time_cv = _draw(rng, sup["lead_time_cv"], n)
    unit_cost_ratio = _draw(rng, sup["unit_cost_ratio"], n)
    capacity_weeks = _draw(rng, sup["capacity_weeks"], n)
    reliability = _draw(rng, sup["reliability"], n)

    moq = rng.choice(sup["moq"]["choices"], size=n, p=sup["moq"]["weights"])
    pack = rng.choice(sup["pack_size"]["choices"], size=n, p=sup["pack_size"]["weights"])

    rows = pd.DataFrame(
        {
            "stock_code": np.repeat(skus, n_per),
            "supplier_id": [
                f"SUP-{i % n_per + 1:02d}" for i in range(n)
            ],
            "lead_time_days": lead_time.astype(int),
            "lead_time_cv": np.round(lead_time_cv, 4),
            "moq": moq.astype(int),
            "pack_size": pack.astype(int),
            "reliability": np.round(reliability, 4),
            "_unit_cost_ratio": np.round(unit_cost_ratio, 4),
            "_capacity_weeks": np.round(capacity_weeks, 4),
        }
    )

    # Cost and capacity are anchored to each SKU's own observed price and
    # demand so the simulated economics stay on a plausible scale.
    ref = sku_master.set_index("stock_code")
    rows["median_price"] = rows["stock_code"].map(ref["median_price"])
    rows["unit_cost"] = (rows["median_price"] * rows["_unit_cost_ratio"]).round(4)

    mean_weekly = (ref["total_units"] / (ref["active_days"] / 7.0)).clip(lower=1.0)
    rows["mean_weekly_units"] = rows["stock_code"].map(mean_weekly)
    rows["capacity_units"] = (
        (rows["mean_weekly_units"] * rows["_capacity_weeks"]).round().astype(int)
    )
    # A supplier that cannot ship its own minimum, or even a single pack,
    # would be unusable by construction.
    rows["capacity_units"] = np.maximum(
        rows["capacity_units"], np.maximum(rows["moq"], rows["pack_size"])
    )

    rows = rows.drop(columns=["_unit_cost_ratio", "_capacity_weeks"])
    rows["is_simulated"] = True

    # Preferred supplier = cheapest landed option, tie-broken by lead time.
    rows = rows.sort_values(["stock_code", "unit_cost", "lead_time_days"])
    rows["supplier_rank"] = rows.groupby("stock_code").cumcount() + 1
    rows["is_preferred"] = (rows["supplier_rank"] == 1)

    return rows.sort_values(["stock_code", "supplier_rank"]).reset_index(drop=True)


def main() -> int:
    cfg = load_config()
    sku_path = resolve(cfg["paths"]["processed"]) / "sku_master.parquet"
    if not sku_path.exists():
        raise FileNotFoundError(
            f"{sku_path} missing. Run scripts/prepare_data.py first."
        )
    sku_master = pd.read_parquet(sku_path)

    suppliers = generate_suppliers(sku_master)

    out = resolve(cfg["paths"]["simulated"]) / "suppliers.parquet"
    ensure_dir(out)
    suppliers.to_parquet(out, index=False)

    print(f"SIMULATED supplier master: {len(suppliers):,} rows "
          f"({suppliers['stock_code'].nunique()} SKUs x "
          f"{load_supply_config()['suppliers']['per_sku']} suppliers)")
    print(f"  lead time  : {suppliers['lead_time_days'].min()}-"
          f"{suppliers['lead_time_days'].max()} days "
          f"(mean {suppliers['lead_time_days'].mean():.1f})")
    print(f"  MOQ        : {sorted(suppliers['moq'].unique())}")
    print(f"  pack size  : {sorted(suppliers['pack_size'].unique())}")
    print(f"  unit cost  : {suppliers['unit_cost'].min():.2f}-"
          f"{suppliers['unit_cost'].max():.2f}")
    print(f"  saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
