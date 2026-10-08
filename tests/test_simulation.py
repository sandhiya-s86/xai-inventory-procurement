"""Reproducibility of the simulated supply side, and sanity of the policy ladder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.generate_supply import generate_suppliers  # noqa: E402

from xaierp.explain.whatif import run_whatif_table  # noqa: E402
from xaierp.utils.config import load_config, load_supply_config, resolve  # noqa: E402


def _sku_master() -> pd.DataFrame:
    path = resolve(load_config()["paths"]["processed"]) / "sku_master.parquet"
    if not path.exists():
        pytest.skip("sku_master not built -- run scripts/prepare_data.py")
    return pd.read_parquet(path)


def test_same_seed_reproduces_the_supplier_table():
    master = _sku_master()
    a = generate_suppliers(master)
    b = generate_suppliers(master)
    pd.testing.assert_frame_equal(a, b)


def test_supplier_values_stay_inside_their_declared_ranges(suppliers):
    scfg = load_supply_config()["suppliers"]
    lt = scfg["lead_time"]
    assert suppliers["lead_time_days"].min() >= lt["min_days"]
    assert suppliers["lead_time_days"].max() <= lt["max_days"]
    assert set(suppliers["moq"].unique()) <= set(scfg["moq"]["choices"])
    assert set(suppliers["pack_size"].unique()) <= set(scfg["pack_size"]["choices"])
    assert (suppliers["reliability"] >= scfg["reliability"]["low"]).all()
    assert (suppliers["reliability"] <= scfg["reliability"]["high"]).all()
    assert (suppliers["unit_cost"] > 0).all()


def test_capacity_is_never_below_moq(suppliers):
    """A supplier that cannot ship its own minimum is unusable by construction."""
    assert (suppliers["capacity_units"] >= suppliers["moq"]).all()


def test_every_sku_has_a_single_preferred_supplier(suppliers):
    counts = suppliers.groupby("stock_code")["is_preferred"].sum()
    assert (counts == 1).all()
    per_sku = load_supply_config()["suppliers"]["per_sku"]
    assert (suppliers.groupby("stock_code").size() == per_sku).all()


def test_simulated_data_is_flagged_as_simulated(suppliers, inventory):
    assert suppliers["is_simulated"].all()
    assert inventory["is_simulated_supply"].all()


def test_whatif_responds_to_every_lever():
    """A counterfactual panel where nothing moves is a broken panel."""
    base = {
        "demand_7d": 350.0, "q50": 330.0, "q95": 520.0,
        "current_stock": 200.0, "lead_time_days": 8,
        "review_period": 7, "horizon": 7,
        "moq": 12, "pack_size": 12,
        "supplier_capacity": 100000,       # deliberately non-binding
        "unit_cost": 1.5, "mean_daily_demand": 50.0,
    }
    rows = run_whatif_table(base)
    qty = {r["scenario"]: r["recommended_qty"] for r in rows}
    current = qty["Current situation"]

    assert len(set(qty.values())) > 1, "no scenario changed the recommendation"

    longer = [k for k in qty if "doubled" in k][0]
    shorter = [k for k in qty if "halved" in k][0]
    assert qty[longer] > current, "a longer lead time must require more stock"
    assert qty[shorter] < current, "a shorter lead time must require less stock"

    assert qty["Stock fully depleted (0 units)"] > current
    assert qty["Stock = 150% of lead-time demand"] < current
    assert qty["Demand surge +30%"] > current
    assert qty["No safety buffer (service level dropped)"] < current


def test_policy_ladder_outputs_are_coherent():
    path = resolve(load_config()["paths"]["reports"]) / "policy_comparison.csv"
    if not path.exists():
        pytest.skip("policy comparison not built -- run scripts/run_simulation.py")
    df = pd.read_csv(path)

    assert len(df) == 5
    assert set(df["policy"]) == {"P0", "P1", "P2", "P3", "P4"}
    assert ((df["fill_rate"] >= 0) & (df["fill_rate"] <= 1)).all()
    assert (df["unmet_demand_units"] >= 0).all()
    assert (df["total_cost"] > 0).all()
    assert (df["n_orders"] > 0).all()

    # Costs must add up.
    recomputed = (
        df["holding_cost"] + df["purchase_cost"]
        + df["shortage_cost"] + df["ordering_cost"]
    )
    assert np.allclose(recomputed, df["total_cost"], atol=0.05)

    # Every policy faces identical demand -- that is what makes it a fair test.
    assert df["total_demand_units"].nunique() == 1

    # Adding a safety buffer must improve service.
    p0 = df.loc[df["policy"] == "P0", "fill_rate"].iloc[0]
    p4 = df.loc[df["policy"] == "P4", "fill_rate"].iloc[0]
    assert p4 > p0, "the full policy should beat the naive baseline on service"
