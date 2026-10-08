"""Inventory conservation: stock cannot be created or destroyed.

    ending_stock = beginning_stock + arrivals - consumed
    consumed     = min(demand, beginning_stock + arrivals)
    unmet_demand = demand - consumed

If this identity fails, every cost and fill-rate number in the project is
built on invented inventory.
"""

from __future__ import annotations

import numpy as np
import pytest

from xaierp.utils.config import load_supply_config

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.inventory_state import simulate_sku  # noqa: E402

TOL = 1e-6


def test_conservation_on_a_synthetic_series():
    rng = np.random.default_rng(0)
    demand = rng.poisson(12, 200).astype(float)
    sim = simulate_sku(demand, lead_time=5, initial_stock=60.0)

    recon = sim["beginning_stock"] + sim["arrivals"] - sim["consumed"]
    assert np.allclose(recon, sim["ending_stock"], atol=TOL)


def test_consumption_is_capped_by_availability():
    rng = np.random.default_rng(1)
    demand = rng.poisson(20, 150).astype(float)
    sim = simulate_sku(demand, lead_time=7, initial_stock=5.0)

    available = sim["beginning_stock"] + sim["arrivals"]
    assert np.all(sim["consumed"] <= available + TOL)
    assert np.all(sim["consumed"] <= demand + TOL)
    assert np.allclose(sim["unmet_demand"], demand - sim["consumed"], atol=TOL)


def test_stock_never_goes_negative():
    rng = np.random.default_rng(2)
    demand = rng.poisson(30, 120).astype(float)
    sim = simulate_sku(demand, lead_time=10, initial_stock=0.0)
    assert np.all(sim["ending_stock"] >= -TOL)
    assert np.all(sim["unmet_demand"] >= -TOL)


def test_beginning_stock_chains_to_previous_ending_stock():
    rng = np.random.default_rng(3)
    demand = rng.poisson(8, 100).astype(float)
    sim = simulate_sku(demand, lead_time=4, initial_stock=40.0)
    # Today's opening position is yesterday's close (arrivals land separately).
    assert np.allclose(sim["beginning_stock"][1:], sim["ending_stock"][:-1], atol=TOL)


def test_zero_demand_never_consumes_stock():
    sim = simulate_sku(np.zeros(50), lead_time=3, initial_stock=100.0)
    assert np.allclose(sim["consumed"], 0.0)
    assert np.allclose(sim["unmet_demand"], 0.0)
    assert sim["ending_stock"][-1] >= 100.0 - TOL


def test_simulation_is_deterministic():
    demand = np.random.default_rng(4).poisson(15, 90).astype(float)
    a = simulate_sku(demand, lead_time=6, initial_stock=50.0)
    b = simulate_sku(demand, lead_time=6, initial_stock=50.0)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} differs between identical runs"


def test_conservation_holds_on_the_real_generated_state(inventory):
    """The identity must hold on the actual artefact, not only on toys."""
    df = inventory
    recon = df["beginning_stock"] + df["arrivals"] - df["consumed"]
    resid = (recon - df["ending_stock"]).abs()
    assert resid.max() < 1e-6, f"max conservation residual {resid.max()}"

    assert (df["ending_stock"] >= -1e-6).all()
    assert (df["unmet_demand"] >= -1e-6).all()
    assert (df["consumed"] <= df["demand"] + 1e-6).all()
    assert np.allclose(df["unmet_demand"], df["demand"] - df["consumed"], atol=1e-6)
