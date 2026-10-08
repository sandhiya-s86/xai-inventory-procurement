"""Shared pytest fixtures.

Tests that need the real artefacts skip (rather than fail) when the pipeline
has not been run, so a fresh clone can still run the pure-logic tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.utils.config import load_config, resolve  # noqa: E402

CFG = load_config()


def _need(path: Path, hint: str):
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run {hint}")
    return path


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    p = _need(
        resolve(CFG["paths"]["processed"]) / "daily_demand_panel.parquet",
        "scripts/prepare_data.py",
    )
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    p = _need(
        resolve(CFG["paths"]["processed"]) / "feature_table.parquet",
        "scripts/build_features.py",
    )
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def inventory() -> pd.DataFrame:
    p = _need(
        resolve(CFG["paths"]["simulated"]) / "inventory_state.parquet",
        "sim/inventory_state.py",
    )
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def suppliers() -> pd.DataFrame:
    p = _need(
        resolve(CFG["paths"]["simulated"]) / "suppliers.parquet",
        "sim/generate_supply.py",
    )
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def recommendations() -> pd.DataFrame:
    p = _need(
        resolve(CFG["paths"]["results"]) / "procurement_recommendations.parquet",
        "scripts/run_simulation.py",
    )
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def cfg() -> dict:
    return CFG
