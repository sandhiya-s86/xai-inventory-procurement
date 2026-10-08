"""Configuration and path resolution.

Everything in this project is addressed relative to the repository root so the
code runs the same from a notebook, a script, or pytest.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# src/xaierp/utils/config.py -> parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"
SUPPLY_CONFIG_PATH = PROJECT_ROOT / "sim" / "supply_config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run commands from the repository root, or check that the file exists."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load config/project.yaml."""
    return _read_yaml(CONFIG_PATH)


@lru_cache(maxsize=1)
def load_supply_config() -> dict[str, Any]:
    """Load sim/supply_config.yaml (the SIMULATED supply-side parameters)."""
    return _read_yaml(SUPPLY_CONFIG_PATH)


def resolve(relative: str | Path) -> Path:
    """Turn a repo-relative path from the config into an absolute path."""
    return (PROJECT_ROOT / Path(relative)).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing, then return it."""
    p = Path(path)
    if p.suffix:  # looks like a file -> make its parent
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p
