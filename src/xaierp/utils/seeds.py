"""Deterministic seeding.

Every stochastic step in this project goes through here so that two runs of
scripts/run_pipeline.py produce byte-identical results.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and hash randomisation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    """A fresh, independent NumPy Generator for one simulation component."""
    return np.random.default_rng(seed)
