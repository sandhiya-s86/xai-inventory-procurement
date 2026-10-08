"""Regression models for 7-day demand forecasting.

Each entry carries the syllabus topic it demonstrates, so
docs/pbl1_rubric_mapping.md can be generated from code rather than written
from memory.

Unit 1 : linear regression, polynomial curve fitting, least squares,
         overfitting vs underfitting
Unit 2 : least squares / MLE equivalence, ridge regression (L2),
         Bayesian linear regression, robust linear regression
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import (
    BayesianRidge,
    HuberRegressor,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class ModelSpec:
    """One entry in the comparison table."""

    key: str
    label: str
    unit: str          # "Unit 1", "Unit 2" or "Unit 1 + Unit 2"
    concept: str       # the exact syllabus phrase this demonstrates
    build: Callable[[], Any]
    notes: str = ""


def _scaled(estimator) -> Pipeline:
    """Standardise before fitting.

    Lag counts, prices and zero-rates live on wildly different scales. Ridge
    and Huber penalise coefficients directly, so without scaling the penalty
    would fall almost entirely on whichever feature happens to be measured in
    small units. Tree models do not need this and do not get it.
    """
    return Pipeline(
        [("scaler", StandardScaler()), ("model", estimator)]
    )


def _polynomial(degree: int, alpha: float = 1.0) -> Pipeline:
    """Polynomial curve fitting of a given degree.

    A small Ridge penalty is used instead of plain least squares because the
    degree-4 expansion of correlated demand features is severely
    ill-conditioned; ordinary least squares there returns coefficients of
    order 1e12 and a meaningless fit. The penalty is held CONSTANT across
    degrees so the degree comparison stays a fair test of model capacity.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", Ridge(alpha=alpha, random_state=None)),
        ]
    )


def build_regression_models(
    lgbm_params: dict | None = None, seed: int = 42
) -> dict[str, ModelSpec]:
    """Return the full regression zoo, unfitted."""
    import lightgbm as lgb

    params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": 2,          # the dev machine is memory-constrained
        "verbose": -1,
    }
    if lgbm_params:
        params.update(lgbm_params)

    specs = [
        ModelSpec(
            key="linear_regression",
            label="Linear Regression (OLS)",
            unit="Unit 1 + Unit 2",
            concept="Linear regression / least squares; equals the MLE under "
                    "i.i.d. Gaussian noise",
            build=lambda: _scaled(LinearRegression()),
            notes="Baseline. Minimising the sum of squared errors is exactly "
                  "maximising the Gaussian likelihood, which is why no separate "
                  "'MLE model' is needed.",
        ),
        ModelSpec(
            key="ridge_regression",
            label="Ridge Regression (L2)",
            unit="Unit 2",
            concept="Ridge regression; MAP estimate under a Gaussian prior on "
                    "the weights",
            build=lambda: _scaled(Ridge(alpha=1.0)),
            notes="alpha is selected on the validation split; see "
                  "reports/ridge_alpha_search.csv.",
        ),
        ModelSpec(
            key="bayesian_ridge",
            label="Bayesian Linear Regression (Bayesian Ridge)",
            unit="Unit 2",
            concept="Bayesian linear regression; posterior over weights, "
                    "predictive variance",
            build=lambda: _scaled(BayesianRidge()),
            notes="Learns the noise and prior precision from the data and "
                  "exposes a predictive standard deviation per row.",
        ),
        ModelSpec(
            key="huber_regression",
            label="Robust Linear Regression (Huber)",
            unit="Unit 2",
            concept="Robust linear regression",
            build=lambda: _scaled(HuberRegressor(epsilon=1.35, max_iter=500)),
            notes="Demand has a long right tail (one SKU-day reaches 9,480 "
                  "units). Huber loss is linear beyond epsilon, so those days "
                  "stop dominating the fit the way squared error lets them.",
        ),
        ModelSpec(
            key="lightgbm",
            label="LightGBM (gradient-boosted trees)",
            unit="Advanced (beyond syllabus)",
            concept="Non-linear ensemble; provides TreeSHAP explanations",
            build=lambda: lgb.LGBMRegressor(**params),
            notes="Included as the strong non-linear benchmark and because "
                  "TreeSHAP gives exact per-prediction attributions.",
        ),
    ]
    return {s.key: s for s in specs}


def build_polynomial_models(degrees: list[int], alpha: float = 1.0) -> dict[str, ModelSpec]:
    """The polynomial degree study (Unit 1: polynomial curve fitting)."""
    out = {}
    for d in degrees:
        out[f"poly_degree_{d}"] = ModelSpec(
            key=f"poly_degree_{d}",
            label=f"Polynomial Regression (degree {d})",
            unit="Unit 1",
            concept="Polynomial curve fitting; overfitting vs underfitting",
            build=(lambda dd=d: _polynomial(dd, alpha=alpha)),
            notes=f"Degree {d} expansion of a 4-feature subset.",
        )
    return out


REGRESSION_MODELS = build_regression_models


def predictive_std(model, X: np.ndarray) -> np.ndarray | None:
    """Per-row predictive standard deviation, where the model provides one.

    Only BayesianRidge does, which is precisely the point of including it:
    it is the one linear model in the zoo that returns a distribution rather
    than a point.
    """
    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not isinstance(inner, BayesianRidge):
        return None
    if hasattr(model, "named_steps"):
        X = model.named_steps["scaler"].transform(X)
    _, std = inner.predict(X, return_std=True)
    return std
