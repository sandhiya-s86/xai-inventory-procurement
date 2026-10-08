"""SHAP attributions for the two models that actually make decisions.

Explainer choice is driven by which model is deployed, not by convenience:

  demand forecast  -> Ridge regression won on test RMSE, so it is the model
                      that must be explained. For a linear model on
                      standardised inputs the Shapley value has a closed
                      form, phi_i = beta_i * (x_i - E[x_i]), so the
                      attributions are EXACT, not sampled.

  reorder decision -> LightGBM won on test F1, so TreeSHAP applies. TreeSHAP
                      is also exact for tree ensembles.

In both cases the additivity property holds:

    base_value + sum(phi_i) == model output

`verify_additivity` checks it, and tests/test_explanations.py enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShapResult:
    values: np.ndarray        # (n_rows, n_features)
    base_value: float
    feature_names: list[str]
    predictions: np.ndarray

    def verify_additivity(self, tol: float = 1e-4) -> float:
        """Max absolute gap between base + sum(phi) and the model output."""
        recon = self.base_value + self.values.sum(axis=1)
        return float(np.max(np.abs(recon - self.predictions)))


def forecast_shap(
    model, X: np.ndarray, feature_names: list[str]
) -> ShapResult:
    """Exact Shapley values for the Ridge/linear demand forecaster.

    The model is a Pipeline(StandardScaler -> linear). Working in the scaled
    space, phi_i = beta_i * (z_i - mean(z_i)). Because the scaler centres the
    training data, mean(z_i) over the TRAINING set is 0 by construction, so
    the reference point is the training mean -- which is exactly the
    reference SHAP assumes.
    """
    scaler = model.named_steps["scaler"]
    linear = model.named_steps["model"]

    Z = scaler.transform(X)
    beta = np.asarray(linear.coef_, dtype=np.float64).ravel()
    intercept = float(np.asarray(linear.intercept_).ravel()[0])

    # Reference = the training mean, which maps to z = 0.
    phi = Z * beta  # (n, p)
    preds = model.predict(X)

    return ShapResult(
        values=phi,
        base_value=intercept,
        feature_names=list(feature_names),
        predictions=np.asarray(preds, dtype=np.float64),
    )


def classifier_shap(
    model, X: np.ndarray, feature_names: list[str], max_rows: int | None = None
) -> ShapResult:
    """TreeSHAP for the LightGBM reorder classifier, in log-odds space.

    Log-odds is the additive space for a boosted classifier: contributions
    sum to the raw margin, which the sigmoid then turns into a probability.
    Explaining in probability space would break additivity.
    """
    import shap

    if max_rows is not None and len(X) > max_rows:
        X = X[:max_rows]

    booster = model.booster_ if hasattr(model, "booster_") else model
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)

    # LightGBM binary returns either (n, p) or a 2-element list.
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim == 3:
        sv = sv[:, :, -1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    base = float(base)

    raw = model.predict_proba(X)[:, 1]
    margin = np.log(np.clip(raw, 1e-12, 1 - 1e-12) / np.clip(1 - raw, 1e-12, 1))

    return ShapResult(
        values=sv,
        base_value=base,
        feature_names=list(feature_names),
        predictions=margin,
    )


def top_contributions(
    result: ShapResult, row: int, feature_values: np.ndarray, k: int = 5
) -> dict[str, list[dict]]:
    """The k strongest positive and negative drivers for one prediction.

    Returns the real feature value alongside each attribution so the text
    explanation can say "28-day mean demand is 41.2 units", never a
    placeholder.
    """
    phi = result.values[row]
    order = np.argsort(phi)

    def pack(indices):
        out = []
        for i in indices:
            out.append(
                {
                    "feature": result.feature_names[i],
                    "shap_value": float(phi[i]),
                    "feature_value": float(feature_values[i]),
                }
            )
        return out

    negatives = pack(order[:k])
    positives = pack(order[::-1][:k])
    return {
        "positive": [p for p in positives if p["shap_value"] > 0],
        "negative": [n for n in negatives if n["shap_value"] < 0],
    }


def global_importance(result: ShapResult) -> pd.DataFrame:
    """Mean |SHAP| per feature -- the global importance ranking."""
    imp = np.abs(result.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": result.feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
