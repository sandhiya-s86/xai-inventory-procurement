"""Explainability: SHAP attributions plus a decision-level account."""

from .shap_forecast import (
    classifier_shap,
    forecast_shap,
    top_contributions,
)
from .decision_explanation import explain_decision, render_explanation

__all__ = [
    "forecast_shap",
    "classifier_shap",
    "top_contributions",
    "explain_decision",
    "render_explanation",
]
