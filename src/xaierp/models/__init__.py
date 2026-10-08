"""Model zoo. Each builder returns an unfitted estimator plus its syllabus tag."""

from .regression import REGRESSION_MODELS, build_regression_models
from .classification import CLASSIFICATION_MODELS, build_classification_models

__all__ = [
    "REGRESSION_MODELS",
    "build_regression_models",
    "CLASSIFICATION_MODELS",
    "build_classification_models",
]
