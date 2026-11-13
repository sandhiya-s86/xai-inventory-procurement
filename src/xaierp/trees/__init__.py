"""Unit 5 — Classification and Regression Trees (CART).

    cart.py   binary recursive partitioning, Gini / entropy / MSE criteria,
              minimal cost-complexity pruning, impurity feature importances,
              and human-readable decision rules

CART is implemented rather than imported because the parts that matter for
this project are the parts a library hides: the pruning sequence, and the
exact chain of if-then tests that produced a given decision. The second of
those is the point. Every other classifier in this repo answers "reorder?"
with a probability and needs SHAP to explain itself; a pruned tree answers it
with a rule a buyer can read, argue with, and overrule.

tests/test_unit5_cart.py pins the implementation against scikit-learn's
DecisionTreeClassifier and DecisionTreeRegressor.
"""

from __future__ import annotations

from .cart import (
    CART,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Node,
    cost_complexity_pruning_path,
    entropy,
    gini,
    mse_impurity,
)

__all__ = [
    "CART",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "Node",
    "gini",
    "entropy",
    "mse_impurity",
    "cost_complexity_pruning_path",
]
