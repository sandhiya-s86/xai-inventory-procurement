"""Classification models for the reorder / stockout-risk decision.

Unit 2 coverage, mapped explicitly:

    Discriminant function          -> LinearDiscriminantAnalysis
    Probabilistic generative model -> GaussianNB (models p(x|y) and p(y))
    Probabilistic discriminative   -> LogisticRegression (models p(y|x))
    Kernel functions / kernel trick-> SVC(linear) vs SVC(rbf)
    SVMs                           -> both SVC variants

Putting the generative and the discriminative model side by side is the point
of the comparison: they are fitted on identical data and differ only in what
they model, which is exactly the distinction the syllabus draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ClfSpec:
    key: str
    label: str
    unit: str
    concept: str
    build: Callable[[], Any]
    # SVC is O(n^2)-ish in the number of samples. Training it on 67k rows is
    # not practical on this machine, so kernel models are fitted on a
    # subsample. The subsample size is declared here rather than hidden in
    # the training script, and it is reported in the results table.
    max_train_rows: int | None = None
    notes: str = ""


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_classification_models(seed: int = 42, svm_sample: int = 15_000) -> dict[str, ClfSpec]:
    import lightgbm as lgb

    specs = [
        ClfSpec(
            key="logistic_regression",
            label="Logistic Regression",
            unit="Unit 2",
            concept="Probabilistic discriminative model; models p(y|x) directly",
            build=lambda: _scaled(
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                )
            ),
            notes="class_weight='balanced' because missing a stockout costs "
                  "more than a surplus order.",
        ),
        ClfSpec(
            key="lda",
            label="Linear Discriminant Analysis",
            unit="Unit 2",
            concept="Discriminant function; linear decision boundary from "
                    "class-conditional Gaussians with shared covariance",
            build=lambda: _scaled(LinearDiscriminantAnalysis()),
            notes="The closed-form discriminant function counterpart to "
                  "logistic regression.",
        ),
        ClfSpec(
            key="gaussian_nb",
            label="Gaussian Naive Bayes",
            unit="Unit 2 (+ Unit 1 Bayes rule)",
            concept="Probabilistic generative model; models p(x|y)p(y) and "
                    "applies Bayes rule to obtain p(y|x)",
            build=lambda: _scaled(GaussianNB()),
            notes="Included as the generative contrast to logistic "
                  "regression, and as the direct application of Bayes rule "
                  "from Unit 1.",
        ),
        ClfSpec(
            key="svm_linear",
            label="SVM (linear kernel)",
            unit="Unit 2",
            concept="Support vector machine; maximum-margin linear boundary",
            build=lambda: _scaled(
                SVC(kernel="linear", C=1.0, class_weight="balanced",
                    cache_size=300, random_state=seed)
            ),
            max_train_rows=svm_sample,
            notes="Fitted on a subsample (see max_train_rows) because SVC "
                  "scales quadratically with sample count.",
        ),
        ClfSpec(
            key="svm_rbf",
            label="SVM (RBF kernel)",
            unit="Unit 2",
            concept="Kernel function / kernel trick; non-linear boundary via "
                    "an implicit high-dimensional feature map",
            build=lambda: _scaled(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    cache_size=300, random_state=seed)
            ),
            max_train_rows=svm_sample,
            notes="Same subsample and same C as the linear SVM, so the "
                  "difference between the two is attributable to the kernel "
                  "alone.",
        ),
        ClfSpec(
            key="lightgbm_clf",
            label="LightGBM Classifier",
            unit="Advanced (beyond syllabus)",
            concept="Non-linear ensemble; TreeSHAP explanations for the "
                    "reorder decision",
            build=lambda: lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=seed,
                n_jobs=2,
                verbose=-1,
            ),
            notes="Strong benchmark and the model the explainability layer "
                  "uses for the reorder decision.",
        ),
    ]
    return {s.key: s for s in specs}


CLASSIFICATION_MODELS = build_classification_models


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    """A monotone score for ROC-AUC, whatever the estimator supports.

    SVC without probability=True has no predict_proba, but decision_function
    is a perfectly valid ranking score for AUC -- and far cheaper than the
    internal cross-validation that probability=True would run.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)
