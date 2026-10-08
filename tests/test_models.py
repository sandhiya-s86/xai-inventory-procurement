"""Models train, predict sanely, and the reported comparison tables are honest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xaierp.models.classification import build_classification_models, decision_scores
from xaierp.models.regression import build_polynomial_models, build_regression_models
from xaierp.utils.config import load_config, resolve
from xaierp.utils.metrics import classification_metrics, pinball_loss, regression_metrics


# ----------------------------------------------------------------- metrics


def test_regression_metrics_are_zero_for_a_perfect_fit():
    y = np.array([1.0, 5.0, 12.0, 100.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["RMSLE"] == 0
    assert m["R2"] == pytest.approx(1.0)


def test_rmse_is_the_root_of_mse():
    rng = np.random.default_rng(0)
    y, p = rng.normal(50, 10, 200), rng.normal(50, 10, 200)
    m = regression_metrics(y, p)
    assert m["RMSE"] == pytest.approx(np.sqrt(m["MSE"]))


def test_classification_metrics_on_a_known_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])   # TP=2 FN=1 FP=1 TN=4
    m = classification_metrics(y, p)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_pinball_loss_penalises_asymmetrically():
    """A high quantile should punish under-prediction harder than over."""
    under = pinball_loss(np.array([100.0]), np.array([80.0]), 0.9)
    over = pinball_loss(np.array([100.0]), np.array([120.0]), 0.9)
    assert under > over


# ----------------------------------------------------------------- training


@pytest.fixture(scope="module")
def tiny():
    """A small, learnable problem: y depends linearly on two of four features."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.3, size=600)
    return X, y


def test_every_regression_model_trains_and_predicts(tiny):
    X, y = tiny
    for key, spec in build_regression_models(seed=0).items():
        model = spec.build()
        model.fit(X[:400], y[:400])
        pred = model.predict(X[400:])
        assert pred.shape == (200,)
        assert np.isfinite(pred).all(), f"{key} produced non-finite predictions"


def test_linear_models_recover_a_known_signal(tiny):
    X, y = tiny
    for key in ("linear_regression", "ridge_regression", "bayesian_ridge"):
        model = build_regression_models(seed=0)[key].build()
        model.fit(X[:400], y[:400])
        r2 = regression_metrics(y[400:], model.predict(X[400:]))["R2"]
        assert r2 > 0.9, f"{key} failed to learn an easy linear signal (R2={r2})"


def test_every_classifier_trains_and_scores(tiny):
    X, y = tiny
    labels = (y > 0).astype(int)
    for key, spec in build_classification_models(seed=0, svm_sample=300).items():
        model = spec.build()
        model.fit(X[:400], labels[:400])
        pred = model.predict(X[400:])
        assert set(np.unique(pred)) <= {0, 1}, f"{key} predicted a non-binary label"
        scores = decision_scores(model, X[400:])
        assert scores.shape == (200,)
        assert np.isfinite(scores).all()


def test_polynomial_degree_increases_model_capacity(tiny):
    X, y = tiny
    counts = []
    for key, spec in build_polynomial_models([1, 2, 3]).items():
        m = spec.build()
        m.fit(X[:400], y[:400])
        counts.append(m.named_steps["poly"].n_output_features_)
    assert counts == sorted(counts) and len(set(counts)) == 3


# ----------------------------------------------------------------- artefacts


def _report(name: str) -> pd.DataFrame:
    path = resolve(load_config()["paths"]["reports"]) / name
    if not path.exists():
        pytest.skip(f"{name} not built -- run scripts/train_models.py")
    return pd.read_csv(path)


def test_regression_comparison_table_is_well_formed():
    df = _report("regression_model_comparison.csv")
    for col in ("test_MAE", "test_RMSE", "test_R2", "test_RMSLE", "train_time_s"):
        assert col in df.columns
    assert (df["test_RMSE"] > 0).all()
    assert (df["test_MAE"] <= df["test_RMSE"] + 1e-9).all(), "MAE cannot exceed RMSE"
    assert df["test_R2"].max() <= 1.0
    # Baselines must be present so the comparison means something.
    assert df["key"].str.startswith("baseline").any()


def test_classification_comparison_table_is_well_formed():
    df = _report("classification_model_comparison.csv")
    for col in ("test_accuracy", "test_precision", "test_recall",
                "test_f1", "test_roc_auc"):
        assert col in df.columns
        assert ((df[col] >= 0) & (df[col] <= 1)).all()
    # A model that beats coin-flipping on AUC is the minimum bar.
    assert df["test_roc_auc"].max() > 0.6
    # All six syllabus models must be reported, including the weak ones.
    assert len(df) == 6


def test_polynomial_study_shows_the_overfitting_curve():
    df = _report("polynomial_degree_study.csv").sort_values("degree")
    # Training error must fall monotonically as capacity grows.
    assert df["train_RMSE"].is_monotonic_decreasing, (
        "a higher-degree polynomial should always fit the training data better"
    )
    # And the generalisation gap must widen.
    assert df["gap_test_minus_train_RMSE"].iloc[-1] > df["gap_test_minus_train_RMSE"].iloc[0]


def test_quantile_models_are_ordered_and_roughly_calibrated():
    df = _report("quantile_model_metrics.csv").sort_values("quantile")
    assert df["mean_prediction"].is_monotonic_increasing, (
        "a higher quantile must predict a higher value"
    )
    assert (df["coverage_error"].abs() < 0.10).all(), (
        "quantile coverage is more than 10 points off its target"
    )


def test_model_summary_matches_the_comparison_tables():
    path = resolve(load_config()["paths"]["reports"]) / "model_summary.json"
    if not path.exists():
        pytest.skip("model summary not built")
    s = json.loads(path.read_text(encoding="utf-8"))

    reg = _report("regression_model_comparison.csv")
    clf = _report("classification_model_comparison.csv")
    assert s["best_regression_test_RMSE"] == pytest.approx(reg["test_RMSE"].min())
    assert s["best_classifier_test_f1"] == pytest.approx(clf["test_f1"].max())
