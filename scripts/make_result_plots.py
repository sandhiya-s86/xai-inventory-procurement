"""Comparison figures for the report: RMSE, R2, F1, ROC, confusion matrix.

Writes reports/figures/model_*.png and reports/figures/policy_*.png.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve  # noqa: E402

from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def save(fig, path: Path) -> None:
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")


def main() -> int:
    cfg = load_config()
    reports = resolve(cfg["paths"]["reports"])
    figs = resolve(cfg["paths"]["figures"])

    reg = pd.read_csv(reports / "regression_model_comparison.csv")
    clf = pd.read_csv(reports / "classification_model_comparison.csv")
    poly = pd.read_csv(reports / "polynomial_degree_study.csv")
    pol = pd.read_csv(reports / "policy_comparison.csv")
    qt = pd.read_csv(reports / "quantile_model_metrics.csv")
    preds = pd.read_parquet(
        resolve(cfg["paths"]["forecasts"]) / "test_predictions.parquet"
    )
    summary = json.loads((reports / "model_summary.json").read_text(encoding="utf-8"))

    # ---- regression RMSE -------------------------------------------
    d = reg.sort_values("test_RMSE")
    colors = ["#718096" if k.startswith("baseline") else "#2b6cb0" for k in d["key"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(d["model"][::-1], d["test_RMSE"][::-1], color=colors[::-1])
    ax.set_xlabel("Test RMSE (units over 7 days) -- lower is better")
    ax.set_title("Regression model comparison: 7-day demand forecast")
    for i, v in enumerate(d["test_RMSE"][::-1]):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    save(fig, figs / "model_regression_rmse.png")

    # ---- regression R2 ---------------------------------------------
    d2 = reg.sort_values("test_R2", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(d2["model"][::-1], d2["test_R2"][::-1], color="#2f855a")
    ax.set_xlabel("Test R² -- higher is better")
    ax.set_title("Regression model comparison: variance explained")
    for i, v in enumerate(d2["test_R2"][::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    save(fig, figs / "model_regression_r2.png")

    # ---- classification F1 / recall ---------------------------------
    d3 = clf.sort_values("test_f1", ascending=False)
    x = np.arange(len(d3))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, d3["test_f1"], 0.4, label="F1", color="#2b6cb0")
    ax.bar(x + 0.2, d3["test_recall"], 0.4, label="Recall", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels(d3["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(
        "Classification model comparison: reorder / stockout risk\n"
        "(recall matters because a missed stockout costs more than a surplus order)"
    )
    ax.legend()
    save(fig, figs / "model_classification_f1.png")

    # ---- ROC curves --------------------------------------------------
    y = preds["reorder_flag"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _, r in clf.sort_values("test_roc_auc", ascending=False).iterrows():
        col = f"score_{r['key']}"
        if col not in preds.columns:
            continue
        fpr, tpr, _ = roc_curve(y, preds[col].to_numpy())
        ax.plot(fpr, tpr, lw=1.6, label=f"{r['model']} (AUC {r['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves -- reorder classifier")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, figs / "model_roc_curves.png")

    # ---- confusion matrix for the best classifier --------------------
    yhat = preds["pred_reorder_best"].astype(int).to_numpy()
    cm = confusion_matrix(y, yhat)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=["No reorder", "Reorder"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=","
    )
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f"Confusion matrix -- {summary['best_classifier_model']}\n"
        f"missed stockouts (FN) = {fn:,}   false alarms (FP) = {fp:,}"
    )
    ax.grid(False)
    save(fig, figs / "model_confusion_matrix.png")

    # ---- polynomial overfitting curve --------------------------------
    p = poly.sort_values("degree")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(p["degree"], p["train_RMSE"], "o-", label="Train RMSE", color="#2f855a")
    ax.plot(p["degree"], p["val_RMSE"], "s-", label="Validation RMSE", color="#dd6b20")
    ax.plot(p["degree"], p["test_RMSE"], "^-", label="Test RMSE", color="#c53030")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE (units over 7 days)")
    ax.set_xticks(p["degree"])
    ax.set_yscale("log")
    ax.set_title(
        "Polynomial curve fitting: underfitting vs overfitting (Unit 1)\n"
        "training error falls with degree while test error explodes"
    )
    ax.legend()
    save(fig, figs / "model_polynomial_overfitting.png")

    # ---- quantile calibration ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=1, label="Perfect calibration")
    ax.plot(qt["quantile"], qt["empirical_coverage_test"], "o-",
            color="#2b6cb0", label="LightGBM quantile models")
    for _, r in qt.iterrows():
        ax.annotate(f"q{int(r['quantile']*100)}",
                    (r["quantile"], r["empirical_coverage_test"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Target quantile")
    ax.set_ylabel("Empirical coverage on test")
    ax.set_title("Quantile calibration: does P90 really cover 90%?")
    ax.legend()
    save(fig, figs / "model_quantile_calibration.png")

    # ---- policy ladder: cost vs service ------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pol))
    ax1.bar(x, pol["total_cost"], 0.55, color="#a0aec0", label="Total cost")
    ax1.set_ylabel("Total simulated cost")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pol["policy"])
    ax1.set_xlabel("Policy")
    ax2 = ax1.twinx()
    ax2.plot(x, pol["fill_rate"] * 100, "o-", color="#c53030", lw=2,
             label="Fill rate (%)")
    ax2.set_ylabel("Fill rate (%)")
    ax2.grid(False)
    for i, v in enumerate(pol["fill_rate"] * 100):
        ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#c53030")
    ax1.set_title(
        "Procurement policy ladder: cost against service level\n"
        "(costs are SIMULATED; the comparison between policies is the result)"
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    save(fig, figs / "policy_cost_vs_fill_rate.png")

    # ---- predicted vs actual for the best forecaster ------------------
    s = preds.sample(min(4000, len(preds)), random_state=0)
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    ax.scatter(s["future_7d_demand"], s["pred_best_regression"], s=6,
               alpha=0.3, color="#2b6cb0")
    lim = float(np.percentile(s["future_7d_demand"], 99.5))
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual 7-day demand (units)")
    ax.set_ylabel("Predicted 7-day demand (units)")
    ax.set_title(
        f"Predicted vs actual -- {summary['best_regression_model']}\n"
        f"test R² = {summary['best_regression_test_R2']:.3f}, "
        f"RMSE = {summary['best_regression_test_RMSE']:.1f}"
    )
    ax.legend()
    save(fig, figs / "model_predicted_vs_actual.png")

    print("\nResult plots complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
