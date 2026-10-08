"""Stage 5: SHAP attributions, decision explanations and what-if analysis.

Writes:
    reports/shap_global_importance_forecast.csv
    reports/shap_global_importance_classifier.csv
    reports/figures/shap_*.png
    reports/generated/case_studies.json
    reports/generated/case_studies.txt
    reports/whatif_examples.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.explain.decision_explanation import explain_decision, render_explanation  # noqa: E402
from xaierp.explain.shap_forecast import (  # noqa: E402
    classifier_shap,
    forecast_shap,
    global_importance,
    top_contributions,
)
from xaierp.explain.whatif import run_whatif_table  # noqa: E402
from xaierp.features.builder import (  # noqa: E402
    classification_features,
    load_features,
    regression_features,
    split_frames,
)
from xaierp.policy.reorder import decide_reorder, safety_stock_from_quantiles  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402

SHAP_SAMPLE = 4000   # rows used for the global importance picture
N_CASES = 6          # worked examples written out in full


def _bar_plot(imp: pd.DataFrame, title: str, path: Path, top: int = 15) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _waterfall(contribs: dict, sku: str, base: float, pred: float, path: Path) -> None:
    items = contribs["positive"][:5] + contribs["negative"][:5]
    items = sorted(items, key=lambda d: d["shap_value"])
    names = [f"{i['feature']}\n= {i['feature_value']:,.2f}" for i in items]
    vals = [i["shap_value"] for i in items]
    colors = ["#c53030" if v < 0 else "#2f855a" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP contribution to predicted 7-day demand (units)")
    ax.set_title(
        f"Why this forecast? SKU {sku}\n"
        f"base {base:,.1f} + contributions = prediction {pred:,.1f} units"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    figs = resolve(cfg["paths"]["figures"])
    reports = resolve(cfg["paths"]["reports"])
    gen = reports / "generated"
    ensure_dir(gen)

    summary = json.loads((reports / "model_summary.json").read_text(encoding="utf-8"))
    best_reg_key = summary["best_regression_key"]
    best_clf_key = summary["best_classifier_key"]

    df = load_features()
    te = split_frames(df)["test"].reset_index(drop=True)
    reg_feats = regression_features(df)
    clf_feats = classification_features(df)

    rec_path = resolve(cfg["paths"]["results"]) / "procurement_recommendations.parquet"
    if not rec_path.exists():
        raise FileNotFoundError(f"{rec_path} missing. Run scripts/run_simulation.py.")
    rec = pd.read_parquet(rec_path)

    models_dir = resolve(cfg["paths"]["models"])
    reg_model = joblib.load(models_dir / f"reg_{best_reg_key}.joblib")
    clf_model = joblib.load(models_dir / f"clf_{best_clf_key}.joblib")

    # ---------------------------------------------------------------
    print("=" * 70)
    print("SHAP -- demand forecast")
    print(f"  model: {summary['best_regression_model']} (exact linear Shapley values)")
    print("=" * 70)
    samp = te.iloc[:SHAP_SAMPLE]
    Xr = samp[reg_feats].to_numpy(dtype=np.float32)
    res_reg = forecast_shap(reg_model, Xr, reg_feats)
    add_err = res_reg.verify_additivity()
    print(f"  additivity check: max |base + sum(phi) - prediction| = {add_err:.2e}")
    if add_err > 1e-3:
        raise AssertionError(f"SHAP additivity violated for the forecaster ({add_err})")

    imp_reg = global_importance(res_reg)
    imp_reg.to_csv(reports / "shap_global_importance_forecast.csv", index=False)
    _bar_plot(
        imp_reg,
        "Global feature importance -- 7-day demand forecast (Ridge, exact SHAP)",
        figs / "shap_global_importance_forecast.png",
    )
    print("  top drivers: " + ", ".join(imp_reg["feature"].head(5)))

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SHAP -- reorder classifier")
    print(f"  model: {summary['best_classifier_model']} (TreeSHAP, log-odds space)")
    print("=" * 70)
    Xc = samp[clf_feats].to_numpy(dtype=np.float32)
    res_clf = classifier_shap(clf_model, Xc, clf_feats)
    add_err_c = res_clf.verify_additivity()
    print(f"  additivity check: max |base + sum(phi) - margin| = {add_err_c:.2e}")

    imp_clf = global_importance(res_clf)
    imp_clf.to_csv(reports / "shap_global_importance_classifier.csv", index=False)
    _bar_plot(
        imp_clf,
        "Global feature importance -- reorder decision (LightGBM, TreeSHAP)",
        figs / "shap_global_importance_classifier.png",
    )
    print("  top drivers: " + ", ".join(imp_clf["feature"].head(5)))

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"CASE STUDIES ({N_CASES} worked examples)")
    print("=" * 70)
    horizon = cfg["target"]["horizon_days"]
    review = 7

    # Pick a spread: highest risk, lowest risk, biggest order, and a few
    # mid-range rows, so the examples are not all the same situation.
    pool = rec.reset_index(drop=True)
    pool = pool.loc[pool.index < SHAP_SAMPLE]
    picks = []
    picks.append(int(pool["prob_best_classifier"].idxmax()))
    picks.append(int(pool["prob_best_classifier"].idxmin()))
    picks.append(int(pool["rec_qty"].idxmax()))
    mid = pool.loc[
        (pool["prob_best_classifier"] > 0.4) & (pool["prob_best_classifier"] < 0.7)
    ]
    picks += [int(i) for i in mid.index[:3]]
    picks = list(dict.fromkeys(picks))[:N_CASES]

    cases, texts, whatif_rows = [], [], []
    for n, idx in enumerate(picks, 1):
        row = pool.loc[idx]
        lead = int(row["lead_time_days"])
        scale = (lead + review) / horizon
        q50 = max(0.0, float(row["pred_q50"]))
        q95 = max(0.0, float(row["pred_q95"]))
        f7 = float(row["pred_best_regression"])

        decision = decide_reorder(
            expected_demand_leadtime=max(0.0, f7) * scale,
            safety_stock=safety_stock_from_quantiles(q95, q50, scale=scale),
            inventory_position=float(row["inventory_position"]),
            lead_time_days=lead,
            moq=int(row["moq"]),
            pack_size=int(row["pack_size"]),
            supplier_capacity=int(row["supplier_capacity"]),
            unit_cost=float(row["unit_cost"]),
            mean_daily_demand=float(row["rolling_mean_28"]),
        )

        contribs = top_contributions(res_reg, idx, Xr[idx], k=5)
        exp = explain_decision(
            sku=str(row["stock_code"]),
            description=str(row.get("description", "")),
            date=row["date"],
            decision=decision,
            forecast_7d=f7,
            reorder_probability=float(row["prob_best_classifier"]),
            shap_top=contribs,
            mean_daily_demand=float(row["rolling_mean_28"]),
        )
        exp["actual_7d_demand"] = float(row["future_7d_demand"])
        exp["actual_reorder_label"] = int(row["reorder_flag"])
        cases.append(exp)

        text = render_explanation(exp)
        text += (
            f"\n\nOutcome check (hindsight, not available at decision time):"
            f"\n  actual demand over the next 7 days : "
            f"{exp['actual_7d_demand']:,.0f} units"
            f"\n  actual reorder-needed label        : "
            f"{'YES' if exp['actual_reorder_label'] else 'NO'}"
        )
        texts.append(f"{'=' * 72}\nCASE STUDY {n}\n{'=' * 72}\n{text}")
        print(f"  case {n}: SKU {row['stock_code']}  "
              f"risk={exp['stockout_risk']}  qty={exp['recommended_qty']:,}  "
              f"reconciles={exp['reconciles']}")

        _waterfall(
            contribs, str(row["stock_code"]),
            res_reg.base_value, float(res_reg.predictions[idx]),
            figs / f"shap_waterfall_case{n}.png",
        )

        # -------- what-if panel for this case -----------------------
        base = {
            "demand_7d": f7, "q50": q50, "q95": q95,
            "current_stock": float(row["inventory_position"]),
            "lead_time_days": lead, "review_period": review, "horizon": horizon,
            "moq": int(row["moq"]), "pack_size": int(row["pack_size"]),
            "supplier_capacity": int(row["supplier_capacity"]),
            "unit_cost": float(row["unit_cost"]),
            "mean_daily_demand": float(row["rolling_mean_28"]),
        }
        for r in run_whatif_table(base):
            r["case"] = n
            r["stock_code"] = str(row["stock_code"])
            whatif_rows.append(r)

    (gen / "case_studies.json").write_text(
        json.dumps(cases, indent=2, default=str), encoding="utf-8"
    )
    (gen / "case_studies.txt").write_text("\n\n".join(texts), encoding="utf-8")
    print(f"  saved -> {(gen / 'case_studies.txt').relative_to(ROOT)}")

    wf = pd.DataFrame(whatif_rows)
    wf.to_csv(reports / "whatif_examples.csv", index=False)
    print(f"\n  what-if scenarios: {len(wf)} rows across {wf['case'].nunique()} cases")
    print(f"  saved -> {(reports / 'whatif_examples.csv').relative_to(ROOT)}")

    # Show one what-if panel so the run is self-evidencing.
    first = wf.loc[wf["case"] == wf["case"].iloc[0]]
    print(f"\n  Example what-if panel (SKU {first['stock_code'].iloc[0]}):")
    for _, r in first.iterrows():
        print(f"    {r['scenario']:<48} qty={r['recommended_qty']:>7,}  "
              f"({r['change_vs_current']:+,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
