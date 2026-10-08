"""Streamlit decision-support demo.

Run from the repository root:
    .venv\\Scripts\\streamlit run dashboard/app.py

This is a research/demo front end, not an ERP. It reads the artefacts the
pipeline produced and lets a reviewer walk the whole story: data -> forecast
-> risk -> recommendation -> why.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.explain.decision_explanation import explain_decision, render_explanation  # noqa: E402
from xaierp.explain.shap_forecast import forecast_shap, top_contributions  # noqa: E402
from xaierp.explain.whatif import Scenario, run_scenario, run_whatif_table  # noqa: E402
from xaierp.features.builder import regression_features  # noqa: E402
from xaierp.policy.reorder import decide_reorder, safety_stock_from_quantiles  # noqa: E402
from xaierp.utils.config import load_config, resolve  # noqa: E402

st.set_page_config(
    page_title="XAI Inventory & Procurement Decision Support",
    page_icon="📦",
    layout="wide",
)

CFG = load_config()
REPORTS = resolve(CFG["paths"]["reports"])
FIGS = resolve(CFG["paths"]["figures"])
REVIEW_PERIOD = 7


# ----------------------------------------------------------------- loading


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    p = resolve(CFG["paths"]["results"]) / "procurement_recommendations.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_panel_df() -> pd.DataFrame:
    p = resolve(CFG["paths"]["processed"]) / "daily_demand_panel.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_features_df() -> pd.DataFrame:
    p = resolve(CFG["paths"]["processed"]) / "feature_table.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_reg_model():
    s = read_json("model_summary.json")
    if not s:
        return None
    p = resolve(CFG["paths"]["models"]) / f"reg_{s['best_regression_key']}.joblib"
    return joblib.load(p) if p.exists() else None


def guard(df: pd.DataFrame, what: str, how: str) -> bool:
    if df.empty:
        st.error(f"{what} not found. Run `{how}` first.")
        return False
    return True


rec = load_recommendations()
summary = read_json("model_summary.json")

st.sidebar.title("📦 XAI Decision Support")
st.sidebar.caption(
    "Explainable AI for Predictive Inventory and Procurement Optimization "
    "in ERP Systems"
)
page = st.sidebar.radio(
    "Page",
    [
        "1 · Overview",
        "2 · Demand Prediction",
        "3 · Inventory Risk",
        "4 · Procurement Recommendation",
        "5 · WHY?  (Explainability)",
        "6 · Model Comparison",
    ],
)
st.sidebar.markdown("---")
st.sidebar.warning(
    "**Data provenance**\n\n"
    "Sales, dates, prices and SKUs are **real** (UCI Online Retail II).\n\n"
    "Stock levels, suppliers, lead times, MOQ and all costs are "
    "**SIMULATED** — the source dataset contains none of them."
)


# ================================================================ PAGE 1
if page.startswith("1"):
    st.title("Overview")
    panel = load_panel_df()
    if guard(panel, "Daily demand panel", "python scripts/prepare_data.py"):
        dq = read_json("data_quality_report.json")
        c = st.columns(4)
        c[0].metric("SKUs modelled", f"{panel['stock_code'].nunique():,}")
        c[1].metric("Trading days", f"{panel['date'].nunique():,}")
        c[2].metric("Units sold", f"{panel['demand'].sum():,.0f}")
        c[3].metric("Revenue", f"{panel['revenue'].sum():,.0f}")

        if dq:
            c = st.columns(4)
            c[0].metric("Raw invoice lines", f"{dq['raw']['n_rows']:,}")
            c[1].metric("After cleaning", f"{dq['clean']['n_rows']:,}")
            c[2].metric("Customers", f"{dq['raw']['n_customers']:,}")
            c[3].metric("Countries", f"{dq['raw']['n_countries']:,}")

        st.subheader("Total daily demand")
        daily = panel.groupby("date")["demand"].sum()
        st.line_chart(daily, height=260)

        left, right = st.columns(2)
        with left:
            st.subheader("Top 15 SKUs by revenue")
            top = (
                panel.groupby(["stock_code", "description"])["revenue"]
                .sum().sort_values(ascending=False).head(15).reset_index()
            )
            top["revenue"] = top["revenue"].round(0)
            st.dataframe(top, hide_index=True, use_container_width=True)
        with right:
            st.subheader("Monthly demand")
            st.bar_chart(daily.resample("MS").sum(), height=300)

        st.subheader("Dataset summary")
        st.dataframe(read_csv("dataset_summary.csv"), hide_index=True,
                     use_container_width=True)


# ================================================================ PAGE 2
elif page.startswith("2"):
    st.title("Demand Prediction")
    if guard(rec, "Predictions", "python scripts/train_models.py"):
        skus = sorted(rec["stock_code"].unique())
        c1, c2 = st.columns([2, 3])
        sku = c1.selectbox("Product (SKU)", skus)
        sub = rec.loc[rec["stock_code"] == sku].sort_values("date")
        desc = str(sub["description"].iloc[0])
        c2.text_input("Description", desc, disabled=True)

        date = st.select_slider(
            "Decision date",
            options=list(sub["date"].dt.date),
            value=list(sub["date"].dt.date)[-1],
        )
        row = sub.loc[sub["date"].dt.date == date].iloc[0]

        st.subheader(f"Forecast for the 7 days from {date}")
        c = st.columns(4)
        c[0].metric("Predicted demand (P50 point)", f"{max(0, row['pred_best_regression']):,.0f} units")
        c[1].metric("P75 scenario", f"{max(0, row['pred_q75']):,.0f} units")
        c[2].metric("P90 scenario", f"{max(0, row['pred_q90']):,.0f} units")
        c[3].metric("P95 scenario", f"{max(0, row['pred_q95']):,.0f} units")
        st.caption(
            "Higher quantiles are progressively more conservative: planning to "
            "P95 means being short only about 5% of the time, at the cost of "
            "holding more stock."
        )

        st.subheader("History and prediction band")
        chart = pd.DataFrame(
            {
                "actual next-7-day demand": sub.set_index("date")["future_7d_demand"],
                "predicted (point)": sub.set_index("date")["pred_best_regression"].clip(lower=0),
                "P95": sub.set_index("date")["pred_q95"].clip(lower=0),
            }
        )
        st.line_chart(chart, height=320)

        err = (sub["pred_best_regression"] - sub["future_7d_demand"]).abs().mean()
        st.caption(f"Mean absolute error for this SKU over the test period: {err:,.1f} units")


# ================================================================ PAGE 3
elif page.startswith("3"):
    st.title("Inventory Risk")
    if guard(rec, "Predictions", "python scripts/train_models.py"):
        st.caption("Stock positions below are SIMULATED (see sim/).")
        latest = rec.sort_values("date").groupby("stock_code").tail(1).copy()
        latest["risk"] = pd.cut(
            latest["prob_best_classifier"],
            [-0.01, 0.40, 0.70, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        latest["days_to_stockout"] = np.where(
            latest["rolling_mean_7"] > 0,
            (latest["current_stock"] / latest["rolling_mean_7"]).round(1),
            np.nan,
        )

        c = st.columns(3)
        for col, lab in zip(c, ["HIGH", "MEDIUM", "LOW"]):
            col.metric(f"{lab} risk SKUs", int((latest["risk"] == lab).sum()))

        choice = st.multiselect("Show risk levels", ["HIGH", "MEDIUM", "LOW"],
                                default=["HIGH", "MEDIUM"])
        view = latest.loc[latest["risk"].isin(choice)].sort_values(
            "prob_best_classifier", ascending=False
        )
        st.dataframe(
            view[
                ["stock_code", "description", "date", "current_stock",
                 "pred_best_regression", "lead_time_days", "days_to_stockout",
                 "prob_best_classifier", "risk"]
            ].rename(
                columns={
                    "pred_best_regression": "predicted_7d_demand",
                    "prob_best_classifier": "reorder_probability",
                }
            ).round(2),
            hide_index=True, use_container_width=True, height=440,
        )


# ================================================================ PAGE 4
elif page.startswith("4"):
    st.title("Procurement Recommendation")
    if guard(rec, "Recommendations", "python scripts/run_simulation.py"):
        latest = rec.sort_values("date").groupby("stock_code").tail(1)
        ordering = latest.loc[latest["rec_reorder"]].sort_values(
            "rec_order_value", ascending=False
        )
        c = st.columns(4)
        c[0].metric("SKUs to reorder", f"{len(ordering):,}")
        c[1].metric("Total units", f"{ordering['rec_qty'].sum():,.0f}")
        c[2].metric("Order value", f"{ordering['rec_order_value'].sum():,.0f}")
        c[3].metric("Mean lead time", f"{ordering['lead_time_days'].mean():.1f} d")

        st.subheader("Purchase recommendations (latest decision date per SKU)")
        st.dataframe(
            ordering[
                ["stock_code", "description", "supplier_id", "current_stock",
                 "rec_expected_demand", "rec_safety_stock", "rec_qty",
                 "moq", "pack_size", "lead_time_days", "unit_cost",
                 "rec_order_value"]
            ].round(2),
            hide_index=True, use_container_width=True, height=420,
        )
        st.caption(
            "Supplier, MOQ, pack size, lead time and cost are SIMULATED. "
            "A human buyer must approve every order; nothing here is auto-submitted."
        )

        st.subheader("Policy comparison (test period)")
        pol = read_csv("policy_comparison.csv")
        if not pol.empty:
            st.dataframe(
                pol[["policy", "policy_label", "fill_rate", "stockout_days",
                     "mean_inventory_units", "total_cost", "cost_vs_P0_pct",
                     "fill_rate_vs_P0_pp"]],
                hide_index=True, use_container_width=True,
            )


# ================================================================ PAGE 5
elif page.startswith("5"):
    st.title("WHY?  Explaining the recommendation")
    feats_df = load_features_df()
    model = load_reg_model()
    if guard(rec, "Recommendations", "python scripts/run_simulation.py"):
        skus = sorted(rec["stock_code"].unique())
        sku = st.selectbox("Product (SKU)", skus)
        sub = rec.loc[rec["stock_code"] == sku].sort_values("date")
        date = st.select_slider(
            "Decision date",
            options=list(sub["date"].dt.date),
            value=list(sub["date"].dt.date)[-1],
        )
        row = sub.loc[sub["date"].dt.date == date].iloc[0]

        lead = int(row["lead_time_days"])
        scale = (lead + REVIEW_PERIOD) / CFG["target"]["horizon_days"]
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

        # SHAP for this single row, if the model and features are available.
        shap_top = None
        if model is not None and not feats_df.empty:
            reg_feats = regression_features(feats_df)
            m = (feats_df["stock_code"] == sku) & (
                feats_df["date"].dt.date == date
            )
            if m.any():
                X = feats_df.loc[m, reg_feats].to_numpy(dtype=np.float32)
                res = forecast_shap(model, X, reg_feats)
                shap_top = top_contributions(res, 0, X[0], k=5)

        exp = explain_decision(
            sku=sku, description=str(row["description"]), date=date,
            decision=decision, forecast_7d=f7,
            reorder_probability=float(row["prob_best_classifier"]),
            shap_top=shap_top,
            mean_daily_demand=float(row["rolling_mean_28"]),
        )

        c = st.columns(4)
        c[0].metric("Stockout risk", exp["stockout_risk"],
                    f"p = {exp['reorder_probability']:.2f}")
        c[1].metric("Predicted 7-day demand", f"{exp['predicted_7d_demand']:,.0f}")
        c[2].metric("Stock + on order", f"{exp['current_stock']:,.0f}")
        c[3].metric("Recommended order", f"{exp['recommended_qty']:,}")

        left, right = st.columns(2)
        with left:
            st.subheader("Why is demand forecast at this level?")
            if exp["demand_reasons"]:
                for r in exp["demand_reasons"]:
                    st.markdown(f"- {r.capitalize()}")
            else:
                st.info("Feature attribution unavailable for this row.")
        with right:
            st.subheader("Why this order quantity?")
            for r in exp["quantity_reasons"]:
                st.markdown(f"- {r}")

        st.subheader("Quantity decomposition (sums exactly to the recommendation)")
        dec = pd.DataFrame(
            {"component": list(exp["decomposition"]),
             "units": list(exp["decomposition"].values())}
        )
        st.dataframe(dec, hide_index=True, use_container_width=True)
        st.success(
            f"Components total {sum(exp['decomposition'].values()):,.2f} units "
            f"= recommended order of {exp['decomposition_total']:,} units."
        )

        st.subheader("What if…?")
        base = {
            "demand_7d": f7, "q50": q50, "q95": q95,
            "current_stock": float(row["inventory_position"]),
            "lead_time_days": lead, "review_period": REVIEW_PERIOD,
            "horizon": CFG["target"]["horizon_days"],
            "moq": int(row["moq"]), "pack_size": int(row["pack_size"]),
            "supplier_capacity": int(row["supplier_capacity"]),
            "unit_cost": float(row["unit_cost"]),
            "mean_daily_demand": float(row["rolling_mean_28"]),
        }
        wc1, wc2, wc3 = st.columns(3)
        new_lead = wc1.slider("Lead time (days)", 1, 40, lead)
        new_stock = wc2.slider(
            "Stock on hand + on order", 0,
            max(100, int(row["inventory_position"] * 3) + 100),
            int(row["inventory_position"]),
        )
        new_moq = wc3.slider("Supplier MOQ", 0, 500, int(row["moq"]), step=12)

        what = run_scenario(
            base,
            Scenario("custom", lead_time_days=new_lead,
                     current_stock=float(new_stock), moq=new_moq),
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Recommended order", f"{what.final_qty:,}",
                  f"{what.final_qty - decision.final_qty:+,} vs current")
        d2.metric("Expected lead-time demand", f"{what.expected_demand:,.0f}")
        d3.metric("Safety stock", f"{what.safety_stock:,.0f}")

        st.markdown("**Standard scenario panel**")
        st.dataframe(pd.DataFrame(run_whatif_table(base)), hide_index=True,
                     use_container_width=True)

        with st.expander("Full text explanation"):
            st.code(render_explanation(exp), language="text")


# ================================================================ PAGE 6
elif page.startswith("6"):
    st.title("Model Comparison")
    if summary:
        c = st.columns(2)
        c[0].success(
            f"**Best forecaster:** {summary['best_regression_model']}  \n"
            f"RMSE {summary['best_regression_test_RMSE']:.2f} · "
            f"MAE {summary['best_regression_test_MAE']:.2f} · "
            f"R² {summary['best_regression_test_R2']:.4f}"
        )
        c[1].success(
            f"**Best classifier:** {summary['best_classifier_model']}  \n"
            f"F1 {summary['best_classifier_test_f1']:.4f} · "
            f"Recall {summary['best_classifier_test_recall']:.4f} · "
            f"AUC {summary['best_classifier_test_roc_auc']:.4f}"
        )

    st.subheader("Regression — 7-day demand forecast")
    st.dataframe(read_csv("regression_model_comparison.csv"), hide_index=True,
                 use_container_width=True)

    st.subheader("Classification — reorder / stockout risk")
    st.dataframe(read_csv("classification_model_comparison.csv"), hide_index=True,
                 use_container_width=True)

    st.subheader("Polynomial degree study (Unit 1: overfitting)")
    poly = read_csv("polynomial_degree_study.csv")
    if not poly.empty:
        st.dataframe(poly, hide_index=True, use_container_width=True)
        st.line_chart(
            poly.set_index("degree")[["train_RMSE", "val_RMSE", "test_RMSE"]],
            height=280,
        )
        st.caption(
            "Training error keeps falling as the degree rises while test error "
            "explodes — textbook overfitting, measured on real data."
        )

    st.subheader("Quantile calibration")
    st.dataframe(read_csv("quantile_model_metrics.csv"), hide_index=True,
                 use_container_width=True)

    for title, fname in [
        ("Feature importance — demand forecast",
         "shap_global_importance_forecast.png"),
        ("Feature importance — reorder decision",
         "shap_global_importance_classifier.png"),
    ]:
        p = FIGS / fname
        if p.exists():
            st.subheader(title)
            st.image(str(p), use_container_width=True)
