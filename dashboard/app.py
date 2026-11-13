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
def load_segments_df() -> pd.DataFrame:
    """Unit 3 output: one row per SKU with its segment and profile."""
    p = resolve(CFG["paths"]["processed"]) / "sku_segments.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_regimes_df() -> pd.DataFrame:
    """Unit 4 output: one row per trading day with regime and Kalman level."""
    p = resolve(CFG["paths"]["processed"]) / "demand_regimes.parquet"
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
        "7 · SKU Segmentation (Unit 3)",
        "8 · Demand Regimes (Unit 4)",
        "9 · Decision Rules (Unit 5)",
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


# ================================================================ PAGE 7
elif page.startswith("7"):
    st.title("SKU Segmentation — Unit 3 (unsupervised learning)")
    seg = load_segments_df()
    if guard(seg, "SKU segments", "python scripts/run_unsupervised.py"):
        find = read_json("unit3_findings.json")
        st.caption(
            "Segments are learned from the **train period only** — the "
            f"profiles use data up to {find.get('train_end', 'the train cut-off')}, "
            "so a segment-based safety stock applied to the test period is not "
            "peeking at its own answer."
        )

        c = st.columns(4)
        c[0].metric("SKUs segmented", f"{len(seg):,}")
        c[1].metric("Segments found", f"{seg['segment'].nunique()}")
        c[2].metric("Profile features", f"{find.get('n_profile_features', '—')}")
        c[3].metric("Latent dimensions (PCA)", f"{find.get('n_latent', '—')}")

        st.subheader("Segment profiles")
        prof = read_csv("unit3_segment_profiles.csv")
        if not prof.empty:
            st.dataframe(
                prof[[
                    "segment", "segment_name", "n_skus", "mean_demand",
                    "cv_demand", "zero_rate", "adi", "q4_ratio",
                    "revenue_share_total",
                ]].round(3),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "Each segment is named after the behavioural axis on which it "
                "is most extreme relative to the others — not by fixed "
                "thresholds, which stop meaning anything if the data changes."
            )

        left, right = st.columns(2)
        with left:
            st.subheader("Segments in the PCA latent space")
            if {"pc1", "pc2"} <= set(seg.columns):
                st.scatter_chart(
                    seg, x="pc1", y="pc2", color="segment_name", height=380
                )
        with right:
            st.subheader("How many clusters? (elbow method)")
            elbow = read_csv("unit3_kmeans_elbow.csv")
            if not elbow.empty:
                st.line_chart(elbow.set_index("k")["inertia"], height=380)
                st.caption(
                    "Inertia always falls as k rises, so the minimum is never "
                    "the answer. The elbow — the point furthest from the chord "
                    "joining the ends — is at k = "
                    f"{find.get('kmeans', {}).get('k_elbow', '—')}."
                )

        st.subheader("Which clustering method, and do they agree?")
        c1, c2 = st.columns([3, 2])
        with c1:
            ev = read_csv("unit3_clustering_evaluation.csv")
            if not ev.empty:
                st.dataframe(ev.round(4), hide_index=True, use_container_width=True)
                st.caption(
                    "Silhouette and Calinski-Harabasz both reward compact "
                    "spherical clusters — exactly what K-means produces by "
                    "construction — so they cannot settle the comparison on "
                    "their own. The agreement table beside them is the check."
                )
        with c2:
            ag = read_csv("unit3_clustering_agreement.csv")
            if not ag.empty:
                st.dataframe(ag.round(3), hide_index=True, use_container_width=True)
                st.caption(
                    "Adjusted Rand Index: 1 = identical partitions, 0 = chance. "
                    "Immune to cluster relabelling."
                )

        st.subheader("Hard vs soft assignment: what K-means throws away")
        gmm = find.get("gmm", {})
        n_amb = gmm.get("n_ambiguous_skus")
        if n_amb is not None:
            st.info(
                "The Gaussian mixture gives every SKU a *responsibility* per "
                f"component instead of one label. **{n_amb} of {len(seg)} SKUs "
                "have no component above 90% responsibility** — they sit "
                "genuinely between segments. K-means reports them as "
                "confidently belonging to one, which for a stocking decision "
                "is a false certainty."
            )
        if "gmm_max_responsibility" in seg.columns:
            amb = seg.nsmallest(12, "gmm_max_responsibility")
            st.dataframe(
                amb[["stock_code", "description", "segment_name",
                     "gmm_max_responsibility", "silhouette"]].round(3),
                hide_index=True, use_container_width=True,
            )
            st.caption("The 12 least confidently assigned SKUs.")

        st.subheader("Choosing the number of latent dimensions")
        c1, c2 = st.columns(2)
        with c1:
            scree = read_csv("unit3_pca_scree.csv")
            if not scree.empty:
                st.line_chart(
                    scree.set_index("component")[
                        ["eigenvalue", "cumulative_variance_ratio"]
                    ],
                    height=300,
                )
        with c2:
            choice = find.get("pca_choice", {})
            if choice:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "rule": ["Kaiser (eigenvalue > 1)",
                                     "90% cumulative variance",
                                     "scree elbow"],
                            "components": [choice.get("kaiser"),
                                           choice.get("variance"),
                                           choice.get("elbow")],
                        }
                    ),
                    hide_index=True, use_container_width=True,
                )
                st.caption(
                    "The three published rules disagree. All three are "
                    "reported rather than quoting whichever gives a "
                    "convenient number; the pipeline uses Kaiser."
                )

        for title, fname, note in [
            ("Dendrogram — hierarchical clustering", "unit3_05_dendrogram.png",
             "Cut height determines the number of clusters. Single and average "
             "linkage score highest on dendrogram fidelity yet put 97% of SKUs "
             "in one cluster (chaining), so the cut is taken from Ward."),
            ("Factor loadings after varimax rotation",
             "unit3_02_factor_loadings.png",
             "An orthogonal rotation leaves the likelihood unchanged, so the "
             "loadings are identified only up to a rotation — varimax picks "
             "the most interpretable member of that family for free."),
            ("EM convergence and model selection", "unit3_04_gmm_em.png",
             "Left: the log-likelihood never decreases, which is EM's "
             "guarantee. Right: BIC has a genuine minimum, unlike inertia."),
        ]:
            p = FIGS / fname
            if p.exists():
                st.subheader(title)
                st.image(str(p), use_container_width=True)
                st.caption(note)

        with st.expander("Browse every SKU and its segment"):
            cols = [c for c in [
                "stock_code", "description", "segment", "segment_name",
                "mean_demand", "cv_demand", "zero_rate", "adi", "q4_ratio",
                "silhouette", "gmm_max_responsibility",
            ] if c in seg.columns]
            st.dataframe(seg[cols].round(3), hide_index=True,
                         use_container_width=True, height=420)


# ================================================================ PAGE 8
elif page.startswith("8"):
    st.title("Demand Regimes — Unit 4 (sequential models)")
    reg_df = load_regimes_df()
    if guard(reg_df, "Demand regimes", "python scripts/run_sequential.py"):
        find = read_json("unit4_findings.json")
        hmm_info = find.get("hmm", {})
        lds_info = find.get("lds", {})

        st.caption(
            "A hidden Markov model over the business-wide daily series. The "
            "regime is never recorded in the data — it is inferred from how "
            "demand behaves, and it is what should set safety stock."
        )

        c = st.columns(4)
        c[0].metric("Trading days", f"{len(reg_df):,}")
        c[1].metric("Regimes (HMM)", f"{hmm_info.get('n_regimes', '—')}")
        c[2].metric("Baum-Welch iterations", f"{hmm_info.get('n_iter', '—')}")
        gain = lds_info.get("rmse_improvement_pct")
        c[3].metric(
            "Kalman vs naive RW",
            f"{gain:+.1f}%" if isinstance(gain, (int, float)) else "—",
            help="Held-out one-step RMSE improvement over a random walk",
        )

        st.subheader("The three learned regimes")
        regimes = read_csv("unit4_hmm_regimes.csv")
        if not regimes.empty:
            st.dataframe(
                regimes[[
                    "label", "mean_demand_units", "std_log1p",
                    "self_transition", "expected_duration_days",
                    "stationary_prob",
                ]].round(3),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "A self-transition near 1 means the regime persists: these are "
                "genuine multi-week states, not day-to-day noise. "
                "Expected duration is 1/(1 - self_transition)."
            )

        st.subheader("Why this matters for stocking")
        mult = read_csv("unit4_regime_safety_multipliers.csv")
        if not mult.empty:
            st.dataframe(mult.round(3), hide_index=True, use_container_width=True)
            di = find.get("decision_integration", {})
            spread = di.get("multiplier_spread")
            if spread:
                st.warning(
                    f"Demand volatility differs **{spread:.1f}x** between the "
                    "calmest and busiest regime. A single pooled safety stock "
                    "is therefore simultaneously too small during SURGE and "
                    "oversized during QUIET — the regime, not the average, is "
                    "what a buffer should be sized against."
                )
            st.caption(
                "Computed from the **filtered** posterior p(z_t | x_1:t), not "
                "the smoothed one. Smoothing uses future observations, which a "
                "buyer standing on day t does not have."
            )

        st.subheader("Regime timeline")
        if "viterbi_regime_name" in reg_df.columns:
            chart_df = reg_df[["date", "total_demand", "viterbi_regime_name"]].copy()
            st.scatter_chart(
                chart_df, x="date", y="total_demand",
                color="viterbi_regime_name", height=360,
            )
            st.caption("Viterbi (MAP) decoding of the whole sequence.")

        st.subheader("Kalman filter: estimating the true demand level")
        lvl_cols = [c for c in ["kalman_filtered_level", "kalman_smoothed_level",
                                "log1p_demand"] if c in reg_df.columns]
        if lvl_cols:
            st.line_chart(reg_df.set_index("date")[lvl_cols], height=340)
            if lds_info:
                var_cut = 100 * (
                    1 - lds_info.get("mean_smoothed_var", 1)
                    / max(lds_info.get("mean_filtered_var", 1), 1e-12)
                )
                st.caption(
                    f"Fitted by maximum likelihood: q = {lds_info.get('q', 0):.5f}, "
                    f"r = {lds_info.get('r', 0):.5f}, giving a steady-state "
                    f"Kalman gain of {lds_info.get('steady_state_gain', 0):.3f}. "
                    "The gain is also the alpha of the equivalent exponential "
                    "smoother — so the smoothing constant is estimated, not "
                    f"guessed. Smoothing cut the mean level variance by "
                    f"{var_cut:.0f}%, but it uses future data and so is for "
                    "retrospective analysis only."
                )

        st.subheader("Two decoders, two different answers")
        dec = read_csv("unit4_decoder_comparison.csv")
        if not dec.empty:
            d = dec.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Days they disagree",
                      f"{int(d['n_disagreements'])}/{int(d['n_timesteps'])}",
                      f"{d['disagreement_rate']:.1%}")
            c2.metric("Viterbi regime switches", int(d["viterbi_n_switches"]))
            c3.metric("Posterior regime switches", int(d["posterior_n_switches"]))
            st.caption(
                "Posterior decoding takes the most likely state at each day "
                "independently and never consults the transition matrix, so it "
                "switches regime more often and can even emit a sequence the "
                "model considers impossible. Viterbi maximises the probability "
                "of the path as a whole and is always a legal sequence — here "
                f"by {d['logprob_advantage_viterbi']:.1f} log-units."
            )

        st.subheader("Markov model of demand states")
        c1, c2 = st.columns(2)
        with c1:
            tm = read_csv("unit4_markov_transition_matrix.csv")
            if not tm.empty:
                st.dataframe(tm.round(3), hide_index=True,
                             use_container_width=True)
                st.caption("Each row sums to 1 — a probability distribution.")
        with c2:
            pers = read_csv("unit4_markov_persistence.csv")
            if not pers.empty:
                st.dataframe(pers.round(3), hide_index=True,
                             use_container_width=True)
                mk = find.get("markov", {})
                pval = mk.get("independence_pvalue")
                if pval is not None:
                    st.caption(
                        "The buckets are equal-sized quantiles, so a uniform "
                        "stationary distribution proves nothing. The test is "
                        "whether the diagonal beats independence: chi-square "
                        f"= {mk.get('independence_chi2', 0):.1f}, p = {pval:.2e} "
                        "— busy days really do cluster."
                    )

        with st.expander("Scaling factors: the underflow they prevent"):
            sc = find.get("scaling", {})
            und = read_csv("unit4_scaling_underflow.csv")
            if sc:
                st.markdown(
                    "- At the real sequence length T = "
                    f"{sc.get('real_sequence_length')} the naive product is about "
                    "1e-217 — small, but still **finite**, so this series does "
                    "not underflow.\n"
                    "- Tiling the series to find the breaking point: the "
                    "unscaled recursion survives T = "
                    f"{sc.get('unscaled_survives_to_T')} and collapses to -inf "
                    f"by T = {sc.get('unscaled_underflow_at_T')}.\n"
                    "- The scaled recursion is still finite at T = "
                    f"{sc.get('scaled_still_finite_at_T')}, because "
                    "log p(x) = sum of log c_t is a **sum** of logs and cannot "
                    "underflow."
                )
            if not und.empty:
                st.line_chart(und.set_index("T")["loglik_scaled"], height=260)

        with st.expander("Sum-product == forward-backward (verification)"):
            sp = read_csv("unit4_sum_product_verification.csv")
            if not sp.empty:
                st.dataframe(sp.T.rename(columns={0: "value"}),
                             use_container_width=True)
                st.caption(
                    "Every difference is at machine precision: forward-backward "
                    "really is sum-product on an HMM chain, and Viterbi really "
                    "is its max-product analogue. These are not claims taken on "
                    "trust — they are computed both ways and compared."
                )

        for title, fname in [
            ("Autocorrelation and the raw series", "unit4_01_sequential.png"),
            ("Scaling factors and EM convergence", "unit4_02_scaling_em.png"),
            ("Kalman level and decoded regimes", "unit4_03_kalman_regimes.png"),
            ("Forward-backward regime posteriors",
             "unit4_04_regime_posteriors.png"),
        ]:
            p = FIGS / fname
            if p.exists():
                st.subheader(title)
                st.image(str(p), use_container_width=True)

        with st.expander("Browse the daily regime table"):
            cols = [c for c in [
                "date", "total_demand", "markov_state_name",
                "viterbi_regime_name", "filtered_regime_name",
                "kalman_filtered_level", "kalman_smoothed_level", "split",
            ] if c in reg_df.columns]
            st.dataframe(reg_df[cols].round(3), hide_index=True,
                         use_container_width=True, height=420)


# ================================================================ PAGE 9
elif page.startswith("9"):
    st.title("Decision Rules — Unit 5 (CART)")
    find = read_json("unit5_findings.json")
    rules_path = REPORTS / "unit5_tree_rules.txt"

    if not find:
        st.error("CART results not found. Run `python scripts/run_cart.py` first.")
    else:
        st.caption(
            "Every other classifier in this project answers \"reorder?\" with a "
            "probability and needs SHAP to explain itself. A pruned tree "
            "answers with a **rule** — one a buyer can read, check against "
            "their own judgement, and overrule."
        )

        pruning = find.get("pruning", {})
        readable = find.get("readable_tree", {})
        comp = find.get("comparison", {})

        c = st.columns(4)
        c[0].metric("Leaves after pruning",
                    f"{pruning.get('pruned_leaves', '—')}",
                    f"from {pruning.get('full_leaves', '—')}")
        c[1].metric("Test F1 (pruned)",
                    f"{find.get('test_pruned', {}).get('f1', 0):.4f}")
        c[2].metric("Rank vs all classifiers",
                    f"{comp.get('cart_rank', '—')} of {comp.get('n_models', '—')}",
                    f"{comp.get('f1_gap_to_best', 0):+.4f} F1 vs best")
        c[3].metric("Features actually used",
                    f"{find.get('n_features_used', '—')} of {find.get('n_features', '—')}")

        # ---------------------------------------------------- the headline
        one = find.get("one_rule_baseline", {})
        if one:
            st.subheader("The finding that matters")
            st.warning(
                f"The tree put **{find.get('top_features', [{}])[0].get('importance', 0):.1%} "
                f"of its importance on a single feature**, so that claim was "
                f"tested directly. A single threshold —\n\n"
                f"### `IF {one.get('feature')} ≤ {one.get('threshold', 0):.2f} THEN reorder`\n\n"
                f"— scores **test F1 {one.get('test_f1', 0):.4f}**, against "
                f"{comp.get('best_test_f1', 0):.4f} for the best of eight "
                f"models ({one.get('f1_vs_best_model', 0):+.4f}). The threshold "
                f"was chosen on the validation split, the same discipline every "
                f"other model got."
            )
            st.markdown(
                f"""
**Why.** The simulator defines the label as

> `{one.get('label_definition')}`

and `{one.get('feature')}` is

> `{one.get('feature_definition')}`

— the leakage-free proxy for that same comparison. The feature is **not**
leaking (it uses only trailing data), but it already contains the decision.
The residual error is *demand-forecast* error, which no classifier can remove
because the forecast is baked into the feature. That is why all eight models
cluster near F1 0.75–0.78: they share a ceiling set by the task, not by the
algorithm.

**This is the argument for interpretable models, made concrete.** LightGBM
reaches the same ceiling and conceals why. CART makes it visible in one split
— and the finding is a property of the *task*, not of the model.
"""
            )

        # ---------------------------------------------------------- rules
        st.subheader("The tree a buyer would actually be shown")
        if readable:
            st.caption(
                f"{readable.get('n_leaves')} leaves, depth "
                f"{readable.get('max_depth')}, test F1 "
                f"{readable.get('test_f1', 0):.4f} "
                f"({readable.get('f1_cost_vs_tuned', 0):+.4f} against the "
                f"tuned tree — readability is close to free here)."
            )
        tree_txt = REPORTS / "unit5_tree_text.txt"
        if tree_txt.exists():
            st.code(tree_txt.read_text(encoding="utf-8"), language="text")

        st.subheader("Worked decisions from the test period")
        ex = read_csv("unit5_rule_examples.csv")
        if not ex.empty:
            for _, r in ex.iterrows():
                hit = r["actual_reorder"] == r["predicted_reorder"]
                with st.expander(
                    f"{'✓' if hit else '✗'} SKU {r['stock_code']} on {r['date']} "
                    f"— predicted {'REORDER' if r['predicted_reorder'] else 'no order'}, "
                    f"actual {'REORDER' if r['actual_reorder'] else 'no order'}"
                ):
                    st.code(str(r["rule"]).replace(" AND ", "\n   AND ")
                            .replace("THEN", "\nTHEN"), language="text")

        # -------------------------------------------------------- pruning
        st.subheader("Cost-complexity pruning")
        st.markdown(
            r"""
What makes this CART rather than just a decision tree. Instead of an
arbitrary depth cap, price each leaf and minimise

$$R_\alpha(T) = R(T) + \alpha\,|\text{leaves}(T)|$$

As $\alpha$ rises, whole subtrees stop paying for themselves. The sequence of
subtrees is **finite and nested**, so every tree worth considering can be
enumerated and $\alpha$ chosen on a validation set rather than guessed.
"""
        )
        pp = read_csv("unit5_pruning_path.csv")
        if not pp.empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.line_chart(
                    pp.set_index("n_leaves")[["train_f1", "val_f1"]], height=320
                )
                st.caption(
                    "Training F1 keeps rising with more leaves; validation F1 "
                    "does not. The gap between them is what pruning buys back."
                )
            with c2:
                st.dataframe(
                    pp[["ccp_alpha", "n_leaves", "train_f1", "val_f1"]].round(5),
                    hide_index=True, use_container_width=True, height=320,
                )
            st.success(
                f"Pruning removed "
                f"{pruning.get('full_leaves', 0) - pruning.get('pruned_leaves', 0)} "
                f"of {pruning.get('full_leaves', 0)} leaves and narrowed the "
                f"train/validation gap from "
                f"{pruning.get('full_train_f1', 0) - pruning.get('full_val_f1', 0):+.4f} "
                f"to "
                f"{pruning.get('pruned_train_f1', 0) - pruning.get('pruned_val_f1', 0):+.4f}"
                f" — and test F1 went *up*, from "
                f"{find.get('test_full', {}).get('f1', 0):.4f} to "
                f"{find.get('test_pruned', {}).get('f1', 0):.4f}."
            )

        # ----------------------------------------------------- comparisons
        st.subheader("Against every other classifier")
        mc = read_csv("unit5_model_comparison.csv")
        if not mc.empty:
            st.dataframe(mc.round(4), hide_index=True, use_container_width=True)
            st.caption(
                "A 16-leaf tree you can read on one page lands within "
                f"{abs(comp.get('f1_gap_to_best', 0)):.4f} F1 of a "
                "gradient-boosted ensemble of hundreds of trees."
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gini vs entropy")
            cc = read_csv("unit5_criterion_comparison.csv")
            if not cc.empty:
                st.dataframe(
                    cc[["criterion", "n_leaves", "fit_seconds", "f1", "roc_auc"]].round(4),
                    hide_index=True, use_container_width=True,
                )
                st.caption(
                    f"They pick almost the same splits (F1 differs by "
                    f"{find.get('criterion', {}).get('f1_difference', 0):.4f}); "
                    "Gini is faster because it needs no logarithm in the inner "
                    "loop. Measured, not assumed."
                )
        with c2:
            st.subheader("Regression tree (the 'R' in CART)")
            rt = read_csv("unit5_regression_tree.csv")
            if not rt.empty:
                st.dataframe(rt[["max_depth", "n_leaves", "RMSE", "R2"]].round(4),
                             hide_index=True, use_container_width=True)
                st.caption(
                    "Reported because it is true: a single regression tree gets "
                    "*worse* with depth here, and its best R² sits well below "
                    "Ridge's 0.510. Trees are high-variance; this one is here "
                    "to be read, not to win."
                )

        st.subheader("What the pruned tree splits on")
        fi = read_csv("unit5_feature_importance.csv")
        if not fi.empty:
            st.dataframe(fi.round(4), hide_index=True, use_container_width=True)
            st.caption(
                "Impurity-decrease importance. Known bias worth stating: it "
                "favours high-cardinality continuous features, which offer more "
                "candidate thresholds. It describes *this tree*, and is not a "
                "causal claim."
            )

        for title, fname in [
            ("Cost-complexity pruning", "unit5_01_pruning.png"),
            ("Depth study", "unit5_02_depth_study.png"),
            ("Feature importance", "unit5_03_feature_importance.png"),
        ]:
            p = FIGS / fname
            if p.exists():
                st.subheader(title)
                st.image(str(p), use_container_width=True)

        if rules_path.exists():
            with st.expander("Full tree and all worked examples (text)"):
                st.code(rules_path.read_text(encoding="utf-8"), language="text")
