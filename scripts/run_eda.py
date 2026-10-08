"""Stage 2a: exploratory data analysis -- figures, summary tables and insights.

Every number quoted in docs/eda_insights.md is computed here and written to
reports/eda_findings.json, so the prose and the data cannot drift apart.

Writes:
    reports/dataset_summary.csv
    reports/eda_findings.json
    reports/figures/eda_*.png
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

from xaierp.ingest.online_retail import load_panel  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402

plt.rcParams.update(
    {
        "figure.figsize": (11, 5.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
    }
)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
F: dict[str, str] = {}   # figure registry


def save(fig, name: str, figs: Path) -> None:
    path = figs / f"eda_{name}.png"
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    F[name] = str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    figs = resolve(cfg["paths"]["figures"])
    reports = resolve(cfg["paths"]["reports"])
    ensure_dir(figs)

    panel = load_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    find: dict[str, object] = {}

    # =================================================================
    # 1. Daily total demand
    # =================================================================
    daily = panel.groupby("date")["demand"].sum()
    fig, ax = plt.subplots()
    ax.plot(daily.index, daily.values, lw=0.8, color="#2b6cb0")
    ax.set_title("Total daily demand across the 200 modelled SKUs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold per day")
    save(fig, "01_daily_total_demand", figs)

    peak_day = daily.idxmax()
    find["daily_demand"] = {
        "mean": float(daily.mean()),
        "median": float(daily.median()),
        "std": float(daily.std()),
        "min": float(daily.min()),
        "max": float(daily.max()),
        "peak_date": str(peak_day.date()),
        "n_days": int(len(daily)),
    }

    # =================================================================
    # 2. Weekly and 3. monthly trend
    # =================================================================
    weekly = daily.resample("W").sum()
    fig, ax = plt.subplots()
    ax.plot(weekly.index, weekly.values, marker="o", ms=3, color="#2f855a")
    ax.set_title("Weekly total demand")
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Units sold per week")
    save(fig, "02_weekly_trend", figs)

    monthly = daily.resample("MS").sum()
    fig, ax = plt.subplots()
    ax.bar(monthly.index, monthly.values, width=20, color="#4c51bf")
    ax.set_title("Monthly total demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units sold per month")
    save(fig, "03_monthly_trend", figs)

    # Peak season: compare Sep-Nov against the rest.
    m = monthly.copy()
    m.index = pd.to_datetime(m.index)
    peak_months = m[m.index.month.isin([9, 10, 11])]
    other_months = m[~m.index.month.isin([9, 10, 11])]
    find["seasonality"] = {
        "peak_month": str(m.idxmax().date()),
        "peak_month_units": float(m.max()),
        "trough_month": str(m.idxmin().date()),
        "trough_month_units": float(m.min()),
        "peak_to_trough_ratio": float(m.max() / m.min()),
        "mean_monthly_units_sep_nov": float(peak_months.mean()),
        "mean_monthly_units_other": float(other_months.mean()),
        "q4_uplift_pct": float(
            100.0 * (peak_months.mean() - other_months.mean()) / other_months.mean()
        ),
    }

    # =================================================================
    # 4/5. Demand distribution (linear and log)
    # =================================================================
    d = panel["demand"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(d[d <= d.quantile(0.99)], bins=60, color="#dd6b20")
    axes[0].set_title("SKU-day demand distribution (below the 99th percentile)")
    axes[0].set_xlabel("Units sold in one day")
    axes[0].set_ylabel("Number of SKU-days")
    axes[1].hist(np.log1p(d), bins=60, color="#dd6b20")
    axes[1].set_title("log(1 + demand) -- the long right tail made visible")
    axes[1].set_xlabel("log(1 + units sold)")
    axes[1].set_ylabel("Number of SKU-days")
    save(fig, "04_demand_distribution", figs)

    find["demand_distribution"] = {
        "mean": float(d.mean()),
        "median": float(d.median()),
        "std": float(d.std()),
        "variance": float(d.var()),
        "skew": float(d.skew()),
        "kurtosis": float(d.kurtosis()),
        "min": float(d.min()),
        "max": float(d.max()),
        "q25": float(d.quantile(0.25)),
        "q50": float(d.quantile(0.50)),
        "q75": float(d.quantile(0.75)),
        "q90": float(d.quantile(0.90)),
        "q95": float(d.quantile(0.95)),
        "q99": float(d.quantile(0.99)),
        "pct_zero_days": float(100.0 * (d == 0).mean()),
        "mean_over_median_ratio": float(d.mean() / max(d.median(), 1e-9)),
    }

    # =================================================================
    # 6. Outliers -- boxplot by top families
    # =================================================================
    top10 = (
        panel.groupby("stock_code")["demand"].sum().sort_values(ascending=False).head(10).index
    )
    sub = panel[panel["stock_code"].isin(top10)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    data = [sub.loc[sub["stock_code"] == s, "demand"].values for s in top10]
    ax.boxplot(data, tick_labels=[str(s) for s in top10], showfliers=True,
               flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    ax.set_title("Daily demand spread and outliers -- top 10 SKUs by total units")
    ax.set_xlabel("Stock code")
    ax.set_ylabel("Units sold per day")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "05_outliers_boxplot", figs)

    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    find["outliers"] = {
        "iqr_q1": float(q1), "iqr_q3": float(q3), "iqr": float(iqr),
        "upper_fence": float(upper),
        "n_above_fence": int((d > upper).sum()),
        "pct_above_fence": float(100.0 * (d > upper).mean()),
        "max_single_sku_day": float(d.max()),
    }

    # =================================================================
    # 7. Top SKUs by revenue
    # =================================================================
    top_rev = (
        panel.groupby(["stock_code", "description"])["revenue"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [
        f"{r.stock_code} {str(r.description)[:28]}" for r in top_rev.itertuples()
    ][::-1]
    ax.barh(labels, top_rev["revenue"].values[::-1], color="#2b6cb0")
    ax.set_title("Top 15 SKUs by total revenue (modelled set)")
    ax.set_xlabel("Total revenue over the period")
    ax.set_ylabel("SKU")
    save(fig, "06_top_skus_revenue", figs)

    sku_rev = panel.groupby("stock_code")["revenue"].sum().sort_values(ascending=False)
    cum = sku_rev.cumsum() / sku_rev.sum()
    n20 = max(1, int(0.2 * len(sku_rev)))
    find["concentration"] = {
        "top_sku": str(sku_rev.index[0]),
        "top_sku_revenue": float(sku_rev.iloc[0]),
        "top_10_pct_of_revenue": float(100.0 * sku_rev.head(10).sum() / sku_rev.sum()),
        "top_20pct_skus_share_of_revenue": float(100.0 * cum.iloc[n20 - 1]),
        "n_skus": int(len(sku_rev)),
    }

    # =================================================================
    # 8. Pareto curve
    # =================================================================
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(cum) + 1), cum.values * 100, color="#c53030", lw=2)
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.axvline(n20, ls="--", color="gray", lw=1)
    ax.set_title("Revenue concentration (Pareto curve)")
    ax.set_xlabel("Number of SKUs (ranked by revenue)")
    ax.set_ylabel("Cumulative share of revenue (%)")
    save(fig, "07_pareto_revenue", figs)

    # =================================================================
    # 9. Day-of-week effect  -- the Saturday finding
    # =================================================================
    panel["dow"] = panel["date"].dt.dayofweek
    dow = panel.groupby("dow")["demand"].agg(["mean", "sum", "count"])
    present = [WEEKDAYS[i] for i in dow.index]
    fig, ax = plt.subplots()
    ax.bar(present, dow["mean"].values, color="#4c51bf")
    ax.set_title("Mean daily demand per SKU by day of week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean units per SKU per day")
    save(fig, "08_day_of_week", figs)

    all_days = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    traded = set(panel["date"].unique())
    missing_dow = pd.Series(
        [d.dayofweek for d in all_days if d not in traded]
    ).value_counts().sort_index()
    find["trading_calendar"] = {
        "calendar_days": int(len(all_days)),
        "trading_days": int(len(traded)),
        "non_trading_days": int(len(all_days) - len(traded)),
        "non_trading_by_weekday": {
            WEEKDAYS[int(i)]: int(v) for i, v in missing_dow.items()
        },
        "mean_demand_by_weekday": {
            WEEKDAYS[int(i)]: float(v) for i, v in dow["mean"].items()
        },
    }

    # =================================================================
    # 10. Correlation heatmap
    # =================================================================
    corr_cols = ["demand", "revenue", "avg_price", "n_invoices",
                 "returned_units", "median_price", "sku_rank"]
    corr = panel[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.55 else "black")
    ax.set_title("Correlation matrix (Pearson) -- panel variables")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.grid(False)
    save(fig, "09_correlation_heatmap", figs)

    find["correlation"] = {
        "demand_vs_n_invoices": float(corr.loc["demand", "n_invoices"]),
        "demand_vs_avg_price": float(corr.loc["demand", "avg_price"]),
        "demand_vs_revenue": float(corr.loc["demand", "revenue"]),
        "demand_vs_returns": float(corr.loc["demand", "returned_units"]),
        "covariance_demand_invoices": float(
            panel[["demand", "n_invoices"]].cov().iloc[0, 1]
        ),
    }

    # =================================================================
    # 11. Invoices vs demand scatter
    # =================================================================
    s = panel.sample(min(6000, len(panel)), random_state=42)
    fig, ax = plt.subplots()
    ax.scatter(s["n_invoices"], s["demand"], s=5, alpha=0.3, color="#2f855a")
    ax.set_title("Daily demand vs number of invoices containing the SKU")
    ax.set_xlabel("Invoices containing the SKU that day")
    ax.set_ylabel("Units sold that day")
    ax.set_yscale("symlog")
    save(fig, "10_invoices_vs_demand", figs)

    # =================================================================
    # 12. Price vs demand
    # =================================================================
    fig, ax = plt.subplots()
    ax.scatter(s["avg_price"], s["demand"], s=5, alpha=0.3, color="#dd6b20")
    ax.set_title("Daily demand vs average unit price (price as a promotion proxy)")
    ax.set_xlabel("Average unit price that day")
    ax.set_ylabel("Units sold that day")
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    save(fig, "11_price_vs_demand", figs)

    # =================================================================
    # 13. Zero / intermittent demand
    # =================================================================
    zero_rate = panel.groupby("stock_code")["demand"].apply(lambda x: (x == 0).mean())
    fig, ax = plt.subplots()
    ax.hist(zero_rate * 100, bins=40, color="#805ad5")
    ax.set_title("Intermittency: share of trading days with zero sales, per SKU")
    ax.set_xlabel("% of days with zero demand")
    ax.set_ylabel("Number of SKUs")
    save(fig, "12_zero_demand", figs)

    find["intermittency"] = {
        "overall_pct_zero_sku_days": float(100.0 * (panel["demand"] == 0).mean()),
        "median_sku_zero_rate_pct": float(100.0 * zero_rate.median()),
        "min_sku_zero_rate_pct": float(100.0 * zero_rate.min()),
        "max_sku_zero_rate_pct": float(100.0 * zero_rate.max()),
        "n_skus_above_50pct_zero": int((zero_rate > 0.5).sum()),
    }

    # =================================================================
    # 14/15. Rolling mean and rolling std
    # =================================================================
    roll_mean = daily.rolling(28, min_periods=28).mean()
    roll_std = daily.rolling(28, min_periods=28).std()
    fig, ax = plt.subplots()
    ax.plot(daily.index, daily.values, lw=0.5, alpha=0.35,
            color="gray", label="Daily total")
    ax.plot(roll_mean.index, roll_mean.values, lw=2,
            color="#2b6cb0", label="28-day rolling mean")
    ax.set_title("Total daily demand with 28-day rolling mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units per day")
    ax.legend()
    save(fig, "13_rolling_mean", figs)

    fig, ax = plt.subplots()
    ax.plot(roll_std.index, roll_std.values, lw=1.6, color="#c53030")
    ax.set_title("28-day rolling standard deviation of total daily demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling std of units per day")
    save(fig, "14_rolling_std", figs)

    find["volatility"] = {
        "mean_rolling_std_28": float(roll_std.mean()),
        "max_rolling_std_28": float(roll_std.max()),
        "max_rolling_std_date": str(roll_std.idxmax().date()),
        "coefficient_of_variation_daily": float(daily.std() / daily.mean()),
    }

    # =================================================================
    # 16. Target distribution (what the model predicts)
    # =================================================================
    tgt = panel.groupby("stock_code")["demand"].rolling(7).sum().dropna()
    fig, ax = plt.subplots()
    ax.hist(tgt[tgt <= tgt.quantile(0.99)], bins=60, color="#2f855a")
    for q, c in [(0.5, "#000000"), (0.9, "#dd6b20"), (0.95, "#c53030")]:
        ax.axvline(tgt.quantile(q), ls="--", color=c, lw=1.5,
                   label=f"P{int(q*100)} = {tgt.quantile(q):,.0f}")
    ax.set_title("Distribution of 7-day demand totals, with planning quantiles")
    ax.set_xlabel("Units over 7 days")
    ax.set_ylabel("Number of SKU-weeks")
    ax.legend()
    save(fig, "15_target_quantiles", figs)

    find["target_quantiles"] = {
        "mean": float(tgt.mean()),
        "P50": float(tgt.quantile(0.50)),
        "P75": float(tgt.quantile(0.75)),
        "P90": float(tgt.quantile(0.90)),
        "P95": float(tgt.quantile(0.95)),
        "P95_over_P50": float(tgt.quantile(0.95) / max(tgt.quantile(0.50), 1e-9)),
    }

    # =================================================================
    # 17. Seasonal decomposition of the weekly series
    # =================================================================
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        dec = seasonal_decompose(weekly.dropna(), model="additive", period=52
                                 if len(weekly) >= 104 else 13)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(dec.trend, color="#2b6cb0")
        axes[0].set_ylabel("Trend")
        axes[1].plot(dec.seasonal, color="#2f855a")
        axes[1].set_ylabel("Seasonal")
        axes[2].plot(dec.resid, color="#c53030", lw=0.8)
        axes[2].set_ylabel("Residual")
        axes[2].set_xlabel("Week")
        axes[0].set_title("Seasonal decomposition of weekly total demand")
        save(fig, "16_seasonal_decomposition", figs)
    except Exception as exc:  # noqa: BLE001
        print(f"  (seasonal decomposition skipped: {exc})")

    # =================================================================
    # Dataset summary table
    # =================================================================
    rows = [
        ("Rows in daily panel", f"{len(panel):,}"),
        ("SKUs modelled", f"{panel['stock_code'].nunique():,}"),
        ("Trading days", f"{panel['date'].nunique():,}"),
        ("Date range", f"{panel['date'].min().date()} to {panel['date'].max().date()}"),
        ("Total units sold", f"{panel['demand'].sum():,.0f}"),
        ("Total revenue", f"{panel['revenue'].sum():,.0f}"),
        ("Mean daily demand per SKU", f"{panel['demand'].mean():.2f}"),
        ("Median daily demand per SKU", f"{panel['demand'].median():.2f}"),
        ("Std of daily demand", f"{panel['demand'].std():.2f}"),
        ("Variance of daily demand", f"{panel['demand'].var():.2f}"),
        ("Skewness of daily demand", f"{panel['demand'].skew():.2f}"),
        ("Min / Max daily demand", f"{panel['demand'].min():.0f} / {panel['demand'].max():.0f}"),
        ("SKU-days with zero demand", f"{100.0 * (panel['demand'] == 0).mean():.1f}%"),
        ("Non-trading calendar days", f"{len(all_days) - len(traded):,}"),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    out = reports / "dataset_summary.csv"
    ensure_dir(out)
    summary.to_csv(out, index=False)

    find["figures"] = F
    (reports / "eda_findings.json").write_text(
        json.dumps(find, indent=2, default=str), encoding="utf-8"
    )

    print(f"EDA complete: {len(F)} figures -> {figs.relative_to(ROOT)}")
    for k, v in find["seasonality"].items():
        print(f"  seasonality.{k}: {v}")
    print(f"  intermittency: {find['intermittency']['overall_pct_zero_sku_days']:.1f}% zero SKU-days")
    print(f"  demand skew  : {find['demand_distribution']['skew']:.1f}")
    print(f"  saved -> {out.relative_to(ROOT)}")
    print(f"  saved -> {(reports / 'eda_findings.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
