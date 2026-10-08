# EDA Insights

Every number below is computed by `scripts/run_eda.py` and stored in
`reports/eda_findings.json`. Figures are in `reports/figures/`. Nothing here is
estimated by eye.

Panel under analysis: **200 SKUs × 604 trading days = 120,800 SKU-days**,
2010-01-12 to 2011-12-09.

---

## 1. Daily total demand — `eda_01_daily_total_demand.png`

**Observation.** Total daily demand across the modelled SKUs averages
**5,725 units** (median 4,921, std 3,600), ranging from **589** to **31,228**
units. The maximum occurs on **2010-12-07**.

**Interpretation.** The coefficient of variation is **0.63** — day-to-day
variation is more than half the mean level. The series is spiky rather than
smooth, with a pronounced run-up into December.

**Business implication.** A replenishment rule tuned to the average day will be
wrong most days. This is the single clearest argument for carrying safety stock
rather than ordering to the mean.

---

## 2–3. Weekly and monthly trend — `eda_02_weekly_trend.png`, `eda_03_monthly_trend.png`

**Observation.** The peak month is **November 2010 at 258,366 units**; the
trough is December 2011 at 67,127 units — but that month is truncated (the
dataset ends 2011-12-09), so the honest seasonal comparison is between
Sep–Nov (**197,135 units/month** on average) and all other months
(**119,746 units/month**): a **+64.6% uplift**.

**Interpretation.** This is a UK giftware wholesaler. Its customers are
retailers stocking up before Christmas, so the business peaks in
September–November, roughly a month ahead of consumer demand.

**Business implication.** Lead times of up to 21 days mean the buying decision
for the November peak must be taken in October. A model trained only on
off-peak behaviour will under-order at exactly the moment errors are most
expensive — which is why the test split deliberately contains the Q4 run-up.

---

## 4. Demand distribution — `eda_04_demand_distribution.png`

**Observation.** Per-SKU daily demand has mean **28.6** but median **6.0** — a
mean/median ratio of **4.77**. Skewness is **24.6** and kurtosis **1,416**.
Quartiles: P25 = 0, P50 = 6, P75 = 25, P90 = 69, P95 = 122, P99 = 326,
max **9,480**.

**Interpretation.** This is nowhere near Normal. The distribution is a spike at
zero with an extremely long right tail.

**Business implication.** The textbook safety-stock formula `z·σ·√L` assumes
Normality. On a distribution this skewed it systematically misprices the tail,
which is the direct motivation for the quantile-regression buffer used in
policy P4.

---

## 5. Outliers — `eda_05_outliers_boxplot.png`

**Observation.** With Q1 = 0 and Q3 = 25, the IQR is 25 and the upper Tukey
fence sits at **62.5 units**. **13,110 SKU-days (10.9%)** lie above it. The
largest single SKU-day is **9,480 units**.

**Interpretation.** At 10.9%, these are not errors — they are the wholesale
bulk orders that constitute the business. Deleting them would remove the
revenue the company actually runs on.

**Business implication.** They are kept. Instead, a robust regressor (Huber)
is included to show the alternative treatment, and squared-error metrics are
reported alongside MAE so the tail's influence is visible rather than hidden.
Only 7 lines above 10,000 units were removed during cleaning, as data-entry
corrections.

---

## 6–7. Revenue concentration — `eda_06_top_skus_revenue.png`, `eda_07_pareto_revenue.png`

**Observation.** The top SKU (**22423**) alone generates **£344,069**. The top
10 SKUs account for **18.2%** of modelled revenue; the top 20% of SKUs account
for **39.6%**.

**Interpretation.** Concentration is real but milder than the classic 80/20
rule — this catalogue is broader than a Pareto assumption would suggest.

**Business implication.** A pure ABC policy that only manages the top decile
would leave 82% of revenue on autopilot. That justifies modelling all 200 SKUs
individually rather than hand-managing a shortlist.

---

## 8. Day-of-week effect — `eda_08_day_of_week.png`

**Observation.** Mean units per SKU per day: Mon 30.1, Tue 32.6, Wed 29.9,
**Thu 36.9**, Fri 26.8, Sun 15.0, **Sat 7.4**. Of 739 calendar days in the
span, **135 are non-trading**, and **104 of those are Saturdays**.

**Interpretation.** The business is effectively closed on Saturdays — the few
remaining Saturday records are residual activity, not a trading day. Thursday
is the busiest day; Sunday runs at about half a weekday.

**Business implication.** This is the most consequential data-handling decision
in the project. Zero-filling 104 Saturdays would have injected ~20,800 fake
zero-demand observations into a 120,800-row panel (17%), dragging every
forecast down and corrupting the intermittency statistics. The panel is built
on trading days only.

---

## 9. Correlation and covariance — `eda_09_correlation_heatmap.png`

**Observation.** Demand correlates **+0.64** with revenue, **+0.38** with the
number of invoices containing the SKU, **−0.13** with average unit price and
**+0.08** with returned units. Cov(demand, invoices) = **89.19**.

**Interpretation.** The demand–revenue correlation is well below 1 because
price varies across SKUs. The negative price correlation is weak but in the
expected direction — cheaper items move in larger quantities. Returns carry
almost no same-day signal.

**Business implication.** Price is a usable but weak feature, which is exactly
how it is treated: a lagged discount proxy rather than a headline driver. The
invoice-count relationship confirms demand is driven by *breadth* of orders,
not single large buyers.

---

## 10–11. Scatter relationships — `eda_10_invoices_vs_demand.png`, `eda_11_price_vs_demand.png`

**Observation.** Demand rises with invoice count but with a widening spread;
the price relationship is a diffuse negative cloud over three orders of
magnitude of price.

**Interpretation.** Neither relationship is tight enough to forecast from
alone. Both carry real but modest information.

**Business implication.** Supports the modelling choice: autoregressive demand
history dominates, and price/invoice features contribute at the margin. The
SHAP results later confirm this — the top five forecast drivers are all
rolling demand statistics.

---

## 12. Intermittency — `eda_12_zero_demand.png`

**Observation.** **31.7%** of all SKU-days have zero demand. The median SKU is
idle **28.1%** of days; the range runs from **0.17%** to **77.6%**, and
**38 of 200 SKUs** are idle on more than half of trading days.

**Interpretation.** This is genuinely intermittent demand, even after
restricting to the busiest 200 SKUs. Many of these zeros are real commercial
information, not missing data — the shop was open and nobody bought the item.

**Business implication.** Intermittency is why `zero_rate_7/14/28` are explicit
features, and why RMSLE is reported alongside RMSE. It is also the main reason
the gradient-boosted model struggles here: tree splits fragment an already
zero-inflated target.

---

## 13–14. Rolling statistics — `eda_13_rolling_mean.png`, `eda_14_rolling_std.png`

**Observation.** The 28-day rolling standard deviation of total daily demand
averages **2,894 units** and peaks at **8,123** on **2011-01-12**.

**Interpretation.** Volatility is not constant — it spikes around the
post-Christmas transition. Demand is heteroscedastic.

**Business implication.** A single fixed safety-stock level is wrong for most
of the year. Buffers must scale with recent volatility, which is why
`rolling_std_7/14/28` are features and why the quantile models (which learn
the conditional spread) outperform a static buffer.

---

## 15. The forecasting target — `eda_15_target_quantiles.png`

**Observation.** Seven-day demand totals have mean **199.5** units, with
P50 = **90**, P75 = **248**, P90 = **497**, P95 = **735**. P95 is
**8.2×** P50.

**Interpretation.** The gap between the median and the 95th percentile is
enormous. Planning to the median leaves you short in half of all weeks.

**Business implication.** This single ratio is the quantitative case for the
whole uncertainty layer. Moving from a P50 plan to a P95 plan means holding
roughly **8× the stock** for those weeks — an expensive decision that must be
made deliberately, per SKU, rather than by a blanket rule. The dashboard
surfaces P50/P75/P90/P95 side by side for exactly this reason.

---

## 16. Seasonal decomposition — `eda_16_seasonal_decomposition.png`

**Observation.** Additive decomposition of the weekly series separates a
rising trend through 2010, a clear annual seasonal component peaking in
Q4, and a residual that is largest around the peak.

**Interpretation.** Trend and seasonality are both real and both recoverable
from calendar features; the residual's concentration at the peak shows that
the hardest weeks to predict are the ones that matter most commercially.

**Business implication.** Calendar features (`month`, `week_of_year`,
`is_q4_peak`) are justified as inputs, but they are not sufficient — the
residual is where the demand-history features earn their place.

---

## Summary of what the EDA determined about the modelling

1. **Trading-day calendar, not calendar days** — 135 non-trading days excluded.
2. **Keep the outliers** — 10.9% above the Tukey fence is the business, not noise.
3. **Do not assume Normality** — skew 24.6 rules out `z·σ·√L` as the only buffer.
4. **Model intermittency explicitly** — 31.7% zero days needs `zero_rate_*` features.
5. **Buffers must be dynamic** — volatility ranges over a factor of ~3 across the year.
6. **Test on the Q4 peak** — the chronological split puts the hardest period in test.
