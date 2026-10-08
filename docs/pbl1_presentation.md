# PBL-1 Presentation — 14 slides

Each slide: content, the visual to show, and a 30–60 second speaking script.
Total speaking time ≈ 11 minutes, leaving room for questions.

All figures referenced are in `reports/figures/`.

---

## Slide 1 — Title

**Content**
> **Explainable AI-Driven Decision Support for Predictive Inventory and
> Procurement Optimization in ERP Systems**
> PBL-1 · Machine Learning · Units 1 & 2
> Dataset: UCI Online Retail II (1,067,371 real transactions)

**Visual:** the pipeline diagram (Data → EDA → Regression/Classification →
Comparison → XAI → Procurement → Dashboard).

**Script (30s)**
"My project builds a decision-support system that tells a retailer three
things: how much of a product will sell, whether they are at risk of running
out, and how much to order — and crucially, *why*. It is built on just over a
million real retail transactions, and every claim I make today comes from code
that runs end to end."

---

## Slide 2 — The problem

**Content**
- Stockouts lose sales; overstock ties up cash
- The controlling decision: *how much to order, and when*
- Baseline in our simulation: **79.9% fill rate** — 1 in 5 units unavailable
- A forecast alone is not a decision

**Visual:** `policy_cost_vs_fill_rate.png` (just the P0 bar, or the full chart).

**Script (40s)**
"Inventory is a two-sided problem. Order too little and you lose the sale;
order too much and your cash sits on a shelf. In our environment the naive
moving-average rule that most small retailers actually use achieves only a
79.9% fill rate — one unit in five that a customer wanted wasn't there. The
key insight driving this project is that a forecast by itself changes nothing.
You have to turn it into a decision."

---

## Slide 3 — Dataset

**Content**
| | |
|---|---|
| Source | UCI Online Retail II, CC BY 4.0 |
| Rows | 1,067,371 invoice lines |
| Period | Dec 2009 – Dec 2011 |
| After cleaning | 1,024,819 (96.01% retained) |
| Modelled panel | 200 SKUs × 604 trading days = 120,800 |

**Honesty note (keep this on the slide):** sales are real; **stock, suppliers,
lead times, MOQ and all costs are SIMULATED** — the dataset has none.

**Visual:** cleaning funnel (the 7 steps and rows removed).

**Script (45s)**
"I used UCI Online Retail II — real transactions from a UK giftware wholesaler.
I chose it over the richer Kaggle Favorita dataset for one reason: it needs no
credentials, so this whole project reproduces from a single command. But I need
to be upfront about something. This is a *sales* log. It contains no stock
levels, no suppliers, no lead times and no costs — and you cannot make a
purchasing decision without those. So I simulate them, from an explicit config
file with a fixed seed, and I label them as simulated everywhere. I will not
present a simulated number as a measurement."

---

## Slide 4 — EDA: the finding that changed everything

**Content**
- Invoices by weekday: Mon 6,245 · Thu 8,167 · **Sat 30**
- 104 of 739 calendar days are non-trading Saturdays
- Zero-filling them would have injected **~17% fake zero-demand rows**

**Visual:** `eda_08_day_of_week.png`

**Script (45s)**
"This is my favourite finding because it nearly broke the project. When I
plotted invoices by day of week, Saturday had thirty. Not thirty thousand —
thirty, against eight thousand on a Thursday. This business doesn't trade on
Saturdays. If I had built a calendar-day panel and filled those with zeros, I
would have injected about seventeen percent fake zero-demand observations and
dragged every single forecast downward. So the panel is built on trading days
only — 604 of 739."

---

## Slide 5 — EDA: why uncertainty is the whole story

**Content**
- Demand skewness **24.6**, mean 28.6 vs median 6.0
- **31.7%** of SKU-days have zero demand
- Q4 uplift **+64.6%** (Sep–Nov vs rest)
- **P95 of 7-day demand is 8.2× P50**

**Visual:** `eda_15_target_quantiles.png`

**Script (50s)**
"Four numbers that shaped every modelling choice. Demand is skewed at 24.6 —
nothing like a bell curve, which rules out the textbook safety-stock formula.
Nearly a third of SKU-days have zero sales, so this is genuinely intermittent
demand. There's a 65% Q4 uplift. But the number that matters most is the last
one: the 95th percentile of weekly demand is **8.2 times** the median. Plan to
the median and you're short half the time. That single ratio is the business
case for everything I built after this point."

---

## Slide 6 — Methodology & leakage control

**Content**
- Row `(sku, t)` = decision at **start of day t**
- Features: data from `t−1` and earlier only (every rolling stat on `.shift(1)`)
- Target: demand over `t … t+6`
- **Chronological** split — train → 2011-05-15, val → 2011-08-23, test → 2011-12-02
- 8 dedicated leakage tests

**Visual:** timeline diagram of the split, with the Q4 peak inside the test window.

**Script (50s)**
"Time-series ML is easy to get wrong, so let me be precise. A row is a decision
taken at the start of day t. Features may only use yesterday and earlier —
every rolling statistic is computed on a shifted series. The target is the next
seven days. I split chronologically, never randomly, because consecutive days
share almost all their rolling history; a random split would leak the answer
and inflate every score. And I don't just assert this — I test it on a
monotonically increasing series where any leak is arithmetically visible."

---

## Slide 7 — Unit 1: polynomial curve fitting and overfitting

**Content**
| Degree | Terms | Train RMSE | Test RMSE | Test R² |
|---|---|---|---|---|
| 1 | 4 | 285.77 | 291.05 | 0.50 |
| 2 | 14 | 282.55 | 314.83 | 0.42 |
| 3 | 34 | 281.00 | 393.52 | 0.09 |
| 4 | 69 | 278.42 | **941.97** | **−4.23** |

**Visual:** `model_polynomial_overfitting.png`

**Script (45s)**
"This is the Unit 1 bias–variance trade-off, measured on real data rather than
drawn as a cartoon. As I raise the polynomial degree, training error falls
every single time — 285 down to 278. But test error goes from 291 to 942, and
R² goes *negative* at degree four, meaning the model is worse than just
predicting the average. Same data, same penalty, only capacity changed. That's
overfitting, and you can watch it happen."

---

## Slide 8 — Unit 2: regression results

**Content**
| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Ridge (L2), α=100** | 136.4 | **288.5** | **0.510** |
| Bayesian Ridge | 136.3 | 288.7 | 0.509 |
| Linear (OLS) | 134.2 | 289.0 | 0.508 |
| Huber (robust) | **124.0** | 297.4 | 0.479 |
| *28-day MA baseline* | 136.7 | 298.1 | 0.477 |
| LightGBM | 142.3 | 311.4 | 0.429 |

**Visual:** `model_regression_rmse.png`

**Script (55s)**
"Here's the regression comparison, and there are two honest things to say.
First: Ridge wins, and MLE connects straight to this — minimising squared error
*is* maximising the Gaussian likelihood, so OLS already is a maximum-likelihood
estimator. Ridge adds an L2 penalty, which I needed because my design matrix is
badly conditioned. Second, the uncomfortable part: **LightGBM came sixth of
seven, behind a moving average.** I'm reporting that rather than hiding it.
Three reasons — zero inflation fragments tree splits, the test period is the Q4
peak and trees can't extrapolate past their training range, and my rolling
features already encode what a tree would have to learn."

---

## Slide 9 — Unit 2: classification and the kernel trick

**Content**
| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **LightGBM** | 0.744 | 0.816 | **0.778** | **0.921** |
| SVM linear | 0.691 | 0.816 | 0.748 | 0.899 |
| Logistic | 0.684 | 0.826 | 0.748 | 0.898 |
| SVM RBF | 0.616 | 0.859 | 0.718 | 0.891 |
| LDA | **0.781** | 0.427 | 0.552 | 0.850 |
| Gaussian NB | 0.374 | **0.969** | 0.540 | 0.769 |

**Visual:** `model_roc_curves.png` and `model_confusion_matrix.png`

**Script (60s)**
"Six classifiers, each mapped to a Unit 2 topic. Two results are worth
dwelling on. **Generative versus discriminative:** Naive Bayes models the
feature distribution and applies Bayes rule — it gets 97% recall but only 37%
precision, because its independence assumption is badly violated by my
correlated rolling means, so it flags almost everything. Logistic regression
models the probability directly and is far better calibrated. **The kernel
trick:** linear SVM scores 0.748, RBF scores 0.718 — same C, same data, only
the kernel differs. The infinite-dimensional feature map *lost*. That tells me
the boundary is close to linear here, and it's a negative result I'm happy to
report."

---

## Slide 10 — Uncertainty: learned quantiles

**Content**
| Target | Coverage | Error |
|---|---|---|
| P50 | 0.550 | +0.050 |
| P75 | 0.758 | +0.008 |
| P90 | 0.892 | −0.008 |
| P95 | 0.929 | −0.021 |

Trained on pinball loss · no Normality assumption

**Visual:** `model_quantile_calibration.png`

**Script (40s)**
"Because demand is skewed at 24.6, the classical `z·sigma·root-L` safety stock
formula misprices the tail. So instead I train LightGBM models directly on the
pinball loss to predict quantiles. The calibration check asks: does my P90
model actually cover 90% of outcomes? It covers 89.2%. Every level is within
2.1 percentage points. That gives me a safety buffer learned from the data's
actual tail rather than assumed from a bell curve."

---

## Slide 11 — From prediction to decision: the policy ladder

**Content**
| Policy | Fill rate | Stockout days | Cost vs P0 |
|---|---|---|---|
| P0 moving average | 79.9% | 2,027 | — |
| P1 classical reorder point | 88.7% | 966 | −2.9% |
| **P2 ML forecast, no buffer** | **79.2%** | 1,570 | −0.8% |
| P3 ML forecast + classical buffer | 89.7% | 767 | −1.6% |
| **P4 ML + quantile buffer** | **93.9%** | **496** | +4.6% |

**Visual:** `policy_cost_vs_fill_rate.png`

**Script (60s)**
"This is the most important slide in the deck. Look at P2: the ML point
forecast with no safety buffer. Its fill rate is **79.2% — worse than the naive
baseline.** A better forecast, on its own, did not help at all. What helps is
explicitly modelling uncertainty. P1 adds a classical buffer with no machine
learning and gains nearly nine points. P4 combines the ML forecast with the
learned quantile buffer and reaches 93.9% — fourteen points above baseline,
with stockout days cut by 75%. It costs 4.6% more, and whether that trade is
worth it depends on how you price a lost sale — which is a parameter I put in a
config file so it can be argued about."

---

## Slide 12 — Explainable AI: why *this* number?

**Content**
```
Product : 20727            Stock + on order : 347.2 units
Predicted 7-day demand     : 486.3 units
Supplier lead time         : 8 days
Stockout risk : HIGH  (p = 0.9999)
Recommended order : 1,380 units

  +1,042.11   expected demand over lead time + review period
    +674.98   safety stock (learned P95 - P50 spread)
    -347.20   less: stock already available / on order
      +0.00   clip at zero (never order negative)
      +0.00   supplier minimum order quantity
     +10.11   pack-size rounding
      +0.00   supplier capacity cap
  ──────────
      1,380   FINAL ORDER QUANTITY
```

**Visual:** `shap_waterfall_case1.png` beside the decomposition.

**Script (55s)**
"Most XAI demos stop at 'SHAP says feature X is important'. That doesn't tell a
buyer why the system asked for 1,380 units. So I built two layers. SHAP
explains the *forecast* — and because my winning model is Ridge, a linear
model, the Shapley values have a closed form, so they're exact, not sampled.
Then the *quantity* gets an exact arithmetic decomposition: expected demand,
plus safety stock, minus stock on hand, plus MOQ, plus pack rounding. Those
components sum to the recommendation with a residual of zero across all 16,689
decisions — and there's a test that fails if that identity ever breaks."

---

## Slide 13 — What-if: making it decision *support*

**Content**
| Scenario | Order | Change |
|---|---|---|
| Current situation | 1,380 | — |
| Lead time halved (8 → 4 days) | 924 | −456 |
| Lead time doubled (8 → 16) | 2,292 | +912 |
| Stock fully depleted | 1,728 | +348 |
| Stock = 150% of lead-time demand | 156 | −1,224 |
| No safety buffer | 696 | −684 |
| Demand surge +30% | 1,692 | +312 |

**Visual:** live dashboard sliders, or the table.

**Script (40s)**
"A system that hands you one number is an oracle. A system you can interrogate
is decision support. Every row here re-runs the real policy function with one
input changed — nothing is interpolated. So a buyer can ask 'what if I
negotiate the lead time down to four days?' and see the order drop by 456 units
*before* committing to the negotiation. That's the difference I was aiming for."

---

## Slide 14 — Demo, limitations, conclusion

**Content**
- **Live:** Streamlit dashboard — Overview · Forecast · Risk · Recommendation · WHY · Comparison
- **Verified:** 81 passing tests, 16 pipeline stages, 10 executed notebooks
- **Limitations, stated:** supply side simulated · one retailer · top 200 SKUs (42% of revenue) · no promotion data · gains over a moving average are modest (3.2%)
- **Conclusion:** the value is in the *uncertainty* layer and the *explanation*, not the point forecast

**Visual:** dashboard page 5 (WHY) live.

**Script (50s)**
"Let me close with what I'd want a reviewer to take away. The system runs
end to end — one command, sixteen stages, eighty-one passing tests. It
implements every Unit 1 and Unit 2 topic with code that actually executes, and
I've mapped each one to its file and its output. And I'd rather be judged on
the honest findings than a polished story: my advanced model lost the
regression task, the RBF kernel lost to the linear one, and a better point
forecast on its own didn't improve service at all. What *did* work was
quantifying uncertainty and explaining the decision. I'll show the dashboard
now."

---

## Backup slides (have ready, don't present)

**B1 — Feature importance.** `shap_global_importance_forecast.png` and
`shap_global_importance_classifier.png`. Note how different they are: demand is
driven by demand history; the *reorder decision* is driven by
`stock_vs_leadtime_demand`, `current_stock` and `lead_time_days`.

**B2 — Statistical significance.** Paired bootstrap, 1,000 resamples:
RMSE(Ridge) − RMSE(LightGBM) 95% CI lies entirely below zero.

**B3 — Cleaning funnel.** All 7 steps with rows removed and the reason for each.

**B4 — Simulated parameter distributions.** From `sim/supply_config.yaml`:
lead time Gamma(4.0, 1.6) clipped to [2,21]; MOQ from {12,24,48,72,96,144};
pack size from {1,6,12,24}; unit cost 45–70% of median price.

**B5 — The ablation table.** `reports/ablation_study.csv`.
