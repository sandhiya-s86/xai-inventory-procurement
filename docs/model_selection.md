# Model Selection

All figures are test-set results from `reports/*.csv`. The test split is
**2011-08-24 → 2011-12-02, 16,689 rows** and was used exactly once per model,
after every hyperparameter choice was fixed on validation.

---

## 1. Regression — forecasting 7-day demand

| Model | MAE | RMSE | R² | RMSLE | Train (s) |
|---|---|---|---|---|---|
| **Ridge Regression (L2)** | 136.36 | **288.53** | **0.510** | 1.133 | 0.17 |
| Bayesian Linear Regression | 136.32 | 288.67 | 0.509 | 1.130 | 0.92 |
| Linear Regression (OLS) | 134.19 | 288.96 | 0.508 | 1.115 | 0.28 |
| Robust Linear Regression (Huber) | **124.04** | 297.38 | 0.479 | **0.939** | 26.77 |
| *Baseline: 28-day mean × horizon* | 136.70 | 298.13 | 0.477 | 0.991 | 0.00 |
| LightGBM | 142.28 | 311.43 | 0.429 | 1.178 | 2.22 |
| *Baseline: 7-day mean × horizon* | 160.32 | 352.71 | 0.267 | 1.053 | 0.00 |

### Selected: Ridge Regression (α = 100)

**Why.** Lowest test RMSE (288.53) and highest R² (0.510). α was chosen on the
validation split from {0.01, 0.1, 1, 10, 100, 1000} — see
`reports/ridge_alpha_search.csv`. The test set played no part in that choice.

**Is the win real?** A paired bootstrap (1,000 resamples of the same test rows
for both models, notebook 09) puts the Ridge−LightGBM RMSE difference entirely
below zero, so Ridge's advantage over the boosted model is not sampling noise.
Against OLS and Bayesian Ridge the gap is small (288.53 vs 288.67 vs 288.96) and
within noise — Ridge is preferred on the tie-break of stability, since the
design matrix is ill-conditioned (sklearn reports rcond ≈ 7.7e-08 for OLS) and
the L2 penalty is what keeps the coefficients bounded.

### The honest surprise: LightGBM loses

The advanced model placed **sixth of seven**, behind even the 28-day moving
average. This is reported as found. Three reasons, in order of confidence:

1. **Zero-inflation.** 31.7% of SKU-days have zero demand. Tree splits
   repeatedly partition an already sparse target, and leaf means get dragged
   toward zero. The linear models interpolate smoothly instead.
2. **Distribution shift into the test period.** The test window is the Q4
   run-up, where demand levels exceed anything in training. Trees cannot
   extrapolate beyond the range of their training leaves; a linear model can.
   This is visible in the validation-vs-test gap: validation RMSE is far lower
   than test RMSE for every model.
3. **Feature set already encodes the non-linearity.** `rolling_mean_*`,
   `demand_momentum_ratio` and `demand_cv_28` hand the model the interactions a
   tree would otherwise have to discover, leaving little for it to add.

Early stopping on validation was applied (it halted at 36 trees and improved
test RMSE from 342.93 to 311.43) but did not close the gap. The result stands.

### Metric disagreement, stated rather than hidden

Huber has the **best MAE (124.04) and best RMSLE (0.939)** but worse RMSE.
That is exactly what a robust loss is supposed to do: it refuses to chase the
9,480-unit outlier days, so typical-day error falls while squared error rises.
**If the business cared about typical-day accuracy, Huber would be the right
choice.** RMSE is the declared primary metric here
(`config/project.yaml: evaluation.regression_primary`) because inventory cost
is driven by large misses, so Ridge is selected — but this is a judgement about
the cost function, not a statement that Huber is worse.

### Do the models beat doing nothing?

Yes, but narrowly. The best model's RMSE of 288.53 against the 28-day moving
average's 298.13 is a **3.2% improvement**. Honest framing: on this dataset,
seven days ahead, a well-chosen moving average is already a strong forecaster,
and the learned models add a modest amount. The larger gains in this project
come from the **uncertainty layer**, not the point forecast — see §3.

---

## 2. Classification — reorder / stockout risk

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---|---|---|---|---|---|
| **LightGBM Classifier** | **0.853** | 0.744 | 0.816 | **0.778** | **0.921** | 13.54 |
| SVM (linear kernel) | 0.826 | 0.691 | 0.816 | 0.748 | 0.899 | 44.46 |
| Logistic Regression | 0.824 | 0.684 | 0.826 | 0.748 | 0.898 | 1.50 |
| SVM (RBF kernel) | 0.786 | 0.616 | 0.859 | 0.718 | 0.891 | 26.72 |
| Linear Discriminant Analysis | 0.780 | **0.781** | 0.427 | 0.552 | 0.850 | 0.63 |
| Gaussian Naive Bayes | 0.476 | 0.374 | **0.969** | 0.540 | 0.769 | 0.24 |

### Selected: LightGBM Classifier

Best F1 (0.778) and best AUC (0.921), with competitive recall (0.816).
Here the boosted model **does** win — the reorder decision is a genuinely
non-linear function of stock position versus lead-time demand, with threshold
effects that trees capture well. Note that the same algorithm that failed at
regression succeeds at classification on the same data; the task, not the
algorithm, decided the outcome.

### Why F1 and recall, not accuracy

The positive rate is 30.8%, so a model that always answers "no reorder" scores
**69.2% accuracy while catching zero stockouts**. Accuracy is reported for
completeness and relied on for nothing. A false negative loses a sale and
customer goodwill; a false positive ties up working capital. Recall is
therefore weighted above precision, and all linear models use
`class_weight='balanced'`.

Confusion matrix for the selected model is in
`reports/figures/model_confusion_matrix.png`.

### What the comparison teaches (Unit 2)

**Generative vs discriminative.** Gaussian Naive Bayes models `p(x|y)p(y)` and
inverts with Bayes rule; Logistic Regression models `p(y|x)` directly. On
identical data NB reaches **0.969 recall but only 0.374 precision** — it flags
almost everything. Its conditional-independence assumption is badly violated
(`rolling_mean_7` and `rolling_mean_28` are strongly correlated), so it
becomes over-confident. Logistic regression makes no claim about how features
are distributed and is far better calibrated. This is the textbook trade-off,
measured.

**Kernel trick.** Linear SVM (F1 0.748) **beats** RBF SVM (F1 0.718) with
identical C, identical features and the same 15,000-row subsample. The extra
flexibility of an infinite-dimensional feature map did not help, which means
the boundary is close to linear in these features. A negative result, reported.

**Discriminant function.** LDA has the **highest precision (0.781)** of any
model but the worst recall (0.427). Its shared-covariance assumption makes it
conservative — it only flags cases it is confident about. Wrong trade-off for
stockouts, but a clean illustration of how the modelling assumption drives the
operating point.

---

## 3. Where the value actually is: the uncertainty layer

| Policy | Fill rate | Stockout days | Total cost | vs P0 |
|---|---|---|---|---|
| P0 moving average, no buffer | 79.9% | 2,027 | 1,086,570 | — |
| P1 classical reorder point | 88.7% | 966 | 1,054,847 | −2.9% |
| P2 ML forecast, no buffer | 79.2% | 1,570 | 1,077,805 | −0.8% |
| P3 ML forecast + classical buffer | 89.7% | 767 | 1,069,475 | −1.6% |
| **P4 ML forecast + quantile buffer** | **93.9%** | **496** | 1,136,500 | +4.6% |

**The headline finding is that a better point forecast alone (P2) does not
improve service** — fill rate actually falls 0.7pp against P0. What improves
service is explicitly modelling uncertainty: P1 adds +8.8pp with a classical
buffer and no ML at all; P4 adds **+14.0pp** using the learned quantile spread.

**P1 is cost-minimal; P4 maximises service.** P4 buys 14 percentage points of
fill rate for 4.6% more total cost, and cuts stockout days by 75% (2,027 → 496).
Which is correct depends on how the business prices a lost sale — a parameter
(`shortage_cost_multiplier`, currently 2.5) that lives in
`sim/supply_config.yaml` precisely so it can be challenged. P4 is the
dashboard's default recommendation because stockout cost in giftware wholesale
includes losing the retailer's next order, which the simulator does not model
and therefore understates.

**Caveat.** All costs are SIMULATED. The ranking between policies is a real
result about the decision rules; the currency amounts are not a claim about the
actual retailer.

---

## 4. Quantile calibration

| Target | Empirical coverage | Error | Pinball loss |
|---|---|---|---|
| P50 | 0.550 | +0.050 | 63.19 |
| P75 | 0.758 | +0.008 | 66.06 |
| P90 | 0.892 | −0.008 | 49.51 |
| P95 | 0.929 | −0.021 | 37.76 |

Coverage is within **2.1 percentage points** of target at every level, and
within 0.8pp at P75 and P90. The models are trained on the pinball loss, the
proper scoring rule for quantiles. P50 is the least accurate (+5.0pp), which is
expected on a zero-inflated target where the median sits near a mass point.

These intervals are what make the P4 buffer work without assuming Normality —
important given the measured skewness of 24.6.

---

## 5. Generalisation and the polynomial study

| Degree | Terms | Train RMSE | Val RMSE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| 1 | 4 | 285.77 | 212.67 | 291.05 | 0.50 |
| 2 | 14 | 282.55 | 211.42 | 314.83 | 0.42 |
| 3 | 34 | 281.00 | 213.98 | 393.52 | 0.09 |
| 4 | 69 | 278.42 | 242.90 | 941.97 | **−4.23** |

Training error falls monotonically with degree while test error rises by
**3.2×** and R² goes negative — degree 4 is worse than predicting the mean.
This is the Unit 1 bias/variance trade-off measured on real data, not
illustrated with a toy curve. A constant Ridge penalty is applied at every
degree so the comparison isolates model capacity.

Note that validation RMSE (≈212) is much lower than test RMSE (≈291) for every
degree. That is not a bug: the validation window (May–Aug 2011) is a quiet
trading period while the test window contains the Q4 peak. It is a useful
reminder that a single validation split can flatter a model, and it is why the
final judgement is made on the held-out test period.

---

## 6. Summary of selections

| Role | Model | Key metric |
|---|---|---|
| Demand forecast (deployed) | **Ridge Regression, α=100** | test RMSE 288.53, R² 0.510 |
| Reorder classifier (deployed) | **LightGBM Classifier** | test F1 0.778, AUC 0.921 |
| Safety-stock buffer | **LightGBM quantile P50/P95** | coverage within 2.1pp |
| Procurement policy | **P4** | fill rate 93.9%, +4.6% cost |

Both deployed models are explainable by construction: Ridge admits exact
closed-form Shapley values, and LightGBM admits exact TreeSHAP.
