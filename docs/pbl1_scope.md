# PBL-1 Scope — what is implemented, what is deferred

**Project:** Explainable AI-Driven Decision Support for Predictive Inventory and
Procurement Optimization in ERP Systems

The repository originally contained a research proposal sized for a final-year
or conference-paper project. PBL-1 is assessed against a five-criterion,
20-mark rubric. This document records exactly where the line was drawn and why.

---

## 1. Implemented for PBL-1

### Data
| Item | Status |
|---|---|
| UCI Online Retail II (1,067,371 invoice lines, Dec 2009 – Dec 2011) | Implemented, real |
| Schema validation, cleaning with a documented reason per rule | Implemented |
| Daily demand panel: 200 SKUs × 604 trading days = 120,800 rows | Implemented |
| Data quality report (`reports/data_quality_report.json`) | Implemented |

### Unit 1 concepts
| Concept | Where |
|---|---|
| Supervised learning: regression + classification | Both problems implemented |
| Polynomial curve fitting, degrees 1–4 | `reports/polynomial_degree_study.csv` |
| Overfitting vs underfitting, measured | Test RMSE 291 → 942 across degrees |
| Least squares | `LinearRegression`, and its MLE equivalence derived |
| Probability, PDFs, quantiles | Notebook 06 |
| Mean, variance, expectation | Notebook 06, verified `Var(X)=E[X²]−E[X]²` |
| Covariance and correlation | Notebook 06, EDA figure 09 |
| Bayes rule | Gaussian Naive Bayes classifier |

### Unit 2 concepts
| Concept | Where |
|---|---|
| Maximum likelihood ↔ least squares | Derived in notebook 04 |
| Ridge regression (L2) | Winning forecaster; α tuned on validation |
| Bayesian linear regression | `BayesianRidge`, with predictive std |
| Robust linear regression | `HuberRegressor` |
| Discriminant function | `LinearDiscriminantAnalysis` |
| Probabilistic generative model | `GaussianNB` (models p(x\|y)p(y)) |
| Probabilistic discriminative model | `LogisticRegression` (models p(y\|x)) |
| Kernel functions, kernel trick | `SVC(linear)` vs `SVC(rbf)`, identical C and data |
| SVMs | Both SVM variants |

### System
- 7-day demand forecasting (regression), 7 models including 2 naive baselines
- Reorder / stockout-risk prediction (classification), 6 models
- LightGBM quantile regression at P50/P75/P90/P95 for safety stock
- Simulated inventory state and supplier master, fixed seed, fully documented
- Procurement policy with MOQ, pack rounding and supplier capacity
- Policy ladder P0→P4 and a matching ablation study
- SHAP explanations (exact linear SHAP for the forecaster, TreeSHAP for the classifier)
- Decision decomposition that sums exactly to the recommended quantity
- What-if / counterfactual analysis
- Streamlit dashboard, 6 pages
- 81 passing tests
- 10 executed notebooks

---

## 2. Deliberately deferred

These appear in the original proposal and are **not** implemented. They are
recorded here as future work rather than silently dropped.

| Deferred | Why it is out of scope for PBL-1 |
|---|---|
| Temporal Fusion Transformer, DeepAR, N-BEATS, N-HiTS | Deep sequence models need far more data per series and GPU time than this task justifies; the syllabus covers neither. |
| MILP procurement optimisation | PBL-1 asks for a decision rule, not a solver. The current policy is auditable by hand, which matters more here. |
| Digital twin at research scale | Requires a validated supply-chain model; the supply side here is openly simulated. |
| Conformal prediction intervals | Quantile regression already supplies calibrated intervals (coverage within 2.1 points of target) at a fraction of the complexity. |
| Laplace approximation / full Bayesian logistic regression | `BayesianRidge` already demonstrates the Bayesian treatment on the regression side; adding it on the classification side would duplicate the concept for no extra marks. |
| Live ERPNext / SAP / Odoo deployment | The project claims ERP-*readiness*, not integration. See `docs/erp_integration_ready.md`. |
| Multi-echelon inventory | Single-echelon is enough to demonstrate the decision logic. |

---

## 3. Scoping decisions that affect the results

These are judgement calls a reviewer should be able to challenge, so they are
stated explicitly rather than buried in code.

**200 SKUs, not all 5,305.**
A dense daily panel over every stock code would be overwhelmingly structural
zeros, which makes "predict 0" a strong baseline and the exercise meaningless.
SKUs are ranked by revenue and must have sold on at least 120 distinct days.
The modelled set covers **42.0% of cleaned revenue**. Configurable via
`panel.top_n_skus`.

**Trading days, not calendar days.**
This retailer does not trade on Saturdays (30 Saturday invoices against
5,987–8,167 on other weekdays) and closes over Christmas. 135 of 739 calendar
days are non-trading. Zero-filling them would invent roughly one fake
zero-demand observation per week per SKU and bias every model downward.

**Price movement as a promotion proxy.**
Online Retail II has no promotion flag. `price_discount_pct` measures the unit
price against the SKU's own 28-day trailing median. This is labelled a proxy
throughout and is never described as a promotion field.

**Supply side is simulated.**
Stock, suppliers, lead times, MOQ, pack sizes, capacities and all costs are
generated by `sim/` from `sim/supply_config.yaml`. The dataset contains none of
them. Cost comparisons between policies are meaningful; the absolute currency
figures are not a claim about the real retailer.

**SVMs are trained on a 15,000-row subsample.**
`SVC` scales roughly quadratically in sample count. The subsample size is
declared in the model spec and reported in the results table rather than hidden.

---

## 4. Dataset choice

The Kaggle *Corporación Favorita — Store Sales* competition was evaluated first
and is richer (it ships promotion flags, oil prices, holidays and store
metadata). It was **not** used because it requires Kaggle credentials and
acceptance of competition rules, which cannot be automated.

UCI Online Retail II was selected instead: real transactional retail data,
CC BY 4.0, downloadable with no credentials, and with genuine returns and
intermittency that make the cleaning and forecasting work substantive. Full
reasoning in `docs/dataset_acquisition.md`.
