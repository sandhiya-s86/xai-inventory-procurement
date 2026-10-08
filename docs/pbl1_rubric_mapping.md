# PBL-1 Rubric Mapping

Every rubric item mapped to the notebook, source file and artefact that
implements it. **No syllabus topic is listed unless there is code that runs it
and an output file that proves it ran.**

Verify mechanically with:

```bash
.venv\Scripts\python scripts\check_pbl1_completeness.py
```

---

## Criterion 1 — Problem Understanding & Dataset Familiarization (4 marks)

| Component | Notebook | Source | Artefact | Unit concept |
|---|---|---|---|---|
| Problem statement & scope | — | — | `docs/pbl1_scope.md` | — |
| Dataset choice & acquisition | — | `scripts/download_data.py` | `docs/dataset_acquisition.md` | — |
| Schema validation | 01 | `src/xaierp/ingest/online_retail.py::validate_schema` | `reports/data_quality_report.json` | — |
| Cleaning, with a reason per rule | 01 | `::clean_transactions` | `data_quality_report.json → cleaning.steps` | — |
| Row/column/type/missing audit | 01 | `src/xaierp/ingest/quality_report.py` | `reports/dataset_summary.csv` | — |
| Daily demand panel | 01 | `::build_daily_panel` | `data/processed/daily_demand_panel.parquet` | Supervised learning setup |

**Evidence.** 1,067,371 raw lines → 1,024,819 cleaned (96.01% retained), with a
documented reason for each of 7 removal steps. 200 SKUs × 604 trading days.

---

## Criterion 2 — Exploratory Data Analysis & Insights (4 marks)

| Component | Notebook | Source | Artefact | Unit concept |
|---|---|---|---|---|
| 16 labelled figures | 02 | `scripts/run_eda.py` | `reports/figures/eda_*.png` | — |
| Distribution: mean, variance, skew | 02, 06 | `run_eda.py` | `eda_findings.json → demand_distribution` | U1: mean, variance |
| Quantiles P25–P99 | 02, 06 | `run_eda.py` | `eda_findings.json → target_quantiles` | U1: quantiles |
| Correlation & covariance | 02, 06 | `run_eda.py` | `eda_09_correlation_heatmap.png` | U1: covariance |
| Outlier / IQR analysis | 02 | `run_eda.py` | `eda_05_outliers_boxplot.png` | U1: variance |
| Seasonality & trend | 02 | `run_eda.py` | `eda_16_seasonal_decomposition.png` | — |
| Rolling mean / std | 02 | `run_eda.py` | `eda_13`, `eda_14` | U1: mean, variance |
| Intermittency | 02 | `run_eda.py` | `eda_12_zero_demand.png` | — |
| Written interpretation | — | — | `docs/eda_insights.md` | — |

**Evidence.** Each figure has Observation / Interpretation / Business
implication written against computed values (skew 24.6, Q4 uplift +64.6%,
31.7% zero SKU-days, 104 non-trading Saturdays).

---

## Criterion 3 — Unit 1 / Unit 2 Implementation, incl. Regression and Classification (4 marks)

### Unit 1

| Syllabus topic | Notebook | Source | Artefact |
|---|---|---|---|
| ML: what and why | — | — | `paper/pbl1_report.md` §3 |
| Supervised learning | 03 | `src/xaierp/features/builder.py` | `feature_table.parquet` |
| Regression | 04 | `src/xaierp/models/regression.py` | `regression_model_comparison.csv` |
| Classification | 05 | `src/xaierp/models/classification.py` | `classification_model_comparison.csv` |
| **Polynomial curve fitting** | 04, 09 | `regression.py::_polynomial` | `polynomial_degree_study.csv` |
| **Overfitting vs underfitting** | 04, 09 | `train_models.py::stage_polynomial` | `model_polynomial_overfitting.png` |
| Least squares / SSE | 04 | `LinearRegression` | `regression_model_comparison.csv` |
| Probability theory | 06 | `run_eda.py` | `eda_findings.json` |
| **Bayes rule** | 05 | `classification.py` → `GaussianNB` | `classification_model_comparison.csv` |
| Discrete / continuous RVs, PDFs | 06 | notebook 06 | `eda_04_demand_distribution.png` |
| **Quantiles** | 06 | `train_models.py::stage_quantile` | `quantile_model_metrics.csv` |
| **Mean and variance** | 06 | notebook 06 | `Var(X)=E[X²]−E[X]²` verified |
| Expectation | 06 | notebook 06 | notebook output |
| **Covariance** | 06 | notebook 06 | covariance matrix printed |

### Unit 2

| Syllabus topic | Notebook | Source | Artefact |
|---|---|---|---|
| **Maximum likelihood estimation** | 04 | derivation + `LinearRegression` | notebook 04 markdown |
| **Least squares** | 04 | `LinearRegression` | RMSE 288.96 |
| **Robust linear regression** | 04 | `HuberRegressor` | MAE 124.04 (best) |
| **Ridge regression** | 04 | `Ridge(alpha=100)` | RMSE 288.53 (best), `ridge_alpha_search.csv` |
| **Bayesian linear regression** | 04 | `BayesianRidge` + `predictive_std` | RMSE 288.67 + predictive σ |
| **Discriminant function** | 05 | `LinearDiscriminantAnalysis` | precision 0.781 (best) |
| **Probabilistic generative model** | 05 | `GaussianNB` | recall 0.969 |
| **Probabilistic discriminative model** | 05 | `LogisticRegression` | F1 0.748 |
| Laplace approximation | — | — | **Deferred** — see `pbl1_scope.md` §2 |
| Bayesian logistic regression | — | — | **Deferred** — Bayesian treatment shown on the regression side |
| **Kernel functions** | 05 | `SVC(kernel='linear')` | F1 0.748 |
| **Kernel trick** | 05 | `SVC(kernel='rbf')` | F1 0.718 |
| **SVMs** | 05 | both `SVC` variants | `classification_model_comparison.csv` |

### Regression models (7, incl. 2 baselines)

Ridge · Bayesian Ridge · OLS · Huber · LightGBM · 28-day MA · 7-day MA

### Classification models (6)

LightGBM · SVM-linear · Logistic · SVM-RBF · LDA · Gaussian NB

---

## Criterion 4 — Comparative Performance Analysis (4 marks)

| Component | Notebook | Artefact |
|---|---|---|
| Regression table (MAE/MSE/RMSE/R²/RMSLE/time) | 04, 09 | `reports/regression_model_comparison.csv` |
| Classification table (Acc/P/R/F1/AUC/time) | 05, 09 | `reports/classification_model_comparison.csv` |
| RMSE comparison chart | 09 | `model_regression_rmse.png` |
| R² comparison chart | 09 | `model_regression_r2.png` |
| F1 / recall chart | 09 | `model_classification_f1.png` |
| ROC curves (all 6 classifiers) | 09 | `model_roc_curves.png` |
| Confusion matrix (best) | 09 | `model_confusion_matrix.png` |
| Predicted vs actual | 09 | `model_predicted_vs_actual.png` |
| Polynomial degree study | 09 | `model_polynomial_overfitting.png` |
| Quantile calibration | 06, 09 | `model_quantile_calibration.png` |
| **Bootstrap 95% CIs** | 09 | notebook 09 output |
| **Paired bootstrap significance test** | 09 | notebook 09 output |
| Policy ladder P0–P4 | 07 | `reports/policy_comparison.csv` |
| Ablation A–E | 07 | `reports/ablation_study.csv` |
| Written selection rationale | — | `docs/model_selection.md` |

---

## Criterion 5 — Result Interpretation & Presentation (4 marks)

| Component | Notebook | Artefact |
|---|---|---|
| SHAP global importance (forecast, exact) | 08 | `shap_global_importance_forecast.csv` |
| SHAP global importance (classifier, TreeSHAP) | 08 | `shap_global_importance_classifier.csv` |
| SHAP additivity verified | 08 | max error 6.1e-04 (linear), 8.5e-13 (tree) |
| Per-case waterfall plots | 08 | `shap_waterfall_case*.png` |
| Natural-language explanations | 08, 10 | `reports/generated/case_studies.txt` |
| **Decision decomposition summing to the order** | 08, 10 | `case_studies.json`, residual 0.000000 |
| What-if / counterfactual | 10 | `reports/whatif_examples.csv` |
| Business interpretation | — | `docs/eda_insights.md`, `docs/model_selection.md` |
| Full report | — | `paper/pbl1_report.md` |
| Slide deck + speaking script | — | `docs/pbl1_presentation.md` |
| Viva preparation | — | `docs/viva_questions.md` |
| Live dashboard | — | `dashboard/app.py` (6 pages) |

---

## Supporting engineering (not directly marked, but assessed indirectly)

| Component | Artefact |
|---|---|
| Reproducible pipeline | `scripts/run_pipeline.py` (16 stages) |
| Fixed seeds everywhere | `src/xaierp/utils/seeds.py` |
| Config-driven, no magic numbers | `config/project.yaml`, `sim/supply_config.yaml` |
| **81 passing tests** | `tests/` |
| Leakage tests | `tests/test_no_leakage.py` (8 tests) |
| Inventory conservation | `tests/test_inventory_conservation.py` (7 tests) |
| Reorder constraint tests | `tests/test_reorder_policy.py` (14 tests) |
| Explanation reconciliation | `tests/test_decision_explanation.py` (8 tests) |
| Simulation determinism | `tests/test_simulation.py` (8 tests) |
| Completeness gate | `scripts/check_pbl1_completeness.py` |

---

## Honesty register

Things this project does **not** claim:

- **ERP integration is not implemented.** The dashboard is a demo on an
  ERP-shaped data model. See `docs/erp_integration_ready.md`.
- **Supply-side data is simulated.** Stock, suppliers, lead times, MOQ, pack
  sizes, capacities and all costs are generated by `sim/`. The dataset has none.
- **LightGBM loses the regression task.** Reported in full
  (6th of 7, behind a moving average) rather than omitted.
- **RBF SVM loses to linear SVM.** A negative kernel result, reported.
- **The ML point forecast alone does not improve fill rate** (P2 is 0.7pp
  *worse* than P0). The gain comes from the uncertainty layer.
- **Gains over a moving-average baseline are modest** (3.2% RMSE).
- **Laplace approximation and Bayesian logistic regression are deferred**, and
  are marked as such rather than claimed.
