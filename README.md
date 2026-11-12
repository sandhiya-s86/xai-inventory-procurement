# Explainable AI-Driven Decision Support for Predictive Inventory and Procurement Optimization in ERP Systems

**PBL-1 · Machine Learning (Units 1–4)**

An end-to-end system that turns 1,067,371 real retail transactions into
auditable purchase recommendations — and explains every one of them.

**Team**
| Name | Register Number |
|---|---|
| Sandhiya S | RA2411003010045 |
| Nishanth Kannan | RA2411003010051 |
| Netti Nithin Chakravarthy | RA2411003010033 |

**Repository:** <https://github.com/sandhiya-s86/xai-inventory-procurement>
· `main` = original proposal · `pbl1-implementation` = the built system (default)

**Presentation:** [`paper/PBL1_Presentation.pptx`](paper/PBL1_Presentation.pptx) — 15 slides with speaker notes


> The original research proposal this repository started from is preserved at
> [`docs/00-original-proposal-readme.md`](docs/00-original-proposal-readme.md).
> What was implemented and what was deliberately deferred is recorded in
> [`docs/pbl1_scope.md`](docs/pbl1_scope.md).

```
Historical sales  →  Cleaning  →  EDA  →  Features  →  Regression (7-day demand)
                                                    →  Classification (reorder?)
                                                    →  Model comparison
                                                    →  Explainable AI
                                                    →  Inventory risk
                                                    →  Procurement recommendation
                                                    →  Streamlit dashboard
```

---

## Headline results

| | |
|---|---|
| **Best forecaster** | Ridge Regression (α=100) — test RMSE **288.53**, R² **0.510** |
| **Best classifier** | LightGBM — F1 **0.778**, recall **0.816**, AUC **0.921** |
| **Uncertainty** | LightGBM quantiles, P90 coverage **89.2%** (target 90%) |
| **Service gain** | Fill rate **79.9% → 93.9%** (+14.0pp), stockout days −75% |
| **Explanation integrity** | Decomposition residual **0.000000** across 16,689 decisions |
| **Demand regimes (Unit 4)** | HMM finds QUIET / NORMAL / **SURGE**; SURGE lands on the Christmas peak in both years |
| **Regime-aware buffers** | Volatility differs **2.1×** across regimes — a pooled safety stock is wrong in both directions |
| **SKU segments (Unit 3)** | **5 segments** from 13 behavioural features; 31 SKUs sit genuinely between them |
| **Verification** | **239 tests passing**, 20 pipeline stages, 10 notebooks, **8/8 dashboard pages** Playwright-verified |

Three findings reported because they are true, not because they flatter:

- **A better point forecast alone did not improve service.** Policy P2 (ML
  forecast, no safety buffer) achieved a 79.2% fill rate — *below* the 79.9%
  naive baseline. The gain came entirely from modelling uncertainty.
- **LightGBM lost the regression task**, placing 6th of 7 behind a 28-day
  moving average, because demand is 31.7% zero-inflated and the test period
  lies outside the training range.
- **The RBF kernel lost to the linear kernel** (SVM F1 0.718 vs 0.748) with
  identical C, features and subsample.
- **The best dendrogram was the worst partition.** Average linkage scored the
  highest cophenetic correlation (0.931) while putting 97% of SKUs in one
  cluster — chaining. Ward scored worst (0.495) and was the only usable cut.
- **This series does not actually underflow.** At T = 604 the raw forward
  product is ~1e-217, above float64's floor; the naive recursion only dies at
  T = 1,000. Reported rather than claiming a textbook underflow that did not
  happen.

---

## Units 3 & 4 — unsupervised learning and sequential models

Full write-up: **[`docs/unit3-unit4.md`](docs/unit3-unit4.md)**

| | |
|---|---|
| **Unit 3** — `src/xaierp/unsupervised/` | measuring dissimilarity · K-means + elbow · mixtures of Gaussians · EM + the alternative view of EM · factor analysis + varimax · PCA + choosing latent dimensions · hierarchical clustering · evaluating clustering output |
| **Unit 4** — `src/xaierp/sequential/` | sequential data · Markov models · HMMs · Baum–Welch · forward–backward · scaling factors · Viterbi · sum–product · linear dynamical systems (Kalman + RTS) |

Every algorithm is written from first principles and pinned against either
scikit-learn / SciPy or **brute-force enumeration over all `N^T` state paths**,
so "from scratch" does not quietly become "from scratch and wrong".

**Deliverables**

- [`paper/Unit3_Unit4_Results_Report.pdf`](paper/Unit3_Unit4_Results_Report.pdf) — 18-page results report; every number read live from the findings JSON
- [`paper/Unit3_Unit4_Study_Guide.pdf`](paper/Unit3_Unit4_Study_Guide.pdf) — 17-page study guide, all 17 topics split evenly across the team of 3
- Dashboard pages **7 · SKU Segmentation** and **8 · Demand Regimes**

```bash
.venv\Scripts\python scriptsun_unsupervised.py   # Unit 3
.venv\Scripts\python scriptsun_sequential.py     # Unit 4
.venv\Scripts\python scriptsuild_unit34_docs.py  # the two PDFs
.venv\Scripts\python scriptserify_dashboard.py   # drive all 8 pages in Chromium
```

> Unit 5 is **not** covered — the Drive link supplied for it returns HTTP 401
> without authentication, so its syllabus could not be read.

---

## Quick start

```bash
# 1. Environment (Python 3.11)
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 2. Everything, end to end (~15 min on first run)
.venv\Scripts\python scripts\run_pipeline.py

# 3. The demo
.venv\Scripts\streamlit run dashboard\app.py
```

Individual stages:

```bash
.venv\Scripts\python scripts\download_data.py          # no credentials needed
.venv\Scripts\python scripts\prepare_data.py
.venv\Scripts\python scripts\run_eda.py
.venv\Scripts\python sim\generate_supply.py
.venv\Scripts\python sim\inventory_state.py
.venv\Scripts\python scripts\build_features.py
.venv\Scripts\python scripts\train_models.py           # or --stage regression
.venv\Scripts\python scripts\run_simulation.py
.venv\Scripts\python scripts\run_explainability.py
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\check_pbl1_completeness.py
```

Resume after a failure: `run_pipeline.py --from train_regression`

---

## Data provenance — read this first

| Category | Fields | Provenance |
|---|---|---|
| Demand side | units sold, date, price, SKU, country, invoice | **REAL** (UCI Online Retail II, CC BY 4.0) |
| Supply side | stock, suppliers, lead time, MOQ, pack size, capacity, costs | **SIMULATED** |

UCI Online Retail II is a **sales log**. It contains no inventory, suppliers,
lead times or costs — and you cannot make a procurement decision without them.
Those values are generated by `sim/` from `sim/supply_config.yaml` under a
fixed seed, are flagged `is_simulated` in every artefact, and are labelled as
simulated in the dashboard, notebooks and report.

Policy comparisons are valid because all policies face identical real demand
and identical simulated terms. **Absolute currency figures are not claims about
the real retailer.**

---

## Dataset

**UCI Online Retail II** — <https://archive.ics.uci.edu/dataset/502/online+retail+ii>

| | |
|---|---|
| Raw invoice lines | 1,067,371 |
| Period | 2009-12-01 → 2011-12-09 |
| After cleaning | 1,024,819 (96.01% retained) |
| Modelled panel | 200 SKUs × 604 trading days = 120,800 rows |
| Credentials | **None** |

Chosen over the richer Kaggle Favorita competition because it needs no
credentials, so the project reproduces from one command. Full reasoning and the
trade-offs accepted: [`docs/dataset_acquisition.md`](docs/dataset_acquisition.md).

---

## Repository layout

```
config/project.yaml              every tunable parameter
sim/
  supply_config.yaml             SIMULATED supply-side distributions
  generate_supply.py             supplier master
  inventory_state.py             daily stock trajectory + reorder labels
src/xaierp/
  ingest/      read, validate, clean, build the daily panel
  features/    calendar, lag, rolling features (no future information)
  models/      regression and classification model zoo
  policy/      reorder quantity with MOQ / pack / capacity
  simulate/    procurement cost simulation, policy ladder
  explain/     SHAP, decision decomposition, what-if
  utils/       config, seeding, metrics
scripts/       download, prepare, eda, features, train, simulate, explain,
               plots, pipeline, completeness, notebooks
notebooks/     01-10, all executed
tests/         81 tests
dashboard/     Streamlit app (6 pages)
reports/       figures, tables, generated explanations
paper/         pbl1_report.md
docs/          scope, rubric mapping, EDA insights, model selection, viva, ...
```

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/pbl1_scope.md`](docs/pbl1_scope.md) | What is implemented, what is deferred, and why |
| [`docs/pbl1_rubric_mapping.md`](docs/pbl1_rubric_mapping.md) | Every rubric item → notebook, file, artefact, syllabus topic |
| [`docs/dataset_acquisition.md`](docs/dataset_acquisition.md) | Dataset choice, download, what it does *not* contain |
| [`docs/eda_insights.md`](docs/eda_insights.md) | 16 figures: observation → interpretation → business implication |
| [`docs/model_selection.md`](docs/model_selection.md) | Why each winner won, including the uncomfortable results |
| [`docs/erp_integration_ready.md`](docs/erp_integration_ready.md) | What integration would take; what is *not* claimed |
| [`docs/viva_questions.md`](docs/viva_questions.md) | Q&A with this project's real numbers |
| [`docs/pbl1_presentation.md`](docs/pbl1_presentation.md) | 14 slides with speaking scripts |
| [`paper/pbl1_report.md`](paper/pbl1_report.md) | Full report |
| [`docs/implementation_audit.md`](docs/implementation_audit.md) | State of the original proposal and what changed |

The original research-proposal documents (`docs/00-executive-summary.md`
through `docs/11-references.md`, `decisions.md`, `lit-log.md`) are retained
unchanged.

---

## Syllabus coverage

**Unit 1** — supervised learning · regression · classification · polynomial
curve fitting (degrees 1-4) · overfitting/underfitting (measured: test R²
0.50 → −4.23) · least squares · probability & PDFs · Bayes rule (Gaussian NB) ·
quantiles (P50-P95) · mean & variance (`Var(X)=E[X²]−E[X]²` verified) ·
expectation · covariance

**Unit 2** — MLE ↔ least squares (derived) · ridge regression · Bayesian linear
regression (with predictive σ) · robust regression (Huber) · discriminant
function (LDA) · probabilistic generative (GaussianNB) · probabilistic
discriminative (Logistic) · kernel functions · kernel trick (linear vs RBF
SVM) · SVMs

*Deferred and marked as such:* Laplace approximation, Bayesian logistic
regression — see [`docs/pbl1_scope.md`](docs/pbl1_scope.md) §2.

---

## Testing

```bash
.venv\Scripts\python -m pytest
# 81 passed
```

| File | Covers |
|---|---|
| `test_no_leakage.py` | No feature sees the present or future; splits don't overlap |
| `test_data_quality.py` | Panel density, no duplicates, cleaning log adds up |
| `test_features.py` | Lag/rolling/price arithmetic against hand calculations |
| `test_models.py` | Every model trains; comparison tables are coherent |
| `test_inventory_conservation.py` | `ending = beginning + arrivals − consumed` |
| `test_reorder_policy.py` | Never negative; MOQ honoured; pack multiples; capacity |
| `test_decision_explanation.py` | Explanation reconciles; refuses to render if tampered |
| `test_simulation.py` | Seed reproducibility; what-if responds to every lever |

---

## Notes on the development environment

Built on Windows 11 / Python 3.11 on a memory-constrained machine. Two
accommodations are in the code and documented where they occur:

- the 45 MB workbook is **streamed** to Parquet rather than loaded
  (`pandas.read_excel` exhausted available memory);
- training runs **one subprocess per stage** with BLAS thread pools capped at 1.

Neither changes any result.

---

## Licence and attribution

Code: for academic use.
Data: UCI Online Retail II, CC BY 4.0 —
Chen, D. (2019), <https://doi.org/10.24432/C5CG6D>.
The dataset is **not** redistributed here; `scripts/download_data.py` fetches it.
