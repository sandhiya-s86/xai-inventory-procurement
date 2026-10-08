# 08 — Implementation Plan

## 1. Repository layout

```
ML-Project/
├── README.md
├── docs/                          ← these documents
├── pyproject.toml                 ← uv / pip, pinned
├── Makefile                       ← one command per pipeline stage
├── dvc.yaml                       ← stage graph, if using DVC
├── config/
│   ├── data.yaml                  ← paths, checksums, fold boundaries
│   ├── features.yaml              ← feature list + availability lags
│   ├── models/{lgbm,tft,arima,...}.yaml
│   ├── supply_config.yaml         ← THE simulated supply parameters
│   ├── scenarios.yaml             ← S1–S4
│   └── policies.yaml              ← P0–P4
├── src/xaierp/
│   ├── ingest/       favorita.py  online_retail.py  quality_report.py
│   ├── features/     builder.py  calendar.py  lags.py  sbc.py
│   ├── models/       base.py  statistical.py  gbm.py  tft.py  ensemble.py
│   ├── calibration/  conformal.py  cqr.py
│   ├── risk/         ltd.py  copula.py  metrics.py
│   ├── policy/       order_up_to.py  rsq.py  min_max.py  safety_stock.py
│   ├── optimize/     milp.py  model_builder.py  duals.py
│   ├── explain/      shap_forecast.py  decompose.py  shapley_ss.py
│   │                 counterfactual.py  supplier_regret.py  render.py
│   ├── simulate/     twin.py  demand.py  supply_events.py  costing.py
│   ├── evaluate/     accuracy.py  inventory.py  stats_tests.py  tables.py
│   └── service/      api.py  schemas.py  erpnext_client.py
├── notebooks/
│   ├── 01_eda_favorita.ipynb
│   ├── 02_eda_online_retail.ipynb
│   ├── 03_sbc_classification.ipynb
│   ├── 04_forecast_benchmark.ipynb
│   ├── 05_simulation_results.ipynb
│   └── 06_explanation_cases.ipynb
├── dashboard/                     ← Streamlit (research) / ERPNext page
├── tests/                         ← pytest; see §4
├── data/                          ← gitignored except .gitkeep + checksums
│   ├── raw/  interim/  processed/  forecasts/  results/
└── paper/
    ├── main.tex  refs.bib  figures/  tables/
```

**Rule:** nothing in `notebooks/` is imported by anything in `src/`. Notebooks consume the library and produce figures. Logic that matters lives in `src/` and is tested.

---

## 2. Technology stack

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Language | Python 3.11 | 3.12+ occasionally breaks forecasting libraries; 3.11 is the safe floor |
| Env/deps | `uv` (or conda) with a lockfile | Reproducibility is a paper requirement |
| Dataframes | `polars` for the 125M-row file, `pandas` elsewhere | pandas will not handle the SKU-level release comfortably on a laptop |
| Query | `duckdb` over parquet | Lets you slice the big file without loading it |
| Statistical forecasting | **`statsforecast`** (Nixtla) | Has Croston/SBA/TSB/ADIDA/IMAPA/AutoARIMA/ETS in one API; AutoARIMA ~20× faster than `pmdarima` |
| ML forecasting | **`lightgbm`** (+ `mlforecast` for lag/rolling automation), `xgboost` | Native quantile objective; TreeSHAP compatible |
| Deep forecasting | `pytorch-forecasting` (TFT, DeepAR) or `neuralforecast` (Nixtla — TFT, N-BEATS, N-HiTS) | `neuralforecast` is the more consistent API if you already use `statsforecast` |
| Conformal | `MAPIE`, or Nixtla conformal utilities | CQR implemented and maintained |
| Explainability | `shap` (TreeExplainer) | Exact for tree ensembles |
| Optimization | **`ortools`** (CP-SAT) | Free; handles the MOQ/pack integer structure well |
| Simulation | `numpy` + `numba` | Vectorised over items; SimPy is ~10–50× slower here for no benefit |
| Statistics | `scipy`, `statsmodels`, `scikit-posthocs` (Nemenyi + CD diagrams) | `scikit-posthocs` draws the critical-difference diagram for you |
| Tracking | `MLflow` | Experiment count will exceed what a spreadsheet can hold |
| Pipeline | `DVC` or a plain `Makefile` | Either is fine; pick one on day one |
| API | `FastAPI` + `pydantic` | Schema validation for the explanation object |
| Dashboard | `Streamlit` first; ERPNext page only if time allows | Do not spend week 3 on React |
| ERP | **ERPNext** in Docker | Real ERP doctypes; Python-native; documented REST API |
| Paper | LaTeX (IEEEtran / Springer LNCS) + `refs.bib` | — |

**Hardware:** LightGBM path runs on any laptop with 16 GB RAM. TFT wants a GPU — use Kaggle (30 GPU-hours/week free) or Colab, and record which. If no GPU is available, cut TFT and say so; the paper survives without it, and pretending you ran it would not.

---

## 3. Environment setup

```bash
cd Documents/ML-Project
uv venv --python 3.11 && .venv\Scripts\activate      # Windows
uv pip install polars duckdb pandas numpy scipy numba pyarrow \
               statsforecast mlforecast neuralforecast lightgbm xgboost \
               torch shap mapie ortools scikit-learn statsmodels \
               scikit-posthocs mlflow fastapi uvicorn streamlit \
               matplotlib seaborn kaggle ucimlrepo pytest ruff
uv pip freeze > requirements.lock
```

Verify `ortools` and `torch` import cleanly before writing any code against them — both are the usual sources of a lost afternoon on Windows.

---

## 4. Testing (do not skip — these are the tests that protect the claims)

| Test | Asserts | Protects |
|---|---|---|
| `test_no_leakage` | For every feature, the maximum source-row date ≤ forecast origin | The entire experiment |
| `test_decomposition_sums` | $\lvert Q - \sum \text{terms}\rvert < 10^{-6}$ on 10,000 sampled decisions | The core novelty claim |
| `test_shapley_efficiency` | $\phi_\mathcal{D} + \phi_\mathcal{L} = SS$ | The safety-stock split |
| `test_counterfactual_validity` | Stated counterfactual outcome == actual re-solve outcome | Explanation honesty |
| `test_simulator_conservation` | $\text{OH}_t = \text{OH}_{t-1} + \text{arrivals} - \text{sales} - \text{spoilage}$, every period, every item | Simulator correctness |
| `test_milp_feasibility` | Known-feasible instances solve; known-infeasible ones return INFEASIBLE not garbage | Optimizer correctness |
| `test_milp_vs_bruteforce` | On a 2-item, 2-supplier, 5-period toy instance, CP-SAT matches exhaustive enumeration | Optimizer correctness |
| `test_crn_identical` | Two policies see byte-identical demand and lead-time draws | Fair comparison |
| `test_conformal_coverage` | On synthetic data with known distribution, empirical coverage ≥ nominal | Calibration correctness |
| `test_sbc_classification` | Hand-computed ADI/CV² on a 20-point toy series matches | Segmentation correctness |

`test_simulator_conservation` and `test_decomposition_sums` are the two that will actually catch real bugs. Write them first, before the code they test.

---

## 5. Sixteen-week schedule

| Wk | Focus | Concrete deliverable | Gate |
|---|---|---|---|
| 1 | Data acquisition, ingest, quality report | Both datasets loaded, every ✅/⚠️ number in [03](03-datasets.md) verified | Numbers confirmed or corrected in writing |
| 2 | EDA + SBC classification | `01`–`03` notebooks; **ADI–CV² figure**; class distribution decides whether you need the SKU-level release | **Go/no-go on RQ1 framing** |
| 3 | Feature pipeline + leakage tests | `features/` + `test_no_leakage` green | No feature ships without a declared lag |
| 4 | Statistical baselines | B0–B5 across all folds, both datasets | Seasonal naive beaten by something |
| 5 | LightGBM quantile model | M1 trained, tuned, quantiles produced | Beats every statistical baseline on WRMSSE |
| 6 | TFT (+ DeepAR if time) | M3 trained | **Cut without regret if the GPU is a problem** |
| 7 | Conformal calibration | Coverage table before/after | Coverage within ±2pp of nominal |
| 8 | **Inventory digital twin** | Simulator + conservation tests + P0/P1 baselines running | Simulator reproduces a hand-worked example exactly |
| 9 | MILP optimizer | CP-SAT model + brute-force test | Matches exhaustive search on toy instances |
| 10 | Supply simulation + scenarios | `supply_config.yaml`, S1–S4 wired | Full P0–P4 × S1–S4 pipeline runs end-to-end |
| 11 | **Decision attribution** | Decomposition, Shapley SS split, supplier regret | `test_decomposition_sums` green |
| 12 | Counterfactuals + SHAP + renderer | Explanation object + NL output | Validity rate 100% |
| 13 | Full experiment runs | All of A–D, both datasets, results in parquet | Every headline table populated |
| 14 | Statistical validation + figures | DM, Friedman–Nemenyi, CD diagram, frontier plot | Figure 1 (cost–service frontier) done |
| 15 | Dashboard + ERPNext integration + user study | Streamlit demo, PO write path, 8–15 participants run | Study data collected |
| 16 | Paper + buffer | Full draft, internal review, submission | — |

**Writing is not week 16.** Draft the methodology section in week 4 and the related-work section in week 2, while you are reading. Papers written entirely in the final week read like it.

---

## 6. Milestone gates (stop and reassess if a gate fails)

| Gate | Condition | If it fails |
|---|---|---|
| G1 (wk 2) | ≥ 15% of series are intermittent/lumpy | Either move to the SKU-level release or **rewrite RQ1** around volatility strata. Do not proceed with an intermittency claim you cannot support |
| G2 (wk 5) | LightGBM beats seasonal naive by a meaningful margin on WRMSSE | Feature or leakage bug. Do not build on a broken forecaster |
| G3 (wk 8) | Simulator reproduces a hand-computed 10-period example exactly | Fix before any results are generated. Every number downstream depends on this |
| G4 (wk 9) | CP-SAT matches brute force on toy instances | Your MILP formulation is wrong |
| G5 (wk 11) | Decomposition residual ~0 on all decisions | The novelty claim is unsupported. **This is the critical gate** |
| G6 (wk 13) | P4 beats P0 on cost at matched fill rate in ≥ 3 of 4 scenarios | Report honestly. A partial win with a clear explanation of *when* it wins is publishable; a fabricated win is not |

---

## 7. Version control and hygiene

```bash
git init
# .gitignore: data/, .venv/, mlruns/, *.parquet, *.csv, *.7z, __pycache__/
```

- Commit `config/` and `tests/` always. Never commit raw data.
- Tag `v1.0-submission` at paper submission and record the commit hash in the paper.
- Keep `docs/lit-log.md` updated as you read — one line per paper: citation, what it does, what it does not do that you do. Writing the related-work section then takes an afternoon instead of a week.
- Keep `docs/decisions.md` — an append-only log of every methodological choice and its justification (why forward-fill oil, why lost-sales not backorder, why these four scenarios). Your paper's methodology section is a rewrite of this file, and your viva answers come from it verbatim.
