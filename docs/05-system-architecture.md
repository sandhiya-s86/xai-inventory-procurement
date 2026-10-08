# 05 — System Architecture

## 1. Two architectures, kept separate

A recurring mistake in projects like this is to build one system that tries to be both the research pipeline and the product. They have opposite requirements: the research pipeline must be reproducible, deterministic and offline; the ERP integration must be responsive, stateful and online.

| | **Research pipeline** | **ERP decision-support service** |
|---|---|---|
| Trigger | manual / DVC pipeline run | nightly batch + on-demand API call |
| Data | parquet files on disk | ERP database |
| Priority | reproducibility, determinism | latency, availability |
| Output | metrics tables, figures, paper | recommendations in the ERP UI |
| Failure mode | rerun it | must degrade gracefully to the rule-based policy |

They **share** the model artefacts and the decision/explanation code. They share nothing else. Build the research pipeline first; the service is a thin wrapper added in week 15.

---

## 2. Research pipeline (build this)

```
                        ┌──────────────────────────────────┐
   data/raw/            │  data/raw/favorita/*.csv         │
                        │  data/raw/online_retail_II.xlsx  │
                        └────────────────┬─────────────────┘
                                         ▼
   ingest/              parse · type-cast · concat sheets · checksum
                                         ▼
   data/interim/        long-format panel  (series_id, date, demand, covariates)
                                         ▼
   features/            causal feature builder  ·  SBC classifier
                                         ▼
   data/processed/      feature matrix (parquet, partitioned by fold)
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
   models/statistical/          models/ml/                    models/deep/
   naive · MA · ARIMA ·          LightGBM-Q · XGBoost          TFT · DeepAR · N-HiTS
   ETS · Croston · SBA ·
   TSB · ADIDA · IMAPA
            └────────────────────────────┼────────────────────────────┘
                                         ▼
   forecasts/           q̂^(τ) per (series, origin, horizon, τ)   [parquet]
                                         ▼
   calibration/         split-conformal / CQR  →  calibrated quantiles
                                         ▼
   risk/                Monte-Carlo LTD  →  P(stockout), E[short], DTS
                                         ▼
   policy/              order-up-to · (R,s,Q) · (R,s,S) · static min-max
                                         ▼
   optimize/            OR-Tools CP-SAT MILP  →  Q, timing, supplier split
                                         ▼
            ┌────────────────────────────┴────────────────────────────┐
            ▼                                                         ▼
   explain/                                              simulate/
   · TreeSHAP over forecast                              inventory digital twin
   · exact decision decomposition                        (numpy/numba, CRN,
   · Shapley split of safety stock                        30 replications)
   · counterfactual thresholds                                        │
   · supplier regret                                                  ▼
   · NL template renderer                                 results/ cost, service,
            │                                                        turnover
            └────────────────────────┬───────────────────────────────┘
                                     ▼
   evaluate/            metrics · DM test · Friedman–Nemenyi · bootstrap CI
                                     ▼
   reports/             tables (LaTeX) · figures (PDF) · paper assets
```

Every arrow is a materialised parquet artefact on disk. No stage recomputes an upstream stage. This makes the pipeline resumable, which matters when the TFT run takes six hours and dies at hour five.

---

## 3. Component responsibilities

| Component | Input | Output | Key design note |
|---|---|---|---|
| `ingest` | raw files | long panel | Emits a data-quality report; fails loudly on schema drift |
| `features` | long panel | feature matrix | **Every feature must declare its availability lag.** Encode this as metadata and assert it, rather than trusting review |
| `models/*` | feature matrix + fold spec | quantile forecasts | Common `fit(train) / predict_quantiles(X, taus)` interface so models are swappable |
| `calibration` | forecasts + calib actuals | calibrated forecasts | Reports coverage before/after |
| `risk` | calibrated forecasts + supplier params | LTD samples, risk metrics | Gaussian-copula temporal dependence; MC size configurable |
| `policy` | risk + state | target level $S$, raw $Q$ | Pure functions, no I/O — trivially unit-testable |
| `optimize` | raw $Q$ + constraints | final $Q$, supplier, timing | Returns solution **plus** solver status, MIP gap, runtime, and duals |
| `explain` | everything above | structured explanation object | Emits JSON, not strings. Rendering is separate from computation |
| `simulate` | policy + optimizer + actuals | cost & service time series | The only component allowed to read held-out actuals |
| `evaluate` | all results | tables, tests, figures | Deterministic; regenerable from results parquet alone |

### The explanation object (contract)

Explanations must be **data**, not prose. Prose is generated from data at the last step. This makes explanations testable, diffable, and machine-checkable for faithfulness.

```json
{
  "series_id": "S44_GROCERY_I",
  "decision_date": "2017-08-01",
  "recommendation": {
    "quantity": 200, "supplier": "SUP_A", "order_date": "2017-08-01",
    "expected_arrival": "2017-08-10"
  },
  "quantity_decomposition": [
    {"term": "expected_lead_time_demand", "units": 178.0, "sign": "+"},
    {"term": "safety_stock",              "units":  71.0, "sign": "+",
      "sub": [{"source": "demand_variability",    "units": 33.0},
              {"source": "lead_time_variability", "units": 38.0}]},
    {"term": "on_hand",   "units": -86.0, "sign": "-"},
    {"term": "in_transit","units": -13.0, "sign": "-", "ref": "PO-2291"},
    {"term": "moq_rounding", "units": 50.0, "sign": "+", "binding_constraint": "MOQ=200"}
  ],
  "decomposition_residual": 0.0,
  "risk": {"p_stockout": 0.89, "expected_units_short": 42.3,
           "days_to_stockout": {"p50": 12, "p10": 8, "p90": 19}},
  "supplier_regret": [
    {"supplier": "SUP_B", "objective_delta": 1240.0,
     "breakdown": {"purchase": -1600.0, "holding": 480.0, "shortage": 2360.0}}
  ],
  "counterfactuals": [
    {"parameter": "lead_time_days", "from": 9, "to": 5, "new_quantity": 140},
    {"parameter": "cycle_service_level", "from": 0.95, "to": 0.90, "new_quantity": 165},
    {"parameter": "moq", "from": 200, "to": 1, "new_quantity": 187}
  ],
  "forecast_attribution": {
    "method": "TreeSHAP", "base_value": 165.2,
    "top_features": [
      {"feature": "rolling_mean_28", "shap": 46.1},
      {"feature": "onpromotion_lead_7", "shap": 18.4},
      {"feature": "is_payday", "shap": 9.8},
      {"feature": "post_holiday_flag", "shap": -6.6}
    ]
  },
  "solver": {"status": "OPTIMAL", "mip_gap": 0.0, "runtime_ms": 412}
}
```

> **`decomposition_residual` must be 0 (within float tolerance) and is asserted in a unit test.** If the parts do not sum to the whole, the explanation is a lie. This assertion is your strongest defence of the "exact decomposition" claim, and it belongs in the paper as a stated property.

---

## 4. ERP integration surface

### 4.1 Choice of ERP

| Option | Verdict |
|---|---|
| **ERPNext** (Frappe, Python, MariaDB) | **Recommended.** Fully open-source, Python-native, clean REST API, real Item / Warehouse / Bin / Supplier / Purchase Order / Stock Ledger doctypes. You get an authentic ERP schema without building one. |
| **Odoo Community** (Python, PostgreSQL) | Equally viable; strong external-JSON-RPC/REST access and a documented module extension model. Slightly heavier to develop against. |
| **Custom Django/FastAPI "ERP-style" app** | Fastest, but weakens the "ERP integration" claim — you would be integrating with your own toy schema. Acceptable fallback if time is short; say so plainly in the paper. |

Recommendation: **ERPNext in Docker**, seeded with Favorita-derived items/warehouses and the simulated supplier master. This gives real ERP doctypes to read from and write to, at low cost.

### 4.2 Integration pattern

```
┌──────────────────────────────────────────────────────────────┐
│  ERPNext                                                     │
│  Item · Warehouse · Bin(actual_qty) · Supplier ·             │
│  Purchase Order · Stock Ledger Entry · Sales Invoice         │
└───────────┬───────────────────────────────▲──────────────────┘
            │ REST (read)                   │ REST (write)
            ▼                               │
┌──────────────────────────────────────────────────────────────┐
│  XAI-ERP Decision Service   (FastAPI)                        │
│                                                              │
│  POST /forecast        → quantile forecasts                  │
│  POST /risk            → stockout probability, DTS           │
│  POST /recommend       → {qty, supplier, date, explanation}  │
│  GET  /explain/{id}    → full explanation object             │
│  POST /whatif          → counterfactual re-solve             │
│  POST /simulate        → policy comparison over a window     │
│                                                              │
│  Model registry (MLflow) · feature store (parquet/DuckDB)    │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│  Dashboard  (Streamlit for research; React page in ERPNext   │
│              if time permits)                                │
│  · replenishment worklist, ranked by risk × value            │
│  · per-item explanation panel (the waterfall above)          │
│  · what-if sliders (lead time, service level, budget)        │
│  · policy comparison view (proposed vs incumbent, £ and %)   │
│  · accept / modify / reject  →  logged as feedback           │
└──────────────────────────────────────────────────────────────┘
```

**Write path:** the service creates ERPNext Purchase Orders in **draft** state only. A human approves. Never auto-submit — this is both good engineering and the honest position on human-in-the-loop decision support, and it makes the "decision support, not automation" framing in the paper true rather than rhetorical.

### 4.3 The accept/modify/reject log

Every recommendation shown to a user is logged with the user's action and, on modification, their chosen quantity.

This log is disproportionately valuable:
- It is the **dependent variable for a real evaluation of explanation utility** — do users override less when shown decision-level explanations than forecast-level SHAP?
- With even 8–10 users × 20 decisions, it yields a genuine within-subjects comparison, which is far stronger than a Likert questionnaire alone.
- It is future work for closing the loop (learning from overrides).

Build this table on day one of the dashboard work; it costs an hour and it is the difference between a measured RQ5 and an asserted one.

### 4.4 Degradation

If the model service is unavailable, the ERP falls back to its native reorder rule and flags the item as "ML unavailable — rule-based". Do not fail the user's workflow. State this in the architecture section of the paper; reviewers with industrial experience notice its absence.

---

## 5. Non-functional targets

| Concern | Target | Note |
|---|---|---|
| Nightly batch (1,782 series, H=28, 9 quantiles) | < 20 min end-to-end | LightGBM global model; TFT scoring is the long pole |
| `/recommend` single item | < 2 s | Pre-computed forecasts; MILP solved per item on demand |
| MILP per item (3 suppliers, 28-day horizon) | < 500 ms | CP-SAT; report the gap if a time limit binds |
| Counterfactual re-solve (5 parameters) | < 3 s | Parallelise the five solves |
| Reproducibility | bit-identical given a seed | Fix seeds for numpy, LightGBM, torch, and the supply generator |

---

## 6. Data governance notes for the paper

- No personal data is used. `CustomerID` in Online Retail II is a pseudonymous integer and is **not needed** for this project — drop the column at ingest and say so. This is a clean, honest one-line answer to any ethics-statement requirement.
- Favorita raw data is **not redistributed**; the repository ships download scripts and SHA-256 checksums.
- The simulated supply layer **is** redistributed, under a permissive licence, with its generator and seed.
