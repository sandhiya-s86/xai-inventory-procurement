# XAI-ERP: Explainable AI-Driven Decision Support for Predictive Inventory and Procurement Optimization in ERP Systems

**Status:** Proposal / pre-implementation
**Document set version:** 1.0 — 18 August 2026
**Target output:** B.Tech final-year project + one conference paper (IEEE / Springer, Scopus-indexed)

---

## What this project actually is

A **decision-support pipeline** that sits on top of ERP transactional data and does four things in sequence:

1. **Forecasts demand probabilistically** (not just a point number — a full distribution per SKU per horizon).
2. **Converts that distribution into inventory risk** (stockout probability, expected units short, days-to-stockout).
3. **Optimizes the procurement decision** under real constraints (MOQ, lead time, budget, supplier capacity) using mixed-integer programming.
4. **Explains the recommendation itself** — not just the forecast — and is evaluated on whether those explanations are faithful and useful.

The ERP dashboard is the *delivery vehicle*, not the contribution. The contribution is items 1–4 evaluated end-to-end on **decision cost**, not only forecast error.

---

## Read the documents in this order

| # | Document | What it answers |
|---|----------|-----------------|
| 00 | [Executive Summary](docs/00-executive-summary.md) | The one-page version. Read first. |
| 01 | [Problem & Motivation](docs/01-problem-and-motivation.md) | Why fixed reorder points fail; what it costs |
| 02 | [Literature Review & Research Gap](docs/02-literature-review-and-research-gap.md) | What exists, what's missing, honest novelty assessment |
| 03 | [Dataset Audit](docs/03-datasets.md) | **Verified** facts on Favorita + Online Retail II, what they lack, how to get them |
| 04 | [Methodology](docs/04-methodology.md) | The full mathematical pipeline, stage by stage |
| 05 | [System Architecture](docs/05-system-architecture.md) | Components, data flow, ERP integration surface |
| 06 | [Experimental Design](docs/06-experimental-design.md) | Baselines, ablations, scenarios, statistical validation |
| 07 | [Evaluation Metrics](docs/07-evaluation-metrics.md) | Every metric, with formulas and why each is there |
| 08 | [Implementation Plan](docs/08-implementation-plan.md) | Repo layout, tech stack, 16-week schedule |
| 09 | [Risks & Mitigations](docs/09-risks-and-mitigations.md) | What can sink this and what to do about it |
| 10 | [Paper Outline](docs/10-paper-outline.md) | Section-by-section paper plan + venue shortlist |
| 11 | [References](docs/11-references.md) | Sources, all verified as of Aug 2026 |

---

## The three things to internalise before you start

**1. Neither dataset contains supplier or inventory data.** Favorita and Online Retail II are *sales* datasets. There are no stock positions, no purchase orders, no lead times, no supplier records anywhere in either. The procurement layer must be a **declared simulation**, and the paper must say so in the methodology section, in the limitations section, and in the figure captions. Reviewers reject papers that blur this line. See [03-datasets](docs/03-datasets.md) §5.

**2. The commonly-used Favorita file is aggregated to product *family*, not SKU.** The popular `store-sales-time-series-forecasting` release has 33 families × 54 stores = 1,782 series — these are smooth, not intermittent. If you want to answer "which model wins under intermittent demand," you need the **original 2017 competition release** with item-level data (~4,100 items, 125.5M rows). Using the family-level file and then claiming intermittency findings is a factual error a reviewer will catch. See [03-datasets](docs/03-datasets.md) §2.

**3. Forecast accuracy is not the deliverable.** A model with lower MAE can produce *worse* inventory decisions. The primary result table in the paper must be **cost and service level from the inventory simulator**, with forecast error as a secondary table. This is the single design choice that separates this from the hundreds of "we compared XGBoost and LSTM on retail sales" papers. See [04-methodology](docs/04-methodology.md) §7 and [06-experimental-design](docs/06-experimental-design.md) §4.

---

## Quick orientation: the pipeline in one diagram

```
  Favorita (primary)          Online Retail II (external validation)
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
              Feature engineering
      (calendar, lags, rolling stats, promo,
       holidays, oil, price, SBC demand class)
                        ▼
        ┌───────────────────────────────┐
        │  PROBABILISTIC FORECASTING    │   ← quantiles, not point estimates
        │  LGBM-quantile / TFT / DeepAR │
        │  vs ARIMA, ETS, Croston, TSB  │
        └───────────────┬───────────────┘
                        ▼
              Conformal calibration          ← distribution-free coverage
                        ▼
        ┌───────────────────────────────┐
        │  LEAD-TIME DEMAND DISTRIBUTION│   ← convolve demand × stochastic lead time
        │  → P(stockout), E[units short]│
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  PROCUREMENT OPTIMIZER (MILP) │   ← OR-Tools; MOQ, capacity, budget,
        │  order qty · timing · supplier│      multi-supplier allocation
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  TWO-LEVEL EXPLANATION        │
        │  L1: SHAP on the forecast     │
        │  L2: DECISION ATTRIBUTION —   │   ← the actual novelty
        │      decompose the order qty  │
        │      + counterfactual "what-if"│
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  INVENTORY DIGITAL TWIN       │   ← where the results come from
        │  replay policy over holdout   │
        │  → cost, stockouts, turnover  │
        └───────────────┬───────────────┘
                        ▼
                 ERP Dashboard
```

---

## Getting started

```bash
cd Documents/ML-Project
# 1. Read docs/00 through docs/03 before touching code.
# 2. Set up environment per docs/08-implementation-plan.md §3
# 3. Acquire data per docs/03-datasets.md §6
```

Nothing in this repository is implemented yet. These documents are the specification.
