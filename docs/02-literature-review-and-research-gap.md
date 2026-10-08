# 02 — Literature Review and Research Gap

> This document is the one to get right. The biggest risk to this project is not implementation — it is choosing a contribution that has already been published. Read §4 before you commit to the title.

---

## 1. Strand A — Retail demand forecasting

### 1.1 Statistical baselines you must include

These are not optional. A forecasting paper without simple baselines is desk-rejected, and the M5 competition demonstrated that naive and simple exponential-smoothing benchmarks beat a large fraction of submitted ML methods.

| Method | Use | Library |
|---|---|---|
| Seasonal naive | The floor. Everything must beat this. | `statsforecast` |
| ARIMA / AutoARIMA | Classical univariate baseline | `statsforecast` (~20× faster than `pmdarima`) |
| ETS / Holt-Winters | Seasonal exponential smoothing | `statsforecast` |
| **Croston / SBA / TSB** | Intermittent demand. TSB (Teunter–Syntetos–Babai) replaces Croston's inter-demand interval with a demand *probability*, handling obsolescence. | `statsforecast` |
| ADIDA / IMAPA | Temporal-aggregation approaches for intermittent series | `statsforecast` |

### 1.2 The intermittency taxonomy (needed for RQ1)

**Syntetos, Boylan & Croston (2005)** classify series by two statistics:

- **ADI** — average demand interval (mean number of periods between non-zero demands)
- **CV²** — squared coefficient of variation of non-zero demand sizes

With cutoffs **ADI = 1.32** and **CV² = 0.49**:

|  | CV² < 0.49 | CV² > 0.49 |
|---|---|---|
| **ADI < 1.32** | **Smooth** | **Erratic** |
| **ADI > 1.32** | **Intermittent** | **Lumpy** |

Guidance from the literature: Croston is preferable for smoother series; the Syntetos–Boylan Approximation (SBA) is preferable for more intermittent, erratic and lumpy series. Note also that the sufficiency of this taxonomy is itself contested in recent literature — treating the cutoffs as a reporting *stratification* rather than a *model-selection rule* is the defensible position, and you should say so.

**Use of this in your paper:** segment every result table by SBC class. That single move converts "we compared five models" into "we characterise *when* each model wins," which is a materially stronger claim.

### 1.3 ML and deep learning

- **Gradient-boosted trees (LightGBM / XGBoost)** dominate tabular retail forecasting in practice and in competition results, particularly with lag/rolling/calendar feature engineering. Reported as strong across group-level revenue and profit WMAPE evaluations.
- **Temporal Fusion Transformer (Lim et al., 2021)** — multi-horizon, handles static covariates, known-future inputs (promotions, holidays) and observed inputs separately, and exposes *variable selection weights* and *interpretable attention*. Studies on M5 (30,490 series, 10 stores) report Transformer-family models including TFT significantly outperforming traditional baselines.
- **N-BEATS / N-HiTS / DeepAR** — additional deep baselines. DeepAR is natively probabilistic, which matters here.
- Recent comparative work evaluating statistical baselines, tree ensembles, and deep architectures (N-BEATS, N-HiTS, TFT) on retail data with intermittency, missingness and product churn is directly relevant prior art — read it before finalising your benchmark set.

**Key honest reading:** the "which model is most accurate on retail sales" question is *saturated*. Dozens of papers answer it. Do not make it your contribution. Make it your Table 3.

### 1.4 Probabilistic forecasting and its evaluation

The M5 competition ran an explicit **Uncertainty** track requiring nine quantiles (0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995), scored by **Scaled Pinball Loss**, aggregated as Weighted SPL. The accuracy track used **RMSSE**, aggregated by dollar-sales weights into **WRMSSE**. Adopt these metrics — they are scale-free, robust to zeros (RMSSE avoids division by near-zero values), and immediately recognisable to reviewers.

**Conformal prediction** provides distribution-free, finite-sample coverage: split conformal guarantees the true value falls in the interval with probability ≥ 1 − α, for *any* black-box model. Recent work adapts it to non-exchangeable time-series data via calibration reweighting, adaptive residual updating, and online target-coverage tuning. It is explicitly recognised as valuable for inventory planning with quantified uncertainty. Practically available via `MAPIE` and Nixtla's conformal utilities.

---

## 2. Strand B — Inventory optimization and predict-then-optimize

- **Newsvendor / critical ratio** — the classical foundation. Optimal order quantity is the critical quantile of demand.
- **Data-driven newsvendor** — a mature literature replacing distribution assumptions with direct quantile estimation. ML-based quantile regression obtains order quantities in a *single step*, avoiding a separate estimation-then-optimization pipeline. Deep learning approaches estimate the target conditional quantile function directly with a DNN.
- **Decision-focused learning / SPO** (Elmachtoub & Grigas, 2022) — a loss measuring decision error rather than prediction error, with a convex, Fisher-consistent surrogate (SPO+). Established that minimising MSE does not minimise decision regret.
- **Safety stock theory** — the widely used **King (2011)** formulation combining both demand and lead-time variability:

  $$SS = z\sqrt{\overline{LT}\,\sigma_D^2 + \overline{D}^2\,\sigma_{LT}^2}$$

  Setting σ_LT = 0 recovers the common simplification SS = z·√LT·σ_D. Note that combining both sources statistically gives a *lower* total than summing the two separate calculations.
- **Cycle service level vs fill rate** — CSL is the probability of no stockout in a replenishment cycle; fill rate is the fraction of demand met from stock. Fill rate is almost always higher than CSL, sometimes substantially. Standard z-based safety stock formulas target **CSL**, not fill rate. Report both, and be explicit about which one your z targets — conflating them is a common and easily-caught error.
- **Supplier selection & lot sizing MILP** — a long-established OR literature: MILP models for multi-product supplier selection with minimum package quantities, minimum order values, and quantity discounts; MINLP for joint supplier selection, economic lot-sizing and order allocation. Most published work uses CPLEX/Gurobi; **OR-Tools** (CP-SAT, SCIP wrapper) is a free, adequate substitute at your problem scale.

---

## 3. Strand C — XAI in supply chain

- SHAP (Lundberg & Lee, 2017) and LIME are the dominant post-hoc methods; TreeSHAP gives exact Shapley values for tree ensembles in polynomial time, which makes LightGBM/XGBoost the pragmatic choice for the explainable forecasting component.
- Applied work exists: SHAP and LIME interpretability analysis identifying cost and inventory drivers in supply-chain decision support systems (*Production Planning & Control*, 2024); permutation-importance and SHAP quantification across retail datasets; TFT variable-selection weights used as native interpretability.
- Reviews consistently identify **explainability as an adoption barrier**, and note that most SC AI approaches afford little to no explainability. A 2023 *IJPR* review surveys neurosymbolic approaches as one response.
- **XAI evaluation** is itself a field. Relevant quantitative metrics: **faithfulness/fidelity** (deletion–insertion AUC, sufficiency/comprehensiveness), **stability** (consistency of explanations under small input perturbations), **infidelity** (Yeh et al.). Evidence suggests SHAP is more robust than LIME, with dataset characteristics driving explanation consistency. Taxonomies of XAI evaluation metrics exist and should be cited when you justify your chosen measures.

---

## 4. The research gap — stated precisely

Lay the three strands side by side:

```
Strand A  Forecasting          →  optimises  ACCURACY.   Stops at the number.
Strand B  Inventory / OR       →  optimises  COST.       Usually assumes demand
                                                          distribution is given.
Strand C  XAI in supply chain  →  explains   PREDICTIONS. Stops at feature
                                                          importance for a forecast.
```

### What is genuinely missing

**Gap 1 — Explanations target the prediction, not the decision.**
Existing XAI-for-supply-chain work reports SHAP values over forecast features. No mainstream work explains the *constrained procurement recommendation*: why 200 units and not 120, how much of that is expected demand vs. buffer vs. lead-time risk vs. MOQ rounding, and what would have to change for the answer to be different. This is the buyer's actual question and it is unanswered.

**Gap 2 — Explanation quality is asserted, not measured, in this domain.**
Papers show a SHAP waterfall plot and declare the system interpretable. Quantitative faithfulness/stability metrics are standard in the XAI methods literature but rarely applied in supply-chain applications.

**Gap 3 — Evaluation on forecast error rather than decision cost.**
Despite the SPO result being nearly a decade old, applied ERP/inventory ML papers overwhelmingly report MAE/RMSE/MAPE and stop. Very few close the loop through an inventory simulator to realised cost, and fewer still show whether the accuracy ranking survives the translation.

**Gap 4 — Uncertainty is truncated at the interface.**
Where probabilistic forecasts are produced, the mean is usually what gets passed to the inventory rule. The full predictive distribution — and its *calibration* — is rarely propagated into the safety-stock/order-quantity computation, and conformal calibration of that interface is largely unexplored in applied retail replenishment.

**Gap 5 — Reproducibility of the supply side.**
Procurement data is commercially sensitive, so published inventory-ML studies use proprietary data and cannot be replicated. There is no shared, declared, parameterised supply-side simulation layer bolted onto a standard public retail dataset.

### Your contribution, mapped

| Gap | Your response | Novelty strength |
|---|---|---|
| 1 | Exact additive decision attribution + counterfactual sensitivity over the MILP output | **High** — this is your claim |
| 2 | Deletion-AUC faithfulness + perturbation stability + small user study on explanations | Medium-high |
| 3 | Rolling-origin inventory digital twin; cost as primary metric; ranking-inversion analysis | Medium-high |
| 4 | Quantile forecasts → conformal calibration → lead-time-demand convolution → order-up-to | Medium |
| 5 | Released parameterised supply simulator + scenario sweep on public data | Medium (high *practical* value) |

**Honest assessment:** Gaps 3–5 are engineering-and-protocol contributions — real, useful, publishable at a good conference, but incremental. **Gap 1 is your novelty.** Write the paper so that Gap 1 is the headline and 2–5 are the supporting apparatus. If you invert that emphasis you will get "insufficient novelty" reviews.

---

## 5. Prior art you MUST check before finalising

Search these queries on Google Scholar, IEEE Xplore, ScienceDirect and arXiv, and record every hit with a one-line note in `docs/lit-log.md`. If any of them already does decision-level attribution over a constrained replenishment optimizer, you must pivot the emphasis.

```
"explainable" AND ("order quantity" OR "replenishment") AND (SHAP OR counterfactual)
"decision-focused" AND "inventory" AND "explanation"
"prescriptive analytics" AND "interpretab*" AND "supply chain"
"counterfactual explanation" AND "optimization" AND ("procurement" OR "sourcing")
"explaining optimization" OR "explainable optimization" OR "XAI for OR"
Favorita AND (SHAP OR explainab*)
"inventory simulation" AND "forecast" AND "cost" AND "benchmark"
```

Also read specifically:
- The counterfactual-explanations-for-optimization literature (there is an emerging "explainable constraint programming / explainable OR" community — check CPAIOR and IJCAI XAI workshops). **If a mature body of work exists there, cite it and position yourself as applying it to replenishment, which is still a valid and defensible contribution.**
- Zalando's 2025/26 publication on replenishment optimization with an extended (R, s, Q) policy and probabilistic models — industrially relevant, strong related-work citation.
- Any recent hybrid forecasting-uncertainty + adaptive inventory planning frameworks in retail supply chains.

### Rule for the related-work section

For each cited paper, one sentence of what it does and one sentence of what it does *not* do that you do. A related-work section that only summarises is filler; one that positions is a weapon.

---

## 6. Suggested related-work table for the paper

| Work | Forecast | Uncertainty | Inventory policy | Optimizer | Explains forecast | **Explains decision** | Decision-cost eval |
|---|---|---|---|---|---|---|---|
| Classical ERP (min-max) | moving avg | ✗ | static ROP | ✗ | ✗ | ✗ | ✗ |
| ML forecasting papers (many) | ML | some | ✗ | ✗ | some SHAP | ✗ | ✗ |
| Data-driven newsvendor | quantile | ✓ | newsvendor | closed form | ✗ | ✗ | ✓ |
| SPO / decision-focused | ML | ✗ | varies | LP/MILP | ✗ | ✗ | ✓ |
| XAI-in-SC applied work | ML | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **XAI-ERP (this work)** | ML + DL | ✓ conformal | (R,s,Q)/order-up-to | MILP | ✓ | **✓** | ✓ |

Fill in the real rows with real citations. This table alone justifies the paper's existence to a reviewer in fifteen seconds — put it on page 2.
