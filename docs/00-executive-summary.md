# 00 — Executive Summary

## Title

**Explainable AI-Driven Decision Support for Predictive Inventory and Procurement Optimization in ERP Systems**

Short form: **XAI-ERP**

Paper-ready title (more specific, better for a venue):

> *From Probabilistic Demand Forecasts to Explainable Procurement Decisions: A Decision-Cost-Evaluated XAI Framework for ERP Inventory Replenishment*

---

## Abstract (draft — 210 words)

Enterprise Resource Planning (ERP) systems continue to drive replenishment through static reorder points and moving-average consumption rules, producing simultaneous overstock and stockout conditions that fixed-parameter policies cannot resolve. Recent work applies machine learning to retail demand forecasting with strong accuracy gains, but stops at the forecast: the downstream procurement decision remains rule-based, and where explainability is offered it attributes the *prediction* rather than the *recommended action*. We propose XAI-ERP, an end-to-end framework that (i) produces calibrated probabilistic demand forecasts using gradient-boosted quantile regression and a Temporal Fusion Transformer, calibrated post-hoc by split conformal prediction; (ii) convolves the demand distribution with a stochastic supplier lead-time distribution to yield lead-time demand and closed-form stockout risk; (iii) solves a mixed-integer procurement program over order quantity, timing, and supplier allocation subject to minimum order quantity, capacity, and budget constraints; and (iv) generates a two-level explanation comprising SHAP attributions over the forecast and a novel additive decomposition of the recommended order quantity into forecast, safety-stock, pipeline, and constraint-rounding components, with counterfactual sensitivity on lead time and service level. The framework is evaluated not on forecast error alone but on realised inventory cost within a rolling-origin inventory digital twin, across four supply-uncertainty scenarios, on two heterogeneous public retail datasets, with Diebold–Mariano and Friedman–Nemenyi significance testing and quantitative explanation-faithfulness measurement.

---

## The research question

> Can an explainable machine-learning system jointly (a) forecast demand probabilistically, (b) quantify stockout risk under joint demand and lead-time uncertainty, and (c) optimize the procurement decision under business constraints — while producing explanations of the **recommendation** that are measurably faithful and demonstrably reduce realised inventory cost relative to conventional ERP reorder-point policies?

## Sub-questions

| RQ | Question | Answered by |
|----|----------|-------------|
| **RQ1** | Which forecasting family performs best across demand regimes (smooth / erratic / intermittent / lumpy per the SBC taxonomy), and does the ranking change between point-error and decision-cost evaluation? | Exp. A + segment analysis |
| **RQ2** | Does propagating *forecast uncertainty* (quantiles + conformal calibration) rather than a point forecast into the inventory policy reduce cost at equal service level? | Exp. B (ablation) |
| **RQ3** | Does jointly modelling supplier lead-time uncertainty with demand uncertainty improve decisions over demand-uncertainty-only safety stock? | Exp. C (scenario sweep) |
| **RQ4** | Does MILP-based multi-supplier procurement optimization outperform greedy single-supplier reorder-point replenishment on total landed cost at matched service level? | Exp. D |
| **RQ5** | Can decision-level explanations be produced that are (a) faithful to the optimizer, (b) stable under input perturbation, and (c) rated more useful than forecast-level SHAP by ERP users? | Exp. E |

RQ5 is the one that makes this a paper rather than a project. RQ1–RQ4 make it *credible*.

---

## Contribution claims (what you may honestly claim)

1. **A decision-level explanation mechanism for constrained replenishment.** An exact additive decomposition of the optimizer's recommended order quantity into interpretable components (expected lead-time demand, safety-stock buffer attributable to demand variance, buffer attributable to lead-time variance, in-transit offset, MOQ/case-pack rounding, budget-rationing adjustment), plus counterfactual sensitivities. This is *exact*, not an approximation like SHAP — because the optimizer's structure is known. **This is the primary novelty.**

2. **Decision-cost evaluation protocol on public data.** A reproducible rolling-origin inventory digital twin that converts any forecaster into realised cost/service metrics, released with declared, parameterised procurement simulation. Enables comparing forecasters on what actually matters.

3. **Empirical evidence that accuracy ranking ≠ decision ranking.** Demonstrated on 1,782+ series across two datasets and four uncertainty scenarios, with significance testing.

4. **A segment-conditioned model-selection finding.** Which forecaster to use per SBC demand class, evaluated on cost.

5. **A working ERP-integrated reference implementation** with the explanation surface exposed to the user.

## Contribution claims (what you must NOT claim)

- ❌ "Novel forecasting architecture" — you are using established models.
- ❌ "Real supplier data" — you have none. It is simulated.
- ❌ "Deployed in industry / validated in production."
- ❌ "First to apply SHAP to demand forecasting" — comprehensively false, see [02](02-literature-review-and-research-gap.md).
- ❌ Any claim about intermittent demand if you use only the family-level Favorita file.

---

## Datasets (verified — full audit in [03](03-datasets.md))

| Role | Dataset | Size | Period | Licence |
|------|---------|------|--------|---------|
| Primary | Corporación Favorita (Kaggle *Store Sales*) | 3,000,888 rows, 1,782 series | 2013-01-01 → 2017-08-15 | Kaggle competition rules; academic use permitted |
| Primary (SKU-level, for RQ1) | Corporación Favorita (original 2017 competition) | 125,497,040 rows, ~4,100 items | 2013 → 2017-08-15 | same |
| External validation | UCI Online Retail II | 1,067,371 transactions | 2009-12-01 → 2011-12-09 | CC BY 4.0, DOI 10.24432/C5CG6D |
| Simulated overlay | Supplier / lead-time / cost / MOQ parameters | generated | — | authored by us, released |

---

## Headline expected results (hypotheses to test, not claims)

| Hypothesis | Expected direction | Risk if false |
|---|---|---|
| H1: Quantile-based order-up-to beats point-forecast + static safety stock at equal CSL | 8–20% total cost reduction | Low — well supported in literature |
| H2: Cost ranking of forecasters ≠ MAE ranking for ≥25% of series | ranking inversion observed | Low — this is a known effect and is itself a finding either way |
| H3: Joint demand+lead-time uncertainty modelling helps most under Scenario C (disruption) | largest gap in disrupted scenario | Low |
| H4: MILP multi-supplier beats greedy single-supplier on landed cost | 3–10% | Medium — depends on simulated price spread |
| H5: Decision-level explanations score higher on user-rated usefulness than SHAP-only | significant on Likert | Medium — small-n study; report honestly |

Note H2 is a *safe* hypothesis: whichever way it resolves, it is publishable content. Design more of your hypotheses this way.

---

## Effort and feasibility

| Phase | Weeks | Risk |
|---|---|---|
| Data acquisition + EDA + SBC classification | 1–2 | Low |
| Forecasting benchmark (statistical + ML + DL) | 3–6 | Medium (TFT training time) |
| Conformal calibration + inventory digital twin | 7–8 | Low |
| MILP procurement optimizer | 9–10 | Medium (modelling correctness) |
| Decision-attribution + SHAP layer | 11–12 | Medium (this is the novel bit) |
| Scenario experiments + statistical validation | 13–14 | Low |
| ERP dashboard + integration | 15 | Low |
| Paper writing | 14–16 (overlap) | — |

**Cut order if time runs short:** drop TFT → drop the SKU-level 125M-row variant → drop the human study (replace with faithfulness metrics only) → drop the ERP UI (replace with a Streamlit demo + screenshots). Never cut: the digital twin, the MILP, the decision attribution. Those are the contribution.
