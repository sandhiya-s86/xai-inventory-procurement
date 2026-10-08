# Methodological Decision Log

Append-only. Every non-obvious choice, with its justification and the date it was made.

**Why this file exists:** the methodology section of your paper is a rewrite of this log, and your viva answers come from it verbatim. Six weeks after making a choice you will not remember why you made it, and "I don't recall" is a bad answer to a examiner asking why you forward-filled oil prices.

**Format:** one entry per decision. Never edit an old entry — if you reverse a decision, add a new entry that supersedes it and note which.

---

## Template

```
### D-NNN — <short title>
Date:        YYYY-MM-DD
Context:     What forced the choice
Options:     A / B / C considered
Decision:    What was chosen
Rationale:   Why, with evidence or citation
Consequences: What this rules in and rules out downstream
Supersedes:  D-NNN (if applicable)
```

---

### D-001 — Two datasets, separate tracks, not merged
Date: 2026-08-18
Context: Favorita and Online Retail II both proposed as data sources.
Options: (A) merge into one corpus; (B) Favorita primary + Online Retail II external validation; (C) Favorita only.
Decision: **B.**
Rationale: The two represent unrelated businesses with incompatible schemas, granularities and demand regimes. Merging produces a meaningless combined panel. Keeping them separate supports a genuine generalisation claim: the framework is evaluated on a smooth Ecuadorian grocery panel *and* a lumpy UK wholesale gift catalogue.
Consequences: Two full experimental tracks to run; Online Retail II gets weekly granularity and 3 folds rather than daily and 5, because its series are shorter and sparser.

---

### D-002 — Supply-side data is simulated and declared
Date: 2026-08-18
Context: Neither dataset contains inventory positions, suppliers, lead times, purchase orders or costs. No public dataset joins these to real sales.
Options: (A) keep searching for real procurement data; (B) simulate and declare; (C) drop the procurement component.
Decision: **B**, with a four-scenario sweep and a released generator.
Rationale: Procurement terms are commercially confidential; the search in (A) has unbounded cost and a known outcome. (C) removes the contribution. Declaring, parameterising and sweeping converts an unavoidable weakness into a sensitivity analysis.
Consequences: Every claim about procurement cost is scenario-conditional and must be stated as such. The generator and its seed become a released artefact.

---

### D-003 — Probabilistic (quantile) forecasting rather than point forecasting
Date: 2026-08-18
Context: The inventory decision requires a quantile of lead-time demand, not a mean.
Options: (A) point forecast + normal-assumption safety stock (the conventional approach); (B) quantile forecasts feeding an empirical order-up-to level.
Decision: **B**, with (A) retained as baseline P2 for the ablation.
Rationale: The newsvendor optimum is a quantile. A point forecast plus a normality assumption is exactly what conventional ERP does, and testing against it is RQ2.
Consequences: Requires nine-quantile forecasts, conformal calibration, Monte-Carlo LTD sampling, and probabilistic metrics (SPL, coverage) alongside point metrics.

---

### D-004 — Evaluation on realised decision cost, not forecast error
Date: 2026-08-18
Context: Standard practice in this application area is to report MAE/RMSE/MAPE and stop.
Options: (A) forecast metrics only; (B) forecast metrics + inventory digital twin producing realised cost and service.
Decision: **B**, with cost as the primary metric and accuracy as secondary.
Rationale: Minimising prediction error does not minimise decision regret (Elmachtoub & Grigas, 2022). A forecast-accuracy table does not demonstrate business value.
Consequences: A digital twin must be built and verified before any headline result exists. Policies must be compared at matched service level, not on raw cost.

---

### D-005 — Explanation targets the decision, not only the prediction
Date: 2026-08-18
Context: Existing XAI work in this domain explains the forecast via SHAP.
Options: (A) SHAP on the forecast only; (B) SHAP plus an exact analytic decomposition of the recommended order quantity, a Shapley split of the safety-stock buffer, counterfactual thresholds, and supplier-choice regret.
Decision: **B.**
Rationale: The buyer's question is "why order 200?", not "why forecast 245?". The optimizer's structure is known, so the decomposition can be *exact* rather than approximate — a stronger property than any post-hoc attribution can offer.
Consequences: This is the paper's primary novelty claim. It is protected by `test_decomposition_sums` and `test_shapley_efficiency`; if those fail, the claim fails.

---

<!-- Add D-006 onward as you go. Suggested near-term entries:
     - lost sales vs backorder, per dataset
     - cancellation handling in Online Retail II (drop vs net)
     - oil price forward-fill
     - handling of the 2016 earthquake window
     - whether to move to the SKU-level Favorita release (decided at gate G1)
     - MOQ/case-pack parameter grounding
     - solver time limit and accepted MIP gap
-->
