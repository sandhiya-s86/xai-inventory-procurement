# 01 — Problem Statement and Motivation

## 1. What ERP systems actually do today

Mainstream ERP inventory modules (SAP MM/IM, Oracle NetSuite, Microsoft Dynamics 365 BC, Odoo, ERPNext) implement replenishment through a small set of parameterised rules:

| Mechanism | How it works | Where it breaks |
|---|---|---|
| **Min-max / reorder point** | Fire a purchase requisition when on-hand ≤ *min*; order up to *max*. Both set manually. | Parameters are set once and rarely revisited. They encode last year's demand. |
| **Consumption-based planning** | Reorder point = average daily usage × lead time + safety stock, where averages come from a trailing window (typically 3–12 months). | A trailing mean is a forecast with no seasonality, no promotions, no trend, no calendar effects. |
| **EOQ / lot-size procedures** | Economic order quantity from fixed ordering and holding cost. | Assumes deterministic, stationary demand and a single supplier with infinite capacity. |
| **Static safety stock** | A fixed number of days or units of cover, set per ABC class. | Same buffer applied to a smooth fast-mover and a lumpy slow-mover. Guarantees over-buffering one and under-buffering the other. |
| **Supplier selection** | A default/preferred vendor on the item master, chosen by price or habit. | Ignores realised lead-time reliability entirely. |

Every one of these is a **point estimate plugged into a deterministic formula**. Uncertainty enters only through a single hand-tuned safety-stock number.

Open-source ERPs are no different in kind. ERPNext exposes reorder level and reorder quantity per item-warehouse; Odoo exposes reordering rules with min/max and a lead-time field. Both are static values a human types in. Both platforms expose full REST/RPC APIs and are Python-based, which is precisely why they are good integration targets for this project — the gap is in the *logic*, not the *plumbing*.

## 2. The three failures that follow

### 2.1 The point-forecast failure

A replenishment decision is a decision under uncertainty. Its optimal solution depends on the **shape of the demand distribution**, not its mean. This is the classical newsvendor result: the optimal stock level is a *quantile* of the demand distribution, set by the ratio of underage to overage cost:

$$q^\* = F^{-1}_D\!\left(\frac{C_u}{C_u + C_o}\right)$$

An ERP that computes a mean and adds a fixed buffer is implicitly asserting that a single, item-independent quantile is right for every product. For an item with a shortage-to-holding cost ratio of 9:1, the correct service quantile is 0.90; for a perishable with a 1:3 ratio it is 0.25. Static safety stock cannot express this.

### 2.2 The decoupling failure

Even ML-enhanced ERP add-ons typically stop at the forecast and hand a number to an unchanged rule engine. But minimising forecast error does not minimise decision cost. This is the core insight of **decision-focused learning / "Smart Predict-then-Optimize"** (Elmachtoub & Grigas, *Management Science*, 2022): the loss that matters is *decision regret*, and a model with lower MSE can induce higher regret. Two forecasts with identical MAE can produce wildly different costs if one errs high on a cheap item and low on an expensive one.

Practically: a system that reports "we reduced MAPE from 18% to 12%" has not demonstrated it saved anyone money.

### 2.3 The opacity failure

When a black-box model tells a procurement manager to order 200 units instead of the 120 they expected, the manager has three options: obey blindly, override, or ignore the system. In practice they override, and the ML investment is wasted.

This is not speculation — it is the consistently reported barrier in the supply-chain AI literature. Systematic reviews of AI in supply chain management identify explainability as a first-order adoption barrier, noting that the majority of AI approaches deployed in supply chain contexts afford little or no explainability and that practitioners find black-box models difficult to trust; mitigation requires supplementing models with explanations and clear accountability for incorrect recommendations (Hosseini & Ivanov and related reviews; Tandfonline *Production Planning & Control*, 2024; *International Journal of Production Research*, 2023).

**And here is the sharper point that motivates this project:** the explainability work that does exist explains the *prediction*. SHAP tells you why the model forecast 245 units. It does not tell you why the system is asking you to buy 200. Those are different questions, and the second is the one the buyer is actually asking. Between the forecast and the purchase order sit safety stock, lead-time convolution, pipeline inventory, MOQ rounding, budget rationing, and supplier allocation — none of which SHAP touches.

## 3. The cost of getting it wrong

Framing for the paper's introduction (cite real figures from literature, do not invent):

- Overstock ties up working capital and incurs holding cost typically estimated at 20–30% of item value per annum (carrying cost = capital + storage + obsolescence + insurance + shrinkage).
- Stockouts cost lost margin, and in grocery/retail, substitution and permanent customer switching.
- Perishables add a third failure mode — the Favorita item master carries a `perishable` flag precisely because the overage cost for these items is qualitatively different (write-off, not just carrying cost). This is a natural and honest source of *asymmetric* cost ratios in your experiments, and it is real data rather than an assumption.

## 4. Why *this* framing is defensible for a final-year research project

| Concern a reviewer/guide will raise | Answer built into the design |
|---|---|
| "This is just applying existing models." | The models are established *on purpose*. The contribution is the decision-attribution layer and the decision-cost evaluation protocol, not a new architecture. |
| "Where is the novelty?" | Explanation of the *constrained optimizer's output*, exactly decomposed, with counterfactuals — see [02](02-literature-review-and-research-gap.md) §4. |
| "How do you know it's better?" | Realised cost from a rolling-origin inventory simulator, not a forecast-error table. With DM and Friedman–Nemenyi tests. |
| "You don't have real supplier data." | Correct, and stated openly. The simulation is parameterised, released, and swept across four scenarios so conclusions are not artefacts of one parameter choice. |
| "Isn't this just a CRUD ERP app?" | The ERP UI is explicitly not claimed as a contribution. It is the delivery surface. |

## 5. Scope boundaries

**In scope**
- Single-echelon inventory (one warehouse/store per series). Periodic-review order-up-to and (R, s, Q) policies.
- Multi-supplier sourcing for a single item per period.
- Simulated but declared and parameterised supply-side data.
- Two public retail datasets.

**Out of scope** (state these in the paper's limitations section)
- Multi-echelon network optimization and inventory pooling.
- Transportation/routing costs.
- Price optimization or demand shaping (promotions are treated as an exogenous input feature, not a decision variable).
- Real-time streaming ERP integration; batch (nightly) scoring is assumed.
- Reinforcement-learning inventory control — noted as future work.
