# 04 — Methodology

## 0. Notation

| Symbol | Meaning |
|---|---|
| $i$ | item (SKU or store×family series) |
| $s$ | supplier |
| $t$ | period (day) |
| $h$ | forecast horizon step, $h = 1 \dots H$ |
| $D_{i,t}$ | demand for item $i$ in period $t$ (random) |
| $\hat{q}^{(\tau)}_{i,t+h}$ | forecast of the $\tau$-quantile of $D_{i,t+h}$ |
| $L_s$ | lead time of supplier $s$ (random), mean $\bar{L}_s$, sd $\sigma_{L_s}$ |
| $R$ | review period (days between replenishment decisions) |
| $\text{LTD}_{i,s}$ | lead-time demand: total demand over $L_s + R$ |
| $\text{OH}_{i,t}$, $\text{IT}_{i,t}$, $\text{BO}_{i,t}$ | on-hand, in-transit, backordered |
| $\text{IP}_{i,t}$ | inventory position $= \text{OH} + \text{IT} - \text{BO}$ |
| $S_{i,t}$ | order-up-to level |
| $SS_{i,t}$ | safety stock |
| $h_i$, $p_i$ | holding and shortage cost per unit per period |
| $c_{i,s}$ | unit purchase cost |
| $K_s$ | fixed ordering cost per PO |
| $Q_{i,t}$ | recommended order quantity |

---

## 1. Pipeline overview

```
 Stage 0  Series construction + SBC classification
 Stage 1  Feature engineering
 Stage 2  Probabilistic demand forecasting        →  q̂^(τ) for τ ∈ T
 Stage 3  Conformal calibration                   →  calibrated quantiles
 Stage 4  Lead-time demand distribution + risk    →  P(stockout), E[short]
 Stage 5  Inventory policy → target level S       →  raw order quantity
 Stage 6  MILP procurement optimization           →  final Q, timing, supplier
 Stage 7  Two-level explanation                   →  SHAP + exact decision attribution
 Stage 8  Inventory digital twin (evaluation)     →  realised cost, service level
```

Stages 4–7 are where the contribution lives. Stages 1–3 are competent standard practice.

---

## 2. Stage 0 — Series construction and demand classification

1. Build a dense daily panel per series; **do not** fill structurally-missing dates (store closures) with zeros — mask them.
2. Distinguish **structural zeros** (product not yet listed / delisted / store closed) from **demand zeros** (listed, no sale). Leading zeros before first sale are structural: truncate. This distinction materially changes ADI.
3. Compute per series over the training window:

$$\text{ADI}_i = \frac{n_i}{z_i}, \qquad CV^2_i = \left(\frac{\sigma_{D^+_i}}{\mu_{D^+_i}}\right)^{2}$$

where $n_i$ = number of periods, $z_i$ = number of non-zero demand periods, and $D^+$ = demand in non-zero periods only.

4. Assign the SBC class using cutoffs **ADI = 1.32**, **CV² = 0.49**:

| | CV² < 0.49 | CV² > 0.49 |
|---|---|---|
| ADI < 1.32 | Smooth | Erratic |
| ADI > 1.32 | Intermittent | Lumpy |

5. **Stratify every downstream result by this class.** Report the class distribution for both datasets as a paper figure.

> Treat SBC as a *reporting stratification*, not a model-selection rule. The literature is not settled on the sufficiency of the ADI/CV² cutoffs, and letting the data tell you which model wins per class is a stronger result than assuming it.

---

## 3. Stage 1 — Feature engineering

All features must be computed **causally** — using only information available at the forecast origin. A single leaked future value invalidates the entire experiment, and it is the most common fatal error in student forecasting projects.

| Group | Features |
|---|---|
| Calendar | day-of-week, day-of-month, week-of-year, month, quarter, year, is_weekend, is_month_start/end, **payday flags (15th, month-end)**, days-since-start |
| Cyclical encodings | sin/cos transforms of dow, doy, month |
| Lags | $y_{t-1}, y_{t-7}, y_{t-14}, y_{t-28}, y_{t-365}$ — lags must be ≥ H to be usable at all horizons, or use direct multi-horizon models |
| Rolling statistics | mean / std / min / max / median over windows {7, 14, 28, 91, 365}, all **shifted by H** |
| Exponentially weighted | EWM mean with α ∈ {0.1, 0.3, 0.5} |
| Intermittency features | days since last non-zero demand, rolling proportion of zero days, rolling ADI |
| Promotion | `onpromotion`, its lags and leads (**known-future** — promotions are planned, so future values are legitimately available), rolling promo intensity |
| Calendar events | holiday type / locale / transferred flags, joined on the **correct locale key** (national → all; regional → match `state`; local → match `city`) |
| Exogenous | oil price (fwd-filled), store `transactions` (lagged only — this is *observed*, not known-future) |
| Static covariates | store_nbr, city, state, store type, cluster, family, class, perishable flag |
| Price (Online Retail II) | unit price, price relative to trailing mean, price-change flag |
| Target transforms | $\log(1+y)$ for training; invert for evaluation. Note this biases the mean but not the quantiles — a further reason to forecast quantiles. |

**Encoding:** LightGBM handles categoricals natively; TFT takes them as learned embeddings. Avoid target encoding unless computed strictly within the training fold — it is a classic leakage vector.

---

## 4. Stage 2 — Probabilistic demand forecasting

### 4.1 Why probabilistic, not point

The downstream inventory decision needs a **quantile** of demand, not its mean (newsvendor). Forecasting a distribution and taking the required quantile is strictly more informative than forecasting a mean and bolting on a normality assumption in the safety-stock formula — which is what conventional ERP does and what most ML-for-inventory papers inherit without questioning.

Target quantile set (borrowed from M5-Uncertainty for comparability):

$$\mathcal{T} = \{0.005,\ 0.025,\ 0.165,\ 0.250,\ 0.500,\ 0.750,\ 0.835,\ 0.975,\ 0.995\}$$

plus any service-level quantiles your cost ratios imply.

### 4.2 Model set

| Tier | Model | Probabilistic? | Role |
|---|---|---|---|
| B0 | Seasonal naive | via empirical residual quantiles | Floor. Must be beaten. |
| B1 | Moving average (28d) | ✗ / empirical | **The ERP incumbent.** This is the policy you claim to beat. |
| B2 | AutoARIMA | ✓ (Gaussian intervals) | Classical baseline |
| B3 | ETS / Holt-Winters | ✓ | Seasonal baseline |
| B4 | Croston / SBA / TSB | point + empirical | **Intermittent specialists** |
| B5 | ADIDA / IMAPA | point | Temporal aggregation |
| M1 | **LightGBM quantile regression** (one model per τ, or one multi-output) | ✓ native pinball objective | Primary ML model |
| M2 | XGBoost | ✓ quantile objective | Comparison |
| M3 | **Temporal Fusion Transformer** | ✓ native quantile heads | Primary DL model; also gives variable-selection weights |
| M4 | DeepAR | ✓ native | Probabilistic DL comparison |
| M5 | N-BEATS / N-HiTS | point (+ conformal) | Strong DL point baselines |
| E1 | Ensemble (weighted quantile average of M1 + M3) | ✓ | Usually wins; report separately |

**Practical recommendation:** M1 (LightGBM-quantile) is your workhorse. It trains in minutes, handles 1,782 series in a single global model, works with TreeSHAP for *exact* Shapley attribution, and is competitive with or better than deep models on tabular retail data. M3 (TFT) is the "we also tried the sophisticated thing" entry and gives you native interpretability to compare against SHAP. If compute is tight, cut M4 and M5 before cutting M1 or M3.

### 4.3 Training objective

Pinball (quantile) loss:

$$\mathcal{L}_\tau(y, \hat{q}) = \begin{cases} \tau\,(y - \hat{q}) & y \ge \hat{q} \\ (1-\tau)(\hat{q} - y) & y < \hat{q}\end{cases}$$

Global model across all series (one model, series identity as a feature) rather than per-series models: better sample efficiency, handles short/new series, and is standard in modern retail forecasting.

**Quantile crossing** ($\hat{q}^{(0.25)} > \hat{q}^{(0.75)}$) will occur when quantiles are fit independently. Fix by post-hoc sorting (isotonic rearrangement) and report that you did.

---

## 5. Stage 3 — Conformal calibration

Raw ML quantiles are frequently miscalibrated: the nominal 90% interval covers 78% of outcomes. Feeding a miscalibrated quantile into a safety-stock formula silently destroys the service-level guarantee — a failure mode almost never checked in applied papers, and worth a paragraph of its own.

**Split conformal prediction** gives distribution-free finite-sample coverage $\ge 1-\alpha$ for any black-box model. Time-series data is not exchangeable, so use a time-series-adapted variant (calibration-set reweighting, adaptive residual updating, or online coverage-target adjustment).

Procedure:
1. Split the training window into proper-train and a **calibration set** that is *temporally after* proper-train.
2. Compute conformity scores on the calibration set. For quantile regression use CQR (Conformalized Quantile Regression): $E_j = \max\{\hat{q}^{(\alpha/2)}(x_j) - y_j,\; y_j - \hat{q}^{(1-\alpha/2)}(x_j)\}$.
3. Take the $\lceil (n+1)(1-\alpha)\rceil$-th order statistic of $\{E_j\}$ as the correction $\hat{E}$.
4. Calibrated interval: $[\hat{q}^{(\alpha/2)} - \hat{E},\ \hat{q}^{(1-\alpha/2)} + \hat{E}]$.

**Report empirical coverage before and after calibration, per SBC class.** This becomes a table in the paper and directly supports RQ2. Tooling: `MAPIE`, or Nixtla's conformal utilities.

---

## 6. Stage 4 — Lead-time demand and risk

The quantity at risk is not demand over one day; it is demand over the **protection interval** $L_s + R$.

### 6.1 Lead-time demand distribution

Two routes, both worth implementing:

**(a) Analytical (fast, used inside the optimizer).** Mean and variance of LTD under random lead time (Hadley–Whitin):

$$\mu_{\text{LTD}} = \bar{L}\,\bar{D}, \qquad \sigma^2_{\text{LTD}} = \bar{L}\,\sigma_D^2 + \bar{D}^2\,\sigma_L^2$$

**(b) Monte Carlo (accurate, used for reported risk metrics).** For $m = 1 \dots M$ ($M = 10{,}000$):
1. Sample $L^{(m)} \sim \text{Gamma}$ (fitted per supplier).
2. Sample a demand path over $\lceil L^{(m)} + R\rceil$ days by inverse-transform sampling from the calibrated forecast quantiles per day (interpolating between the nine quantiles), preserving a fitted temporal dependence structure (Gaussian copula on the ranks — independent daily sampling badly understates the variance of the sum).
3. Sum to get $\text{LTD}^{(m)}$.

The empirical distribution of $\{\text{LTD}^{(m)}\}$ makes everything below straightforward and assumption-light. **This is the recommended primary method** — using MC rather than the normal approximation is defensible, easy, and separates you from the textbook approach.

### 6.2 Risk metrics delivered to the user

$$P(\text{stockout}) = \Pr(\text{LTD} > \text{IP}) \approx \frac{1}{M}\sum_m \mathbb{1}[\text{LTD}^{(m)} > \text{IP}]$$

$$\mathbb{E}[\text{units short}] = \frac{1}{M}\sum_m \max(0,\ \text{LTD}^{(m)} - \text{IP})$$

$$\text{days-to-stockout} = \min\Big\{ d : \sum_{k=1}^{d}\hat{q}^{(0.5)}_{t+k} > \text{IP} \Big\}$$

Report days-to-stockout as an interval, not a point — the 10th and 90th percentiles across MC paths. "Expected stockout in 12 days (80% CI: 8–19)" is honest; "expected stockout: 12 days" is not.

---

## 7. Stage 5 — Inventory policy and target level

### 7.1 Safety stock — two variants, compared

**Variant A — classical (the King formulation), the incumbent:**

$$SS^{\text{King}} = z_{\text{CSL}}\sqrt{\bar{L}\,\sigma_D^2 + \bar{D}^2\,\sigma_L^2}$$

with $\sigma_D$ from historical demand and $z$ from the normal quantile. Setting $\sigma_L = 0$ recovers the simpler $SS = z\sqrt{L}\,\sigma_D$ used by most ERPs.

**Variant B — quantile/empirical (yours):**

$$S = F^{-1}_{\text{LTD}}(\text{CR}), \qquad SS^{\text{quant}} = S - \mu_{\text{LTD}}$$

where the critical ratio comes from the cost structure:

$$\text{CR} = \frac{C_u}{C_u + C_o} = \frac{p}{p + h}$$

and $F^{-1}_{\text{LTD}}$ is the empirical quantile of the Monte-Carlo LTD samples — **no normality assumption**.

The A-vs-B comparison *is* RQ2. It is a clean, well-posed ablation.

> **Service-level discipline.** $z_{\text{CSL}}$ targets **cycle service level** — the probability of no stockout in a cycle — not **fill rate**, which is the fraction of demand met from stock and is almost always higher. Report both. Conflating them is a common error, and stating the distinction explicitly signals competence to a reviewer.

### 7.2 Policies implemented

| Policy | Rule | Role |
|---|---|---|
| **Static min-max** | Fixed $s$, $S$ from trailing 12-month averages, refreshed annually | **The ERP baseline you must beat** |
| $(R, s, Q)$ | Periodic review; if $\text{IP} \le s$, order fixed $Q$ (EOQ) | Classical |
| $(R, S)$ order-up-to | Periodic review; order $S - \text{IP}$ | Primary policy |
| $(R, s, S)$ | Order up to $S$ only when $\text{IP} \le s$ | Adds fixed-cost awareness |

Raw order quantity under order-up-to:

$$Q^{\text{raw}}_{i,t} = \max\left(0,\ S_{i,t} - \text{IP}_{i,t}\right) = \max\left(0,\ \mu_{\text{LTD}} + SS - \text{OH} - \text{IT} + \text{BO}\right)$$

This expression is **exactly additive in interpretable terms** — which is precisely what Stage 7 exploits.

---

## 8. Stage 6 — Procurement optimization (MILP)

$Q^{\text{raw}}$ ignores MOQ, case packs, supplier capacity, budget, fixed ordering cost, and multi-supplier allocation. The MILP handles all of it jointly over a rolling planning horizon $T$ (28 days).

### 8.1 Decision variables

| Var | Domain | Meaning |
|---|---|---|
| $x_{i,s,t}$ | $\mathbb{Z}_{\ge 0}$ | units of $i$ ordered from $s$ at $t$ |
| $w_{i,s,t}$ | $\mathbb{Z}_{\ge 0}$ | number of case packs (so $x = \text{pack}_{i,s}\, w$) |
| $y_{i,s,t}$ | $\{0,1\}$ | line placed for $(i,s,t)$ |
| $z_{s,t}$ | $\{0,1\}$ | PO issued to $s$ at $t$ (triggers fixed cost) |
| $I_{i,t}$ | $\mathbb{R}_{\ge 0}$ | projected on-hand at end of $t$ |
| $B_{i,t}$ | $\mathbb{R}_{\ge 0}$ | projected shortage in $t$ |

### 8.2 Objective

$$\min \sum_{t\in T}\Bigg[ \underbrace{\sum_{i,s} c_{i,s}\,x_{i,s,t}}_{\text{purchase}} + \underbrace{\sum_{s} K_s\, z_{s,t}}_{\text{ordering}} + \underbrace{\sum_i h_i\, I_{i,t}}_{\text{holding}} + \underbrace{\sum_i p_i\, B_{i,t}}_{\text{shortage}} + \underbrace{\sum_{i,s}\rho_{s}\,x_{i,s,t}}_{\text{supplier risk penalty}} \Bigg]$$

where $\rho_s$ is a per-unit risk penalty derived from supplier unreliability (e.g. $\rho_s \propto \text{CV}(L_s)\cdot(1-\text{OTD}_s)$). Including it makes this genuinely **multi-objective**, and $\rho$ becomes a knob for a Pareto-frontier plot (cost vs. supplier risk) — a strong, cheap figure for the paper.

### 8.3 Constraints

**Inventory balance** (goods ordered at $t$ arrive at $t + L_s$):

$$I_{i,t} - B_{i,t} = I_{i,t-1} - B_{i,t-1} + \sum_{s} x_{i,s,\,t-\lceil \bar{L}_s\rceil} - d_{i,t}, \qquad \forall i,t$$

with $d_{i,t}$ set to the calibrated **quantile** forecast $\hat{q}^{(\text{CR})}_{i,t}$, not the mean. This is the deterministic-equivalent / robust approximation of the underlying stochastic program — **state this explicitly as a modelling assumption; do not pretend it solves the stochastic program.**

**Case packs:** $x_{i,s,t} = \text{pack}_{i,s}\, w_{i,s,t}$

**Minimum order quantity (big-M linking):**
$$\text{MOQ}_{i,s}\, y_{i,s,t} \le x_{i,s,t} \le M_{i}\, y_{i,s,t}$$

**Line implies PO:** $y_{i,s,t} \le z_{s,t}$

**Supplier capacity:** $\sum_i x_{i,s,t} \le \text{Cap}_{s,t}\, z_{s,t}$

**Budget per period:** $\sum_{i,s} c_{i,s}\, x_{i,s,t} \le \text{Budget}_t$

**Safety-stock floor (soft, with slack $\xi$ penalised in the objective):**
$$I_{i,t} + \xi_{i,t} \ge SS_{i,t}$$
Soft rather than hard, so the model degrades gracefully instead of returning INFEASIBLE — an infeasible optimizer in a live ERP is worse than a suboptimal one.

**Warehouse capacity (optional):** $\sum_i v_i I_{i,t} \le V_{\max}$ with $v_i$ the unit volume.

### 8.4 Solver

`OR-Tools` **CP-SAT** for the integer-heavy formulation (excellent on MOQ/pack/binary structure, free, and handles this scale comfortably), or the OR-Tools MPSolver wrapper over SCIP if you prefer an LP-relaxation-based branch-and-bound view. Note in the paper that published supplier-selection/lot-sizing work typically uses CPLEX/Gurobi and that OR-Tools is an open substitute at this scale — that is an honest and reproducibility-positive framing.

**Decomposition for tractability:** solve per item (or per small item group) when the only coupling constraints are the budget and supplier capacity; otherwise solve the full model with a time limit and report the MIP gap. **Always report the gap and the wall-clock time** — a paper that says "we solved it" without either is not reproducible.

---

## 9. Stage 7 — Two-level explanation

> This is the contribution. Everything above is infrastructure for this section.

### 9.1 Level 1 — Forecast explanation (established)

**TreeSHAP** on the LightGBM quantile model gives exact Shapley values in polynomial time.

- **Global:** mean |SHAP| ranking, beeswarm, dependence plots. Answers "what drives demand in this catalogue."
- **Local:** waterfall for the specific item-date. Answers "why 245 units for *this* item *this* week."
- **TFT native:** variable-selection weights and attention over the encoder. Compare against SHAP as a cross-check; agreement between two independent attribution mechanisms is itself a reportable finding.

This part is standard. Do not oversell it.

### 9.2 Level 2 — Decision attribution (the novel part)

SHAP explains a *prediction*. The user is asking about a *recommendation*. Because the recommendation's structure is known analytically, it can be decomposed **exactly** — no approximation, no sampling.

**Step 1 — Additive decomposition of the raw quantity.**

$$Q^{\text{raw}} = \underbrace{\mu_{\text{LTD}}}_{\text{expected demand over cover}} + \underbrace{SS}_{\text{uncertainty buffer}} - \underbrace{\text{OH}}_{\text{stock you already hold}} - \underbrace{\text{IT}}_{\text{stock already inbound}} + \underbrace{\text{BO}}_{\text{unmet backlog}}$$

Each term is signed, in units, and sums exactly to $Q^{\text{raw}}$.

**Step 2 — Exact Shapley split of the safety-stock buffer.**

$SS$ mixes two sources of uncertainty and is *not* additive in them (it is a square root of a sum). With exactly two "players" — demand variability $\mathcal{D}$ and lead-time variability $\mathcal{L}$ — the Shapley value has a closed form:

$$\phi_{\mathcal{D}} = \tfrac{1}{2}\Big[SS(\mathcal{D}) - SS(\emptyset)\Big] + \tfrac{1}{2}\Big[SS(\mathcal{D},\mathcal{L}) - SS(\mathcal{L})\Big]$$
$$\phi_{\mathcal{L}} = \tfrac{1}{2}\Big[SS(\mathcal{L}) - SS(\emptyset)\Big] + \tfrac{1}{2}\Big[SS(\mathcal{D},\mathcal{L}) - SS(\mathcal{D})\Big]$$

with $SS(\emptyset)=0$, $SS(\mathcal{D}) = z\sqrt{\bar{L}\sigma_D^2}$, $SS(\mathcal{L}) = z\bar{D}\sigma_L$, $SS(\mathcal{D},\mathcal{L}) = z\sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$.

Efficiency holds exactly: $\phi_{\mathcal{D}} + \phi_{\mathcal{L}} = SS$. This lets the system say *"38 of the 71 buffer units exist because your supplier's lead time is unreliable, not because demand is volatile"* — an actionable statement no SHAP plot produces. It also connects Level 1 and Level 2 conceptually: the same axiomatic machinery, applied once to the model and once to the decision.

**Step 3 — Constraint adjustments.**

$$Q^{\text{final}} = Q^{\text{raw}} + \Delta_{\text{pack}} + \Delta_{\text{MOQ}} + \Delta_{\text{ration}} + \Delta_{\text{consolidate}}$$

- $\Delta_{\text{pack}}$: rounding up to the case-pack multiple
- $\Delta_{\text{MOQ}}$: lifting to the supplier minimum (or dropping to zero if MOQ is not worth meeting)
- $\Delta_{\text{ration}}$: reduction because a budget or capacity constraint bound — **recover this from the constraint's dual value / from a re-solve with the constraint relaxed**
- $\Delta_{\text{consolidate}}$: extra units pulled forward to share a fixed ordering cost $K_s$

Each is recoverable exactly by re-solving the MILP with the corresponding constraint removed and differencing. Costly but exact; cache aggressively.

**Step 4 — Supplier-choice explanation by counterfactual regret.**

For each non-selected supplier $s'$, re-solve with $s'$ forced and report the objective delta:

$$\text{Regret}(s') = Z^\*(\text{force } s') - Z^\*$$

decomposed into its cost components (purchase, holding, shortage, risk penalty). The system then states: *"Supplier B is ₹8/unit cheaper but its lead-time CV of 0.42 requires 34 more buffer units; net penalty ₹1,240 over the horizon."* That is a complete, checkable answer to "why not the cheaper one."

**Step 5 — Counterfactual thresholds ("what would change my mind").**

For each key parameter, find the minimal perturbation that flips the recommendation, by bisection over the pipeline:

- *Lead time*: "If Supplier A delivered in ≤ 5 days instead of 9, the order would drop to 140 units."
- *Service level*: "At 90% CSL instead of 95%, order 165 instead of 200."
- *MOQ*: "Without the 250-unit MOQ, the optimal order is 187."
- *Current stock*: "You would not need to order at all until 26 Aug if on-hand were 130."

These map directly onto the actions a procurement manager can actually take, which is what makes them useful rather than merely interesting.

### 9.3 Explanation rendering

Natural-language templates over the numeric decomposition — deterministic, auditable, no LLM in the decision path.

```
Order 200 units of SKU-4471 from Supplier A, place today (18 Aug).

WHY 200 UNITS
  +178   expected demand over the 9-day lead time + 7-day review period
   +71   safety buffer for 95% cycle service level
           ├─ 33 units ← demand variability
           └─ 38 units ← Supplier A lead-time variability   ⚠ dominant driver
   −86   stock on hand
   −13   already in transit (PO-2291, due 21 Aug)
  ─────
   150   unconstrained requirement
   +50   rounded up to Supplier A minimum order quantity (200)
  ─────
   200   recommended order

WHY NOW      Days-to-stockout: 12 (80% interval 8–19). Ordering after 20 Aug
             pushes arrival past the projected stockout date.

WHY SUPPLIER A
             Supplier B: −₹8/unit purchase cost, but lead-time CV 0.42 requires
             +34 buffer units. Net penalty over horizon: ₹1,240.

WHAT WOULD CHANGE THIS
             Lead time ≤ 5 days      → order 140
             Service level 90%       → order 165
             MOQ removed             → order 187

TOP FORECAST DRIVERS (SHAP)
             +28%  4-week demand trend
             +11%  promotion scheduled 22–25 Aug
              +6%  month-end payday effect
              −4%  post-holiday slowdown
```

Every number in that block is traceable to a formula in this document. That traceability is the paper's core claim and should be stated as such.

---

## 10. Stage 8 — Inventory digital twin (how results are produced)

Forecast metrics do not measure the system. The simulator does.

### 10.1 Loop

```
for each backtest fold:
  initialise OH, IT, BO from the warm-up window
  for t in evaluation window:
      1. observe state (OH, IT, BO)                    # information available at t
      2. if t is a review period:
             forecast → calibrate → LTD distribution → policy → MILP → Q, supplier
             place PO; schedule arrival at t + L_realised
      3. realise arrivals due at t (apply late/short delivery per supplier OTD)
      4. realise ACTUAL demand D_t from held-out data
      5. fulfil min(OH, D_t); record lost sales or backorder
      6. accrue holding cost on end-of-period OH
      7. accrue shortage cost on unmet demand
      8. apply perishable spoilage/write-off where applicable
  aggregate costs and service metrics
```

**Discipline:** the simulator must be the *only* consumer of held-out actuals. Any forecast or policy touching step 4's data before step 4 is leakage.

### 10.2 Design details that matter

- **Unmet demand:** decide **lost sales** (grocery — realistic for Favorita) vs **backorder** (wholesale — realistic for Online Retail II) and state it. The two produce materially different cost profiles; using backorder for grocery would be wrong.
- **Warm-up:** discard the first 28–56 days of each fold so results are not dominated by arbitrary initial inventory.
- **Common random numbers:** use identical demand realisations, lead-time draws, and initial conditions across every policy compared. This is *essential* — it removes simulation noise from the comparison and dramatically tightens confidence intervals. Fix seeds per (fold, scenario), not per policy.
- **Replications:** ≥ 30 stochastic replications per (policy × scenario × fold) for confidence intervals on cost.
- Implement in `numpy`/`numba` (vectorised over items) rather than `SimPy` — the process-based abstraction buys nothing for periodic review and costs an order of magnitude in speed.

---

## 11. Backtesting protocol

**Rolling-origin evaluation with an expanding window.** Never a single random train/test split — that is leakage in a time-series setting.

```
Favorita (2013-01-01 → 2017-08-15)
  fold 1: train ≤ 2016-08-15 | calib 2016-08-16→09-15 | test 2016-09-16→10-13
  fold 2: train ≤ 2016-11-15 | calib ...               | test ...
  fold 3: train ≤ 2017-02-15 | ...
  fold 4: train ≤ 2017-05-15 | ...
  fold 5: train ≤ 2017-07-15 | calib 07-16→07-31       | test 2017-08-01→08-15
```

- Forecast horizon $H$ = 28 days (covers the longest simulated lead time + review period).
- Refit cadence: retrain per fold; **also** run a "no-refit" variant to measure degradation from concept drift — that is a cheap extra result with real practical relevance.
- Calibration set always sits strictly between train and test, temporally.
- Identical fold boundaries for every model. No exceptions.

For Online Retail II (2 years), use 3 folds at weekly granularity rather than 5 at daily — the series are too short and too sparse for a daily 5-fold scheme, and forcing it would produce noise you would then have to explain away.
