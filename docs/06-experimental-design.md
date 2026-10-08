# 06 — Experimental Design

> Design the experiments before writing any modelling code. Every experiment below must produce a specific, pre-named table or figure in the paper. If an experiment does not map to an output, delete it.

---

## 1. Experiment inventory

| ID | Experiment | Answers | Output |
|---|---|---|---|
| **A** | Forecasting benchmark, all models × both datasets × 5 folds, stratified by SBC class | RQ1 | Table 3 (accuracy), Table 4 (by class), Fig. 4 (CD diagram) |
| **B** | Uncertainty ablation: point vs quantile vs conformal-calibrated quantile → inventory cost | RQ2 | Table 6, Fig. 5 (coverage) |
| **C** | Scenario sweep across four supply-uncertainty regimes | RQ3 | Table 7, Fig. 6 |
| **D** | Policy/optimizer comparison: static min-max vs (R,s,Q) vs order-up-to vs MILP multi-supplier | RQ4 | **Table 5 — the headline table** |
| **E** | Explanation evaluation: faithfulness, stability, user study | RQ5 | Table 8, Table 9, Fig. 7 |
| **F** | Ranking-inversion analysis: does accuracy ranking = cost ranking? | cross-cutting | Fig. 8 (scatter), Table 10 |
| **G** | Sensitivity: service level, cost ratio, MOQ severity, no-refit drift | robustness | Appendix tables |

---

## 2. Experiment A — Forecasting benchmark

**Factors**

| Factor | Levels |
|---|---|
| Dataset | Favorita (1,782 series), Online Retail II (products surviving the filter) |
| Model | B0–B5, M1–M5, E1 (see [04](04-methodology.md) §4.2) — 12–14 levels |
| Fold | 5 (Favorita), 3 (Online Retail II) |
| SBC class | Smooth / Erratic / Intermittent / Lumpy (stratification, not a manipulated factor) |
| Horizon | h = 1, 7, 14, 28 (reported separately — accuracy degrades with h and models rank differently at different h) |

**Controls**
- Identical folds, identical feature matrix, identical target transform for every model.
- Hyperparameters tuned on the **validation slice inside the training window only**, never on test.
- Report tuning budget (number of trials, search space) — an untuned baseline against a tuned proposal is the most common form of unfair comparison in this literature, and reviewers look for it.
- Fixed seeds; report hardware and wall-clock training time per model. Training cost is a legitimate finding: if TFT takes 40× longer than LightGBM for a 2% gain, say so.

**Primary metrics:** WRMSSE (point), Weighted Scaled Pinball Loss (probabilistic), MASE. Secondary: RMSE, MAE, sMAPE.
Do **not** use MAPE as a primary metric — it is undefined at zero demand and this data is full of zeros. Mention this explicitly; it signals you understand the metric, and MAPE misuse is a standard reviewer complaint.

---

## 3. Experiment C — The four scenarios

The scenario sweep is what converts "our supplier data is simulated" from a weakness into a sensitivity analysis. Every scenario reuses the same demand data and the same forecasts; only the supply-side parameters change.

| Scenario | Lead-time mean | Lead-time CV | On-time rate | Supplier capacity | Demand regime |
|---|---|---|---|---|---|
| **S1 — Stable** | 5 d | 0.15 | 0.98 | ample (4× mean demand) | as observed |
| **S2 — High / seasonal demand** | 5 d | 0.15 | 0.98 | ample | observed × seasonal amplification (peak weeks ×1.5) |
| **S3 — Supplier disruption** | 12 d | 0.50 | 0.80 | tight (1.5×) | as observed |
| **S4 — Compound uncertainty** | 12 d | 0.50 | 0.80 | tight | amplified |

**Hypothesis:** the advantage of joint demand + lead-time uncertainty modelling (and of the MILP's supplier-risk penalty) is smallest in S1 and largest in S3/S4. If the gap is *flat* across scenarios, that is a negative result — report it. A flat result means lead-time modelling does not pay, which is genuinely useful to know and still publishable when honestly framed.

**Run every policy in Experiment D under all four scenarios.** 4 policies × 4 scenarios × 5 folds × 30 replications = 2,400 simulator runs per dataset. This is why the simulator must be vectorised numpy, not SimPy.

---

## 4. Experiment D — The headline comparison

This is the table that decides whether the paper is accepted.

| Policy | Forecast input | Safety stock | Supplier | Constraints |
|---|---|---|---|---|
| **P0 — ERP incumbent** | 28-day moving average | fixed days-of-cover by ABC class | default (cheapest) | MOQ only |
| **P1 — Classical ROP** | 28-day MA | King formula, σ from history | default | MOQ + pack |
| **P2 — ML point forecast** | LightGBM median | King formula, σ from residuals | default | MOQ + pack |
| **P3 — ML quantile** | LightGBM quantiles + conformal | empirical LTD quantile at CR | default | MOQ + pack |
| **P4 — Full proposal** | LightGBM quantiles + conformal | empirical LTD quantile at CR | **MILP-selected** | full constraint set + budget + capacity + risk penalty |

Reporting P0→P4 as a ladder shows exactly where the gains come from. It is simultaneously the comparison *and* the ablation, which is efficient.

**Fair comparison discipline — read this twice.**

Policies with different service levels are not comparable on cost alone. A policy can always cut cost by holding less and taking stockouts. Therefore:

1. **Match service level, then compare cost.** Tune each policy's service parameter so all achieve ~95% realised fill rate on a *validation* fold, then compare total cost on test. This is the primary comparison.
2. **Also report the efficient frontier.** Sweep each policy's service parameter across a range and plot cost vs realised fill rate. If P4's curve dominates P0's everywhere, the claim is unimpeachable. **This plot is worth more than any table** — make it Figure 1.
3. Report cost decomposed into purchase / ordering / holding / shortage, so the reader can see *which* component moved.

**Common random numbers across policies** — same demand realisations, same lead-time draws, same initial state. Non-negotiable.

---

## 5. Experiment E — Explanation evaluation

Three independent angles. Report all three; do not lean on the user study alone.

### 5.1 Faithfulness (does the explanation reflect the actual mechanism?)

- **Decomposition exactness (decision level).** Assert $|Q^{\text{final}} - \sum \text{terms}| < 10^{-6}$ across every recommendation in the test set. Report the count and max residual. This is a property, not a metric — but it is the strongest possible faithfulness claim and no SHAP-based work can make it.
- **Deletion / insertion AUC (forecast level).** Progressively remove the top-k SHAP features (replacing with a baseline) and measure prediction degradation; compute the AUC. Compare TreeSHAP against permutation importance and against TFT variable-selection weights. Beware the out-of-distribution problem in perturbation-based XAI evaluation — use a marginal-distribution baseline and acknowledge the limitation.
- **Counterfactual validity.** For every generated counterfactual ("lead time ≤ 5 → order 140"), actually re-run the full pipeline with that parameter and verify the stated outcome. Report the **validity rate**; it should be 100% by construction, and demonstrating it is exactly the point.

### 5.2 Stability

Perturb inputs by small amounts ($\epsilon$ = 1%, 5% multiplicative noise on continuous features), re-generate explanations, and measure:

- Rank correlation (Spearman) of the top-10 SHAP features before/after.
- $L_1$ distance between decision-decomposition vectors, normalised by $Q$.

Report both for SHAP (forecast level) and for the decomposition (decision level). **Expect the decision-level decomposition to be markedly more stable**, because it is analytic rather than estimated. If so, that is a clean, quantified argument for your method, and it is a result you can predict but must still measure.

### 5.3 User study

Small-n, within-subjects, honestly reported.

- **Participants:** 8–15. Realistically: classmates with supply-chain coursework, a faculty member, and if at all possible 1–2 practitioners. **Report exactly who they were** — inflating "domain experts" is easily caught and fatal to credibility.
- **Design:** within-subjects, 3 conditions × ~6 decision scenarios each, condition order counterbalanced.
  - C1: recommendation only ("order 200 units")
  - C2: recommendation + forecast SHAP
  - C3: recommendation + full decision explanation (decomposition + counterfactuals + supplier regret)
- **Measures:**
  - *Objective:* decision agreement with the system; time to decide; **accuracy on a comprehension probe** ("which factor contributed most to the buffer?") — objective comprehension beats self-reported understanding.
  - *Subjective:* 5-point Likert on understanding, trust, usefulness, and confidence. Consider a validated instrument (e.g. an explanation-satisfaction scale) rather than inventing items.
  - *Behavioural:* appropriate-reliance measure — inject 2 scenarios where the recommendation is deliberately poor (built on a degraded forecast) and check whether C3 users are **better at catching it**. Blind trust is a failure, not a success; an explanation that increases override of *bad* recommendations while reducing override of good ones is the real win.
- **Analysis:** Friedman test across the three conditions, Wilcoxon signed-rank post-hoc with Holm correction.
- **Ethics:** informed consent, anonymised responses, note institutional approval status.
- **Limitations paragraph (write it now, not at the end):** small n, non-representative participants, single session, no real financial stakes. State it plainly. A modest study honestly reported is respected; an overclaimed one is not.

---

## 6. Experiment F — Ranking inversion

For each series, rank models by (a) WRMSSE and (b) realised total cost in the simulator. Then:

- Spearman correlation between the two rankings, overall and per SBC class.
- Percentage of series where the accuracy-best model is not the cost-best model.
- Scatter: Δ accuracy vs Δ cost, one point per (series, model pair). Highlight the quadrant where accuracy improved but cost worsened.

This is the empirical demonstration of the decision-focused-learning argument on real retail data with an explicit inventory simulator. It is cheap — the runs already exist from A and D — and it is one of the most quotable results in the paper.

---

## 7. Statistical validation

Statistical testing is not optional decoration. It is the difference between "our model got a lower number" and "our model is better."

| Comparison | Test | Notes |
|---|---|---|
| Two forecasting models, one series | **Diebold–Mariano** | Use HLN small-sample correction; Newey–West variance for autocorrelated multi-step errors. Do not apply to nested models without care. |
| Many models across many series | **Friedman test** → **Nemenyi post-hoc** | Rank models per series, test for overall differences, then pairwise. Plot the **critical-difference diagram**: $CD = q_\alpha\sqrt{k(k+1)/(6N)}$. This is the standard presentation and reviewers expect it. |
| Simulator cost, policy A vs B | **Paired bootstrap** on per-series cost differences (10,000 resamples), 95% CI | Paired because of common random numbers |
| Multiple pairwise tests | **Holm–Bonferroni** correction | With 14 models you have 91 pairs; uncorrected p-values are meaningless |
| User-study conditions | Friedman + Wilcoxon signed-rank, Holm-corrected | Non-parametric; n is small |
| Effect size | Report alongside every p-value | Cliff's delta or rank-biserial for non-parametric tests |

**Report confidence intervals everywhere, not just p-values.** With ~1,782 series, statistical significance is nearly free and nearly meaningless; a 0.3% cost reduction will be significant and irrelevant. Lead with effect sizes and CIs, and say which differences are *practically* meaningful.

---

## 8. Ablation table (for the paper)

| Variant | Forecast | Conformal | LT uncertainty | Quantile SS | MILP | Risk penalty | Explanations |
|---|---|---|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| − conformal | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| − LT uncertainty | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| − quantile SS (King formula instead) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| − MILP (greedy single supplier) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| − risk penalty ($\rho = 0$) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| − ML forecast (moving average) | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |

Each row reports Δ total cost and Δ fill rate against Full, with bootstrap CIs. This single table pre-empts most "how do you know component X matters?" reviewer questions.

Note that explanations are present in every row: they do not affect the numbers. Their evaluation is entirely Experiment E. Being clear about this prevents the reasonable objection that you are claiming cost savings *from* explainability, which you are not and must not.

---

## 9. Reproducibility checklist

- [ ] All seeds fixed and recorded (numpy, LightGBM, torch, supply generator, simulator)
- [ ] Environment pinned (`uv.lock` / `requirements.txt` with hashes) + Python version
- [ ] Hardware and runtime recorded per experiment
- [ ] Fold boundaries stored as data, not code constants
- [ ] Raw data checksummed; download scripts, not data, in the repo
- [ ] Supply-simulation config in YAML with the seed committed
- [ ] Every table and figure regenerable by one command from stored results parquet
- [ ] Solver version, time limits and MIP gaps reported
- [ ] Public repository link in the paper
