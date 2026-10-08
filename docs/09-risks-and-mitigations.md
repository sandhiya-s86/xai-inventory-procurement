# 09 — Risks and Mitigations

Ordered by expected damage, not by probability.

---

## R1 — Insufficient novelty (severity: fatal | likelihood: medium)

**The risk.** "Forecast retail demand with ML and explain it with SHAP" has been published many times over. If the paper's contribution reads as a combination of existing components, a reviewer writes *"the individual components are standard and the integration is engineering rather than research"* and it is rejected.

**Mitigations**
1. Do the prior-art search in [02](02-literature-review-and-research-gap.md) §5 **in week 1–2**, not after the experiments. Log every hit.
2. Lead with the **decision-level attribution** (exact decomposition + Shapley split of safety stock + counterfactual thresholds + supplier regret). Make the SHAP-on-forecast layer explicitly a *baseline you compare against*, not a contribution.
3. Put the related-work comparison table (§6 of [02](02-literature-review-and-research-gap.md)) on page 2, so the delta is visible before the reviewer forms an opinion.
4. The decision-cost evaluation protocol is a second, independent contribution. If the novelty of (2) is challenged, (3) still carries weight.

**Early warning:** if week-2 searching finds a paper doing exact attribution over a constrained replenishment optimizer, pivot immediately — most likely toward the multi-objective supplier-risk frontier or the accuracy-vs-decision inversion analysis as the headline.

---

## R2 — The simulated supply layer discredits the results (severity: high | likelihood: medium-high)

**The risk.** A reviewer writes: *"the procurement results depend entirely on invented parameters; nothing is validated against reality."* This is a fair criticism and cannot be fully answered.

**Mitigations**
1. **Never hide it.** Abstract, methodology, every relevant figure caption, and the limitations section all state that supply-side parameters are simulated.
2. **Sweep, don't pick.** Four scenarios ([06](06-experimental-design.md) §3) plus sensitivity runs on cost ratio and MOQ severity. Conclusions that hold across all four are robust to the parameterisation; conclusions that hold in only one must be reported as scenario-specific.
3. **Ground the parameters in literature** — carrying cost as a % of item value, standard service-level targets, Gamma-distributed lead times — with citations, not round numbers.
4. **Release the generator** with its seed. The mitigation for unverifiable data is verifiable *procedure*.
5. Use the real `perishable` flag (SKU-level Favorita) to derive at least one genuinely data-grounded cost asymmetry.
6. **Frame the released simulator as a contribution.** "There is no public dataset joining real sales to real procurement terms, so we release a declared, parameterised supply layer to make this class of study reproducible" turns the weakness into a stated contribution.

---

## R3 — The proposed system does not beat the baseline (severity: high | likelihood: medium)

**The risk.** P4 fails to beat P0/P1 on cost at matched service level in some or all scenarios.

**Mitigations**
1. **Design for informativeness, not for winning.** Experiment F (ranking inversion) and the ablation table are publishable regardless of direction.
2. If it loses, diagnose rather than tune-until-it-wins: is the forecast weak? is the MILP over-constrained? is the simulator's cost structure making the difference irrelevant? The diagnosis *is* the finding.
3. Check for the standard causes first: unmatched service levels between policies (the most common false comparison), leakage inflating the baseline, or a simulator bug.
4. A paper titled around *"when does ML-based replenishment actually pay?"* with an honest mixed result is more valuable and more credible than a paper claiming a uniform 20% win. Keep that pivot available.

**What you must not do:** tune the proposal on test data, or select the scenario where you win and report only that. Both are detectable and both end careers.

---

## R4 — Data leakage invalidates everything (severity: fatal | likelihood: medium)

**The risk.** A rolling feature computed without shifting, a target encoding fitted across folds, a calibration set overlapping training. Results look excellent and are worthless. This is the single most common fatal error in student time-series projects.

**Mitigations**
1. `test_no_leakage` as an automated test, run in CI, asserting every feature's maximum source date ≤ forecast origin.
2. Feature metadata declaring availability lag; the builder refuses to emit an undeclared feature.
3. Sanity check: if any model achieves near-perfect accuracy on a noisy retail series, assume leakage until proven otherwise. Retail demand is not that predictable.
4. Have someone else review the feature builder specifically. It is 200 lines and it decides whether the project is real.

---

## R5 — MILP intractability (severity: medium | likelihood: medium)

**The risk.** The full multi-item model over a 28-day horizon with integer case packs does not solve in reasonable time.

**Mitigations**
1. **Decompose by item** where the only coupling is budget/capacity; use a Lagrangian or a simple iterative budget-allocation heuristic across items if needed.
2. Set a solver time limit and **report the MIP gap**. A 1%-gap solution is fine; concealing that a gap exists is not.
3. Shorten the planning horizon to 14 days; rolling re-solve limits the damage.
4. Relax case-pack integrality to continuous with post-hoc rounding if necessary — but then report the rounding loss.
5. Fall back to a greedy/closed-form order-up-to with constraint post-processing. The decision attribution still works (the decomposition does not require a MILP), so the primary contribution survives even if the optimizer is downgraded. **Note this explicitly — it means R5 cannot sink the project.**

---

## R6 — Compute for deep models (severity: low-medium | likelihood: medium-high)

**The risk.** TFT training on 1,782 series exceeds available GPU time; the SKU-level 125M-row release exceeds available RAM.

**Mitigations**
1. Kaggle Notebooks (free GPU quota) or Colab; record which and report runtimes.
2. Stratified item subsample for the SKU-level release (300–500 items), with a documented sampling procedure.
3. `polars`/`duckdb` and aggressive dtype downcasting.
4. **Cut TFT.** It is a nice-to-have. LightGBM-quantile + statistical baselines is a complete and defensible model set, and the contribution does not depend on a transformer. Saying "we did not evaluate deep models due to compute constraints" is entirely acceptable; faking that you did is not.

---

## R7 — The user study is too weak to support RQ5 (severity: medium | likelihood: high)

**The risk.** n = 9 classmates, no domain expertise, obvious demand characteristics, results not generalisable.

**Mitigations**
1. **Do not make the user study the sole evidence for RQ5.** Faithfulness and stability metrics are objective, automated, and carry the claim on their own.
2. Include the **appropriate-reliance** manipulation (deliberately degraded recommendations). Whether users catch bad advice is a much stronger measure than whether they say they understood.
3. Use a within-subjects design with counterbalancing — it extracts far more power from small n.
4. Recruit at least 1–2 people with real procurement or supply-chain experience, and report participant backgrounds honestly.
5. Write the limitations paragraph before running the study, so you are not tempted to overstate afterwards.

---

## R8 — Scope overrun (severity: medium | likelihood: high)

**The risk.** Two datasets × 14 models × 4 policies × 4 scenarios × 5 folds × 30 replications, plus an ERP, plus a dashboard, plus a user study, in 16 weeks, alongside other coursework.

**Mitigations**
1. **Pre-committed cut order** (from [00](00-executive-summary.md)):
   TFT → SKU-level Favorita release → user study → ERPNext integration (→ Streamlit only) → Online Retail II external validation.
2. **Never cut:** the digital twin, the MILP (or its fallback), the decision attribution, the statistical validation. These are the paper.
3. Milestone gates in [08](08-implementation-plan.md) §6 force an honest reassessment at weeks 2, 5, 8, 9, 11 and 13.
4. Timebox the dashboard to five days, hard.

---

## R9 — Kaggle data access friction (severity: low | likelihood: low-medium)

Competition rules must be accepted in the browser before the API will serve the files, and the SKU-level archive is 7z-compressed. Do this in week 1, not week 3. Have Online Retail II (a direct download, no account needed) working as an unblocked fallback.

---

## R10 — Reproducibility failure at write-up time (severity: medium | likelihood: medium)

**The risk.** You cannot regenerate the number in Table 5 because the script changed.

**Mitigations**
1. Every result written to parquet with the config hash and git commit that produced it.
2. Tables and figures generated from stored results, never transcribed by hand into LaTeX.
3. MLflow for every run.
4. Tag the submission commit.

---

## R11 — Ethical / licensing missteps (severity: medium | likelihood: low)

1. Do not redistribute Kaggle competition data. Ship download scripts and checksums.
2. Attribute Online Retail II per CC BY 4.0, with its DOI.
3. Drop `CustomerID` at ingest — it is unnecessary here, and doing so gives a clean answer to any data-protection question.
4. Obtain consent for the user study; anonymise responses; state institutional approval status.
5. Do not describe simulated results as observed. This is the ethical failure most likely to occur by accident, through sloppy phrasing rather than intent — which is exactly why it needs a rule rather than good intentions.

---

## Risk summary

| ID | Risk | Sev | Lik | Primary mitigation |
|---|---|---|---|---|
| R1 | Insufficient novelty | Fatal | Med | Week-1 prior-art search; lead with decision attribution |
| R4 | Data leakage | Fatal | Med | Automated leakage test; external review of the feature builder |
| R2 | Simulation credibility | High | Med-High | Declare, sweep, ground, release |
| R3 | No improvement over baseline | High | Med | Design experiments that inform either way; matched service levels |
| R5 | MILP intractability | Med | Med | Decompose; report gap; heuristic fallback preserves the contribution |
| R7 | Weak user study | Med | High | Objective XAI metrics carry RQ5; appropriate-reliance measure |
| R8 | Scope overrun | Med | High | Pre-committed cut order; milestone gates |
| R6 | Compute limits | Low-Med | Med-High | Free GPU quota; subsample; cut TFT |
| R10 | Reproducibility | Med | Med | Config-hashed results; auto-generated tables |
| R11 | Licensing/ethics | Med | Low | Checksums not data; CC BY attribution; drop CustomerID |
| R9 | Data access friction | Low | Low-Med | Do it in week 1 |
