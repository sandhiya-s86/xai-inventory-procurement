# 10 — Paper Outline and Publication Plan

## 1. Target venues

Decide the venue **before** writing — page limits and format differ substantially and retrofitting is painful.

| Tier | Venue type | Examples | Notes |
|---|---|---|---|
| Realistic primary | IEEE international conferences with Xplore indexing in AI/data-science/computing | ICCCNT, ICACCS, ICICCS, ICIRCA, ICSCDS, INDICON, CONECCT | 6 pages, IEEEtran double-column. Accessible for a strong B.Tech project. Check the venue is genuinely IEEE-sponsored, not a lookalike |
| Realistic primary | Springer proceedings series | LNNS, LNEE, SIST, AISC (via conferences such as ICDECT, ICTIS, SoCTA, ICCIDS) | ~10–12 pages, Springer LNCS format |
| Stretch | Domain conferences / workshops | INFORMS-adjacent workshops, ECML PKDD applied-data-science track, IJCAI/AAAI XAI workshops | Higher bar, much stronger line on a CV |
| Stretch | Journals (if results are strong and you have time) | *Expert Systems with Applications*, *Computers & Industrial Engineering*, *IEEE Access*, *International Journal of Production Research* | Months-long review; only if you have a full year |

**Recommendation:** target a **Springer LNNS-series conference** (10–12 pages gives room for the methodology and the ablations, which a 6-page IEEE format will force you to gut), with an IEEE conference as the fallback.

**Warning on predatory venues.** Verify: is it indexed in Scopus/DBLP? Is it genuinely IEEE/Springer-sponsored (check the sponsor's own conference listing, not the conference's own website)? Are the programme committee members real, findable researchers? Is the review turnaround plausible — a promise of acceptance in 5 days is a definitive red flag? Is the fee proportionate? A publication in a predatory venue is worse than no publication.

---

## 2. Full paper structure (Springer, ~11 pages)

### Title
*From Probabilistic Demand Forecasts to Explainable Procurement Decisions: A Decision-Cost-Evaluated XAI Framework for ERP Inventory Replenishment*

Do not title it "AI Inventory Management ERP." That title reads as a software project, and reviewers form an expectation from the title before the abstract.

### Abstract (~200 words)
Draft in [00](00-executive-summary.md). Structure: context (1) → gap (2) → what we do (3–4) → how we evaluate (2) → key results with numbers (2). Fill the numbers in last, and only with numbers you actually obtained.

### 1. Introduction (~1 page)
1. ERP replenishment is rule-based and static; the cost of that. [→ [01](01-problem-and-motivation.md)]
2. ML forecasting has advanced substantially but stops at the forecast.
3. Explainability is the recognised adoption barrier — **and existing XAI in this domain explains the prediction, not the decision.** This sentence is the pivot of the whole paper; make it unmissable.
4. Contributions, as a numbered list of four or five.
5. Paper roadmap.

### 2. Related Work (~1.5 pages)
Four subsections: demand forecasting (incl. intermittent); inventory optimization and predict-then-optimize; XAI in supply chain; ERP decision support.
**End with the comparison table** from [02](02-literature-review-and-research-gap.md) §6 and an explicit gap paragraph. Every citation gets one sentence of what it does and one of what it does not do that you do.

### 3. Problem Formulation (~0.75 page)
Notation table. Formal statement: given history and covariates, choose $(Q, s, t_{\text{order}})$ minimising expected total cost subject to constraints, while producing an explanation $E$ with stated faithfulness properties. Making the *explanation* part of the formal problem statement is a small move that signals the contribution before the reader reaches the method.

### 4. Proposed Framework (~3 pages) — the core
- 4.1 Architecture overview + pipeline figure
- 4.2 Feature engineering and demand classification (SBC)
- 4.3 Probabilistic forecasting + conformal calibration
- 4.4 Lead-time demand distribution and risk quantification
- 4.5 Procurement optimization (full MILP, stated as such)
- 4.6 **Two-level explanation** — the longest subsection
  - 4.6.1 Forecast attribution (SHAP) — brief, positioned as the baseline
  - 4.6.2 Exact decision decomposition (with the additivity property stated as a proposition)
  - 4.6.3 Shapley split of the safety-stock buffer (closed form for two players, with the efficiency property)
  - 4.6.4 Counterfactual thresholds and supplier regret
  - 4.6.5 Explanation rendering + **the worked example box** (from [04](04-methodology.md) §9.3)

The worked example is the single most persuasive element in the paper. Give it a full-width figure or boxed listing and refer to it from the introduction.

### 5. Experimental Setup (~1.25 pages)
Datasets (with the honest table of what is real and what is simulated); simulated supply parameters + four scenarios; baselines P0–P4; rolling-origin protocol; metrics; implementation and hardware; reproducibility statement with the repository link.

### 6. Results and Discussion (~3 pages)
| Order | Content |
|---|---|
| 6.1 | Demand characterisation (SBC distribution) — Fig. 3 |
| 6.2 | Forecast accuracy — Table 3, by-class Table 4, CD diagram Fig. 4 |
| 6.3 | Calibration — coverage before/after, Fig. 5 |
| 6.4 | **Inventory and cost results — Table 5 + the cost–service frontier, Fig. 1** |
| 6.5 | Scenario sensitivity — Table 7, Fig. 6 |
| 6.6 | Ablation — Table 11 |
| 6.7 | **Accuracy-vs-decision ranking inversion — Fig. 8** |
| 6.8 | Explanation evaluation — faithfulness, stability, user study |
| 6.9 | Discussion: what generalises, what is scenario-specific, what surprised you |

Lead Section 6 with the **cost–service frontier plot**, not the forecast-accuracy table. That ordering enacts the paper's central argument.

### 7. Threats to Validity / Limitations (~0.4 page)
Do not bury this. Simulated supply parameters; single-echelon; family-level aggregation (if applicable); small user study; two datasets from retail only; deterministic-equivalent MILP approximation; no real deployment. A frank limitations section increases reviewer trust — reviewers find these limitations anyway, and finding them *unacknowledged* is much worse.

### 8. Conclusion and Future Work (~0.4 page)
Restate contributions with actual numbers. Future work: multi-echelon; decision-focused (SPO+) training end-to-end; RL replenishment; learning from the accept/modify/reject override log; live deployment.

### References
30–45 entries. Predominantly 2020–2026, plus the classical anchors (Croston 1972; Syntetos–Boylan–Croston 2005; Lundberg & Lee 2017; Lim et al. 2021 TFT; Elmachtoub & Grigas 2022; Makridakis et al. M5).

---

## 3. Figure and table plan

| # | Asset | Source |
|---|---|---|
| **Fig. 1** | **Cost–service efficiency frontier, all policies** | Exp. D — the money figure |
| Fig. 2 | System architecture | [05](05-system-architecture.md) |
| Fig. 3 | ADI–CV² scatter with SBC cutoffs, both datasets | EDA |
| Fig. 4 | Critical-difference diagram (Nemenyi) | Exp. A |
| Fig. 5 | Coverage before/after conformal calibration | Exp. B |
| Fig. 6 | Cost by policy × scenario (grouped bars, with CIs) | Exp. C |
| **Fig. 7** | **Worked explanation example (waterfall + counterfactuals)** | [04](04-methodology.md) §9.3 |
| Fig. 8 | Δ accuracy vs Δ cost scatter (ranking inversion) | Exp. F |
| Fig. 9 | Dashboard screenshot | Week 15 |
| Table 1 | Notation | §3 |
| Table 2 | Dataset characteristics — real vs simulated columns marked | [03](03-datasets.md) |
| Table 3 | Forecast accuracy, all models | Exp. A |
| Table 4 | Accuracy by SBC class | Exp. A |
| **Table 5** | **Inventory cost and service, P0–P4 × scenarios** | Exp. D |
| Table 6 | Coverage and interval width | Exp. B |
| Table 7 | Scenario sensitivity | Exp. C |
| Table 8 | Explanation faithfulness | Exp. E |
| Table 9 | User study results | Exp. E |
| Table 10 | Ranking correlation, accuracy vs cost | Exp. F |
| Table 11 | Ablation | §8 of [06](06-experimental-design.md) |
| Table 12 | Related-work comparison | [02](02-literature-review-and-research-gap.md) §6 |

For a 6-page IEEE version, cut to Figs. 1, 2, 7, 8 and Tables 3, 5, 8, 12. Note that the surviving set still tells the complete argument — that is the test of whether your figure plan is sound.

---

## 4. Writing rules

1. **One claim per sentence, one argument per paragraph.**
2. **Never write "our model achieved 95% accuracy"** without saying accuracy of what, measured how, against what baseline.
3. Every number in the abstract must appear in a table.
4. Use "simulated" and "assumed" wherever they apply. Every time.
5. Past tense for what you did, present tense for what the method does.
6. No unexplained acronyms after the abstract.
7. Do not claim generalisation beyond two retail datasets.
8. Report at least one thing that did not work. It is the cheapest credibility available.
9. Read the paper aloud once before submitting. Sentences you stumble on are sentences the reviewer will stumble on.

---

## 5. Submission checklist

- [ ] Venue verified as genuinely IEEE/Springer-sponsored and Scopus/DBLP-indexed
- [ ] Format template applied from the start (IEEEtran / Springer LNCS)
- [ ] Page limit respected without shrinking margins or fonts
- [ ] Anonymised if the venue is double-blind (including the repo link — use an anonymised mirror)
- [ ] All figures vector (PDF/EPS), legible at 100% and in greyscale
- [ ] Every table has a caption stating what is measured and in what units
- [ ] Reproducibility statement with repository link
- [ ] Dataset attributions: Online Retail II CC BY 4.0 with DOI; Favorita competition + Corporación Favorita as provider
- [ ] Ethics statement for the user study
- [ ] Explicit statement that supply-side parameters are simulated — in abstract, method, and limitations
- [ ] Grammar/spell check; consistent British or American spelling
- [ ] Supervisor and co-authors have read the final version
- [ ] arXiv preprint considered (check the venue permits it)

---

## 6. Realistic expectations

A well-executed version of this project is a **solid conference paper**, not a top-tier journal article. That is the correct target for a B.Tech final year, and there is no shame in it — a clean, honest, reproducible applied paper at a decent indexed venue is worth considerably more than an inflated claim at a venue nobody reads.

The three things that will most improve your odds:
1. The **decision-attribution** contribution, presented as the headline rather than buried in Section 4.6.
2. **Decision-cost evaluation** rather than forecast-error evaluation.
3. **Transparent handling of the simulation**, turning the obvious weakness into a declared, swept, released sensitivity analysis.
