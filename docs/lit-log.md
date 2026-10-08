# Literature Log

One row per paper you actually read. Fill this in as you go — it converts directly into Section 2 of the paper and into the related-work comparison table.

**Rule:** the "What it does NOT do" column is the important one. If you cannot fill it in, you have not read the paper carefully enough to cite it as related work.

| # | Citation (author, year, venue) | Strand | What it does | What it does NOT do (that we do) | Dataset used | Cite in § |
|---|---|---|---|---|---|---|
| 1 | | forecasting / inventory / XAI / ERP | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

---

## Priority reading queue (week 1–3)

**Must read before finalising the topic** — these determine whether the novelty claim survives:

- [ ] *Enabling explainable AI capabilities in supply chain decision support making*, Production Planning & Control (2024) — the closest XAI-in-SC applied work
- [ ] *A review of XAI in supply chain management using neurosymbolic approaches*, IJPR (2023) — the survey that frames the gap
- [ ] Elmachtoub & Grigas (2022), *Smart "Predict, then Optimize"*, Management Science — the theoretical basis for decision-cost evaluation
- [ ] *A hybrid learning framework for forecasting uncertainty and adaptive inventory planning in retail supply chains* (2025) — possibly very close to your framework
- [ ] *A practical approach to replenishment optimization with extended (R,s,Q) policy and probabilistic models*, Sci. Reports (2025) — industrial related work
- [ ] Syntetos, Boylan & Croston (2005) — the SBC taxonomy you use throughout
- [ ] Makridakis et al., M5 accuracy + M5 uncertainty papers — your metrics come from here
- [ ] *Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices* (arXiv:2203.10716) — read early; it will prevent mistakes
- [ ] Demšar (2006) — for the CD diagram
- [ ] King (2011) safety stock — the formula you use and ablate against

**Searches to run and log** (record hit counts and anything relevant, even if you find nothing — "we searched X and found no prior work doing Y" is a legitimate and useful statement to be able to make):

```
"explainable" AND ("order quantity" OR "replenishment") AND (SHAP OR counterfactual)
"decision-focused" AND "inventory" AND "explanation"
"prescriptive analytics" AND "interpretab*" AND "supply chain"
"counterfactual explanation" AND "optimization" AND ("procurement" OR "sourcing")
"explainable optimization" OR "explainable constraint programming" OR "XAI for operations research"
Favorita AND (SHAP OR explainab*)
"inventory simulation" AND "forecast" AND "cost" AND benchmark
"order-up-to" AND "quantile forecast" AND "conformal"
```

Databases: Google Scholar, IEEE Xplore, ScienceDirect, SpringerLink, arXiv, DBLP.
Also check: CPAIOR, IJCAI/AAAI XAI workshops, ECML PKDD applied data science track.

---

## Novelty verdict log

Record your conclusion after each search round, with the date. If a search overturns the novelty claim, record that too — an honest pivot in week 3 is cheap; discovering the problem in week 15 is not.

| Date | Search round | Closest prior work found | Does it threaten the claim? | Action |
|---|---|---|---|---|
| | | | | |
