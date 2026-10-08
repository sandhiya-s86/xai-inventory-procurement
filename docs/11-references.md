# 11 — References and Sources

> **Verification policy.** Entries marked ✅ were confirmed at the primary source during this research pass (August 2026). Entries marked ⚠️ are known works whose exact bibliographic details (volume, pages, year) you must confirm against the publisher before they enter `refs.bib`. **Never cite a work you have not read at least the abstract of** — fabricated or mis-attributed citations are the fastest route to a desk rejection, and they are checked.

---

## 1. Datasets (primary sources)

| Source | URL | Status |
|---|---|---|
| UCI Online Retail II — Chen, D. (2012). UCI Machine Learning Repository. DOI 10.24432/C5CG6D. 1,067,371 instances, 8 features, 2009-12-01 → 2011-12-09, CC BY 4.0 | https://archive.ics.uci.edu/dataset/502/online+retail+ii | ✅ verified at source |
| UCI Online Retail (the smaller 541,909-row predecessor) | https://archive.ics.uci.edu/dataset/352/online+retail | ✅ |
| Kaggle mirror of Online Retail II (renamed columns) | https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci | ✅ |
| Kaggle — *Store Sales: Time Series Forecasting* (Favorita, family-level; 3,000,888 train rows, 54 stores, 33 families, 2013-01-01 → 2017-08-15) | https://www.kaggle.com/competitions/store-sales-time-series-forecasting | ✅ figures corroborated; verify test-horizon length yourself |
| Kaggle — *Corporación Favorita Grocery Sales Forecasting* (original, SKU-level; ~125.5M train rows, 54 stores in 22 cities / 16 states, 33 families, 337 classes) | https://www.kaggle.com/c/favorita-grocery-sales-forecasting | ⚠️ row/item counts from secondary sources — confirm |

---

## 2. Forecasting — methods and competitions

| Work | Note | Status |
|---|---|---|
| Makridakis, Spiliotis & Assimakopoulos — *The M5 accuracy competition: results, findings and conclusions*, **International Journal of Forecasting** | Source of RMSSE / WRMSSE; weights from cumulative dollar sales over the last 28 observations | ✅ paper exists; confirm vol/pages |
| Makridakis et al. — *The M5 uncertainty competition: results, findings and conclusions*, **IJF** | Scaled Pinball Loss; the nine quantiles {0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995} | ✅ |
| Lim, Arık, Loeff & Pfister — *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*, **IJF** 37(4), 2021 | TFT; variable selection networks; interpretable attention | ⚠️ confirm pages |
| Croston, J.D. (1972) — *Forecasting and stock control for intermittent demands*, **Operational Research Quarterly** 23(3) | The original intermittent-demand method | ⚠️ |
| Syntetos, Boylan & Croston (2005) — *On the categorization of demand patterns*, **JORS** 56(5) | **ADI = 1.32, CV² = 0.49 cutoffs**; smooth/erratic/intermittent/lumpy | ✅ cutoffs and scheme confirmed; verify pages |
| Teunter, Syntetos & Babai — TSB method | Replaces the inter-demand interval with demand probability; handles obsolescence | ⚠️ |
| Petropoulos & Kourentzes — ADIDA / MAPA / temporal aggregation | Aggregation-based intermittent forecasting | ⚠️ |
| Oreshkin et al. — *N-BEATS* (ICLR 2020); Challu et al. — *N-HiTS* (AAAI 2023); Salinas et al. — *DeepAR* (IJF 2020) | Deep baselines | ⚠️ |
| *Evaluating the Effectiveness of Time Series Transformers for Demand Forecasting in Retail*, **Mathematics** 12(17), 2728 (2024) | Transformer family vs baselines on M5 (30,490 series) | ✅ located |
| *Comparative Analysis of Modern Machine Learning Models for Retail Sales Forecasting*, arXiv:2506.05941 | Statistical vs GBM vs N-BEATS/N-HiTS/TFT under intermittency, missingness, product churn | ✅ located |
| Hyndman & Koehler (2006) — *Another look at measures of forecast accuracy*, **IJF** | MASE; the case against MAPE | ⚠️ |
| Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (3rd ed.), OTexts | Free, citable, excellent on rolling-origin evaluation | ✅ https://otexts.com/fpp3/ |

---

## 3. Uncertainty quantification

| Work | Note | Status |
|---|---|---|
| Romano, Patterson & Candès (2019) — *Conformalized Quantile Regression*, **NeurIPS** | CQR — the method you implement | ⚠️ |
| Lei et al. (2018) — *Distribution-free predictive inference for regression*, **JASA** | Split conformal foundations | ⚠️ |
| Barber, Candès, Ramdas & Tibshirani — conformal prediction beyond exchangeability | Required for time-series validity | ⚠️ |
| *A Gentle Introduction to Conformal Time Series Forecasting*, arXiv:2511.13608 (2025) | Recent survey; covers calibration reweighting, adaptive residual updating, online coverage tuning | ✅ located |
| Nixtla — conformal prediction tutorial | Practical implementation | ✅ https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html |

---

## 4. Inventory theory and optimization

| Work | Note | Status |
|---|---|---|
| King, P.L. (2011) — *Understanding safety stock and mastering its equations*, **APICS Magazine** | The combined demand + lead-time variability formula | ✅ https://web.mit.edu/2.810/www/files/readings/King_SafetyStock.pdf |
| APICS — *Cycle Service Level and Fill Rate* | The CSL vs fill-rate distinction | ✅ |
| Silver, Pyke & Thomas — *Inventory and Production Management in Supply Chains* (4th ed.) | Standard textbook; (R,s,Q) and (R,s,S) policies | ⚠️ |
| Zipkin — *Foundations of Inventory Management* | Rigorous reference | ⚠️ |
| Elmachtoub & Grigas (2022) — *Smart "Predict, then Optimize"*, **Management Science** 68(1) | SPO loss; MSE ≠ decision regret; SPO+ convex surrogate | ✅ https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922 |
| Oroojlooyjadid, Snyder & Takáč — *Applying deep learning to the newsvendor problem* | DNN for the critical quantile | ⚠️ |
| Ban & Rudin — *The Big Data Newsvendor: Practical Insights from Machine Learning*, **Operations Research** | Feature-based newsvendor | ⚠️ |
| *From predictive to prescriptive analytics: a data-driven multi-item newsvendor model*, **Decision Support Systems** (2020) | ML quantile regression giving order quantities in one step | ✅ located |
| *Quantile forecasting and data-driven inventory management under nonstationary demand*, **ORL** | Direct link forecast → inventory | ✅ located |
| MILP supplier selection / lot sizing with MOQ, minimum order value and quantity discounts | Multiple works; see the Springer/ScienceDirect hits logged in `lit-log.md` | ⚠️ pick 2–3 |
| *A practical approach to replenishment optimization with extended (R, s, Q) policy and probabilistic models*, **Scientific Reports** (2025), Zalando | Strong industrial related work | ✅ located |
| Google OR-Tools documentation | CP-SAT and MPSolver | ✅ https://developers.google.com/optimization/introduction |

---

## 5. Explainable AI

| Work | Note | Status |
|---|---|---|
| Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*, **NeurIPS** | SHAP | ⚠️ |
| Lundberg et al. (2020) — *From local explanations to global understanding with explainable AI for trees*, **Nature Machine Intelligence** 2 | **TreeSHAP** — exact, polynomial-time Shapley for tree ensembles | ⚠️ |
| Ribeiro, Singh & Guestrin (2016) — *"Why Should I Trust You?"*, **KDD** | LIME | ⚠️ |
| Yeh et al. (2019) — *On the (in)fidelity and sensitivity of explanations*, **NeurIPS** | Infidelity and sensitivity metrics | ⚠️ |
| *Evaluation Metrics for XAI: A Review, Taxonomy, and Practical Applications*, IEEE INES 2023 (DFKI) | Taxonomy for justifying your metric choice | ✅ located |
| Wachter, Mittelstadt & Russell (2017) — counterfactual explanations without opening the black box | Counterfactual explanation foundations | ⚠️ |
| Hoffman et al. — *Metrics for Explainable AI: Challenges and Prospects* | Explanation-satisfaction and trust scales for the user study | ⚠️ |

---

## 6. XAI and AI in supply chain / ERP

| Work | Note | Status |
|---|---|---|
| *Enabling explainable artificial intelligence capabilities in supply chain decision support making*, **Production Planning & Control** (2024) | SHAP + LIME identifying cost and inventory drivers in SC DSS | ✅ https://www.tandfonline.com/doi/full/10.1080/09537287.2024.2313514 |
| *A review of explainable artificial intelligence in supply chain management using neurosymbolic approaches*, **International Journal of Production Research** (2023) | Most SC AI affords little/no explainability; explainability as adoption barrier | ✅ https://www.tandfonline.com/doi/full/10.1080/00207543.2023.2281663 |
| *Artificial intelligence in supply chain management: a systematic literature review of empirical studies and research directions*, **JBR/IJPE** (2024) | Adoption barriers, trust, SME gap | ✅ located via ScienceDirect |
| *Intelligent Supply Chain Management: A Systematic Literature Review on Artificial Intelligence Contributions*, **Information** 16(5), 399 (2025) | Recent SLR | ✅ https://doi.org/10.3390/info16050399 |
| *Perceptions and Trust in AI-Driven Decision Support Systems in Supply Chain Management* (2025) | Practitioner trust evidence for the introduction | ✅ located |
| *A generic framework for decision support in retail inventory management*, arXiv:2207.13923 | Related framework paper | ✅ located |
| *A hybrid learning framework for forecasting uncertainty and adaptive inventory planning in retail supply chains*, ScienceDirect (2025) | Close related work — read carefully | ✅ located |

---

## 7. Statistical testing

| Work | Note | Status |
|---|---|---|
| Diebold & Mariano (1995) — *Comparing predictive accuracy*, **JBES** 13(3) | The DM test | ⚠️ |
| Harvey, Leybourne & Newbold (1997) — small-sample correction to DM | Use this, not raw DM | ⚠️ |
| Demšar (2006) — *Statistical comparisons of classifiers over multiple data sets*, **JMLR** 7 | Friedman + Nemenyi; the **critical-difference diagram**; $CD = q_\alpha\sqrt{k(k+1)/(6N)}$ | ⚠️ — the canonical citation for your Fig. 4 |
| Koning et al. / Kourentzes — Nemenyi applied to forecasting comparisons | `tsutils::nemenyi` in R; `scikit-posthocs` in Python | ✅ located |
| *Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices*, arXiv:2203.10716 | Excellent, citable, and will save you from several mistakes — **read this early** | ✅ located |

---

## 8. Software

| Tool | URL |
|---|---|
| Nixtla `statsforecast` (AutoARIMA, ETS, Croston, SBA, TSB, ADIDA, IMAPA) | https://github.com/Nixtla/statsforecast |
| Nixtla `mlforecast` / `neuralforecast` | https://nixtlaverse.nixtla.io/ |
| LightGBM / XGBoost | https://lightgbm.readthedocs.io/ |
| `pytorch-forecasting` (TFT, DeepAR) | https://pytorch-forecasting.readthedocs.io/ |
| SHAP | https://github.com/shap/shap |
| MAPIE (conformal) | https://github.com/scikit-learn-contrib/MAPIE |
| Google OR-Tools | https://developers.google.com/optimization |
| `scikit-posthocs` (Nemenyi, CD diagrams) | https://scikit-posthocs.readthedocs.io/ |
| ERPNext | https://github.com/frappe/erpnext |

Cite the software you actually use, with version numbers, in the experimental-setup section.

---

## 9. `refs.bib` hygiene

1. Pull BibTeX from the **publisher's** page or DBLP, never from a blog or a search-result snippet.
2. Prefer the journal/conference version over the arXiv preprint when one exists.
3. Check every author name, year, venue and page range against the publisher record.
4. Read at least the abstract and the results section of everything you cite as related work. If a reviewer asks "how does your method differ from [12]?" you must be able to answer.
5. Keep `docs/lit-log.md` current — it converts directly into Section 2 of the paper.
