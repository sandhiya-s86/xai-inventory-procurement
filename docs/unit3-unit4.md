# Units 3 & 4 — Unsupervised Learning and Sequential Models

Everything in the Unit 3 and Unit 4 syllabi, implemented from first
principles, run on the real 1,067,371-transaction dataset, and verified.

- **Results report (PDF):** [`paper/Unit3_Unit4_Results_Report.pdf`](../paper/Unit3_Unit4_Results_Report.pdf) — 18 pages, every number read live from the findings JSON
- **Study guide (PDF):** [`paper/Unit3_Unit4_Study_Guide.pdf`](../paper/Unit3_Unit4_Study_Guide.pdf) — 17 pages, all 17 topics split across the team of 3
- **Dashboard:** page 7 (SKU Segmentation) and page 8 (Demand Regimes)

```bash
.venv\Scripts\python scripts\run_unsupervised.py   # Unit 3
.venv\Scripts\python scripts\run_sequential.py     # Unit 4
.venv\Scripts\python scripts\build_unit34_docs.py  # the two PDFs
.venv\Scripts\python scripts\verify_dashboard.py   # Playwright check
```

---

## Syllabus coverage

| Unit 3 topic | Module | Unit 4 topic | Module |
|---|---|---|---|
| K-means clustering | `unsupervised/kmeans.py` | Sequential data | `sequential/sequential.py` |
| Mixtures of Gaussians | `unsupervised/gmm.py` | Markov models | `sequential/markov.py` |
| EM algorithm | `unsupervised/gmm.py` | HMM | `sequential/hmm.py` |
| An alternative view of EM | `gmm.em_lower_bound` | Maximum likelihood / EM | `hmm.fit` (Baum–Welch) |
| Factor analysis | `unsupervised/factor_analysis.py` | Forward–backward | `sequential/hmm.py` |
| PCA | `unsupervised/pca.py` | Sum–product | `sequential/sum_product.py` |
| Choosing latent dimensions | `pca.choose_n_components` | Scaling factors | `hmm.forward_scaled` |
| Measuring dissimilarity | `unsupervised/dissimilarity.py` | Viterbi | `sequential/viterbi.py` |
| Evaluating clustering output | `unsupervised/evaluation.py` | Linear dynamical systems | `sequential/lds.py` |
| Hierarchical clustering | `unsupervised/hierarchical.py` | — | — |

Unit 5 is **not** covered: the Drive link supplied for it returns HTTP 401
without authentication, so its syllabus could not be read.

---

## Headline results

**Unit 4 found three demand regimes nobody labelled.** A 3-state Gaussian HMM
fitted by Baum–Welch on log1p daily demand:

| Regime | Mean demand | Self-transition | Expected duration | Share of days |
|---|---|---|---|---|
| QUIET | 3,605 units/day | 0.991 | 106 days | 42.1% |
| NORMAL | 4,664 units/day | 0.987 | 80 days | 47.6% |
| SURGE | 9,244 units/day | 0.942 | 17 days | 10.3% |

Viterbi decoding places SURGE on the October–December trading peak in **both**
2010 and 2011 — the model was never told Christmas exists.

**Why it matters for the decision:** demand volatility differs **2.1×** between
the calmest and busiest regime, so a single pooled safety stock is
simultaneously ~1.42× too small during SURGE and ~0.67× oversized during QUIET.
The multipliers are computed from the **filtered** posterior, so they are
available to a buyer on the morning of the decision.

**Unit 3 found five SKU segments** from 13 behavioural features, learned on the
training window only:

| Segment | SKUs | Mean demand | ADI | Q4 ratio | Revenue |
|---|---|---|---|---|---|
| High-volume sells-daily | 129 | 28.8/day | 1.31 | 1.59 | 64.2% |
| Mid-volume weekly-cyclical | 50 | 12.8/day | 3.37 | 14.10 | 18.4% |
| High-volume fast-moving | 8 | 133.6/day | 1.30 | 4.90 | 15.4% |
| Mid-volume erratic | 10 | 7.3/day | 41.94 | 0.96 | 2.0% |
| Dormant (no sales in window) | 3 | 0.0/day | — | — | 0.0% |

---

## Results that are worth the marks

Three findings reported because they are true, not because they flatter:

**1. A faithful dendrogram is not a useful partition.** Average linkage scored
the *best* cophenetic correlation (0.931) and put **97% of SKUs in one
cluster** — textbook chaining. Ward scored the *worst* (0.495) and was the only
method giving balanced groups, so the cut was taken from Ward.

**2. The Markov stationary distribution proves nothing here.** The four demand
states are equal-sized quantile buckets, so a uniform stationary distribution
is guaranteed *by construction*. The real test is persistence against
independence: lift up to **1.96×** for PEAK, chi-square 63.0 on 9 df,
p = 3.5e-10. Busy days genuinely cluster.

**3. This series does not actually underflow.** Claiming an underflow without
checking would have been dishonest. At the real T = 604 the raw forward product
is ~1e-217 — small, but above float64's 1e-308 floor — and the scaled and
unscaled answers agree to the last decimal. Tiling the series to find the true
breaking point: the naive recursion survives T = 900 and is dead by **T =
1,000**, while the scaled one is still finite at T = 3,600.

---

## How it is verified

| Layer | What it catches |
|---|---|
| **Reference tests** | From-scratch code vs scikit-learn / SciPy. K-means inertia, PCA components, all 4 linkage methods, silhouette, Davies–Bouldin, Calinski–Harabasz and ARI all match. |
| **Brute-force tests** | Dynamic programming vs enumerating all `N^T` paths. Forward recursion, forward–backward posteriors and the Viterbi path match exactly on a 3-state, 7-step HMM (2,187 paths). |
| **Property tests** | The guarantees the lectures state: EM monotonicity, EM bound tightness, rotational invariance of the FA likelihood, `pi = pi A`, smoothed variance never exceeding filtered. |
| **Browser verification** | `scripts/verify_dashboard.py` drives the real Streamlit server in Chromium. Streamlit traps exceptions *per widget*, so a broken chart yields an app that starts fine and is wrong — only a browser catches it. 8/8 pages pass. |

```
239 tests passing  =  81 (Units 1–2)  +  79 (Unit 3)  +  79 (Unit 4)
 87 rubric checks  =  51 (Units 1–2)  +  36 (Units 3–4)
```

`scripts/check_pbl1_completeness.py` was extended to audit Units 3 and 4. The
new checks test **properties, not file existence**: a findings file that exists
but records `loglik_monotone = false`, or a sum–product discrepancy above
1e-9, fails the rubric rather than ticking a box because the file was written.

Two identities worth singling out, both measured rather than asserted:

- **The EM lower bound is tight at the E-step posterior.** Gap = `2.3e-13`
  (machine zero); at any other `q` the gap equals the KL divergence exactly.
  That identity *is* the proof that EM cannot decrease the likelihood.
- **Sum–product *is* forward–backward.** Messages vs α, beliefs vs γ, pairwise
  beliefs vs ξ, log-partition vs log-likelihood — **every difference is exactly
  0.0**. Swapping sum for max reproduces the Viterbi path.

And one that is a genuine warning:

- **Factor loadings are not unique.** Rotating the fitted `W` by a random
  orthogonal matrix moved individual loadings by up to **1.42** while the
  log-likelihood moved by **4.5e-13**. Never report loadings as if unique.

---

## Outputs

```
reports/unit3_*.csv              10 tables      reports/unit4_*.csv       12 tables
reports/unit3_findings.json                     reports/unit4_findings.json
reports/figures/unit3_*.png      6 figures      reports/figures/unit4_*.png  4 figures
data/processed/sku_segments.parquet             data/processed/demand_regimes.parquet
reports/dashboard_verification.json             reports/screenshots/page_*.png
```

## Known limitations

- **Unit 5 not covered** — the Drive link is auth-gated (HTTP 401).
- **The HMM is fitted on the aggregate series, not per SKU.** Business-wide
  regimes are what the Christmas peak actually is, but an individual SKU can be
  on promotion while the business is QUIET.
- **Factor analysis converges slowly** — 2,378 EM iterations at tol 1e-8; tol
  1e-9 needs ~8,200 for another 0.06 of log-likelihood. The tolerance was
  chosen by measuring that knee, not by guessing.
- **The scree-elbow rule returns 1 component** on this spectrum. Reported as-is
  rather than quietly dropped; Kaiser is used for the actual choice.
- **Regime-aware safety stock is quantified, not yet A/B tested end to end.**
  Re-running the full policy ladder under regime-conditioned buffers is the
  natural next step.
- **The supply side remains simulated** — unchanged from the existing project,
  and flagged throughout the dashboard.
