# 07 — Evaluation Metrics

Every metric here exists to answer a specific research question. If you cannot name the RQ a metric serves, drop it — a results section padded with metrics reads as uncertainty about which one matters.

---

## 1. Forecast accuracy (RQ1)

### Primary

**RMSSE / WRMSSE** — the M5 accuracy metric. Scale-free, and safe on series containing zeros because it never divides by an actual value.

$$\text{RMSSE}_i = \sqrt{\frac{\frac{1}{h}\sum_{t=n+1}^{n+h}(y_t - \hat{y}_t)^2}{\frac{1}{n-1}\sum_{t=2}^{n}(y_t - y_{t-1})^2}}$$

The denominator is the in-sample one-step naive MSE. Aggregate with weights $w_i$ from cumulative actual sales value over a recent window:

$$\text{WRMSSE} = \sum_i w_i \cdot \text{RMSSE}_i$$

Weighting by revenue is the right call for a procurement paper: a 10% error on a high-value fast-mover matters more than on a slow-moving trinket.

**MASE** — the mean-absolute analogue; report alongside RMSSE since some readers expect it.

$$\text{MASE}_i = \frac{\frac{1}{h}\sum|y_t - \hat{y}_t|}{\frac{1}{n-1}\sum_{t=2}^{n}|y_t - y_{t-1}|}$$

For seasonal data use the seasonal naive denominator ($y_{t-m}$, $m=7$) and say which you used.

### Probabilistic

**Scaled Pinball Loss / Weighted SPL** — the M5 uncertainty metric, over the nine-quantile set.

$$\text{PL}_\tau(y,\hat q) = \begin{cases}\tau(y-\hat q) & y\ge\hat q\\ (1-\tau)(\hat q-y) & y<\hat q\end{cases}$$

$$\text{SPL}_i = \frac{\frac{1}{h}\sum_t \sum_{\tau} \text{PL}_\tau(y_t, \hat q^{(\tau)}_t)}{\frac{1}{n-1}\sum_{t=2}^n |y_t - y_{t-1}|}$$

**Empirical coverage** at nominal levels — the calibration check:

$$\text{Cov}(1-\alpha) = \frac{1}{N}\sum \mathbb{1}\left[ y \in \left[\hat q^{(\alpha/2)},\, \hat q^{(1-\alpha/2)}\right]\right]$$

Report **before and after conformal calibration**, per SBC class. Also report **mean interval width** — coverage alone is gameable by predicting infinitely wide intervals, and a reviewer will note if you report one without the other.

**CRPS** — optional; useful if you produce full predictive distributions rather than a quantile set.

### Secondary
RMSE, MAE (interpretable in units), sMAPE.

### Do not use
**MAPE.** Undefined at $y=0$, and both datasets are full of zeros. State this rejection explicitly in the paper — it is a one-line demonstration of metric literacy.

---

## 2. Inventory and service performance (RQ2, RQ3, RQ4)

All produced by the digital twin over the evaluation window, per series, then aggregated.

| Metric | Formula | Note |
|---|---|---|
| **Cycle service level (CSL)** | cycles with no stockout / total cycles | What the $z$ in safety-stock formulas targets |
| **Fill rate (unit fill)** | units satisfied from stock / units demanded | Almost always higher than CSL. **Report both** — see [04](04-methodology.md) §7.1 |
| **Stockout rate** | periods with unmet demand / total periods | Simple, intuitive |
| **Stockout events** | count of distinct stockout episodes | Distinguishes one long outage from many short ones |
| **Expected units short** | $\sum_t \max(0, D_t - \text{OH}_t)$ | Volume, not just frequency |
| **Average on-hand inventory** | $\frac{1}{T}\sum_t \text{OH}_t$ | In units and in currency |
| **Inventory turnover** | COGS over window / average inventory value | Annualise for readability |
| **Days of supply / DIO** | 365 / turnover | The metric practitioners actually quote |
| **Excess/obsolete inventory** | units held > 90 days of forward cover | Captures the overstock failure mode explicitly |
| **Spoilage / write-off** | units expired (perishables) | Only with the SKU-level Favorita release |

---

## 3. Cost (the headline)

$$\text{TotalCost} = \underbrace{\sum c_{i,s} x_{i,s,t}}_{\text{purchase}} + \underbrace{\sum K_s z_{s,t}}_{\text{ordering}} + \underbrace{\sum h_i \text{OH}_{i,t}}_{\text{holding}} + \underbrace{\sum p_i B_{i,t}}_{\text{shortage}} + \underbrace{\sum \text{writeoff}_{i,t}}_{\text{spoilage}}$$

**Always report the decomposition, not just the total.** A total that improves by 12% is interesting; knowing that holding fell 30% while shortage rose 8% is *informative*, and it is what tells the reader the mechanism.

Also report:
- **Cost per unit sold** — normalises across scenarios of different demand volume.
- **Cost relative to P0 (%)** — the incumbent-relative number the abstract will quote.
- **Decision regret** vs a clairvoyant oracle that knows realised demand:

$$\text{Regret} = \text{Cost}(\text{policy}) - \text{Cost}(\text{oracle})$$

The oracle is easy to compute (perfect-foresight order quantities under the same constraints) and gives an absolute scale: "we close 61% of the gap between the incumbent policy and perfect foresight" is a far more meaningful sentence than "we reduced cost by 12%". Include it.

**Procurement-specific**
- Number of purchase orders issued (ordering effort — practitioners care)
- Average order quantity and order frequency
- Supplier mix (share of spend per supplier)
- Weighted average lead time realised
- Expedite/emergency order count, if you model expediting

---

## 4. Explanation quality (RQ5)

### 4.1 Faithfulness

| Metric | Definition | Applies to |
|---|---|---|
| **Decomposition residual** | $\lvert Q^{\text{final}} - \sum_k \text{term}_k\rvert$ | Decision level. Must be ~0; report max and mean over all decisions |
| **Shapley efficiency check** | $\lvert SS - (\phi_\mathcal{D} + \phi_\mathcal{L})\rvert$ | Safety-stock split. Must be ~0 by construction |
| **Deletion AUC** | Area under the prediction-degradation curve as top-$k$ features are removed. **Lower is better** | Forecast level |
| **Insertion AUC** | Prediction recovery as top-$k$ features are added back. **Higher is better** | Forecast level |
| **Comprehensiveness / sufficiency** | Δ prediction when top-$k$ removed / retained only | Forecast level |
| **Counterfactual validity rate** | fraction of stated counterfactuals that hold on actual re-solve | Decision level. Target 100% |

Caveat to state: perturbation-based faithfulness metrics push inputs off-distribution, which is a known confound in the XAI evaluation literature. Use marginal-distribution baselines and acknowledge it.

### 4.2 Stability

| Metric | Definition |
|---|---|
| **Top-$k$ rank correlation** | Spearman ρ between feature rankings under $\epsilon$-perturbation |
| **Attribution $L_1$ drift** | $\lVert \phi(x) - \phi(x+\epsilon)\rVert_1 / \lVert \phi(x)\rVert_1$ |
| **Decision-term drift** | Same, over the decomposition vector |
| **Explanation consistency across folds** | Do the same drivers dominate in every fold? |

### 4.3 Human-centred

| Metric | Instrument |
|---|---|
| Comprehension accuracy | Objective probe questions with a correct answer |
| Decision time | Seconds per scenario |
| Self-reported understanding / trust / usefulness | 5-point Likert, ideally a validated scale |
| **Appropriate reliance** | Agreement rate on good recommendations *minus* agreement rate on deliberately degraded ones. **This is the metric that matters** — it separates warranted trust from blind trust |
| Override rate | From the accept/modify/reject log |

---

## 5. Computational cost

Reviewers of applied ML papers increasingly ask for this, and it is cheap to record.

| Metric | Why |
|---|---|
| Training time per model (wall-clock, hardware stated) | LightGBM vs TFT trade-off is a real finding |
| Inference time for the full catalogue | Feasibility of nightly batch |
| MILP solve time per item; MIP gap; % hitting the time limit | Honesty about optimizer tractability |
| Explanation generation time (per decision) | Counterfactual re-solves are the expensive part |
| Peak memory | Reproducibility on modest hardware |

---

## 6. Metric-to-RQ map

| Metric family | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 |
|---|:--:|:--:|:--:|:--:|:--:|
| WRMSSE / MASE | ● | ○ | | | |
| Scaled Pinball Loss | ● | ● | | | |
| Coverage & interval width | | ● | | | |
| Fill rate / CSL | ○ | ● | ● | ● | |
| Total cost + decomposition | ○ | ● | ● | ● | |
| Regret vs oracle | | ● | ● | ● | |
| Turnover / DIO / excess | | ○ | ○ | ● | |
| Supplier mix, PO count | | | ○ | ● | |
| Faithfulness / stability | | | | | ● |
| Human-centred measures | | | | | ● |
| Compute cost | ○ | | | ○ | ○ |

● primary  ○ supporting

---

## 7. Reporting conventions

1. **Effect size and confidence interval before p-value.** With 1,782 series everything is significant; only some things matter.
2. **Fixed decimal places** per metric, chosen once (WRMSSE to 4 dp, cost to 0 dp, percentages to 1 dp) and applied consistently.
3. **Bold the best value per column**, and mark values not significantly different from the best — do not bold a winner that is statistically indistinguishable from second place.
4. **Report the incumbent (P0) in every table.** A result that is not relative to the thing being replaced is not a result.
5. **Report negative results.** If conformal calibration does not improve cost, say so, in the abstract. Reviewers trust papers that report a component that did not help far more than papers where every single idea worked.
6. Currency: pick one (₹ or £) and convert consistently; Online Retail II is natively in GBP, and the Favorita costs are simulated so you may denominate them as you choose — state the choice.
