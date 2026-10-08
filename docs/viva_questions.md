# Viva Preparation

Answers use the project's actual numbers. Where a result is unflattering, the
honest answer is given — examiners reward knowing your own weaknesses.

---

## A. Problem and motivation

**Why this problem?**
Retailers lose money two ways at once: stockouts lose sales, and overstock ties
up cash. The decision that controls both is "how much do I order, and when".
In our simulated environment the naive baseline achieves only a **79.9% fill
rate** — one unit in five that a customer wanted was not there.

**Why inventory optimisation specifically?**
It is a decision problem, not just a prediction problem. A forecast on its own
changes nothing; our own results prove it — policy P2 uses the ML forecast with
no buffer and achieves a fill rate 0.7 points *worse* than the baseline. The
value appears only when the forecast is converted into an ordering decision
that accounts for uncertainty.

**Why does ERP matter?**
Because that is where the data and the action both live. An ERP holds item,
stock, supplier and purchase-order records. A recommendation that cannot be
turned into a draft PO in the system the buyer already uses will not be used.

**What exactly does your system output?**
For a given SKU on a given date: predicted 7-day demand, a stockout-risk
classification with probability, a recommended order quantity, the supplier,
the expected cost, and a written explanation of both the forecast and the
quantity.

---

## B. Dataset

**Which dataset and why?**
UCI Online Retail II: 1,067,371 invoice lines from a UK giftware wholesaler,
Dec 2009 – Dec 2011, CC BY 4.0. Chosen primarily because it needs **no
credentials**, so the whole project reproduces from one command.

**Why not Favorita, which is richer?**
It is richer — it has promotion flags, oil prices and a holiday calendar. But
the Kaggle API returns HTTP 401 without a personal token, and getting one needs
a human to log in and accept competition rules. I judged reproducibility more
valuable than the extra columns, and documented what I gave up: no promotion
flag (I use a price-discount proxy instead) and no external regressor.

**What are its limitations?**
Three that matter. One: it is a single business in a single country, so
findings may not transfer. Two: it has no promotion field. Three — and this is
the big one — **it contains no supply-side data at all.**

**What did you simulate, and why is that acceptable?**
Stock levels, suppliers, lead times, MOQ, pack sizes, capacities and all costs.
No public sales dataset has these, and you cannot make a procurement decision
without them. They are generated in `sim/` from an explicit config under a
fixed seed, every column is flagged `is_simulated`, and the dashboard carries a
provenance warning. The alternative — inventing numbers and calling them data —
would have been dishonest.

**So are your cost savings real?**
No, and I do not claim they are. The *ranking* of the policies is a real
result about the decision rules, because all five policies face identical real
demand and identical simulated supply terms. The currency amounts are not a
claim about the actual retailer.

---

## C. EDA

**What did you discover that changed the modelling?**
Five things.
1. **The business does not trade on Saturdays** — 30 Saturday invoices versus
   5,987–8,167 on weekdays; 104 of the 135 non-trading days are Saturdays. If I
   had zero-filled those I would have injected ~17% fake zero-demand rows.
2. **Demand is skewed 24.6** with mean 28.6 and median 6.0. That kills the
   Normality assumption behind the standard safety-stock formula.
3. **31.7% of SKU-days are zero.** Demand is genuinely intermittent.
4. **Q4 uplift is +64.6%** (Sep–Nov versus the rest).
5. **P95 of 7-day demand is 8.2× P50.** That ratio is the entire business case
   for the uncertainty layer.

**Why a rolling mean?**
It estimates the current demand level while smoothing day-to-day noise. Critically
every rolling feature is computed on a `.shift(1)` series, so the window ends
*yesterday* and never includes the day being predicted.

**Why correlation, and what did it show?**
To see which variables carry signal. Demand correlates +0.64 with revenue,
+0.38 with invoice count, −0.13 with price. The weak negative price
relationship is in the expected direction but too weak to lean on, which is why
price enters only as a lagged proxy feature.

**How did you detect outliers?**
Tukey's rule: Q1 = 0, Q3 = 25, IQR = 25, upper fence 62.5. That flags
**10.9% of SKU-days**. I kept them — at 10.9% they are the wholesale orders the
business runs on, not errors. Only 7 lines above 10,000 units were removed as
data-entry corrections. I included a Huber regressor to show the robust
alternative treatment.

---

## D. Unit 1

**What is supervised learning?**
Learning a mapping from inputs to outputs using labelled examples. Here:
43 features → 7-day demand (regression), and 52 features → reorder yes/no
(classification).

**Regression vs classification?**
Regression predicts a continuous value (425 units); classification predicts a
category (reorder: yes/no). This project does both, on the same underlying data.

**Why polynomial regression?**
To demonstrate model capacity and its cost. I fitted degrees 1–4 on a 4-feature
subset. Training RMSE falls monotonically 285.77 → 278.42 while test RMSE rises
291.05 → **941.97** and test R² goes to **−4.23** — worse than predicting the
mean.

**What is overfitting? Underfitting?**
Overfitting is learning noise in the training data so that performance on new
data degrades — my degree-4 model, with a 3.2× test-error blow-up. Underfitting
is too little capacity to capture the real pattern — the 7-day moving average
baseline, test R² 0.267.

**What is a quantile?**
A value below which a given proportion of the distribution falls. P90 of 7-day
demand is 497 units, meaning 90% of SKU-weeks sell 497 or fewer. I use them for
safety stock: planning to P95 rather than P50 means being short in only ~5% of
weeks.

**What is variance?**
Expected squared deviation from the mean, `Var(X) = E[X²] − (E[X])²`. I verify
that identity numerically in notebook 06. Daily SKU demand has variance 8,891
(std 94.3) against a mean of 28.6 — variance far exceeds the mean, which is why
a Poisson assumption would also be wrong.

**What is covariance? How does it differ from correlation?**
Covariance measures joint variation: `Cov(X,Y) = E[XY] − E[X]E[Y]`.
Cov(demand, invoices) = 89.19. It is scale-dependent, so it cannot be compared
across variable pairs. Correlation normalises by both standard deviations to
give a unit-free number in [−1, 1] — the same pair correlates +0.38.

**What is Bayes rule?**
`P(H|E) = P(E|H)·P(H) / P(E)`. It inverts a conditional. I use it directly:
Gaussian Naive Bayes models `p(x|y)` and the prior `p(y)`, then applies Bayes
rule to obtain `p(y|x)` for the reorder decision.

---

## E. Unit 2

**What is MLE?**
Choosing parameters that maximise the probability of the observed data.

**How does it relate to least squares?**
They are the same estimator under Gaussian noise. With `y = Xβ + ε`,
`ε ~ N(0, σ²I)`, the log-likelihood is
`−n/2·log(2πσ²) − (1/2σ²)·‖y − Xβ‖²`. Only the last term depends on β, so
maximising likelihood *is* minimising squared error. That is why there is no
separate "MLE model" in my table — `LinearRegression` already is one.

**What is ridge regression and why use it?**
Least squares plus an L2 penalty: minimise `‖y − Xβ‖² + λ‖β‖²`. It shrinks
coefficients toward zero, trading a little bias for a lot less variance. I
needed it because my design matrix is ill-conditioned — sklearn reports
rcond ≈ 7.7e-08 for plain OLS. With α = 100, chosen on validation, Ridge gives
the best test RMSE of 288.53.

**Is ridge Bayesian?**
Yes — it is the MAP estimate under a Gaussian prior `β ~ N(0, τ²I)`, with
`λ = σ²/τ²`.

**What does Bayesian linear regression add?**
A distribution rather than a point. `BayesianRidge` learns the noise and prior
precisions from the data and returns a **predictive standard deviation per
row**, so the system can say how confident it is, not just what it expects.

**What is logistic regression?**
A discriminative classifier modelling `p(y=1|x) = σ(wᵀx + b)`. It gives
F1 0.748, recall 0.826 here.

**Generative vs discriminative — show me in your results.**
Gaussian Naive Bayes (generative) versus Logistic Regression (discriminative),
same features, same data. NB gets **recall 0.969 but precision 0.374** — it
flags nearly everything. Its conditional-independence assumption is violated
(my rolling means are strongly correlated), making it over-confident. Logistic
regression models `p(y|x)` directly, assumes nothing about the feature
distribution, and reaches F1 0.748 against NB's 0.540.

**What is an SVM?**
A maximum-margin classifier: it finds the separating hyperplane with the
largest distance to the nearest points (the support vectors).

**Linear vs non-linear SVM, and what happened in your results?**
Linear SVM: F1 **0.748**. RBF SVM: F1 **0.718**. Same C, same features, same
15,000-row subsample — so the only difference is the kernel. The non-linear
kernel **lost**, which tells me the decision boundary is close to linear in
these features. That is a negative result and I report it.

**What is a kernel? What is the kernel trick?**
A kernel `K(x,z)` is an inner product in some feature space:
`K(x,z) = φ(x)ᵀφ(z)`. The RBF kernel `exp(−γ‖x−z‖²)` corresponds to an
*infinite-dimensional* φ. The **kernel trick** is that SVM training and
prediction only ever need inner products, so you compute `K` directly and never
construct φ — making an infinite-dimensional model tractable.

**Why did you subsample for the SVMs?**
`SVC` scales roughly quadratically in sample count; 67k rows was not practical
on this machine. I trained on 15,000 rows, declared it in the model spec, and
reported `n_train_rows_used` in the results table rather than hiding it.

**What is a discriminant function?**
A function assigning a score per class, with the decision going to the highest.
LDA derives it in closed form from class-conditional Gaussians with shared
covariance. It gives my **highest precision (0.781)** but worst recall (0.427) —
the shared-covariance assumption makes it conservative.

---

## F. Methodology and results

**Why a chronological split, not random?**
Because consecutive days for the same SKU share almost all of their
rolling-window history. A random split would put day *t+1* in training and day
*t* in test, leaking the answer and inflating every score. I split on global
date cut-offs: train to 2011-05-15, validation to 2011-08-23, test
2011-08-24 → 2011-12-02, with no date in two splits.

**How did you prevent leakage?**
A stated timing convention — a row `(sku, t)` is a decision at the *start* of
day t; features may use `t−1` and earlier; the target is demand over `t … t+6`.
Every lag and rolling statistic is built on `.shift(1)`, and every rolling
window uses `min_periods = window` so partial windows become NaN and are
dropped. Then I test it: `tests/test_no_leakage.py` runs a strictly increasing
toy series where any leak is arithmetically visible, checks that
`rolling_max_7` equals exactly yesterday's demand, and flags any feature
correlating above 0.97 with the target.

**Which model won and why?**
Ridge for forecasting (test RMSE 288.53, R² 0.510); LightGBM for the reorder
classifier (F1 0.778, AUC 0.921).

**Your LightGBM lost the regression task. Why?**
It placed 6th of 7, behind the 28-day moving average. Three reasons: 31.7%
zero-inflation fragments tree splits; the test period is the Q4 peak and trees
cannot extrapolate beyond their training leaves while linear models can; and my
rolling features already encode the interactions a tree would have to discover.
I applied early stopping (it halted at 36 trees, improving RMSE from 342.93 to
311.43) and it still lost. I report it rather than dropping the model.

**Is Ridge significantly better, or is that noise?**
Against LightGBM, yes — a paired bootstrap over 1,000 resamples of the same
test rows puts the RMSE difference entirely below zero. Against OLS and
Bayesian Ridge (288.53 vs 288.96 vs 288.67) the difference is within noise; I
chose Ridge on the tie-break of numerical stability given the ill-conditioning.

**But Huber has a better MAE than Ridge. Why didn't you pick it?**
Huber wins on MAE (124.04 vs 136.36) and RMSLE (0.939 vs 1.133) precisely
because it refuses to chase the 9,480-unit outlier days. If the business cared
about typical-day accuracy, Huber would be the right choice. I declared RMSE as
the primary metric in config because inventory cost is driven by large misses.
That is a judgement about the cost function, not a claim that Huber is worse.

**Did ML actually beat the naive baseline?**
Modestly: 288.53 versus 298.13 RMSE for the 28-day moving average — a 3.2%
improvement. I am not going to oversell that. The real gain in this project
comes from the uncertainty layer, not the point forecast.

**Why SHAP? And why two different SHAP methods?**
SHAP gives per-prediction attributions with a guarantee that they sum to the
prediction. I use the method matched to the deployed model: for Ridge the
Shapley values have a **closed form** (`φᵢ = βᵢ(zᵢ − E[zᵢ])`) so they are exact;
for LightGBM, TreeSHAP is exact. I verify additivity numerically — max error
6.1e-04 for the linear model and 8.5e-13 for the tree model.

**How does the reorder decision work?**
`order = expected lead-time demand + safety stock − inventory position`, then
clip at zero, apply the supplier MOQ, round up to whole packs, and cap at
supplier capacity (rounding the cap *down* to a whole pack, since a part pack
cannot be bought).

**How is the recommended quantity explained?**
As an exact decomposition. Every one of those seven steps is a named component,
and the components sum to the final quantity to a residual of 0.000000 across
all 16,689 decisions. `tests/test_decision_explanation.py` fails if that
identity ever breaks — including a test that deliberately tampers with a
component and asserts the explainer refuses to render.

**Why does that matter?**
Because "SHAP says `rolling_mean_28` is important" does not tell a buyer why
the system asked for 1,380 units. The decomposition does: 1,042 expected
demand + 675 safety − 0 stock + 0 MOQ + pack rounding, totalling the order.

**What happens when the prediction is wrong?**
Three layers of protection. The safety buffer absorbs ordinary error. The
what-if panel lets a buyer stress-test before committing. And nothing is
auto-submitted — the output is a draft recommendation requiring human approval.
Our classifier's recall is 0.816, so ~18% of genuine reorder situations are
missed; human review of the queue is how those get caught.

**What are the main limitations?**
Five. Supply-side data is simulated. One retailer, one country, two years.
Only the top 200 SKUs by revenue (42.0% of cleaned revenue) are modelled. No
promotion data. And the policy comparison assumes the cost parameters in
`supply_config.yaml`, particularly the shortage multiplier of 2.5, which is a
guess that materially affects whether P1 or P4 is "best".

**If you had another month, what would you do?**
Get real inventory data — that single change would remove the project's biggest
caveat. After that, test on a second retailer to check transfer, and add a
cost-sensitive decision threshold on the classifier rather than the default 0.5.

---

## G. Questions you should be ready to be challenged on

**"Your fill rate improvement comes with higher cost — isn't P1 better?"**
On these cost parameters, P1 is cost-minimal (−2.9% vs baseline at 88.7% fill).
P4 costs +4.6% for 93.9% fill and cuts stockout days by 75%. Which wins depends
entirely on the shortage multiplier, currently 2.5. I put it in a config file
precisely so it can be challenged. In giftware wholesale a stockout also risks
the retailer's next order, which my simulator does not model — so I believe 2.5
understates it.

**"91% of your rows recommend a reorder. Isn't that suspicious?"**
It reflects a weekly-review policy evaluated on *every* day. A recommendation is
computed for each SKU-day independently; in operation the buyer reviews weekly,
so the relevant figure is the per-review decision, not the per-day one. The
inventory position used does net off stock already on order.

**"Why only 200 SKUs?"**
A dense panel over all 5,305 stock codes would be mostly structural zeros,
making "predict 0" a strong baseline and the exercise hollow. I rank by revenue,
require 120+ active days, and state that the modelled set covers 42.0% of
cleaned revenue. It is a config parameter, not a hard limit.

**"Can you prove no data leaked?"**
I can show the tests. Eight of them, including one that constructs a
monotonically increasing series where leakage is arithmetically impossible to
hide, and one that scans every feature for correlation above 0.97 with the
target. All 81 tests pass.
