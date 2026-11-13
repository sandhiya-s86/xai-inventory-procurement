# Unit 5 — Classification and Regression Trees (CART)

Topic 40 (CART) and **Practice 1** (implement CART to perform categorization),
implemented from first principles and run on the real reorder decision.

```bash
.venv\Scripts\python scripts\run_cart.py      # the whole stage
```

- Code: [`src/xaierp/trees/cart.py`](../src/xaierp/trees/cart.py)
- Tests: [`tests/test_unit5_cart.py`](../tests/test_unit5_cart.py) — 56 passing
- Dashboard: page **9 · Decision Rules**
- Rules output: `reports/unit5_tree_rules.txt`

---

## Why CART, out of the seven Unit 5 topics

This project's entire thesis is explainability. Every other classifier in it
answers *"reorder?"* with a probability and needs SHAP to explain itself. A
pruned tree answers with a **rule** — one a buyer can read, check against their
own judgement, and overrule.

CART is also the foundation the rest of Unit 5 is built on: boosting, ensemble
learning and adaptive basis function models all use trees as their base
learner, so this is the piece the others would extend.

---

## What was implemented

| Component | Detail |
|---|---|
| Splitting | Binary, exhaustive search over every feature × threshold |
| Criteria | Gini, entropy (classification); MSE (regression) |
| Vectorisation | Cumulative class counts over a sorted column evaluate every candidate threshold at once — a Python loop over thresholds is what makes naive tree code unusable at 67k rows |
| Stopping | `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` |
| **Pruning** | **Minimal cost-complexity pruning** — the weakest-link sequence |
| Class weights | `balanced`, because missing a stockout costs more than a surplus order |
| Importance | Weighted impurity decrease |
| Explanation | `decision_path`, `rule_for`, `export_text` |

### What makes it CART and not just "a decision tree"

Cost-complexity pruning. Instead of an arbitrary depth cap, price each leaf:

```
R_alpha(T) = R(T) + alpha * |leaves(T)|
```

At `alpha = 0` the full tree wins; as `alpha` rises, whole subtrees stop paying
for themselves. The weakest link is the internal node minimising

```
g(t) = [R(t) - R(T_t)] / (|leaves(T_t)| - 1)
```

The resulting sequence of subtrees is **finite and nested**, so every tree
worth considering can be enumerated and `alpha` chosen on a validation set
rather than guessed.

> **One deliberate deviation from scikit-learn**, documented because it
> surprises people: pruning here uses Breiman's *misclassification* error
> `R(t)`, where scikit-learn uses the impurity criterion. Both are correct; the
> alpha *values* are simply not comparable. The tests check the pruning
> **sequence** rather than raw alphas. A visible consequence is that
> `prune(0.0)` is not always the identity — Breiman's `T(0)` is the *smallest*
> subtree with minimal training error, so a split that lowered Gini without
> fixing a single misclassification is dropped.

---

## Results on the reorder decision

67k train / 16k val / 17k test rows, 52 features, chronological split.

### Pruning works exactly as advertised

| | Leaves | Train F1 | Val F1 | Gap | **Test F1** |
|---|---|---|---|---|---|
| Full tree | 52 | 0.7949 | 0.7785 | +0.0164 | 0.7732 |
| **Pruned** (α = 7.5e-5) | **16** | 0.7910 | 0.7812 | **+0.0099** | **0.7758** |

Pruning removed **69% of the leaves**, narrowed the train/validation gap — and
test F1 went *up*. That is the textbook outcome, measured rather than asserted.

### CART against every classifier already trained

| Model | Test F1 | Recall |
|---|---|---|
| LightGBM Classifier | 0.7780 | 0.8155 |
| **CART (readable, depth 4 — 8 leaves)** | **0.7773** | 0.7497 |
| **CART (pruned — 16 leaves)** | **0.7758** | 0.7463 |
| SVM (linear) | 0.7483 | 0.8157 |
| Logistic Regression | 0.7482 | 0.8257 |
| SVM (RBF) | 0.7179 | 0.8594 |
| LDA | 0.5517 | 0.4265 |
| Gaussian NB | 0.5397 | 0.9694 |

A 16-leaf tree you can read on one page lands within **0.0022 F1** of a
gradient-boosted ensemble of hundreds of trees. The 8-leaf tree is second.

---

## The finding that matters

The pruned tree put **98.5% of its importance on one feature**. That is a claim
about the *task*, not a quirk — so it was tested directly.

A single threshold, chosen on the validation split like every other model:

```
IF stock_vs_leadtime_demand <= 1.01  THEN reorder
```

scores **test F1 0.7777** — against **0.7780** for LightGBM, the best of eight
models. A difference of 0.0003.

**Why.** The simulator defines the label as

```
actual future lead-time demand  >  current stock * (1 + tol)
```

and `stock_vs_leadtime_demand` is

```
current stock / (trailing 28-day mean * lead time)
```

— the leakage-free proxy for that same comparison. The feature is **not**
leaking (it uses only trailing data, and `tests/test_no_leakage.py` still
passes), but it already contains the decision. The residual error is
*demand-forecast* error, which no classifier can remove because the forecast is
baked into the feature. That is why all eight models cluster near F1 0.75–0.78:
they share a ceiling set by the task, not by the algorithm.

**This is the argument for interpretable models, made concrete.** LightGBM
reaches the same ceiling and conceals why. CART makes it visible in a single
split — and the finding is a property of the task, which means it would have
gone unnoticed with black-box models alone.

---

## Other results reported because they are true

- **A single regression tree gets *worse* with depth** here: test RMSE 325.6 at
  depth 3, 361.4 at depth 5, 371.1 at depth 8. Its best R² (0.376) sits well
  below Ridge's 0.510. Trees are high-variance; this one is here to be read.
- **Gini and entropy pick almost the same splits** — test F1 differs by 0.0023.
  Gini is ~1.4× faster because it needs no logarithm in the inner loop.
- **Depth alone is a poor regulariser.** Training F1 rises monotonically with
  depth while validation F1 peaks at depth 8 and then falls; the gap widens
  from +0.006 to +0.032. Pruning beat every depth cap tried.

---

## Verification

56 tests, all passing. Two kinds:

- **Reference** — leaf counts and predictions match scikit-learn's
  `DecisionTreeClassifier` / `DecisionTreeRegressor` exactly at depths 1, 2, 3
  and 5 for both Gini and entropy, and on regression MSE. At unbounded depth
  the structures can differ by a couple of leaves purely from split tie-breaking
  (scikit-learn shuffles feature order; this takes the first best in column
  order), so that case asserts both memorise the training set and end up a
  similar size.
- **Property** — every split partitions its node exactly; no split increases
  weighted impurity; importances sum to 1 and an irrelevant feature gets
  exactly 0; alphas strictly increase while leaves strictly decrease; a pruned
  tree is a genuine *subtree* of its parent; `prune` does not mutate the
  original; **every test printed in a rule is actually satisfied by that row** —
  which is what makes the explanation an explanation rather than a
  plausible-sounding story.

---

## Not implemented

The other six Unit 5 topics — Bayesian model averaging, boosting, adaptive
basis function models, generalized additive models, ensemble learning, and
Practice 2 (ensemble classification) — are **not** covered. You asked for one
topic. CART is the natural base for boosting and ensembles if you want those
next; the `CART` class is already usable as a weak learner.
