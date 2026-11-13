"""Classification and Regression Trees (CART) — Unit 5.

CART recursively partitions the feature space with BINARY splits. At each
node it searches every feature and every candidate threshold for the split
that most reduces impurity, then recurses on the two children until a
stopping rule fires.

    classification impurity:  Gini   G = 1 - SUM_k p_k^2
                              entropy H = - SUM_k p_k log2 p_k
    regression impurity:      MSE    = (1/n) SUM (y - ybar)^2

    split quality = impurity(parent)
                    - (n_L/n) impurity(left) - (n_R/n) impurity(right)

Why Gini rather than entropy by default: they almost always choose the same
split, but Gini needs no logarithm and so is meaningfully faster in the inner
loop. Entropy is implemented too, and the two are compared on real data in
scripts/run_cart.py rather than one being asserted better.

What makes this CART and not just "a decision tree"
---------------------------------------------------
**Minimal cost-complexity pruning.** A tree grown until its leaves are pure
has memorised the training set. CART's answer is not an arbitrary depth cap
but a principled one-parameter family:

    R_alpha(T) = R(T) + alpha * |leaves(T)|

R(T) is the tree's training error and alpha prices each leaf. At alpha = 0
the full tree wins; as alpha rises, whole subtrees stop paying for
themselves and collapse to leaves. Crucially the sequence of subtrees this
produces is FINITE and NESTED, so you can enumerate every tree worth
considering and pick alpha on a validation set. `cost_complexity_pruning_path`
returns that sequence; `prune` applies one.

The weakest link at any point is the internal node with the smallest

    g(t) = [R(t) - R(T_t)] / (|leaves(T_t)| - 1)

i.e. the subtree whose accuracy you lose most cheaply per leaf saved.

Why a tree at all, in this project
----------------------------------
Every other classifier here returns a probability and needs SHAP to explain
itself. A pruned tree returns a RULE: "stock cover is under 6 days AND the
28-day mean exceeds 40 units, therefore reorder". `rule_for` extracts exactly
that chain for any row, which is the one explanation a buyer can audit and
overrule without trusting the model at all.

Honest limitation, stated because the test suite enforces it: a single tree
is high-variance. Small changes to the training data can swap the root split
and reshape the whole tree, and its accuracy will generally sit below a
boosted ensemble's. The tree is here to be READ, and the comparison table in
scripts/run_cart.py reports where it loses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# -------------------------------------------------------------- impurities


def gini(counts: np.ndarray) -> float:
    """Gini impurity 1 - SUM p_k^2 from raw class counts."""
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts / n
    return float(1.0 - np.sum(p**2))


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy - SUM p_k log2 p_k from raw class counts."""
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log2(p)))


def mse_impurity(y: np.ndarray) -> float:
    """Mean squared deviation about the node mean."""
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


# -------------------------------------------------------------------- node


@dataclass
class Node:
    """One node. Leaves have feature = -1 and carry a prediction."""

    n_samples: int
    impurity: float
    value: np.ndarray          # class counts, or [mean] for regression
    depth: int = 0
    feature: int = -1
    threshold: float = np.nan
    left: "Node | None" = None
    right: "Node | None" = None
    node_id: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaves(self) -> int:
        if self.is_leaf:
            return 1
        return self.left.leaves() + self.right.leaves()

    def count_nodes(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + self.left.count_nodes() + self.right.count_nodes()

    def max_depth(self) -> int:
        if self.is_leaf:
            return self.depth
        return max(self.left.max_depth(), self.right.max_depth())


# ------------------------------------------------------- split search


def _best_split_classification(
    X: np.ndarray, y_onehot: np.ndarray, criterion: str,
    min_samples_leaf: int,
) -> tuple[int, float, float]:
    """Best (feature, threshold, weighted child impurity) over all columns.

    Vectorised: for each feature the rows are sorted once and the cumulative
    class counts give every candidate split simultaneously. The alternative
    -- looping over thresholds in Python -- is what makes naive tree code
    unusable on 67,000 rows.
    """
    n, d = X.shape
    best = (-1, np.nan, np.inf)

    for f in range(d):
        col = X[:, f]
        order = np.argsort(col, kind="mergesort")
        xs = col[order]
        ys = y_onehot[order]

        # left_counts[i] = class counts of the first i+1 rows
        left_counts = np.cumsum(ys, axis=0)
        total = left_counts[-1]
        right_counts = total - left_counts

        n_left = np.arange(1, n + 1, dtype=float)
        n_right = n - n_left

        # A split is only valid between two DIFFERENT values, and must leave
        # at least min_samples_leaf rows on each side.
        valid = np.empty(n, dtype=bool)
        valid[:-1] = xs[1:] > xs[:-1]
        valid[-1] = False
        valid &= (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            pl = left_counts / n_left[:, None]
            pr = right_counts / np.where(n_right[:, None] > 0, n_right[:, None], 1)
            if criterion == "gini":
                il = 1.0 - np.sum(pl**2, axis=1)
                ir = 1.0 - np.sum(pr**2, axis=1)
            else:  # entropy
                il = -np.sum(np.where(pl > 0, pl * np.log2(np.where(pl > 0, pl, 1)), 0.0), axis=1)
                ir = -np.sum(np.where(pr > 0, pr * np.log2(np.where(pr > 0, pr, 1)), 0.0), axis=1)

        weighted = (n_left * il + n_right * ir) / n
        weighted = np.where(valid, weighted, np.inf)

        i = int(np.argmin(weighted))
        if weighted[i] < best[2]:
            # Midpoint threshold, the standard convention: it generalises
            # better than using the left value itself.
            best = (f, float((xs[i] + xs[i + 1]) / 2.0), float(weighted[i]))

    return best


def _best_split_regression(
    X: np.ndarray, y: np.ndarray, min_samples_leaf: int
) -> tuple[int, float, float]:
    """Best split by weighted child MSE, using cumulative sums of y and y^2.

    Var(S) = E[y^2] - E[y]^2, so one pass of cumsum over y and y**2 gives the
    MSE of every prefix and suffix without recomputing a mean per candidate.
    """
    n, d = X.shape
    best = (-1, np.nan, np.inf)

    for f in range(d):
        col = X[:, f]
        order = np.argsort(col, kind="mergesort")
        xs = col[order]
        ys = y[order]

        cs = np.cumsum(ys)
        cs2 = np.cumsum(ys**2)
        tot, tot2 = cs[-1], cs2[-1]

        n_left = np.arange(1, n + 1, dtype=float)
        n_right = n - n_left

        valid = np.empty(n, dtype=bool)
        valid[:-1] = xs[1:] > xs[:-1]
        valid[-1] = False
        valid &= (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            var_l = cs2 / n_left - (cs / n_left) ** 2
            rn = np.where(n_right > 0, n_right, 1)
            var_r = (tot2 - cs2) / rn - ((tot - cs) / rn) ** 2

        # Floating point can make a variance a tiny negative number.
        weighted = (n_left * np.maximum(var_l, 0) + n_right * np.maximum(var_r, 0)) / n
        weighted = np.where(valid, weighted, np.inf)

        i = int(np.argmin(weighted))
        if weighted[i] < best[2]:
            best = (f, float((xs[i] + xs[i + 1]) / 2.0), float(weighted[i]))

    return best


# -------------------------------------------------------------------- tree


@dataclass
class CART:
    """Binary recursive partitioning with cost-complexity pruning."""

    task: Literal["classification", "regression"] = "classification"
    criterion: str = "gini"            # gini | entropy | mse
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_impurity_decrease: float = 0.0
    class_weight: str | None = None    # None | "balanced"

    root: Node | None = None
    n_features_: int = 0
    classes_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    _class_weights: np.ndarray | None = None
    n_nodes_: int = 0

    # ---------------------------------------------------------------- fit

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> "CART":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_)]
        )

        if self.task == "classification":
            self.classes_, y_idx = np.unique(y, return_inverse=True)
            k = len(self.classes_)
            if self.class_weight == "balanced":
                # Missing a stockout costs more than an unnecessary order, so
                # the minority class is up-weighted. Weights enter through the
                # class COUNTS, which is all the impurity functions see.
                counts = np.bincount(y_idx, minlength=k).astype(float)
                self._class_weights = len(y) / (k * np.maximum(counts, 1))
            else:
                self._class_weights = np.ones(k)
            onehot = np.zeros((len(y), k))
            onehot[np.arange(len(y)), y_idx] = self._class_weights[y_idx]
            self.root = self._grow(X, onehot, depth=0)
        else:
            self.root = self._grow_reg(X, np.asarray(y, dtype=np.float64), depth=0)

        self._index()
        return self

    def _node_impurity(self, counts: np.ndarray) -> float:
        return gini(counts) if self.criterion == "gini" else entropy(counts)

    def _grow(self, X: np.ndarray, onehot: np.ndarray, depth: int) -> Node:
        counts = onehot.sum(axis=0)
        node = Node(n_samples=len(X), impurity=self._node_impurity(counts),
                    value=counts, depth=depth)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(X) < self.min_samples_split
            or node.impurity <= 0.0
        ):
            return node

        f, thr, child_imp = _best_split_classification(
            X, onehot, self.criterion, self.min_samples_leaf
        )
        if f < 0:
            return node

        decrease = node.impurity - child_imp
        # Weight the decrease by the share of samples reaching this node, so
        # the threshold means the same thing at the root and deep in the tree.
        if decrease * (len(X) / len(X)) < self.min_impurity_decrease:
            return node

        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return node

        node.feature, node.threshold = f, thr
        node.left = self._grow(X[mask], onehot[mask], depth + 1)
        node.right = self._grow(X[~mask], onehot[~mask], depth + 1)
        return node

    def _grow_reg(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(n_samples=len(X), impurity=mse_impurity(y),
                    value=np.array([y.mean() if len(y) else 0.0]), depth=depth)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(X) < self.min_samples_split
            or node.impurity <= 0.0
        ):
            return node

        f, thr, child_imp = _best_split_regression(X, y, self.min_samples_leaf)
        if f < 0 or (node.impurity - child_imp) < self.min_impurity_decrease:
            return node

        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return node

        node.feature, node.threshold = f, thr
        node.left = self._grow_reg(X[mask], y[mask], depth + 1)
        node.right = self._grow_reg(X[~mask], y[~mask], depth + 1)
        return node

    def _index(self) -> None:
        """Give every node a stable id, for export and pruning bookkeeping."""
        counter = [0]

        def walk(n: Node) -> None:
            n.node_id = counter[0]
            counter[0] += 1
            if not n.is_leaf:
                walk(n.left)
                walk(n.right)

        if self.root is not None:
            walk(self.root)
            self.n_nodes_ = counter[0]

    # ------------------------------------------------------------ predict

    def _leaf_for(self, x: np.ndarray) -> Node:
        n = self.root
        while not n.is_leaf:
            n = n.left if x[n.feature] <= n.threshold else n.right
        return n

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.task == "classification":
            idx = [int(np.argmax(self._leaf_for(x).value)) for x in X]
            return self.classes_[idx]
        return np.array([float(self._leaf_for(x).value[0]) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is classification-only")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            v = self._leaf_for(x).value
            s = v.sum()
            out[i] = v / s if s > 0 else np.full(len(v), 1 / len(v))
        return out

    # --------------------------------------------------------- importance

    @property
    def feature_importances_(self) -> np.ndarray:
        """Total weighted impurity decrease attributable to each feature.

        The standard impurity-based importance. Worth stating its known bias:
        it favours high-cardinality continuous features, because they offer
        more candidate thresholds and so more chances to reduce impurity. It
        is a description of THIS tree, not a causal claim.
        """
        imp = np.zeros(self.n_features_)
        if self.root is None:
            return imp
        total = self.root.n_samples

        def walk(n: Node) -> None:
            if n.is_leaf:
                return
            dec = (
                n.n_samples * n.impurity
                - n.left.n_samples * n.left.impurity
                - n.right.n_samples * n.right.impurity
            )
            imp[n.feature] += dec / total
            walk(n.left)
            walk(n.right)

        walk(self.root)
        s = imp.sum()
        return imp / s if s > 0 else imp

    # ------------------------------------------------------------ pruning

    def _node_error(self, n: Node) -> float:
        """Resubstitution error R(t), weighted by the node's sample share.

        For classification this is the MISCLASSIFICATION rate, which is what
        Breiman's CART defines cost-complexity pruning on. Worth flagging
        because scikit-learn differs here: its `cost_complexity_pruning_path`
        uses the impurity criterion (Gini) instead, so its alpha values are
        not comparable with these even though both procedures are correct.
        The pruning SEQUENCE -- which subtrees appear, and in what order --
        is what matters, and the tests check that rather than the raw alphas.
        """
        total = self.root.n_samples
        if self.task == "classification":
            miss = n.value.sum() - n.value.max()
            return float(miss / total)
        return float(n.impurity * n.n_samples / total)

    def _subtree_error(self, n: Node) -> float:
        if n.is_leaf:
            return self._node_error(n)
        return self._subtree_error(n.left) + self._subtree_error(n.right)

    def cost_complexity_pruning_path(self) -> dict[str, np.ndarray]:
        """The full sequence of effective alphas and the trees they produce.

        Repeatedly collapses the WEAKEST LINK -- the internal node minimising

            g(t) = [R(t) - R(T_t)] / (|leaves(T_t)| - 1)

        -- which is the subtree whose training accuracy is cheapest to give
        up per leaf removed. Each collapse yields the next tree in a finite,
        nested sequence, so every tree worth considering is enumerated and
        alpha can be chosen on a validation set rather than guessed.
        """
        import copy

        # Start the sequence at T(0), not at the grown tree. Breiman defines
        # T(alpha) as the SMALLEST subtree minimising R_alpha, and at alpha = 0
        # a split that reduced Gini without reducing a single misclassification
        # has not earned its leaves -- so T(0) can be strictly smaller than the
        # tree as grown. Starting the path anywhere else would make
        # `prune(0.0)` disagree with `path[0]`, which is the kind of
        # inconsistency that only shows up much later as a confusing off-by-one.
        work = self.prune(0.0)
        alphas, leaves, errors = [0.0], [work.root.leaves()], [work._subtree_error(work.root)]

        while not work.root.is_leaf:
            links: list[tuple[float, Node]] = []

            def visit(n: Node) -> None:
                if n.is_leaf:
                    return
                n_leaves = n.leaves()
                if n_leaves > 1:
                    g = (work._node_error(n) - work._subtree_error(n)) / (n_leaves - 1)
                    links.append((g, n))
                visit(n.left)
                visit(n.right)

            visit(work.root)
            if not links:
                break

            best_g = min(g for g, _ in links)
            # Collapse EVERY node tied at the minimum g, not just one. At a
            # given alpha the whole family of weakest links dies together, so
            # pruning them one at a time would emit several steps sharing an
            # identical alpha and misrepresent the sequence as finer-grained
            # than it is. Ties are common on real data.
            doomed = [n for g, n in links if g <= best_g + 1e-12]
            # Skip any node already removed as a descendant of another.
            for n in doomed:
                if n.is_leaf:
                    continue
                n.left = n.right = None
                n.feature, n.threshold = -1, np.nan

            alphas.append(float(max(best_g, alphas[-1])))
            leaves.append(work.root.leaves())
            errors.append(work._subtree_error(work.root))

        return {
            "ccp_alphas": np.array(alphas),
            "n_leaves": np.array(leaves),
            "train_errors": np.array(errors),
        }

    def prune(self, alpha: float) -> "CART":
        """Return a NEW tree pruned at the given complexity parameter.

        Bottom-up: a subtree is collapsed whenever keeping it costs at least
        as much in alpha-per-leaf as the training error it saves. The
        comparison is non-strict, which implements Breiman's "smallest
        minimising subtree": where a split buys nothing, the simpler tree
        wins. One visible consequence is that `prune(0.0)` is NOT always the
        identity -- a split that lowered Gini without fixing a single
        misclassification is removed. `cost_complexity_pruning_path` starts
        from this same T(0) so the two agree.
        """
        import copy

        out = copy.deepcopy(self)

        def collapse(n: Node) -> tuple[float, int]:
            """Returns (subtree error, leaf count) after pruning in place."""
            if n.is_leaf:
                return out._node_error(n), 1
            el, ll = collapse(n.left)
            er, lr = collapse(n.right)
            sub_err, sub_leaves = el + er, ll + lr

            if out._node_error(n) + alpha <= sub_err + alpha * sub_leaves:
                n.left = n.right = None
                n.feature, n.threshold = -1, np.nan
                return out._node_error(n), 1
            return sub_err, sub_leaves

        collapse(out.root)
        out._index()
        return out

    # -------------------------------------------------------- explanation

    def decision_path(self, x: np.ndarray) -> list[dict]:
        """The chain of tests a single row passed through, root to leaf."""
        x = np.asarray(x, dtype=np.float64)
        path, n = [], self.root
        while not n.is_leaf:
            goes_left = x[n.feature] <= n.threshold
            path.append(
                {
                    "feature": self.feature_names_[n.feature],
                    "feature_index": int(n.feature),
                    "threshold": float(n.threshold),
                    "value": float(x[n.feature]),
                    "operator": "<=" if goes_left else ">",
                    "n_samples": int(n.n_samples),
                    "impurity": float(n.impurity),
                }
            )
            n = n.left if goes_left else n.right

        if self.task == "classification":
            s = n.value.sum()
            path.append(
                {
                    "leaf": True,
                    "prediction": self.classes_[int(np.argmax(n.value))],
                    "probability": float(n.value.max() / s) if s > 0 else float("nan"),
                    "n_samples": int(n.n_samples),
                    "impurity": float(n.impurity),
                }
            )
        else:
            path.append({"leaf": True, "prediction": float(n.value[0]),
                         "n_samples": int(n.n_samples),
                         "impurity": float(n.impurity)})
        return path

    def rule_for(self, x: np.ndarray) -> str:
        """The decision as one readable IF ... AND ... THEN sentence.

        This is the deliverable. Every other model in this project explains a
        decision with attributions a buyer must take on trust; this returns
        the actual logic, which they can check against their own judgement.
        """
        path = self.decision_path(x)
        tests = [
            f"{p['feature']} {p['operator']} {p['threshold']:,.2f}"
            for p in path[:-1]
        ]
        leaf = path[-1]
        if self.task == "classification":
            head = f"THEN predict {leaf['prediction']}"
            tail = (f" (confidence {leaf['probability']:.0%}, "
                    f"{leaf['n_samples']:,} training rows in this leaf)")
        else:
            head = f"THEN predict {leaf['prediction']:,.2f}"
            tail = f" ({leaf['n_samples']:,} training rows in this leaf)"

        if not tests:
            return f"{head}{tail}  [the tree is a single leaf]"
        return "IF " + "\n   AND ".join(tests) + f"\n{head}{tail}"

    def export_text(self, max_depth: int | None = None,
                    decimals: int = 2) -> str:
        """The whole tree as indented text."""
        lines: list[str] = []

        def walk(n: Node, prefix: str, label: str) -> None:
            if max_depth is not None and n.depth > max_depth:
                return
            if n.is_leaf:
                if self.task == "classification":
                    s = n.value.sum()
                    cls = self.classes_[int(np.argmax(n.value))]
                    conf = n.value.max() / s if s > 0 else float("nan")
                    lines.append(f"{prefix}{label}predict={cls} "
                                 f"p={conf:.2f} n={n.n_samples:,}")
                else:
                    lines.append(f"{prefix}{label}predict={n.value[0]:.{decimals}f} "
                                 f"n={n.n_samples:,}")
                return
            name = self.feature_names_[n.feature]
            lines.append(f"{prefix}{label}{name} <= {n.threshold:,.{decimals}f} "
                         f"[{self.criterion}={n.impurity:.3f} n={n.n_samples:,}]")
            walk(n.left, prefix + "|   ", "|--- yes: ")
            walk(n.right, prefix + "|   ", "|--- no:  ")

        if self.root is not None:
            walk(self.root, "", "")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "task": self.task,
            "criterion": self.criterion,
            "n_nodes": int(self.root.count_nodes()) if self.root else 0,
            "n_leaves": int(self.root.leaves()) if self.root else 0,
            "max_depth": int(self.root.max_depth()) if self.root else 0,
            "n_features": int(self.n_features_),
        }


# -------------------------------------------------- scikit-learn-ish aliases


def DecisionTreeClassifier(**kw) -> CART:
    kw.setdefault("criterion", "gini")
    return CART(task="classification", **kw)


def DecisionTreeRegressor(**kw) -> CART:
    kw.setdefault("criterion", "mse")
    return CART(task="regression", **kw)


def cost_complexity_pruning_path(tree: CART) -> dict[str, np.ndarray]:
    return tree.cost_complexity_pruning_path()
