"""Unit 5 -- CART computes what it claims to.

Reference tests pin the from-scratch tree against scikit-learn's
DecisionTreeClassifier / DecisionTreeRegressor: same depths, same criteria,
same leaf counts, same predictions. Property tests check the guarantees that
make it CART rather than just "a tree" -- the nested pruning sequence, the
monotone alpha path, and the fact that a pruned tree is a subtree of the one
it came from.

One deliberate deviation is documented and tested for: cost-complexity
pruning here uses Breiman's MISCLASSIFICATION error R(t), where scikit-learn
uses the impurity criterion. Both are correct procedures; their alpha values
are simply not comparable, so the tests check the pruning SEQUENCE rather
than the raw alphas.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier as SKClf
from sklearn.tree import DecisionTreeRegressor as SKReg

from xaierp.trees.cart import (
    CART,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    entropy,
    gini,
    mse_impurity,
)

SEED = 0


@pytest.fixture(scope="module")
def clf_data():
    return make_classification(
        n_samples=500, n_features=8, n_informative=5, random_state=SEED
    )


@pytest.fixture(scope="module")
def reg_data():
    return make_regression(
        n_samples=400, n_features=6, noise=8.0, random_state=SEED
    )


# ================================================================ impurity


def test_gini_of_a_pure_node_is_zero():
    assert gini(np.array([100.0, 0.0])) == pytest.approx(0.0)


def test_gini_of_a_balanced_binary_node_is_one_half():
    assert gini(np.array([50.0, 50.0])) == pytest.approx(0.5)


def test_gini_of_k_balanced_classes_is_one_minus_one_over_k():
    for k in (2, 3, 4, 5):
        assert gini(np.ones(k) * 10) == pytest.approx(1 - 1 / k)


def test_entropy_of_a_balanced_binary_node_is_one_bit():
    assert entropy(np.array([50.0, 50.0])) == pytest.approx(1.0)


def test_entropy_of_a_pure_node_is_zero():
    assert entropy(np.array([0.0, 77.0])) == pytest.approx(0.0)


def test_entropy_of_k_balanced_classes_is_log2_k():
    for k in (2, 4, 8):
        assert entropy(np.ones(k)) == pytest.approx(np.log2(k))


def test_impurity_of_an_empty_node_is_zero():
    assert gini(np.zeros(3)) == 0.0
    assert entropy(np.zeros(3)) == 0.0
    assert mse_impurity(np.array([])) == 0.0


def test_mse_impurity_matches_numpy_variance():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    assert mse_impurity(y) == pytest.approx(float(np.var(y)))


# ====================================================== vs scikit-learn


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_depth", [1, 2, 3, 5])
def test_classifier_matches_sklearn(clf_data, criterion, max_depth):
    X, y = clf_data
    mine = CART(task="classification", criterion=criterion,
                max_depth=max_depth).fit(X, y)
    theirs = SKClf(criterion=criterion, max_depth=max_depth,
                   random_state=SEED).fit(X, y)
    assert mine.root.leaves() == theirs.get_n_leaves()
    assert accuracy_score(y, mine.predict(X)) == pytest.approx(
        accuracy_score(y, theirs.predict(X))
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_unconstrained_classifier_agrees_with_sklearn_up_to_tie_breaking(
    clf_data, criterion
):
    """Grown to purity, leaf counts can differ by a handful. That is expected.

    When several candidate splits reduce impurity by exactly the same amount,
    the winner is arbitrary. scikit-learn breaks those ties by shuffling the
    feature order with its RNG; this implementation takes the first best in
    column order. Deep in a tree, where nodes are small and ties are common,
    the two make different-but-equally-valid choices and the structures drift
    slightly apart. What must still hold is the thing that matters: both
    memorise the training set perfectly and end up a similar size.
    """
    X, y = clf_data
    mine = CART(task="classification", criterion=criterion).fit(X, y)
    theirs = SKClf(criterion=criterion, random_state=SEED).fit(X, y)

    assert accuracy_score(y, mine.predict(X)) == pytest.approx(1.0)
    assert accuracy_score(y, theirs.predict(X)) == pytest.approx(1.0)
    assert abs(mine.root.leaves() - theirs.get_n_leaves()) <= 3


@pytest.mark.parametrize("max_depth", [1, 2, 4, None])
def test_regressor_matches_sklearn(reg_data, max_depth):
    X, y = reg_data
    mine = CART(task="regression", criterion="mse", max_depth=max_depth).fit(X, y)
    theirs = SKReg(max_depth=max_depth, random_state=SEED).fit(X, y)
    assert mine.root.leaves() == theirs.get_n_leaves()
    assert mean_squared_error(y, mine.predict(X)) == pytest.approx(
        mean_squared_error(y, theirs.predict(X)), rel=1e-9
    )


def test_min_samples_leaf_is_respected(clf_data):
    X, y = clf_data
    t = CART(task="classification", min_samples_leaf=40).fit(X, y)

    def walk(n):
        if n.is_leaf:
            assert n.n_samples >= 40
            return
        walk(n.left)
        walk(n.right)

    walk(t.root)


def test_max_depth_is_respected(clf_data):
    X, y = clf_data
    for d in (1, 2, 4):
        assert CART(task="classification", max_depth=d).fit(X, y).root.max_depth() <= d


# =================================================== structural properties


def test_an_unconstrained_tree_fits_the_training_data_perfectly(clf_data):
    """Grown to purity, a tree memorises. That is the motivation for pruning."""
    X, y = clf_data
    t = CART(task="classification").fit(X, y)
    assert accuracy_score(y, t.predict(X)) == pytest.approx(1.0)


def test_every_leaf_is_pure_in_an_unconstrained_tree(clf_data):
    X, y = clf_data
    t = CART(task="classification").fit(X, y)

    def walk(n):
        if n.is_leaf:
            assert n.impurity == pytest.approx(0.0)
            return
        walk(n.left)
        walk(n.right)

    walk(t.root)


def test_leaf_sample_counts_sum_to_the_training_size(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=4).fit(X, y)
    total = []

    def walk(n):
        if n.is_leaf:
            total.append(n.n_samples)
            return
        # Every split partitions its node exactly: nothing lost, nothing shared.
        assert n.left.n_samples + n.right.n_samples == n.n_samples
        walk(n.left)
        walk(n.right)

    walk(t.root)
    assert sum(total) == len(X)


def test_a_split_never_increases_weighted_impurity(clf_data):
    """The defining property of a greedy impurity-reducing split."""
    X, y = clf_data
    t = CART(task="classification", max_depth=5).fit(X, y)

    def walk(n):
        if n.is_leaf:
            return
        child = (n.left.n_samples * n.left.impurity
                 + n.right.n_samples * n.right.impurity) / n.n_samples
        assert child <= n.impurity + 1e-12
        walk(n.left)
        walk(n.right)

    walk(t.root)


def test_predictions_are_constant_within_a_leaf(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=3).fit(X, y)
    leaves = {}
    for x in X:
        n = t._leaf_for(x)
        p = int(t.predict(x.reshape(1, -1))[0])
        leaves.setdefault(id(n), set()).add(p)
    assert all(len(v) == 1 for v in leaves.values())


def test_predict_proba_rows_are_distributions(clf_data):
    X, y = clf_data
    p = CART(task="classification", max_depth=4).fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_a_single_class_target_yields_a_single_leaf():
    X = np.random.default_rng(SEED).normal(size=(50, 3))
    t = CART(task="classification").fit(X, np.zeros(50, dtype=int))
    assert t.root.is_leaf
    assert t.root.leaves() == 1


def test_identical_rows_cannot_be_split():
    """No valid threshold exists between equal values."""
    X = np.ones((40, 3))
    y = np.array([0, 1] * 20)
    assert CART(task="classification").fit(X, y).root.is_leaf


# ------------------------------------------------------------ importances


def test_feature_importances_sum_to_one(clf_data):
    X, y = clf_data
    imp = CART(task="classification", max_depth=5).fit(X, y).feature_importances_
    assert imp.sum() == pytest.approx(1.0)
    assert (imp >= 0).all()


def test_an_irrelevant_feature_gets_zero_importance():
    """A pure-noise column appended to a separable problem earns nothing."""
    rng = np.random.default_rng(SEED)
    signal = np.concatenate([np.zeros(100), np.ones(100)])
    X = np.column_stack([signal, rng.normal(size=200)])
    y = signal.astype(int)
    imp = CART(task="classification", max_depth=1).fit(X, y).feature_importances_
    assert imp[0] == pytest.approx(1.0)
    assert imp[1] == pytest.approx(0.0)


def test_a_single_leaf_tree_has_zero_importances():
    X = np.ones((30, 4))
    t = CART(task="classification").fit(X, np.zeros(30, dtype=int))
    assert t.feature_importances_.sum() == pytest.approx(0.0)


# =============================================== cost-complexity pruning


def test_pruning_alphas_increase_and_leaves_decrease(clf_data):
    """The nested sequence: each step costs more per leaf and keeps fewer."""
    X, y = clf_data
    path = CART(task="classification").fit(X, y).cost_complexity_pruning_path()
    assert np.all(np.diff(path["ccp_alphas"]) > 0), "alphas must strictly increase"
    assert np.all(np.diff(path["n_leaves"]) < 0), "leaves must strictly decrease"


def test_pruning_path_starts_at_T0_and_ends_at_a_stump(clf_data):
    """The path begins at T(0) and must agree with prune(0.0).

    T(0) is the SMALLEST subtree with minimal training error, so it can be
    strictly smaller than the tree as grown: a split that lowered Gini
    without fixing a single misclassification has not earned its leaves.
    The path and prune() must start from the same place.
    """
    X, y = clf_data
    t = CART(task="classification").fit(X, y)
    path = t.cost_complexity_pruning_path()
    assert path["ccp_alphas"][0] == 0.0
    assert path["n_leaves"][0] == t.prune(0.0).root.leaves()
    assert path["n_leaves"][0] <= t.root.leaves()
    assert path["n_leaves"][-1] == 1


def test_training_error_rises_monotonically_as_the_tree_is_pruned(clf_data):
    """Pruning can only ever hurt the TRAINING fit -- that is the trade."""
    X, y = clf_data
    path = CART(task="classification").fit(X, y).cost_complexity_pruning_path()
    assert np.all(np.diff(path["train_errors"]) >= -1e-12)


def test_larger_alpha_never_yields_more_leaves(clf_data):
    X, y = clf_data
    t = CART(task="classification").fit(X, y)
    sizes = [t.prune(a).root.leaves() for a in [0.0, 0.001, 0.01, 0.05, 0.2, 1.0]]
    assert all(b <= a for a, b in zip(sizes, sizes[1:])), sizes


def test_pruning_at_zero_removes_only_splits_that_bought_nothing(clf_data):
    """prune(0.0) is NOT the identity, and that is correct.

    At alpha = 0 the cost is pure training error, so Breiman's smallest
    minimising subtree drops any split that reduced Gini without correcting a
    single misclassification. The tree may therefore shrink -- but the
    training error must not move at all.
    """
    X, y = clf_data
    t = CART(task="classification", max_depth=6).fit(X, y)
    p = t.prune(0.0)
    assert p.root.leaves() <= t.root.leaves()
    assert accuracy_score(y, p.predict(X)) == pytest.approx(
        accuracy_score(y, t.predict(X))
    )


def test_pruning_at_zero_is_idempotent(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=6).fit(X, y)
    once = t.prune(0.0)
    assert once.prune(0.0).root.leaves() == once.root.leaves()


def test_a_large_alpha_collapses_the_tree_to_its_root(clf_data):
    X, y = clf_data
    t = CART(task="classification").fit(X, y)
    assert t.prune(1e9).root.is_leaf


def test_pruning_does_not_mutate_the_original_tree(clf_data):
    """prune() must return a new tree; callers rely on reusing the original."""
    X, y = clf_data
    t = CART(task="classification").fit(X, y)
    before = t.root.leaves()
    t.prune(0.05)
    assert t.root.leaves() == before


def test_a_pruned_tree_is_a_subtree_of_its_parent(clf_data):
    """Every surviving split must exist, identically, in the original."""
    X, y = clf_data
    t = CART(task="classification", max_depth=6).fit(X, y)
    p = t.prune(0.01)

    def compare(pn, tn):
        if pn.is_leaf:
            return
        assert not tn.is_leaf, "pruned tree split where the original did not"
        assert pn.feature == tn.feature
        assert pn.threshold == pytest.approx(tn.threshold)
        compare(pn.left, tn.left)
        compare(pn.right, tn.right)

    compare(p.root, t.root)


def test_pruning_reduces_the_train_test_gap():
    """The point of pruning: trade training fit for generalisation."""
    X, y = make_classification(n_samples=600, n_features=10, n_informative=4,
                               flip_y=0.25, random_state=SEED)
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    full = CART(task="classification").fit(Xtr, ytr)
    pruned = full.prune(0.02)

    gap_full = (accuracy_score(ytr, full.predict(Xtr))
                - accuracy_score(yte, full.predict(Xte)))
    gap_pruned = (accuracy_score(ytr, pruned.predict(Xtr))
                  - accuracy_score(yte, pruned.predict(Xte)))
    assert pruned.root.leaves() < full.root.leaves()
    assert gap_pruned < gap_full


# ==================================================== explanation output


def test_decision_path_ends_at_a_leaf_and_matches_the_prediction(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=4).fit(X, y)
    for i in range(10):
        path = t.decision_path(X[i])
        assert path[-1]["leaf"] is True
        assert path[-1]["prediction"] == t.predict(X[i:i + 1])[0]


def test_decision_path_tests_are_all_satisfied_by_the_row(clf_data):
    """Every test in the reported rule must actually hold for that row.

    This is what makes the explanation an explanation: if a printed condition
    were not true of the input, the rule would be a plausible-sounding story
    rather than the arithmetic that produced the answer.
    """
    X, y = clf_data
    t = CART(task="classification", max_depth=5).fit(X, y)
    for i in range(20):
        for step in t.decision_path(X[i])[:-1]:
            v, thr = step["value"], step["threshold"]
            if step["operator"] == "<=":
                assert v <= thr
            else:
                assert v > thr


def test_decision_path_length_matches_the_leaf_depth(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=4).fit(X, y)
    for i in range(10):
        assert len(t.decision_path(X[i])) - 1 == t._leaf_for(X[i]).depth


def test_rule_text_names_real_features(clf_data):
    X, y = clf_data
    names = [f"feat_{i}" for i in range(X.shape[1])]
    t = CART(task="classification", max_depth=3).fit(X, y, feature_names=names)
    rule = t.rule_for(X[0])
    assert "IF" in rule and "THEN predict" in rule
    assert any(n in rule for n in names)


def test_export_text_lists_every_node(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=3).fit(X, y)
    assert len(t.export_text().strip().splitlines()) == t.root.count_nodes()


def test_a_single_leaf_tree_still_produces_a_rule():
    X = np.ones((20, 2))
    t = CART(task="classification").fit(X, np.zeros(20, dtype=int))
    assert "single leaf" in t.rule_for(X[0])


# ======================================================== class weighting


def test_balanced_class_weight_raises_recall_on_a_rare_class():
    """Missing a stockout costs more than an unnecessary order."""
    rng = np.random.default_rng(SEED)
    n = 1000
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.6 * rng.normal(size=n) > 1.6).astype(int)  # ~5% positive
    assert 0.02 < y.mean() < 0.15

    plain = CART(task="classification", max_depth=3).fit(X, y)
    weighted = CART(task="classification", max_depth=3,
                    class_weight="balanced").fit(X, y)

    def recall(m):
        p = m.predict(X)
        return float(((p == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))

    assert recall(weighted) >= recall(plain)


# ========================================================= aliases / misc


def test_the_constructor_aliases_set_the_right_task(clf_data, reg_data):
    X, y = clf_data
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).task == "classification"
    Xr, yr = reg_data
    assert DecisionTreeRegressor(max_depth=2).fit(Xr, yr).task == "regression"


def test_predict_proba_is_rejected_for_a_regression_tree(reg_data):
    X, y = reg_data
    t = CART(task="regression", criterion="mse", max_depth=2).fit(X, y)
    with pytest.raises(ValueError, match="classification-only"):
        t.predict_proba(X)


def test_summary_reports_consistent_structure(clf_data):
    X, y = clf_data
    t = CART(task="classification", max_depth=4).fit(X, y)
    s = t.summary()
    assert s["n_leaves"] == t.root.leaves()
    assert s["n_nodes"] == t.root.count_nodes()
    assert s["max_depth"] <= 4
    # A binary tree with L leaves has exactly 2L - 1 nodes.
    assert s["n_nodes"] == 2 * s["n_leaves"] - 1


def test_regression_tree_leaf_value_is_the_mean_of_its_members(reg_data):
    X, y = reg_data
    t = CART(task="regression", criterion="mse", max_depth=3).fit(X, y)
    groups: dict[int, list[float]] = {}
    for x, target in zip(X, y):
        groups.setdefault(id(t._leaf_for(x)), []).append(target)
    for x in X[:50]:
        leaf = t._leaf_for(x)
        assert leaf.value[0] == pytest.approx(np.mean(groups[id(leaf)]))
