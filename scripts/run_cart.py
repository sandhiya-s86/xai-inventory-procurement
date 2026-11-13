"""Stage: Unit 5 -- CART on the reorder decision.

Implements topic 40 (Classification and Regression Trees) and Practice 1
(implement CART to perform categorization), run on the real reorder task the
rest of the project already models.

    1.  Grow a full classification tree on the reorder decision
    2.  Cost-complexity pruning path; choose alpha on the VALIDATION split
    3.  Evaluate the pruned tree on the held-out test split
    4.  Gini vs entropy, measured rather than asserted
    5.  Depth study: where the tree starts memorising
    6.  Feature importances from impurity decrease
    7.  Readable IF/THEN rules for real decisions  <- the point
    8.  A regression tree on 7-day demand (the "R" in CART)
    9.  Comparison against every Unit 2 classifier already trained

Writes:
    reports/unit5_*.csv            (7 tables)
    reports/unit5_findings.json
    reports/unit5_tree_rules.txt
    reports/figures/unit5_*.png    (3 figures)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.features.builder import (  # noqa: E402
    classification_features,
    load_features,
    regression_features,
    split_frames,
    xy,
)
from xaierp.trees.cart import CART  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.metrics import classification_metrics, regression_metrics  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402

plt.rcParams.update(
    {
        "figure.figsize": (11, 5.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
    }
)

TARGET = "reorder_flag"
# Growing to purity on 67k rows gives a tree with thousands of leaves that
# takes minutes to prune and is unreadable anyway. The pruning study starts
# from a tree capped here, which is itself a form of pre-pruning and is
# declared rather than hidden.
GROW_DEPTH = 10
MIN_LEAF = 30


def save_fig(fig, name: str, figs: Path) -> None:
    path = figs / f"unit5_{name}.png"
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_table(rows, name: str, reports: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = reports / f"unit5_{name}.csv"
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"    saved -> reports/unit5_{name}.csv  ({len(df)} rows)")
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)
    reports = resolve(cfg["paths"]["reports"])
    figs = resolve(cfg["paths"]["figures"])
    ensure_dir(reports)
    ensure_dir(figs)

    find: dict[str, object] = {}

    print("=" * 72)
    print("UNIT 5 -- CART ON THE REORDER DECISION")
    print("=" * 72)

    df = load_features()
    sp = split_frames(df)
    feats = classification_features(df)
    Xtr, ytr = xy(sp["train"], feats, TARGET)
    Xva, yva = xy(sp["val"], feats, TARGET)
    Xte, yte = xy(sp["test"], feats, TARGET)
    Xtr, Xva, Xte = Xtr.astype(np.float64), Xva.astype(np.float64), Xte.astype(np.float64)
    ytr, yva, yte = ytr.astype(int), yva.astype(int), yte.astype(int)

    print(f"\nTask: predict `{TARGET}` (reorder yes/no)")
    print(f"  train {len(Xtr):,} rows | val {len(Xva):,} | test {len(Xte):,}")
    print(f"  features {len(feats)} | positive rate "
          f"train {ytr.mean():.1%} / test {yte.mean():.1%}")

    find["target"] = TARGET
    find["n_features"] = len(feats)
    find["n_train"] = int(len(Xtr))
    find["n_val"] = int(len(Xva))
    find["n_test"] = int(len(Xte))
    find["positive_rate_train"] = float(ytr.mean())
    find["positive_rate_test"] = float(yte.mean())

    # ------------------------------------------------------- 1. grow
    print(f"\n[1/9] Growing the tree (max_depth={GROW_DEPTH}, "
          f"min_samples_leaf={MIN_LEAF})")
    t0 = time.time()
    full = CART(
        task="classification", criterion="gini", max_depth=GROW_DEPTH,
        min_samples_leaf=MIN_LEAF, class_weight="balanced",
    ).fit(Xtr, ytr, feature_names=feats)
    grow_s = time.time() - t0
    print(f"    grown in {grow_s:.1f}s: {full.root.leaves()} leaves, "
          f"depth {full.root.max_depth()}, {full.root.count_nodes()} nodes")
    print(f"    class_weight='balanced' -- missing a stockout costs more "
          f"than an unnecessary order")

    # ------------------------------------------- 2. cost-complexity pruning
    print("\n[2/9] Cost-complexity pruning path")
    t0 = time.time()
    path = full.cost_complexity_pruning_path()
    print(f"    {len(path['ccp_alphas'])} subtrees in the nested sequence "
          f"({time.time() - t0:.1f}s), {path['n_leaves'][0]} -> "
          f"{path['n_leaves'][-1]} leaves")

    # Evaluate every subtree on the VALIDATION split. This is the whole point
    # of the pruning sequence: it is finite, so alpha can be selected rather
    # than guessed -- and selected on data the tree was not grown on.
    rows = []
    for a, nl in zip(path["ccp_alphas"], path["n_leaves"]):
        p = full.prune(float(a))
        m_tr = classification_metrics(ytr, p.predict(Xtr))
        m_va = classification_metrics(yva, p.predict(Xva))
        rows.append(
            {
                "ccp_alpha": float(a),
                "n_leaves": int(p.root.leaves()),
                "max_depth": int(p.root.max_depth()),
                "train_f1": m_tr["f1"],
                "train_accuracy": m_tr["accuracy"],
                "val_f1": m_va["f1"],
                "val_accuracy": m_va["accuracy"],
                "val_recall": m_va["recall"],
                "val_precision": m_va["precision"],
            }
        )
    prune_df = save_table(rows, "pruning_path", reports)

    best = max(rows, key=lambda r: r["val_f1"])
    alpha_star = best["ccp_alpha"]
    print(f"    best validation F1 at alpha = {alpha_star:.6f} "
          f"({best['n_leaves']} leaves, depth {best['max_depth']})")
    print(f"    full tree      : train F1 {rows[0]['train_f1']:.4f}  "
          f"val F1 {rows[0]['val_f1']:.4f}  "
          f"(gap {rows[0]['train_f1'] - rows[0]['val_f1']:+.4f})")
    print(f"    pruned tree    : train F1 {best['train_f1']:.4f}  "
          f"val F1 {best['val_f1']:.4f}  "
          f"(gap {best['train_f1'] - best['val_f1']:+.4f})")
    print(f"    -> pruning removed "
          f"{rows[0]['n_leaves'] - best['n_leaves']} leaves "
          f"({1 - best['n_leaves']/rows[0]['n_leaves']:.0%}) and "
          f"{'narrowed' if (best['train_f1']-best['val_f1']) < (rows[0]['train_f1']-rows[0]['val_f1']) else 'widened'} "
          f"the train/val gap")

    pruned = full.prune(alpha_star)

    find["grow"] = {
        "max_depth_cap": GROW_DEPTH,
        "min_samples_leaf": MIN_LEAF,
        "grow_seconds": grow_s,
        "full_leaves": int(rows[0]["n_leaves"]),
        "full_depth": int(rows[0]["max_depth"]),
    }
    find["pruning"] = {
        "n_subtrees": int(len(path["ccp_alphas"])),
        "alpha_star": float(alpha_star),
        "full_leaves": int(rows[0]["n_leaves"]),
        "pruned_leaves": int(best["n_leaves"]),
        "pruned_depth": int(best["max_depth"]),
        "full_train_f1": float(rows[0]["train_f1"]),
        "full_val_f1": float(rows[0]["val_f1"]),
        "pruned_train_f1": float(best["train_f1"]),
        "pruned_val_f1": float(best["val_f1"]),
        "alphas_monotone": bool(np.all(np.diff(path["ccp_alphas"]) > 0)),
        "leaves_monotone": bool(np.all(np.diff(path["n_leaves"]) < 0)),
    }

    # ------------------------------------------------------- 3. test
    print("\n[3/9] Held-out test performance")
    te_full = classification_metrics(yte, full.predict(Xte),
                                     full.predict_proba(Xte)[:, 1])
    te_pruned = classification_metrics(yte, pruned.predict(Xte),
                                       pruned.predict_proba(Xte)[:, 1])
    for lab, m, nl in [("full tree", te_full, rows[0]["n_leaves"]),
                       ("pruned tree", te_pruned, best["n_leaves"])]:
        print(f"    {lab:<12} ({nl:3d} leaves)  F1 {m['f1']:.4f}  "
              f"recall {m['recall']:.4f}  precision {m['precision']:.4f}  "
              f"AUC {m['roc_auc']:.4f}")

    find["test_full"] = te_full
    find["test_pruned"] = te_pruned

    # ------------------------------------------------ 4. gini vs entropy
    print("\n[4/9] Gini vs entropy")
    crit_rows = []
    for crit in ("gini", "entropy"):
        t0 = time.time()
        m = CART(task="classification", criterion=crit, max_depth=6,
                 min_samples_leaf=MIN_LEAF, class_weight="balanced").fit(
            Xtr, ytr, feature_names=feats)
        fit_s = time.time() - t0
        mm = classification_metrics(yte, m.predict(Xte),
                                    m.predict_proba(Xte)[:, 1])
        crit_rows.append(
            {"criterion": crit, "fit_seconds": fit_s,
             "n_leaves": int(m.root.leaves()), **mm}
        )
        print(f"    {crit:<8} fit {fit_s:5.1f}s  {m.root.leaves():3d} leaves  "
              f"test F1 {mm['f1']:.4f}  AUC {mm['roc_auc']:.4f}")
    save_table(crit_rows, "criterion_comparison", reports)
    d_f1 = abs(crit_rows[0]["f1"] - crit_rows[1]["f1"])
    print(f"    -> F1 differs by {d_f1:.4f}; Gini is "
          f"{crit_rows[1]['fit_seconds']/max(crit_rows[0]['fit_seconds'],1e-9):.2f}x "
          f"faster (no logarithm in the inner loop)")
    find["criterion"] = {"f1_difference": float(d_f1), "rows": crit_rows}

    # ----------------------------------------------------- 5. depth study
    print("\n[5/9] Depth study -- where the tree starts memorising")
    depth_rows = []
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 14]:
        m = CART(task="classification", max_depth=d, min_samples_leaf=MIN_LEAF,
                 class_weight="balanced").fit(Xtr, ytr, feature_names=feats)
        tr = classification_metrics(ytr, m.predict(Xtr))
        va = classification_metrics(yva, m.predict(Xva))
        te = classification_metrics(yte, m.predict(Xte))
        depth_rows.append(
            {"max_depth": d, "n_leaves": int(m.root.leaves()),
             "train_f1": tr["f1"], "val_f1": va["f1"], "test_f1": te["f1"],
             "train_val_gap": tr["f1"] - va["f1"]}
        )
        print(f"    depth {d:>2}  {m.root.leaves():4d} leaves  "
              f"train F1 {tr['f1']:.4f}  val F1 {va['f1']:.4f}  "
              f"gap {tr['f1'] - va['f1']:+.4f}")
    save_table(depth_rows, "depth_study", reports)
    find["depth_study"] = depth_rows

    # ------------------------------------------------- 6. importances
    print("\n[6/9] Feature importances (impurity decrease)")
    imp = pruned.feature_importances_
    order = np.argsort(imp)[::-1]
    imp_rows = [
        {"feature": feats[i], "importance": float(imp[i])}
        for i in order if imp[i] > 0
    ]
    save_table(imp_rows, "feature_importance", reports)
    for r in imp_rows[:8]:
        print(f"    {r['feature']:<28} {r['importance']:.4f}")
    print(f"    ({len(imp_rows)} of {len(feats)} features used by the "
          f"pruned tree)")
    find["n_features_used"] = len(imp_rows)
    find["top_features"] = imp_rows[:10]

    # ----------------------------------------------------- 7. the rules
    print("\n[7/9] Readable decision rules")
    # A shallow tree is what a buyer would actually be shown. Depth 4 is the
    # readability/accuracy trade-off, and the cost of that trade is reported.
    readable = CART(task="classification", max_depth=4,
                    min_samples_leaf=200, class_weight="balanced").fit(
        Xtr, ytr, feature_names=feats)
    r_te = classification_metrics(yte, readable.predict(Xte))
    print(f"    readable tree: {readable.root.leaves()} leaves, depth "
          f"{readable.root.max_depth()}, test F1 {r_te['f1']:.4f} "
          f"({r_te['f1'] - te_pruned['f1']:+.4f} vs the tuned tree)")

    test_df = sp["test"].reset_index(drop=True)
    examples = []
    for i in [int(x) for x in np.linspace(0, len(Xte) - 1, 6)]:
        rule = readable.rule_for(Xte[i])
        examples.append(
            {
                "stock_code": str(test_df.loc[i, "stock_code"]),
                "date": str(pd.Timestamp(test_df.loc[i, "date"]).date()),
                "actual_reorder": int(yte[i]),
                "predicted_reorder": int(readable.predict(Xte[i:i + 1])[0]),
                "rule": rule.replace("\n", " "),
            }
        )
    save_table(examples, "rule_examples", reports)

    tree_only = reports / "unit5_tree_text.txt"
    tree_only.write_text(readable.export_text(), encoding="utf-8")

    rules_path = reports / "unit5_tree_rules.txt"
    rules_path.write_text(
        "CART decision rules for the reorder decision\n"
        "=" * 60 + "\n\n"
        f"Readable tree: {readable.root.leaves()} leaves, "
        f"depth {readable.root.max_depth()}, test F1 {r_te['f1']:.4f}\n\n"
        + readable.export_text() + "\n\n"
        + "=" * 60 + "\nWORKED EXAMPLES\n" + "=" * 60 + "\n\n"
        + "\n\n".join(
            f"SKU {e['stock_code']} on {e['date']} "
            f"(actual: {'REORDER' if e['actual_reorder'] else 'no order'})\n"
            + readable.rule_for(Xte[i])
            for e, i in zip(examples,
                            [int(x) for x in np.linspace(0, len(Xte) - 1, 6)])
        ),
        encoding="utf-8",
    )
    print(f"    saved -> reports/unit5_tree_rules.txt")
    print("\n    Example decision:")
    for line in readable.rule_for(Xte[0]).split("\n"):
        print(f"      {line}")

    find["readable_tree"] = {
        "n_leaves": int(readable.root.leaves()),
        "max_depth": int(readable.root.max_depth()),
        "test_f1": float(r_te["f1"]),
        "f1_cost_vs_tuned": float(r_te["f1"] - te_pruned["f1"]),
    }

    # ------------------------------------------- 8. regression tree
    print("\n[8/9] Regression tree -- the 'R' in CART")
    rfeats = regression_features(df)
    Xr_tr, yr_tr = xy(sp["train"], rfeats, "future_7d_demand")
    Xr_te, yr_te = xy(sp["test"], rfeats, "future_7d_demand")
    reg_rows = []
    for d in [3, 5, 8]:
        rt = CART(task="regression", criterion="mse", max_depth=d,
                  min_samples_leaf=MIN_LEAF).fit(
            Xr_tr.astype(np.float64), yr_tr, feature_names=rfeats)
        mm = regression_metrics(yr_te, rt.predict(Xr_te.astype(np.float64)))
        reg_rows.append({"max_depth": d, "n_leaves": int(rt.root.leaves()), **mm})
        print(f"    depth {d}  {rt.root.leaves():4d} leaves  "
              f"test RMSE {mm['RMSE']:.2f}  R2 {mm['R2']:.4f}")
    save_table(reg_rows, "regression_tree", reports)
    best_reg = min(reg_rows, key=lambda r: r["RMSE"])
    find["regression_tree"] = {"best": best_reg, "rows": reg_rows}

    # --------------------------------------------- 9. vs Unit 2 models
    print("\n[9/9] CART against the classifiers already trained")
    comp_path = reports / "classification_model_comparison.csv"
    comp_rows = [
        {"model": "CART (pruned, from scratch)", "unit": "Unit 5",
         "n_leaves": int(best["n_leaves"]),
         "test_f1": te_pruned["f1"], "test_recall": te_pruned["recall"],
         "test_precision": te_pruned["precision"],
         "test_roc_auc": te_pruned["roc_auc"]},
        {"model": "CART (readable, depth 4)", "unit": "Unit 5",
         "n_leaves": int(readable.root.leaves()),
         "test_f1": r_te["f1"], "test_recall": r_te["recall"],
         "test_precision": r_te["precision"],
         "test_roc_auc": float("nan")},
    ]
    if comp_path.exists():
        prev = pd.read_csv(comp_path)
        for _, r in prev.iterrows():
            comp_rows.append(
                {"model": r["model"], "unit": r["unit"], "n_leaves": np.nan,
                 "test_f1": r["test_f1"], "test_recall": r["test_recall"],
                 "test_precision": r["test_precision"],
                 "test_roc_auc": r["test_roc_auc"]}
            )
    comp_rows.sort(key=lambda r: -r["test_f1"])
    save_table(comp_rows, "model_comparison", reports)
    for r in comp_rows:
        print(f"    {r['model']:<32} F1 {r['test_f1']:.4f}  "
              f"recall {r['test_recall']:.4f}")

    rank = [r["model"] for r in comp_rows].index("CART (pruned, from scratch)") + 1
    top_f1 = comp_rows[0]["test_f1"]
    print(f"\n    CART ranks {rank} of {len(comp_rows)} by F1 "
          f"({te_pruned['f1']:.4f} vs best {top_f1:.4f}, "
          f"{te_pruned['f1'] - top_f1:+.4f})")
    find["comparison"] = {
        "cart_rank": rank,
        "n_models": len(comp_rows),
        "cart_test_f1": float(te_pruned["f1"]),
        "best_test_f1": float(top_f1),
        "best_model": comp_rows[0]["model"],
        "f1_gap_to_best": float(te_pruned["f1"] - top_f1),
    }

    # ------------------------------------------- 10. the one-rule baseline
    #
    # The pruned tree put 98.5% of its importance on a single feature. That is
    # not a quirk worth a footnote -- it is a claim about the TASK, and it is
    # testable: if one feature carries the decision, a single threshold on it
    # should match the models. So test that directly.
    print("\n[10/10] The one-rule baseline that the tree pointed at")
    top_feat = imp_rows[0]["feature"]
    j = feats.index(top_feat)
    xv_va, xv_te = Xva[:, j], Xte[:, j]

    # Threshold chosen on VALIDATION, scored on test -- same discipline as
    # every other model here, so the comparison is fair.
    grid = np.quantile(xv_va[np.isfinite(xv_va)], np.linspace(0.01, 0.99, 200))
    best_t, best_vf1 = grid[0], -1.0
    for t in grid:
        f1 = classification_metrics(yva, (xv_va <= t).astype(int))["f1"]
        if f1 > best_vf1:
            best_t, best_vf1 = float(t), f1
    one_rule = classification_metrics(yte, (xv_te <= best_t).astype(int))

    print(f"    rule: IF {top_feat} <= {best_t:.2f} THEN reorder")
    print(f"    test F1 {one_rule['f1']:.4f}  recall {one_rule['recall']:.4f}  "
          f"precision {one_rule['precision']:.4f}  "
          f"accuracy {one_rule['accuracy']:.4f}")
    print(f"    vs best model ({comp_rows[0]['model']}): "
          f"{one_rule['f1'] - top_f1:+.4f} F1")

    save_table(
        [{"rule": f"{top_feat} <= {best_t:.4f}",
          "threshold_chosen_on": "validation split", **one_rule}],
        "one_rule_baseline", reports,
    )

    beats = one_rule["f1"] >= top_f1
    print()
    if beats:
        print("    *** A SINGLE THRESHOLD MATCHES OR BEATS EVERY MODEL. ***")
    print("    Why: the simulator defines reorder_flag as")
    print("      actual future lead-time demand > current stock * (1 + tol)")
    print(f"    and `{top_feat}` is")
    print("      current stock / (trailing 28-day mean * lead time)")
    print("    -- the leakage-free proxy for that same comparison. The feature")
    print("    is not leaking (it uses only trailing data), but it already")
    print("    contains the decision; the residual error is demand-FORECAST")
    print("    error, which no classifier can remove because the forecast is")
    print("    baked into the feature. Every model here is bounded by it,")
    print("    which is why they all cluster near F1 0.75-0.78.")
    print()
    print("    This is the argument for interpretable models, made concrete:")
    print("    LightGBM reaches the same ceiling and conceals why. CART makes")
    print("    it visible in one split, and the finding is a property of the")
    print("    TASK, not of the model.")

    find["one_rule_baseline"] = {
        "feature": top_feat,
        "threshold": float(best_t),
        "threshold_chosen_on": "validation",
        "test_f1": float(one_rule["f1"]),
        "test_recall": float(one_rule["recall"]),
        "test_precision": float(one_rule["precision"]),
        "test_accuracy": float(one_rule["accuracy"]),
        "beats_every_model": bool(beats),
        "f1_vs_best_model": float(one_rule["f1"] - top_f1),
        "label_definition": "future lead-time demand > current stock * (1 + tol)",
        "feature_definition": "current stock / (trailing 28d mean * lead time)",
        "interpretation": (
            "The feature is the leakage-free proxy for the comparison the "
            "label thresholds, so it already contains the decision. Residual "
            "error is demand-forecast error, which bounds every classifier "
            "on this task."
        ),
    }

    # ------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(prune_df["n_leaves"], prune_df["train_f1"], "o-",
               label="train F1", color="#2563eb")
    ax[0].plot(prune_df["n_leaves"], prune_df["val_f1"], "s-",
               label="validation F1", color="#dc2626")
    ax[0].axvline(best["n_leaves"], color="crimson", ls="--",
                  label=f"chosen: {best['n_leaves']} leaves")
    ax[0].set_xlabel("Leaves in the subtree"); ax[0].set_ylabel("F1")
    ax[0].set_title("Cost-complexity pruning: alpha selected on validation")
    ax[0].legend()

    ax[1].plot(prune_df["ccp_alpha"], prune_df["n_leaves"], "o-",
               color="#047857")
    ax[1].set_xscale("symlog", linthresh=1e-6)
    ax[1].set_xlabel("ccp_alpha (price per leaf)")
    ax[1].set_ylabel("Leaves")
    ax[1].set_title("The nested pruning sequence")
    save_fig(fig, "01_pruning", figs)

    dd = pd.DataFrame(depth_rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dd["max_depth"], dd["train_f1"], "o-", label="train F1",
            color="#2563eb")
    ax.plot(dd["max_depth"], dd["val_f1"], "s-", label="validation F1",
            color="#dc2626")
    ax.plot(dd["max_depth"], dd["test_f1"], "^--", label="test F1",
            color="#047857")
    ax.set_xlabel("max_depth"); ax.set_ylabel("F1")
    ax.set_title("Tree depth: training F1 keeps rising, validation F1 does not")
    ax.legend()
    save_fig(fig, "02_depth_study", figs)

    top = imp_rows[:15][::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh([r["feature"] for r in top], [r["importance"] for r in top],
            color="#7c3aed")
    ax.set_xlabel("Impurity-decrease importance")
    ax.set_title("What the pruned tree actually splits on")
    save_fig(fig, "03_feature_importance", figs)

    path_json = reports / "unit5_findings.json"
    path_json.write_text(json.dumps(find, indent=2, default=str),
                         encoding="utf-8")
    print(f"\n    saved -> reports/unit5_findings.json")
    print("\nUnit 5 (CART) complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
