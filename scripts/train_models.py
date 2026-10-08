"""Stage 3: train and evaluate every regression and classification model.

The work is split into STAGES that can each run in a fresh process:

    regression      linear / ridge / bayesian / huber / lightgbm  + baselines
    polynomial      Unit 1 degree study (1..4)
    quantile        LightGBM quantile models for safety stock
    classification  logistic / LDA / GaussianNB / SVM linear / SVM RBF / LGBM
    predictions     reload saved models, write the test prediction table

Why stages: this machine runs close to its commit limit, and holding every
fitted model plus several design matrices in one interpreter was enough to
trigger allocation failures. One stage per process keeps peak memory low and
makes a failure recoverable without redoing the earlier stages.

    .venv\\Scripts\\python scripts\\train_models.py              # all stages
    .venv\\Scripts\\python scripts\\train_models.py --stage regression
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP thread pools BEFORE numpy is imported. Each OpenBLAS thread
# reserves a private buffer; the default (one per core) exhausts the commit
# limit on this machine and numpy aborts with
# "Memory allocation still failed after 10 retries".
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xaierp.features.builder import (  # noqa: E402
    classification_features,
    load_features,
    regression_features,
    split_frames,
    xy,
)
from xaierp.models.classification import build_classification_models, decision_scores  # noqa: E402
from xaierp.models.regression import build_polynomial_models, build_regression_models  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.metrics import (  # noqa: E402
    classification_metrics,
    pinball_loss,
    regression_metrics,
)
from xaierp.utils.seeds import set_global_seed  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STAGES = ["regression", "polynomial", "quantile", "classification", "predictions"]


def _save(obj, name: str) -> Path:
    path = resolve(load_config()["paths"]["models"]) / f"{name}.joblib"
    ensure_dir(path)
    joblib.dump(obj, path, compress=3)
    return path


def _load_model(name: str):
    path = resolve(load_config()["paths"]["models"]) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the earlier training stage.")
    return joblib.load(path)


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = resolve(load_config()["paths"]["reports"]) / name
    ensure_dir(path)
    df.to_csv(path, index=False)
    return path


def _read_csv(name: str) -> pd.DataFrame:
    path = resolve(load_config()["paths"]["reports"]) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the earlier training stage.")
    return pd.read_csv(path)


# ======================================================================
# Stage: regression
# ======================================================================


def tune_ridge_alpha(Xtr, ytr, Xva, yva) -> tuple[float, pd.DataFrame]:
    """Select the L2 strength on VALIDATION. The test split is never consulted."""
    rows = []
    for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
        pipe.fit(Xtr, ytr)
        rows.append({"alpha": a, **regression_metrics(yva, pipe.predict(Xva))})
        del pipe
        gc.collect()
    tbl = pd.DataFrame(rows)
    return float(tbl.loc[tbl["RMSE"].idxmin(), "alpha"]), tbl


def stage_regression(seed: int) -> None:
    cfg = load_config()
    target = cfg["target"]["regression_target"]
    horizon = cfg["target"]["horizon_days"]

    df = load_features()
    frames = split_frames(df)
    feats = regression_features(df)
    tr, va, te = frames["train"], frames["val"], frames["test"]

    Xtr, ytr = xy(tr, feats, target)
    Xva, yva = xy(va, feats, target)
    Xte, yte = xy(te, feats, target)
    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  (float32)")

    print("  tuning Ridge alpha on validation...")
    best_alpha, alpha_tbl = tune_ridge_alpha(Xtr, ytr, Xva, yva)
    _write_csv(alpha_tbl, "ridge_alpha_search.csv")
    print(f"    best alpha = {best_alpha}")

    rows = []

    # --- naive baselines -------------------------------------------------
    # The bar a learned model has to clear. Without these the comparison
    # table cannot say whether machine learning added anything at all.
    for key, label, col in [
        ("baseline_last7_x_h", "Baseline: 7-day mean x horizon", "rolling_mean_7"),
        ("baseline_last28_x_h", "Baseline: 28-day mean x horizon", "rolling_mean_28"),
    ]:
        pv = va[col].to_numpy() * horizon
        pt = te[col].to_numpy() * horizon
        rows.append(
            {
                "model": label, "key": key,
                "unit": "Baseline (no learning)",
                "concept": "Moving-average rule of thumb",
                "train_time_s": 0.0,
                **{f"val_{k}": v for k, v in regression_metrics(yva, pv).items()},
                **{f"test_{k}": v for k, v in regression_metrics(yte, pt).items()},
            }
        )

    specs = build_regression_models(seed=seed)
    specs["ridge_regression"].build = lambda a=best_alpha: Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=a))]
    )

    for key, spec in specs.items():
        print(f"  fitting {spec.label} ...", end="", flush=True)
        t0 = time.time()
        model = spec.build()
        if key == "lightgbm":
            # Early stopping on the validation split. Without it the boosted
            # model runs all 600 rounds and overfits the training period
            # badly -- see docs/model_selection.md.
            import lightgbm as lgb

            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(Xtr, ytr)
        dt = time.time() - t0

        vm = regression_metrics(yva, model.predict(Xva))
        tm = regression_metrics(yte, model.predict(Xte))
        extra = ""
        if key == "lightgbm" and getattr(model, "best_iteration_", None):
            extra = f"  (stopped at {model.best_iteration_} trees)"
        print(f" {dt:6.1f}s   test RMSE={tm['RMSE']:.2f}  R2={tm['R2']:.4f}{extra}")

        rows.append(
            {
                "model": spec.label, "key": key,
                "unit": spec.unit, "concept": spec.concept,
                "train_time_s": round(dt, 3),
                **{f"val_{k}": v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            }
        )
        _save(model, f"reg_{key}")
        del model
        gc.collect()

    tbl = pd.DataFrame(rows).sort_values("test_RMSE").reset_index(drop=True)
    _write_csv(tbl, "regression_model_comparison.csv")
    _write_csv(pd.DataFrame({"feature": feats}), "regression_feature_list.csv")
    print(f"  best by test RMSE: {tbl.loc[0, 'model']} ({tbl.loc[0, 'test_RMSE']:.2f})")


# ======================================================================
# Stage: polynomial degree study (Unit 1)
# ======================================================================


def stage_polynomial(seed: int) -> None:
    cfg = load_config()
    subset = cfg["polynomial"]["feature_subset"]
    degrees = cfg["polynomial"]["degrees"]
    target = cfg["target"]["regression_target"]

    df = load_features()
    frames = split_frames(df)
    tr, va, te = frames["train"], frames["val"], frames["test"]
    Xtr, ytr = xy(tr, subset, target)
    Xva, yva = xy(va, subset, target)
    Xte, yte = xy(te, subset, target)

    print(f"  feature subset: {subset}")
    rows = []
    for key, spec in build_polynomial_models(degrees).items():
        d = int(key.rsplit("_", 1)[1])
        t0 = time.time()
        model = spec.build()
        model.fit(Xtr, ytr)
        dt = time.time() - t0
        n_terms = int(model.named_steps["poly"].n_output_features_)
        trm = regression_metrics(ytr, model.predict(Xtr))
        vm = regression_metrics(yva, model.predict(Xva))
        tm = regression_metrics(yte, model.predict(Xte))
        print(
            f"    degree {d}: {n_terms:4d} terms   "
            f"train RMSE={trm['RMSE']:7.2f}   val RMSE={vm['RMSE']:7.2f}   "
            f"test RMSE={tm['RMSE']:8.2f}   ({dt:.1f}s)"
        )
        rows.append(
            {
                "degree": d, "n_polynomial_terms": n_terms,
                "train_time_s": round(dt, 3),
                "train_RMSE": trm["RMSE"], "train_R2": trm["R2"],
                "val_RMSE": vm["RMSE"], "val_R2": vm["R2"],
                "test_RMSE": tm["RMSE"], "test_R2": tm["R2"],
                "gap_val_minus_train_RMSE": vm["RMSE"] - trm["RMSE"],
                "gap_test_minus_train_RMSE": tm["RMSE"] - trm["RMSE"],
            }
        )
        _save(model, f"reg_poly_degree_{d}")
        del model
        gc.collect()
    _write_csv(pd.DataFrame(rows), "polynomial_degree_study.csv")


# ======================================================================
# Stage: quantile regression
# ======================================================================


def stage_quantile(seed: int) -> None:
    import lightgbm as lgb

    cfg = load_config()
    levels = cfg["quantiles"]["levels"]
    target = cfg["target"]["regression_target"]

    df = load_features()
    frames = split_frames(df)
    feats = regression_features(df)
    Xtr, ytr = xy(frames["train"], feats, target)
    Xva, yva = xy(frames["val"], feats, target)
    Xte, yte = xy(frames["test"], feats, target)

    rows = []
    for q in levels:
        t0 = time.time()
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        dt = time.time() - t0
        pt = m.predict(Xte)
        # Coverage = share of actuals at or below the predicted quantile.
        # A calibrated q=0.90 model should cover about 90%.
        cov = float((yte <= pt).mean())
        rows.append(
            {
                "quantile": q,
                "pinball_loss_test": pinball_loss(yte, pt, q),
                "empirical_coverage_test": cov,
                "coverage_error": cov - q,
                "mean_prediction": float(pt.mean()),
                "train_time_s": round(dt, 3),
            }
        )
        print(f"    q={q:.2f}: coverage={cov:.3f} (target {q:.2f})  "
              f"pinball={rows[-1]['pinball_loss_test']:.3f}  ({dt:.1f}s)")
        _save(m, f"reg_quantile_q{int(q * 100)}")
        del m
        gc.collect()
    _write_csv(pd.DataFrame(rows), "quantile_model_metrics.csv")


# ======================================================================
# Stage: classification
# ======================================================================


def stage_classification(seed: int) -> None:
    target = "reorder_flag"
    df = load_features()
    frames = split_frames(df)
    feats = classification_features(df)
    tr, va, te = frames["train"], frames["val"], frames["test"]

    Xtr, ytr = xy(tr, feats, target)
    Xva, yva = xy(va, feats, target)
    Xte, yte = xy(te, feats, target)
    ytr, yva, yte = ytr.astype(np.int8), yva.astype(np.int8), yte.astype(np.int8)

    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  (float32)")
    print(f"  positive rate: train {ytr.mean():.2%}  val {yva.mean():.2%}  "
          f"test {yte.mean():.2%}")

    rng = np.random.default_rng(seed)
    rows = []

    for key, spec in build_classification_models(seed=seed).items():
        Xf, yf, n_used = Xtr, ytr, len(ytr)
        if spec.max_train_rows and len(ytr) > spec.max_train_rows:
            idx = np.sort(rng.choice(len(ytr), spec.max_train_rows, replace=False))
            Xf, yf, n_used = Xtr[idx], ytr[idx], spec.max_train_rows

        print(f"  fitting {spec.label} (n={n_used:,}) ...", end="", flush=True)
        t0 = time.time()
        model = spec.build()
        model.fit(Xf, yf)
        dt = time.time() - t0

        vm = classification_metrics(yva, model.predict(Xva), decision_scores(model, Xva))
        tm = classification_metrics(yte, model.predict(Xte), decision_scores(model, Xte))
        print(f" {dt:7.1f}s   test F1={tm['f1']:.4f}  recall={tm['recall']:.4f}  "
              f"AUC={tm['roc_auc']:.4f}")

        rows.append(
            {
                "model": spec.label, "key": key,
                "unit": spec.unit, "concept": spec.concept,
                "n_train_rows_used": int(n_used),
                "train_time_s": round(dt, 3),
                **{f"val_{k}": v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            }
        )
        _save(model, f"clf_{key}")
        del model, Xf, yf
        gc.collect()

    tbl = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    _write_csv(tbl, "classification_model_comparison.csv")
    _write_csv(pd.DataFrame({"feature": feats}), "classification_feature_list.csv")
    print(f"  best by test F1: {tbl.loc[0, 'model']} ({tbl.loc[0, 'test_f1']:.4f})")


# ======================================================================
# Stage: predictions (reloads models from disk)
# ======================================================================


def stage_predictions(seed: int) -> None:
    cfg = load_config()
    target = cfg["target"]["regression_target"]

    df = load_features()
    te = split_frames(df)["test"]
    reg_feats = regression_features(df)
    clf_feats = classification_features(df)

    reg_tbl = _read_csv("regression_model_comparison.csv")
    clf_tbl = _read_csv("classification_model_comparison.csv")
    best_reg = reg_tbl.loc[0, "key"]
    best_clf = clf_tbl.loc[0, "key"]

    out = te[
        ["stock_code", "date", "demand", target, "reorder_flag",
         "current_stock", "inventory_position", "on_order",
         "lead_time_days", "moq", "pack_size",
         "unit_cost", "supplier_id", "rolling_mean_7", "rolling_mean_28",
         "rolling_std_28", "description"]
    ].copy()

    Xte_reg = te[reg_feats].to_numpy(dtype=np.float32)
    for key in reg_tbl["key"]:
        if str(key).startswith("baseline"):
            continue
        m = _load_model(f"reg_{key}")
        out[f"pred_{key}"] = m.predict(Xte_reg)
        del m
        gc.collect()
    out["pred_best_regression"] = out[f"pred_{best_reg}"]

    for q in cfg["quantiles"]["levels"]:
        m = _load_model(f"reg_quantile_q{int(q * 100)}")
        out[f"pred_q{int(q * 100)}"] = m.predict(Xte_reg)
        del m
        gc.collect()
    del Xte_reg
    gc.collect()

    Xte_clf = te[clf_feats].to_numpy(dtype=np.float32)
    for key in clf_tbl["key"]:
        m = _load_model(f"clf_{key}")
        out[f"score_{key}"] = decision_scores(m, Xte_clf)
        if key == best_clf:
            out["pred_reorder_best"] = m.predict(Xte_clf)
        del m
        gc.collect()
    out["prob_best_classifier"] = out[f"score_{best_clf}"]
    del Xte_clf
    gc.collect()

    path = resolve(cfg["paths"]["forecasts"]) / "test_predictions.parquet"
    ensure_dir(path)
    out.to_parquet(path, index=False)

    summary = {
        "best_regression_model": str(reg_tbl.loc[0, "model"]),
        "best_regression_key": str(best_reg),
        "best_regression_test_RMSE": float(reg_tbl.loc[0, "test_RMSE"]),
        "best_regression_test_MAE": float(reg_tbl.loc[0, "test_MAE"]),
        "best_regression_test_R2": float(reg_tbl.loc[0, "test_R2"]),
        "best_classifier_model": str(clf_tbl.loc[0, "model"]),
        "best_classifier_key": str(best_clf),
        "best_classifier_test_f1": float(clf_tbl.loc[0, "test_f1"]),
        "best_classifier_test_recall": float(clf_tbl.loc[0, "test_recall"]),
        "best_classifier_test_precision": float(clf_tbl.loc[0, "test_precision"]),
        "best_classifier_test_roc_auc": float(clf_tbl.loc[0, "test_roc_auc"]),
        "n_regression_features": len(reg_feats),
        "n_classification_features": len(clf_feats),
        "test_rows": int(len(te)),
        "test_date_min": str(te["date"].min().date()),
        "test_date_max": str(te["date"].max().date()),
    }
    spath = resolve(cfg["paths"]["reports"]) / "model_summary.json"
    spath.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  best regression : {summary['best_regression_model']} "
          f"(test RMSE {summary['best_regression_test_RMSE']:.2f}, "
          f"R2 {summary['best_regression_test_R2']:.4f})")
    print(f"  best classifier : {summary['best_classifier_model']} "
          f"(test F1 {summary['best_classifier_test_f1']:.4f}, "
          f"AUC {summary['best_classifier_test_roc_auc']:.4f})")
    print(f"  saved -> {path.relative_to(ROOT)}")


# ======================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=STAGES, help="run a single stage")
    args = ap.parse_args()

    seed = load_config()["project"]["random_seed"]
    set_global_seed(seed)

    runners = {
        "regression": stage_regression,
        "polynomial": stage_polynomial,
        "quantile": stage_quantile,
        "classification": stage_classification,
        "predictions": stage_predictions,
    }
    todo = [args.stage] if args.stage else STAGES

    for name in todo:
        print("=" * 70)
        print(name.upper())
        print("=" * 70)
        runners[name](seed)
        gc.collect()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
