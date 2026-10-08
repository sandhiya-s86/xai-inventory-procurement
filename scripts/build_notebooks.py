"""Generate and execute the ten PBL-1 notebooks.

The notebooks are deliberately THIN. All business logic lives in src/ and all
heavy computation happens in scripts/; a notebook loads the saved artefacts,
calls the same functions the pipeline calls, and displays the result. That
keeps the analysis and the shipped code from drifting apart, and it means a
notebook can be re-run in seconds rather than re-training everything.

    .venv\\Scripts\\python scripts\\build_notebooks.py
    .venv\\Scripts\\python scripts\\build_notebooks.py --no-execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

BOOT = """\
import sys, os
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

from xaierp.utils.config import load_config, resolve
CFG = load_config()
REPORTS = resolve(CFG["paths"]["reports"])
FIGS = resolve(CFG["paths"]["figures"])
print("project root:", ROOT)
"""

PROVENANCE = """\
> **Data provenance.** Sales, dates, prices, SKUs and countries are **real**
> (UCI Online Retail II, CC BY 4.0, 1,067,371 invoice lines).
> Stock levels, suppliers, lead times, MOQ, pack sizes and every cost are
> **SIMULATED** — the source dataset contains none of them. See `sim/` and
> `docs/pbl1_scope.md`.
"""


def nb(title: str, subtitle: str, cells: list) -> nbf.NotebookNode:
    out = new_notebook()
    out.cells = [
        new_markdown_cell(f"# {title}\n\n{subtitle}\n\n{PROVENANCE}"),
        new_code_cell(BOOT),
        *cells,
    ]
    out.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return out


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


# ======================================================================
NOTEBOOKS: dict[str, tuple[str, str, list]] = {}

# ---------------------------------------------------------------- 01
NOTEBOOKS["01_dataset_familiarization.ipynb"] = (
    "01 · Dataset Familiarization",
    "What is in this dataset, how big is it, and what state is it in? "
    "Covers rubric criterion 1 (Problem Understanding & Dataset Familiarization).",
    [
        md("## 1. The raw dataset at a glance"),
        code(
            "dq = json.loads((REPORTS / 'data_quality_report.json').read_text())\n"
            "raw = dq['raw']\n"
            "print('Source      :', dq['dataset']['name'])\n"
            "print('URL         :', dq['dataset']['source_url'])\n"
            "print('Licence     :', dq['dataset']['licence'])\n"
            "print()\n"
            "print(f\"Rows        : {raw['n_rows']:,}\")\n"
            "print(f\"Columns     : {raw['n_cols']}\")\n"
            "print(f\"Invoices    : {raw['n_invoices']:,}\")\n"
            "print(f\"Stock codes : {raw['n_stock_codes']:,}\")\n"
            "print(f\"Customers   : {raw['n_customers']:,}\")\n"
            "print(f\"Countries   : {raw['n_countries']}\")\n"
            "print(f\"Date range  : {raw['date_min']}  ->  {raw['date_max']}\")"
        ),
        md("## 2. Columns, data types and missing values"),
        code(
            "miss = pd.DataFrame(raw['missing_by_column']).T\n"
            "miss[['dtype', 'n_missing', 'pct_missing', 'n_unique']]"
        ),
        md(
            "`customer_id` is missing on a substantial share of rows. Those are "
            "genuine transactions without a registered customer, so they are kept: "
            "the forecasting target is units sold, which does not need a customer id."
        ),
        md("## 3. Numeric summary of the raw columns"),
        code("pd.DataFrame(raw['numeric_summary']).T"),
        md(
            "`quantity` is extremely right-skewed and contains negatives — the "
            "negatives are credit notes, handled explicitly in cleaning."
        ),
        md("## 4. The cleaning pipeline, step by step"),
        code(
            "steps = pd.DataFrame(dq['cleaning']['steps'])\n"
            "print(f\"rows in : {dq['cleaning']['rows_in']:,}\")\n"
            "print(f\"rows out: {dq['cleaning']['rows_out']:,} \"\n"
            "      f\"({100 - dq['cleaning']['pct_removed_total']:.2f}% retained)\")\n"
            "steps[['step', 'rows_before', 'rows_after', 'rows_removed', 'pct_removed']]"
        ),
        code(
            "for s in dq['cleaning']['steps']:\n"
            "    print(f\"{s['step']}\\n    {s['reason'][:300]}\\n\")"
        ),
        md("## 5. Trading-day structure — a real finding"),
        code(
            "wk = dq['clean']['invoices_by_weekday']\n"
            "s = pd.Series(wk)\n"
            "display(s.to_frame('invoices'))\n"
            "ax = s.plot.bar(figsize=(8, 4), color='#4c51bf')\n"
            "ax.set_title('Invoices by day of week (cleaned data)')\n"
            "ax.set_ylabel('Number of invoices'); ax.set_xlabel('Day of week')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('\\nSaturday invoices:', wk['Sat'], 'vs weekday average:',\n"
            "      int(np.mean([wk[d] for d in ['Mon','Tue','Wed','Thu','Fri']])))"
        ),
        md(
            "**This retailer does not trade on Saturdays.** Treating Saturdays as "
            "zero-demand days would invent roughly one fake zero per week per SKU "
            "and bias every forecast downward. The panel therefore uses only days "
            "on which the business actually transacted."
        ),
        md("## 6. The modelling panel"),
        code(
            "p = dq['panel']\n"
            "for k, v in p.items():\n"
            "    if k != 'demand_summary':\n"
            "        print(f'{k:<36} {v}')\n"
            "print()\n"
            "pd.Series(p['demand_summary']).to_frame('daily demand per SKU')"
        ),
        md("## 7. Summary table used in the report"),
        code("pd.read_csv(REPORTS / 'dataset_summary.csv')"),
    ],
)

# ---------------------------------------------------------------- 02
NOTEBOOKS["02_eda.ipynb"] = (
    "02 · Exploratory Data Analysis",
    "Sixteen figures and the numbers behind them. "
    "Covers rubric criterion 2 (EDA & Insights).",
    [
        md(
            "All figures are generated by `scripts/run_eda.py` and saved to "
            "`reports/figures/`. Every number quoted in `docs/eda_insights.md` "
            "comes from `reports/eda_findings.json`, loaded below."
        ),
        code(
            "find = json.loads((REPORTS / 'eda_findings.json').read_text())\n"
            "list(find.keys())"
        ),
        md("## Key statistics"),
        code(
            "for section in ['daily_demand', 'demand_distribution', 'seasonality',\n"
            "                'intermittency', 'concentration', 'outliers',\n"
            "                'volatility', 'target_quantiles', 'correlation']:\n"
            "    print(f'--- {section} ---')\n"
            "    for k, v in find[section].items():\n"
            "        print(f'    {k:<34} {v}')\n"
            "    print()"
        ),
        md("## The figures"),
        code(
            "from IPython.display import Image, display, Markdown\n"
            "for name in sorted(find['figures']):\n"
            "    display(Markdown(f'### {name}'))\n"
            "    display(Image(filename=str(ROOT / find['figures'][name])))"
        ),
        md("## Trading calendar"),
        code("pd.Series(find['trading_calendar']['mean_demand_by_weekday'],\n"
             "          name='mean units per SKU per day').to_frame()"),
        md(
            "The written interpretation of every figure — observation, "
            "interpretation and business implication — is in "
            "[`docs/eda_insights.md`](../docs/eda_insights.md)."
        ),
    ],
)

# ---------------------------------------------------------------- 03
NOTEBOOKS["03_feature_engineering.ipynb"] = (
    "03 · Feature Engineering and Leakage Control",
    "How features are built and why none of them can see the future.",
    [
        md(
            "### Timing convention\n\n"
            "A row keyed `(sku, t)` is a decision taken at the **start of trading "
            "day t**.\n\n"
            "* features — may use demand observed on `t-1` and earlier\n"
            "* target — `future_7d_demand` = demand on days `t .. t+6`\n\n"
            "So demand on day `t` is part of the **label**, never an input."
        ),
        code(
            "from xaierp.features.builder import (\n"
            "    FEATURE_GROUPS, classification_features, load_features,\n"
            "    regression_features)\n"
            "feats = load_features()\n"
            "print(f'{len(feats):,} rows x {feats.shape[1]} columns')\n"
            "print(f\"{feats['stock_code'].nunique()} SKUs, \"\n"
            "      f\"{feats['date'].min().date()} -> {feats['date'].max().date()}\")"
        ),
        md("## Feature groups"),
        code(
            "for g, cols in FEATURE_GROUPS.items():\n"
            "    present = [c for c in cols if c in feats.columns]\n"
            "    print(f'{g:<10} ({len(present):>2}): {present}')"
        ),
        code(
            "reg = regression_features(feats)\n"
            "clf = classification_features(feats)\n"
            "print(f'regression features    : {len(reg)}')\n"
            "print(f'classification features: {len(clf)}')\n"
            "print(f'inventory-only extras  : {sorted(set(clf) - set(reg))}')"
        ),
        md(
            "The forecaster deliberately does **not** see stock level. How much is "
            "on the shelf does not change how much customers want; including it "
            "would model the replenishment policy rather than demand."
        ),
        md("## Demonstrating that the lag really lags"),
        code(
            "from xaierp.features.lags import add_lag_features, add_rolling_features\n"
            "toy = pd.DataFrame({'stock_code': ['A'] * 12,\n"
            "                    'date': pd.date_range('2024-01-01', periods=12),\n"
            "                    'demand': np.arange(1, 13, dtype=float)})\n"
            "t = add_rolling_features(add_lag_features(toy, [1, 7]), [3])\n"
            "t[['date', 'demand', 'lag_1', 'lag_7', 'rolling_mean_3', 'rolling_max_3']]"
        ),
        md(
            "On a strictly increasing series every 'past' feature must be strictly "
            "below today's value — and `rolling_max_3` equals exactly yesterday's "
            "demand, proving the window ends at `t-1`. This is asserted in "
            "`tests/test_no_leakage.py`."
        ),
        md("## The target"),
        code(
            "from xaierp.features.builder import add_target\n"
            "tt = add_target(toy, horizon=7)\n"
            "tt[['date', 'demand', 'future_7d_demand']]"
        ),
        md("## Chronological split"),
        code(
            "g = feats.groupby('split')['date'].agg(['min', 'max', 'count'])\n"
            "display(g.loc[['train', 'val', 'test']])\n"
            "print('\\nNo date appears in two splits:',\n"
            "      g.loc['train','max'] < g.loc['val','min'] <= g.loc['val','max'] < g.loc['test','min'])"
        ),
        md(
            "A **random** split would place day `t+1` in train and day `t` in test "
            "for the same SKU. Consecutive days share nearly all of their "
            "rolling-window history, so that leaks the answer and inflates every "
            "score. Splitting on global date cut-offs prevents it."
        ),
        md("## Target distribution"),
        code(
            "y = feats['future_7d_demand']\n"
            "print(y.describe())\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
            "ax[0].hist(y[y <= y.quantile(0.99)], bins=60, color='#2b6cb0')\n"
            "ax[0].set_title('future_7d_demand (below P99)'); ax[0].set_xlabel('units / 7 days')\n"
            "ax[1].hist(np.log1p(y), bins=60, color='#2b6cb0')\n"
            "ax[1].set_title('log(1 + future_7d_demand)'); ax[1].set_xlabel('log units')\n"
            "plt.tight_layout(); plt.show()"
        ),
    ],
)

# ---------------------------------------------------------------- 04
NOTEBOOKS["04_regression_models.ipynb"] = (
    "04 · Regression Models — forecasting 7-day demand",
    "Unit 1: linear regression, polynomial curve fitting, least squares, "
    "overfitting. Unit 2: MLE/least-squares, ridge, Bayesian linear "
    "regression, robust regression.",
    [
        md("## Results"),
        code(
            "reg = pd.read_csv(REPORTS / 'regression_model_comparison.csv')\n"
            "reg[['model', 'unit', 'test_MAE', 'test_RMSE', 'test_R2',\n"
            "     'test_RMSLE', 'train_time_s']].round(4)"
        ),
        code(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'model_regression_rmse.png')))\n"
            "display(Image(filename=str(FIGS / 'model_regression_r2.png')))\n"
            "display(Image(filename=str(FIGS / 'model_predicted_vs_actual.png')))"
        ),
        md("## Syllabus mapping for each model"),
        code("reg[['model', 'unit', 'concept']]"),
        md(
            "### Least squares and maximum likelihood are the same estimator\n\n"
            "Assume `y = Xβ + ε` with `ε ~ N(0, σ²I)`. The log-likelihood is\n\n"
            "    log L(β) = −n/2·log(2πσ²) − (1/2σ²)·‖y − Xβ‖²\n\n"
            "Only the final term depends on β, so **maximising the likelihood is "
            "exactly minimising the sum of squared errors**. That is why the table "
            "has no separate 'MLE model': `LinearRegression` already is one."
        ),
        md("### Ridge: the MAP estimate under a Gaussian prior"),
        code(
            "alpha = pd.read_csv(REPORTS / 'ridge_alpha_search.csv')\n"
            "display(alpha.round(4))\n"
            "best = alpha.loc[alpha['RMSE'].idxmin(), 'alpha']\n"
            "print(f'Selected alpha = {best} (chosen on VALIDATION, not on test)')\n"
            "ax = alpha.plot(x='alpha', y='RMSE', logx=True, marker='o', figsize=(7,4))\n"
            "ax.set_ylabel('Validation RMSE'); ax.set_title('Ridge regularisation path')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "Putting a prior `β ~ N(0, τ²I)` on the weights and taking the MAP "
            "estimate gives the ridge objective `‖y − Xβ‖² + λ‖β‖²` with "
            "`λ = σ²/τ²`. Ridge is therefore the Bayesian counterpart of OLS."
        ),
        md("### Bayesian linear regression gives a distribution, not a point"),
        code(
            "import joblib\n"
            "from xaierp.features.builder import load_features, regression_features, split_frames, xy\n"
            "from xaierp.models.regression import predictive_std\n"
            "feats = load_features(); te = split_frames(feats)['test']\n"
            "rf = regression_features(feats)\n"
            "X, y = xy(te, rf, 'future_7d_demand')\n"
            "br = joblib.load(resolve(CFG['paths']['models']) / 'reg_bayesian_ridge.joblib')\n"
            "mu = br.predict(X); sd = predictive_std(br, X)\n"
            "print('Predictive std available:', sd is not None)\n"
            "print(f'mean predictive std = {sd.mean():.2f} units')\n"
            "k = np.argsort(-sd)[:5]\n"
            "pd.DataFrame({'prediction': mu[k].round(1), 'predictive_std': sd[k].round(1),\n"
            "              'actual': y[k].round(1)})"
        ),
        md("### Robust regression: why Huber is in the table"),
        code(
            "d = pd.read_parquet(resolve(CFG['paths']['processed']) / 'daily_demand_panel.parquet')['demand']\n"
            "print(f'max single SKU-day demand : {d.max():,.0f} units')\n"
            "print(f'99th percentile           : {d.quantile(0.99):,.0f} units')\n"
            "print(f'skewness                  : {d.skew():.1f}')\n"
            "print('\\nSquared error weights that 9,480-unit day ~ (9480/69)^2 ≈ '\n"
            "      f'{(d.max()/d.quantile(0.90))**2:,.0f}x a typical day.')"
        ),
        md("## What actually won, and the honest surprise"),
        code(
            "s = json.loads((REPORTS / 'model_summary.json').read_text())\n"
            "print('Best:', s['best_regression_model'])\n"
            "print(f\"  test RMSE {s['best_regression_test_RMSE']:.2f}\")\n"
            "print(f\"  test MAE  {s['best_regression_test_MAE']:.2f}\")\n"
            "print(f\"  test R2   {s['best_regression_test_R2']:.4f}\")\n"
            "print()\n"
            "lg = reg.loc[reg['key'] == 'lightgbm'].iloc[0]\n"
            "rd = reg.loc[reg['key'] == 'ridge_regression'].iloc[0]\n"
            "print(f\"LightGBM test RMSE {lg['test_RMSE']:.2f} vs Ridge {rd['test_RMSE']:.2f}\")\n"
            "print('The gradient-boosted model LOSES to a regularised linear model here.')"
        ),
        md(
            "That result is reported as found, not hidden. The reasoning is in "
            "[`docs/model_selection.md`](../docs/model_selection.md)."
        ),
    ],
)

# ---------------------------------------------------------------- 05
NOTEBOOKS["05_classification_models.ipynb"] = (
    "05 · Classification Models — reorder / stockout risk",
    "Unit 2: discriminant functions, probabilistic generative vs "
    "discriminative models, kernels, the kernel trick and SVMs.",
    [
        md("## Results"),
        code(
            "clf = pd.read_csv(REPORTS / 'classification_model_comparison.csv')\n"
            "clf[['model', 'unit', 'n_train_rows_used', 'test_accuracy',\n"
            "     'test_precision', 'test_recall', 'test_f1', 'test_roc_auc',\n"
            "     'train_time_s']].round(4)"
        ),
        code(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'model_classification_f1.png')))\n"
            "display(Image(filename=str(FIGS / 'model_roc_curves.png')))\n"
            "display(Image(filename=str(FIGS / 'model_confusion_matrix.png')))"
        ),
        md("## Syllabus mapping"),
        code("clf[['model', 'unit', 'concept']]"),
        md("## Generative vs discriminative — the clearest contrast in the table"),
        code(
            "pair = clf[clf['key'].isin(['gaussian_nb', 'logistic_regression'])]\n"
            "display(pair[['model', 'test_precision', 'test_recall', 'test_f1',\n"
            "              'test_roc_auc']].round(4))\n"
            "nb_ = clf.loc[clf['key'] == 'gaussian_nb'].iloc[0]\n"
            "lr = clf.loc[clf['key'] == 'logistic_regression'].iloc[0]\n"
            "print(f\"Naive Bayes recall {nb_['test_recall']:.3f} but F1 only {nb_['test_f1']:.3f}\")\n"
            "print(f\"Logistic   recall {lr['test_recall']:.3f} and F1 {lr['test_f1']:.3f}\")"
        ),
        md(
            "Gaussian Naive Bayes models `p(x|y)p(y)` and inverts it with **Bayes "
            "rule** (Unit 1). Its independence assumption is badly violated here — "
            "`rolling_mean_7` and `rolling_mean_28` are strongly correlated — so it "
            "becomes over-confident and flags almost everything as needing a "
            "reorder: very high recall, poor precision. Logistic regression models "
            "`p(y|x)` directly, makes no claim about how the features are "
            "distributed, and is far better calibrated."
        ),
        md("## The kernel trick: linear vs RBF SVM"),
        code(
            "k = clf[clf['key'].isin(['svm_linear', 'svm_rbf'])]\n"
            "k[['model', 'n_train_rows_used', 'test_precision', 'test_recall',\n"
            "   'test_f1', 'test_roc_auc', 'train_time_s']].round(4)"
        ),
        md(
            "Both SVMs use the same C, the same features and the same training "
            "subsample, so the only difference is the kernel.\n\n"
            "* **Linear kernel** `K(x,z) = xᵀz` — a hyperplane in the original space.\n"
            "* **RBF kernel** `K(x,z) = exp(−γ‖x−z‖²)` — a hyperplane in an "
            "infinite-dimensional space.\n\n"
            "The **kernel trick** is that the algorithm only ever needs inner "
            "products, so that infinite-dimensional map is never computed — only "
            "`K`. Note the result below: the extra flexibility did **not** help "
            "here, which is itself worth saying out loud."
        ),
        code(
            "lin = clf.loc[clf['key'] == 'svm_linear', 'test_f1'].iloc[0]\n"
            "rbf = clf.loc[clf['key'] == 'svm_rbf', 'test_f1'].iloc[0]\n"
            "print(f'Linear SVM F1 {lin:.4f} vs RBF SVM F1 {rbf:.4f} -> '\n"
            "      f'{\"RBF\" if rbf > lin else \"LINEAR\"} wins')\n"
            "print('The decision boundary is close to linear in these features.')"
        ),
        md("## Why F1 and recall, not accuracy"),
        code(
            "feats_rate = pd.read_parquet(\n"
            "    resolve(CFG['paths']['processed']) / 'feature_table.parquet')\n"
            "rate = feats_rate['reorder_flag'].mean()\n"
            "print(f'positive (reorder needed) rate: {rate:.2%}')\n"
            "print(f'a model that always says NO scores {1-rate:.2%} accuracy '\n"
            "      'while catching zero stockouts')\n"
            "print(f\"best model accuracy: {clf['test_accuracy'].max():.2%}\")"
        ),
        md(
            "A missed stockout (false negative) loses a sale and customer goodwill; "
            "a false alarm just ties up working capital. Recall is therefore "
            "weighted more heavily than precision, and accuracy alone is reported "
            "but never relied on."
        ),
    ],
)

# ---------------------------------------------------------------- 06
NOTEBOOKS["06_uncertainty_and_quantiles.ipynb"] = (
    "06 · Uncertainty, Probability and Quantiles",
    "Unit 1: mean, variance, probability densities, expectation, covariance, "
    "quantiles — connected to safety-stock sizing.",
    [
        md("## Mean, variance and the shape of demand"),
        code(
            "panel = pd.read_parquet(\n"
            "    resolve(CFG['paths']['processed']) / 'daily_demand_panel.parquet')\n"
            "d = panel['demand']\n"
            "stats = pd.Series({\n"
            "    'mean E[X]': d.mean(), 'variance Var(X)': d.var(),\n"
            "    'std sqrt(Var)': d.std(), 'median': d.median(),\n"
            "    'skewness': d.skew(), 'kurtosis': d.kurtosis(),\n"
            "    'min': d.min(), 'max': d.max()})\n"
            "display(stats.to_frame('daily demand per SKU').round(3))\n"
            "print('Var(X) = E[X^2] - (E[X])^2 check:',\n"
            "      round((d**2).mean() - d.mean()**2, 4), 'vs', round(d.var(ddof=0), 4))"
        ),
        md(
            "The identity `Var(X) = E[X²] − (E[X])²` from Unit 1 is verified "
            "numerically above. Mean far exceeds median and skewness is large — "
            "this is nothing like a Normal distribution, which matters later."
        ),
        md("## Empirical quantiles of 7-day demand"),
        code(
            "feats = pd.read_parquet(\n"
            "    resolve(CFG['paths']['processed']) / 'feature_table.parquet')\n"
            "y = feats['future_7d_demand']\n"
            "qs = [0.5, 0.75, 0.9, 0.95, 0.99]\n"
            "q = pd.DataFrame({'quantile': qs, 'units over 7 days': [y.quantile(v) for v in qs]})\n"
            "q['x P50'] = (q['units over 7 days'] / y.quantile(0.5)).round(2)\n"
            "display(q.round(1))\n"
            "print('Planning to P95 instead of P50 means holding',\n"
            "      f\"{y.quantile(0.95)/y.quantile(0.5):.1f}x the stock.\")"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ax.hist(y[y <= y.quantile(0.99)], bins=70, color='#2f855a', alpha=0.85)\n"
            "for v, c in zip([0.5, 0.75, 0.9, 0.95], ['k', '#2b6cb0', '#dd6b20', '#c53030']):\n"
            "    ax.axvline(y.quantile(v), ls='--', color=c, lw=1.8,\n"
            "               label=f'P{int(v*100)} = {y.quantile(v):,.0f}')\n"
            "ax.set_xlabel('7-day demand (units)'); ax.set_ylabel('SKU-weeks')\n"
            "ax.set_title('Empirical distribution of 7-day demand with planning quantiles')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md("## Covariance and correlation (Unit 1)"),
        code(
            "cols = ['demand', 'revenue', 'avg_price', 'n_invoices', 'returned_units']\n"
            "print('Covariance matrix:'); display(panel[cols].cov().round(2))\n"
            "print('Correlation matrix:'); display(panel[cols].corr().round(3))"
        ),
        md(
            "Covariance is scale-dependent (its units are units×currency), which is "
            "why correlation — covariance divided by the two standard deviations — "
            "is the comparable quantity."
        ),
        md("## Learned quantiles: does P90 really cover 90%?"),
        code(
            "qt = pd.read_csv(REPORTS / 'quantile_model_metrics.csv')\n"
            "display(qt.round(4))\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'model_quantile_calibration.png')))"
        ),
        md(
            "These are LightGBM models trained on the **pinball loss**, which is the "
            "proper scoring rule for a quantile. Coverage is within a few points of "
            "target, so the predicted upper quantile is a usable basis for safety "
            "stock — and unlike the classical `z·σ·√L` formula it makes no Normality "
            "assumption, which matters given the skew measured above."
        ),
        md("## From quantile to safety stock"),
        code(
            "from xaierp.policy.reorder import safety_stock_from_quantiles, safety_stock_from_sigma\n"
            "sigma, lead = float(panel['demand'].std()), 7\n"
            "rows = []\n"
            "for sl in [0.50, 0.75, 0.90, 0.95, 0.99]:\n"
            "    rows.append({'service level': sl,\n"
            "                 'classical z*sigma*sqrt(L)': round(safety_stock_from_sigma(sigma, lead, sl), 1)})\n"
            "display(pd.DataFrame(rows))\n"
            "preds = pd.read_parquet(resolve(CFG['paths']['forecasts']) / 'test_predictions.parquet')\n"
            "gap = (preds['pred_q95'].clip(lower=0) - preds['pred_q50'].clip(lower=0)).clip(lower=0)\n"
            "print(f'\\nLearned P95-P50 buffer: mean {gap.mean():.1f}, median {gap.median():.1f} units')"
        ),
    ],
)

# ---------------------------------------------------------------- 07
NOTEBOOKS["07_inventory_and_procurement.ipynb"] = (
    "07 · Inventory Simulation and Procurement Decisions",
    "The SIMULATED supply side, the reorder policy, and the P0→P4 policy ladder.",
    [
        md(
            "### Everything about supply in this notebook is simulated\n\n"
            "Online Retail II is a sales log. Stock, suppliers, lead times, MOQ and "
            "costs do not exist in it and are generated by `sim/` from "
            "`sim/supply_config.yaml` under a fixed seed."
        ),
        md("## The simulated supplier master"),
        code(
            "sup = pd.read_parquet(resolve(CFG['paths']['simulated']) / 'suppliers.parquet')\n"
            "print(f'{len(sup):,} rows, {sup[\"stock_code\"].nunique()} SKUs')\n"
            "display(sup.head(9))\n"
            "sup[['lead_time_days', 'moq', 'pack_size', 'unit_cost',\n"
            "     'capacity_units', 'reliability']].describe().round(2)"
        ),
        md("## Simulated inventory trajectory for one SKU"),
        code(
            "inv = pd.read_parquet(resolve(CFG['paths']['simulated']) / 'inventory_state.parquet')\n"
            "sku = inv.groupby('stock_code')['unmet_demand'].sum().idxmax()\n"
            "g = inv[inv['stock_code'] == sku].sort_values('date').tail(180)\n"
            "fig, ax = plt.subplots(figsize=(12, 5))\n"
            "ax.plot(g['date'], g['ending_stock'], label='Ending stock', color='#2b6cb0')\n"
            "ax.bar(g['date'], g['demand'], color='#a0aec0', label='Daily demand')\n"
            "ax.bar(g['date'], g['unmet_demand'], color='#c53030', label='Unmet demand')\n"
            "ax.set_title(f'SIMULATED inventory trajectory — SKU {sku} (worst stockouts)')\n"
            "ax.set_xlabel('Date'); ax.set_ylabel('Units'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## Conservation holds exactly"),
        code(
            "recon = inv['beginning_stock'] + inv['arrivals'] - inv['consumed']\n"
            "resid = (recon - inv['ending_stock']).abs()\n"
            "print(f'max |beginning + arrivals - consumed - ending| = {resid.max():.12f}')\n"
            "print('negative stock rows:', int((inv['ending_stock'] < -1e-9).sum()))\n"
            "print(f\"baseline fill rate: {1 - inv['unmet_demand'].sum()/inv['demand'].sum():.3%}\")"
        ),
        md("## The reorder calculation, component by component"),
        code(
            "from xaierp.policy.reorder import decide_reorder\n"
            "d = decide_reorder(expected_demand_leadtime=420, safety_stock=95,\n"
            "                   inventory_position=160, lead_time_days=8,\n"
            "                   moq=48, pack_size=12, supplier_capacity=5000,\n"
            "                   unit_cost=1.75, mean_daily_demand=30)\n"
            "for k, v in d.components().items():\n"
            "    print(f'{v:>+12,.2f}   {k}')\n"
            "print('-' * 46)\n"
            "print(f'{d.final_qty:>12,}   FINAL ORDER QUANTITY')\n"
            "print('\\nComponents reconcile:', d.reconciles())"
        ),
        md("## The policy ladder"),
        code(
            "pol = pd.read_csv(REPORTS / 'policy_comparison.csv')\n"
            "pol[['policy', 'policy_label', 'fill_rate', 'stockout_days', 'n_orders',\n"
            "     'mean_inventory_units', 'holding_cost', 'shortage_cost',\n"
            "     'total_cost', 'cost_vs_P0_pct', 'fill_rate_vs_P0_pp']].round(3)"
        ),
        code(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'policy_cost_vs_fill_rate.png')))"
        ),
        md("## Ablation — what each capability adds"),
        code("pd.read_csv(REPORTS / 'ablation_study.csv').round(3)"),
        md(
            "Read this honestly: a better **forecast alone** (P2) does not improve "
            "service — it is the explicit treatment of **uncertainty** (a safety "
            "buffer, P1/P3/P4) that does. P4 buys the highest fill rate at a higher "
            "total cost; whether that trade is worth taking depends on how the "
            "business prices a lost sale, which is exactly the judgement a decision "
            "support tool should surface rather than make."
        ),
        md("## Recommendations produced"),
        code(
            "rec = pd.read_parquet(resolve(CFG['paths']['results']) / 'procurement_recommendations.parquet')\n"
            "print(f'{len(rec):,} SKU-day decisions, {rec[\"rec_reorder\"].mean():.1%} recommend a reorder')\n"
            "comp = ['rec_expected_demand','rec_safety_stock','rec_inventory_offset',\n"
            "        'rec_clip_adjustment','rec_moq_adjustment','rec_pack_adjustment',\n"
            "        'rec_capacity_adjustment']\n"
            "print('max decomposition residual:',\n"
            "      float((rec[comp].sum(axis=1) - rec['rec_qty']).abs().max()))\n"
            "rec.loc[rec['rec_reorder'], ['stock_code','date','current_stock',\n"
            "    'rec_expected_demand','rec_safety_stock','rec_qty','moq','pack_size',\n"
            "    'rec_order_value']].head(12).round(1)"
        ),
    ],
)

# ---------------------------------------------------------------- 08
NOTEBOOKS["08_xai_explanations.ipynb"] = (
    "08 · Explainable AI",
    "SHAP attributions for the forecast, TreeSHAP for the reorder decision, "
    "and the decision-level decomposition that actually answers 'why this number?'.",
    [
        md("## Global importance — demand forecast"),
        code(
            "imp = pd.read_csv(REPORTS / 'shap_global_importance_forecast.csv')\n"
            "display(imp.head(15).round(4))\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'shap_global_importance_forecast.png')))"
        ),
        md(
            "The winning forecaster is Ridge — a linear model — so the Shapley "
            "values are computed in **closed form** "
            "(`φᵢ = βᵢ·(zᵢ − E[zᵢ])`) and are therefore exact, not sampled."
        ),
        md("## Global importance — reorder decision"),
        code(
            "impc = pd.read_csv(REPORTS / 'shap_global_importance_classifier.csv')\n"
            "display(impc.head(15).round(4))\n"
            "display(Image(filename=str(FIGS / 'shap_global_importance_classifier.png')))"
        ),
        md(
            "Note how different the two rankings are. Demand is driven by demand "
            "history; the *reorder decision* is driven mostly by "
            "`stock_vs_leadtime_demand`, `current_stock` and `lead_time_days` — "
            "i.e. by the inventory position, exactly as a buyer would expect."
        ),
        md("## Additivity — the property that makes SHAP trustworthy"),
        code(
            "import joblib\n"
            "from xaierp.explain.shap_forecast import forecast_shap\n"
            "from xaierp.features.builder import load_features, regression_features, split_frames\n"
            "s = json.loads((REPORTS / 'model_summary.json').read_text())\n"
            "feats = load_features(); te = split_frames(feats)['test'].reset_index(drop=True)\n"
            "rf = regression_features(feats)\n"
            "model = joblib.load(resolve(CFG['paths']['models']) / f\"reg_{s['best_regression_key']}.joblib\")\n"
            "X = te.iloc[:500][rf].to_numpy(dtype=np.float32)\n"
            "res = forecast_shap(model, X, rf)\n"
            "print(f'base value           : {res.base_value:.3f}')\n"
            "print(f'max |base + sum(phi) - prediction| = {res.verify_additivity():.2e}')\n"
            "print('\\nEvery prediction is exactly the base value plus its attributions.')"
        ),
        md("## Worked case studies"),
        code(
            "print((REPORTS / 'generated' / 'case_studies.txt').read_text()[:6000])"
        ),
        md("## Every case reconciles"),
        code(
            "cases = json.loads((REPORTS / 'generated' / 'case_studies.json').read_text())\n"
            "rows = [{'sku': c['sku'], 'risk': c['stockout_risk'],\n"
            "         'recommended_qty': c['recommended_qty'],\n"
            "         'components_sum': round(sum(c['decomposition'].values()), 2),\n"
            "         'reconciles': c['reconciles']} for c in cases]\n"
            "pd.DataFrame(rows)"
        ),
        md(
            "This is the part most explainability demos skip. Saying "
            "'`rolling_mean_28` is important' does not tell a buyer why the system "
            "asked for 1,380 units. The decomposition does, and "
            "`tests/test_decision_explanation.py` fails if it ever stops adding up."
        ),
        md("## Per-case SHAP waterfalls"),
        code(
            "for p in sorted(FIGS.glob('shap_waterfall_case*.png')):\n"
            "    display(Image(filename=str(p)))"
        ),
    ],
)

# ---------------------------------------------------------------- 09
NOTEBOOKS["09_model_comparison.ipynb"] = (
    "09 · Comparative Performance Analysis",
    "All models side by side, with statistical confidence intervals. "
    "Covers rubric criterion 4.",
    [
        md("## Regression"),
        code(
            "reg = pd.read_csv(REPORTS / 'regression_model_comparison.csv')\n"
            "reg[['model','unit','test_MAE','test_RMSE','test_R2','test_RMSLE','train_time_s']].round(4)"
        ),
        md("## Classification"),
        code(
            "clf = pd.read_csv(REPORTS / 'classification_model_comparison.csv')\n"
            "clf[['model','unit','test_accuracy','test_precision','test_recall',\n"
            "     'test_f1','test_roc_auc','train_time_s']].round(4)"
        ),
        md("## Polynomial degree: underfitting vs overfitting"),
        code(
            "poly = pd.read_csv(REPORTS / 'polynomial_degree_study.csv')\n"
            "display(poly.round(3))\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGS / 'model_polynomial_overfitting.png')))"
        ),
        md(
            "Training RMSE falls monotonically with degree while test RMSE "
            "explodes — the canonical bias/variance picture from Unit 1, measured "
            "on real data rather than illustrated with a toy curve."
        ),
        md("## Bootstrap confidence intervals on the test metrics"),
        code(
            "preds = pd.read_parquet(resolve(CFG['paths']['forecasts']) / 'test_predictions.parquet')\n"
            "y = preds['future_7d_demand'].to_numpy()\n"
            "rng = np.random.default_rng(42); B = 500\n"
            "rows = []\n"
            "for key, label in [('ridge_regression','Ridge'), ('linear_regression','Linear'),\n"
            "                   ('lightgbm','LightGBM'), ('huber_regression','Huber')]:\n"
            "    col = f'pred_{key}'\n"
            "    if col not in preds.columns: continue\n"
            "    p = preds[col].to_numpy()\n"
            "    stats = []\n"
            "    for _ in range(B):\n"
            "        i = rng.integers(0, len(y), len(y))\n"
            "        stats.append(np.sqrt(np.mean((y[i] - p[i])**2)))\n"
            "    lo, hi = np.percentile(stats, [2.5, 97.5])\n"
            "    rows.append({'model': label, 'test_RMSE': np.sqrt(np.mean((y-p)**2)),\n"
            "                 'CI_lower_95': lo, 'CI_upper_95': hi})\n"
            "pd.DataFrame(rows).round(2)"
        ),
        md("## Paired bootstrap: is Ridge genuinely better than LightGBM?"),
        code(
            "a = preds['pred_ridge_regression'].to_numpy()\n"
            "b = preds['pred_lightgbm'].to_numpy()\n"
            "diffs = []\n"
            "for _ in range(1000):\n"
            "    i = rng.integers(0, len(y), len(y))\n"
            "    diffs.append(np.sqrt(np.mean((y[i]-a[i])**2)) - np.sqrt(np.mean((y[i]-b[i])**2)))\n"
            "diffs = np.array(diffs)\n"
            "lo, hi = np.percentile(diffs, [2.5, 97.5])\n"
            "print(f'RMSE(Ridge) - RMSE(LightGBM) = {diffs.mean():.2f}  95% CI [{lo:.2f}, {hi:.2f}]')\n"
            "print('Ridge is significantly better:' , hi < 0)"
        ),
        md(
            "The paired bootstrap resamples the *same* test rows for both models, "
            "so it controls for which weeks happened to be easy or hard. An "
            "interval entirely below zero means the difference is not noise."
        ),
        md("## Figures"),
        code(
            "for f in ['model_regression_rmse.png','model_regression_r2.png',\n"
            "          'model_classification_f1.png','model_roc_curves.png',\n"
            "          'model_confusion_matrix.png','model_predicted_vs_actual.png',\n"
            "          'model_quantile_calibration.png']:\n"
            "    display(Image(filename=str(FIGS / f)))"
        ),
    ],
)

# ---------------------------------------------------------------- 10
NOTEBOOKS["10_final_case_studies.ipynb"] = (
    "10 · End-to-end Case Studies",
    "The complete chain for individual products: data → forecast → risk → "
    "recommendation → explanation → what-if.",
    [
        md("## The six questions this project set out to answer"),
        code(
            "rec = pd.read_parquet(resolve(CFG['paths']['results']) / 'procurement_recommendations.parquet')\n"
            "s = json.loads((REPORTS / 'model_summary.json').read_text())\n"
            "print('Test period:', s['test_date_min'], '->', s['test_date_max'])\n"
            "print(f\"{len(rec):,} decisions across {rec['stock_code'].nunique()} SKUs\")"
        ),
        code(
            "from xaierp.explain.decision_explanation import explain_decision, render_explanation\n"
            "from xaierp.explain.whatif import run_whatif_table\n"
            "from xaierp.policy.reorder import decide_reorder, safety_stock_from_quantiles\n"
            "\n"
            "def full_story(row, review=7):\n"
            "    lead = int(row['lead_time_days'])\n"
            "    scale = (lead + review) / CFG['target']['horizon_days']\n"
            "    q50, q95 = max(0.0, row['pred_q50']), max(0.0, row['pred_q95'])\n"
            "    f7 = float(row['pred_best_regression'])\n"
            "    d = decide_reorder(\n"
            "        expected_demand_leadtime=max(0.0, f7) * scale,\n"
            "        safety_stock=safety_stock_from_quantiles(q95, q50, scale=scale),\n"
            "        inventory_position=float(row['inventory_position']),\n"
            "        lead_time_days=lead, moq=int(row['moq']),\n"
            "        pack_size=int(row['pack_size']),\n"
            "        supplier_capacity=int(row['supplier_capacity']),\n"
            "        unit_cost=float(row['unit_cost']),\n"
            "        mean_daily_demand=float(row['rolling_mean_28']))\n"
            "    exp = explain_decision(str(row['stock_code']), str(row['description']),\n"
            "                           row['date'], d, f7,\n"
            "                           float(row['prob_best_classifier']), None,\n"
            "                           float(row['rolling_mean_28']))\n"
            "    return d, exp"
        ),
        md("## Case A — a high-risk, stocked-out product"),
        code(
            "row = rec.loc[rec['prob_best_classifier'].idxmax()]\n"
            "d, exp = full_story(row)\n"
            "print(render_explanation(exp))\n"
            "print(f\"\\nActual demand over the next 7 days (hindsight): \"\n"
            "      f\"{row['future_7d_demand']:,.0f} units\")"
        ),
        md("## Case B — a well-stocked product that needs nothing"),
        code(
            "row = rec.loc[rec['prob_best_classifier'].idxmin()]\n"
            "d, exp = full_story(row)\n"
            "print(render_explanation(exp))\n"
            "print(f\"\\nActual demand over the next 7 days (hindsight): \"\n"
            "      f\"{row['future_7d_demand']:,.0f} units\")"
        ),
        md("## What-if on the high-risk case"),
        code(
            "row = rec.loc[rec['prob_best_classifier'].idxmax()]\n"
            "base = {'demand_7d': float(row['pred_best_regression']),\n"
            "        'q50': max(0.0, float(row['pred_q50'])),\n"
            "        'q95': max(0.0, float(row['pred_q95'])),\n"
            "        'current_stock': float(row['inventory_position']),\n"
            "        'lead_time_days': int(row['lead_time_days']),\n"
            "        'review_period': 7, 'horizon': CFG['target']['horizon_days'],\n"
            "        'moq': int(row['moq']), 'pack_size': int(row['pack_size']),\n"
            "        'supplier_capacity': int(row['supplier_capacity']),\n"
            "        'unit_cost': float(row['unit_cost']),\n"
            "        'mean_daily_demand': float(row['rolling_mean_28'])}\n"
            "pd.DataFrame(run_whatif_table(base))"
        ),
        md(
            "Each row re-runs the **real** policy function with one input changed — "
            "nothing is interpolated. This is what makes the system decision "
            "*support*: a buyer can test a negotiation ('what if I get the lead "
            "time down to 4 days?') before committing."
        ),
        md("## Aggregate view of the recommendations"),
        code(
            "latest = rec.sort_values('date').groupby('stock_code').tail(1)\n"
            "o = latest[latest['rec_reorder']]\n"
            "print(f'SKUs needing a reorder : {len(o)} of {len(latest)}')\n"
            "print(f'Total units            : {o[\"rec_qty\"].sum():,.0f}')\n"
            "print(f'Total value (SIMULATED): {o[\"rec_order_value\"].sum():,.0f}')\n"
            "o[['stock_code','description','current_stock','pred_best_regression',\n"
            "   'rec_qty','lead_time_days','rec_order_value']].sort_values(\n"
            "   'rec_order_value', ascending=False).head(15).round(1)"
        ),
        md("## Final scorecard"),
        code(
            "pol = pd.read_csv(REPORTS / 'policy_comparison.csv')\n"
            "p0 = pol[pol['policy']=='P0'].iloc[0]; p4 = pol[pol['policy']=='P4'].iloc[0]\n"
            "print(f\"Forecast  : {s['best_regression_model']}  \"\n"
            "      f\"RMSE {s['best_regression_test_RMSE']:.1f}  R2 {s['best_regression_test_R2']:.3f}\")\n"
            "print(f\"Classifier: {s['best_classifier_model']}  \"\n"
            "      f\"F1 {s['best_classifier_test_f1']:.3f}  AUC {s['best_classifier_test_roc_auc']:.3f}\")\n"
            "print(f\"Fill rate : {p0['fill_rate']:.1%} (P0 baseline) -> {p4['fill_rate']:.1%} (P4)\")\n"
            "print(f\"Cost      : {p4['cost_vs_P0_pct']:+.1f}% vs the P0 baseline\")"
        ),
    ],
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true")
    args = ap.parse_args()

    NB_DIR.mkdir(parents=True, exist_ok=True)
    built = []
    for fname, (title, subtitle, cells) in NOTEBOOKS.items():
        doc = nb(title, subtitle, cells)
        path = NB_DIR / fname
        nbf.write(doc, path)
        built.append(path)
        print(f"  wrote {path.relative_to(ROOT)}")

    if args.no_execute:
        print("\nSkipping execution (--no-execute).")
        return 0

    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    print("\nExecuting notebooks...")
    failures = []
    for path in built:
        doc = nbf.read(path, as_version=4)
        client = NotebookClient(
            doc, timeout=900, kernel_name="python3",
            resources={"metadata": {"path": str(NB_DIR)}},
            allow_errors=False,
        )
        try:
            client.execute()
            nbf.write(doc, path)
            print(f"  OK   {path.name}")
        except CellExecutionError as exc:
            failures.append((path.name, str(exc).splitlines()[-1][:160]))
            nbf.write(doc, path)
            print(f"  FAIL {path.name}")

    if failures:
        print("\nFailed notebooks:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print(f"\nAll {len(built)} notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
