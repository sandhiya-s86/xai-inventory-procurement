"""Verify that every PBL-1 rubric requirement has a real artefact behind it.

This checks for EVIDENCE, not intent: a criterion passes only when the file
exists and, where applicable, contains what it claims to. Exits non-zero if
anything required is missing, so it can gate a release.

    .venv\\Scripts\\python scripts\\check_pbl1_completeness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.utils.config import load_config, resolve  # noqa: E402

CFG = load_config()
REPORTS = resolve(CFG["paths"]["reports"])
FIGS = resolve(CFG["paths"]["figures"])


def exists(rel: str) -> tuple[bool, str]:
    p = ROOT / rel
    if p.exists():
        size = p.stat().st_size
        return True, f"{rel} ({size / 1024:.0f} KB)"
    return False, f"MISSING: {rel}"


def figures_matching(pattern: str, minimum: int) -> tuple[bool, str]:
    n = len(list(FIGS.glob(pattern)))
    return n >= minimum, f"{n} figures matching {pattern} (need >= {minimum})"


def csv_has_rows(rel: str, minimum: int) -> tuple[bool, str]:
    p = ROOT / rel
    if not p.exists():
        return False, f"MISSING: {rel}"
    n = len(pd.read_csv(p))
    return n >= minimum, f"{rel}: {n} rows (need >= {minimum})"


def csv_contains(rel: str, column: str, needles: list[str]) -> tuple[bool, str]:
    p = ROOT / rel
    if not p.exists():
        return False, f"MISSING: {rel}"
    vals = " ".join(pd.read_csv(p)[column].astype(str)).lower()
    missing = [n for n in needles if n.lower() not in vals]
    if missing:
        return False, f"{rel} missing: {missing}"
    return True, f"{rel} contains {needles}"


def json_key(rel: str, key: str) -> tuple[bool, str]:
    p = ROOT / rel
    if not p.exists():
        return False, f"MISSING: {rel}"
    data = json.loads(p.read_text(encoding="utf-8"))
    return key in data, f"{rel}[{key}] = {data.get(key, 'ABSENT')}"


def pptx_is_valid(rel: str, min_slides: int) -> tuple[bool, str]:
    """The deck must open, have enough slides, and carry speaker notes."""
    p = ROOT / rel
    if not p.exists():
        return False, f"MISSING: {rel}"
    try:
        from pptx import Presentation
    except ImportError:
        return False, "python-pptx not installed"
    prs = Presentation(str(p))
    n = len(prs.slides)
    with_notes = sum(
        1 for s in prs.slides if s.notes_slide.notes_text_frame.text.strip()
    )
    if n < min_slides:
        return False, f"{rel}: only {n} slides (need >= {min_slides})"
    if with_notes < n:
        return False, f"{rel}: {with_notes}/{n} slides have speaker notes"
    return True, f"{rel}: {n} slides, all with speaker notes ({p.stat().st_size/1024:.0f} KB)"


def all_cases_reconcile() -> tuple[bool, str]:
    p = REPORTS / "generated" / "case_studies.json"
    if not p.exists():
        return False, "MISSING: case_studies.json"
    cases = json.loads(p.read_text(encoding="utf-8"))
    bad = [c["sku"] for c in cases if not c.get("reconciles")]
    if bad:
        return False, f"{len(bad)} case studies do not reconcile: {bad}"
    return True, f"{len(cases)} case studies, all decompositions reconcile"


# (rubric criterion, check label, callable)
CHECKS: list[tuple[str, str, callable]] = [
    # ---- 1. Problem understanding & dataset familiarisation (4 marks)
    ("1. Problem Understanding", "Problem & scope documented",
     lambda: exists("docs/pbl1_scope.md")),
    ("1. Problem Understanding", "Dataset acquisition documented",
     lambda: exists("docs/dataset_acquisition.md")),
    ("2. Dataset Familiarization", "Raw data acquired",
     lambda: exists("data/raw/online_retail/online_retail_II.xlsx")),
    ("2. Dataset Familiarization", "Data quality report",
     lambda: exists("reports/data_quality_report.json")),
    ("2. Dataset Familiarization", "Dataset summary table",
     lambda: csv_has_rows("reports/dataset_summary.csv", 10)),
    ("2. Dataset Familiarization", "Familiarisation notebook",
     lambda: exists("notebooks/01_dataset_familiarization.ipynb")),
    ("2. Dataset Familiarization", "Preprocessing pipeline",
     lambda: exists("src/xaierp/ingest/online_retail.py")),
    ("2. Dataset Familiarization", "Processed daily panel",
     lambda: exists("data/processed/daily_demand_panel.parquet")),

    # ---- 3. EDA & insights (4 marks)
    ("3. EDA", "EDA notebook",
     lambda: exists("notebooks/02_eda.ipynb")),
    ("3. EDA", "EDA figures",
     lambda: figures_matching("eda_*.png", 14)),
    ("3. EDA", "EDA insights written up",
     lambda: exists("docs/eda_insights.md")),
    ("3. EDA", "EDA findings json (numbers behind the prose)",
     lambda: exists("reports/eda_findings.json")),

    # ---- 4. Unit 1 / Unit 2 implementation incl. regression & classification
    ("4. Regression", "Regression comparison table",
     lambda: csv_has_rows("reports/regression_model_comparison.csv", 5)),
    ("4. Regression", "Linear + Ridge + Bayesian + Huber present",
     lambda: csv_contains(
         "reports/regression_model_comparison.csv", "model",
         ["Linear Regression", "Ridge", "Bayesian", "Huber"])),
    ("4. Regression", "Polynomial degree study (Unit 1)",
     lambda: csv_has_rows("reports/polynomial_degree_study.csv", 4)),
    ("4. Regression", "Quantile regression (uncertainty)",
     lambda: csv_has_rows("reports/quantile_model_metrics.csv", 4)),
    ("5. Classification", "Classification comparison table",
     lambda: csv_has_rows("reports/classification_model_comparison.csv", 5)),
    ("5. Classification", "Logistic + LDA + NaiveBayes + SVM present",
     lambda: csv_contains(
         "reports/classification_model_comparison.csv", "model",
         ["Logistic", "Discriminant", "Naive Bayes", "SVM (linear", "SVM (RBF"])),
    ("5. Classification", "Confusion matrix figure",
     lambda: exists("reports/figures/model_confusion_matrix.png")),
    ("5. Classification", "ROC curves figure",
     lambda: exists("reports/figures/model_roc_curves.png")),
    ("6. Unit 1 Implementation", "Unit 1 concepts mapped to code",
     lambda: exists("docs/pbl1_rubric_mapping.md")),
    ("6. Unit 1 Implementation", "Probability / quantile analysis notebook",
     lambda: exists("notebooks/06_uncertainty_and_quantiles.ipynb")),
    ("7. Unit 2 Implementation", "Kernel comparison (linear vs RBF SVM)",
     lambda: csv_contains(
         "reports/classification_model_comparison.csv", "concept",
         ["kernel"])),
    ("7. Unit 2 Implementation", "Ridge alpha search (regularisation)",
     lambda: csv_has_rows("reports/ridge_alpha_search.csv", 4)),

    # ---- 8. Comparative performance analysis (4 marks)
    ("8. Comparison", "Model summary json",
     lambda: json_key("reports/model_summary.json", "best_regression_model")),
    ("8. Comparison", "Regression RMSE figure",
     lambda: exists("reports/figures/model_regression_rmse.png")),
    ("8. Comparison", "Classification F1 figure",
     lambda: exists("reports/figures/model_classification_f1.png")),
    ("8. Comparison", "Polynomial overfitting figure",
     lambda: exists("reports/figures/model_polynomial_overfitting.png")),
    ("8. Comparison", "Model selection rationale",
     lambda: exists("docs/model_selection.md")),

    # ---- 9. Explainability
    ("9. Explainability", "SHAP importance (forecast)",
     lambda: csv_has_rows("reports/shap_global_importance_forecast.csv", 10)),
    ("9. Explainability", "SHAP importance (classifier)",
     lambda: csv_has_rows("reports/shap_global_importance_classifier.csv", 10)),
    ("9. Explainability", "Case studies with real explanations",
     lambda: all_cases_reconcile()),
    ("9. Explainability", "What-if / counterfactual analysis",
     lambda: csv_has_rows("reports/whatif_examples.csv", 8)),
    ("9. Explainability", "XAI notebook",
     lambda: exists("notebooks/08_xai_explanations.ipynb")),

    # ---- 10. Procurement decision
    ("10. Procurement Decision", "Reorder policy implementation",
     lambda: exists("src/xaierp/policy/reorder.py")),
    ("10. Procurement Decision", "Procurement recommendations",
     lambda: exists("data/results/procurement_recommendations.parquet")),
    ("10. Procurement Decision", "Policy ladder comparison",
     lambda: csv_has_rows("reports/policy_comparison.csv", 5)),
    ("10. Procurement Decision", "Ablation study",
     lambda: csv_has_rows("reports/ablation_study.csv", 5)),
    ("10. Procurement Decision", "SIMULATED supply clearly configured",
     lambda: exists("sim/supply_config.yaml")),

    # ---- 11. Testing
    ("11. Testing", "Leakage tests",
     lambda: exists("tests/test_no_leakage.py")),
    ("11. Testing", "Inventory conservation tests",
     lambda: exists("tests/test_inventory_conservation.py")),
    ("11. Testing", "Reorder policy tests",
     lambda: exists("tests/test_reorder_policy.py")),
    ("11. Testing", "Decision explanation tests",
     lambda: exists("tests/test_decision_explanation.py")),
    ("11. Testing", "Simulation reproducibility tests",
     lambda: exists("tests/test_simulation.py")),

    # ---- 12. Presentation & delivery
    ("12. Presentation", "PBL report",
     lambda: exists("paper/pbl1_report.md")),
    ("12. Presentation", "Slide deck outline with speaking script",
     lambda: exists("docs/pbl1_presentation.md")),
    ("12. Presentation", "PowerPoint deck (.pptx)",
     lambda: pptx_is_valid("paper/PBL1_Presentation.pptx", 14)),
    ("12. Presentation", "Viva question bank",
     lambda: exists("docs/viva_questions.md")),
    ("12. Presentation", "Streamlit dashboard",
     lambda: exists("dashboard/app.py")),
    ("12. Presentation", "ERP integration note",
     lambda: exists("docs/erp_integration_ready.md")),
    ("12. Presentation", "Reproducible pipeline entry point",
     lambda: exists("scripts/run_pipeline.py")),
]


def main() -> int:
    results: list[tuple[str, str, bool, str]] = []
    for criterion, label, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        results.append((criterion, label, ok, detail))

    print("=" * 78)
    print("PBL-1 COMPLETENESS")
    print("=" * 78)

    criteria: dict[str, list] = {}
    for crit, label, ok, detail in results:
        criteria.setdefault(crit, []).append((label, ok, detail))

    all_ok = True
    for crit, items in criteria.items():
        passed = sum(1 for _, ok, _ in items if ok)
        status = "PASS" if passed == len(items) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"\n{crit:<34} {status}   ({passed}/{len(items)})")
        for label, ok, detail in items:
            mark = "  [x]" if ok else "  [ ]"
            print(f"{mark} {label:<48} {detail}")

    total = len(results)
    passed = sum(1 for _, _, ok, _ in results if ok)
    print("\n" + "=" * 78)
    print(f"TOTAL: {passed}/{total} checks passed")

    if all_ok:
        print("\nAll rubric criteria have supporting artefacts.")
    else:
        print("\nSome criteria are unsupported. See the [ ] rows above.")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
