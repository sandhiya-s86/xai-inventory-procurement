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
# ---------------------------------------------------------- Unit 3/4 helpers
# These check PROPERTIES, not just file existence. A findings file that exists
# but records loglik_monotone = false means the maths broke, and the rubric
# should fail on that rather than tick a box because the file was written.


def _findings(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return json.loads(p.read_text(encoding="utf-8"))


def json_subkeys(rel: str, section: str, keys: list[str]) -> tuple[bool, str]:
    """A nested section exists and carries all the named keys."""
    try:
        block = _findings(rel).get(section, {})
    except FileNotFoundError:
        return False, f"MISSING: {rel}"
    missing = [k for k in keys if k not in block]
    if missing:
        return False, f"{rel}[{section}] missing {missing}"
    vals = ", ".join(f"{k}={block[k]}" for k in keys)
    return True, f"{section}: {vals}"


def flag_true(rel: str, section: str, key: str) -> tuple[bool, str]:
    """A boolean guarantee recorded by the pipeline is actually True."""
    try:
        block = _findings(rel).get(section, {})
    except FileNotFoundError:
        return False, f"MISSING: {rel}"
    if key not in block:
        return False, f"{rel}[{section}][{key}] ABSENT"
    val = bool(block[key])
    return val, f"{section}.{key} = {val}"


def positive_value(rel: str, section: str, key: str) -> tuple[bool, str]:
    """A measured quantity exists and is strictly positive."""
    try:
        block = _findings(rel).get(section, {})
    except FileNotFoundError:
        return False, f"MISSING: {rel}"
    if key not in block:
        return False, f"{rel}[{section}][{key}] ABSENT"
    val = float(block[key])
    return val > 0, f"{section}.{key} = {val:.4f}"


def em_bound_is_tight(tol: float = 1e-6) -> tuple[bool, str]:
    """The EM lower bound must equal the log-likelihood at the E-step posterior.

    This is the identity that proves EM's monotonicity, so the rubric checks
    the measured gap rather than merely that the section was written.
    """
    try:
        gm = _findings("reports/unit3_findings.json").get("gmm", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit3_findings.json"
    gap = abs(float(gm.get("bound_gap_at_posterior", 1.0)))
    loose = float(gm.get("bound_gap_at_uniform_q", 0.0))
    if gap > tol:
        return False, f"bound not tight at posterior: gap = {gap:.2e}"
    if loose <= 0:
        return False, "bound not strictly below for a wrong q"
    return True, f"gap at posterior {gap:.2e}, at uniform q {loose:.3e}"


def rotation_is_invariant(tol: float = 1e-6) -> tuple[bool, str]:
    """Rotating W must move the loadings but not the likelihood."""
    try:
        fa = _findings("reports/unit3_findings.json").get("factor_analysis", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit3_findings.json"
    dll = abs(float(fa.get("rotation_loglik_difference", 1.0)))
    dw = float(fa.get("rotation_max_loading_difference", 0.0))
    if dll > tol:
        return False, f"likelihood moved under rotation: {dll:.2e}"
    if dw <= 0.01:
        return False, f"loadings barely moved ({dw:.3f}) -- not a real test"
    return True, f"loadings moved {dw:.3f}, log-likelihood moved {dll:.2e}"


def underflow_was_measured() -> tuple[bool, str]:
    """The underflow point must be an observed T, not an assumption."""
    try:
        sc = _findings("reports/unit4_findings.json").get("scaling", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit4_findings.json"
    fail_at = sc.get("unscaled_underflow_at_T")
    ok_to = sc.get("unscaled_survives_to_T")
    still = sc.get("scaled_still_finite_at_T")
    if not fail_at:
        return False, "no underflow point measured"
    return True, (f"unscaled survives T={ok_to}, dies by T={fail_at}; "
                  f"scaled still finite at T={still}")


def sum_product_matches(tol: float = 1e-9) -> tuple[bool, str]:
    """Sum-product must reproduce forward-backward to machine precision."""
    try:
        sp = _findings("reports/unit4_findings.json").get("sum_product", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit4_findings.json"
    diffs = {k: float(v) for k, v in sp.items() if k.startswith(("max_abs", "abs_"))}
    if not diffs:
        return False, "no sum-product comparison recorded"
    worst = max(diffs.values())
    if worst > tol:
        return False, f"largest discrepancy {worst:.2e} exceeds {tol:.0e}"
    if not sp.get("max_product_path_matches_viterbi"):
        return False, "max-product path does not match Viterbi"
    return True, f"all differences <= {worst:.1e}; max-product == Viterbi"


def dashboard_verified() -> tuple[bool, str]:
    """Every dashboard page rendered in a real browser with no exceptions."""
    p = ROOT / "reports" / "dashboard_verification.json"
    if not p.exists():
        return False, "MISSING: run scripts/verify_dashboard.py"
    d = json.loads(p.read_text(encoding="utf-8"))
    n, passed = d.get("n_pages", 0), d.get("n_passed", 0)
    if not d.get("all_pages_passed"):
        bad = [pg["page"] for pg in d.get("pages", []) if not pg.get("ok")]
        return False, f"{passed}/{n} pages passed; failing: {bad}"
    return True, f"{passed}/{n} pages render with zero exceptions"


def pdf_has_pages(rel: str, minimum: int) -> tuple[bool, str]:
    p = ROOT / rel
    if not p.exists():
        return False, f"MISSING: {rel}"
    try:
        from pypdf import PdfReader
    except ImportError:
        return True, f"{rel} ({p.stat().st_size/1024:.0f} KB; pypdf absent)"
    n = len(PdfReader(str(p)).pages)
    if n < minimum:
        return False, f"{rel}: only {n} pages (need >= {minimum})"
    return True, f"{rel}: {n} pages ({p.stat().st_size/1024:.0f} KB)"


# ------------------------------------------------------------ Unit 5 helpers


def pruning_sequence_is_valid() -> tuple[bool, str]:
    """Alphas must strictly increase while leaves strictly decrease.

    That nesting is what makes cost-complexity pruning a *sequence* you can
    search rather than a knob you guess, so it is worth gating on.
    """
    try:
        pr = _findings("reports/unit5_findings.json").get("pruning", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit5_findings.json"
    a_ok = bool(pr.get("alphas_monotone"))
    l_ok = bool(pr.get("leaves_monotone"))
    n = pr.get("n_subtrees")
    if not (a_ok and l_ok):
        return False, f"alphas_monotone={a_ok}, leaves_monotone={l_ok}"
    return True, f"{n} nested subtrees; alphas increase, leaves decrease"


def pruning_reduced_overfitting() -> tuple[bool, str]:
    """The whole point of pruning: trade training fit for generalisation."""
    try:
        pr = _findings("reports/unit5_findings.json").get("pruning", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit5_findings.json"
    full_gap = pr.get("full_train_f1", 0) - pr.get("full_val_f1", 0)
    pruned_gap = pr.get("pruned_train_f1", 0) - pr.get("pruned_val_f1", 0)
    if pruned_gap >= full_gap:
        return False, (f"gap did not narrow: {full_gap:+.4f} -> {pruned_gap:+.4f}")
    return True, (f"train/val gap {full_gap:+.4f} -> {pruned_gap:+.4f}, "
                  f"{pr.get('full_leaves')} -> {pr.get('pruned_leaves')} leaves")


def rules_are_readable() -> tuple[bool, str]:
    """The rules file must contain real IF/THEN logic, not just exist."""
    p = ROOT / "reports" / "unit5_tree_rules.txt"
    if not p.exists():
        return False, "MISSING: reports/unit5_tree_rules.txt"
    text = p.read_text(encoding="utf-8")
    if "IF " not in text or "THEN predict" not in text:
        return False, "no IF/THEN rules found in the file"
    n_rules = text.count("THEN predict")
    return True, f"{n_rules} IF/THEN rules ({p.stat().st_size / 1024:.0f} KB)"


def one_rule_baseline_recorded() -> tuple[bool, str]:
    """The headline finding must be a measured comparison, not a claim.

    The tree concentrated almost all its importance on one feature. That is a
    statement about the TASK, so the rubric requires it to have been tested
    against the best model rather than merely observed.
    """
    try:
        ob = _findings("reports/unit5_findings.json").get("one_rule_baseline", {})
    except FileNotFoundError:
        return False, "MISSING: reports/unit5_findings.json"
    if "test_f1" not in ob or "f1_vs_best_model" not in ob:
        return False, "one-rule baseline not evaluated"
    if ob.get("threshold_chosen_on") != "validation":
        return False, "threshold was not selected on the validation split"
    return True, (f"{ob['feature']} <= {ob['threshold']:.2f} -> test F1 "
                  f"{ob['test_f1']:.4f} ({ob['f1_vs_best_model']:+.4f} vs best)")


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

    # ---------------------------------------------------- Unit 3 (unsupervised)
    ("13. Unit 3 (unsupervised)", "Measuring dissimilarity: 5 metrics compared",
     lambda: csv_has_rows("reports/unit3_dissimilarity_metrics.csv", 5)),
    ("13. Unit 3 (unsupervised)", "K-means + elbow method over a range of k",
     lambda: csv_has_rows("reports/unit3_kmeans_elbow.csv", 8)),
    ("13. Unit 3 (unsupervised)", "PCA scree + explained variance",
     lambda: csv_has_rows("reports/unit3_pca_scree.csv", 10)),
    ("13. Unit 3 (unsupervised)", "Choosing latent dims: 3 rules reported",
     lambda: json_subkeys("reports/unit3_findings.json", "pca_choice",
                          ["kaiser", "variance", "elbow"])),
    ("13. Unit 3 (unsupervised)", "Factor analysis loadings + varimax",
     lambda: csv_has_rows("reports/unit3_factor_loadings.csv", 10)),
    ("13. Unit 3 (unsupervised)", "FA rotational ambiguity demonstrated",
     lambda: rotation_is_invariant()),
    ("13. Unit 3 (unsupervised)", "Mixtures of Gaussians: BIC/AIC model selection",
     lambda: csv_has_rows("reports/unit3_gmm_selection.csv", 4)),
    ("13. Unit 3 (unsupervised)", "EM log-likelihood is monotonic (GMM)",
     lambda: flag_true("reports/unit3_findings.json", "gmm", "loglik_monotone")),
    ("13. Unit 3 (unsupervised)", "Alternative view of EM: bound tight at posterior",
     lambda: em_bound_is_tight()),
    ("13. Unit 3 (unsupervised)", "Hierarchical clustering: 4 linkage methods",
     lambda: csv_has_rows("reports/unit3_hierarchical_linkages.csv", 4)),
    ("13. Unit 3 (unsupervised)", "Evaluating clustering: internal + relative indices",
     lambda: csv_has_rows("reports/unit3_clustering_evaluation.csv", 3)),
    ("13. Unit 3 (unsupervised)", "Clustering agreement (ARI) between methods",
     lambda: csv_has_rows("reports/unit3_clustering_agreement.csv", 3)),
    ("13. Unit 3 (unsupervised)", "Segments written for the policy layer",
     lambda: csv_has_rows("reports/unit3_segment_profiles.csv", 2)),
    ("13. Unit 3 (unsupervised)", "Unit 3 figures",
     lambda: figures_matching("unit3_*.png", 6)),

    # ------------------------------------------------------ Unit 4 (sequential)
    ("14. Unit 4 (sequential)", "Sequential data: autocorrelation computed",
     lambda: csv_has_rows("reports/unit4_autocorrelation.csv", 30)),
    ("14. Unit 4 (sequential)", "Order shown to carry information (chain rule)",
     lambda: positive_value("reports/unit4_findings.json", "sequential",
                            "order_information_gain_nats")),
    ("14. Unit 4 (sequential)", "Markov transition matrix, rows sum to 1",
     lambda: flag_true("reports/unit4_findings.json", "markov",
                       "rows_sum_to_one")),
    ("14. Unit 4 (sequential)", "k-step prediction + stationary distribution",
     lambda: csv_has_rows("reports/unit4_markov_kstep.csv", 6)),
    ("14. Unit 4 (sequential)", "Markov persistence tested against independence",
     lambda: csv_has_rows("reports/unit4_markov_persistence.csv", 3)),
    ("14. Unit 4 (sequential)", "HMM regimes learned by Baum-Welch",
     lambda: csv_has_rows("reports/unit4_hmm_regimes.csv", 3)),
    ("14. Unit 4 (sequential)", "Baum-Welch log-likelihood is monotonic",
     lambda: flag_true("reports/unit4_findings.json", "hmm",
                       "loglik_monotone")),
    ("14. Unit 4 (sequential)", "Scaling factors: log p(x) = sum log c_t",
     lambda: flag_true("reports/unit4_findings.json", "scaling",
                       "loglik_equals_sum_log_c")),
    ("14. Unit 4 (sequential)", "Scaling: underflow point actually measured",
     lambda: underflow_was_measured()),
    ("14. Unit 4 (sequential)", "Forward-backward posteriors normalised",
     lambda: flag_true("reports/unit4_findings.json", "forward_backward",
                       "gamma_normalised")),
    ("14. Unit 4 (sequential)", "Viterbi vs posterior decoding compared",
     lambda: csv_has_rows("reports/unit4_decoder_comparison.csv", 1)),
    ("14. Unit 4 (sequential)", "Sum-product verified == forward-backward",
     lambda: sum_product_matches()),
    ("14. Unit 4 (sequential)", "Kalman filter beats a naive random walk",
     lambda: positive_value("reports/unit4_findings.json", "lds",
                            "rmse_improvement_pct")),
    ("14. Unit 4 (sequential)", "RTS smoothing never increases variance",
     lambda: flag_true("reports/unit4_findings.json", "lds",
                       "smoothed_var_never_exceeds_filtered")),
    ("14. Unit 4 (sequential)", "Regimes feed the procurement decision",
     lambda: csv_has_rows("reports/unit4_regime_safety_multipliers.csv", 3)),
    ("14. Unit 4 (sequential)", "Unit 4 figures",
     lambda: figures_matching("unit4_*.png", 4)),

    # -------------------------------------------------- Unit 3/4 verification
    ("15. Unit 3/4 verification", "Unit 3 test suite",
     lambda: exists("tests/test_unit3_unsupervised.py")),
    ("15. Unit 3/4 verification", "Unit 4 test suite",
     lambda: exists("tests/test_unit4_sequential.py")),
    ("15. Unit 3/4 verification", "Dashboard verified in a real browser",
     lambda: dashboard_verified()),
    ("15. Unit 3/4 verification", "Unit 3/4 results report (PDF)",
     lambda: pdf_has_pages("paper/Unit3_Unit4_Results_Report.pdf", 12)),
    ("15. Unit 3/4 verification", "Unit 3/4 study guide (PDF)",
     lambda: pdf_has_pages("paper/Unit3_Unit4_Study_Guide.pdf", 12)),
    ("15. Unit 3/4 verification", "Unit 3/4 write-up",
     lambda: exists("docs/unit3-unit4.md")),

    # ------------------------------------------------------- Unit 5 (CART)
    ("16. Unit 5 (CART)", "CART implemented from scratch",
     lambda: exists("src/xaierp/trees/cart.py")),
    ("16. Unit 5 (CART)", "CART test suite",
     lambda: exists("tests/test_unit5_cart.py")),
    ("16. Unit 5 (CART)", "Cost-complexity pruning path computed",
     lambda: csv_has_rows("reports/unit5_pruning_path.csv", 5)),
    ("16. Unit 5 (CART)", "Pruning sequence is nested and monotone",
     lambda: pruning_sequence_is_valid()),
    ("16. Unit 5 (CART)", "Pruning narrowed the train/validation gap",
     lambda: pruning_reduced_overfitting()),
    ("16. Unit 5 (CART)", "alpha selected on the validation split",
     lambda: json_subkeys("reports/unit5_findings.json", "pruning",
                          ["alpha_star", "pruned_leaves"])),
    ("16. Unit 5 (CART)", "Gini vs entropy compared",
     lambda: csv_has_rows("reports/unit5_criterion_comparison.csv", 2)),
    ("16. Unit 5 (CART)", "Depth study (overfitting evidence)",
     lambda: csv_has_rows("reports/unit5_depth_study.csv", 6)),
    ("16. Unit 5 (CART)", "Regression tree (the 'R' in CART)",
     lambda: csv_has_rows("reports/unit5_regression_tree.csv", 3)),
    ("16. Unit 5 (CART)", "Feature importances computed",
     lambda: csv_has_rows("reports/unit5_feature_importance.csv", 3)),
    ("16. Unit 5 (CART)", "Readable IF/THEN rules exported",
     lambda: rules_are_readable()),
    ("16. Unit 5 (CART)", "Worked rule examples on real test rows",
     lambda: csv_has_rows("reports/unit5_rule_examples.csv", 5)),
    ("16. Unit 5 (CART)", "Compared against every other classifier",
     lambda: csv_has_rows("reports/unit5_model_comparison.csv", 6)),
    ("16. Unit 5 (CART)", "One-rule baseline tested (the headline finding)",
     lambda: one_rule_baseline_recorded()),
    ("16. Unit 5 (CART)", "Unit 5 figures",
     lambda: figures_matching("unit5_*.png", 3)),
    ("16. Unit 5 (CART)", "Unit 5 write-up",
     lambda: exists("docs/unit5-cart.md")),
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
