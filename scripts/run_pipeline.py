"""Run the whole project end to end, one stage per subprocess.

    .venv\\Scripts\\python scripts\\run_pipeline.py
    .venv\\Scripts\\python scripts\\run_pipeline.py --from train_regression
    .venv\\Scripts\\python scripts\\run_pipeline.py --skip download

Each stage runs in its own interpreter. That keeps peak memory low (this
project was developed on a machine running near its commit limit) and means
a failure in one stage does not force the earlier ones to be redone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (name, argv, what it produces)
STAGES: list[tuple[str, list[str], str]] = [
    ("download", ["scripts/download_data.py"],
     "data/raw/online_retail/online_retail_II.xlsx"),
    ("prepare_data", ["scripts/prepare_data.py"],
     "daily panel + data quality report"),
    ("eda", ["scripts/run_eda.py"],
     "16 EDA figures + dataset summary + findings json"),
    ("generate_supply", ["sim/generate_supply.py"],
     "SIMULATED supplier master"),
    ("inventory_state", ["sim/inventory_state.py"],
     "SIMULATED daily inventory + reorder labels"),
    ("build_features", ["scripts/build_features.py"],
     "feature table with chronological splits"),
    ("train_regression", ["scripts/train_models.py", "--stage", "regression"],
     "regression comparison table"),
    ("train_polynomial", ["scripts/train_models.py", "--stage", "polynomial"],
     "polynomial degree study (Unit 1)"),
    ("train_quantile", ["scripts/train_models.py", "--stage", "quantile"],
     "quantile models for safety stock"),
    ("train_classification", ["scripts/train_models.py", "--stage", "classification"],
     "classification comparison table"),
    ("predictions", ["scripts/train_models.py", "--stage", "predictions"],
     "test prediction table + model summary"),
    ("simulation", ["scripts/run_simulation.py"],
     "policy ladder, ablation, procurement recommendations"),
    ("unsupervised", ["scripts/run_unsupervised.py"],
     "Unit 3: dissimilarity, PCA, factor analysis, K-means, GMM/EM, hierarchical"),
    ("sequential", ["scripts/run_sequential.py"],
     "Unit 4: Markov, HMM/Baum-Welch, forward-backward, Viterbi, sum-product, Kalman"),
    ("explainability", ["scripts/run_explainability.py"],
     "SHAP, case studies, what-if"),
    ("plots", ["scripts/make_result_plots.py"],
     "model comparison figures"),
    ("notebooks", ["scripts/build_notebooks.py"],
     "10 executed notebooks"),
    ("presentation", ["scripts/build_presentation.py"],
     "paper/PBL1_Presentation.pptx"),
    ("unit34_docs", ["scripts/build_unit34_docs.py"],
     "Unit 3/4 results report PDF + 3-person study guide PDF"),
    ("tests", [], "pytest suite"),
    ("completeness", ["scripts/check_pbl1_completeness.py"],
     "PBL-1 rubric completeness report"),
]


def run(name: str, argv: list[str]) -> tuple[bool, float]:
    t0 = time.time()
    if name == "tests":
        cmd = [PY, "-m", "pytest", "-q"]
    else:
        cmd = [PY, *argv]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode == 0, time.time() - t0


def main() -> int:
    names = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=names,
                    help="begin at this stage")
    ap.add_argument("--only", choices=names, help="run just this stage")
    ap.add_argument("--skip", nargs="*", default=[], choices=names,
                    help="stages to skip")
    args = ap.parse_args()

    todo = STAGES
    if args.only:
        todo = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        i = names.index(args.start)
        todo = STAGES[i:]
    todo = [s for s in todo if s[0] not in args.skip]

    print("=" * 72)
    print("XAI INVENTORY & PROCUREMENT -- FULL PIPELINE")
    print(f"{len(todo)} stages")
    print("=" * 72)

    results = []
    for n, (name, argv, produces) in enumerate(todo, 1):
        print(f"\n{'#' * 72}")
        print(f"# [{n}/{len(todo)}] {name}  ->  {produces}")
        print(f"{'#' * 72}")
        ok, dt = run(name, argv)
        results.append((name, ok, dt))
        if not ok:
            print(f"\nSTAGE FAILED: {name} (after {dt:.1f}s)")
            print("Fix the error, then resume with:")
            print(f"  python scripts/run_pipeline.py --from {name}")
            break

    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)
    total = 0.0
    for name, ok, dt in results:
        total += dt
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {dt:8.1f}s")
    print(f"  {'':6}{'TOTAL':<22} {total:8.1f}s")

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\nFailed stages: {failed}")
        return 1
    if len(results) == len(STAGES):
        print("\nAll stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
