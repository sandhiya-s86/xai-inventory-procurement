"""Stage: Unit 3 -- unsupervised analysis of the SKU demand profiles.

Runs the whole Unit 3 syllabus on the real 200-SKU panel and writes every
number it quotes to a file, so the report and the data cannot drift apart.

    1.  SKU profile matrix (train period only -- no leakage)
    2.  Measuring dissimilarity: 5 metrics compared
    3.  PCA + scree + choosing the number of latent dimensions
    4.  Factor analysis (EM), varimax, rotational ambiguity
    5.  K-means + elbow method
    6.  Mixtures of Gaussians (EM), BIC/AIC, the alternative view of EM
    7.  Hierarchical clustering: 4 linkages, dendrogram, cophenetic corr
    8.  Evaluating the output of clustering methods
    9.  Segment assignment -> data/processed/sku_segments.parquet

Writes:
    reports/unit3_*.csv            (9 tables)
    reports/unit3_findings.json
    reports/figures/unit3_*.png    (6 figures)
    data/processed/sku_segments.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.unsupervised.dissimilarity import (  # noqa: E402
    dissimilarity_matrix,
    metric_comparison,
    standardize,
)
from xaierp.unsupervised.evaluation import (  # noqa: E402
    adjusted_rand_index,
    evaluate_clustering,
    silhouette_samples,
)
from xaierp.unsupervised.factor_analysis import (  # noqa: E402
    FactorAnalysisEM,
    rotational_ambiguity_check,
    varimax,
)
from xaierp.unsupervised.gmm import (  # noqa: E402
    GaussianMixtureEM,
    em_lower_bound,
    select_n_components,
)
from xaierp.unsupervised.hierarchical import (  # noqa: E402
    compare_linkages,
    fcluster,
    linkage,
)
from xaierp.unsupervised.kmeans import KMeans, elbow_curve, knee_point  # noqa: E402
from xaierp.unsupervised.pca import PCA, choose_n_components, scree_table  # noqa: E402
from xaierp.unsupervised.profiles import (  # noqa: E402
    PROFILE_COLUMNS,
    build_sku_profiles,
    name_segments,
    profile_matrix,
    segment_summary,
)
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
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

K_RANGE = list(range(2, 11))


def save_fig(fig, name: str, figs: Path) -> None:
    path = figs / f"unit3_{name}.png"
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_table(rows, name: str, reports: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = reports / f"unit3_{name}.csv"
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"    saved -> reports/unit3_{name}.csv  ({len(df)} rows)")
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)
    reports = resolve(cfg["paths"]["reports"])
    figs = resolve(cfg["paths"]["figures"])
    processed = resolve(cfg["paths"]["processed"])
    ensure_dir(reports)
    ensure_dir(figs)

    find: dict[str, object] = {}

    print("=" * 72)
    print("UNIT 3 -- UNSUPERVISED LEARNING ON SKU DEMAND PROFILES")
    print("=" * 72)

    # ---------------------------------------------------------- 1. profiles
    print("\n[1/9] Building SKU profiles (train period only)")
    panel = pd.read_parquet(processed / "daily_demand_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    feats = pd.read_parquet(processed / "feature_table.parquet")
    train_end = pd.Timestamp(feats.loc[feats["split"] == "train", "date"].max())
    print(f"    train period ends {train_end.date()} "
          f"(test data is NOT used to build segments)")

    profiles = build_sku_profiles(panel, end_date=train_end)
    X_raw, cols = profile_matrix(profiles)
    X, mean, std = standardize(X_raw)
    print(f"    profile matrix: {X.shape[0]} SKUs x {X.shape[1]} behavioural features")

    find["n_skus"] = int(X.shape[0])
    find["n_profile_features"] = int(X.shape[1])
    find["train_end"] = str(train_end.date())
    find["profile_columns"] = cols

    save_table(
        [
            {
                "feature": c,
                "mean": float(X_raw[:, i].mean()),
                "std": float(X_raw[:, i].std()),
                "min": float(X_raw[:, i].min()),
                "max": float(X_raw[:, i].max()),
            }
            for i, c in enumerate(cols)
        ],
        "profile_features",
        reports,
    )

    # ------------------------------------------------------ 2. dissimilarity
    print("\n[2/9] Measuring dissimilarity -- comparing 5 metrics")
    metrics = metric_comparison(X)
    save_table(metrics, "dissimilarity_metrics", reports)
    for m in metrics:
        print(f"    {m['metric']:<12} mean d={m['mean_distance']:7.3f}  "
              f"rank agreement with Euclidean rho={m['spearman_vs_euclidean']:.3f}")
    find["dissimilarity_metrics"] = metrics

    # ----------------------------------------------------------- 3. PCA
    print("\n[3/9] PCA and choosing the number of latent dimensions")
    scree = scree_table(X)
    save_table(scree, "pca_scree", reports)
    choice = choose_n_components(X, variance_threshold=0.90)
    print(f"    Kaiser (eigenvalue>1): {choice['kaiser']} components")
    print(f"    90% variance rule    : {choice['variance']} components")
    print(f"    scree elbow          : {choice['elbow']} components")

    n_latent = int(choice["kaiser"])
    pca = PCA(n_components=n_latent).fit(X)
    scores = pca.transform(X)
    var_kept = float(pca.explained_variance_ratio_.sum())
    print(f"    -> using L={n_latent} (Kaiser), capturing {var_kept:.1%} of variance")
    print(f"    reconstruction MSE at L={n_latent}: {pca.reconstruction_error(X):.4f}")

    find["pca_choice"] = choice
    find["n_latent"] = n_latent
    find["pca_variance_captured"] = var_kept
    find["pca_reconstruction_mse"] = float(pca.reconstruction_error(X))

    save_table(
        [
            {"feature": c, **{f"PC{j+1}": float(pca.components_[j, i])
                              for j in range(n_latent)}}
            for i, c in enumerate(cols)
        ],
        "pca_loadings",
        reports,
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ev = [r["eigenvalue"] for r in scree]
    ax[0].plot(range(1, len(ev) + 1), ev, "o-", color="#2563eb")
    ax[0].axhline(1.0, color="crimson", ls="--", label="Kaiser: eigenvalue = 1")
    ax[0].set_xlabel("Component"); ax[0].set_ylabel("Eigenvalue")
    ax[0].set_title("Scree plot"); ax[0].legend()
    cum = [r["cumulative_variance_ratio"] for r in scree]
    ax[1].plot(range(1, len(cum) + 1), cum, "o-", color="#059669")
    ax[1].axhline(0.90, color="crimson", ls="--", label="90% of variance")
    ax[1].set_xlabel("Components retained")
    ax[1].set_ylabel("Cumulative explained variance")
    ax[1].set_title("How many latent dimensions?"); ax[1].legend()
    save_fig(fig, "01_pca_scree", figs)

    # ------------------------------------------------- 4. factor analysis
    print("\n[4/9] Factor analysis (EM) + varimax + rotational ambiguity")
    fa = FactorAnalysisEM(n_factors=n_latent, random_state=seed).fit(X)
    fa_ll = fa.log_likelihood(X)
    mono = all(
        fa.loglik_history_[i] <= fa.loglik_history_[i + 1] + 1e-6
        for i in range(len(fa.loglik_history_) - 1)
    )
    print(f"    EM ran {fa.n_iter_} iterations "
          f"({'converged' if fa.converged_ else 'HIT ITERATION CAP'}), "
          f"log-likelihood {fa_ll:,.2f}")
    print(f"    log-likelihood non-decreasing every iteration: {mono}")

    W_rot, R = varimax(fa.loadings_)
    amb = rotational_ambiguity_check(fa, X)
    print(f"    rotational ambiguity: loadings change by up to "
          f"{amb['max_loading_difference']:.3f} while the log-likelihood "
          f"changes by {amb['loglik_difference']:.2e}")
    print(f"    -> W is identified only up to {amb['n_free_rotations']} free rotations")

    comm = fa.communalities()
    save_table(
        [
            {
                "feature": c,
                **{f"factor{j+1}_raw": float(fa.loadings_[i, j]) for j in range(n_latent)},
                **{f"factor{j+1}_varimax": float(W_rot[i, j]) for j in range(n_latent)},
                "communality": float(comm[i]),
                "noise_variance": float(fa.noise_variance_[i]),
            }
            for i, c in enumerate(cols)
        ],
        "factor_loadings",
        reports,
    )

    find["factor_analysis"] = {
        "n_factors": n_latent,
        "loglik": fa_ll,
        "n_iter": fa.n_iter_,
        "converged": bool(fa.converged_),
        "loglik_monotone": bool(mono),
        "n_parameters": fa.n_parameters(),
        "mean_communality": float(comm.mean()),
        **{f"rotation_{k}": v for k, v in amb.items()},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(W_rot, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(W_rot).max(), vmax=np.abs(W_rot).max())
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_xticks(range(n_latent))
    ax.set_xticklabels([f"Factor {i+1}" for i in range(n_latent)])
    ax.set_title("Factor loadings after varimax rotation")
    for i in range(len(cols)):
        for j in range(n_latent):
            ax.text(j, i, f"{W_rot[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="loading")
    ax.grid(False)
    save_fig(fig, "02_factor_loadings", figs)

    # ------------------------------------------------------- 5. K-means
    print("\n[5/9] K-means + elbow method")
    elbow = elbow_curve(X, K_RANGE, n_init=10, random_state=seed)
    save_table(elbow, "kmeans_elbow", reports)
    k_star = knee_point([r["k"] for r in elbow], [r["inertia"] for r in elbow])
    print(f"    elbow at k = {k_star}")

    km = KMeans(n_clusters=k_star, n_init=20, random_state=seed).fit(X)
    inertia_monotone = all(
        km.inertia_history_[i] >= km.inertia_history_[i + 1] - 1e-9
        for i in range(len(km.inertia_history_) - 1)
    )
    print(f"    inertia {km.inertia_:.2f} in {km.n_iter_} iterations; "
          f"non-increasing every iteration: {inertia_monotone}")

    find["kmeans"] = {
        "k_elbow": int(k_star),
        "inertia": float(km.inertia_),
        "n_iter": int(km.n_iter_),
        "inertia_monotone": bool(inertia_monotone),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ks = [r["k"] for r in elbow]
    ax.plot(ks, [r["inertia"] for r in elbow], "o-", color="#2563eb")
    ax.axvline(k_star, color="crimson", ls="--", label=f"elbow: k = {k_star}")
    ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("Inertia (within-cluster SS)")
    ax.set_title("K-means elbow method"); ax.legend()
    save_fig(fig, "03_kmeans_elbow", figs)

    # ---------------------------------------------------------- 6. GMM
    print("\n[6/9] Mixtures of Gaussians (EM) + BIC/AIC + the EM lower bound")
    # Full covariances in 13 dimensions need ~91 parameters per component,
    # which 200 SKUs cannot support. The mixture is therefore fitted in the
    # PCA latent space, where the dimension is small enough to estimate.
    gmm_rows = select_n_components(scores, range(2, 7), random_state=seed)
    save_table(gmm_rows, "gmm_selection", reports)
    best_gmm_row = min(gmm_rows, key=lambda r: r["BIC"])
    k_gmm = int(best_gmm_row["n_components"])
    print(f"    BIC selects K = {k_gmm} components "
          f"(BIC {best_gmm_row['BIC']:.1f})")

    gmm = GaussianMixtureEM(n_components=k_gmm, random_state=seed).fit(scores)
    gmm_mono = all(
        gmm.loglik_history_[i] <= gmm.loglik_history_[i + 1] + 1e-7
        for i in range(len(gmm.loglik_history_) - 1)
    )
    resp = gmm.responsibilities_
    print(f"    EM: {gmm.n_iter_} iterations, log-likelihood "
          f"{gmm.loglik_history_[0]:.2f} -> {gmm.loglik_history_[-1]:.2f}")
    print(f"    log-likelihood non-decreasing every iteration: {gmm_mono}")

    # The alternative view of EM: the bound is TIGHT at the true posterior
    # and strictly below it for any other q.
    tight = em_lower_bound(scores, gmm, resp)
    # A mildly wrong q: the true posterior blended 10% towards uniform. This
    # gives a readable number; a fully uniform q is so far from this very
    # peaked posterior that the KL runs to ~1e9 and stops being informative.
    mild_q = 0.90 * resp + 0.10 / k_gmm
    mild_q /= mild_q.sum(axis=1, keepdims=True)
    mild = em_lower_bound(scores, gmm, mild_q)
    uniform_q = np.full_like(resp, 1.0 / k_gmm)
    loose = em_lower_bound(scores, gmm, uniform_q)
    print(f"    EM lower bound at the E-step posterior: gap = {tight['gap']:.2e} "
          f"(bound is TIGHT, KL = {tight['kl_divergence']:.2e})")
    print(f"    EM lower bound at a slightly wrong q  : gap = {mild['gap']:.2f} "
          f"= KL divergence {mild['kl_divergence']:.2f}")
    print(f"    EM lower bound at a uniform q         : gap = {loose['gap']:.3e} "
          f"= KL divergence {loose['kl_divergence']:.3e}")

    n_ambiguous = int((resp.max(axis=1) < 0.90).sum())
    print(f"    soft assignment: {n_ambiguous}/{len(resp)} SKUs have no component "
          f"above 90% responsibility")

    find["gmm"] = {
        "k_bic": k_gmm,
        "fitted_in": "PCA latent space",
        "loglik_first": float(gmm.loglik_history_[0]),
        "loglik_final": float(gmm.loglik_history_[-1]),
        "n_iter": int(gmm.n_iter_),
        "loglik_monotone": bool(gmm_mono),
        "bound_gap_at_posterior": float(tight["gap"]),
        "kl_at_posterior": float(tight["kl_divergence"]),
        "bound_gap_at_mild_q": float(mild["gap"]),
        "kl_at_mild_q": float(mild["kl_divergence"]),
        "bound_gap_at_uniform_q": float(loose["gap"]),
        "kl_at_uniform_q": float(loose["kl_divergence"]),
        "n_ambiguous_skus": n_ambiguous,
        "mean_max_responsibility": float(resp.max(axis=1).mean()),
    }

    save_table(
        [
            {
                "iteration": i,
                "log_likelihood": float(v),
                "improvement": float(v - gmm.loglik_history_[i - 1]) if i else float("nan"),
            }
            for i, v in enumerate(gmm.loglik_history_)
        ],
        "gmm_em_convergence",
        reports,
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(gmm.loglik_history_, "o-", color="#7c3aed")
    ax[0].set_xlabel("EM iteration"); ax[0].set_ylabel("Log-likelihood")
    ax[0].set_title("EM is monotonic: log-likelihood never decreases")
    ax[1].plot([r["n_components"] for r in gmm_rows],
               [r["BIC"] for r in gmm_rows], "o-", label="BIC", color="#2563eb")
    ax[1].plot([r["n_components"] for r in gmm_rows],
               [r["AIC"] for r in gmm_rows], "s--", label="AIC", color="#059669")
    ax[1].axvline(k_gmm, color="crimson", ls="--", label=f"BIC min: K = {k_gmm}")
    ax[1].set_xlabel("K (components)"); ax[1].set_ylabel("Criterion (lower is better)")
    ax[1].set_title("Choosing K for the mixture"); ax[1].legend()
    save_fig(fig, "04_gmm_em", figs)

    # ------------------------------------------------- 7. hierarchical
    print("\n[7/9] Hierarchical clustering -- 4 linkage methods")
    link_rows = compare_linkages(X, k_star)
    save_table(link_rows, "hierarchical_linkages", reports)
    for r in link_rows:
        print(f"    {r['linkage']:<9} cophenetic r={r['cophenetic_corr']:.3f}  "
              f"silhouette={r['silhouette']:.3f}  "
              f"largest cluster={r['largest_cluster_share']:.1%}")

    best_link = max(link_rows, key=lambda r: r["cophenetic_corr"])
    print(f"    best dendrogram fidelity: {best_link['linkage']} "
          f"(cophenetic r = {best_link['cophenetic_corr']:.3f})")

    # The two criteria disagree, and the disagreement is the finding. Single
    # and average linkage score highest on cophenetic correlation while
    # putting 97% of SKUs in ONE cluster -- classic chaining. A faithful
    # dendrogram is not the same thing as a useful partition, so the cut is
    # taken from Ward, which is the only method here that produces balanced
    # groups. Reported rather than quietly resolved.
    chaining = [r for r in link_rows if r["largest_cluster_share"] > 0.90]
    for r in chaining:
        print(f"    WARNING: {r['linkage']} linkage put "
              f"{r['largest_cluster_share']:.1%} of SKUs in one cluster "
              f"(chaining) -- high cophenetic r, unusable partition")
    print("    -> cutting the WARD tree: the only linkage giving balanced groups")

    Z_ward = linkage(X, "ward")
    hier_labels = fcluster(Z_ward, k_star)
    find["hierarchical"] = {
        "linkages": link_rows,
        "best_by_cophenetic": best_link["linkage"],
        "chaining_methods": [r["linkage"] for r in chaining],
        "cut_taken_from": "ward",
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z_ward, ax=ax, no_labels=True, color_threshold=Z_ward[-(k_star - 1), 2])
    ax.axhline(Z_ward[-(k_star - 1), 2], color="crimson", ls="--",
               label=f"cut for {k_star} clusters")
    ax.set_title("Hierarchical clustering of 200 SKUs (Ward linkage)")
    ax.set_xlabel("SKUs"); ax.set_ylabel("Merge distance")
    ax.legend()
    save_fig(fig, "05_dendrogram", figs)

    # -------------------------------------------------- 8. evaluation
    print("\n[8/9] Evaluating the output of clustering methods")
    gmm_hard = gmm.predict(scores)
    methods = {
        "K-means": km.labels_,
        "GMM (hard assignment)": gmm_hard,
        "Hierarchical (Ward)": hier_labels,
    }
    eval_rows = [evaluate_clustering(X, lab, name) for name, lab in methods.items()]
    save_table(eval_rows, "clustering_evaluation", reports)
    for r in eval_rows:
        print(f"    {r['method']:<22} silhouette={r['silhouette']:.3f}  "
              f"DB={r['davies_bouldin']:.3f}  CH={r['calinski_harabasz']:.1f}")

    names = list(methods)
    ari_rows = [
        {
            "method_a": a,
            "method_b": b,
            "adjusted_rand_index": adjusted_rand_index(methods[a], methods[b]),
        }
        for i, a in enumerate(names)
        for b in names[i + 1:]
    ]
    save_table(ari_rows, "clustering_agreement", reports)
    for r in ari_rows:
        print(f"    ARI {r['method_a']} vs {r['method_b']}: "
              f"{r['adjusted_rand_index']:.3f}")

    find["clustering_evaluation"] = eval_rows
    find["clustering_agreement"] = ari_rows

    best_method = max(eval_rows, key=lambda r: r["silhouette"])["method"]
    print(f"    best by silhouette: {best_method}")
    find["best_clustering_method"] = best_method

    sil = silhouette_samples(X, km.labels_)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y0 = 0
    for c in range(k_star):
        vals = np.sort(sil[km.labels_ == c])
        ax[0].barh(np.arange(y0, y0 + len(vals)), vals, height=1.0)
        y0 += len(vals) + 5
    ax[0].axvline(sil.mean(), color="crimson", ls="--",
                  label=f"mean = {sil.mean():.3f}")
    ax[0].set_xlabel("Silhouette coefficient"); ax[0].set_ylabel("SKUs by cluster")
    ax[0].set_title("Silhouette plot (K-means)"); ax[0].legend()

    sc = ax[1].scatter(scores[:, 0], scores[:, 1], c=km.labels_,
                       cmap="viridis", s=36, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax[1].set_title("SKU segments in the PCA latent space")
    fig.colorbar(sc, ax=ax[1], label="segment")
    save_fig(fig, "06_clusters", figs)

    # --------------------------------------------------- 9. segments out
    print("\n[9/9] Writing segment assignments")
    labels = km.labels_ if best_method == "K-means" else (
        gmm_hard if best_method.startswith("GMM") else hier_labels
    )
    summary = segment_summary(profiles, labels)
    seg_names = name_segments(summary)
    summary["segment_name"] = summary["segment"].map(seg_names)
    save_table(summary.to_dict("records"), "segment_profiles", reports)

    out = profiles.reset_index()[["stock_code", "description"]].copy()
    out["segment"] = labels
    out["segment_name"] = out["segment"].map(seg_names)
    out["silhouette"] = sil
    out["gmm_max_responsibility"] = resp.max(axis=1)
    out["gmm_component"] = gmm_hard
    for j in range(n_latent):
        out[f"pc{j+1}"] = scores[:, j]
    for c in PROFILE_COLUMNS:
        out[c] = profiles[c].to_numpy()

    seg_path = processed / "sku_segments.parquet"
    ensure_dir(seg_path)
    out.to_parquet(seg_path, index=False)
    print(f"    saved -> data/processed/sku_segments.parquet ({len(out)} SKUs)")

    for _, r in summary.iterrows():
        print(f"    segment {int(r['segment'])}: {r['segment_name']:<28} "
              f"{int(r['n_skus']):>3} SKUs, mean demand {r['mean_demand']:7.1f}/day, "
              f"{r['revenue_share_total']:.1%} of revenue")

    find["segments"] = [
        {
            "segment": int(r["segment"]),
            "name": r["segment_name"],
            "n_skus": int(r["n_skus"]),
            "mean_demand": float(r["mean_demand"]),
            "cv_demand": float(r["cv_demand"]),
            "adi": float(r["adi"]),
            "q4_ratio": float(r["q4_ratio"]),
            "revenue_share": float(r["revenue_share_total"]),
        }
        for _, r in summary.iterrows()
    ]

    path = reports / "unit3_findings.json"
    path.write_text(json.dumps(find, indent=2, default=str), encoding="utf-8")
    print(f"\n    saved -> reports/unit3_findings.json")
    print("\nUnit 3 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
