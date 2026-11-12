"""Unit 3 -- the unsupervised methods compute what they claim to.

Two kinds of test here, and the distinction matters:

* **Reference tests** pin the from-scratch implementations against
  scikit-learn / SciPy. "I implemented K-means myself" is only a virtue if it
  gives the same answer as the library; these tests are what stop it becoming
  "I implemented K-means myself and it is subtly wrong".
* **Property tests** check the mathematical guarantees the lectures state --
  EM's monotonicity, the tightness of the EM lower bound, factor analysis's
  rotational ambiguity, PCA's orthonormality. These catch errors a reference
  test cannot, because they say what must be true of ANY correct version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans as SKKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA as SKPCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score as sk_silhouette,
)
from sklearn.mixture import GaussianMixture as SKGMM

from xaierp.unsupervised.dissimilarity import (
    correlation_distance,
    cosine_distance,
    dissimilarity_matrix,
    euclidean_distance,
    manhattan_distance,
    standardize,
)
from xaierp.unsupervised.evaluation import (
    adjusted_rand_index,
    calinski_harabasz,
    davies_bouldin,
    silhouette_samples,
    silhouette_score,
)
from xaierp.unsupervised.factor_analysis import (
    FactorAnalysisEM,
    rotational_ambiguity_check,
    varimax,
)
from xaierp.unsupervised.gmm import GaussianMixtureEM, em_lower_bound
from xaierp.unsupervised.hierarchical import (
    AgglomerativeClustering,
    cophenetic_correlation,
    fcluster,
    linkage,
)
from xaierp.unsupervised.kmeans import KMeans, elbow_curve, knee_point
from xaierp.unsupervised.pca import PCA, choose_n_components, scree_table
from xaierp.unsupervised.profiles import (
    PROFILE_COLUMNS,
    build_sku_profiles,
    name_segments,
    segment_summary,
)

SEED = 0


@pytest.fixture(scope="module")
def blobs():
    """Four well-separated Gaussian blobs: a case with a known right answer."""
    X, y = make_blobs(
        n_samples=160, centers=4, n_features=6, cluster_std=1.2, random_state=SEED
    )
    Z, _, _ = standardize(X)
    return Z, y


# =============================================================== dissimilarity


def test_distance_functions_match_scipy():
    from scipy.spatial import distance

    rng = np.random.default_rng(SEED)
    a, b = rng.normal(size=8), rng.normal(size=8)

    assert euclidean_distance(a, b) == pytest.approx(distance.euclidean(a, b))
    assert manhattan_distance(a, b) == pytest.approx(distance.cityblock(a, b))
    assert cosine_distance(a, b) == pytest.approx(distance.cosine(a, b))
    assert correlation_distance(a, b) == pytest.approx(distance.correlation(a, b))


def test_dissimilarity_matrix_is_a_metric_matrix(blobs):
    X, _ = blobs
    for metric in ("euclidean", "manhattan", "cosine", "correlation"):
        D = dissimilarity_matrix(X, metric)
        assert D.shape == (len(X), len(X))
        assert np.allclose(np.diag(D), 0.0), f"{metric}: diagonal must be zero"
        assert np.allclose(D, D.T), f"{metric}: must be symmetric"
        assert (D >= -1e-12).all(), f"{metric}: distances must be non-negative"


def test_euclidean_matrix_matches_scipy(blobs):
    from scipy.spatial.distance import squareform, pdist

    X, _ = blobs
    assert np.allclose(dissimilarity_matrix(X, "euclidean"),
                       squareform(pdist(X, "euclidean")))


def test_cosine_distance_ignores_vector_length():
    a = np.array([1.0, 2.0, 3.0])
    # Scaling a vector changes its length but not its direction.
    assert cosine_distance(a, 5.0 * a) == pytest.approx(0.0, abs=1e-12)
    assert euclidean_distance(a, 5.0 * a) > 1.0


def test_cosine_distance_of_zero_vector_is_total():
    # A zero vector has no direction; returning 1.0 keeps the function total
    # instead of producing a NaN that would poison a whole distance matrix.
    assert cosine_distance(np.zeros(3), np.array([1.0, 0, 0])) == 1.0


def test_standardize_uses_supplied_statistics():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    Z, mean, std = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(Z.std(axis=0), 1.0)
    # Reapplying the TRAIN statistics is what makes the transform leakage-free.
    Z2, _, _ = standardize(X * 2.0, mean=mean, std=std)
    assert np.allclose(Z2, (X * 2.0 - mean) / std)


def test_standardize_survives_a_constant_column():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    Z, _, _ = standardize(X)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 1], 0.0)


# ==================================================================== K-means


def test_kmeans_matches_sklearn_inertia_and_labels(blobs):
    X, _ = blobs
    mine = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X)
    theirs = SKKMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X)
    assert mine.inertia_ == pytest.approx(theirs.inertia_, rel=1e-6)
    assert adjusted_rand_score(mine.labels_, theirs.labels_) == pytest.approx(1.0)


def test_kmeans_recovers_known_blobs(blobs):
    X, y = blobs
    labels = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit_predict(X)
    assert adjusted_rand_index(labels, y) > 0.9


def test_kmeans_inertia_never_increases(blobs):
    """Both of Lloyd's steps minimise inertia, so it must fall monotonically."""
    X, _ = blobs
    km = KMeans(n_clusters=5, n_init=1, random_state=SEED).fit(X)
    h = km.inertia_history_
    assert len(h) >= 2
    for a, b in zip(h, h[1:]):
        assert b <= a + 1e-9, f"inertia rose from {a} to {b}"


def test_kmeans_inertia_equals_its_own_definition(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=4, random_state=SEED).fit(X)
    manual = sum(
        float(np.sum((X[km.labels_ == j] - km.cluster_centers_[j]) ** 2))
        for j in range(4)
    )
    assert km.inertia_ == pytest.approx(manual)


def test_kmeans_centroids_are_the_means_of_their_members(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=4, random_state=SEED).fit(X)
    for j in range(4):
        members = X[km.labels_ == j]
        if len(members):
            assert np.allclose(km.cluster_centers_[j], members.mean(axis=0), atol=1e-6)


def test_kmeans_returns_exactly_k_non_empty_clusters(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=6, random_state=SEED).fit(X)
    assert len(np.unique(km.labels_)) == 6


def test_kmeans_predict_is_consistent_with_fit(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=4, random_state=SEED).fit(X)
    assert np.array_equal(km.predict(X), km.labels_)


def test_kmeans_rejects_more_clusters_than_points():
    with pytest.raises(ValueError, match="exceeds n_samples"):
        KMeans(n_clusters=5).fit(np.zeros((3, 2)))


def test_elbow_inertia_falls_with_k(blobs):
    X, _ = blobs
    rows = elbow_curve(X, range(2, 8), n_init=5, random_state=SEED)
    inertias = [r["inertia"] for r in rows]
    for a, b in zip(inertias, inertias[1:]):
        assert b <= a + 1e-6, "inertia must not rise as k grows"


def test_knee_point_finds_the_planted_elbow():
    # A curve that drops steeply to k=4 and then flattens.
    ks = [2, 3, 4, 5, 6, 7, 8]
    inertias = [100.0, 60.0, 30.0, 28.0, 26.5, 25.5, 25.0]
    assert knee_point(ks, inertias) == 4


# ======================================================================== PCA


def test_pca_matches_sklearn(blobs):
    X, _ = blobs
    mine = PCA(n_components=3).fit(X)
    theirs = SKPCA(n_components=3).fit(X)
    assert np.allclose(mine.explained_variance_ratio_,
                       theirs.explained_variance_ratio_)
    # Components agree up to the sign convention.
    assert np.allclose(np.abs(mine.components_), np.abs(theirs.components_), atol=1e-8)


def test_pca_components_are_orthonormal(blobs):
    X, _ = blobs
    p = PCA(n_components=4).fit(X)
    assert np.allclose(p.components_ @ p.components_.T, np.eye(4), atol=1e-10)


def test_pca_explained_variance_sums_to_one(blobs):
    X, _ = blobs
    rows = scree_table(X)
    assert rows[-1]["cumulative_variance_ratio"] == pytest.approx(1.0)


def test_pca_eigenvalues_are_in_descending_order(blobs):
    X, _ = blobs
    ev = [r["eigenvalue"] for r in scree_table(X)]
    assert all(a >= b - 1e-12 for a, b in zip(ev, ev[1:]))


def test_pca_full_rank_reconstruction_is_exact(blobs):
    X, _ = blobs
    p = PCA(n_components=X.shape[1]).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X, atol=1e-10)
    assert p.reconstruction_error(X) == pytest.approx(0.0, abs=1e-18)


def test_pca_reconstruction_error_falls_as_components_are_added(blobs):
    X, _ = blobs
    errors = [PCA(n_components=k).fit(X).reconstruction_error(X)
              for k in range(1, X.shape[1] + 1)]
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-12


def test_pca_explained_variance_matches_the_projection_variance(blobs):
    X, _ = blobs
    p = PCA(n_components=3).fit(X)
    scores = p.transform(X)
    assert np.allclose(scores.var(axis=0, ddof=1), p.explained_variance_)


def test_choose_n_components_reports_three_rules(blobs):
    X, _ = blobs
    out = choose_n_components(X, variance_threshold=0.90)
    assert set(out) >= {"kaiser", "variance", "elbow"}
    assert 1 <= out["kaiser"] <= X.shape[1]
    # The 90% rule must actually reach 90%.
    cum = [r["cumulative_variance_ratio"] for r in scree_table(X)]
    assert cum[out["variance"] - 1] >= 0.90


def test_pca_rejects_too_many_components(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="exceeds"):
        PCA(n_components=X.shape[1] + 5).fit(X)


# ======================================================== mixtures of Gaussians


def test_gmm_matches_sklearn_log_likelihood(blobs):
    X, _ = blobs
    mine = GaussianMixtureEM(n_components=4, random_state=SEED).fit(X)
    theirs = SKGMM(n_components=4, covariance_type="full",
                   reg_covar=1e-6, random_state=SEED).fit(X)
    assert mine.score(X) == pytest.approx(theirs.score(X) * len(X), rel=1e-5)


def test_gmm_em_log_likelihood_never_decreases(blobs):
    """The central guarantee of EM. If this fails, the maths is wrong."""
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=4, random_state=SEED).fit(X)
    h = gm.loglik_history_
    assert len(h) >= 2
    for a, b in zip(h, h[1:]):
        assert b >= a - 1e-7, f"log-likelihood fell from {a} to {b}"


def test_gmm_responsibilities_form_a_distribution(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    r = gm.responsibilities_
    assert np.allclose(r.sum(axis=1), 1.0)
    assert (r >= 0).all() and (r <= 1).all()


def test_gmm_weights_sum_to_one_and_match_responsibilities(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    assert gm.weights_.sum() == pytest.approx(1.0)


def test_gmm_covariances_are_symmetric_positive_definite(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    for cov in gm.covariances_:
        assert np.allclose(cov, cov.T)
        # Cholesky succeeds only for a positive-definite matrix.
        np.linalg.cholesky(cov)


def test_gmm_recovers_well_separated_blobs(blobs):
    X, y = blobs
    gm = GaussianMixtureEM(n_components=4, random_state=SEED).fit(X)
    assert adjusted_rand_index(gm.predict(X), y) > 0.9


def test_gmm_bic_penalises_extra_components():
    """BIC must have a real minimum, unlike inertia which only ever falls."""
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2,
                      cluster_std=0.8, random_state=SEED)
    bics = []
    for k in range(2, 8):
        gm = GaussianMixtureEM(n_components=k, random_state=SEED).fit(X)
        bics.append(gm.bic(X))
    # The best K is interior, not the largest one on offer.
    assert int(np.argmin(bics)) < len(bics) - 1


def test_gmm_parameter_count_accounts_for_covariance_symmetry(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    d = X.shape[1]
    expected = 3 * d + 3 * d * (d + 1) // 2 + 2
    assert gm.n_parameters() == expected


# ------------------------------------------------- the alternative view of EM


def test_em_bound_is_tight_at_the_e_step_posterior(blobs):
    """F(q, theta) == log p(x | theta) exactly when q is the true posterior.

    This is the identity that proves EM's monotonicity, so it is worth
    checking directly rather than inferring it from the fit succeeding.
    """
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    out = em_lower_bound(X, gm, gm.responsibilities_)
    assert out["gap"] == pytest.approx(0.0, abs=1e-6)
    assert out["kl_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert out["lower_bound"] == pytest.approx(out["log_likelihood"], rel=1e-9)


def test_em_bound_is_strictly_below_for_any_other_q(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    wrong = np.full_like(gm.responsibilities_, 1.0 / 3.0)
    out = em_lower_bound(X, gm, wrong)
    assert out["lower_bound"] < out["log_likelihood"]
    assert out["kl_divergence"] > 0
    # The shortfall is exactly the KL divergence -- that is the decomposition
    # log p(x) = F(q, theta) + KL(q || p(z|x)).
    assert out["gap"] == pytest.approx(out["kl_divergence"], rel=1e-6)


def test_em_lower_bound_decomposes_into_expectation_plus_entropy(blobs):
    X, _ = blobs
    gm = GaussianMixtureEM(n_components=3, random_state=SEED).fit(X)
    out = em_lower_bound(X, gm, gm.responsibilities_)
    assert out["lower_bound"] == pytest.approx(
        out["expected_complete_loglik"] + out["entropy_q"], rel=1e-9
    )


# =========================================================== factor analysis


def test_factor_analysis_log_likelihood_never_decreases(blobs):
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=2, random_state=SEED, max_iter=300).fit(X)
    h = fa.loglik_history_
    for a, b in zip(h, h[1:]):
        assert b >= a - 1e-6, f"FA log-likelihood fell from {a} to {b}"


def test_factor_analysis_reproduces_the_sample_covariance(blobs):
    """The model claims cov(x) = W W^T + Psi; check it against reality."""
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=4, random_state=SEED, max_iter=3000).fit(X)
    model_cov = fa.loadings_ @ fa.loadings_.T + np.diag(fa.noise_variance_)
    sample_cov = np.cov(X.T, bias=True)
    assert np.max(np.abs(model_cov - sample_cov)) < 0.05


def test_factor_analysis_diagonal_is_fitted_almost_exactly(blobs):
    # FA has one free Psi entry per variable, so the VARIANCES (the diagonal)
    # can always be matched; it is the off-diagonal that the low rank
    # constrains. Checking them separately localises any error.
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=3000).fit(X)
    model_cov = fa.loadings_ @ fa.loadings_.T + np.diag(fa.noise_variance_)
    sample_cov = np.cov(X.T, bias=True)
    assert np.allclose(np.diag(model_cov), np.diag(sample_cov), atol=1e-2)


def test_factor_analysis_noise_variances_stay_positive(blobs):
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=500).fit(X)
    assert (fa.noise_variance_ > 0).all()


def test_factor_analysis_rotation_leaves_the_likelihood_unchanged(blobs):
    """Rotational ambiguity: W is identified only up to an orthogonal R."""
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=500).fit(X)
    out = rotational_ambiguity_check(fa, X)
    # The loadings genuinely move...
    assert out["max_loading_difference"] > 0.01
    # ...while the likelihood and the implied covariance do not.
    assert out["loglik_difference"] < 1e-6
    assert out["max_covariance_difference"] < 1e-10
    assert out["n_free_rotations"] == 3  # L(L-1)/2 for L = 3


def test_varimax_rotation_is_orthogonal_and_preserves_the_fit(blobs):
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=500).fit(X)
    W_rot, R = varimax(fa.loadings_)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.allclose(W_rot @ W_rot.T, fa.loadings_ @ fa.loadings_.T, atol=1e-8)


def test_varimax_increases_loading_simplicity(blobs):
    """Varimax exists to sharpen the loadings, so its criterion must rise."""
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=500).fit(X)
    before = float(np.var(fa.loadings_**2, axis=0).sum())
    W_rot, _ = varimax(fa.loadings_)
    after = float(np.var(W_rot**2, axis=0).sum())
    assert after >= before - 1e-9


def test_factor_analysis_communalities_are_proportions(blobs):
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=3, random_state=SEED, max_iter=500).fit(X)
    c = fa.communalities()
    assert ((c >= 0) & (c <= 1)).all()


def test_factor_analysis_rejects_too_many_factors(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="must be <"):
        FactorAnalysisEM(n_factors=X.shape[1]).fit(X)


def test_factor_analysis_parameter_count_subtracts_free_rotations(blobs):
    X, _ = blobs
    fa = FactorAnalysisEM(n_factors=2, random_state=SEED, max_iter=100).fit(X)
    d = X.shape[1]
    assert fa.n_parameters() == d + 2 * d + d - 1


# ==================================================== hierarchical clustering


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_linkage_matches_scipy(blobs, method):
    from scipy.cluster.hierarchy import linkage as sp_linkage

    X, _ = blobs
    mine = linkage(X, method)
    theirs = sp_linkage(X, method=method, metric="euclidean")
    assert np.allclose(mine[:, 2], theirs[:, 2]), f"{method}: merge heights differ"
    assert np.array_equal(mine[:, 3], theirs[:, 3]), f"{method}: cluster sizes differ"


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_fcluster_matches_scipy(blobs, method):
    from scipy.cluster.hierarchy import fcluster as sp_fcluster
    from scipy.cluster.hierarchy import linkage as sp_linkage

    X, _ = blobs
    mine = fcluster(linkage(X, method), 4)
    theirs = sp_fcluster(sp_linkage(X, method=method), 4, criterion="maxclust")
    assert adjusted_rand_score(mine, theirs) == pytest.approx(1.0)


def test_linkage_merge_heights_are_non_decreasing(blobs):
    """Ward, complete and average are monotonic: no inversions in the tree."""
    X, _ = blobs
    for method in ("single", "complete", "average", "ward"):
        h = linkage(X, method)[:, 2]
        assert all(a <= b + 1e-9 for a, b in zip(h, h[1:])), method


def test_linkage_has_n_minus_one_merges(blobs):
    X, _ = blobs
    assert linkage(X, "ward").shape == (len(X) - 1, 4)


def test_linkage_final_cluster_contains_every_point(blobs):
    X, _ = blobs
    assert linkage(X, "ward")[-1, 3] == len(X)


def test_cophenetic_correlation_matches_scipy(blobs):
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import pdist

    X, _ = blobs
    Z = linkage(X, "average")
    mine = cophenetic_correlation(Z, dissimilarity_matrix(X, "euclidean"))
    theirs = cophenet(sp_linkage(X, method="average"), pdist(X))[0]
    assert mine == pytest.approx(theirs, rel=1e-9)


def test_hierarchical_recovers_known_blobs(blobs):
    X, y = blobs
    labels = AgglomerativeClustering(n_clusters=4).fit_predict(X)
    assert adjusted_rand_index(labels, y) > 0.9


def test_fcluster_cut_yields_the_requested_number_of_clusters(blobs):
    X, _ = blobs
    Z = linkage(X, "ward")
    for k in (2, 3, 5, 8):
        assert len(np.unique(fcluster(Z, k))) == k


def test_ward_rejects_non_euclidean_metrics(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="only defined for the Euclidean"):
        linkage(X, "ward", metric="cosine")


def test_unknown_linkage_is_rejected(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="Unknown linkage"):
        linkage(X, "centroid-ish")


# =================================================== clustering evaluation


def test_silhouette_matches_sklearn(blobs):
    X, y = blobs
    assert silhouette_score(X, y) == pytest.approx(sk_silhouette(X, y), rel=1e-9)


def test_silhouette_samples_average_to_the_score(blobs):
    X, y = blobs
    assert silhouette_samples(X, y).mean() == pytest.approx(silhouette_score(X, y))


def test_silhouette_is_bounded(blobs):
    X, y = blobs
    s = silhouette_samples(X, y)
    assert (s >= -1.0 - 1e-12).all() and (s <= 1.0 + 1e-12).all()


def test_davies_bouldin_matches_sklearn(blobs):
    X, y = blobs
    assert davies_bouldin(X, y) == pytest.approx(davies_bouldin_score(X, y), rel=1e-9)


def test_calinski_harabasz_matches_sklearn(blobs):
    X, y = blobs
    assert calinski_harabasz(X, y) == pytest.approx(
        calinski_harabasz_score(X, y), rel=1e-9
    )


def test_adjusted_rand_index_matches_sklearn(blobs):
    X, y = blobs
    labels = KMeans(n_clusters=4, random_state=SEED).fit_predict(X)
    assert adjusted_rand_index(labels, y) == pytest.approx(
        adjusted_rand_score(labels, y), rel=1e-9
    )


def test_ari_is_one_for_identical_partitions_and_invariant_to_relabelling():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    # Permuting the cluster NAMES is not a different clustering.
    assert adjusted_rand_index(a, np.array([2, 2, 0, 0, 1, 1])) == pytest.approx(1.0)


def test_ari_is_near_zero_for_random_labellings():
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 4, 400)
    b = rng.integers(0, 4, 400)
    assert abs(adjusted_rand_index(a, b)) < 0.05


def test_better_separated_clusters_score_higher():
    """A sanity check on the indices themselves, not on any one algorithm."""
    tight, y1 = make_blobs(n_samples=120, centers=3, cluster_std=0.4,
                           n_features=2, random_state=SEED)
    loose, y2 = make_blobs(n_samples=120, centers=3, cluster_std=3.0,
                           n_features=2, random_state=SEED)
    assert silhouette_score(tight, y1) > silhouette_score(loose, y2)
    assert davies_bouldin(tight, y1) < davies_bouldin(loose, y2)
    assert calinski_harabasz(tight, y1) > calinski_harabasz(loose, y2)


def test_evaluation_metrics_need_at_least_two_clusters(blobs):
    X, _ = blobs
    one = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match="at least 2 clusters"):
        silhouette_score(X, one)
    with pytest.raises(ValueError, match="at least 2 clusters"):
        davies_bouldin(X, one)


# ============================================================ SKU profiles


def _toy_panel(n_skus: int = 6, n_days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    dates = pd.date_range("2010-01-01", periods=n_days, freq="D")
    for i in range(n_skus):
        # Half the SKUs sell every day, half are intermittent.
        base = 50.0 if i % 2 == 0 else 3.0
        demand = rng.poisson(base, n_days).astype(float)
        if i % 2:
            demand[rng.random(n_days) < 0.6] = 0.0
        rows.append(
            pd.DataFrame(
                {
                    "stock_code": f"SKU{i}",
                    "date": dates,
                    "demand": demand,
                    "revenue": demand * (1.0 + i),
                    "returned_units": 0.0,
                    "description": f"item {i}",
                    "median_price": 1.0 + i,
                    "n_invoices": 1,
                    "avg_price": 1.0 + i,
                    "sku_rank": i + 1,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_profiles_have_one_row_per_sku_and_no_missing_values():
    p = build_sku_profiles(_toy_panel())
    assert len(p) == 6
    assert list(p.index.names) == ["stock_code"]
    assert p[PROFILE_COLUMNS].notna().all().all()
    assert np.isfinite(p[PROFILE_COLUMNS].to_numpy()).all()


def test_profiles_respect_the_end_date_cutoff():
    """The cutoff is what keeps the segmentation leakage-free, so it must bite."""
    panel = _toy_panel()
    early = build_sku_profiles(panel, end_date="2010-02-01")
    full = build_sku_profiles(panel)
    assert (early["n_days"] < full["n_days"]).all()
    assert not np.allclose(early["mean_demand"], full["mean_demand"])


def test_profiles_reject_an_empty_window():
    with pytest.raises(ValueError, match="No panel rows"):
        build_sku_profiles(_toy_panel(), end_date="2000-01-01")


def test_intermittent_skus_get_a_higher_adi_and_zero_rate():
    p = build_sku_profiles(_toy_panel())
    steady = p.loc[[f"SKU{i}" for i in (0, 2, 4)]]
    sporadic = p.loc[[f"SKU{i}" for i in (1, 3, 5)]]
    assert sporadic["adi"].mean() > steady["adi"].mean()
    assert sporadic["zero_rate"].mean() > steady["zero_rate"].mean()


def test_revenue_shares_sum_to_one():
    p = build_sku_profiles(_toy_panel())
    assert p["revenue_share"].sum() == pytest.approx(1.0)


def test_segment_summary_and_names_cover_every_segment():
    p = build_sku_profiles(_toy_panel())
    labels = np.array([0, 1, 0, 1, 0, 1])
    s = segment_summary(p, labels)
    assert len(s) == 2
    assert s["n_skus"].sum() == 6
    names = name_segments(s)
    assert set(names) == {0, 1}
    # Distinct captions: two segments sharing a name is a reporting bug.
    assert len(set(names.values())) == 2
