"""Unit 3 — unsupervised learning on the SKU demand profiles.

Everything the syllabus lists for Unit 3 is implemented here from first
principles rather than imported from scikit-learn:

    dissimilarity.py    measuring dissimilarity (the metric choice itself)
    kmeans.py           K-means (Lloyd + k-means++), elbow method
    gmm.py              mixtures of Gaussians, EM, the alternative view of EM
    factor_analysis.py  factor analysis (EM), varimax, rotational ambiguity
    pca.py              PCA, scree, choosing the number of latent dimensions
    hierarchical.py     agglomerative clustering, linkage, dendrogram
    evaluation.py       evaluating the output of clustering methods
    profiles.py         builds the SKU x feature matrix everything runs on

The from-scratch implementations are the deliverable; the test-suite pins
each of them against scikit-learn / SciPy so "from scratch" does not quietly
become "from scratch and wrong".
"""

from __future__ import annotations

from .dissimilarity import (
    DISSIMILARITIES,
    correlation_distance,
    cosine_distance,
    dissimilarity_matrix,
    euclidean_distance,
    manhattan_distance,
    standardize,
)
from .evaluation import (
    adjusted_rand_index,
    calinski_harabasz,
    davies_bouldin,
    evaluate_clustering,
    silhouette_samples,
    silhouette_score,
)
from .factor_analysis import FactorAnalysisEM, varimax
from .gmm import GaussianMixtureEM, em_lower_bound
from .hierarchical import AgglomerativeClustering, cophenetic_correlation, fcluster
from .kmeans import KMeans, elbow_curve, knee_point
from .pca import PCA, choose_n_components, scree_table
from .profiles import build_sku_profiles

__all__ = [
    "DISSIMILARITIES",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "correlation_distance",
    "dissimilarity_matrix",
    "standardize",
    "KMeans",
    "elbow_curve",
    "knee_point",
    "GaussianMixtureEM",
    "em_lower_bound",
    "FactorAnalysisEM",
    "varimax",
    "PCA",
    "scree_table",
    "choose_n_components",
    "AgglomerativeClustering",
    "fcluster",
    "cophenetic_correlation",
    "silhouette_score",
    "silhouette_samples",
    "davies_bouldin",
    "calinski_harabasz",
    "adjusted_rand_index",
    "evaluate_clustering",
    "build_sku_profiles",
]
