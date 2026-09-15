"""
Unsupervised learning utilities for DataLab Pro.

This module provides reusable APIs for:

- Feature standardization
- K-Means clustering
- Agglomerative clustering
- DBSCAN clustering
- Principal Component Analysis (PCA)
- Silhouette analysis
- K-Means inertia / elbow analysis
- Cluster profiling
- DBSCAN k-distance analysis

The functions are intentionally framework-independent so they can be used
from Streamlit, notebooks, scripts, tests, or other applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NotFittedError(RuntimeError):
    """Raised when an operation requires a fitted model."""


@dataclass
class ClusteringResult:
    """Container for a fitted clustering model and its labels."""

    model: object
    labels: np.ndarray


@dataclass
class PCAResult:
    """Container for PCA output and fitted PCA metadata."""

    model: PCA
    transformed: pd.DataFrame
    explained_variance_ratio: np.ndarray


def _validate_dataframe(
    frame: pd.DataFrame,
    *,
    require_numeric: bool = True,
) -> None:
    """Validate a DataFrame before numerical unsupervised learning."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("frame must not be empty.")

    if frame.shape[1] == 0:
        raise ValueError("frame must contain at least one column.")

    if frame.isnull().any().any():
        raise ValueError("frame must not contain missing values.")

    if require_numeric:
        non_numeric = frame.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise ValueError(
                "All features must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )


def _validate_cluster_count(n_clusters: int, n_samples: int) -> None:
    """Validate a requested number of clusters."""
    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer.")

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= n_samples:
        raise ValueError(
            "n_clusters must be smaller than the number of samples."
        )


def _validate_finite_values(frame: pd.DataFrame) -> None:
    """Reject infinite numerical values."""
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("frame must contain only finite numeric values.")


def standardize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features using StandardScaler.

    Parameters
    ----------
    frame:
        Numeric feature DataFrame.

    Returns
    -------
    pandas.DataFrame
        Standardized features with the original index and column names.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    scaler = StandardScaler()
    transformed = scaler.fit_transform(frame)

    return pd.DataFrame(
        transformed,
        index=frame.index,
        columns=frame.columns,
    )


def fit_kmeans(
    frame: pd.DataFrame,
    *,
    n_clusters: int = 3,
    random_state: int | None = 42,
    n_init: int | str = 10,
    **kwargs,
) -> ClusteringResult:
    """
    Fit a K-Means clustering model.

    Parameters
    ----------
    frame:
        Numeric feature DataFrame.
    n_clusters:
        Number of clusters.
    random_state:
        Random seed.
    n_init:
        Number of centroid initializations.
    **kwargs:
        Additional KMeans parameters.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)
    _validate_cluster_count(n_clusters, len(frame))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        **kwargs,
    )

    labels = model.fit_predict(frame)

    return ClusteringResult(
        model=model,
        labels=np.asarray(labels),
    )


def fit_agglomerative(
    frame: pd.DataFrame,
    *,
    n_clusters: int = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
    **kwargs,
) -> ClusteringResult:
    """
    Fit an agglomerative hierarchical clustering model.

    Parameters
    ----------
    frame:
        Numeric feature DataFrame.
    n_clusters:
        Number of clusters.
    linkage:
        Linkage criterion.
    metric:
        Distance metric supported by the selected linkage method.
    **kwargs:
        Additional AgglomerativeClustering parameters.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)
    _validate_cluster_count(n_clusters, len(frame))

    if linkage == "ward" and metric != "euclidean":
        raise ValueError(
            "linkage='ward' requires metric='euclidean'."
        )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
        **kwargs,
    )

    labels = model.fit_predict(frame)

    return ClusteringResult(
        model=model,
        labels=np.asarray(labels),
    )


def fit_dbscan(
    frame: pd.DataFrame,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    **kwargs,
) -> ClusteringResult:
    """
    Fit a DBSCAN clustering model.

    DBSCAN labels noise observations as ``-1``.

    Parameters
    ----------
    frame:
        Numeric feature DataFrame.
    eps:
        Maximum neighborhood radius.
    min_samples:
        Minimum number of samples required to form a dense region.
    metric:
        Distance metric.
    **kwargs:
        Additional DBSCAN parameters.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    if not isinstance(min_samples, (int, np.integer)):
        raise TypeError("min_samples must be an integer.")

    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        **kwargs,
    )

    labels = model.fit_predict(frame)

    return ClusteringResult(
        model=model,
        labels=np.asarray(labels),
    )


def apply_pca(
    frame: pd.DataFrame,
    *,
    n_components: int | float | None = 2,
    random_state: int | None = 42,
    prefix: str = "PC",
    **kwargs,
) -> pd.DataFrame:
    """
    Apply Principal Component Analysis.

    Parameters
    ----------
    frame:
        Numeric feature DataFrame.
    n_components:
        Number or proportion of components.
    random_state:
        Random seed where applicable.
    prefix:
        Prefix used for transformed component names.
    **kwargs:
        Additional PCA parameters.

    Returns
    -------
    pandas.DataFrame
        PCA-transformed features.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    n_features = frame.shape[1]

    if isinstance(n_components, (int, np.integer)):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        if n_components > n_features:
            raise ValueError(
                "n_components cannot exceed the number of features."
            )

    elif isinstance(n_components, float):
        if not 0 < n_components <= 1:
            raise ValueError(
                "Float n_components must be between 0 and 1."
            )

    elif n_components is not None:
        raise TypeError(
            "n_components must be an integer, float, or None."
        )

    model = PCA(
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )

    transformed = model.fit_transform(frame)

    columns = [
        f"{prefix}{index + 1}"
        for index in range(transformed.shape[1])
    ]

    return pd.DataFrame(
        transformed,
        index=frame.index,
        columns=columns,
    )


def silhouette_score(
    frame: pd.DataFrame,
    labels: Iterable[int],
    *,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """
    Calculate the mean silhouette coefficient.

    Silhouette scores range from -1 to 1, where larger values generally
    indicate better-defined clusters.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    labels_array = np.asarray(labels)

    if len(labels_array) != len(frame):
        raise ValueError(
            "labels must contain one value for every observation."
        )

    unique_labels = np.unique(labels_array)

    if len(unique_labels) < 2:
        raise ValueError(
            "Silhouette score requires at least two clusters."
        )

    if len(unique_labels) >= len(frame):
        raise ValueError(
            "Silhouette score requires fewer clusters than observations."
        )

    return float(
        sklearn_silhouette_score(
            frame,
            labels_array,
            metric=metric,
            **kwargs,
        )
    )


def kmeans_inertia(result: ClusteringResult) -> float:
    """
    Return the inertia from a fitted K-Means result.
    """
    if not isinstance(result, ClusteringResult):
        raise TypeError(
            "result must be a ClusteringResult."
        )

    model = result.model

    if not hasattr(model, "inertia_"):
        raise ValueError(
            "The supplied result does not contain a K-Means model."
        )

    return float(model.inertia_)


def kmeans_elbow(
    frame: pd.DataFrame,
    *,
    k_range: Iterable[int] = range(2, 11),
    random_state: int | None = 42,
    n_init: int | str = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate K-Means inertia across a range of cluster counts.

    Returns a DataFrame with columns:

    - ``n_clusters``
    - ``inertia``
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    values = list(k_range)

    if not values:
        raise ValueError("k_range must contain at least one value.")

    rows = []

    for k in values:
        _validate_cluster_count(k, len(frame))

        result = fit_kmeans(
            frame,
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            **kwargs,
        )

        rows.append(
            {
                "n_clusters": int(k),
                "inertia": kmeans_inertia(result),
            }
        )

    return pd.DataFrame(rows)


def cluster_summary(
    frame: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """
    Generate a numerical profile for each cluster.

    The output contains:

    - cluster identifier
    - observation count
    - mean for each numeric feature
    - standard deviation for each numeric feature
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("frame must not be empty.")

    if cluster_column not in frame.columns:
        raise ValueError(
            f"Cluster column '{cluster_column}' was not found."
        )

    if frame[cluster_column].isnull().any():
        raise ValueError(
            "The cluster column must not contain missing values."
        )

    feature_columns = [
        column
        for column in frame.select_dtypes(include=np.number).columns
        if column != cluster_column
    ]

    if not feature_columns:
        raise ValueError(
            "At least one numeric feature is required for profiling."
        )

    grouped = frame.groupby(
        cluster_column,
        sort=True,
        dropna=False,
    )

    counts = grouped.size().rename("count")

    means = grouped[feature_columns].mean().add_suffix("_mean")
    stds = grouped[feature_columns].std().fillna(0).add_suffix("_std")

    result = pd.concat(
        [
            counts,
            means,
            stds,
        ],
        axis=1,
    ).reset_index()

    return result


def k_distance_analysis(
    frame: pd.DataFrame,
    *,
    k: int = 5,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Calculate each observation's distance to its k-th nearest neighbor.

    The returned distances are sorted ascending, which makes the result
    suitable for identifying an approximate DBSCAN ``eps`` value.
    """
    _validate_dataframe(frame)
    _validate_finite_values(frame)

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if k >= len(frame):
        raise ValueError(
            "k must be smaller than the number of observations."
        )

    neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
    )

    distances, _ = neighbors.fit(frame).kneighbors(frame)

    kth_distances = distances[:, k]

    return pd.DataFrame(
        {
            "distance": np.sort(kth_distances),
        }
    )


__all__ = [
    "ClusteringResult",
    "NotFittedError",
    "PCAResult",
    "apply_pca",
    "cluster_summary",
    "fit_agglomerative",
    "fit_dbscan",
    "fit_kmeans",
    "k_distance_analysis",
    "kmeans_elbow",
    "kmeans_inertia",
    "silhouette_score",
    "standardize_features",
]