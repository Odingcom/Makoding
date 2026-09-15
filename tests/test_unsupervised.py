import numpy as np
import pandas as pd
import pytest

from makoding.unsupervised import (
    NotFittedError,
    standardize_features,
    fit_kmeans,
    fit_agglomerative,
    fit_dbscan,
    apply_pca,
    silhouette_score,
    kmeans_inertia,
    kmeans_elbow,
    cluster_summary,
    k_distance_analysis,
)


@pytest.fixture
def clustering_data():
    X = pd.DataFrame(
        {
            "feature_1": [
                1.0, 1.2, 0.8, 1.1, 0.9,
                8.0, 8.2, 7.8, 8.1, 7.9,
                15.0, 15.2, 14.8, 15.1, 14.9,
            ],
            "feature_2": [
                1.0, 0.8, 1.2, 0.9, 1.1,
                8.0, 7.8, 8.2, 7.9, 8.1,
                15.0, 14.8, 15.2, 14.9, 15.1,
            ],
        }
    )
    return X


@pytest.fixture
def clustered_data():
    return pd.DataFrame(
        {
            "feature_1": [
                1.0, 1.2, 0.8, 1.1, 0.9,
                8.0, 8.2, 7.8, 8.1, 7.9,
            ],
            "feature_2": [
                1.0, 0.8, 1.2, 0.9, 1.1,
                8.0, 7.8, 8.2, 7.9, 8.1,
            ],
            "cluster": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_standardize_features(clustering_data):
    result = standardize_features(clustering_data)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == clustering_data.shape
    assert list(result.columns) == list(clustering_data.columns)
    assert np.allclose(result.mean(), 0.0, atol=1e-10)
    assert np.allclose(result.std(ddof=0), 1.0, atol=1e-10)


def test_standardize_features_rejects_non_numeric():
    data = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Nairobi", "Kisumu", "Mombasa"],
        }
    )

    with pytest.raises(ValueError):
        standardize_features(data)


def test_fit_kmeans(clustering_data):
    result = fit_kmeans(
        clustering_data,
        n_clusters=3,
        random_state=42,
    )

    assert hasattr(result, "model")
    assert hasattr(result, "labels")
    assert len(result.labels) == len(clustering_data)
    assert len(np.unique(result.labels)) == 3


def test_fit_kmeans_reproducible(clustering_data):
    first = fit_kmeans(
        clustering_data,
        n_clusters=3,
        random_state=42,
    )
    second = fit_kmeans(
        clustering_data,
        n_clusters=3,
        random_state=42,
    )

    assert np.array_equal(first.labels, second.labels)


def test_fit_kmeans_invalid_clusters(clustering_data):
    with pytest.raises(ValueError):
        fit_kmeans(clustering_data, n_clusters=1)

    with pytest.raises(ValueError):
        fit_kmeans(clustering_data, n_clusters=len(clustering_data) + 1)


def test_fit_agglomerative(clustering_data):
    result = fit_agglomerative(
        clustering_data,
        n_clusters=3,
    )

    assert hasattr(result, "model")
    assert hasattr(result, "labels")
    assert len(result.labels) == len(clustering_data)
    assert len(np.unique(result.labels)) == 3


def test_fit_agglomerative_invalid_clusters(clustering_data):
    with pytest.raises(ValueError):
        fit_agglomerative(clustering_data, n_clusters=1)

    with pytest.raises(ValueError):
        fit_agglomerative(
            clustering_data,
            n_clusters=len(clustering_data) + 1,
        )


def test_fit_dbscan(clustering_data):
    result = fit_dbscan(
        clustering_data,
        eps=1.0,
        min_samples=2,
    )

    assert hasattr(result, "model")
    assert hasattr(result, "labels")
    assert len(result.labels) == len(clustering_data)
    assert len(np.unique(result.labels)) >= 1


def test_fit_dbscan_identifies_noise():
    data = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 10.0],
            "y": [0.0, 0.1, 0.2, 10.0],
        }
    )

    result = fit_dbscan(
        data,
        eps=0.5,
        min_samples=2,
    )

    assert -1 in result.labels


def test_fit_dbscan_invalid_parameters(clustering_data):
    with pytest.raises(ValueError):
        fit_dbscan(clustering_data, eps=0)

    with pytest.raises(ValueError):
        fit_dbscan(clustering_data, min_samples=0)


def test_apply_pca(clustering_data):
    result = apply_pca(
        clustering_data,
        n_components=1,
        random_state=42,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(clustering_data), 1)
    assert result.columns.tolist() == ["PC1"]


def test_apply_pca_multiple_components(clustering_data):
    result = apply_pca(
        clustering_data,
        n_components=2,
    )

    assert result.shape == clustering_data.shape
    assert result.columns.tolist() == ["PC1", "PC2"]


def test_apply_pca_invalid_components(clustering_data):
    with pytest.raises(ValueError):
        apply_pca(clustering_data, n_components=0)

    with pytest.raises(ValueError):
        apply_pca(
            clustering_data,
            n_components=len(clustering_data.columns) + 1,
        )


def test_silhouette_score(clustering_data):
    result = fit_kmeans(
        clustering_data,
        n_clusters=3,
        random_state=42,
    )

    score = silhouette_score(
        clustering_data,
        result.labels,
    )

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    assert score > 0.8


def test_silhouette_score_rejects_invalid_labels(clustering_data):
    labels = np.zeros(len(clustering_data), dtype=int)

    with pytest.raises(ValueError):
        silhouette_score(clustering_data, labels)


def test_kmeans_inertia(clustering_data):
    result = fit_kmeans(
        clustering_data,
        n_clusters=3,
        random_state=42,
    )

    inertia = kmeans_inertia(result)

    assert isinstance(inertia, float)
    assert inertia >= 0


def test_kmeans_elbow(clustering_data):
    result = kmeans_elbow(
        clustering_data,
        k_range=range(2, 6),
        random_state=42,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["n_clusters", "inertia"]
    assert len(result) == 4
    assert result["n_clusters"].tolist() == [2, 3, 4, 5]
    assert (result["inertia"] >= 0).all()


def test_cluster_summary(clustered_data):
    result = cluster_summary(
        clustered_data,
        cluster_column="cluster",
    )

    assert isinstance(result, pd.DataFrame)
    assert "cluster" in result.columns
    assert "count" in result.columns
    assert result["count"].sum() == len(clustered_data)


def test_cluster_summary_numeric_statistics(clustered_data):
    result = cluster_summary(
        clustered_data,
        cluster_column="cluster",
    )

    assert "feature_1_mean" in result.columns
    assert "feature_2_mean" in result.columns
    assert "feature_1_std" in result.columns
    assert "feature_2_std" in result.columns


def test_k_distance_analysis(clustering_data):
    result = k_distance_analysis(
        clustering_data,
        k=3,
    )

    assert isinstance(result, pd.DataFrame)
    assert "distance" in result.columns
    assert len(result) == len(clustering_data)
    assert result["distance"].is_monotonic_increasing


def test_k_distance_analysis_invalid_k(clustering_data):
    with pytest.raises(ValueError):
        k_distance_analysis(clustering_data, k=0)

    with pytest.raises(ValueError):
        k_distance_analysis(
            clustering_data,
            k=len(clustering_data),
        )


def test_empty_dataframe_rejected():
    empty = pd.DataFrame()

    with pytest.raises(ValueError):
        standardize_features(empty)


def test_missing_values_rejected(clustering_data):
    data = clustering_data.copy()
    data.loc[0, "feature_1"] = np.nan

    with pytest.raises(ValueError):
        fit_kmeans(data, n_clusters=3)


def test_cluster_summary_missing_cluster_column(clustered_data):
    with pytest.raises(ValueError):
        cluster_summary(
            clustered_data,
            cluster_column="missing_cluster",
        )