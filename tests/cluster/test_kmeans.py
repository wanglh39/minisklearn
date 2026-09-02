"""KMeans 测试。"""

import numpy as np
import pytest
from minisklearn.cluster import KMeans
from minisklearn.base import clone


def make_blobs(n_per_cluster=50, centers=3, seed=42):
    rng = np.random.RandomState(seed)
    center_coords = rng.randn(centers, 2) * 5
    X = np.vstack([
        rng.randn(n_per_cluster, 2) * 0.5 + center_coords[i]
        for i in range(centers)
    ])
    return X, center_coords


class TestKMeans:
    def test_fit_basic(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_.shape == (3, 2)
        assert kmeans.labels_.shape == (len(X),)

    def test_clusters_separated(self):
        """明显分离的簇应该被正确聚类。"""
        X = np.array([
            [0, 0], [0.1, 0.1], [-0.1, 0.05],
            [10, 10], [10.1, 9.9], [9.95, 10.05],
            [-10, -10], [-10.1, -9.9], [-9.95, -10.05],
        ])
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        # 同簇样本应该有相同标签
        assert kmeans.labels_[0] == kmeans.labels_[1] == kmeans.labels_[2]
        assert kmeans.labels_[3] == kmeans.labels_[4] == kmeans.labels_[5]
        assert kmeans.labels_[6] == kmeans.labels_[7] == kmeans.labels_[8]
        # 不同簇标签不同
        assert len(set([kmeans.labels_[0], kmeans.labels_[3], kmeans.labels_[6]])) == 3

    def test_inertia_nonnegative(self):
        X, _ = make_blobs()
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        assert kmeans.inertia_ >= 0

    def test_predict(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        assert np.array_equal(labels, kmeans.labels_)

    def test_fit_predict(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        assert np.array_equal(labels, kmeans.labels_)

    def test_kmeans_plus_plus_init(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_.shape == (3, 2)

    def test_random_init(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, init="random", random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_.shape == (3, 2)

    def test_reproducibility(self):
        X, _ = make_blobs(centers=3)
        km1 = KMeans(n_clusters=3, random_state=42)
        km1.fit(X)
        km2 = KMeans(n_clusters=3, random_state=42)
        km2.fit(X)
        assert np.allclose(km1.cluster_centers_, km2.cluster_centers_)

    def test_n_init(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_.shape == (3, 2)

    def test_transform(self):
        X, _ = make_blobs(centers=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        distances = kmeans.transform(X)
        assert distances.shape == (len(X), 3)

    def test_clone(self):
        kmeans = KMeans(n_clusters=5)
        cloned = clone(kmeans)
        assert cloned.n_clusters == 5
        assert not hasattr(cloned, "cluster_centers_")

    def test_repr(self):
        kmeans = KMeans(n_clusters=5)
        assert "KMeans" in repr(kmeans)