"""PCA 测试。"""

import numpy as np
import pytest
from minisklearn.decomposition import PCA
from minisklearn.base import clone


class TestPCA:
    def test_fit_transform_basic(self):
        X = np.random.RandomState(42).randn(100, 5)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (100, 2)

    def test_components_shape(self):
        X = np.random.RandomState(42).randn(50, 4)
        pca = PCA(n_components=2)
        pca.fit(X)
        assert pca.components_.shape == (2, 4)

    def test_explained_variance(self):
        X = np.random.RandomState(42).randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        assert pca.explained_variance_.shape == (5,)
        assert pca.explained_variance_ratio_.shape == (5,)
        # 保留全部主成分时，方差比例总和应为 1
        assert np.isclose(np.sum(pca.explained_variance_ratio_), 1.0, atol=1e-10)

    def test_variance_ordered(self):
        """解释方差应按降序排列。"""
        X = np.random.RandomState(42).randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        assert np.all(np.diff(pca.explained_variance_) <= 0)

    def test_inverse_transform(self):
        """降维后逆变换应近似恢复（有信息损失）。"""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        pca = PCA(n_components=3)  # 保留全部维度
        X_reduced = pca.fit_transform(X)
        X_recovered = pca.inverse_transform(X_reduced)
        assert np.allclose(X, X_recovered, atol=1e-10)

    def test_mean_removed(self):
        """降维后数据的均值应接近 0。"""
        X = np.random.RandomState(42).randn(100, 5) + 10
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        assert np.allclose(np.mean(X_reduced, axis=0), 0, atol=1e-10)

    def test_orthogonal_components(self):
        """主成分应正交。"""
        X = np.random.RandomState(42).randn(100, 4)
        pca = PCA(n_components=3)
        pca.fit(X)
        # components_ @ components_.T 应接近单位阵
        identity = pca.components_ @ pca.components_.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)

    def test_variance_ratio_auto(self):
        """n_components 为 float 时自动选择维度。"""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        pca = PCA(n_components=0.95)  # 保留 95% 方差
        pca.fit(X)
        assert np.sum(pca.explained_variance_ratio_) >= 0.95

    def test_whiten(self):
        """白化后各主成分方差为 1（ddof=1 样本方差）。"""
        X = np.random.RandomState(42).randn(100, 5)
        pca = PCA(n_components=3, whiten=True)
        X_reduced = pca.fit_transform(X)
        variances = np.var(X_reduced, axis=0, ddof=1)
        assert np.allclose(variances, 1.0, atol=1e-10)

    def test_singular_values(self):
        X = np.random.RandomState(42).randn(100, 5)
        pca = PCA(n_components=3)
        pca.fit(X)
        assert pca.singular_values_.shape == (3,)
        assert np.all(pca.singular_values_ > 0)

    def test_clone(self):
        pca = PCA(n_components=3)
        cloned = clone(pca)
        assert cloned.n_components == 3
        assert not hasattr(cloned, "components_")

    def test_repr(self):
        pca = PCA(n_components=3)
        assert "PCA" in repr(pca)

    def test_known_data(self):
        """已知数据的主成分方向。"""
        # 数据沿 x 轴展开，第一主成分应接近 [1, 0]
        X = np.array([[i, 0] for i in range(100)], dtype=float)
        pca = PCA(n_components=1)
        pca.fit(X)
        # 第一主成分方向应接近 [1, 0] 或 [-1, 0]
        comp = pca.components_[0]
        assert np.abs(comp[0]) > 0.99
        assert np.abs(comp[1]) < 0.01