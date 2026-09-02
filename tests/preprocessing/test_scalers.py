"""StandardScaler 和 MinMaxScaler 测试。

测试目录镜像 minisklearn/preprocessing/ 的结构。
"""

import numpy as np
import pytest
from minisklearn.preprocessing import StandardScaler, MinMaxScaler
from minisklearn.base import clone


class TestStandardScaler:
    """StandardScaler 测试。"""

    def test_fit_transform_basic(self):
        scaler = StandardScaler()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0)
        assert np.allclose(X_scaled.std(axis=0), 1)

    def test_mean_and_scale_stored(self):
        scaler = StandardScaler()
        X = np.array([[1], [2], [3], [4]])
        scaler.fit(X)
        assert np.isclose(scaler.mean_[0], 2.5)
        assert np.isclose(scaler.scale_[0], np.std(X))

    def test_transform_new_data(self):
        scaler = StandardScaler()
        X_train = np.array([[1], [2], [3]])
        scaler.fit(X_train)
        X_new = np.array([[4]])
        result = scaler.transform(X_new)
        expected = (4 - 2) / np.std(X_train)
        assert np.isclose(result[0, 0], expected)

    def test_inverse_transform(self):
        scaler = StandardScaler()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)
        assert np.allclose(X, X_restored)

    def test_with_mean_false(self):
        scaler = StandardScaler(with_mean=False)
        X = np.array([[1], [2], [3]])
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(scaler.mean_, 0)
        assert not np.allclose(X_scaled.mean(axis=0), 0)

    def test_with_std_false(self):
        scaler = StandardScaler(with_std=False)
        X = np.array([[1], [2], [3]])
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(scaler.scale_, 1)
        assert np.allclose(X_scaled.mean(axis=0), 0)

    def test_constant_feature(self):
        """常量特征（std=0）不应除零。"""
        scaler = StandardScaler()
        X = np.array([[1, 5], [2, 5], [3, 5]])
        X_scaled = scaler.fit_transform(X)
        assert not np.any(np.isnan(X_scaled))
        assert np.allclose(X_scaled[:, 1], 0)

    def test_clone(self):
        scaler = StandardScaler(with_mean=False)
        cloned = clone(scaler)
        assert cloned.with_mean == False
        assert not hasattr(cloned, "mean_")

    def test_repr(self):
        scaler = StandardScaler()
        assert "StandardScaler" in repr(scaler)


class TestMinMaxScaler:
    """MinMaxScaler 测试。"""

    def test_fit_transform_default_range(self):
        scaler = MinMaxScaler()
        X = np.array([[1], [2], [3], [4]])
        X_scaled = scaler.fit_transform(X)
        assert np.isclose(X_scaled.min(), 0)
        assert np.isclose(X_scaled.max(), 1)

    def test_custom_range(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = np.array([[1], [2], [3], [4]])
        X_scaled = scaler.fit_transform(X)
        assert np.isclose(X_scaled.min(), -1)
        assert np.isclose(X_scaled.max(), 1)

    def test_inverse_transform(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)
        assert np.allclose(X, X_restored)

    def test_constant_feature(self):
        """常量特征（range=0）不应除零。"""
        scaler = MinMaxScaler()
        X = np.array([[1, 5], [2, 5], [3, 5]])
        X_scaled = scaler.fit_transform(X)
        assert not np.any(np.isnan(X_scaled))

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            scaler = MinMaxScaler(feature_range=(1, 0))
            scaler.fit([[1], [2]])

    def test_clone(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        cloned = clone(scaler)
        assert cloned.feature_range == (-1, 1)