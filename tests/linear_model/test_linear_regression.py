"""LinearRegression 测试。"""

import numpy as np
import pytest
from minisklearn.linear_model import LinearRegression
from minisklearn.base import clone


class TestLinearRegressionNormal:
    """正规方程求解的测试。"""

    def test_simple_line(self):
        """y = 2x + 1 的数据。"""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 5, 7, 9])
        reg = LinearRegression()
        reg.fit(X, y)
        assert np.isclose(reg.coef_[0], 2.0, atol=1e-10)
        assert np.isclose(reg.intercept_, 1.0, atol=1e-10)

    def test_multivariate(self):
        """y = 2x1 + 3x2 + 0.5。"""
        X = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [4, 5]])
        y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5
        reg = LinearRegression()
        reg.fit(X, y)
        assert np.allclose(reg.coef_, [2.0, 3.0], atol=1e-10)
        assert np.isclose(reg.intercept_, 0.5, atol=1e-10)

    def test_predict(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict([[5]])
        assert np.isclose(y_pred[0], 10.0)

    def test_no_intercept(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        assert np.isclose(reg.intercept_, 0.0)
        assert np.isclose(reg.coef_[0], 2.0, atol=1e-10)

    def test_score_perfect(self):
        X = np.array([[1], [2], [3], [4]])
        y = 2 * X.ravel() + 1
        reg = LinearRegression().fit(X, y)
        assert np.isclose(reg.score(X, y), 1.0)

    def test_collinear_features(self):
        """共线性特征：lstsq 应能处理。"""
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([2, 4, 6])
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        assert np.allclose(y_pred, y, atol=1e-8)

    def test_clone(self):
        reg = LinearRegression(fit_intercept=False)
        cloned = clone(reg)
        assert cloned.fit_intercept == False
        assert not hasattr(cloned, "coef_")

    def test_repr(self):
        reg = LinearRegression()
        assert "LinearRegression" in repr(reg)


class TestLinearRegressionSGD:
    """SGD 求解的测试。"""

    def test_approximate_line(self):
        """SGD 是近似解，允许一定误差。"""
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])
        reg = LinearRegression(method="sgd")
        reg.fit(X, y)
        y_pred = reg.predict(X)
        # SGD 应该接近正确解
        assert np.allclose(y_pred, y, atol=0.5)

    def test_invalid_method(self):
        reg = LinearRegression(method="invalid")
        with pytest.raises(ValueError):
            reg.fit([[1], [2]], [1, 2])