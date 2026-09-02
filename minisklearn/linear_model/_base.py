"""线性回归：LinearRegression。

本模块实现线性回归的两种求解方式：

    1. 正规方程（Normal Equation）：w = (X^T X)^{-1} X^T y
       - 精确解，一步到位
       - 特征数多时矩阵求逆代价大 O(d³)
       - X^T X 不可逆时需用伪逆

    2. 随机梯度下降（SGD）：逐步迭代 w -= lr * ∇L
       - 近似解，需迭代
       - 适合大规模数据（每次只用一个样本）
       - 适合特征数多的场景

==============================================================================
数学原理
==============================================================================

线性模型：ŷ = Xw + b

损失函数（均方误差）：
    L(w, b) = (1/n) Σ (y_i - ŷ_i)² = (1/n) ||y - Xw - b||²

正规方程的推导：
    对 L 求偏导并令其为 0：
    ∂L/∂w = -(2/n) X^T (y - Xw - b) = 0
    → X^T X w = X^T (y - b)
    → w = (X^T X)^{-1} X^T (y - b)

    截距通过中心化处理：
    令 X' = X - mean(X)，y' = y - mean(y)
    w = (X'^T X')^{-1} X'^T y'
    b = mean(y) - mean(X) @ w

SGD 的梯度：
    对单个样本 (x_i, y_i)：
    ∂L_i/∂w = -2 x_i (y_i - x_i w - b)
    更新：w -= lr * ∂L_i/∂w
"""

import numpy as np
from ..base import BaseEstimator, RegressorMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted


class LinearRegression(BaseEstimator, RegressorMixin):
    """普通最小二乘线性回归。

    参数：
        fit_intercept: 是否拟合截距项（偏置）
        method: 求解方法
            - 'normal': 正规方程（默认，小数据集首选）
            - 'sgd': 随机梯度下降（大数据集或特征多时用）

    fit 后的属性：
        coef_: 回归系数，shape (n_features,)
        intercept_: 截距，float
        n_features_in_: 输入特征数

    使用示例：
        >>> reg = LinearRegression()
        >>> reg.fit([[1], [2], [3]], [2, 4, 6])
        >>> reg.coef_
        array([2.])
        >>> reg.intercept_
        0.0
    """

    def __init__(self, fit_intercept=True, method="normal"):
        self.fit_intercept = fit_intercept
        self.method = method

    def fit(self, X, y):
        """拟合线性模型。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.method == "normal":
            self._fit_normal(X, y)
        elif self.method == "sgd":
            self._fit_sgd(X, y)
        else:
            raise ValueError(
                f"method 必须为 'normal' 或 'sgd'，得到 '{self.method}'"
            )

        return self

    def _fit_normal(self, X, y):
        """正规方程求解。

        w = (X^T X)^{-1} X^T y

        用中心化处理截距：
            X' = X - mean(X)
            y' = y - mean(y)
            w = (X'^T X')^{-1} X'^T y'
            b = mean(y) - mean(X) @ w

        用 np.linalg.lstsq 代替直接求逆，更数值稳定：
            lstsq 用 SVD 分解，能处理 X^T X 不可逆的情况
        """
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
            # lstsq 比 inv 更稳定，处理共线性
            self.coef_, _, _, _ = np.linalg.lstsq(
                X_centered, y_centered, rcond=None
            )
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept_ = 0.0

    def _fit_sgd(self, X, y, learning_rate=0.01, n_epochs=100):
        """随机梯度下降求解。

        每次取一个样本，计算梯度并更新参数：
            grad_w = -2 * x_i * (y_i - x_i @ w - b)
            grad_b = -2 * (y_i - x_i @ w - b)
            w -= lr * grad_w
            b -= lr * grad_b

        用固定的学习率（简化教学，sklearn 用学习率调度器）。
        """
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0 if self.fit_intercept else 0.0

        for epoch in range(n_epochs):
            # 随机打乱样本顺序
            indices = np.random.permutation(n_samples)
            for i in indices:
                x_i = X[i]
                y_i = y[i]
                y_pred = x_i @ self.coef_ + self.intercept_
                error = y_pred - y_i

                # 梯度
                grad_w = 2 * error * x_i
                self.coef_ -= learning_rate * grad_w

                if self.fit_intercept:
                    grad_b = 2 * error
                    self.intercept_ -= learning_rate * grad_b

    def predict(self, X):
        """预测：ŷ = X @ w + b。"""
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_