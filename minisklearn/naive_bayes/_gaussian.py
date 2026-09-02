"""高斯朴素贝叶斯（GaussianNB）。

==============================================================================
朴素贝叶斯原理
==============================================================================

基于贝叶斯定理 + 条件独立假设：

    P(y | x) ∝ P(y) * P(x | y) = P(y) * Π_j P(x_j | y)

    预测：ŷ = argmax_y P(y) * Π_j P(x_j | y)

取对数避免下溢：

    ŷ = argmax_y [ log P(y) + Σ_j log P(x_j | y) ]

高斯假设：每个特征在给定类别下服从高斯分布：

    P(x_j | y=c) = N(x_j | μ_cj, σ²_cj)

fit 阶段：对每个类别 c，计算每个特征的均值和方差。

为什么"朴素"？
    因为假设特征条件独立（P(x|y) = Π P(x_j|y)），这在现实中几乎不成立。
    但实践效果出奇地好，尤其在高维数据上。
"""

import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted


class GaussianNB(BaseEstimator, ClassifierMixin):
    """高斯朴素贝叶斯分类器。

    参数：
        var_smoothing: 方差平滑因子（防止方差为 0），默认 1e-9

    fit 后的属性：
        classes_: 类别标签
        class_prior_: 每个类别的先验概率
        theta_: 每个类别每个特征的均值，shape (n_classes, n_features)
        var_: 每个类别每个特征的方差，shape (n_classes, n_features)

    使用示例：
        >>> clf = GaussianNB()
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """拟合高斯朴素贝叶斯。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        n_samples, n_features = X.shape
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.var_[i] = np.var(X_c, axis=0)
            self.class_count_[i] = len(X_c)

        # 方差平滑：防止方差为 0
        epsilon = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += epsilon

        # 先验概率
        self.class_prior_ = self.class_count_ / n_samples

        return self

    def _joint_log_likelihood(self, X):
        """计算每个样本每个类别的对数联合概率。

        log P(c) + Σ_j log N(x_j | μ_cj, σ²_cj)
        = log P(c) + Σ_j [-0.5 * log(2π σ²_cj) - (x_j - μ_cj)² / (2σ²_cj)]
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        jll = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            log_prior = np.log(self.class_prior_[i])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[i])
                + (X - self.theta_[i]) ** 2 / self.var_[i],
                axis=1
            )
            jll[:, i] = log_prior + log_likelihood

        return jll

    def predict(self, X):
        """预测：取对数联合概率最大的类别。"""
        check_is_fitted(self, ["theta_", "var_"])
        X = check_array(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """预测类别概率（softmax 归一化对数联合概率）。"""
        check_is_fitted(self, ["theta_", "var_"])
        X = check_array(X)
        jll = self._joint_log_likelihood(X)

        # softmax 归一化（减去最大值防止溢出）
        jll_max = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - jll_max)
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)
