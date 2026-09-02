"""主成分分析（PCA）。

==============================================================================
PCA 原理
==============================================================================

目标：找到数据方差最大的方向，将高维数据投影到低维子空间。

数学推导：

    1. 中心化：X' = X - mean(X)
    2. SVD 分解：X' = U S V^T
       - U: 左奇异向量 (n × n)
       - S: 奇异值对角阵 (n × d)
       - V: 右奇异向量 (d × d)，即主成分方向
    3. 主成分 = V 的前 k 列（对应最大奇异值）
    4. 降维：X_new = X' @ V[:, :k] = U[:, :k] @ S[:k, :k]

为什么 SVD？

    协方差矩阵 C = X'^T X' / (n-1) = V S² V^T / (n-1)
    → 特征值 = S² / (n-1)，特征向量 = V

    SVD 直接给出 V 和 S，无需显式计算 X'^T X'（避免精度损失）。
    且 SVD 对数值条件更鲁棒。

解释方差：

    explained_variance_k = S_k² / (n-1)
    explained_variance_ratio_k = explained_variance_k / total_variance

    total_variance = Σ all explained_variance = trace(C)

白化（whiten）：

    白化使各主成分方差为 1：X_whiten = X_new / sqrt(explained_variance)
    去除特征间的相关性，常用于预处理。
"""

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array, check_is_fitted


class PCA(BaseEstimator, TransformerMixin):
    """主成分分析降维。

    参数：
        n_components: 降维后的维度
            - int: 指定维度数
            - float (0, 1): 保留的方差比例，自动选择维度数
            - None: min(n_samples, n_features)
        whiten: 是否白化（使各主成分方差为 1）

    fit 后的属性：
        components_: 主成分方向，shape (n_components, n_features)
        explained_variance_: 各主成分解释的方差
        explained_variance_ratio_: 方差比例
        singular_values_: 奇异值
        mean_: 数据均值
        n_components_: 实际保留的维度数

    使用示例：
        >>> pca = PCA(n_components=2)
        >>> pca.fit(X)
        >>> X_reduced = pca.transform(X)
        >>> X_recovered = pca.inverse_transform(X_reduced)
    """

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        """用 SVD 拟合 PCA。"""
        X = check_array(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD 分解：X_centered = U S V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 确定保留的维度数
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # 按方差比例自动选择
            total_var = np.sum(S ** 2) / (n_samples - 1)
            ratio_cumsum = np.cumsum(S ** 2) / ((n_samples - 1) * total_var)
            n_components = np.searchsorted(ratio_cumsum, self.n_components) + 1
        else:
            n_components = int(self.n_components)

        n_components = min(n_components, min(n_samples, n_features))
        self.n_components_ = n_components

        # 取前 n_components 个主成分
        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]

        # 解释方差
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = self.explained_variance_[:n_components]

        total_var = np.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # 白化预处理：除以 sqrt(explained_variance_) 使各主成分方差为 1
        if self.whiten:
            self.whiten_scale_ = np.sqrt(self.explained_variance_)
        else:
            self.whiten_scale_ = np.ones(n_components)

        return self

    def transform(self, X):
        """降维：X_new = (X - mean) @ components_.T。"""
        check_is_fitted(self, ["components_", "mean_"])
        X = check_array(X)
        X_centered = X - self.mean_

        # 投影到主成分空间
        X_transformed = X_centered @ self.components_.T

        # 白化
        if self.whiten:
            X_transformed /= self.whiten_scale_

        return X_transformed

    def inverse_transform(self, X):
        """逆变换：X = X_new @ components_ + mean。"""
        check_is_fitted(self, ["components_", "mean_"])
        X = np.asarray(X, dtype=np.float64)

        # 反白化
        if self.whiten:
            X = X * self.whiten_scale_

        # 逆投影
        X_reconstructed = X @ self.components_
        return X_reconstructed + self.mean_

    def fit_transform(self, X, y=None):
        """拟合并降维。"""
        return self.fit(X).transform(X)