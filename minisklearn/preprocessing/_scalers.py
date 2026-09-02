"""数值缩放器：StandardScaler 和 MinMaxScaler。

本模块实现两种最常用的特征缩放方法。它们都是 Transformer，遵循
fit → transform 的标准流程。

==============================================================================
为什么要缩放？
==============================================================================

很多算法对特征尺度敏感：

    - KNN 用距离度量，大量级特征会主导距离
    - 梯度下降法中，不同量级的特征导致不同方向的学习率需求
    - 正则化（L1/L2）对不同量级特征惩罚不均

缩放让所有特征处于可比的量级，消除量纲差异。

两种缩放的区别：
    - StandardScaler：去均值 + 除标准差 → 均值0、标准差1（保留分布形状）
    - MinMaxScaler：线性映射到 [min, max] → 有界范围（不保留分布形状）
"""

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array, check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    """标准化缩放器：去均值、除标准差。

    对每个特征独立计算：
        z = (x - mean) / std

    变换后每个特征的均值为 0、标准差为 1。

    参数：
        with_mean: 是否去均值。稀疏矩阵下需设为 False（否则破坏稀疏结构）。
        with_std: 是否除标准差。设为 False 则只去均值。

    fit 后的属性（下划线结尾）：
        mean_: 每个特征的均值，shape (n_features,)
        scale_: 每个特征的标准差，shape (n_features,)
        var_: 每个特征的方差，shape (n_features,)

    数学原理：
        mean = (1/n) Σ x_i
        var  = (1/n) Σ (x_i - mean)²
        std  = √var
        z_i  = (x_i - mean) / std

    使用示例：
        >>> scaler = StandardScaler()
        >>> scaler.fit([[1], [2], [3]])
        >>> scaler.transform([[1], [2], [3]])
        array([[-1.22464487], [0.], [1.22464487]])
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        """从数据中计算均值和标准差。

        向量化实现：np.mean / np.std 沿 axis=0 对每个特征独立计算，
        无需 Python 循环。
        """
        X = check_array(X, force_all_finite=True)
        self.n_features_in_ = X.shape[1]

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])

        if self.with_std:
            self.var_ = np.var(X, axis=0)
            self.scale_ = np.sqrt(self.var_)
            # 标准差为 0 的特征（常量特征）：缩放因子设为 1，避免除零
            # 变换后该特征全为 0（因为 x - mean = 0）
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
            self.var_ = np.ones(X.shape[1])

        return self

    def transform(self, X):
        """应用标准化变换：z = (x - mean) / scale。"""
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        """逆变换：x = z * scale + mean。"""
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """最小最大缩放器：线性映射到指定范围。

    对每个特征独立计算：
        z = (x - min) / (max - min) * (range_max - range_min) + range_min

    默认映射到 [0, 1]。

    参数：
        feature_range: 目标范围 (min, max)，默认 (0, 1)

    fit 后的属性：
        min_: 每个特征的最小值，shape (n_features,)
        scale_: 缩放因子，shape (n_features,)
        data_min_: 原始数据每特征最小值
        data_max_: 原始数据每特征最大值

    数学原理：
        scale = (range_max - range_min) / (data_max - data_min)
        z = (x - data_min) * scale + range_min

    使用示例：
        >>> scaler = MinMaxScaler()
        >>> scaler.fit([[1], [2], [3]])
        >>> scaler.transform([[1], [2], [3]])
        array([[0.], [0.5], [1.]])
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """从数据中计算最小值、最大值和缩放因子。"""
        X = check_array(X, force_all_finite=True)
        self.n_features_in_ = X.shape[1]

        range_min, range_max = self.feature_range
        if range_min >= range_max:
            raise ValueError(
                f"feature_range 的最小值 {range_min} 必须小于最大值 {range_max}"
            )

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # 范围为 0 的常量特征：缩放因子设为 1，避免除零
        with np.errstate(divide="ignore"):
            self.scale_ = (range_max - range_min) / self.data_range_
        self.scale_[self.data_range_ == 0] = 1.0

        self.min_ = range_min - self.data_min_ * self.scale_

        return self

    def transform(self, X):
        """应用最小最大变换：z = x * scale + min。"""
        check_is_fitted(self, ["scale_", "min_"])
        X = check_array(X)
        return X * self.scale_ + self.min_

    def inverse_transform(self, X):
        """逆变换：x = (z - min) / scale。"""
        check_is_fitted(self, ["scale_", "min_"])
        X = check_array(X)
        return (X - self.min_) / self.scale_