"""K 近邻分类器和回归器。

==============================================================================
KNN 算法原理
==============================================================================

KNN 是最简单的机器学习算法之一——它没有"训练"过程，只是记住数据。

分类：
    1. 计算查询点到所有训练点的距离
    2. 找到距离最近的 K 个训练点
    3. 对这 K 个点的标签投票，多数类为预测结果

回归：
    1. 同上找到 K 个最近邻
    2. 对这 K 个点的目标值取平均（或距离加权平均）

加权投票：
    - uniform: 每个近邻权重相同
    - distance: 权重 = 1 / distance（近的邻居影响更大）

KNN 的特点：
    - 优点：无需训练、概念简单、天然支持多分类
    - 缺点：预测慢（每次都要算所有距离）、需要大量内存、对尺度敏感
    - 本质：非参数方法，决策边界由数据分布决定

为什么 KNN 不需要 fit？
    KNN 的"模型"就是训练数据本身。fit 只是存储数据，
    真正的计算在 predict 时进行。这叫"惰性学习"（lazy learning）。
"""

import numpy as np
from collections import Counter

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted
from ._distances import find_k_neighbors


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """K 近邻分类器。

    参数：
        n_neighbors: 近邻数 K（默认 5）
        weights: 权重方式
            - 'uniform': 均匀权重
            - 'distance': 距离倒数权重
        metric: 距离度量 ('euclidean' 或 'manhattan')

    fit 后的属性：
        _X: 训练特征（存储数据，KNN 是惰性学习）
        _y: 训练标签
        classes_: 类别标签
        n_features_in_: 输入特征数

    使用示例：
        >>> clf = KNeighborsClassifier(n_neighbors=3)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        """存储训练数据（KNN 是惰性学习，fit 只存数据）。"""
        X, y = check_X_y(X, y)
        self._X = X
        self._y = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """预测：找 K 个近邻，加权投票。"""
        check_is_fitted(self, ["_X", "_y"])
        X = check_array(X)

        indices, distances = find_k_neighbors(
            X, self._X, self.n_neighbors, metric=self.metric
        )

        # 对每个查询点投票
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self._y.dtype)

        for i in range(n_samples):
            neighbor_labels = self._y[indices[i]]
            neighbor_distances = distances[i]

            if self.weights == "uniform":
                # 均匀投票：多数类
                counts = Counter(neighbor_labels)
                y_pred[i] = counts.most_common(1)[0][0]
            elif self.weights == "distance":
                # 距离加权投票
                # 处理距离为 0 的情况（查询点与训练点重合）
                with np.errstate(divide="ignore"):
                    weights = np.where(
                        neighbor_distances > 0,
                        1.0 / neighbor_distances,
                        1e10  # 距离为 0 时给极大权重
                    )
                # 按类别累加权重
                class_weights = {}
                for label, w in zip(neighbor_labels, weights):
                    class_weights[label] = class_weights.get(label, 0) + w
                y_pred[i] = max(class_weights, key=class_weights.get)
            else:
                raise ValueError(f"未知 weights: {self.weights}")

        return y_pred

    def predict_proba(self, X):
        """预测类别概率（近邻中各类别的比例）。"""
        check_is_fitted(self, ["_X", "_y"])
        X = check_array(X)

        indices, distances = find_k_neighbors(
            X, self._X, self.n_neighbors, metric=self.metric
        )

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for i in range(n_samples):
            neighbor_labels = self._y[indices[i]]
            neighbor_distances = distances[i]

            if self.weights == "uniform":
                weights = np.ones(len(neighbor_labels))
            elif self.weights == "distance":
                weights = np.where(
                    neighbor_distances > 0,
                    1.0 / neighbor_distances,
                    1e10
                )
            else:
                raise ValueError(f"未知 weights: {self.weights}")

            total_weight = np.sum(weights)
            for label, w in zip(neighbor_labels, weights):
                proba[i, class_to_idx[label]] += w
            proba[i] /= total_weight

        return proba


class KNeighborsRegressor(BaseEstimator, RegressorMixin):
    """K 近邻回归器。

    参数同 KNeighborsClassifier。

    预测方式：
        - uniform: K 个近邻目标值的平均
        - distance: K 个近邻目标值的距离加权平均

    使用示例：
        >>> reg = KNeighborsRegressor(n_neighbors=3)
        >>> reg.fit(X_train, y_train)
        >>> reg.predict(X_test)
    """

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        """存储训练数据。"""
        X, y = check_X_y(X, y)
        self._X = X
        self._y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """预测：找 K 个近邻，加权平均。"""
        check_is_fitted(self, ["_X", "_y"])
        X = check_array(X)

        indices, distances = find_k_neighbors(
            X, self._X, self.n_neighbors, metric=self.metric
        )

        neighbor_values = self._y[indices]  # shape (n_samples, k)

        if self.weights == "uniform":
            return np.mean(neighbor_values, axis=1)
        elif self.weights == "distance":
            # 距离加权平均
            with np.errstate(divide="ignore"):
                weights = np.where(
                    distances > 0,
                    1.0 / distances,
                    1e10
                )
            # 处理距离为 0 的情况：如果某距离为 0，直接返回该近邻的值
            zero_mask = distances == 0
            if np.any(zero_mask):
                # 有精确匹配的近邻，直接用其值
                result = np.empty(neighbor_values.shape[0])
                for i in range(neighbor_values.shape[0]):
                    if np.any(zero_mask[i]):
                        # 取距离为 0 的近邻的平均
                        result[i] = np.mean(neighbor_values[i][zero_mask[i]])
                    else:
                        result[i] = np.average(
                            neighbor_values[i], weights=weights[i]
                        )
                return result
            else:
                return np.average(neighbor_values, weights=weights, axis=1)
        else:
            raise ValueError(f"未知 weights: {self.weights}")