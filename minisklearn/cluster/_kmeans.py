"""KMeans 聚类算法。

==============================================================================
KMeans 原理
==============================================================================

目标：将 n 个样本分为 K 个簇，使簇内平方和（WCSS）最小：

    min Σ_{k=1}^{K} Σ_{x ∈ C_k} ||x - μ_k||²

其中 μ_k 是簇 k 的中心（均值）。

Lloyd 算法（交替优化）：
    1. 分配步：固定中心，每个样本分配到最近中心
       C_k = {x : k = argmin_j ||x - μ_j||²}
    2. 更新步：固定分配，中心移到簇内均值
       μ_k = (1/|C_k|) Σ_{x ∈ C_k} x
    3. 重复 1-2 直到中心不再变化（收敛）

    保证单调下降 WCSS，但可能收敛到局部最优。

KMeans++ 初始化：
    随机初始化中心可能导致局部最优。KMeans++ 用概率方式选初始中心：
    1. 随机选第一个中心
    2. 对每个后续中心，以 D(x)² 的概率选择（D(x) 是到最近已选中心的距离）
    3. 这样中心分散开来，避免初始聚集

    KMeans++ 保证期望 WCSS <= 8 * OPT（理论保证）。
"""

import numpy as np
from ..base import BaseEstimator, ClusterMixin
from ..utils.validation import check_array, check_is_fitted
from ..neighbors._distances import euclidean_distances


class KMeans(BaseEstimator, ClusterMixin):
    """K 均值聚类。

    参数：
        n_clusters: 簇数 K
        init: 初始化方式
            - 'k-means++': KMeans++ 概率初始化（默认）
            - 'random': 随机选 K 个样本作为中心
        n_init: 不同初始化运行次数，取最优（默认 10）
        max_iter: 单次运行最大迭代次数
        tol: 收敛阈值（中心移动的 Frobenius 范数）
        random_state: 随机种子

    fit 后的属性：
        cluster_centers_: 簇中心，shape (n_clusters, n_features)
        labels_: 每个样本的簇标签，shape (n_samples,)
        inertia_: WCSS（簇内平方和）
        n_iter_: 实际迭代次数

    使用示例：
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> kmeans.fit(X)
        >>> kmeans.labels_  # 聚类标签
        >>> kmeans.predict(X_new)  # 预测新样本的簇
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """运行 KMeans 聚类。"""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        rng = np.random.RandomState(self.random_state)

        # 多次运行取最优
        best_inertia = float("inf")
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def _single_run(self, X, rng):
        """单次 KMeans 运行。"""
        n_samples = X.shape[0]

        # 初始化中心
        centers = self._init_centroids(X, rng)

        for iteration in range(self.max_iter):
            # 分配步：每个样本到最近中心
            distances = euclidean_distances(X, centers, squared=True)
            labels = np.argmin(distances, axis=1)

            # 更新步：中心移到簇内均值
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # 空簇：保留旧中心（或重新初始化）
                    new_centers[k] = centers[k]

            # 收敛检查
            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if center_shift < self.tol:
                break

        # 最终分配
        distances = euclidean_distances(X, centers, squared=True)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1))

        return centers, labels, inertia, iteration + 1

    def _init_centroids(self, X, rng):
        """初始化聚类中心。"""
        n_samples = X.shape[0]

        if self.init == "k-means++":
            return self._kmeans_plus_plus(X, rng)
        elif self.init == "random":
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()
        else:
            raise ValueError(f"未知 init: {self.init}")

    def _kmeans_plus_plus(self, X, rng):
        """KMeans++ 初始化。

        1. 随机选第一个中心
        2. 后续中心以 D(x)² 的概率选择
        """
        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))

        # 第一个中心：均匀随机
        centers[0] = X[rng.randint(n_samples)]

        # 后续中心
        for k in range(1, self.n_clusters):
            # 计算每个样本到最近已选中心的距离平方
            distances = euclidean_distances(
                X, centers[:k], squared=True
            )
            min_distances = np.min(distances, axis=1)

            # 以 D(x)² 为概率选择
            probs = min_distances / np.sum(min_distances)
            centers[k] = X[rng.choice(n_samples, p=probs)]

        return centers

    def predict(self, X):
        """预测每个样本最近的簇标签。"""
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X)
        distances = euclidean_distances(X, self.cluster_centers_, squared=True)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        """拟合并返回聚类标签。"""
        return self.fit(X).labels_

    def transform(self, X):
        """计算每个样本到各簇中心的距离。"""
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X)
        return euclidean_distances(X, self.cluster_centers_)