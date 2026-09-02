"""KNN 测试：距离计算 + 分类器 + 回归器。"""

import numpy as np
import pytest
from minisklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    euclidean_distances,
    manhattan_distances,
    find_k_neighbors,
)
from minisklearn.base import clone


class TestEuclideanDistances:
    """向量化欧氏距离测试。"""

    def test_single_pair(self):
        x = np.array([[1, 0]])
        y = np.array([[0, 0]])
        dist = euclidean_distances(x, y)
        assert dist.shape == (1, 1)
        assert np.isclose(dist[0, 0], 1.0)

    def test_matrix(self):
        X = np.array([[0, 0], [3, 0]])
        Y = np.array([[0, 0], [0, 4]])
        dist = euclidean_distances(X, Y)
        assert dist.shape == (2, 2)
        assert np.isclose(dist[0, 0], 0.0)
        assert np.isclose(dist[0, 1], 4.0)
        assert np.isclose(dist[1, 0], 3.0)
        assert np.isclose(dist[1, 1], 5.0)

    def test_squared(self):
        X = np.array([[1, 0]])
        Y = np.array([[0, 0]])
        dist_sq = euclidean_distances(X, Y, squared=True)
        assert np.isclose(dist_sq[0, 0], 1.0)

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 3)
        dist_xy = euclidean_distances(X, X)
        assert np.allclose(dist_xy, dist_xy.T)

    def test_zero_distance(self):
        X = np.array([[1, 2, 3]])
        dist = euclidean_distances(X, X)
        assert np.isclose(dist[0, 0], 0.0)


class TestManhattanDistances:
    def test_basic(self):
        X = np.array([[0, 0]])
        Y = np.array([[3, 4]])
        dist = manhattan_distances(X, Y)
        assert np.isclose(dist[0, 0], 7.0)


class TestFindKNeighbors:
    def test_basic(self):
        X_query = np.array([[0, 0]])
        X_train = np.array([[1, 0], [0, 1], [2, 0], [0, 2]])
        indices, distances = find_k_neighbors(X_query, X_train, k=2)
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        # 最近的两个应该是 [1,0] 和 [0,1]，距离都是 1
        assert np.isclose(distances[0, 0], 1.0)
        assert np.isclose(distances[0, 1], 1.0)

    def test_sorted(self):
        """返回的近邻应按距离排序。"""
        X_query = np.array([[0, 0]])
        X_train = np.array([[3, 0], [1, 0], [2, 0]])
        indices, distances = find_k_neighbors(X_query, X_train, k=3)
        assert distances[0, 0] <= distances[0, 1] <= distances[0, 2]


class TestKNNClassifier:
    """KNN 分类器测试。"""

    def test_basic_classification(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        y = np.array([0, 0, 1, 1])
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        y_pred = clf.predict([[0.5, 0.5], [10.5, 10.5]])
        assert y_pred[0] == 0
        assert y_pred[1] == 1

    def test_k3_majority_vote(self):
        X = np.array([[0], [0.1], [0.2], [10], [10.1]])
        y = np.array([0, 0, 1, 1, 1])
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
        # 查询 [0.05]，最近 3 个是 [0], [0.1], [0.2]，标签 0,0,1 → 多数 0
        y_pred = clf.predict([[0.05]])
        assert y_pred[0] == 0

    def test_distance_weights(self):
        """距离加权：近的邻居影响更大。"""
        X = np.array([[0], [0.5], [10]])
        y = np.array([0, 1, 1])
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        clf.fit(X, y)
        # 查询 [0]，距离 0, 0.5, 10
        # 权重: inf(精确匹配), 2, 0.1
        # 类 0 权重 inf，类 1 权重 2.1 → 预测 0
        y_pred = clf.predict([[0]])
        assert y_pred[0] == 0

    def test_predict_proba(self):
        X = np.array([[0], [1], [10], [11]])
        y = np.array([0, 0, 1, 1])
        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(X, y)
        proba = clf.predict_proba([[0.5]])
        assert proba.shape == (1, 2)
        assert np.isclose(proba.sum(axis=1)[0], 1.0)
        # [0.5] 最近邻是 [0] 和 [1]，都是类 0
        assert proba[0, 0] == 1.0

    def test_multiclass(self):
        X = np.array([[0, 0], [10, 0], [0, 10]])
        y = np.array([0, 1, 2])
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        y_pred = clf.predict([[0.1, 0.1], [9.9, 0.1], [0.1, 9.9]])
        assert list(y_pred) == [0, 1, 2]

    def test_string_labels(self):
        X = np.array([[0], [10]])
        y = np.array(["猫", "狗"])
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        y_pred = clf.predict([[1]])
        assert y_pred[0] == "猫"

    def test_manhattan_metric(self):
        X = np.array([[0, 0], [10, 10]])
        y = np.array([0, 1])
        clf = KNeighborsClassifier(n_neighbors=1, metric="manhattan")
        clf.fit(X, y)
        y_pred = clf.predict([[1, 1]])
        assert y_pred[0] == 0

    def test_clone(self):
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        cloned = clone(clf)
        assert cloned.n_neighbors == 3
        assert cloned.weights == "distance"
        assert not hasattr(cloned, "_X")

    def test_repr(self):
        clf = KNeighborsClassifier(n_neighbors=7)
        assert "KNeighborsClassifier" in repr(clf)
        assert "n_neighbors=7" in repr(clf)

    def test_k_larger_than_samples(self):
        """K 大于样本数时应自动截断。"""
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        clf = KNeighborsClassifier(n_neighbors=10)
        clf.fit(X, y)
        y_pred = clf.predict([[0.4]])
        assert y_pred[0] in [0, 1]


class TestKNNRegressor:
    """KNN 回归器测试。"""

    def test_basic_regression(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 1, 2, 3])
        reg = KNeighborsRegressor(n_neighbors=1)
        reg.fit(X, y)
        y_pred = reg.predict([[0.4], [2.6]])
        assert np.isclose(y_pred[0], 0.0)
        assert np.isclose(y_pred[1], 3.0)

    def test_k2_average(self):
        X = np.array([[0], [1], [2]])
        y = np.array([0, 10, 20])
        reg = KNeighborsRegressor(n_neighbors=2)
        reg.fit(X, y)
        # [0.4] 最近邻 [0] 和 [1]，平均 (0+10)/2 = 5
        y_pred = reg.predict([[0.4]])
        assert np.isclose(y_pred[0], 5.0)

    def test_distance_weights(self):
        X = np.array([[0], [1], [2]])
        y = np.array([0, 10, 20])
        reg = KNeighborsRegressor(n_neighbors=2, weights="distance")
        reg.fit(X, y)
        # [0.4] 最近邻 [0](d=0.4) 和 [1](d=0.6)
        # 加权: (0/0.4 + 10/0.6) / (1/0.4 + 1/0.6) = (0 + 16.67) / (2.5 + 1.67)
        y_pred = reg.predict([[0.4]])
        expected = (10 / 0.6) / (1 / 0.4 + 1 / 0.6)
        assert np.isclose(y_pred[0], expected)

    def test_exact_match(self):
        """查询点与训练点重合时返回该点值。"""
        X = np.array([[0], [1], [2]])
        y = np.array([0, 10, 20])
        reg = KNeighborsRegressor(n_neighbors=2, weights="distance")
        reg.fit(X, y)
        y_pred = reg.predict([[1]])
        assert np.isclose(y_pred[0], 10.0)

    def test_clone(self):
        reg = KNeighborsRegressor(n_neighbors=3)
        cloned = clone(reg)
        assert cloned.n_neighbors == 3
        assert not hasattr(cloned, "_X")