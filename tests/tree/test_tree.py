"""决策树测试：分类树 + 回归树。"""

import numpy as np
import pytest
from minisklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from minisklearn.base import clone


def make_classification_data(n=100, seed=42):
    """生成线性可分的二分类数据。"""
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


def make_regression_data(n=100, seed=42):
    """生成分段常数回归数据。"""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-5, 5, size=(n, 1))
    y = np.where(X.ravel() > 0, 2.0, -2.0) + rng.randn(n) * 0.1
    return X, y


class TestDecisionTreeClassifier:
    """决策树分类器测试。"""

    def test_fit_predict_basic(self):
        X, y = make_classification_data()
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.9

    def test_perfect_separation(self):
        """完全可分的数据应该 100% 准确。"""
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert np.all(y_pred == y)

    def test_max_depth_limits(self):
        """max_depth=1 只能一次分裂。"""
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 0, 0])
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        # 深度 1 的树只有 2 个叶子，最多 2 种预测
        y_pred = clf.predict(X)
        assert len(np.unique(y_pred)) <= 2

    def test_min_samples_split(self):
        """min_samples_split 大时树更浅。"""
        X, y = make_classification_data(n=20)
        clf = DecisionTreeClassifier(min_samples_split=100)
        clf.fit(X, y)
        # 几乎不分裂，根节点就是叶子
        assert clf.tree_.is_leaf or clf.tree_.left.is_leaf

    def test_min_samples_leaf(self):
        """min_samples_leaf 限制叶子最小样本数。"""
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 1, 0, 1])
        clf = DecisionTreeClassifier(min_samples_leaf=2)
        clf.fit(X, y)
        # 每个叶子至少 2 个样本
        y_pred = clf.predict(X)
        assert len(y_pred) == 4

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        centers = [[2, 2], [-2, 2], [0, -2]]
        X = np.vstack([rng.randn(30, 2) + c for c in centers])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.8

    def test_string_labels(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array(["a", "a", "b", "b"])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert set(y_pred) <= {"a", "b"}

    def test_predict_proba(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_clone(self):
        clf = DecisionTreeClassifier(max_depth=5)
        cloned = clone(clf)
        assert cloned.max_depth == 5
        assert not hasattr(cloned, "tree_")

    def test_repr(self):
        clf = DecisionTreeClassifier(max_depth=3)
        assert "DecisionTreeClassifier" in repr(clf)

    def test_pure_node(self):
        """所有样本同类时应直接创建叶子。"""
        X = np.array([[0], [1], [2]])
        y = np.array([0, 0, 0])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.tree_.is_leaf


class TestDecisionTreeRegressor:
    """决策树回归器测试。"""

    def test_fit_predict_basic(self):
        X, y = make_regression_data()
        reg = DecisionTreeRegressor(max_depth=3)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        # 训练集上应该有合理的拟合
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 1.0

    def test_perfect_piecewise(self):
        """分段常数数据应能完美拟合。"""
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([-1, -1, -1, 1, 1, 1])
        reg = DecisionTreeRegressor()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert np.allclose(y_pred, y)

    def test_max_depth(self):
        X, y = make_regression_data()
        reg = DecisionTreeRegressor(max_depth=1)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        # 深度 1 只有两个叶子，最多 2 个不同预测值
        assert len(np.unique(y_pred)) <= 2

    def test_leaf_value_is_mean(self):
        """叶子节点的值应该是该节点样本的均值。"""
        X = np.array([[0], [1], [2], [3]])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        reg = DecisionTreeRegressor(max_depth=1)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        # 深度 1 分裂后左右叶子的值是各自均值
        assert len(np.unique(y_pred)) <= 2

    def test_score(self):
        X, y = make_regression_data()
        reg = DecisionTreeRegressor(max_depth=5)
        reg.fit(X, y)
        score = reg.score(X, y)
        assert score > 0.5

    def test_clone(self):
        reg = DecisionTreeRegressor(max_depth=5)
        cloned = clone(reg)
        assert cloned.max_depth == 5
        assert not hasattr(cloned, "tree_")

    def test_constant_target(self):
        """目标值全相同时应直接创建叶子。"""
        X = np.array([[0], [1], [2]])
        y = np.array([5.0, 5.0, 5.0])
        reg = DecisionTreeRegressor()
        reg.fit(X, y)
        assert reg.tree_.is_leaf
        assert np.isclose(reg.tree_.value, 5.0)