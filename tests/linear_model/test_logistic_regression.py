"""LogisticRegression 测试。"""

import numpy as np
import pytest
from minisklearn.linear_model import LogisticRegression
from minisklearn.base import clone


def make_binary_data(n=100, seed=42):
    """生成线性可分的二分类数据。"""
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


def make_multiclass_data(n=80, seed=42):
    """生成三分类数据。"""
    rng = np.random.RandomState(seed)
    centers = [[2, 2], [-2, 2], [0, -2]]
    Xs = [rng.randn(n, 2) + np.array(c) for c in centers]
    X = np.vstack(Xs)
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


class TestLogisticRegressionBinary:
    """二分类测试。"""

    def test_fit_predict(self):
        X, y = make_binary_data()
        clf = LogisticRegression(max_iter=500, learning_rate=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.9

    def test_predict_proba(self):
        X, y = make_binary_data()
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_decision_function(self):
        X, y = make_binary_data()
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(y),)

    def test_classes_stored(self):
        X, y = make_binary_data()
        clf = LogisticRegression()
        clf.fit(X, y)
        assert set(clf.classes_) == {0, 1}

    def test_string_labels(self):
        X, y = make_binary_data()
        y_str = np.where(y == 1, "正", "负")
        clf = LogisticRegression(max_iter=500, learning_rate=0.5)
        clf.fit(X, y_str)
        y_pred = clf.predict(X)
        assert set(y_pred) <= {"正", "负"}
        assert np.mean(y_pred == y_str) > 0.9

    def test_regularization(self):
        """强正则化（小 C）应使系数更小。"""
        X, y = make_binary_data()
        clf_weak = LogisticRegression(C=100, max_iter=500)
        clf_strong = LogisticRegression(C=0.01, max_iter=500)
        clf_weak.fit(X, y)
        clf_strong.fit(X, y)
        assert np.linalg.norm(clf_strong.coef_) < np.linalg.norm(clf_weak.coef_)

    def test_clone(self):
        clf = LogisticRegression(C=2.0)
        cloned = clone(clf)
        assert cloned.C == 2.0
        assert not hasattr(cloned, "coef_")

    def test_repr(self):
        clf = LogisticRegression(C=2.0)
        assert "LogisticRegression" in repr(clf)
        assert "C=2.0" in repr(clf)


class TestLogisticRegressionMulticlass:
    """多分类（OvR）测试。"""

    def test_fit_predict(self):
        X, y = make_multiclass_data()
        clf = LogisticRegression(max_iter=500, learning_rate=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.8

    def test_classes_stored(self):
        X, y = make_multiclass_data()
        clf = LogisticRegression()
        clf.fit(X, y)
        assert set(clf.classes_) == {0, 1, 2}

    def test_predict_proba(self):
        X, y = make_multiclass_data()
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_coef_shape(self):
        X, y = make_multiclass_data()
        clf = LogisticRegression()
        clf.fit(X, y)
        assert clf.coef_.shape == (3, 2)