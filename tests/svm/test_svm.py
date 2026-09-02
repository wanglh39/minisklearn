"""LinearSVC 测试。"""

import numpy as np
import pytest
from minisklearn.svm import LinearSVC
from minisklearn.base import clone


def make_binary_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


class TestLinearSVC:
    def test_fit_predict_binary(self):
        X, y = make_binary_data()
        clf = LinearSVC(C=1.0, max_iter=1000, learning_rate=0.01)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.9

    def test_decision_function(self):
        X, y = make_binary_data()
        clf = LinearSVC(max_iter=1000)
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(y),)

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        centers = [[2, 2], [-2, 2], [0, -2]]
        X = np.vstack([rng.randn(40, 2) + c for c in centers])
        y = np.array([0] * 40 + [1] * 40 + [2] * 40)
        clf = LinearSVC(C=1.0, max_iter=1000, learning_rate=0.01)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.8

    def test_string_labels(self):
        X, y = make_binary_data()
        y_str = np.where(y == 1, "正", "负")
        clf = LinearSVC(max_iter=1000)
        clf.fit(X, y_str)
        y_pred = clf.predict(X)
        assert set(y_pred) <= {"正", "负"}

    def test_clone(self):
        clf = LinearSVC(C=2.0)
        cloned = clone(clf)
        assert cloned.C == 2.0
        assert not hasattr(cloned, "coef_")

    def test_repr(self):
        clf = LinearSVC(C=2.0)
        assert "LinearSVC" in repr(clf)