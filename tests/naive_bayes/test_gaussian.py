"""GaussianNB 测试。"""

import numpy as np
import pytest
from minisklearn.naive_bayes import GaussianNB
from minisklearn.base import clone


def make_data(n=80, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


class TestGaussianNB:
    def test_fit_predict(self):
        X, y = make_data()
        clf = GaussianNB()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.9

    def test_predict_proba(self):
        X, y = make_data()
        clf = GaussianNB()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        centers = [[2, 2], [-2, 2], [0, -2]]
        X = np.vstack([rng.randn(40, 2) + c for c in centers])
        y = np.array([0] * 40 + [1] * 40 + [2] * 40)
        clf = GaussianNB()
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.85

    def test_string_labels(self):
        X, y = make_data()
        y_str = np.where(y == 1, "a", "b")
        clf = GaussianNB()
        clf.fit(X, y_str)
        y_pred = clf.predict(X)
        assert set(y_pred) <= {"a", "b"}

    def test_theta_and_var_stored(self):
        X, y = make_data()
        clf = GaussianNB()
        clf.fit(X, y)
        assert clf.theta_.shape == (2, 2)
        assert clf.var_.shape == (2, 2)

    def test_clone(self):
        clf = GaussianNB(var_smoothing=1e-5)
        cloned = clone(clf)
        assert cloned.var_smoothing == 1e-5
        assert not hasattr(cloned, "theta_")

    def test_repr(self):
        clf = GaussianNB()
        assert "GaussianNB" in repr(clf)

    def test_single_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        X[:50] += 5
        y = np.array([1] * 50 + [0] * 50)
        clf = GaussianNB()
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.9