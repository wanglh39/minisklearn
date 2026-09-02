"""随机森林测试。"""

import numpy as np
import pytest
from minisklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from minisklearn.base import clone


def make_classification_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 4) + np.array([2, 2, 0, 0])
    X2 = rng.randn(n, 4) + np.array([-2, -2, 0, 0])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


def make_regression_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-5, 5, size=(n, 3))
    y = X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2] + rng.randn(n) * 0.5
    return X, y


class TestRandomForestClassifier:
    def test_fit_predict_basic(self):
        X, y = make_classification_data()
        clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.85

    def test_multiple_trees(self):
        X, y = make_classification_data()
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        assert len(clf.estimators_) == 10

    def test_predict_proba(self):
        X, y = make_classification_data()
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_reproducibility(self):
        X, y = make_classification_data()
        clf1 = RandomForestClassifier(n_estimators=10, random_state=42)
        clf1.fit(X, y)
        clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
        clf2.fit(X, y)
        assert np.all(clf1.predict(X) == clf2.predict(X))

    def test_no_bootstrap(self):
        X, y = make_classification_data(n=30)
        clf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=42)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        centers = [[2, 2], [-2, 2], [0, -2]]
        X = np.vstack([rng.randn(40, 2) + c for c in centers])
        y = np.array([0] * 40 + [1] * 40 + [2] * 40)
        clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.8

    def test_clone(self):
        clf = RandomForestClassifier(n_estimators=50, max_depth=3)
        cloned = clone(clf)
        assert cloned.n_estimators == 50
        assert cloned.max_depth == 3
        assert not hasattr(cloned, "estimators_")

    def test_repr(self):
        clf = RandomForestClassifier(n_estimators=50)
        assert "RandomForestClassifier" in repr(clf)

    def test_string_labels(self):
        X, y = make_classification_data(n=30)
        y_str = np.where(y == 1, "正", "负")
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y_str)
        y_pred = clf.predict(X)
        assert set(y_pred) <= {"正", "负"}


class TestRandomForestRegressor:
    def test_fit_predict_basic(self):
        X, y = make_regression_data()
        reg = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 5.0

    def test_score(self):
        X, y = make_regression_data()
        reg = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42)
        reg.fit(X, y)
        score = reg.score(X, y)
        assert score > 0.5

    def test_multiple_trees(self):
        X, y = make_regression_data(n=50)
        reg = RandomForestRegressor(n_estimators=15, random_state=42)
        reg.fit(X, y)
        assert len(reg.estimators_) == 15

    def test_reproducibility(self):
        X, y = make_regression_data(n=50)
        reg1 = RandomForestRegressor(n_estimators=10, random_state=42)
        reg1.fit(X, y)
        reg2 = RandomForestRegressor(n_estimators=10, random_state=42)
        reg2.fit(X, y)
        assert np.allclose(reg1.predict(X), reg2.predict(X))

    def test_clone(self):
        reg = RandomForestRegressor(n_estimators=50, max_depth=3)
        cloned = clone(reg)
        assert cloned.n_estimators == 50
        assert not hasattr(cloned, "estimators_")