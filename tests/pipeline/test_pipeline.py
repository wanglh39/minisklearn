"""Pipeline 测试。"""

import numpy as np
import pytest
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression, LinearRegression
from minisklearn.base import clone


def make_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


class TestPipeline:
    def test_fit_predict(self):
        X, y = make_data()
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, learning_rate=0.5)),
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        acc = np.mean(y_pred == y)
        assert acc > 0.9

    def test_named_steps(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_score(self):
        X, y = make_data()
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, learning_rate=0.5)),
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert score > 0.9

    def test_set_params_nested(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0)),
        ])
        pipe.set_params(clf__C=10.0)
        assert pipe.named_steps["clf"].C == 10.0

    def test_get_params_nested(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0)),
        ])
        params = pipe.get_params(deep=True)
        assert "clf__C" in params
        assert params["clf__C"] == 1.0

    def test_clone(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        cloned = clone(pipe)
        assert "scaler" in cloned.named_steps
        assert "clf" in cloned.named_steps

    def test_regression_pipeline(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.randn(100) * 0.1
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert score > 0.8

    def test_repr(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        assert "Pipeline" in repr(pipe)