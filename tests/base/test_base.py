"""BaseEstimator 测试 —— 验证参数管理、克隆、repr。

测试目录镜像 minisklearn/base/ 的结构（tests/base/test_base.py）。
"""

import numpy as np
import pytest
from minisklearn.base import BaseEstimator, clone, ClassifierMixin, RegressorMixin
from minisklearn.exceptions import NotFittedError


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """用于测试的虚拟分类器。"""

    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class DummyRegressor(BaseEstimator, RegressorMixin):
    """用于测试的虚拟回归器。"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y):
        self.coef_ = np.mean(X, axis=0)
        return self

    def predict(self, X):
        return X @ self.coef_


class TestGetParams:
    """测试 get_params 的反射机制。"""

    def test_returns_all_init_params(self):
        clf = DummyClassifier(C=2.0, max_iter=50)
        params = clf.get_params()
        assert params == {"C": 2.0, "max_iter": 50}

    def test_deep_with_nested_estimator(self):
        class Wrapper(BaseEstimator):
            def __init__(self, inner=DummyClassifier()):
                self.inner = inner

        wrapper = Wrapper(inner=DummyClassifier(C=3.0))
        params = wrapper.get_params(deep=True)
        assert "inner" in params
        assert "inner__C" in params
        assert params["inner__C"] == 3.0

    def test_shallow_excludes_nested(self):
        class Wrapper(BaseEstimator):
            def __init__(self, inner=DummyClassifier()):
                self.inner = inner

        wrapper = Wrapper()
        params = wrapper.get_params(deep=False)
        assert "inner" in params
        assert "inner__C" not in params


class TestSetParams:
    """测试 set_params。"""

    def test_set_single_param(self):
        clf = DummyClassifier()
        clf.set_params(C=5.0)
        assert clf.C == 5.0

    def test_set_multiple_params(self):
        clf = DummyClassifier()
        clf.set_params(C=5.0, max_iter=200)
        assert clf.C == 5.0
        assert clf.max_iter == 200

    def test_set_invalid_param_raises(self):
        clf = DummyClassifier()
        with pytest.raises(ValueError, match="无效参数"):
            clf.set_params(nonexistent=1.0)

    def test_set_nested_param(self):
        class Wrapper(BaseEstimator):
            def __init__(self, inner=DummyClassifier()):
                self.inner = inner

        wrapper = Wrapper()
        wrapper.set_params(inner__C=10.0)
        assert wrapper.inner.C == 10.0

    def test_returns_self(self):
        clf = DummyClassifier()
        result = clf.set_params(C=1.0)
        assert result is clf


class TestClone:
    """测试 clone 机制。"""

    def test_clone_same_type(self):
        clf = DummyClassifier(C=2.0)
        cloned = clone(clf)
        assert type(cloned) == DummyClassifier

    def test_clone_same_params(self):
        clf = DummyClassifier(C=2.0, max_iter=50)
        cloned = clone(clf)
        assert cloned.C == 2.0
        assert cloned.max_iter == 50

    def test_clone_is_unfitted(self):
        """clone 出的对象应该是未训练的。"""
        clf = DummyClassifier()
        clf.fit(np.array([[1], [2]]), np.array([0, 1]))
        assert hasattr(clf, "coef_")

        cloned = clone(clf)
        assert not hasattr(cloned, "coef_")

    def test_clone_is_different_object(self):
        clf = DummyClassifier(C=2.0)
        cloned = clone(clf)
        assert cloned is not clf

    def test_clone_preserves_nested(self):
        class Wrapper(BaseEstimator):
            def __init__(self, inner=DummyClassifier()):
                self.inner = inner

        wrapper = Wrapper(inner=DummyClassifier(C=5.0))
        cloned = clone(wrapper)
        assert cloned.inner.C == 5.0


class TestRepr:
    """测试 __repr__ 自动生成。"""

    def test_repr_contains_class_name(self):
        clf = DummyClassifier()
        assert "DummyClassifier" in repr(clf)

    def test_repr_contains_params(self):
        clf = DummyClassifier(C=2.0, max_iter=50)
        r = repr(clf)
        assert "C=2.0" in r
        assert "max_iter=50" in r


class TestEstimatorType:
    """测试 _estimator_type 标识。"""

    def test_classifier_type(self):
        clf = DummyClassifier()
        assert clf._estimator_type == "classifier"

    def test_regressor_type(self):
        reg = DummyRegressor()
        assert reg._estimator_type == "regressor"


class TestScore:
    """测试 Mixin 提供的 score 方法。"""

    def test_classifier_score(self):
        clf = DummyClassifier()
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 0, 0])
        clf.fit(X, y)
        score = clf.score(X, y)
        assert 0 <= score <= 1

    def test_regressor_score(self):
        reg = DummyRegressor()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])
        reg.fit(X, y)
        score = reg.score(X, y)
        assert score <= 1.0