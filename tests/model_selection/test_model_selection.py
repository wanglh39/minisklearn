"""model_selection 测试：train_test_split + KFold + cross_val_score + GridSearchCV。"""

import numpy as np
import pytest
from minisklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV,
)
from minisklearn.linear_model import LogisticRegression
from minisklearn.preprocessing import StandardScaler


def make_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    return X, y


class TestTrainTestSplit:
    def test_basic_split(self):
        X, y = make_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert X_train.shape[0] == int(len(X) * 0.8)
        assert X_test.shape[0] == int(len(X) * 0.2)
        assert y_train.shape[0] == X_train.shape[0]

    def test_no_shuffle(self):
        X = np.arange(100).reshape(-1, 1)
        X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
        assert np.all(X_train.ravel() == np.arange(80))
        assert np.all(X_test.ravel() == np.arange(80, 100))

    def test_reproducibility(self):
        X, y = make_data()
        splits1 = train_test_split(X, y, test_size=0.2, random_state=42)
        splits2 = train_test_split(X, y, test_size=0.2, random_state=42)
        for a, b in zip(splits1, splits2):
            assert np.array_equal(a, b)

    def test_multiple_arrays(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        result = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(result) == 4

    def test_int_test_size(self):
        X = np.arange(100).reshape(-1, 1)
        X_train, X_test = train_test_split(X, test_size=30, random_state=42)
        assert X_test.shape[0] == 30


class TestKFold:
    def test_basic(self):
        X = np.arange(100).reshape(-1, 1)
        kf = KFold(n_splits=5)
        splits = list(kf.split(X))
        assert len(splits) == 5

    def test_no_overlap(self):
        X = np.arange(100).reshape(-1, 1)
        kf = KFold(n_splits=5)
        for train_idx, test_idx in kf.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(train_idx) + len(test_idx) == 100

    def test_all_samples_tested(self):
        """K 折中每个样本都应恰好被测试一次。"""
        X = np.arange(100).reshape(-1, 1)
        kf = KFold(n_splits=5)
        all_test = []
        for _, test_idx in kf.split(X):
            all_test.extend(test_idx)
        assert sorted(all_test) == list(range(100))

    def test_get_n_splits(self):
        kf = KFold(n_splits=7)
        assert kf.get_n_splits() == 7


class TestCrossValScore:
    def test_basic(self):
        X, y = make_data()
        clf = LogisticRegression(max_iter=500, learning_rate=0.5)
        scores = cross_val_score(clf, X, y, cv=5)
        assert scores.shape == (5,)
        assert np.all(scores > 0.7)

    def test_does_not_modify_estimator(self):
        """cross_val_score 不应修改原估计器。"""
        X, y = make_data()
        clf = LogisticRegression(max_iter=500)
        cross_val_score(clf, X, y, cv=3)
        assert not hasattr(clf, "coef_")  # 原对象未被 fit


class TestGridSearchCV:
    def test_basic_search(self):
        X, y = make_data()
        grid = GridSearchCV(
            LogisticRegression(max_iter=500, learning_rate=0.5),
            param_grid={"C": [0.1, 1.0, 10.0]},
            cv=3,
        )
        grid.fit(X, y)
        assert "C" in grid.best_params_
        assert grid.best_score_ > 0.7

    def test_best_estimator_predict(self):
        X, y = make_data()
        grid = GridSearchCV(
            LogisticRegression(max_iter=500, learning_rate=0.5),
            param_grid={"C": [0.1, 1.0]},
            cv=3,
        )
        grid.fit(X, y)
        y_pred = grid.predict(X)
        assert len(y_pred) == len(y)

    def test_cv_results(self):
        X, y = make_data()
        grid = GridSearchCV(
            LogisticRegression(max_iter=500, learning_rate=0.5),
            param_grid={"C": [0.1, 1.0, 10.0]},
            cv=3,
        )
        grid.fit(X, y)
        assert len(grid.cv_results_) == 3  # 3 个参数组合

    def test_multiple_params(self):
        X, y = make_data()
        grid = GridSearchCV(
            LogisticRegression(max_iter=500, learning_rate=0.5),
            param_grid={"C": [0.1, 1.0], "max_iter": [200, 500]},
            cv=3,
        )
        grid.fit(X, y)
        assert len(grid.cv_results_) == 4  # 2 * 2 = 4 个组合