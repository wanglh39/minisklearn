"""评估指标测试。"""

import numpy as np
import pytest
from minisklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


class TestAccuracyScore:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 0])
        assert accuracy_score(y, y) == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy_score(y_true, y_pred) == 0.0

    def test_partial(self):
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 2])
        assert accuracy_score(y_true, y_pred) == 0.75

    def test_normalize_false(self):
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 2])
        assert accuracy_score(y_true, y_pred, normalize=False) == 3

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            accuracy_score([0, 1], [0])


class TestMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        assert np.isclose(mean_squared_error(y_true, y_pred), 1 / 3)

    def test_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 5])
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        assert np.isclose(rmse, np.sqrt(mse))


class TestMAE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 5])
        assert np.isclose(mean_absolute_error(y_true, y_pred), 2 / 3)


class TestR2Score:
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3, 4])
        assert r2_score(y, y) == 1.0

    def test_mean_prediction(self):
        """预测恒为均值时 R² = 0。"""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.full(4, np.mean(y_true))
        assert np.isclose(r2_score(y_true, y_pred), 0.0)

    def test_worse_than_mean(self):
        """比均值还差时 R² < 0。"""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([4, 3, 2, 1])
        assert r2_score(y_true, y_pred) < 0


class TestConfusionMatrix:
    def test_basic(self):
        y_true = np.array([0, 1, 2, 2, 0])
        y_pred = np.array([0, 2, 2, 2, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2
        assert cm[1, 2] == 1
        assert cm[2, 2] == 2

    def test_diagonal_perfect(self):
        y = np.array([0, 1, 2])
        cm = confusion_matrix(y, y)
        assert np.allclose(np.diag(cm), [1, 1, 1])
        assert cm.sum() == 3

    def test_custom_labels(self):
        y_true = np.array(["a", "b", "a"])
        y_pred = np.array(["a", "a", "a"])
        cm = confusion_matrix(y_true, y_pred, labels=["a", "b"])
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2
        assert cm[1, 0] == 1


class TestPrecisionRecallF1:
    def test_perfect_binary(self):
        y = np.array([0, 1, 0, 1])
        assert precision_score(y, y) == 1.0
        assert recall_score(y, y) == 1.0
        assert f1_score(y, y) == 1.0

    def test_known_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        # TP=2, FP=1, FN=0
        assert np.isclose(precision_score(y_true, y_pred), 2 / 3)
        assert np.isclose(recall_score(y_true, y_pred), 1.0)

    def test_macro_average(self):
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 2])
        p = precision_score(y_true, y_pred, average="macro")
        assert 0 <= p <= 1