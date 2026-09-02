"""LabelEncoder 和 OneHotEncoder 测试。"""

import numpy as np
import pytest
from minisklearn.preprocessing import LabelEncoder, OneHotEncoder
from minisklearn.base import clone


class TestLabelEncoder:
    """LabelEncoder 测试。"""

    def test_fit_transform_basic(self):
        le = LabelEncoder()
        result = le.fit_transform(["猫", "狗", "鸟", "猫"])
        assert len(result) == 4
        assert result[0] == result[3]
        assert len(np.unique(result)) == 3

    def test_classes_sorted(self):
        le = LabelEncoder()
        le.fit(["c", "a", "b"])
        assert list(le.classes_) == ["a", "b", "c"]

    def test_transform_new_data(self):
        le = LabelEncoder()
        le.fit(["a", "b", "c"])
        result = le.transform(["c", "a"])
        assert list(result) == [2, 0]

    def test_inverse_transform(self):
        le = LabelEncoder()
        y = ["猫", "狗", "鸟"]
        encoded = le.fit_transform(y)
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == y

    def test_unknown_label_raises(self):
        le = LabelEncoder()
        le.fit(["a", "b"])
        with pytest.raises(ValueError, match="未见"):
            le.transform(["c"])

    def test_numeric_labels(self):
        le = LabelEncoder()
        result = le.fit_transform([10, 20, 10, 30])
        assert len(np.unique(result)) == 3
        assert result[0] == result[2]

    def test_clone(self):
        le = LabelEncoder()
        le.fit(["a", "b"])
        cloned = clone(le)
        assert not hasattr(cloned, "classes_")


class TestOneHotEncoder:
    """OneHotEncoder 测试。"""

    def test_fit_transform_basic(self):
        enc = OneHotEncoder()
        X = np.array([["猫"], ["狗"], ["鸟"]])
        result = enc.fit_transform(X)
        assert result.shape == (3, 3)
        assert np.allclose(result.sum(axis=1), 1)

    def test_transform_new_data(self):
        enc = OneHotEncoder()
        X_train = np.array([["a"], ["b"], ["c"]])
        enc.fit(X_train)
        X_new = np.array([["c"], ["a"]])
        result = enc.transform(X_new)
        assert result.shape == (2, 3)
        assert np.allclose(result[0], [0, 0, 1])
        assert np.allclose(result[1], [1, 0, 0])

    def test_multiple_features(self):
        enc = OneHotEncoder()
        X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
        result = enc.fit_transform(X)
        assert result.shape == (3, 4)

    def test_inverse_transform(self):
        enc = OneHotEncoder()
        X = np.array([["a"], ["b"], ["c"]])
        encoded = enc.fit_transform(X)
        decoded = enc.inverse_transform(encoded)
        assert np.all(decoded == X)

    def test_unknown_category_raises(self):
        enc = OneHotEncoder()
        enc.fit(np.array([["a"], ["b"]]))
        with pytest.raises(ValueError, match="未见"):
            enc.transform(np.array([["c"]]))

    def test_1d_input(self):
        enc = OneHotEncoder()
        result = enc.fit_transform(["a", "b", "c"])
        assert result.shape == (3, 3)

    def test_clone(self):
        enc = OneHotEncoder()
        cloned = clone(enc)
        assert not hasattr(cloned, "categories_")