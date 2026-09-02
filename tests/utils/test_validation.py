"""数据校验工具测试。

测试目录镜像 minisklearn/utils/ 的结构。
"""

import numpy as np
import pytest
from minisklearn.utils.validation import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_random_state,
)
from minisklearn.base import BaseEstimator
from minisklearn.exceptions import NotFittedError


class TestCheckArray:
    """测试 check_array。"""

    def test_accepts_list(self):
        result = check_array([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_accepts_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        result = check_array(arr)
        assert isinstance(result, np.ndarray)

    def test_1d_becomes_2d(self):
        result = check_array([1, 2, 3])
        assert result.shape == (3, 1)

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="二维"):
            check_array(np.zeros((2, 2, 2)))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            check_array([[1, np.nan], [3, 4]])

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="无穷大"):
            check_array([[1, np.inf], [3, 4]])

    def test_allows_nan_when_disabled(self):
        result = check_array([[1, np.nan]], force_all_finite=False)
        assert result.shape == (1, 2)

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            check_array(None)

    def test_object_dtype_converted(self):
        result = check_array([["1", "2"], ["3", "4"]])
        assert result.dtype == np.float64


class TestCheckXY:
    """测试 check_X_y。"""

    def test_basic(self):
        X, y = check_X_y([[1, 2], [3, 4]], [0, 1])
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_sample_count_mismatch(self):
        with pytest.raises(ValueError, match="不一致"):
            check_X_y([[1], [2], [3]], [0, 1])

    def test_rejects_none_y(self):
        with pytest.raises(ValueError):
            check_X_y([[1], [2]], None)


class TestCheckIsFitted:
    """测试 check_is_fitted。"""

    def test_fitted_estimator_passes(self):
        class Dummy(BaseEstimator):
            def __init__(self):
                pass

        est = Dummy()
        est.coef_ = np.array([1, 2])
        check_is_fitted(est, ["coef_"])

    def test_unfitted_raises(self):
        class Dummy(BaseEstimator):
            def __init__(self):
                pass

        est = Dummy()
        with pytest.raises(NotFittedError):
            check_is_fitted(est, ["coef_"])

    def test_auto_detect_attributes(self):
        class Dummy(BaseEstimator):
            def __init__(self):
                pass

        est = Dummy()
        est.coef_ = np.array([1])
        check_is_fitted(est)


class TestCheckRandomState:
    """测试 check_random_state。"""

    def test_none_returns_random_state(self):
        rs = check_random_state(None)
        assert isinstance(rs, np.random.RandomState)

    def test_int_returns_seeded(self):
        rs1 = check_random_state(42)
        rs2 = check_random_state(42)
        assert rs1.rand() == rs2.rand()

    def test_passes_through_random_state(self):
        rs = np.random.RandomState(123)
        assert check_random_state(rs) is rs

    def test_rejects_invalid(self):
        with pytest.raises(TypeError):
            check_random_state("invalid")