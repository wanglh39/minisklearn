"""minisklearn.utils —— 工具与校验。

本包提供与具体算法无关的通用工具：

    - validation.check_array:      校验并规范化输入数组
    - validation.check_X_y:        校验特征矩阵和标签
    - validation.check_is_fitted:  检查估计器是否已训练
    - validation.check_random_state: 统一随机种子处理
"""

from .validation import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_random_state,
)

__all__ = [
    "check_array",
    "check_X_y",
    "check_is_fitted",
    "check_random_state",
]