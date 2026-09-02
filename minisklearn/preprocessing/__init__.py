"""minisklearn.preprocessing —— 数据预处理。

本包提供特征缩放和编码工具：

    - StandardScaler: 标准化（均值0、标准差1）
    - MinMaxScaler:   归一化到指定范围
    - LabelEncoder:   标签编码（类别 → 整数）
    - OneHotEncoder:  独热编码（类别 → 二值向量）
"""

from ._scalers import StandardScaler, MinMaxScaler
from ._encoders import LabelEncoder, OneHotEncoder

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "OneHotEncoder",
]