"""minisklearn.model_selection —— 模型选择工具。

    - train_test_split: 随机划分训练/测试集
    - KFold:            K 折交叉验证
    - cross_val_score:  交叉验证评估
    - GridSearchCV:     网格搜索 + 交叉验证
"""

from ._split import train_test_split, KFold
from ._search import cross_val_score, GridSearchCV

__all__ = [
    "train_test_split",
    "KFold",
    "cross_val_score",
    "GridSearchCV",
]