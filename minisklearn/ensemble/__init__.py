"""minisklearn.ensemble —— 集成学习。

本包提供随机森林：

    - RandomForestClassifier: 随机森林分类
    - RandomForestRegressor:  随机森林回归
"""

from ._forest import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
]