"""minisklearn.tree —— 决策树。

本包提供 CART 决策树分类器和回归器：

    - DecisionTreeClassifier: 分类决策树（基尼系数）
    - DecisionTreeRegressor:  回归决策树（MSE）
"""

from ._tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]