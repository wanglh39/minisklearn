"""minisklearn.neighbors —— K 近邻算法。

本包提供 KNN 分类器和回归器：

    - KNeighborsClassifier: K 近邻分类
    - KNeighborsRegressor:  K 近邻回归
"""

from ._knn import KNeighborsClassifier, KNeighborsRegressor
from ._distances import euclidean_distances, manhattan_distances, find_k_neighbors

__all__ = [
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "euclidean_distances",
    "manhattan_distances",
    "find_k_neighbors",
]