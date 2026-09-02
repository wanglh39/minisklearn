"""minisklearn.base —— 基类系统。

本包是理解 sklearn 架构的起点，包含：

    - BaseEstimator:      所有估计器的根基类（参数管理、克隆、repr）
    - ClassifierMixin:    分类器协议（提供 score = accuracy）
    - RegressorMixin:     回归器协议（提供 score = R²）
    - TransformerMixin:   转换器协议（提供 fit_transform）
    - ClusterMixin:       聚类器协议（提供 fit_predict）
    - clone:              克隆函数（元估计器的基石）

使用示例：
    from minisklearn.base import BaseEstimator, ClassifierMixin

    class MyClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, C=1.0):
            self.C = C
        def fit(self, X, y):
            ...
            return self
        def predict(self, X):
            ...
"""

from .base import BaseEstimator, clone
from .mixin import ClassifierMixin, RegressorMixin, TransformerMixin, ClusterMixin

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    "ClusterMixin",
    "clone",
]