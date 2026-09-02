"""minisklearn —— 从零实现的 sklearn 教学项目。

本包旨在通过从零实现 sklearn 的核心算法，帮助学习者理解：

    1. sklearn 的架构设计哲学（统一 API、Mixin、元估计器等）
    2. 机器学习算法的数学原理与底层实现
    3. 工程实践中的防御式编程、参数管理、克隆机制

快速开始：
    >>> from minisklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> scaler.fit([[1], [2], [3]])
    >>> scaler.transform([[1], [2], [3]])

架构导览：
    - base:          基类系统（BaseEstimator + 4 个 Mixin）
    - utils:         数据校验与通用工具
    - preprocessing: 数据预处理
    - linear_model:  线性模型
    - neighbors:     K 近邻
    - tree:          决策树
    - ensemble:      集成学习
    - cluster:       聚类
    - decomposition: 降维
    - svm:           支持向量机
    - naive_bayes:   朴素贝叶斯
    - model_selection: 模型选择
    - pipeline:      流水线
    - metrics:       评估指标
"""

__version__ = "0.1.0"

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    ClusterMixin,
    clone,
)

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    "ClusterMixin",
    "clone",
    "__version__",
]