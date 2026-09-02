"""ClassifierMixin —— 分类器混入。

==============================================================================
Mixin 设计哲学
==============================================================================

为什么用 Mixin 而不是大基类或抽象基类？

    场景：分类器需要 score 方法（返回准确率），回归器也需要 score（返回 R²），
    但两者计算方式完全不同。如果用一个大的 BaseEstimator 包揽所有，
    score 方法里就要写 if-else 判断"我是分类器还是回归器"——这是坏味道。

    sklearn 的解法：拆成 Mixin。ClassifierMixin 只提供分类版的 score，
    RegressorMixin 只提供回归版的 score。算法类按需多继承：

        class LogisticRegression(BaseEstimator, ClassifierMixin): ...
        class LinearRegression(BaseEstimator, RegressorMixin): ...

    好处：
        1. 职责单一：每个 Mixin 只管自己那一份协议
        2. 组合灵活：一个类可以同时是 Transformer + Classifier
           （如 LDA 既降维又分类）
        3. 无状态：Mixin 不定义 __init__，不存任何属性，纯方法集合

    代价：
        1. 多继承的 MRO 需要理解（但 sklearn 的 Mixin 很简单，不冲突）
        2. 缺少编译期检查（靠 check_estimator 测试套件弥补）
"""

import numpy as np
from ..utils.validation import check_array


class ClassifierMixin:
    """分类器 Mixin，提供分类器的通用方法。

    子类需要实现：
        - fit(X, y)
        - predict(X)

    本 Mixin 提供：
        - score(X, y) —— 返回分类准确率
        - _estimator_type —— 标识为 "classifier"
    """

    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """返回预测准确率（accuracy）。

        为什么 score 默认用 accuracy？
            因为 accuracy 是分类问题最通用的指标，无需额外参数。
            如果要算 F1、AUC 等，应该用 metrics 模块的对应函数。
            score 的存在是为了让 GridSearchCV 等元估计器有一个默认优化目标。

        参数：
            X: 测试特征，shape (n_samples, n_features)
            y: 测试标签，shape (n_samples,)
            sample_weight: 样本权重（可选）

        返回：
            float: 准确率，范围 [0, 1]
        """
        X = check_array(X)
        y = np.asarray(y)
        y_pred = self.predict(X)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            return float(np.average(y_pred == y, weights=sample_weight))
        return float(np.mean(y_pred == y))


class RegressorMixin:
    """回归器 Mixin，提供回归器的通用方法。

    子类需要实现：
        - fit(X, y)
        - predict(X)

    本 Mixin 提供：
        - score(X, y) —— 返回 R² 决定系数
        - _estimator_type —— 标识为 "regressor"
    """

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """返回 R² 决定系数。

        为什么回归用 R² 而不是 MSE？
            因为 R² 是无量纲的，范围通常在 [0, 1]（可以为负），
            方便跨问题比较。而 MSE 依赖 y 的量纲，不同数据集无法直接比。
            GridSearchCV 默认用 score 作为优化目标，R² 比 MSE 更适合。

        R² = 1 - SS_res / SS_tot
            SS_res = Σ(y_true - y_pred)²
            SS_tot = Σ(y_true - mean(y_true))²

        参数：
            X: 测试特征
            y: 测试真值
            sample_weight: 样本权重（可选）

        返回：
            float: R² 分数，越接近 1 越好
        """
        X = check_array(X)
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            weight_sum = np.sum(sample_weight)
            mean_y = np.sum(y * sample_weight) / weight_sum
            ss_res = np.sum(sample_weight * (y - y_pred) ** 2)
            ss_tot = np.sum(sample_weight * (y - mean_y) ** 2)
        else:
            mean_y = np.mean(y)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - mean_y) ** 2)

        if ss_tot == 0:
            # 真值全相同的情况，R² 无定义，按 sklearn 惯例返回 1.0
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)


class TransformerMixin:
    """转换器 Mixin，提供转换器的通用方法。

    子类需要实现：
        - fit(X[, y])
        - transform(X)

    本 Mixin 提供：
        - fit_transform(X[, y]) —— 默认实现为 fit 后 transform
        - _estimator_type —— 标识为 "transformer"
    """

    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, **fit_params):
        """拟合数据然后转换。

        为什么要有 fit_transform？
            1. 语义清晰：一步到位
            2. 性能优化机会：某些转换器（如 PCA）在 fit 时已经算出了
               转换结果，fit_transform 可以复用，避免 transform 重复计算。
               子类可覆盖此方法以优化性能，默认实现就是 fit + transform。
            3. Pipeline 的需要：Pipeline.fit_transform 会调用每一步的
               fit_transform，没有就用 fit+transform 兜底。
        """
        if y is None:
            self.fit(X, **fit_params)
        else:
            self.fit(X, y, **fit_params)
        return self.transform(X)


class ClusterMixin:
    """聚类器 Mixin，提供聚类器的通用方法。

    子类需要实现：
        - fit(X)
        - predict(X)（可选，有些聚类器不实现）

    本 Mixin 提供：
        - fit_predict(X) —— 拟合并返回聚类标签
        - _estimator_type —— 标识为 "clusterer"
    """

    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **fit_params):
        """拟合并返回聚类标签。

        与 TransformerMixin.fit_transform 类似的设计动机：
        语义 + 性能。某些聚类算法（如 DBSCAN）在 fit 过程中就已经
        确定了所有点的标签，fit_predict 可以直接返回，无需再走 predict。
        """
        self.fit(X, **fit_params)
        return self.labels_