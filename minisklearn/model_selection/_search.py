"""模型选择：GridSearchCV + cross_val_score。

==============================================================================
GridSearchCV 原理
==============================================================================

网格搜索 + 交叉验证：

    1. 枚举所有参数组合（param_grid 的笛卡尔积）
    2. 对每个组合用 K 折交叉验证评估
    3. 选平均分最高的组合为最优参数
    4. 用最优参数在全量数据上重训

关键：clone 机制
    每次评估都要 clone 出干净的估计器副本，
    防止前一个参数组合的 fit 状态污染后一个。

==============================================================================
cross_val_score 原理
==============================================================================

    1. 用 KFold 划分数据
    2. 每折 clone 估计器，fit 训练集，score 测试集
    3. 返回 K 个分数
"""

import numpy as np
from itertools import product
from ..base import clone, BaseEstimator
from ..utils.validation import check_X_y, check_array, check_is_fitted
from ._split import KFold


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """用交叉验证评估模型。

    参数：
        estimator: 估计器（会被 clone，不修改原对象）
        X, y: 数据
        cv: 折数（int）或 KFold 对象
        scoring: 评分函数，None 则用 estimator.score

    返回：
        np.ndarray: 每折的分数，shape (cv,)
    """
    X, y = check_X_y(X, y)

    if isinstance(cv, int):
        cv = KFold(n_splits=cv)

    scores = []
    for train_idx, test_idx in cv.split(X):
        # clone 出干净副本
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])

        if scoring is not None:
            score = scoring(y[test_idx], est.predict(X[test_idx]))
        else:
            score = est.score(X[test_idx], y[test_idx])
        scores.append(score)

    return np.array(scores)


class GridSearchCV(BaseEstimator):
    """网格搜索 + 交叉验证的元估计器。

    参数：
        estimator: 基础估计器
        param_grid: 参数网格，dict
            例如 {'C': [0.1, 1, 10], 'max_iter': [100, 200]}
        cv: 折数
        scoring: 评分函数，None 则用 estimator.score

    fit 后的属性：
        best_params_: 最优参数组合
        best_score_: 最优交叉验证分数
        best_estimator_: 用最优参数在全量数据上重训的估计器
        cv_results_: 所有组合的交叉验证结果

    使用示例：
        >>> grid = GridSearchCV(LogisticRegression(), {'C': [0.1, 1, 10]}, cv=5)
        >>> grid.fit(X, y)
        >>> grid.best_params_
        >>> grid.predict(X_test)  # 用 best_estimator_ 预测
    """

    def __init__(self, estimator, param_grid, cv=5, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        """执行网格搜索。"""
        X, y = check_X_y(X, y)

        if isinstance(self.cv, int):
            cv = KFold(n_splits=self.cv)
        else:
            cv = self.cv

        # 枚举所有参数组合（笛卡尔积）
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        best_score = -np.inf
        best_params = None
        all_results = []

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            # 交叉验证评估此参数组合
            scores = []
            for train_idx, test_idx in cv.split(X):
                est = clone(self.estimator)
                est.set_params(**params)
                est.fit(X[train_idx], y[train_idx])

                if self.scoring is not None:
                    score = self.scoring(y[test_idx], est.predict(X[test_idx]))
                else:
                    score = est.score(X[test_idx], y[test_idx])
                scores.append(score)

            mean_score = np.mean(scores)
            all_results.append({
                "params": params,
                "mean_score": mean_score,
                "std_score": np.std(scores),
                "scores": scores,
            })

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = all_results

        # 用最优参数在全量数据上重训
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**best_params)
        self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """用最优估计器预测。"""
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """用最优估计器评分。"""
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.score(X, y)