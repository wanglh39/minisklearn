"""Pipeline —— 元估计器流水线。

==============================================================================
Pipeline 原理
==============================================================================

Pipeline 串联多个步骤，前几步是转换器，最后一步是估计器。

fit 链 vs predict 链（核心设计）：

    fit:
        对每步（除最后一步）：X = step.fit_transform(X, y)
        最后一步：step.fit(X, y)
        → 转换器要学习参数 + 转换数据

    predict:
        对每步（除最后一步）：X = step.transform(X)
        最后一步：step.predict(X)
        → 转换器只用已学参数转换，不重新学习

    这个区别保证了 predict 用的转换参数是 fit 时学到的，不会数据泄露。

嵌套参数命名：
    Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1))])
    → get_params 返回 {'scaler__with_mean': True, 'clf__C': 1, ...}
    → set_params(clf__C=10) 设置嵌套参数
    → GridSearchCV 可以搜索 Pipeline 内部任意步骤的参数
"""

import numpy as np
from ..base import BaseEstimator, clone
from ..utils.validation import check_array, check_is_fitted


class Pipeline(BaseEstimator):
    """估计器流水线。

    参数：
        steps: 步骤列表 [(name, estimator), ...]
            最后一步必须是预测器（有 predict），
            前面的步骤是转换器（有 transform）。

    fit 后的属性：
        named_steps: 按名称访问步骤的字典

    使用示例：
        >>> pipe = Pipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('clf', LogisticRegression()),
        ... ])
        >>> pipe.fit(X_train, y_train)
        >>> pipe.predict(X_test)
        >>> pipe.set_params(clf__C=10)  # 设置嵌套参数
    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        """按名称访问步骤。"""
        return dict(self.steps)

    def get_params(self, deep=True):
        """获取 Pipeline 的所有超参数，包括嵌套步骤参数。

        Pipeline 的 __init__ 只有 steps 一个参数，但用户需要访问各步骤
        内部估计器的参数（如 clf__C）。因此覆盖 BaseEstimator.get_params，
        遍历每个步骤并展平为 'stepname__param' 格式。

        参数：
            deep: 是否递归获取各步骤估计器的参数

        返回：
            dict: 包含 'steps' 和所有 'stepname__param' 键的字典
        """
        out = {"steps": self.steps}
        if deep:
            for name, step in self.steps:
                if hasattr(step, "get_params"):
                    for sub_key, sub_value in step.get_params(deep=True).items():
                        out[f"{name}__{sub_key}"] = sub_value
        return out

    def set_params(self, **params):
        """设置 Pipeline 的参数，支持嵌套命名 'stepname__param'。

        这是 GridSearchCV 能搜索 Pipeline 内部参数的关键：
        pipe.set_params(clf__C=10) 会找到名为 'clf' 的步骤，
        并在其估计器上调用 set_params(C=10)。

        参数：
            **params: 要设置的参数，支持 'stepname__param' 嵌套命名

        返回：
            self
        """
        if not params:
            return self

        step_names = [name for name, _ in self.steps]
        nested_params = {}

        for key, value in params.items():
            if "__" in key:
                step_name, sub_param = key.split("__", 1)
                if step_name not in step_names:
                    raise ValueError(
                        f"无效参数 {step_name}：不是 Pipeline 的步骤名"
                    )
                nested_params.setdefault(step_name, {})[sub_param] = value
            else:
                if key != "steps":
                    raise ValueError(
                        f"无效参数 {key}：不是 Pipeline 的超参数"
                    )
                setattr(self, key, value)

        for step_name, sub_params in nested_params.items():
            step_dict = dict(self.steps)
            step_dict[step_name].set_params(**sub_params)

        return self

    def __sklearn_clone__(self):
        """克隆 Pipeline：对每个步骤的估计器调用 clone。

        覆盖默认克隆行为，确保每个步骤得到一个未训练的新副本，
        而非 deepcopy（会复制拟合状态）。
        """
        new_steps = [(name, clone(step)) for name, step in self.steps]
        return Pipeline(steps=new_steps)

    def _validate_steps(self):
        """校验步骤：前几步有 transform，最后一步有 predict。"""
        if len(self.steps) < 1:
            raise ValueError("Pipeline 至少需要一个步骤")

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"中间步骤 '{name}' 必须有 transform 方法"
                )

        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "fit"):
            raise TypeError(
                f"最后一步 '{final_name}' 必须有 fit 方法"
            )

    def fit(self, X, y=None):
        """拟合流水线：每步 fit_transform，最后一步 fit。"""
        self._validate_steps()

        # fit 链：转换器 fit_transform，数据逐步传递
        for name, step in self.steps[:-1]:
            if y is not None:
                X = step.fit_transform(X, y)
            else:
                X = step.fit_transform(X)

        # 最后一步只 fit
        final_name, final_step = self.steps[-1]
        if y is not None:
            final_step.fit(X, y)
        else:
            final_step.fit(X)

        return self

    def predict(self, X):
        """预测：每步 transform，最后一步 predict。"""
        check_is_fitted(self, ["steps"])
        self._validate_steps()

        # predict 链：转换器只 transform
        for name, step in self.steps[:-1]:
            X = step.transform(X)

        return self.steps[-1][1].predict(X)

    def transform(self, X):
        """转换：所有步骤都 transform（当最后一步也是转换器时）。"""
        self._validate_steps()
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """拟合并转换。"""
        self._validate_steps()

        for name, step in self.steps[:-1]:
            if y is not None:
                X = step.fit_transform(X, y)
            else:
                X = step.fit_transform(X)

        final_step = self.steps[-1][1]
        if hasattr(final_step, "fit_transform"):
            if y is not None:
                return final_step.fit_transform(X, y)
            else:
                return final_step.fit_transform(X)
        else:
            if y is not None:
                final_step.fit(X, y)
            else:
                final_step.fit(X)
            return final_step.predict(X)

    def score(self, X, y):
        """评分：用最后一步估计器的 score。"""
        check_is_fitted(self, ["steps"])
        self._validate_steps()

        for name, step in self.steps[:-1]:
            X = step.transform(X)

        return self.steps[-1][1].score(X, y)