"""线性支持向量机（LinearSVC）。

==============================================================================
SVM 原理
==============================================================================

SVM 寻找最大间隔超平面：

    max margin = 2 / ||w||
    s.t. y_i (w·x_i + b) >= 1  for all i

等价的软间隔形式（hinge loss）：

    min (1/2) ||w||² + C Σ max(0, 1 - y_i(w·x_i + b))

hinge loss 的次梯度：

    ∂L/∂w = w - C Σ_{i ∈ violated} y_i x_i
    ∂L/∂b = - C Σ_{i ∈ violated} y_i

    其中 violated = {i : y_i(w·x_i + b) < 1}

用梯度下降求解（Pegasos 风格）：
    每次取一个样本，如果违反间隔约束则更新：
    w -= lr * (w - C * y_i * x_i)
    b -= lr * (-C * y_i)
    否则只正则化：
    w -= lr * w

多分类用 OvR 策略。
"""

import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted


class LinearSVC(BaseEstimator, ClassifierMixin):
    """线性支持向量机分类器。

    用梯度下降优化 hinge loss，支持 L2 正则化。

    参数：
        C: 正则化强度的倒数（默认 1.0）
        max_iter: 最大迭代次数
        learning_rate: 学习率
        tol: 收敛阈值

    fit 后的属性：
        coef_: 权重，shape (n_classes, n_features) 多分类，或 (n_features,) 二分类
        intercept_: 截距
        classes_: 类别标签

    使用示例：
        >>> clf = LinearSVC(C=1.0, max_iter=1000)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.01, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y):
        """拟合 SVM 模型。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # 二分类：y ∈ {-1, +1}
            y_pm = np.where(y == self.classes_[1], 1, -1).astype(np.float64)
            self.coef_, self.intercept_ = self._fit_binary(X, y_pm)
        else:
            # 多分类：OvR
            self._fit_ovr(X, y)

        return self

    def _fit_binary(self, X, y):
        """训练二分类 SVM（y ∈ {-1, +1}）。"""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.max_iter):
            # 计算所有样本的间隔
            margins = y * (X @ w + b)

            # hinge loss 的次梯度
            # 违反间隔的样本：y_i(w·x_i + b) < 1
            violated = margins < 1

            grad_w = w - self.C * np.sum(X[violated] * y[violated, None], axis=0)
            grad_b = -self.C * np.sum(y[violated])

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

            if np.linalg.norm(grad_w) < self.tol:
                break

        return w, b

    def _fit_ovr(self, X, y):
        """多分类 OvR 策略。"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            y_pm = np.where(y == cls, 1, -1).astype(np.float64)
            w, b = self._fit_binary(X, y_pm)
            self.coef_[i] = w
            self.intercept_[i] = b

    def decision_function(self, X):
        """计算决策函数值。"""
        check_is_fitted(self, ["coef_"])
        X = check_array(X)


        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """预测类别标签。"""
        check_is_fitted(self, ["coef_", "classes_"])
        scores = self.decision_function(X)

        if scores.ndim == 1:
            idx = (scores >= 0).astype(int)
        else:
            idx = np.argmax(scores, axis=1)

        return self.classes_[idx]