"""逻辑回归：LogisticRegression。

本模块实现二分类和多分类逻辑回归，使用梯度下降优化 + L2 正则化。

==============================================================================
数学原理
==============================================================================

二分类逻辑回归：

    模型：p = σ(Xw + b) = 1 / (1 + exp(-(Xw + b)))
    预测：ŷ = 1[p > 0.5]

    损失函数（交叉熵 + L2 正则）：
    L(w) = -(1/n) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)] + (1/2C) ||w||²

    梯度：
    ∂L/∂w = (1/n) X^T (p - y) + (1/C) w
    ∂L/∂b = (1/n) Σ (p_i - y_i)

    梯度下降更新：
    w -= lr * ∂L/∂w
    b -= lr * ∂L/∂b

多分类（OvR 策略）：
    对每个类别训练一个二分类器，预测时取置信度最高的类别。
    sklearn 默认也是 OvR（对 LogisticRegression 用 lbfgs 时是 multinomial，
    但 liblinear 求解器是 OvR）。

==============================================================================
数值稳定性
==============================================================================

sigmoid 函数在 z 很大或很小时会溢出：
    z = -100 → exp(100) = inf → 1/(1+inf) = 0  （OK）
    z = +100 → exp(-100) ≈ 0 → 1/(1+0) = 1    （OK）

但中间计算可能溢出。稳定实现：
    σ(z) = z >= 0 ? 1/(1+exp(-z)) : exp(z)/(1+exp(z))

log-sigmoid 也需要稳定实现以避免 log(0)。
"""

import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted


def _sigmoid(z):
    """数值稳定的 sigmoid 函数。

    对 z >= 0: σ(z) = 1 / (1 + exp(-z))
    对 z < 0:  σ(z) = exp(z) / (1 + exp(z))

    分段处理避免 exp 溢出。
    """
    # 向量化实现：对正负分别处理
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)

    return out


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """逻辑回归分类器。

    用梯度下降优化交叉熵损失，支持 L2 正则化。

    参数：
        C: 正则化强度的倒数（默认 1.0）。C 越大正则越弱。
           等价于 sklearn 的 C 参数。
        max_iter: 最大迭代次数
        learning_rate: 梯度下降学习率
        fit_intercept: 是否拟合截距
        tol: 收敛阈值（梯度范数小于此值时提前停止）

    fit 后的属性：
        coef_: 系数，shape (n_classes, n_features) 多分类，或 (n_features,) 二分类
        intercept_: 截距
        classes_: 类别标签，shape (n_classes,)
        n_features_in_: 输入特征数

    使用示例：
        >>> clf = LogisticRegression(C=1.0, max_iter=200)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, C=1.0, max_iter=100, learning_rate=0.1,
                 fit_intercept=True, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y):
        """拟合逻辑回归模型。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # 二分类
            y_binary = (y == self.classes_[1]).astype(np.float64)
            self.coef_, self.intercept_ = self._fit_binary(X, y_binary)
        else:
            # 多分类：OvR
            self._fit_ovr(X, y)

        return self

    def _fit_binary(self, X, y):
        """训练单个二分类逻辑回归，返回 (w, b)。

        梯度下降（sklearn 等价形式）：
            p = σ(Xw + b)
            grad_w = C * (1/n) X^T (p - y) + w
            grad_b = C * (1/n) Σ (p_i - y_i)

    注：sklearn 的损失为 L = C * data_loss + 0.5 * ||w||²，
    与 data_loss + 0.5/C * ||w||² 等价（乘以 C），
    但梯度形式中 C 缩放数据项而非放大正则项，数值更稳定。
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.max_iter):
            z = X @ w + b
            p = _sigmoid(z)

            # 梯度（sklearn 等价形式：C 缩放数据项，正则项为 w）
            error = p - y
            grad_w = self.C * (X.T @ error) / n_samples + w
            grad_b = self.C * np.mean(error) if self.fit_intercept else 0.0

            # 梯度裁剪：防止强正则化时梯度爆炸
            grad_w = np.clip(grad_w, -1e4, 1e4)
            grad_b = np.clip(grad_b, -1e4, 1e4)

            # 更新
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * grad_b

            # 收敛检查
            grad_norm = np.linalg.norm(grad_w)
            if grad_norm < self.tol:
                break

        return w, b

    def _fit_ovr(self, X, y):
        """多分类：One-vs-Rest 策略。

        对每个类别 c，训练一个二分类器：
            正类：y == c
            负类：y != c

        预测时取所有分类器中置信度最高的类别。
        """
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 存储每个二分类器的参数
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            y_binary = (y == cls).astype(np.float64)
            w, b = self._fit_binary(X, y_binary)
            self.coef_[i] = w
            self.intercept_[i] = b

    def decision_function(self, X):
        """计算置信度分数：z = X @ w + b。

        二分类返回 shape (n_samples,)，
        多分类返回 shape (n_samples, n_classes)。
        """
        check_is_fitted(self, ["coef_"])
        X = check_array(X)

        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """预测类别标签。"""
        check_is_fitted(self, ["coef_", "classes_"])
        scores = self.decision_function(X)

        if scores.ndim == 1:
            # 二分类
            idx = (scores >= 0).astype(int)
        else:
            # 多分类：取最大置信度
            idx = np.argmax(scores, axis=1)

        return self.classes_[idx]

    def predict_proba(self, X):
        """预测类别概率。

        二分类返回 shape (n_samples, 2)，
        多分类返回 shape (n_samples, n_classes)（softmax 归一化）。
        """
        check_is_fitted(self, ["coef_", "classes_"])
        scores = self.decision_function(X)

        if scores.ndim == 1:
            p1 = _sigmoid(scores)
            return np.column_stack([1 - p1, p1])
        else:
            # 多分类：softmax
            scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores_shifted)
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)