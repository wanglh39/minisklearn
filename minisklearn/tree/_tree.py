"""决策树（CART 算法）—— 分类树与回归树。

==============================================================================
CART 算法原理
==============================================================================

CART（Classification and Regression Trees）的核心是递归二分：

    1. 在当前节点寻找最佳分裂（特征 + 阈值）
    2. 按分裂将样本分为左右子集
    3. 对左右子集递归建树
    4. 达到停止条件时创建叶子节点

分裂选择：
    对每个特征的每个可能阈值，计算分裂后的加权纯度：
        impurity_split = (n_L/n) * impurity(L) + (n_R/n) * impurity(R)
    选择使 impurity_split 最小的分裂。

分类纯度度量：基尼系数
    Gini(D) = 1 - Σ_k (p_k)²
    其中 p_k 是类别 k 在 D 中的比例。
    Gini 越小越纯（纯节点 Gini=0）。

回归纯度度量：均方误差
    MSE(D) = (1/n) Σ (y_i - ȳ)²
    MSE 越小越纯（纯节点 MSE=0）。

停止条件：
    - 样本数 < min_samples_split
    - 深度 >= max_depth
    - 节点已纯（impurity=0）
    - 分裂后某侧样本 < min_samples_leaf

==============================================================================
实现优化
==============================================================================

1. 阈值选择：对特征排序后，只在相邻不同值的中点处尝试分裂
   （相邻相同值之间分裂无意义）

2. 向量化分裂评估：对排序后的特征，用累积统计量快速计算
   左右子集的纯度，避免重复遍历。教学版简化为遍历所有阈值。

3. 预测：沿树遍历到叶子，返回叶子值（多数类或均值）
"""

import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted


class _TreeNode:
    """决策树节点。

    内部节点存储分裂信息，叶子节点存储预测值。

    属性：
        feature: 分裂特征索引（内部节点）
        threshold: 分裂阈值（内部节点）
        left: 左子树（特征值 <= threshold）
        right: 右子树（特征值 > threshold）
        value: 叶子节点的预测值
        is_leaf: 是否为叶子
        n_samples: 该节点的样本数
        impurity: 该节点的纯度值
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, is_leaf=False, n_samples=0, impurity=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
        self.n_samples = n_samples
        self.impurity = impurity

    def predict_one(self, x):
        """对单个样本沿树遍历到叶子，返回预测值。"""
        node = self
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# ============================================================================
# 分裂准则
# ============================================================================

def _gini_impurity(y):
    """基尼系数：Gini(D) = 1 - Σ (p_k)²

    向量化实现：用 bincount 统计各类别频数，计算概率平方和。
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def _mse_impurity(y):
    """均方误差：MSE(D) = (1/n) Σ (y_i - ȳ)²

    等价于 var(y)。
    """
    if len(y) == 0:
        return 0.0
    return np.var(y)


def _majority_vote(y):
    """多数投票：返回出现最多的类别。"""
    counts = np.bincount(y)
    return np.argmax(counts)


# ============================================================================
# 决策树分类器
# ============================================================================

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """CART 决策树分类器。

    参数：
        criterion: 分裂准则（目前只支持 'gini'）
        max_depth: 树的最大深度
        min_samples_split: 分裂所需最小样本数
        min_samples_leaf: 叶子节点最小样本数
        random_state: 随机种子

    fit 后的属性：
        tree_: 树的根节点
        classes_: 类别标签
        n_features_in_: 输入特征数

    使用示例：
        >>> clf = DecisionTreeClassifier(max_depth=3)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """递归构建决策树。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 将标签编码为 0, 1, ..., n_classes-1
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._idx_to_label = {i: c for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_to_idx[c] for c in y])

        self._rng = np.random.RandomState(self.random_state)
        self.tree_ = self._build_tree(X, y_encoded, depth=0)
        return self

    def _get_feature_subset(self, n_features):
        """选择本次分裂要考虑的特征子集。

        max_features=None: 考虑所有特征（普通决策树）
        max_features=int: 随机选 max_features 个特征（随机森林）
        """
        if self.max_features is None or self.max_features >= n_features:
            return range(n_features)
        return self._rng.choice(n_features, self.max_features, replace=False)

    def _build_tree(self, X, y, depth):
        """递归构建子树。"""
        n_samples = len(y)
        n_classes = len(self.classes_)

        # 计算当前节点纯度
        impurity = _gini_impurity(y)

        # 创建叶子节点的条件
        is_leaf = (
            n_samples < self.min_samples_split or          # 样本太少
            impurity == 0.0 or                              # 已纯
            (self.max_depth is not None and depth >= self.max_depth)  # 深度到顶
        )

        if is_leaf:
            return _TreeNode(
                is_leaf=True,
                value=_majority_vote(y),
                n_samples=n_samples,
                impurity=impurity,
            )

        # 寻找最佳分裂
        best_split = self._find_best_split(X, y)

        if best_split is None:
            # 找不到有效分裂，创建叶子
            return _TreeNode(
                is_leaf=True,
                value=_majority_vote(y),
                n_samples=n_samples,
                impurity=impurity,
            )

        feature, threshold, left_mask, right_mask = best_split

        # 递归构建左右子树
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            n_samples=n_samples,
            impurity=impurity,
        )

    def _find_best_split(self, X, y):
        """寻找最佳分裂点。

        遍历所有特征的所有可能阈值，找使基尼系数最小的分裂。

        优化：对特征排序后只在相邻不同值的中点尝试分裂。
        """
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_split = None

        # max_features: 随机选择特征子集（随机森林的核心）
        features = self._get_feature_subset(n_features)

        for feature in features:
            feature_values = X[:, feature]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            # 在相邻不同值的中点处尝试分裂
            for i in range(n_samples - 1):
                if sorted_values[i] == sorted_values[i + 1]:
                    continue  # 相同值之间不分裂

                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

                # 分割
                left_y = sorted_y[:i + 1]
                right_y = sorted_y[i + 1:]

                # 检查 min_samples_leaf
                if (len(left_y) < self.min_samples_leaf or
                        len(right_y) < self.min_samples_leaf):
                    continue

                # 加权基尼系数
                n_left = len(left_y)
                n_right = len(right_y)
                gini_split = (n_left / n_samples) * _gini_impurity(left_y) + \
                             (n_right / n_samples) * _gini_impurity(right_y)

                if gini_split < best_gini:
                    best_gini = gini_split
                    # 构建左右掩码（基于原始顺序）
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    best_split = (feature, threshold, left_mask, right_mask)

        return best_split

    def predict(self, X):
        """预测：沿树遍历到叶子，返回多数类。"""
        check_is_fitted(self, ["tree_"])
        X = check_array(X)
        y_pred_idx = np.array([self.tree_.predict_one(x) for x in X])
        return np.array([self._idx_to_label[i] for i in y_pred_idx])

    def predict_proba(self, X):
        """预测类别概率（叶子中各类别比例）。"""
        check_is_fitted(self, ["tree_"])
        X = check_array(X)
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # 叶子节点的 value 是多数类索引，概率为 1.0
            proba[i, node.value] = 1.0
        return proba


# ============================================================================
# 决策树回归器
# ============================================================================

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """CART 决策树回归器。

    参数同 DecisionTreeClassifier，但用 MSE 作为分裂准则。
    叶子节点的预测值为该节点样本目标值的均值。

    使用示例：
        >>> reg = DecisionTreeRegressor(max_depth=3)
        >>> reg.fit(X_train, y_train)
        >>> reg.predict(X_test)
    """

    def __init__(self, criterion="mse", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """递归构建回归决策树。"""
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]

        self._rng = np.random.RandomState(self.random_state)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _get_feature_subset(self, n_features):
        """选择本次分裂要考虑的特征子集。"""
        if self.max_features is None or self.max_features >= n_features:
            return range(n_features)
        return self._rng.choice(n_features, self.max_features, replace=False)

    def _build_tree(self, X, y, depth):
        """递归构建子树。"""
        n_samples = len(y)
        impurity = _mse_impurity(y)

        is_leaf = (
            n_samples < self.min_samples_split or
            impurity == 0.0 or
            (self.max_depth is not None and depth >= self.max_depth)
        )

        if is_leaf:
            return _TreeNode(
                is_leaf=True,
                value=np.mean(y),
                n_samples=n_samples,
                impurity=impurity,
            )

        best_split = self._find_best_split(X, y)

        if best_split is None:
            return _TreeNode(
                is_leaf=True,
                value=np.mean(y),
                n_samples=n_samples,
                impurity=impurity,
            )

        feature, threshold, left_mask, right_mask = best_split
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            n_samples=n_samples,
            impurity=impurity,
        )

    def _find_best_split(self, X, y):
        """寻找最佳分裂点（最小化加权 MSE）。"""
        n_samples, n_features = X.shape
        best_mse = float("inf")
        best_split = None

        features = self._get_feature_subset(n_features)

        for feature in features:
            feature_values = X[:, feature]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            for i in range(n_samples - 1):
                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

                left_y = sorted_y[:i + 1]
                right_y = sorted_y[i + 1:]

                if (len(left_y) < self.min_samples_leaf or
                        len(right_y) < self.min_samples_leaf):
                    continue

                n_left = len(left_y)
                n_right = len(right_y)
                mse_split = (n_left / n_samples) * _mse_impurity(left_y) + \
                            (n_right / n_samples) * _mse_impurity(right_y)

                if mse_split < best_mse:
                    best_mse = mse_split
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    best_split = (feature, threshold, left_mask, right_mask)

        return best_split

    def predict(self, X):
        """预测：沿树遍历到叶子，返回均值。"""
        check_is_fitted(self, ["tree_"])
        X = check_array(X)
        return np.array([self.tree_.predict_one(x) for x in X])