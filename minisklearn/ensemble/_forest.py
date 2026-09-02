"""随机森林：RandomForestClassifier + RandomForestRegressor。

==============================================================================
随机森林原理
==============================================================================

随机森林 = Bagging + 随机子空间 + 决策树

三个随机性来源：

    1. 样本随机（Bagging）：每棵树从训练集中有放回采样（bootstrap）
    2. 特征随机（随机子空间）：每次分裂只考虑部分特征
    3. （隐含）样本权重随机：被采样的样本在树中权重相同

为什么随机森林比单棵决策树好？

    - 单棵决策树容易过拟合（记住训练数据）
    - 随机森林通过多棵树的平均/投票降低方差
    - 数学基础：若 n 棵独立同分布的树，方差 σ²/n
    - 实际树之间有相关性，方差降低不如理想，但仍显著

Bagging（Bootstrap Aggregating）：
    1. 从训练集 D 中有放回采样 |D| 个样本，得到 D₁
    2. 重复 n_estimators 次，得到 D₁, D₂, ..., D_n
    3. 每个子集训练一棵树
    4. 预测时投票/平均

    有放回采样：约 63.2% 的样本被选中（去重后），其余 36.8% 是 OOB（袋外）样本

max_features 的选择：
    - 分类：默认 sqrt(n_features)
    - 回归：默认 n_features / 3
    - 越小 → 树差异越大 → 方差降低越多，但偏差增加
"""

import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted
from ..tree._tree import DecisionTreeClassifier, DecisionTreeRegressor


def _bootstrap_sample(X, y, rng):
    """有放回采样生成子数据集。

    从 n 个样本中有放回采 n 个，约 63.2% 的唯一样本。
    """
    n_samples = X.shape[0]
    indices = rng.randint(0, n_samples, size=n_samples)
    return X[indices], y[indices]


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """随机森林分类器。

    参数：
        n_estimators: 树的数量（默认 100）
        max_depth: 每棵树的最大深度
        min_samples_split: 分裂所需最小样本数
        min_samples_leaf: 叶子节点最小样本数
        max_features: 每次分裂考虑的特征数
            - None: 默认 sqrt(n_features)
            - int: 指定数量
        bootstrap: 是否使用 bootstrap 采样（默认 True）
        random_state: 随机种子

    fit 后的属性：
        estimators_: 训练好的决策树列表
        classes_: 类别标签
        n_features_in_: 输入特征数

    使用示例：
        >>> clf = RandomForestClassifier(n_estimators=100, max_depth=5)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        """训练随机森林：每棵树用不同的 bootstrap 样本。"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # 默认 max_features = sqrt(n_features)（分类）
        if self.max_features is None:
            max_features = max(1, int(np.sqrt(n_features)))
        else:
            max_features = self.max_features

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        for i in range(self.n_estimators):
            # 每棵树用不同的随机种子
            tree_seed = rng.randint(0, 2 ** 31 - 1)

            # Bootstrap 采样
            if self.bootstrap:
                X_subset, y_subset = _bootstrap_sample(X, y, rng)
            else:
                X_subset, y_subset = X, y

            # 创建并训练决策树
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=tree_seed,
            )
            tree.fit(X_subset, y_subset)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """预测：多棵树投票，多数类为结果。"""
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        # 收集所有树的预测
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        # predictions shape: (n_estimators, n_samples)

        # 对每个样本投票
        from collections import Counter
        y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            counts = Counter(votes)
            y_pred[i] = counts.most_common(1)[0][0]

        return y_pred

    def predict_proba(self, X):
        """预测概率：所有树的平均概率。"""
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        proba = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            proba += tree.predict_proba(X)
        proba /= self.n_estimators
        return proba


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """随机森林回归器。

    参数同 RandomForestClassifier，但：
        - 默认 max_features = n_features / 3（回归）
        - 预测时取所有树的平均

    使用示例：
        >>> reg = RandomForestRegressor(n_estimators=100, max_depth=5)
        >>> reg.fit(X_train, y_train)
        >>> reg.predict(X_test)
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        """训练随机森林回归。"""
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

        # 默认 max_features = n_features / 3（回归）
        if self.max_features is None:
            max_features = max(1, n_features // 3)
        else:
            max_features = self.max_features

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        for i in range(self.n_estimators):
            tree_seed = rng.randint(0, 2 ** 31 - 1)

            if self.bootstrap:
                X_subset, y_subset = _bootstrap_sample(X, y, rng)
            else:
                X_subset, y_subset = X, y

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=tree_seed,
            )
            tree.fit(X_subset, y_subset)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """预测：所有树的平均。"""
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)