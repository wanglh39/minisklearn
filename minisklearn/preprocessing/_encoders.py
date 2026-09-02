"""编码器：LabelEncoder 和 OneHotEncoder。

本模块处理类别型数据的编码，将非数值标签转为算法可处理的数值形式。

==============================================================================
为什么需要编码？
==============================================================================

大多数算法只接受数值输入。但现实数据中有大量类别型特征：

    - 分类标签："猫" / "狗" / "鸟"
    - 枚举特征："红" / "绿" / "蓝"
    - 有序特征："低" / "中" / "高"

两种编码策略：
    - LabelEncoder：类别 → 整数（0, 1, 2, ...）
      适合标签编码，不适合特征（因为引入了虚假的序关系）

    - OneHotEncoder：类别 → 独热向量（[1,0,0] / [0,1,0] / [0,0,1]）
      适合特征编码，不引入序关系，但维度可能爆炸
"""

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array, check_is_fitted


class LabelEncoder(BaseEstimator, TransformerMixin):
    """标签编码器：将标签映射为连续整数 [0, n_classes-1]。

    主要用于编码目标标签 y，不用于编码特征矩阵 X。
    因为对特征用整数编码会引入虚假的序关系（"蓝"=2 > "红"=0 没有意义）。

    fit 后的属性：
        classes_: 排序后的唯一标签，shape (n_classes,)

    使用示例：
        >>> le = LabelEncoder()
        >>> le.fit(["猫", "狗", "鸟"])
        >>> le.transform(["猫", "鸟"])
        array([0, 1])
        >>> le.inverse_transform([0, 1])
        array(['猫', '鸟'], dtype='<U1')
    """

    def __init__(self):
        pass

    def fit(self, y):
        """学习标签到整数的映射。

        实现：np.unique 会排序并去重，排序保证了映射的确定性
        （同一组标签无论输入顺序如何，编码结果一致）。
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """将标签转为整数索引。

        向量化实现：np.searchsorted 在排序数组中二分查找位置，
        比 dict 查找 + 列表推导更快。
        """
        check_is_fitted(self, ["classes_"])
        y = np.asarray(y)

        # searchsorted: 对每个 y_i，找到它在 classes_ 中的位置
        encoded = np.searchsorted(self.classes_, y)

        # 检查是否有未知标签
        mask = encoded < len(self.classes_)
        if not np.all(mask):
            unknown = y[~mask]
            raise ValueError(
                f"包含训练时未见过的标签: {np.unique(unknown)}"
            )
        # 还需检查 searchsorted 是否匹配（边界情况）
        if not np.all(self.classes_[encoded] == y):
            unknown = y[self.classes_[encoded] != y]
            raise ValueError(
                f"包含训练时未见过的标签: {np.unique(unknown)}"
            )

        return encoded

    def inverse_transform(self, y):
        """将整数索引映射回原始标签。"""
        check_is_fitted(self, ["classes_"])
        y = np.asarray(y)

        if np.any(y < 0) or np.any(y >= len(self.classes_)):
            raise ValueError(
                f"索引超出范围 [0, {len(self.classes_) - 1}]"
            )
        return self.classes_[y]

    def fit_transform(self, y):
        """拟合并转换。"""
        return self.fit(y).transform(y)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """独热编码器：将类别特征转为独热向量。

    对每个类别特征列，生成 n_categories 个二值列，对应类别位置为 1，其余为 0。

    参数：
        categories: 'auto' 自动推断，或显式指定每列的类别

    fit 后的属性：
        categories_: 每列的类别列表，list of ndarray
        n_features_in_: 输入特征数

    使用示例：
        >>> enc = OneHotEncoder()
        >>> enc.fit([["猫"], ["狗"], ["鸟"]])
        >>> enc.transform([["猫"], ["鸟"]])
        array([[1., 0., 0.],
               [0., 1., 0.]])
    """

    def __init__(self, categories="auto"):
        self.categories = categories

    def fit(self, X, y=None):
        """学习每列的类别集合。"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        if self.categories == "auto":
            # 对每列独立找唯一值（排序）
            self.categories_ = [
                np.unique(X[:, i]) for i in range(X.shape[1])
            ]
        else:
            self.categories_ = [
                np.asarray(cat) for cat in self.categories
            ]

        return self

    def transform(self, X):
        """将类别转为独热编码。

        向量化实现思路：
            对每列，用 searchsorted 找到类别索引，
            然后用高级索引填充独热矩阵。
        """
        check_is_fitted(self, ["categories_"])
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"输入特征数 {X.shape[1]} 与训练时 {self.n_features_in_} 不一致"
            )

        n_samples = X.shape[0]
        n_total_cols = sum(len(cats) for cats in self.categories_)
        output = np.zeros((n_samples, n_total_cols), dtype=np.float64)

        col_offset = 0
        for i, cats in enumerate(self.categories_):
            n_cats = len(cats)
            # 找到每行该列的类别在 cats 中的位置
            indices = np.searchsorted(cats, X[:, i])

            # 校验未知类别
            valid = indices < n_cats
            if not np.all(valid):
                unknown = X[~valid, i]
                raise ValueError(
                    f"第 {i} 列包含未见过的类别: {np.unique(unknown)}"
                )
            if not np.all(cats[indices] == X[:, i]):
                mask = cats[indices] != X[:, i]
                raise ValueError(
                    f"第 {i} 列包含未见过的类别: {np.unique(X[mask, i])}"
                )

            # 向量化填充独热：output[row, col_offset + index] = 1
            output[np.arange(n_samples), col_offset + indices] = 1.0
            col_offset += n_cats

        return output

    def inverse_transform(self, X):
        """将独热编码转回原始类别。"""
        check_is_fitted(self, ["categories_"])
        X = np.asarray(X)

        n_samples = X.shape[0]
        output = np.empty((n_samples, self.n_features_in_), dtype=object)

        col_offset = 0
        for i, cats in enumerate(self.categories_):
            n_cats = len(cats)
            block = X[:, col_offset:col_offset + n_cats]
            indices = np.argmax(block, axis=1)
            output[:, i] = cats[indices]
            col_offset += n_cats

        return output