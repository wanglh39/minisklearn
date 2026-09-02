"""数据划分工具：train_test_split + KFold。

==============================================================================
交叉验证原理
==============================================================================

train_test_split：
    随机将数据分为训练集和测试集。
    训练集用于 fit，测试集用于评估泛化能力。
    关键：测试集不能参与训练，否则评估偏乐观。

KFold 交叉验证：
    将数据分为 K 份，每次用 K-1 份训练、1 份测试，循环 K 次。
    取 K 次评估的平均值作为最终评估。

    优点：每个样本都参与过测试，评估更稳定。
    K=5 或 10 是常用值。

    K=n 时是留一法（LOO），方差小但计算贵。
"""

import numpy as np
from ..utils.validation import check_random_state


def train_test_split(*arrays, test_size=0.25, random_state=None,
                      shuffle=True, stratify=None):
    """将数据随机划分为训练集和测试集。

    参数：
        *arrays: 要划分的一个或多个数组（X, y 等），要求第一维长度一致
        test_size: 测试集比例（float）或数量（int）
        random_state: 随机种子
        shuffle: 是否打乱后再划分
        stratify: 按此数组的类别比例分层划分（保证训练/测试集类别比例一致）

    返回：
        list: 交替排列 [train_1, test_1, train_2, test_2, ...]

    使用示例：
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    n_samples = arrays[0].shape[0]
    rng = check_random_state(random_state)

    # 确定测试集大小
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test

    if shuffle:
        if stratify is not None:
            # 分层抽样：按类别比例划分
            indices = _stratified_indices(stratify, n_train, rng)
        else:
            indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result


def _stratified_indices(y, n_train, rng):
    """分层抽样索引：保证训练/测试集类别比例一致。"""
    classes = np.unique(y)
    train_indices = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_cls = len(cls_indices)
        n_cls_train = int(n_cls * n_train / len(y))
        train_indices.extend(cls_indices[:n_cls_train])

    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    all_indices = np.arange(len(y))
    test_indices = np.setdiff1d(all_indices, train_indices)

    return np.concatenate([train_indices, test_indices])


class KFold:
    """K 折交叉验证划分器。

    参数：
        n_splits: 折数 K
        shuffle: 是否打乱后再划分
        random_state: 随机种子

    使用示例：
        kf = KFold(n_splits=5)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """生成 K 组 (train_indices, test_indices)。"""
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        if self.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, X=None, y=None):
        return self.n_splits