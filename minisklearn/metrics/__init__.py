"""评估指标 —— 衡量模型好坏的函数。

本模块提供分类和回归的常用评估指标。与估计器不同，指标是纯函数，
不维护状态，调用即返回结果。

==============================================================================
设计哲学：为什么指标是函数而非方法？
==============================================================================

sklearn 中指标有两种存在形式：

    1. 函数形式：accuracy_score(y_true, y_pred) → float
       纯计算，无状态，灵活组合

    2. 方法形式：classifier.score(X, y) → float
       估计器内置的默认指标，内部调用 predict + 函数形式指标

函数形式更灵活（可以传入任意 y_pred，不限于某个模型），
方法形式更便捷（一行代码评估模型）。两者互补。

指标分类：
    - 分类指标：accuracy、precision、recall、F1、confusion_matrix
    - 回归指标：MSE、MAE、R²
    - 距离指标：euclidean、manhattan（用于 KNN 等）
"""

import numpy as np


def _check_y(y_true, y_pred):
    """校验 y_true 和 y_pred 并转为 ndarray。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} 与 y_pred shape {y_pred.shape} 不一致"
        )
    return y_true, y_pred


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """分类准确率。

    accuracy = (预测正确的样本数) / (总样本数)

    参数：
        normalize: True 返回比例 [0,1]，False 返回正确计数
        sample_weight: 样本权重

    数学原理：
        normalize=True:  acc = (1/n) Σ 1[y_i == ŷ_i]
        normalize=False: acc = Σ 1[y_i == ŷ_i]

    向量化实现：y_true == y_pred 产生布尔数组，mean/sum 即得结果。
    """
    y_true, y_pred = _check_y(y_true, y_pred)

    correct = (y_true == y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if normalize:
            return float(np.average(correct, weights=sample_weight))
        return float(np.sum(correct * sample_weight))

    if normalize:
        return float(np.mean(correct))
    return float(np.sum(correct))


def mean_squared_error(y_true, y_pred, sample_weight=None, squared=True):
    """均方误差（MSE）。

    MSE = (1/n) Σ (y_true - y_pred)²

    参数：
        sample_weight: 样本权重
        squared: True 返回 MSE，False 返回 RMSE（均方根误差）

    数学原理：
        MSE = (1/n) Σ (y_i - ŷ_i)²
        RMSE = √MSE

    为什么用平方而非绝对值？
        - 平方可导（绝对值在 0 点不可导），方便梯度优化
        - 对大误差惩罚更重（平方放大）
        - 与正态分布的 MLE 有数学联系
    """
    y_true, y_pred = _check_y(y_true, y_pred)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    errors = (y_true - y_pred) ** 2

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        mse = np.average(errors, weights=sample_weight)
    else:
        mse = np.mean(errors)

    if squared:
        return float(mse)
    return float(np.sqrt(mse))


def mean_absolute_error(y_true, y_pred, sample_weight=None):
    """平均绝对误差（MAE）。

    MAE = (1/n) Σ |y_true - y_pred|

    与 MSE 的区别：对异常值不敏感（线性惩罚而非平方惩罚）。
    """
    y_true, y_pred = _check_y(y_true, y_pred)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    errors = np.abs(y_true - y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        return float(np.average(errors, weights=sample_weight))
    return float(np.mean(errors))


def r2_score(y_true, y_pred, sample_weight=None):
    """R² 决定系数。

    R² = 1 - SS_res / SS_tot

    其中：
        SS_res = Σ (y_true - y_pred)²    残差平方和
        SS_tot = Σ (y_true - mean(y))²   总平方和

    含义：
        R² = 1：完美预测
        R² = 0：和总是预测均值一样差
        R² < 0：比预测均值还差（模型有系统性偏差）

    为什么回归默认用 R² 而非 MSE？
        R² 是无量纲的，方便跨数据集比较。
        MSE 依赖 y 的量纲，不同问题间不可直接比。
    """
    y_true, y_pred = _check_y(y_true, y_pred)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        weight_sum = np.sum(sample_weight)
        mean_y = np.sum(y_true * sample_weight) / weight_sum
        ss_res = np.sum(sample_weight * (y_true - y_pred) ** 2)
        ss_tot = np.sum(sample_weight * (y_true - mean_y) ** 2)
    else:
        mean_y = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - mean_y) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def confusion_matrix(y_true, y_pred, labels=None):
    """混淆矩阵。

    返回矩阵 C，其中 C[i, j] 表示真实类别为 labels[i]、预测为 labels[j] 的样本数。

    参数：
        labels: 类别列表，决定矩阵行列顺序。None 则自动推断并排序。

    返回：
        ndarray, shape (n_classes, n_classes)

    示例：
        y_true = [0, 1, 2, 2, 0]
        y_pred = [0, 2, 2, 2, 0]
        confusion_matrix(y_true, y_pred)
        → [[2, 0, 0],
           [0, 0, 1],
           [0, 0, 2]]

    向量化实现：
        用 bincount 对 (true * n + pred) 编码，再 reshape。
        比 for 循环遍历样本快得多。
    """
    y_true, y_pred = _check_y(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # 将标签映射为索引
    true_idx = np.array([label_to_idx.get(v, -1) for v in y_true])
    pred_idx = np.array([label_to_idx.get(v, -1) for v in y_pred])

    # 过滤未知标签
    mask = (true_idx >= 0) & (pred_idx >= 0)
    true_idx = true_idx[mask]
    pred_idx = pred_idx[mask]

    # 向量化：用线性索引编码 (true, pred) 对，bincount 统计
    linear_idx = true_idx * n_classes + pred_idx
    counts = np.bincount(linear_idx, minlength=n_classes * n_classes)

    return counts.reshape(n_classes, n_classes)


def precision_score(y_true, y_pred, labels=None, average="binary"):
    """精确率。

    precision = TP / (TP + FP)
    即"预测为正的样本中，真正为正的比例"。

    参数：
        average: 'binary' 二分类正类；'macro' 各类平均；'micro' 全局
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp

    if average == "binary" and cm.shape[0] == 2:
        return float(tp[1] / (tp[1] + fp[1])) if (tp[1] + fp[1]) > 0 else 0.0
    elif average == "macro":
        with np.errstate(divide="ignore", invalid="ignore"):
            precisions = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        return float(np.mean(precisions))
    elif average == "micro":
        return float(tp.sum() / (tp.sum() + fp.sum())) if (tp.sum() + fp.sum()) > 0 else 0.0
    raise ValueError(f"未知 average 类型: {average}")


def recall_score(y_true, y_pred, labels=None, average="binary"):
    """召回率。

    recall = TP / (TP + FN)
    即"真正为正的样本中，被预测为正的比例"。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp

    if average == "binary" and cm.shape[0] == 2:
        return float(tp[1] / (tp[1] + fn[1])) if (tp[1] + fn[1]) > 0 else 0.0
    elif average == "macro":
        with np.errstate(divide="ignore", invalid="ignore"):
            recalls = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        return float(np.mean(recalls))
    elif average == "micro":
        return float(tp.sum() / (tp.sum() + fn.sum())) if (tp.sum() + fn.sum()) > 0 else 0.0
    raise ValueError(f"未知 average 类型: {average}")


def f1_score(y_true, y_pred, labels=None, average="binary"):
    """F1 分数：精确率和召回率的调和平均。

    F1 = 2 * precision * recall / (precision + recall)
    """
    p = precision_score(y_true, y_pred, labels=labels, average=average)
    r = recall_score(y_true, y_pred, labels=labels, average=average)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))