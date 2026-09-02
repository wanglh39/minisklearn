"""KNN 距离计算工具 —— 向量化实现。

==============================================================================
向量化距离计算的核心技巧
==============================================================================

朴素实现：对每对 (query_i, train_j) 计算欧氏距离，需要双重循环。

向量化实现：利用欧氏距离的展开公式

    ||x - y||² = ||x||² - 2 x·y + ||y||²

对一组查询点 Q (m, d) 和训练点 T (n, d)，距离矩阵 D (m, n)：

    D[i, j] = ||Q[i]||² - 2 Q[i]·T[j] + ||T[j]||²

矩阵形式：

    D = ||Q||²[:, None] - 2 Q @ T^T + ||T||²[None, :]

一次矩阵乘法 Q @ T^T 计算所有点积，无需循环。

复杂度：
    - 朴素：O(m × n × d) 逐对计算
    - 向量化：O(m × n × d) 但利用 BLAS 矩阵乘法，常数小 10-100 倍
"""

import numpy as np


def euclidean_distances(X_query, X_train, squared=False):
    """计算查询点到训练点的欧氏距离矩阵。

    参数：
        X_query: 查询点，shape (m, d)
        X_train: 训练点，shape (n, d)
        squared: True 返回距离平方，False 返回距离

    返回：
        dist: 距离矩阵，shape (m, n)，dist[i, j] = ||X_query[i] - X_train[j]||

    数学原理：
        ||x - y||² = ||x||² - 2 x·y + ||y||²
        展开后三项均可向量化：
        - ||x||²: 沿 axis=1 对 X_query 求平方和
        - x·y:   矩阵乘法 X_query @ X_train.T
        - ||y||²: 沿 axis=1 对 X_train 求平方和
    """
    # ||x||² 和 ||y||²
    query_sq_norm = np.sum(X_query ** 2, axis=1)  # shape (m,)
    train_sq_norm = np.sum(X_train ** 2, axis=1)  # shape (n,)

    # 交叉项 -2 x·y
    cross = X_query @ X_train.T  # shape (m, n)

    # 距离平方：||x||² - 2 x·y + ||y||²
    dist_sq = query_sq_norm[:, None] - 2 * cross + train_sq_norm[None, :]

    # 数值误差可能导致小的负值，裁剪到 0
    dist_sq = np.maximum(dist_sq, 0.0)

    if squared:
        return dist_sq
    return np.sqrt(dist_sq)


def manhattan_distances(X_query, X_train):
    """曼哈顿距离（L1 距离）。

    ||x - y||_1 = Σ |x_j - y_j|

    向量化实现：利用广播
        X_query[:, None, :] - X_train[None, :, :]  → shape (m, n, d)
        对最后一维求绝对值再求和

    注意：曼哈顿距离的广播会创建 (m, n, d) 的中间数组，
    内存消耗比欧氏距离大。对大数据集需分批处理。
    """
    diff = X_query[:, None, :] - X_train[None, :, :]  # (m, n, d)
    return np.sum(np.abs(diff), axis=2)  # (m, n)


def find_k_neighbors(X_query, X_train, k, metric="euclidean"):
    """找到每个查询点的 k 个最近邻索引和距离。

    参数：
        X_query: 查询点，shape (m, d)
        X_train: 训练点，shape (n, d)
        k: 近邻数
        metric: 距离度量 ('euclidean' 或 'manhattan')

    返回：
        indices: 近邻索引，shape (m, k)
        distances: 近邻距离，shape (m, k)

    实现细节：
        用 np.argpartition 而非 np.argsort 找前 k 小：
        - argpartition: O(n)（只分区不排序）
        - argsort: O(n log n)（全排序）
        找前 k 个不需要全排序，argpartition 更快。

        argpartition 返回的 k 个元素是无序的，需要再排序。
    """
    if metric == "euclidean":
        dist_matrix = euclidean_distances(X_query, X_train)
    elif metric == "manhattan":
        dist_matrix = manhattan_distances(X_query, X_train)
    else:
        raise ValueError(f"未知距离度量: {metric}")

    n_train = X_train.shape[0]
    k = min(k, n_train)

    # argpartition 找前 k 小的索引（无序）
    # kth=k-1：保证第 k-1 位置排好，前 k-1 个比它小
    partition_indices = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]

    # 取出对应的距离
    m = X_query.shape[0]
    partition_distances = dist_matrix[np.arange(m)[:, None], partition_indices]

    # 对 k 个近邻按距离排序
    sort_order = np.argsort(partition_distances, axis=1)
    indices = partition_indices[np.arange(m)[:, None], sort_order]
    distances = partition_distances[np.arange(m)[:, None], sort_order]

    return indices, distances