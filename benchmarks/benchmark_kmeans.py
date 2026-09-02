"""性能对比基准：KMeans 聚类。

对比三者：
    1. minisklearn（纯 Python/NumPy 向量化）
    2. minisklearn._fast（C++ pybind11 扩展，核心循环）
    3. sklearn（参考实现，底层 C/Cython）
"""

import time
import numpy as np
from minisklearn.cluster import KMeans as MiniKMeans
from minisklearn._fast import kmeans_assign as cpp_assign
from minisklearn._fast import kmeans_update as cpp_update
from minisklearn._fast import kmeans_inertia as cpp_inertia
from sklearn.cluster import KMeans as SklearnKMeans


def cpp_kmeans_fit(X, k, max_iter=100, tol=1e-4, seed=42):
    """用 C++ 核心函数实现完整 KMeans fit。"""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # KMeans++ 初始化（简化：随机选 k 个点）
    centroids = X[rng.choice(n, k, replace=False)].copy()

    for _ in range(max_iter):
        old_centroids = centroids.copy()
        assignments = cpp_assign(X, centroids)
        centroids = cpp_update(X, assignments, k)
        shift = np.sqrt(((centroids - old_centroids) ** 2).sum(axis=1)).max()
        if shift < tol:
            break

    assignments = cpp_assign(X, centroids)
    inertia = cpp_inertia(X, assignments, centroids)
    return centroids, assignments, inertia


def py_kmeans_core(X, k, max_iter=100, tol=1e-4, seed=42):
    """用纯 NumPy 实现同样的 KMeans 核心循环（对比公平）。"""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    centroids = X[rng.choice(n, k, replace=False)].copy()

    for _ in range(max_iter):
        old_centroids = centroids.copy()
        # 分配步
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = dists.argmin(axis=1)
        # 更新步
        for j in range(k):
            mask = assignments == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
        shift = np.sqrt(((centroids - old_centroids) ** 2).sum(axis=1)).max()
        if shift < tol:
            break

    dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    assignments = dists.argmin(axis=1)
    inertia = dists[np.arange(n), assignments].sum()
    return centroids, assignments, inertia


def benchmark_kmeans_core(sizes):
    """对比 KMeans 核心循环性能。"""
    print("=" * 75)
    print("KMeans 核心循环（分配 + 更新 + 收敛判断）")
    print("=" * 75)
    print(f"{'n':>8} {'d':>4} {'k':>4} | {'NumPy':>10} {'C++':>10} | {'C++/Py':>8} {'加速比':>8}")
    print("-" * 75)

    for n, d, k in sizes:
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)

        # NumPy
        t0 = time.perf_counter()
        for _ in range(3):
            py_kmeans_core(X, k, seed=42)
        t_py = (time.perf_counter() - t0) / 3

        # C++
        t0 = time.perf_counter()
        for _ in range(3):
            cpp_kmeans_fit(X, k, seed=42)
        t_cpp = (time.perf_counter() - t0) / 3

        speedup = t_py / t_cpp
        print(f"{n:>8} {d:>4} {k:>4} | {t_py:>9.4f}s {t_cpp:>9.4f}s | {speedup:>7.2f}x {speedup:>7.2f}x")

    print()


def benchmark_kmeans_full(sizes):
    """对比完整 KMeans fit 性能。"""
    print("=" * 75)
    print("KMeans 完整 fit（含 KMeans++ 初始化）")
    print("=" * 75)
    print(f"{'n':>8} {'d':>4} {'k':>4} | {'minisklearn':>12} {'sklearn':>12} | {'sk/mini':>8}")
    print("-" * 75)

    for n, d, k in sizes:
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)

        # minisklearn
        t0 = time.perf_counter()
        for _ in range(3):
            km = MiniKMeans(n_clusters=k, random_state=42, n_init=1)
            km.fit(X)
        t_mini = (time.perf_counter() - t0) / 3

        # sklearn
        t0 = time.perf_counter()
        for _ in range(3):
            sk_km = SklearnKMeans(n_clusters=k, random_state=42, n_init=1)
            sk_km.fit(X)
        t_sk = (time.perf_counter() - t0) / 3

        speedup = t_mini / t_sk
        print(f"{n:>8} {d:>4} {k:>4} | {t_mini:>11.4f}s {t_sk:>11.4f}s | {speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    print("\nminisklearn 性能对比基准 —— KMeans\n")

    core_sizes = [
        (1000, 10, 5),
        (2000, 10, 5),
        (5000, 10, 10),
        (2000, 50, 10),
        (2000, 100, 10),
    ]
    benchmark_kmeans_core(core_sizes)

    full_sizes = [
        (1000, 10, 5),
        (2000, 10, 5),
        (5000, 10, 10),
        (2000, 50, 10),
    ]
    benchmark_kmeans_full(full_sizes)