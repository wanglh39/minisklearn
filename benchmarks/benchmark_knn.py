"""性能对比基准：KNN 距离计算 + 预测。

对比三者：
    1. minisklearn（纯 Python/NumPy 向量化）
    2. minisklearn._fast（C++ pybind11 扩展）
    3. sklearn（参考实现，底层 C/Cython）
"""

import time
import numpy as np
from minisklearn.neighbors import KNeighborsClassifier
from minisklearn.neighbors._distances import euclidean_distances as py_euclidean_distances
from minisklearn._fast import euclidean_distances as cpp_euclidean_distances
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN


def benchmark_euclidean_distances(sizes):
    """对比欧氏距离矩阵计算性能。"""
    print("=" * 70)
    print("欧氏距离矩阵计算")
    print("=" * 70)
    print(f"{'n_train':>8} {'n_query':>8} {'d':>4} | {'Python':>10} {'C++':>10} {'sklearn':>10} | {'C++/Py':>8} {'sk/Py':>8}")
    print("-" * 70)

    for n_train, n_query, d in sizes:
        rng = np.random.RandomState(42)
        X_train = rng.randn(n_train, d)
        X_query = rng.randn(n_query, d)

        # Python/NumPy
        t0 = time.perf_counter()
        for _ in range(5):
            D_py = py_euclidean_distances(X_query, X_train)
        t_py = (time.perf_counter() - t0) / 5

        # C++
        t0 = time.perf_counter()
        for _ in range(5):
            D_cpp = cpp_euclidean_distances(X_query, X_train)
        t_cpp = (time.perf_counter() - t0) / 5

        # sklearn
        from sklearn.metrics.pairwise import euclidean_distances as sk_dist
        t0 = time.perf_counter()
        for _ in range(5):
            D_sk = sk_dist(X_query, X_train)
        t_sk = (time.perf_counter() - t0) / 5

        speedup_cpp = t_py / t_cpp
        speedup_sk = t_py / t_sk

        print(f"{n_train:>8} {n_query:>8} {d:>4} | {t_py:>9.4f}s {t_cpp:>9.4f}s {t_sk:>9.4f}s | {speedup_cpp:>7.2f}x {speedup_sk:>7.2f}x")

    print()


def benchmark_knn_predict(sizes):
    """对比 KNN 预测性能。"""
    print("=" * 70)
    print("KNN 分类预测 (k=5)")
    print("=" * 70)
    print(f"{'n_train':>8} {'n_test':>8} {'d':>4} | {'minisklearn':>12} {'sklearn':>12} | {'sk/mini':>8}")
    print("-" * 70)

    for n_train, n_test, d in sizes:
        rng = np.random.RandomState(42)
        X_train = rng.randn(n_train, d)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(n_test, d)

        # minisklearn
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        t0 = time.perf_counter()
        for _ in range(3):
            y_pred = knn.predict(X_test)
        t_mini = (time.perf_counter() - t0) / 3

        # sklearn
        sk_knn = SklearnKNN(n_neighbors=5)
        sk_knn.fit(X_train, y_train)
        t0 = time.perf_counter()
        for _ in range(3):
            y_pred_sk = sk_knn.predict(X_test)
        t_sk = (time.perf_counter() - t0) / 3

        speedup = t_mini / t_sk

        print(f"{n_train:>8} {n_test:>8} {d:>4} | {t_mini:>11.4f}s {t_sk:>11.4f}s | {speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    print("\nminisklearn 性能对比基准 —— KNN\n")

    # 距离矩阵：(n_train, n_query, d)
    dist_sizes = [
        (1000, 100, 10),
        (2000, 200, 10),
        (5000, 500, 10),
        (2000, 200, 50),
        (2000, 200, 100),
    ]
    benchmark_euclidean_distances(dist_sizes)

    # KNN 预测
    knn_sizes = [
        (1000, 200, 10),
        (2000, 500, 10),
        (5000, 1000, 10),
        (2000, 500, 50),
    ]
    benchmark_knn_predict(knn_sizes)