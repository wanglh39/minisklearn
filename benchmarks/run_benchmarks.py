"""性能对比基准主脚本。

运行所有基准测试，生成结果表格。

用法：
    python benchmarks/run_benchmarks.py

输出：
    控制台表格 + benchmarks/results/ 目录下的结果文件
"""

import sys
import os
import time

# 确保项目根目录在 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minisklearn._fast import is_available


def main():
    print("=" * 70)
    print("  minisklearn 性能对比基准")
    print("  对比：纯 Python/NumPy  vs  C++ 扩展  vs  sklearn")
    print("=" * 70)
    print()

    if not is_available():
        print("⚠ C++ 扩展未加载，部分对比将跳过。")
        print("  编译方法: python cpp/build.py")
        print()

    # KNN 基准
    from benchmarks.benchmark_knn import benchmark_euclidean_distances, benchmark_knn_predict

    print("▶ KNN 基准测试\n")
    benchmark_euclidean_distances([
        (1000, 100, 10),
        (2000, 200, 10),
        (5000, 500, 10),
        (2000, 200, 50),
        (2000, 200, 100),
    ])
    benchmark_knn_predict([
        (1000, 200, 10),
        (2000, 500, 10),
        (5000, 1000, 10),
        (2000, 500, 50),
    ])

    # KMeans 基准
    from benchmarks.benchmark_kmeans import benchmark_kmeans_core, benchmark_kmeans_full

    print("▶ KMeans 基准测试\n")
    benchmark_kmeans_core([
        (1000, 10, 5),
        (2000, 10, 5),
        (5000, 10, 10),
        (2000, 50, 10),
        (2000, 100, 10),
    ])
    benchmark_kmeans_full([
        (1000, 10, 5),
        (2000, 10, 5),
        (5000, 10, 10),
        (2000, 50, 10),
    ])

    print("=" * 70)
    print("  基准测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()