// kmeans.cpp —— KMeans 核心迭代的 C++ 实现
//
// KMeans 的 Lloyd 算法每轮两步：
//   1. 分配步：每个点分配到最近质心
//   2. 更新步：每个质心更新为簇内均值
//
// C++ 优势在于紧凑的双重循环 + 连续内存访问，
// 避免 NumPy 广播产生的临时数组。

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>

namespace py = pybind11;

// 分配步：每个点分配到最近质心
// 输入：X (n x d), centroids (k x d)
// 输出：assignments (n,)，每个点的簇标签
py::array_t<int> kmeans_assign(py::array_t<double> X,
                                py::array_t<double> centroids) {
    auto buf_x = X.request();
    auto buf_c = centroids.request();

    int n = buf_x.shape[0];
    int d = buf_x.shape[1];
    int k = buf_c.shape[0];

    auto result = py::array_t<int>(n);
    auto buf_res = result.request();

    double *ptr_x = static_cast<double *>(buf_x.ptr);
    double *ptr_c = static_cast<double *>(buf_c.ptr);
    int *ptr_res = static_cast<int *>(buf_res.ptr);

    // 预计算质心行平方和
    std::vector<double> c_norm_sq(k);
    for (int j = 0; j < k; ++j) {
        double s = 0.0;
        for (int f = 0; f < d; ++f)
            s += ptr_c[j * d + f] * ptr_c[j * d + f];
        c_norm_sq[j] = s;
    }

    for (int i = 0; i < n; ++i) {
        double x_norm_sq = 0.0;
        for (int f = 0; f < d; ++f)
            x_norm_sq += ptr_x[i * d + f] * ptr_x[i * d + f];

        double best_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;

        for (int j = 0; j < k; ++j) {
            double dot = 0.0;
            for (int f = 0; f < d; ++f)
                dot += ptr_x[i * d + f] * ptr_c[j * d + f];
            double dist_sq = x_norm_sq - 2.0 * dot + c_norm_sq[j];
            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_cluster = j;
            }
        }
        ptr_res[i] = best_cluster;
    }

    return result;
}

// 更新步：重新计算每个簇的质心
// 输入：X (n x d), assignments (n,), k
// 输出：new_centroids (k x d)
py::array_t<double> kmeans_update(py::array_t<double> X,
                                   py::array_t<int> assignments,
                                   int k) {
    auto buf_x = X.request();
    auto buf_a = assignments.request();

    int n = buf_x.shape[0];
    int d = buf_x.shape[1];

    auto result = py::array_t<double>({k, d});
    auto buf_res = result.request();

    double *ptr_x = static_cast<double *>(buf_x.ptr);
    int *ptr_a = static_cast<int *>(buf_a.ptr);
    double *ptr_res = static_cast<double *>(buf_res.ptr);

    // 累加器
    std::vector<double> sums(k * d, 0.0);
    std::vector<int> counts(k, 0);

    for (int i = 0; i < n; ++i) {
        int c = ptr_a[i];
        counts[c]++;
        for (int f = 0; f < d; ++f)
            sums[c * d + f] += ptr_x[i * d + f];
    }

    for (int j = 0; j < k; ++j) {
        if (counts[j] > 0) {
            for (int f = 0; f < d; ++f)
                ptr_res[j * d + f] = sums[j * d + f] / counts[j];
        } else {
            // 空簇：保持零（调用方处理）
            for (int f = 0; f < d; ++f)
                ptr_res[j * d + f] = 0.0;
        }
    }

    return result;
}

// 计算 inertia（簇内距离平方和）
// 输入：X (n x d), assignments (n,), centroids (k x d)
// 输出：double
double kmeans_inertia(py::array_t<double> X,
                      py::array_t<int> assignments,
                      py::array_t<double> centroids) {
    auto buf_x = X.request();
    auto buf_a = assignments.request();
    auto buf_c = centroids.request();

    int n = buf_x.shape[0];
    int d = buf_x.shape[1];

    double *ptr_x = static_cast<double *>(buf_x.ptr);
    int *ptr_a = static_cast<int *>(buf_a.ptr);
    double *ptr_c = static_cast<double *>(buf_c.ptr);

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        int c = ptr_a[i];
        double dist_sq = 0.0;
        for (int f = 0; f < d; ++f) {
            double diff = ptr_x[i * d + f] - ptr_c[c * d + f];
            dist_sq += diff * diff;
        }
        total += dist_sq;
    }
    return total;
}