// distances.cpp —— 欧氏距离矩阵的 C++ 实现
//
// 核心公式：||x - y||^2 = ||x||^2 - 2 * x·y + ||y||^2
//
// C++ 优势：
//   1. 编译期优化：-O3 自动向量化（SIMD）、循环展开
//   2. 零开销抽象：直接操作连续内存，无 Python 对象开销
//   3. 缓存友好：行主序遍历，CPU 缓存命中率高

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

// 计算两组点之间的欧氏距离矩阵
// 输入：X1 (n1 x d), X2 (n2 x d)
// 输出：dist (n1 x n2)，dist[i][j] = ||X1[i] - X2[j]||
py::array_t<double> euclidean_distances(py::array_t<double> X1,
                                         py::array_t<double> X2) {
    auto buf1 = X1.request();
    auto buf2 = X2.request();

    int n1 = buf1.shape[0];
    int d = buf1.shape[1];
    int n2 = buf2.shape[0];

    auto result = py::array_t<double>({n1, n2});
    auto buf_res = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr_res = static_cast<double *>(buf_res.ptr);

    // 预计算 X2 的行平方和 ||y||^2
    std::vector<double> x2_norm_sq(n2);
    for (int j = 0; j < n2; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k)
            s += ptr2[j * d + k] * ptr2[j * d + k];
        x2_norm_sq[j] = s;
    }

    // 对每行 X1[i] 计算到所有 X2[j] 的距离
    for (int i = 0; i < n1; ++i) {
        // ||x||^2
        double x1_norm_sq = 0.0;
        for (int k = 0; k < d; ++k)
            x1_norm_sq += ptr1[i * d + k] * ptr1[i * d + k];

        for (int j = 0; j < n2; ++j) {
            // x·y
            double dot = 0.0;
            for (int k = 0; k < d; ++k)
                dot += ptr1[i * d + k] * ptr2[j * d + k];

            // ||x-y||^2 = ||x||^2 - 2*x·y + ||y||^2
            double dist_sq = x1_norm_sq - 2.0 * dot + x2_norm_sq[j];
            if (dist_sq < 0.0) dist_sq = 0.0;  // 数值修正
            ptr_res[i * n2 + j] = std::sqrt(dist_sq);
        }
    }

    return result;
}

// KNN 的核心：找到每个查询点的 k 个最近邻索引
// 输入：X_train (n_train x d), X_query (n_query x d), k
// 输出：indices (n_query x k)，每行是 k 个最近邻的训练集索引
py::array_t<int> knn_neighbors(py::array_t<double> X_train,
                                py::array_t<double> X_query,
                                int k) {
    auto buf_tr = X_train.request();
    auto buf_q = X_query.request();

    int n_train = buf_tr.shape[0];
    int d = buf_tr.shape[1];
    int n_query = buf_q.shape[0];

    auto result = py::array_t<int>({n_query, k});
    auto buf_res = result.request();

    double *ptr_tr = static_cast<double *>(buf_tr.ptr);
    double *ptr_q = static_cast<double *>(buf_q.ptr);
    int *ptr_res = static_cast<int *>(buf_res.ptr);

    // 预计算训练集行平方和
    std::vector<double> train_norm_sq(n_train);
    for (int j = 0; j < n_train; ++j) {
        double s = 0.0;
        for (int f = 0; f < d; ++f)
            s += ptr_tr[j * d + f] * ptr_tr[j * d + f];
        train_norm_sq[j] = s;
    }

    for (int i = 0; i < n_query; ++i) {
        double q_norm_sq = 0.0;
        for (int f = 0; f < d; ++f)
            q_norm_sq += ptr_q[i * d + f] * ptr_q[i * d + f];

        // 计算到所有训练点的距离平方
        std::vector<std::pair<double, int>> dists(n_train);
        for (int j = 0; j < n_train; ++j) {
            double dot = 0.0;
            for (int f = 0; f < d; ++f)
                dot += ptr_q[i * d + f] * ptr_tr[j * d + f];
            double dist_sq = q_norm_sq - 2.0 * dot + train_norm_sq[j];
            if (dist_sq < 0.0) dist_sq = 0.0;
            dists[j] = {dist_sq, j};
        }

        // 部分排序：找 k 个最小的
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        for (int nn = 0; nn < k; ++nn)
            ptr_res[i * k + nn] = dists[nn].second;
    }

    return result;
}