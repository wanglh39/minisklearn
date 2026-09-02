// bindings.cpp —— pybind11 绑定入口
//
// pybind11 的工作原理：
//   1. PYBIND11_MODULE 宏展开为 Python 模块初始化函数
//   2. m.def() 注册 C++ 函数为 Python 可调用对象
//   3. py::array_t<double> 自动在 C++ 数组和 numpy 数组间转换
//   4. 编译后生成 .pyd 文件（Windows）或 .so 文件（Linux），Python 可直接 import

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// 声明外部函数
py::array_t<double> euclidean_distances(py::array_t<double>, py::array_t<double>);
py::array_t<int> knn_neighbors(py::array_t<double>, py::array_t<double>, int);
py::array_t<int> kmeans_assign(py::array_t<double>, py::array_t<double>);
py::array_t<double> kmeans_update(py::array_t<double>, py::array_t<int>, int);
double kmeans_inertia(py::array_t<double>, py::array_t<int>, py::array_t<double>);

PYBIND11_MODULE(_minisklearn_fast, m) {
    m.doc() = "minisklearn C++ 加速模块";

    m.def("euclidean_distances", &euclidean_distances,
          "计算两组点之间的欧氏距离矩阵 (C++ 实现)");

    m.def("knn_neighbors", &knn_neighbors,
          "找到每个查询点的 k 个最近邻索引 (C++ 实现)");

    m.def("kmeans_assign", &kmeans_assign,
          "KMeans 分配步：每个点分配到最近质心 (C++ 实现)");

    m.def("kmeans_update", &kmeans_update,
          "KMeans 更新步：重新计算质心 (C++ 实现)");

    m.def("kmeans_inertia", &kmeans_inertia,
          "计算 KMeans 的 inertia (C++ 实现)");
}