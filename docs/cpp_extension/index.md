# C++ 扩展与性能对比

> minisklearn 的纯 Python/NumPy 实现清晰易读，但性能不是最优。本章节用 C++ + pybind11 加速核心计算循环，并三方对比（纯 Python vs C++ vs sklearn），揭示"为什么 C++ 更快"。
>
> 本章节不仅展示结果，更拆解每一层原因：从 CPU 缓存到 SIMD 向量化，从内存分配器到编译器优化，让你理解性能差异的**物理根因**。

---

## 目录

- [一、为什么需要 C++ 扩展？](#一为什么需要-c-扩展)
- [二、pybind11 工作原理](#二pybind11-工作原理)
- [三、CMake 构建系统详解](#三cmake-构建系统详解)
- [四、实现的 C++ 函数逐行解析](#四实现的-c-函数逐行解析)
- [五、性能对比结果](#五性能对比结果)
- [六、性能差异的物理根因](#六性能差异的物理根因)
- [七、CPU 缓存与内存布局](#七cpu-缓存与内存布局)
- [八、SIMD 向量化与编译优化](#八simd-向量化与编译优化)
- [九、与 sklearn 的深度对比](#九与-sklearn-的深度对比)
- [十、跨平台构建差异](#十跨平台构建差异)
- [十一、常见问题与解决方案](#十一常见问题与解决方案)
- [十二、扩展开发指南](#十二扩展开发指南)
- [十三、编译与使用](#十三编译与使用)
- [十四、架构回扣](#十四架构回扣)
- [十五、性能调优 Checklist](#十五性能调优-checklist)

---

## 一、为什么需要 C++ 扩展？

### 1.1 NumPy 向量化的极限

NumPy 的向量化已经很快——它把 Python 循环下沉到 C 层。但向量化有代价：

```python
# KMeans 分配步的 NumPy 写法
dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#        广播产生 (n, k, d) 临时数组！
```

当 n=5000, k=10, d=10 时，临时数组有 5000×10×10 = 50万个元素。每次迭代都分配/释放这个数组，开销巨大。

让我们更详细地分析这行代码的执行过程：

```python
# 步骤 1：广播
# X[:, None, :]        → shape (n, 1, d)   ← 视图，无拷贝
# centroids[None, :, :] → shape (1, k, d)  ← 视图，无拷贝
# X[:, None, :] - centroids[None, :, :] → shape (n, k, d)  ← ★ 新分配！

# 步骤 2：平方
# result ** 2 → shape (n, k, d)  ← ★ 可能原地，也可能新分配

# 步骤 3：求和
# .sum(axis=2) → shape (n, k)  ← ★ 再分配
```

每次迭代涉及 **2-3 次大数组分配/释放**。Python 的内存分配器（`malloc`/`free`）虽然快，但在热循环中反复调用会成为瓶颈。

### 1.2 Python 解释器的开销

即使不考虑临时数组，Python 解释器本身也有固有开销：

```python
# Python 循环：每次迭代有解释器开销
for i in range(n):
    for j in range(k):
        dist = 0.0
        for f in range(d):
            dist += (X[i, f] - C[j, f]) ** 2  # 每次操作都有：
            #   1. 类型检查（X[i,f] 是什么类型？）
            #   2. 引用计数（增加/减少）
            #   3. 创建临时 Python 对象
            #   4. 字节码解释
```

每次 Python 操作涉及：

| 开销来源 | 说明 | 典型耗时 |
|---------|------|---------|
| **字节码解释** | Python 虚拟机执行字节码 | ~50ns |
| **类型检查** | 动态类型需要运行时检查 | ~10ns |
| **引用计数** | 每个对象的引用计数增减 | ~5ns |
| **对象创建** | 临时 Python 对象的分配/释放 | ~30ns |
| **全局解释器锁** | GIL 阻止真正的并行 | 间接开销 |

一个简单的浮点加法在 Python 中约 50-100ns，而 C++ 中是 0.3ns（3GHz CPU），**差距 150-300 倍**。

### 1.3 C++ 的优势

```cpp
// C++ 直接遍历，无临时数组
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
        double dist_sq = 0.0;
        for (int f = 0; f < d; ++f)
            dist_sq += (X[i*d+f] - C[j*d+f]) * (X[i*d+f] - C[j*d+f]);
        // ...
    }
}
```

C++ 优势：

| 优势 | 说明 | 量级 |
|------|------|------|
| **零临时数组** | 直接累加，不分配中间结果 | 省去 2-3 次 malloc/free |
| **缓存友好** | 行主序连续访问，CPU 缓存命中率高 | L1 缓存命中 ~1ns vs 内存访问 ~100ns |
| **编译优化** | `-O3` 自动向量化（SIMD）、循环展开 | 2-4x 加速 |
| **零开销抽象** | 无 Python 对象引用计数、类型检查 | 省去每次操作 50-100ns |
| **寄存器分配** | 编译器将热变量放在寄存器中 | 寄存器 ~0.3ns vs 内存 ~100ns |
| **分支预测** | 编译器优化分支模式 | 减少流水线气泡 |

### 1.4 实测加速比

| 场景 | C++ vs NumPy | 原因 |
|------|-------------|------|
| KMeans 核心循环 (n=5000, d=10, k=10) | **18x** | 消除了 (n,k,d) 临时数组 |
| KMeans 核心循环 (n=2000, d=100, k=10) | **15x** | 同上 |
| 欧氏距离矩阵 (n=5000, q=500, d=10) | **4.5x** | 预计算范数 + 连续访问 |
| 欧氏距离矩阵 (d=100) | **0.9x** | 高维时 NumPy 已足够高效 |

**关键发现**：维度低时 C++ 优势最大（临时数组开销占比高）；维度高时优势减小（计算本身成为瓶颈，NumPy 的 BLAS 调用已经很快）。

### 1.5 什么时候不该用 C++？

C++ 扩展不是银弹。以下场景不适合：

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| **I/O 密集型** | 瓶颈在磁盘/网络，不在 CPU | 异步 I/O |
| **调用 NumPy/SciPy 库函数** | 底层已经是 C/Fortran | 直接用库 |
| **逻辑密集型** | 大量 if/else 分支，少数值计算 | 纯 Python |
| **需要快速迭代** | 编译-调试循环慢 | 先用 Python 原型 |
| **小数据量** | Python 开销占比小 | 纯 Python |

**经验法则**：先用 `cProfile` 找到瓶颈，确认是数值计算密集型循环，再考虑 C++ 扩展。

---

## 二、pybind11 工作原理

### 2.1 架构

```
Python 代码
    ↓  import
minisklearn._fast._minisklearn_fast  (.pyd / .so)
    ↓  pybind11 绑定
C++ 函数 (distances.cpp, kmeans.cpp)
    ↓  编译
机器码 (-O3 优化)
```

### 2.2 为什么选 pybind11 而非其他方案？

Python C 扩展有多种方案，各有优劣：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Python C API** | 最灵活、最快 | 开发极其繁琐、容易出错 | 需要精细控制 |
| **ctypes** | 无需编译 C 扩展、标准库 | 手动类型转换、无类型安全 | 调用现有 C 库 |
| **cffi** | 比 ctypes 更安全 | 仍需手动管理 | 调用 C 库 |
| **Cython** | Python-like 语法、成熟 | 需学习 Cython 语法、生成 .c 文件 | 大规模扩展（sklearn 用） |
| **pybind11** | 现代 C++、类型安全、自动转换 | 只支持 C++、编译较慢 | 中小规模扩展 ★ |

pybind11 的核心优势：

1. **纯 C++ 头文件库**：无需额外编译工具，`#include` 即可
2. **自动类型转换**：Python `list` ↔ C++ `std::vector`，numpy `ndarray` ↔ `py::array_t`
3. **异常自动转换**：C++ `std::runtime_error` → Python `RuntimeError`
4. **numpy 零拷贝**：直接操作 numpy 数组的底层内存
5. **现代 C++ 风格**：RAII、智能指针、lambda，无手动内存管理

### 2.3 pybind11 的核心机制

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// C++ 函数：接收 numpy 数组，返回 numpy 数组
py::array_t<double> euclidean_distances(
    py::array_t<double> X1,
    py::array_t<double> X2
) {
    // request() 获取数组的内存信息（不拷贝数据）
    auto buf1 = X1.request();
    int n1 = buf1.shape[0];
    int d = buf1.shape[1];

    // 直接通过指针访问数据
    double *ptr1 = static_cast<double *>(buf1.ptr);

    // 创建输出数组
    auto result = py::array_t<double>({n1, n2});
    // ... 计算逻辑 ...
    return result;
}

// 注册为 Python 模块
PYBIND11_MODULE(_minisklearn_fast, m) {
    m.def("euclidean_distances", &euclidean_distances);
}
```

**关键设计**：

#### `py::array_t<T>`：numpy 数组包装器

```cpp
py::array_t<double> X1  // 接收 numpy float64 数组
auto buf = X1.request(); // 获取 buffer_info（不拷贝数据）

// buf 包含：
//   buf.ptr   → 数据指针（直接访问底层内存）
//   buf.shape → 各维度大小 vector
//   buf.strides → 各维度步长 vector
//   buf.ndim  → 维度数
//   buf.itemsize → 每个元素的字节数
```

**零拷贝**是关键：`request()` 返回的是对 numpy 数组底层内存的直接引用，不复制数据。C++ 代码直接在 numpy 的内存上操作，完成后返回新创建的 numpy 数组。

#### `PYBIND11_MODULE`：模块注册宏

```cpp
PYBIND11_MODULE(_minisklearn_fast, m) {
    m.doc() = "minisklearn C++ 加速模块";
    m.def("euclidean_distances", &euclidean_distances, "计算欧氏距离矩阵");
}
```

这个宏展开为 Python 模块初始化函数（`PyInit__minisklearn_fast`），它在 Python `import` 时被调用。`m.def()` 注册 C++ 函数为 Python 可调用对象。

编译后生成 `.pyd`（Windows）或 `.so`（Linux），Python 可直接 `import`：

```python
from minisklearn._fast._minisklearn_fast import euclidean_distances
```

#### 类型转换系统

pybind11 自动在 Python 类型和 C++ 类型之间转换：

| Python 类型 | C++ 类型 | 转换方式 |
|------------|---------|---------|
| `int` | `int`, `long` | 直接 |
| `float` | `double`, `float` | 直接 |
| `str` | `std::string` | 拷贝（UTF-8） |
| `list` | `std::vector<T>` | 逐元素 |
| `dict` | `std::map<K,V>` | 逐键值对 |
| `numpy.ndarray` | `py::array_t<T>` | 零拷贝 |
| `None` | `std::nullopt` / `nullptr` | 直接 |
| `tuple` | `std::tuple<T...>` | 逐元素 |

### 2.4 异常处理

pybind11 自动将 C++ 异常转换为 Python 异常：

```cpp
// C++ 抛出异常
if (k <= 0) {
    throw std::runtime_error("k 必须为正数");
}

// Python 端收到
// >>> knn_neighbors(X, Q, -1)
// RuntimeError: k 必须为正数
```

| C++ 异常 | Python 异常 |
|---------|------------|
| `std::runtime_error` | `RuntimeError` |
| `std::invalid_argument` | `ValueError` |
| `std::out_of_range` | `IndexError` |
| `std::bad_alloc` | `MemoryError` |
| 自定义 | 需注册 |

### 2.5 GIL（全局解释器锁）管理

Python 的 GIL 阻止多线程真正并行执行 Python 代码。但在 C++ 扩展中可以释放 GIL：

```cpp
#include <pybind11/pybind11.h>

py::array_t<double> heavy_compute(py::array_t<double> X) {
    // 释放 GIL，允许其他 Python 线程运行
    py::gil_scoped_release release;

    // 纯 C++ 计算，不触碰 Python 对象
    // ... 大量计算 ...

    // 重新获取 GIL（返回 Python 对象时需要）
    // py::gil_scoped_acquire acquire;  // 自动在函数返回时获取
    return result;
}
```

本项目的函数都是纯数值计算，不回调 Python，因此可以安全释放 GIL。但为了教学简洁，我们暂未添加 `gil_scoped_release`。

### 2.6 构建流程

```
CMakeLists.txt
    ↓  cmake configure
    ↓  cmake build
g++ -O3 -shared -o _minisklearn_fast.cp313-win_amd64.pyd
    distances.cpp
    kmeans.cpp
    bindings.cpp
    -lpython3.13
    -I<python-headers>
    -I<pybind11-headers>
```

**CMakeLists.txt** 核心：

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(_minisklearn_fast
    src/distances.cpp
    src/kmeans.cpp
    src/bindings.cpp
)
```

`pybind11_add_module` 是 pybind11 提供的 CMake 宏，自动设置：
- 包含 Python 头文件（`Python.h`）
- 链接 Python 库（`libpython3.x`）
- 生成正确命名的扩展文件（`.cp313-win_amd64.pyd`）
- 设置 C++ 标准（C++11 或更高）
- 添加 pybind11 头文件路径

---

## 三、CMake 构建系统详解

### 3.1 为什么用 CMake？

CMake 是 C/C++ 项目的标准构建系统。它的优势：

| 优势 | 说明 |
|------|------|
| **跨平台** | 同一份 CMakeLists.txt 生成 Windows/Linux/macOS 的构建文件 |
| **依赖管理** | `find_package` 自动查找依赖库 |
| **IDE 集成** | 可生成 Visual Studio / Xcode / Makefiles 项目 |
| **构建变体** | Debug/Release/RelWithDebInfo 等配置 |
| **广泛使用** | C/C++ 生态的事实标准 |

### 3.2 CMakeLists.txt 逐行解析

```cmake
# 指定 CMake 最低版本（3.15 支持 pybind11 的现代 CMake target）
cmake_minimum_required(VERSION 3.15)

# 项目名和语言
project(minisklearn_fast LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release 构建（-O3 优化）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找 pybind11（通过 pip 安装的 cmake 配置）
find_package(pybind11 REQUIRED)

# 定义扩展模块
pybind11_add_module(_minisklearn_fast
    src/distances.cpp      # 距离计算函数
    src/kmeans.cpp         # KMeans 核心函数
    src/bindings.cpp       # pybind11 绑定入口
)

# 设置输出目录为 minisklearn/_fast/
set_target_properties(_minisklearn_fast PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../minisklearn/_fast
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../minisklearn/_fast
)
```

### 3.3 构建变体

CMake 支持多种构建类型，对应不同的编译器优化级别：

| 构建类型 | GCC/Clang 标志 | 用途 |
|---------|---------------|------|
| `Debug` | `-O0 -g` | 调试，无优化，有调试信息 |
| `Release` | `-O3 -DNDEBUG` | 生产，最大优化 |
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | 带调试信息的优化构建 |
| `MinSizeRel` | `-Os -DNDEBUG` | 最小体积 |

```bash
# 指定构建类型
cmake -DCMAKE_BUILD_TYPE=Debug ...
cmake -DCMAKE_BUILD_TYPE=Release ...
```

### 3.4 Generator 选择

CMake 通过 generator 生成不同构建系统的文件：

| Generator | 平台 | 说明 |
|-----------|------|------|
| `MinGW Makefiles` | Windows | 用 MinGW 的 g++ 编译 |
| `MSVC Makefiles` | Windows | 用 Visual Studio 的 cl 编译 |
| `Visual Studio 17 2022` | Windows | 生成 VS 项目文件 |
| `Unix Makefiles` | Linux/macOS | 用系统 make |
| `Ninja` | 全平台 | 最快的构建系统 |

本项目在 Windows 上用 `MinGW Makefiles`（因为系统已安装 MinGW）：

```bash
cmake -G "MinGW Makefiles" ...
```

### 3.5 构建流程详解

```bash
# 步骤 1：配置（生成构建文件）
cmake -S cpp/ -B cpp/build/ -G "MinGW Makefiles" -Dpybind11_DIR=...
#   -S: 源码目录（含 CMakeLists.txt）
#   -B: 构建目录（生成 Makefile 等）
#   -G: 指定 generator
#   -D: 设置变量（pybind11 的 CMake 配置目录）

# 步骤 2：构建（编译 + 链接）
cmake --build cpp/build/ --config Release
#   --config: 指定构建类型（对多配置 generator 有效）

# 生成的文件：
#   minisklearn/_fast/_minisklearn_fast.cp313-win_amd64.pyd
```

### 3.6 pybind11 的 CMake 集成

pybind11 通过 `find_package` 与 CMake 集成：

```cmake
find_package(pybind11 REQUIRED)
# 这会设置以下变量：
#   pybind11_INCLUDE_DIRS  → pybind11 头文件路径
#   pybind11_LIBRARIES     → 需要链接的库
#   pybind11_PYTHON_VERSION → 检测到的 Python 版本

pybind11_add_module(my_module src.cpp)
# 这个宏做了以下事情：
#   1. 创建一个 SHARED library target
#   2. 添加 pybind11 和 Python 头文件路径
#   3. 链接 Python 库
#   4. 设置正确的输出文件名（含 Python 版本标签）
#   5. 设置 C++ 标准
```

---

## 四、实现的 C++ 函数逐行解析

### 4.1 欧氏距离矩阵（`distances.cpp`）

#### 数学原理

欧氏距离的展开式：

$$
\|x - y\|^2 = \|x\|^2 - 2 x \cdot y + \|y\|^2
$$

这个展开式的优势：

1. **预计算**：$\|y\|^2$ 可以预先计算，对所有 $x$ 复用
2. **矩阵化**：$X \cdot Y^T$ 是矩阵乘法，可用 BLAS 加速
3. **减少计算**：从 $O(d)$ 次减法 + $O(d)$ 次平方 + $O(d)$ 次加法，变为 $O(d)$ 次乘加

#### 逐行解析

```cpp
py::array_t<double> euclidean_distances(
    py::array_t<double> X1,   // (n1, d) 查询点
    py::array_t<double> X2    // (n2, d) 参考点
) {
    // 获取数组内存信息（零拷贝）
    auto buf1 = X1.request();
    auto buf2 = X2.request();

    int n1 = buf1.shape[0];   // 查询点数量
    int d = buf1.shape[1];    // 维度
    int n2 = buf2.shape[0];   // 参考点数量

    // 创建输出数组 (n1, n2)
    auto result = py::array_t<double>({n1, n2});
    auto buf_res = result.request();

    // 获取原始数据指针
    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr_res = static_cast<double *>(buf_res.ptr);

    // ★ 优化 1：预计算 X2 的行平方和 ||y||^2
    std::vector<double> x2_norm_sq(n2);
    for (int j = 0; j < n2; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k)
            s += ptr2[j * d + k] * ptr2[j * d + k];
        x2_norm_sq[j] = s;
    }
    // 复杂度：O(n2 * d)
    // 如果不预计算，每个距离都要重新算 ||y||^2，总共 O(n1 * n2 * d)

    // 对每行 X1[i] 计算到所有 X2[j] 的距离
    for (int i = 0; i < n1; ++i) {
        // ★ 优化 2：预计算 ||x||^2（对每个 i 只算一次）
        double x1_norm_sq = 0.0;
        for (int k = 0; k < d; ++k)
            x1_norm_sq += ptr1[i * d + k] * ptr1[i * d + k];

        for (int j = 0; j < n2; ++j) {
            // 内积 x·y
            double dot = 0.0;
            for (int k = 0; k < d; ++k)
                dot += ptr1[i * d + k] * ptr2[j * d + k];

            // ||x-y||^2 = ||x||^2 - 2*x·y + ||y||^2
            double dist_sq = x1_norm_sq - 2.0 * dot + x2_norm_sq[j];

            // ★ 数值修正：浮点误差可能导致 dist_sq 为极小负数
            if (dist_sq < 0.0) dist_sq = 0.0;

            ptr_res[i * n2 + j] = std::sqrt(dist_sq);
        }
    }

    return result;
}
```

#### 复杂度分析

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 预计算 `||y||^2` | $O(n_2 \cdot d)$ | 对每个参考点算一次 |
| 主循环 | $O(n_1 \cdot n_2 \cdot d)$ | 双重循环 + 内积 |
| 总计 | $O(n_1 \cdot n_2 \cdot d)$ | 与朴素实现相同，但常数因子更小 |

### 4.2 KNN 最近邻（`distances.cpp`）

```cpp
py::array_t<int> knn_neighbors(
    py::array_t<double> X_train,   // (n_train, d)
    py::array_t<double> X_query,   // (n_query, d)
    int k                           // 近邻数
) {
    // ... 内存设置 ...

    // 预计算训练集行平方和
    std::vector<double> train_norm_sq(n_train);
    for (int j = 0; j < n_train; ++j) {
        double s = 0.0;
        for (int f = 0; f < d; ++f)
            s += ptr_tr[j * d + f] * ptr_tr[j * d + f];
        train_norm_sq[j] = s;
    }

    for (int i = 0; i < n_query; ++i) {
        // 查询点的 ||q||^2
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
            dists[j] = {dist_sq, j};  // (距离, 原始索引)
        }

        // ★ 优化：partial_sort 只排序前 k 个
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        // 输出 k 个最近邻的索引
        for (int nn = 0; nn < k; ++nn)
            ptr_res[i * k + nn] = dists[nn].second;
    }

    return result;
}
```

#### `std::partial_sort` 的优势

```cpp
// 完全排序：O(n log n)
std::sort(dists.begin(), dists.end());

// 部分排序：O(n + k log k)，当 k << n 时快很多
std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
```

当 n=5000, k=5 时：
- `sort`: 5000 × log(5000) ≈ 5000 × 12.3 = 61,500 次比较
- `partial_sort`: 5000 + 5 × log(5) ≈ 5000 + 11.6 = 5,012 次比较

**约 12x 更少的比较操作**。

### 4.3 KMeans 分配步（`kmeans.cpp`）

```cpp
py::array_t<int> kmeans_assign(
    py::array_t<double> X,          // (n, d) 数据点
    py::array_t<double> centroids   // (k, d) 质心
) {
    // ... 内存设置 ...

    // 预计算质心行平方和
    std::vector<double> c_norm_sq(k);
    for (int j = 0; j < k; ++j) {
        double s = 0.0;
        for (int f = 0; f < d; ++f)
            s += ptr_c[j * d + f] * ptr_c[j * d + f];
        c_norm_sq[j] = s;
    }

    for (int i = 0; i < n; ++i) {
        // 数据点的 ||x||^2
        double x_norm_sq = 0.0;
        for (int f = 0; f < d; ++f)
            x_norm_sq += ptr_x[i * d + f] * ptr_x[i * d + f];

        // 找最近质心
        double best_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;

        for (int j = 0; j < k; ++j) {
            // 内积 x · c_j
            double dot = 0.0;
            for (int f = 0; f < d; ++f)
                dot += ptr_x[i * d + f] * ptr_c[j * d + f];

            // 距离平方
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
```

#### 对比 NumPy 实现

```python
# NumPy 写法：广播产生 (n, k, d) 临时数组
dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
assignments = dists.argmin(axis=1)
```

**内存对比**：

| 实现 | 临时数组 | 大小 (n=5000, k=10, d=10) |
|------|---------|--------------------------|
| NumPy 广播 | `(n, k, d)` 差值数组 | 5000×10×10×8 = 4 MB |
| NumPy 广播 | `(n, k, d)` 平方数组 | 4 MB（可能原地） |
| NumPy 广播 | `(n, k)` 求和数组 | 5000×10×8 = 400 KB |
| C++ | 无 | 0（在寄存器中累加） |

C++ 版本**零临时数组**，直接在寄存器中累加，缓存命中率远高于 NumPy 广播。

### 4.4 KMeans 更新步（`kmeans.cpp`）

```cpp
py::array_t<double> kmeans_update(
    py::array_t<double> X,           // (n, d)
    py::array_t<int> assignments,    // (n,)
    int k
) {
    // ... 内存设置 ...

    // 累加器：每个簇的各维度之和
    std::vector<double> sums(k * d, 0.0);
    std::vector<int> counts(k, 0);

    // 单次遍历：累加每个点到对应簇
    for (int i = 0; i < n; ++i) {
        int c = ptr_a[i];           // 该点所属簇
        counts[c]++;                // 簇大小 +1
        for (int f = 0; f < d; ++f)
            sums[c * d + f] += ptr_x[i * d + f];  // 累加各维度
    }

    // 求均值
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
```

**优化点**：单次遍历 $O(n \cdot d)$，而非对每个簇单独遍历 $O(k \cdot n \cdot d)$。

### 4.5 KMeans Inertia（`kmeans.cpp`）

```cpp
double kmeans_inertia(
    py::array_t<double> X,
    py::array_t<int> assignments,
    py::array_t<double> centroids
) {
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
```

**注意**：这里直接用差值平方计算（不展开），因为 inertia 只需距离平方和，不需要预计算范数。直接展开反而多了减法开销。

---

## 五、性能对比结果

### 5.1 欧氏距离矩阵

| n_train | n_query | d | Python | C++ | sklearn | C++/Python | sklearn/Python |
|---------|---------|---|--------|-----|---------|------------|----------------|
| 1000 | 100 | 10 | 0.0014s | 0.0007s | 0.0035s | 2.1x | 0.4x |
| 2000 | 200 | 10 | 0.0130s | 0.0027s | 0.0064s | 4.9x | 2.0x |
| 5000 | 500 | 10 | 0.1075s | 0.0240s | 0.0645s | 4.5x | 1.7x |
| 2000 | 200 | 50 | 0.0258s | 0.0086s | 0.0142s | 3.0x | 1.8x |
| 2000 | 200 | 100 | 0.0185s | 0.0210s | 0.0145s | 0.9x | 1.3x |

**发现**：
- 低维时 C++ 快 3-5x（临时数组开销占比大）
- 高维（d=100）时 C++ 反而慢于 NumPy（NumPy 调用 BLAS 优化矩阵乘法）
- sklearn 在小数据量时反而慢（固定开销大），大数据量时优势显现

### 5.2 KMeans 核心循环

| n | d | k | NumPy | C++ | C++/NumPy |
|---|---|---|-------|-----|-----------|
| 1000 | 10 | 5 | 0.0107s | 0.0014s | **7.7x** |
| 2000 | 10 | 5 | 0.0197s | 0.0029s | **6.9x** |
| 5000 | 10 | 10 | 0.7754s | 0.0430s | **18.0x** |
| 2000 | 50 | 10 | 0.7637s | 0.0443s | **17.2x** |
| 2000 | 100 | 10 | 0.7910s | 0.0513s | **15.4x** |

**发现**：KMeans 核心循环的 C++ 加速最显著（7-18x），因为 NumPy 的 `(n, k, d)` 广播数组在这里是最大的瓶颈。

### 5.3 KNN 完整预测

| n_train | n_test | d | minisklearn | sklearn | sklearn/minisklearn |
|---------|--------|---|-------------|---------|---------------------|
| 1000 | 200 | 10 | 0.0091s | 0.0042s | 2.1x |
| 2000 | 500 | 10 | 0.0521s | 0.0141s | 3.7x |
| 5000 | 1000 | 10 | 0.2514s | 0.0750s | 3.4x |
| 2000 | 500 | 50 | 0.0621s | 0.0568s | 1.1x |

**发现**：sklearn 的 KNN 底层用 C/Cython 实现，比 minisklearn 快 1-3.7x。高维时差距缩小。

### 5.4 KMeans 完整 fit

| n | d | k | minisklearn | sklearn | sklearn/minisklearn |
|---|---|---|-------------|---------|---------------------|
| 1000 | 10 | 5 | 0.0101s | 0.0137s | 0.7x |
| 2000 | 10 | 5 | 0.0166s | 0.0186s | 0.9x |
| 5000 | 10 | 10 | 0.0610s | 0.0513s | 1.2x |
| 2000 | 50 | 10 | 0.0186s | 0.0174s | 1.1x |

**发现**：minisklearn 的 KMeans 与 sklearn 性能接近，因为 minisklearn 的 KMeans 已经用了向量化优化。

---

## 六、性能差异的物理根因

### 6.1 为什么 C++ 在 KMeans 上加速最大？

NumPy 的 KMeans 分配步：

```python
dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
```

这行代码的内存开销：

```
X[:, None, :]        → (n, 1, d)   视图，无拷贝
centroids[None, :, :] → (1, k, d)  视图，无拷贝
相减                  → (n, k, d)   ★ 新分配的临时数组！
平方                  → (n, k, d)   ★ 又一个临时数组（或原地操作）
sum(axis=2)           → (n, k)      ★ 再分配
```

当 n=5000, k=10, d=10 时，临时数组 5000×10×10×8字节 = 4MB。每次迭代都分配/释放，**内存分配器成为瓶颈**。

C++ 版本完全不需要临时数组——所有计算在寄存器中完成。

### 6.2 内存分配器开销

Python/NumPy 的内存分配器基于 `malloc`/`free`，每次调用涉及：

```
malloc(4MB):
  1. 遍历空闲链表找合适块      → ~100-1000ns
  2. 可能触发系统调用 (mmap)   → ~10000ns
  3. 更新分配器内部数据结构    → ~50ns

free(4MB):
  1. 将块加入空闲链表          → ~50ns
  2. 可能合并相邻空闲块        → ~100ns
  3. 可能触发系统调用 (munmap) → ~10000ns
```

在 KMeans 的 100 次迭代中，每次迭代 2-3 次分配/释放 4MB 数组：

```
总分配开销 ≈ 100 次迭代 × 3 次分配 × 1000ns = 300,000ns = 0.3ms
```

这看似不大，但加上 **缓存失效**（新分配的内存不在缓存中）的开销，总影响可达 10-50ms，占 KMeans 总时间的 20-60%。

### 6.3 为什么高维时 C++ 优势减小？

高维（d=100）时：
- 计算量 = n×k×d 次乘加，计算本身成为瓶颈
- NumPy 内部调用 BLAS（高度优化的矩阵乘法库），利用 SIMD 向量化
- C++ 的 `-O3` 也会向量化，但 BLAS 更针对性优化
- 临时数组开销占比下降（计算时间上升）

**量化分析**：

```
低维 (d=10):
  计算时间 ≈ n×k×d×0.3ns = 5000×10×10×0.3ns = 150,000ns = 0.15ms
  分配开销 ≈ 0.3ms
  总时间 ≈ 0.45ms（分配占 67%）

高维 (d=100):
  计算时间 ≈ 5000×10×100×0.3ns = 1,500,000ns = 1.5ms
  分配开销 ≈ 0.3ms
  总时间 ≈ 1.8ms（分配占 17%）
```

高维时计算成为瓶颈，C++ 消除分配开销的收益占比下降。

### 6.4 为什么 sklearn 比 minisklearn 整体快？

sklearn 的关键算法用 C/Cython 实现：
- KNN：`scipy.spatial.cKDTree`（C++ 的 k-d 树）
- KMeans：Cython 编写的核心循环
- 决策树：Cython 的 splitter

但 sklearn 的优势主要在**算法层面**（如 k-d 树加速 KNN），而非单纯的循环优化。minisklearn 的教学目标是清晰易懂，不追求极致性能。

### 6.5 浮点运算的精度差异

C++ 和 NumPy 在浮点运算上可能有微小差异，因为：

1. **运算顺序**：不同的循环展开/向量化导致浮点加法顺序不同
2. **FMA 指令**：C++ `-O3` 可能使用 Fused Multiply-Add（`a*b+c` 一次完成，精度更高）
3. **中间精度**：x87 FPU 用 80 位中间精度，SSE 用 64 位

```python
# 这些差异通常在 1e-15 量级，不影响机器学习结果
# 但在测试中需要用 np.allclose 而非 np.array_equal
```

---

## 七、CPU 缓存与内存布局

### 7.1 CPU 缓存层次结构

现代 CPU 有多级缓存：

| 缓存层级 | 典型大小 | 访问延迟 | 说明 |
|---------|---------|---------|------|
| 寄存器 | ~256B | 0.3ns | 最快，编译器分配 |
| L1 缓存 | 32-64KB | 1ns | 每核独享 |
| L2 缓存 | 256KB-1MB | 3-10ns | 每核独享 |
| L3 缓存 | 8-32MB | 10-30ns | 多核共享 |
| 主内存 | GB 级 | 100-300ns | 所有核共享 |

**关键**：L1 缓存比主内存快 **100-300 倍**。缓存命中率对性能影响巨大。

### 7.2 行主序 vs 列主序

NumPy 默认行主序（C order），即 `X[i, j]` 的内存地址为 `base + (i * ncols + j) * itemsize`。

```cpp
// 行主序遍历（缓存友好）★ 推荐
for (int i = 0; i < n; ++i)
    for (int j = 0; j < d; ++j)
        sum += X[i * d + j];  // 连续访问，缓存命中率高

// 列主序遍历（缓存不友好）✗ 避免
for (int j = 0; j < d; ++j)
    for (int i = 0; i < n; ++i)
        sum += X[i * d + j];  // 步长为 d，每次跨一个缓存行
```

**量化**：当 n=5000, d=10 时，行主序遍历的缓存命中率约 95%，列主序约 50%。性能差距可达 2-3x。

### 7.3 本项目的缓存分析

```cpp
// KMeans 分配步的内存访问模式
for (int i = 0; i < n; ++i) {        // 外层：遍历每个数据点
    for (int j = 0; j < k; ++j) {    // 中层：遍历每个质心
        double dot = 0.0;
        for (int f = 0; f < d; ++f)  // 内层：内积
            dot += X[i*d+f] * C[j*d+f];
            //      ^^^^^^^^   ^^^^^^^^
            //      连续访问    连续访问
    }
}
```

- `X[i*d+f]`：对固定 i，f 从 0 到 d-1，**连续访问**，缓存友好
- `C[j*d+f]`：对固定 j，f 从 0 到 d-1，**连续访问**，缓存友好
- 当 d 较小时（如 d=10），整个 `X[i]` 和 `C[j]` 都在 L1 缓存中

### 7.4 缓行大小与数据对齐

现代 CPU 的缓存行通常为 64 字节：

```
一个 double 占 8 字节
一个缓存行 = 64 字节 = 8 个 double

如果数据未对齐到 64 字节边界：
  - 读取一个 double 可能需要两个缓存行
  - 性能下降约 10-20%
```

NumPy 的数组分配通常对齐到 64 字节，C++ 的 `new`/`malloc` 也对齐到 16 字节（但不保证 64 字节）。对性能要求极高的场景可以用 `aligned_alloc`。

---

## 八、SIMD 向量化与编译优化

### 8.1 SIMD 指令集

SIMD（Single Instruction, Multiple Data）一条指令处理多个数据：

| 指令集 | 寄存器宽度 | 同时处理 double 数 | 支持的 CPU |
|--------|-----------|-------------------|-----------|
| SSE2 | 128 位 | 2 | 所有 x86-64 |
| AVX | 256 位 | 4 | Intel Sandy Bridge+ |
| AVX-512 | 512 位 | 8 | Intel Skylake-X+ |
| NEON | 128 位 | 2 | ARM |

### 8.2 自动向量化

编译器在 `-O3` 下会自动将循环向量化：

```cpp
// 原始循环
for (int f = 0; f < d; ++f)
    dot += X[i*d+f] * C[j*d+f];

// AVX 向量化后（伪代码）
// 一次处理 4 个 double
__m256d sum = _mm256_setzero_pd();
for (int f = 0; f < d; f += 4) {
    __m256d x = _mm256_load_pd(&X[i*d+f]);
    __m256d c = _mm256_load_pd(&C[j*d+f]);
    sum = _mm256_fmadd_pd(x, c, sum);  // Fused Multiply-Add
}
dot = _mm256_reduce_add_pd(sum);
```

**加速比**：AVX 理论上 4x（4 个 double 同时处理），实际 2-3x（受内存带宽限制）。

### 8.3 编译优化级别

| 级别 | 标志 | 优化内容 |
|------|------|---------|
| `-O0` | 无优化 | 调试用，最快编译 |
| `-O1` | 基本优化 | 常量折叠、死代码消除 |
| `-O2` | 标准优化 | 循环优化、内联、向量化 |
| `-O3` | 激进优化 | 循环展开、FMA、更激进向量化 |
| `-Ofast` | 最激进 | `-O3` + `-ffast-math`（可能改变浮点语义） |

本项目用 `-O3`（Release 构建），平衡性能和数值正确性。

### 8.4 循环展开

`-O3` 会自动展开循环：

```cpp
// 原始循环
for (int f = 0; f < d; ++f)
    dot += X[f] * C[f];

// 展开后（展开因子 4）
for (int f = 0; f < d; f += 4) {
    dot += X[f] * C[f];
    dot += X[f+1] * C[f+1];
    dot += X[f+2] * C[f+2];
    dot += X[f+3] * C[f+3];
}
// 剩余部分单独处理
```

**优势**：
1. 减少循环判断开销（`f < d` 检查减少 4 倍）
2. 允许 CPU 流水线并行执行多条指令
3. 为向量化创造条件

### 8.5 Fused Multiply-Add (FMA)

FMA 指令将乘法和加法合并为一条指令：

```cpp
// 没有 FMA
double tmp = a * b;    // 乘法
double result = tmp + c;  // 加法
// 两次运算，两次舍入

// 有 FMA
double result = fma(a, b, c);  // 一次运算，一次舍入
// 更快且更精确
```

`-O3` 会自动使用 FMA 指令（如果 CPU 支持）。

### 8.6 内联函数

编译器会将小函数内联，消除调用开销：

```cpp
// 如果 std::sqrt 被内联，直接生成 sqrtsd 指令
// 而非 call sqrt + ret
```

---

## 九、与 sklearn 的深度对比

### 9.1 sklearn 的混合语言架构

sklearn 的性能秘密在于大量使用 Cython 和 C：

| 模块 | 实现语言 | 加速技术 |
|------|---------|---------|
| `sklearn.neighbors` | C++ (scipy cKDTree) | k-d 树、ball 树 |
| `sklearn.cluster.KMeans` | Cython | 释放 GIL、C 循环 |
| `sklearn.tree` | Cython | 结构体存储树节点 |
| `sklearn.svm` | C (libsvm) | SMO 算法优化 |
| `sklearn.linear_model` | Cython | SGD 循环 |
| `sklearn.decomposition.PCA` | Fortran (LAPACK) | SVD 分解 |

### 9.2 Cython vs pybind11

| 方面 | Cython | pybind11 |
|------|--------|----------|
| 语言 | Python-like (.pyx) | 纯 C++ |
| 学习曲线 | 需学 Cython 语法 | 只需 C++ 知识 |
| 编译 | .pyx → .c → .so | .cpp → .so |
| 类型系统 | 需显式声明 `cdef` | C++ 原生类型 |
| numpy 集成 | 原生支持 | `py::array_t` |
| 性能 | 相近 | 相近 |
| 适合场景 | 大规模扩展 | 中小规模、已有 C++ 代码 |

sklearn 选 Cython 是历史原因（2010 年代初 pybind11 还不存在）。新项目更推荐 pybind11。

### 9.3 算法层面 vs 实现层面优化

sklearn 的很多优势来自**算法层面**的优化，而非单纯的循环加速：

| 优化 | sklearn | minisklearn | 效果 |
|------|---------|-------------|------|
| KNN k-d 树 | ✅ | ❌ 暴力搜索 | O(log n) vs O(n) |
| KMeans++ 初始化 | ✅ | ✅ | 更快收敛 |
| KMeans Elkan 算法 | ✅ | ❌ | 三角不等式避免距离计算 |
| 决策树 presort | ✅ | ❌ | 预排序加速分裂搜索 |

**教学取舍**：minisklearn 用最直接的算法（暴力 KNN、标准 Lloyd KMeans），让读者理解原理。C++ 扩展只加速**循环层面**，不改算法。

---

## 十、跨平台构建差异

### 10.1 三平台对比

| 方面 | Windows | Linux | macOS |
|------|---------|-------|-------|
| 扩展后缀 | `.pyd` | `.so` | `.so` |
| 文件名标签 | `.cp313-win_amd64` | `.cpython-313-x86_64-linux-gnu` | `.cpython-313-darwin` |
| 默认编译器 | MSVC / MinGW | GCC / Clang | Clang |
| CMake Generator | `MinGW Makefiles` / `Visual Studio` | `Unix Makefiles` | `Unix Makefiles` |
| DLL 依赖 | 需 `os.add_dll_directory` | `LD_LIBRARY_PATH` | `DYLD_LIBRARY_PATH` |

### 10.2 Windows 的 DLL 依赖问题

MinGW 编译的 `.pyd` 依赖 MinGW 运行时 DLL：

```
_minisklearn_fast.cp313-win_amd64.pyd
  → libstdc++-6.dll    (C++ 标准库)
  → libgcc_s_seh-1.dll  (GCC 运行时)
  → libwinpthread-1.dll (线程库)
```

Python 3.8+ 改变了 DLL 搜索机制，需要显式添加搜索路径：

```python
import os
os.add_dll_directory(r"C:\mingw64\bin")
# 或
os.environ["PATH"] = r"C:\mingw64\bin;" + os.environ["PATH"]
```

本项目的 `minisklearn/_fast/__init__.py` 自动处理了这个依赖。

### 10.3 Linux/macOS 构建

Linux 和 macOS 的构建更简单，不需要处理 DLL 依赖：

```bash
# Linux
cmake -S cpp/ -B cpp/build/ -Dpybind11_DIR=...
cmake --build cpp/build/
# 生成: minisklearn/_fast/_minisklearn_fast.cpython-313-x86_64-linux-gnu.so

# macOS
cmake -S cpp/ -B cpp/build/ -Dpybind11_DIR=...
cmake --build cpp/build/
# 生成: minisklearn/_fast/_minisklearn_fast.cpython-313-darwin.so
```

### 10.4 跨平台 CMakeLists.txt

```cmake
# 自动选择 generator
if(WIN32)
    # Windows: 优先 MinGW，回退 MSVC
    if(EXISTS "C:/mingw64/bin/g++.exe")
        set(CMAKE_GENERATOR "MinGW Makefiles")
    else()
        set(CMAKE_GENERATOR "Visual Studio 17 2022")
    endif()
else()
    # Linux/macOS: Unix Makefiles
    set(CMAKE_GENERATOR "Unix Makefiles")
endif()
```

---

## 十一、常见问题与解决方案

### 11.1 编译错误

#### 问题：`py::array_t` 未定义

```
error: 'array_t' in namespace 'py' does not name a template type
```

**原因**：缺少 `pybind11/numpy.h` 头文件。

**解决**：在 bindings.cpp 中添加：

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // ← 必须包含
```

#### 问题：`cmake` 命令找不到

```
FileNotFoundError: [WinError 2] 系统找不到指定的文件
```

**原因**：cmake 不在 PATH 中。

**解决**：用完整路径或通过 pip 安装：

```bash
pip install cmake  # 安装到 venv/Scripts/cmake.exe
```

#### 问题：链接器找不到 Python 库

```
error: cannot find -lpython3.13
```

**原因**：Python 库路径未正确设置。

**解决**：确保 CMake 能找到 Python：

```cmake
find_package(Python3 COMPONENTS Development REQUIRED)
```

### 11.2 运行时错误

#### 问题：`DLL load failed`

```
ImportError: DLL load failed while importing _minisklearn_fast: 找不到指定的模块。
```

**原因**：MinGW 运行时 DLL 不在搜索路径中。

**解决**：

```python
import os
os.add_dll_directory(r"C:\mingw64\bin")
```

本项目的 `__init__.py` 已自动处理。

#### 问题：`undefined symbol`

```
ImportError: undefined symbol: PyInit__minisklearn_fast
```

**原因**：模块名与 PYBIND11_MODULE 宏中的名称不匹配。

**解决**：确保两者一致：

```cpp
PYBIND11_MODULE(_minisklearn_fast, m) { ... }
// 文件名必须是 _minisklearn_fast.xxx.pyd
```

### 11.3 性能问题

#### 问题：C++ 扩展反而更慢

**可能原因**：

1. **未开优化**：Debug 构建（`-O0`）比 NumPy 慢
2. **数据拷贝**：`py::array_t` 的 `request()` 应该零拷贝，但如果传入了非连续数组可能拷贝
3. **维度不匹配**：高维时 NumPy 的 BLAS 调用可能更快

**诊断方法**：

```python
# 检查数组是否连续
print(X.flags['C_CONTIGUOUS'])  # 应为 True

# 检查构建类型
# CMakeLists.txt 中应设置 Release
```

#### 问题：数值结果不一致

**原因**：浮点运算顺序不同导致舍入差异。

**解决**：测试中用 `np.allclose` 而非 `np.array_equal`：

```python
assert np.allclose(D_cpp, D_py, rtol=1e-10)
```

---

## 十二、扩展开发指南

### 12.1 如何添加新的 C++ 函数

以添加一个"曼哈顿距离矩阵"函数为例：

#### 步骤 1：编写 C++ 函数

在 `cpp/src/distances.cpp` 中添加：

```cpp
py::array_t<double> manhattan_distances(
    py::array_t<double> X1,
    py::array_t<double> X2
) {
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

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            double dist = 0.0;
            for (int f = 0; f < d; ++f)
                dist += std::abs(ptr1[i*d+f] - ptr2[j*d+f]);
            ptr_res[i*n2+j] = dist;
        }
    }

    return result;
}
```

#### 步骤 2：在 bindings.cpp 中注册

```cpp
// 声明
py::array_t<double> manhattan_distances(py::array_t<double>, py::array_t<double>);

// 在 PYBIND11_MODULE 中添加
m.def("manhattan_distances", &manhattan_distances,
      "计算曼哈顿距离矩阵 (C++ 实现)");
```

#### 步骤 3：在 Python 接口中导出

在 `minisklearn/_fast/__init__.py` 中添加 `manhattan_distances` 到导入列表。

#### 步骤 4：重新编译

```bash
python cpp/build.py
```

#### 步骤 5：测试

```python
from minisklearn._fast import manhattan_distances
D = manhattan_distances(X1, X2)
```

### 12.2 编写测试

```python
def test_manhattan_distances():
    X1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    X2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    D = manhattan_distances(X1, X2)
    expected = np.array([[0, 1, 1], [2, 1, 1]])
    assert np.allclose(D, expected)
```

### 12.3 性能优化技巧

1. **预计算**：将循环不变的计算提到循环外
2. **连续访问**：确保内存访问是连续的（行主序）
3. **避免分支**：减少 if/else 在热循环中
4. **使用 `restrict`**：告诉编译器指针不重叠，允许更激进优化
5. **循环展开**：手动或让编译器自动展开
6. **FMA**：用 `std::fma` 或让编译器自动使用

```cpp
// restrict 关键字（C++ 中用 __restrict）
void compute(double* __restrict X, double* __restrict Y) {
    // 编译器知道 X 和 Y 不重叠，可以更激进优化
}
```

---

## 十三、编译与使用

### 13.1 环境准备

```bash
# 创建虚拟环境
uv venv --python 3.13

# 安装依赖
uv pip install numpy cmake pybind11 scikit-learn matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 minisklearn
uv pip install -e .
```

### 13.2 编译 C++ 扩展

```bash
# 编译
python cpp/build.py

# 输出：
# minisklearn/_fast/_minisklearn_fast.cp313-win_amd64.pyd
```

### 13.3 使用

```python
from minisklearn._fast import is_available, euclidean_distances, kmeans_assign

if is_available():
    D = euclidean_distances(X1, X2)  # C++ 版本
else:
    # 回退到纯 Python 版本
    from minisklearn.neighbors._distances import euclidean_distances
    D = euclidean_distances(X1, X2)
```

### 13.4 运行性能对比

```bash
python benchmarks/run_benchmarks.py
```

### 13.5 从源码重新编译

```bash
# 清除旧构建
rm -rf cpp/build/

# 重新编译
python cpp/build.py
```

---

## 十四、架构回扣

### 14.1 混合语言架构

sklearn 本身就是混合语言架构：Python（API 层）+ C/Cython（计算层）。本项目的 C++ 扩展复现了这个模式：

```
Python（minisklearn）         ← API 层：统一接口、参数管理、文档
    ↓
C++（_minisklearn_fast）      ← 计算层：核心循环加速
```

这是第 1 讲[统一 API](../architecture/01-unified-api.md)的延伸——用户不需要知道底层是 Python 还是 C++，接口一致。

### 14.2 渐进式优化

本项目的性能优化路径：

```
纯 Python 循环    →    NumPy 向量化    →    C++ 扩展
（最清晰）           （清晰 + 快）        （最快但需编译）
```

这反映了实际工程中的渐进式优化策略：先写正确的代码，再向量化，最后才考虑 C++ 扩展。

**Donald Knuth 的名言**："过早优化是万恶之源。" 先用最清晰的方式实现，确认正确后再优化瓶颈。

### 14.3 与基类系统的协作

C++ 扩展不参与 sklearn 的基类系统（BaseEstimator / Mixin），它是**纯函数**层面的加速：

```python
# minisklearn 的 KMeans 用 NumPy 实现核心循环
class KMeans(BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        # ... 初始化 ...
        for _ in range(max_iter):
            # 这里可以用 C++ 加速
            assignments = kmeans_assign(X, centroids)  # C++ 版
            centroids = kmeans_update(X, assignments, k)  # C++ 版
```

C++ 函数是**无状态**的，不继承 BaseEstimator，不参与 clone/get_params。它们只是被 Python 类调用的工具函数。

### 14.4 回扣各讲架构设计

| 架构讲 | 与 C++ 扩展的关系 |
|--------|------------------|
| [第 1 讲 统一 API](../architecture/01-unified-api.md) | C++ 函数遵循相同的接口约定 |
| [第 4 讲 元估计器](../architecture/04-meta-estimator.md) | C++ 扩展可被 Pipeline/GridSearchCV 透明使用 |
| [第 5 讲 数据约定](../architecture/05-data-convention.md) | C++ 函数接收 numpy 数组，遵循相同数据格式 |
| [第 6 讲 一致性测试](../architecture/06-consistency-testing.md) | C++ 结果需与 Python 版本一致（np.allclose） |

---

## 十五、性能调优 Checklist

在考虑用 C++ 加速前，先过一遍这个清单：

### 15.1 优化前

- [ ] 用 `cProfile` 找到真正的瓶颈（而非凭直觉）
- [ ] 确认瓶颈是数值计算密集型（而非 I/O 或逻辑）
- [ ] 已用 NumPy 向量化（`X @ Y` 而非循环）
- [ ] 已避免不必要的临时数组（用 `out=` 参数）
- [ ] 已用 `np.ascontiguousarray` 确保数组连续
- [ ] 已尝试 `numba.jit`（比 C++ 扩展更简单的加速方式）

### 15.2 编写 C++ 扩展时

- [ ] 用 `py::array_t` 零拷贝访问 numpy 数组
- [ ] 预计算循环不变量
- [ ] 确保内存访问连续（行主序）
- [ ] 用 `partial_sort` 而非 `sort`（只需前 k 个时）
- [ ] 添加数值修正（`if (dist_sq < 0) dist_sq = 0`）
- [ ] 释放 GIL（`py::gil_scoped_release`，如果纯 C++ 计算）

### 15.3 编译时

- [ ] 用 Release 构建（`-O3`）
- [ ] 启用 LTO（链接时优化，`-flto`）
- [ ] 针对目标 CPU 架构编译（`-march=native`）
- [ ] 检查编译器向量化报告（`-fopt-info-vec`）

### 15.4 验证时

- [ ] 结果与 Python 版本一致（`np.allclose`）
- [ ] 性能确实提升（用 `timeit` 基准测试）
- [ ] 无内存泄漏（用 valgrind 或 AddressSanitizer）
- [ ] 线程安全（如果用了多线程）
- [ ] 跨平台兼容（Windows/Linux/macOS）

### 15.5 性能分析工具

| 工具 | 用途 | 平台 |
|------|------|------|
| `cProfile` | Python 级性能分析 | 全平台 |
| `line_profiler` | 逐行分析 Python 代码 | 全平台 |
| `perf` | Linux 性能分析（缓存命中、分支预测） | Linux |
| `Instruments` | macOS 性能分析 | macOS |
| `VTune` | Intel 性能分析器 | 全平台 |
| `valgrind --tool=callgrind` | 调用图分析 | Linux |
| `-fopt-info-vec` | GCC 向量化报告 | GCC |

---

## 附录：项目文件结构

```
cpp/                            # C++ 扩展源码
├── CMakeLists.txt              # CMake 构建配置
├── build.py                    # 编译脚本
├── src/
│   ├── distances.cpp           # 距离计算（欧氏距离 + KNN近邻）
│   ├── kmeans.cpp              # KMeans 核心循环（分配 + 更新 + inertia）
│   └── bindings.cpp            # pybind11 绑定入口
└── build/                      # CMake 构建目录（自动生成）

minisklearn/
└── _fast/                      # C++ 加速模块
    ├── __init__.py             # 自动加载 + DLL 依赖处理
    └── _minisklearn_fast.*.pyd # 编译生成的扩展文件

benchmarks/                     # 性能对比基准
├── __init__.py
├── run_benchmarks.py           # 主运行脚本
├── benchmark_knn.py            # KNN 性能对比
└── benchmark_kmeans.py         # KMeans 性能对比
```

---

[← 返回算法列表](../algorithms/index.md)
