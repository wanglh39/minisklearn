# minisklearn —— 从零实现 sklearn

> 理解设计哲学，拆解底层原理，从零手写一个 mini scikit-learn。

---

## 这是什么项目？

`scikit-learn`（sklearn）是 Python 机器学习生态的基石。它的伟大**不在于算法多**——很多算法的实现在专用库中更快——而在于**架构设计**：

- **统一 API**：所有算法遵循 `fit` / `predict` / `transform` 契约，学一个就会用上百个
- **Mixin 多继承**：用极简的基类组合出分类器、回归器、转换器等不同身份
- **元估计器**：`Pipeline`、`GridSearchCV` 用组合而非继承串联一切
- **参数管理**：`get_params` / `set_params` / `clone` 的反射机制支撑了整个生态

本项目通过**从零实现**这些机制和核心算法，让你真正理解 sklearn 的设计思想，而不是停留在"会调 API"的层面。

### 这个项目适合谁？

- **想真正搞懂 sklearn 的 Python 工程师**：你已经用 sklearn 做过几个项目，但每次写 `pipeline.fit(X, y)` 时心里没底，想知道背后到底发生了什么
- **机器学习初学者**：你刚学完吴恩达的课，会推导梯度下降，但不知道怎么把数学公式变成工业级代码
- **架构设计爱好者**：你想看一个"小而美"的 Python 库是怎么设计的，Mixin、元估计器、反射参数管理这些模式在真实项目里怎么用
- **准备面试的同学**：面试官问你"sklearn 的 Pipeline 是怎么实现的"，你能不能从 `__init__` 一路讲到 `fit`？

### 这个项目不适合谁？

- 想找一个生产级 ML 库的人：请直接用 sklearn
- 想学深度学习的人：请看 PyTorch / TensorFlow
- 只想调 API 不关心原理的人：直接看 sklearn 文档更高效

---

## 为什么不直接读 sklearn 源码？

sklearn 是工业级库，代码量大、边界处理多、历史包袱重。直接读它的源码，很容易陷入细节泥潭：

- `LogisticRegression` 的源码有 1000+ 行，光处理多分类策略就几百行
- `BaseEstimator` 散落在多个文件里，`get_params` 的实现涉及 `_get_param_names`、`_get_param_indices` 等多个辅助函数
- 大量代码处理稀疏矩阵、joblib 并行、deprecation 警告，这些对理解核心逻辑没有帮助

minisklearn 做的是**剥离噪声、保留骨架**：

- 每个算法只保留最核心的实现，通常 100-200 行
- 架构机制完整保留，且配详细注释说明"为什么这么写"
- 每个设计决策都有对应的文档讲清楚来龙去脉

打个比方：sklearn 是一座运行中的城市，minisklearn 是这座城市的建筑蓝图。读蓝图比在城市里乱逛更容易理解城市规划。

---

## 设计哲学

### 哲学一：统一 API 是最大的抽象

sklearn 最了不起的地方，是定义了一套**所有算法都遵守的契约**：

```
估计器（Estimator）：
    fit(X, y) → self          # 从数据中学习

预测器（Predictor）：
    predict(X) → y_pred       # 预测

转换器（Transformer）：
    transform(X) → X_new      # 转换数据
    fit_transform(X, y) → X_new  # 拟合 + 转换（通常有优化实现）

评分（所有估计器）：
    score(X, y) → float       # 评估
```

就这四个方法，撑起了整个生态。为什么这这么重要？

**1. 学习成本极低**：你学会了 `LinearRegression`，就学会了 `RandomForest`、`SVC`、`KMeans`——它们都是 `fit` + `predict`。sklearn 有 100+ 个算法，但你只需要学一套 API。

**2. 组合性极强**：因为接口统一，`Pipeline` 可以串联任意算法，`GridSearchCV` 可以包装任意算法。如果每个算法的接口不一样，写一个通用的 Pipeline 几乎不可能。

**3. 测试可以复用**：sklearn 有一套"通用检查"（`check_estimator`），一个测试套件能测所有算法。因为所有算法都遵守同一契约，检查契约本身就成了测试。

本项目第一章就带你从零实现这套契约，让你体会"接口设计"的威力。

### 哲学二：Mixin 多继承优于大基类

很多库的设计是：一个大基类 `BaseModel`，里面有一堆方法，子类通过覆写来定制。sklearn 反其道而行之：

```
BaseEstimator              # 只管参数管理（get_params/set_params/clone）
    ↑
ClassifierMixin            # 只加一个 score = accuracy
RegressorMixin             # 只加一个 score = R²
TransformerMixin           # 只加一个 fit_transform
ClusterMixin               # 只加一个 fit_predict
```

每个 Mixin 只做一件事，算法通过多继承组合身份：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    # 既是估计器（有参数管理），又是分类器（score 是准确率）
    ...

class StandardScaler(BaseEstimator, TransformerMixin):
    # 既是估计器，又是转换器（有 fit_transform）
    ...
```

为什么这比大基类好？

- **单一职责**：每个 Mixin 只管一件事，改一个不影响其他
- **灵活组合**：一个类可以同时是分类器和转换器（比如 `LogisticRegression` 可以当特征提取器用）
- **避免父类地狱**：不需要在 `BaseModel` 里写一堆 `if` 判断子类类型

### 哲学三：元估计器用组合而非继承

`Pipeline`、`GridSearchCV`、`RandomForest` 都是"管理其他估计器的估计器"。sklearn 的做法是**组合**而非继承：

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # 持有其他估计器的引用
    
    def fit(self, X, y):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return self
```

`Pipeline` 不关心 `steps` 里是什么算法，只要它们遵守 `fit_transform` 契约就行。这就是为什么 Pipeline 能串联任意算法——它依赖的是**接口**而非**具体类型**。

### 哲学四：参数管理是元估计器的基石

`GridSearchCV` 要在参数空间里搜索，必须能：

1. **克隆**估计器（否则每次迭代都修改同一个对象）
2. **设置**参数（`set_params({'C': 1.0})`）
3. **获取**参数（`get_params()`）

sklearn 用反射实现了这套机制：

```python
class BaseEstimator:
    def get_params(self):
        # 反射 __init__ 的签名，返回 self 上同名的属性
        ...
    
    def set_params(self, **params):
        # 反射设置 self 上的属性
        ...
```

关键约定：**`__init__` 只存参数，不做任何计算**。这样 `get_params` 才能可靠地通过反射工作。这个约定看似简单，却是整个 sklearn 生态的基石。

---

## 学习路线

```
架构设计（地基）              算法实现（砌墙）              进阶（装修）
┌───────────────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│ 1. 统一 API 契约        │    │ preprocessing        │    │ 元估计器           │
│ 2. Mixin 多继承架构     │    │   StandardScaler     │    │   Pipeline        │
│ 3. 参数管理机制         │ →  │   LabelEncoder       │ →  │   GridSearchCV    │
│ 4. 数据约定与校验       │    │ linear_model         │    │                   │
│ 5. 元估计器模式         │    │   LinearRegression   │    │ C++ 性能对比       │
│ 6. 一致性测试机制       │    │   LogisticRegression │    │   pybind11 扩展    │
│ 7. 全局配置与演进       │    │ neighbors / tree     │    │                   │
│                        │    │ ensemble / cluster   │    │                   │
└───────────────────────┘    └──────────────────────┘    └───────────────────┘
```

**建议顺序**：先通读「架构设计」7 讲打地基，再按算法章节砌墙，每实现一个算法都回扣架构——体会"为什么有了这套架构，加新算法只需要写 `fit` 和 `predict`"。

### 详细路线表

| 阶段 | 内容 | 预计耗时 | 前置要求 | 产出 |
|------|------|----------|----------|------|
| 0 | 环境配置 | 0.5 小时 | Python 3.9+ | 能跑测试和文档 |
| 1 | 统一 API 契约 | 2 小时 | 会写 Python 类 | `BaseEstimator` 骨架 |
| 2 | Mixin 多继承 | 2 小时 | 阶段 1 | 4 个 Mixin |
| 3 | 参数管理机制 | 3 小时 | 阶段 1-2 | `get_params`/`set_params`/`clone` |
| 4 | 数据约定与校验 | 2 小时 | NumPy 基础 | `check_X_y`/`check_array` |
| 5 | 元估计器模式 | 3 小时 | 阶段 1-3 | 理解 Pipeline 原理 |
| 6 | 一致性测试 | 2 小时 | 阶段 1-4 | 通用测试套件 |
| 7 | 全局配置 | 1 小时 | 阶段 1 | `config_context` |
| 8 | 预处理算法 | 3 小时 | 阶段 1-4 | 4 个预处理器 |
| 9 | 线性模型 | 4 小时 | 线性代数基础 | 线性/逻辑回归 |
| 10 | KNN | 2 小时 | 阶段 4 | KNN 分类/回归 |
| 11 | 决策树 | 6 小时 | 信息论基础 | CART 树 |
| 12 | 集成学习 | 4 小时 | 阶段 11 | 随机森林 |
| 13 | 聚类 | 3 小时 | 阶段 4 | KMeans |
| 14 | 降维 | 3 小时 | 线性代数 | PCA |
| 15 | SVM | 4 小时 | 凸优化基础 | LinearSVC |
| 16 | 朴素贝叶斯 | 2 小时 | 概率论基础 | GaussianNB |
| 17 | 元算法实现 | 4 小时 | 阶段 5 | Pipeline/GridSearchCV |
| 18 | C++ 扩展 | 4 小时 | C++ 基础 | pybind11 模块 |
| 19 | 性能对比 | 2 小时 | 阶段 18 | 基准报告 |

总计约 50 小时，建议每天 1-2 小时，1-2 个月学完。

### 不同背景的推荐路线

**纯初学者（会 Python，没学过 ML）**：
1. 先看阶段 1-3（架构基础）
2. 跳过数学推导，直接用阶段 8-9 的算法
3. 遇到不懂的数学再补

**有 ML 基础（学过吴恩达）**：
1. 快速过阶段 1-3
2. 重点看阶段 9-16（算法实现）
3. 对比自己写的和本项目的实现

**资深工程师（想学架构）**：
1. 精读阶段 1-7
2. 选 2-3 个算法看实现
3. 重点看阶段 17（元估计器）

---

## 核心问题

这个项目试图回答以下问题：

### 架构层面

- 为什么所有算法都用 `fit` / `predict` 而不是 `train` / `infer`？
- 为什么 `__init__` 只存参数不做任何计算？
- 为什么用 Mixin 多继承而不是大基类？
- `clone` 为什么不用 `copy.deepcopy`？
- `Pipeline` 凭什么能串联任意算法？
- 为什么能写一个测试套件测所有算法？
- `get_params` 是怎么通过反射拿到参数的？
- 为什么 `fit` 必须返回 `self`？
- `check_is_fitted` 是怎么知道一个估计器有没有拟合的？

### 算法层面

- 线性回归的正规方程和梯度下降各有什么优劣？
- 逻辑回归的损失函数为什么是交叉熵而不是均方误差？
- KNN 的距离计算如何向量化？
- 决策树 CART 算法如何选择分裂点？
- 随机森林的 bagging 和特征随机选择各起什么作用？
- KMeans 的初始化为什么影响结果这么大？KMeans++ 好在哪？
- PCA 的 SVD 实现和协方差矩阵实现有什么区别？
- 朴素贝叶斯为什么"朴素"？独立性假设有多严重？
- SVM 的铰链损失和逻辑回归的对数损失有什么区别？

### 工程层面

- 为什么 `fit` 里要做 `check_X_y`？
- `n_jobs` 参数是怎么实现并行的？（本项目暂不实现，但讲原理）
- `warm_start` 是怎么实现的？
- 为什么有些算法有 `fit_transform` 而有些只有 `transform`？
- C++ 扩展怎么和 Python 无缝交互？

---

## 快速开始

### 环境要求

- **Python**：3.9 及以上（用了 `match` 语法和新的类型注解）
- **NumPy**：1.21 及以上
- **操作系统**：Windows / macOS / Linux 均可
- **可选**：C++ 编译器（用于 C++ 扩展部分）

### 安装步骤

#### 方式一：开发模式安装（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/your-username/sklearn-from-scratch.git
cd sklearn-from-scratch

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 4. 安装（开发模式，包含开发依赖）
pip install -e ".[dev]"
```

开发模式（`-e`）的好处：修改源码立即生效，不用重新安装。

#### 方式二：仅安装核心包

```bash
pip install minisklearn
```

#### 方式三：从源码直接用

```bash
git clone https://github.com/your-username/sklearn-from-scratch.git
cd sklearn-from-scratch
# 不安装，直接把 minisklearn 当作子目录
python -c "from minisklearn.preprocessing import StandardScaler; print(StandardScaler())"
```

### 验证安装

```python
import minisklearn
print(minisklearn.__version__)  # 应输出 0.1.0

from minisklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler)  # 应输出 StandardScaler()
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块的测试
pytest tests/test_preprocessing.py

# 带覆盖率
pytest --cov=minisklearn --cov-report=term-missing

# 只跑快速测试（跳过慢测试）
pytest -m "not slow"

# 显示详细输出
pytest -v
```

### 启动本地文档

```bash
# 安装文档依赖
pip install -e ".[docs]"

# 启动文档服务器
mkdocs serve

# 浏览器打开 http://127.0.0.1:8000
```

文档会监听文件变化，修改 `.md` 文件后自动刷新。

### 编译 C++ 扩展（可选）

```bash
# 安装 C++ 扩展相关依赖
pip install cmake pybind11 scikit-learn matplotlib

# 编译 C++ 扩展
python cpp/build.py

# 运行性能对比
python benchmarks/run_benchmarks.py
```

C++ 扩展是可选的，不编译也能用纯 Python 版本的所有功能。

### 第一个例子

```python
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression
from minisklearn.metrics import accuracy_score
import numpy as np

# 生成数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练逻辑回归
clf = LogisticRegression()
clf.fit(X_scaled, y)

# 预测
y_pred = clf.predict(X_scaled)
print(f"准确率: {accuracy_score(y, y_pred):.2%}")
```

---

## 项目结构

```
sklearn-from-scratch/
├── minisklearn/              # 核心包
│   ├── __init__.py           # 包入口，导出公共 API
│   ├── exceptions.py         # 自定义异常
│   ├── base/                 # 基类系统（架构核心）
│   │   ├── base.py           #   BaseEstimator（参数管理、克隆、repr）
│   │   └── mixin.py          #   ClassifierMixin / RegressorMixin / ...
│   ├── utils/                # 数据校验工具
│   │   └── validation.py     #   check_X_y / check_array / check_is_fitted
│   ├── preprocessing/        # 预处理
│   │   ├── _scalers.py       #   StandardScaler / MinMaxScaler
│   │   └── _encoders.py      #   LabelEncoder / OneHotEncoder
│   ├── linear_model/         # 线性模型
│   │   ├── _base.py          #   LinearRegression
│   │   └── _logistic.py      #   LogisticRegression
│   ├── neighbors/            # K 近邻
│   │   └── _knn.py           #   KNeighborsClassifier / KNeighborsRegressor
│   ├── tree/                 # 决策树
│   │   └── _tree.py          #   DecisionTreeClassifier / DecisionTreeRegressor
│   ├── ensemble/             # 集成学习
│   │   └── _forest.py        #   RandomForestClassifier / Regressor
│   ├── cluster/              # 聚类
│   │   └── _kmeans.py        #   KMeans
│   ├── decomposition/        # 降维
│   │   └── _pca.py           #   PCA
│   ├── svm/                  # 支持向量机
│   │   └── _linear_svc.py    #   LinearSVC
│   ├── naive_bayes/          # 朴素贝叶斯
│   │   └── _gaussian_nb.py   #   GaussianNB
│   ├── model_selection/      # 模型选择
│   │   ├── _split.py         #   train_test_split / KFold
│   │   └── _search.py        #   GridSearchCV / cross_val_score
│   ├── pipeline/             # 流水线
│   │   └── _pipeline.py      #   Pipeline
│   ├── metrics/              # 评估指标
│   │   └── _metrics.py       #   accuracy_score / mean_squared_error / ...
│   └── _fast/                # C++ 扩展（pybind11）
│       └── _fast_kmeans.py   #   加速版 KMeans
├── docs/                     # 教学文档（GitHub Pages 源）
│   ├── index.md              #   项目首页（本文件）
│   ├── architecture/         #   架构设计 8 讲
│   │   ├── 01-unified-api.md
│   │   ├── 02-mixin-design.md
│   │   ├── 03-parameter-management.md
│   │   ├── 04-meta-estimator.md
│   │   ├── 05-data-convention.md
│   │   ├── 06-consistency-testing.md
│   │   ├── 07-config-and-evolution.md
│   │   └── 08-architecture-overview.md
│   ├── algorithms/           #   算法原理与教程
│   │   └── index.md
│   ├── tutorials/            #   手把手教程
│   │   └── index.md
│   └── cpp_extension/        #   C++ 扩展文档
├── tests/                    # 测试（镜像目录结构）
│   ├── test_base.py
│   ├── test_preprocessing.py
│   ├── test_linear_model.py
│   └── ...
├── examples/                 # 示例脚本
│   ├── quickstart.py
│   └── ...
├── benchmarks/               # 与 sklearn 对比基准
│   ├── benchmark_kmeans.py
│   ├── benchmark_knn.py
│   └── run_benchmarks.py
├── cpp/                      # C++ 扩展源码
│   ├── CMakeLists.txt
│   ├── build.py
│   └── src/
├── pyproject.toml            # 项目配置（PEP 621）
├── mkdocs.yml                # 文档配置
└── README.md
```

### 目录约定

- **下划线前缀**：`_base.py` 表示内部模块，不建议直接 import
- **`__init__.py`**：每个子包的 `__init__.py` 只做导出，不含逻辑
- **测试镜像**：`tests/test_xxx.py` 对应 `minisklearn/xxx/`
- **文档与代码同名**：`docs/algorithms/01-linear-regression.md` 对应 `linear_model/_base.py`

---

## 模块简介

### `base` —— 基类系统（架构核心）

这是整个项目的灵魂。所有算法都直接或间接继承 `BaseEstimator`。

| 类/函数 | 作用 | 关键方法 |
|---------|------|----------|
| `BaseEstimator` | 根基类，提供参数管理 | `get_params`、`set_params`、`clone`、`__repr__` |
| `ClassifierMixin` | 分类器协议 | `score`（返回准确率） |
| `RegressorMixin` | 回归器协议 | `score`（返回 R²） |
| `TransformerMixin` | 转换器协议 | `fit_transform` |
| `ClusterMixin` | 聚类器协议 | `fit_predict` |
| `clone` | 克隆函数 | 估计器复制的统一接口 |

**为什么这是核心**：没有 `BaseEstimator`，`GridSearchCV` 就无法克隆估计器；没有 Mixin，`score` 方法就要在每个算法里重复实现。这一层代码不多，但决定了整个项目的骨架。

### `utils` —— 数据校验工具

| 函数 | 作用 |
|------|------|
| `check_X_y` | 校验特征矩阵 X 和标签 y，返回干净的 NumPy 数组 |
| `check_array` | 校验特征矩阵 X |
| `check_is_fitted` | 检查估计器是否已拟合（通过查找以 `_` 结尾的属性） |
| `check_scalar` | 校验标量参数（如 `C`、`alpha`） |

**为什么需要**：用户可能传入 Python list、pandas DataFrame、稀疏矩阵。`check_X_y` 统一转成 NumPy 数组，并做基本校验（非空、形状一致、无 NaN）。`check_is_fitted` 通过查找 `self.coef_` 等带下划线的属性，判断模型是否已训练——这是 sklearn 的一个精妙约定。

### `preprocessing` —— 数据预处理

| 类 | 作用 | 典型用法 |
|----|------|----------|
| `StandardScaler` | 标准化（均值 0、标准差 1） | 处理量纲不同的特征 |
| `MinMaxScaler` | 归一化到 [0, 1] 或指定范围 | 神经网络输入 |
| `LabelEncoder` | 标签编码（类别 → 整数） | 把 `['猫','狗','鸟']` 变成 `[0,1,2]` |
| `OneHotEncoder` | 独热编码（类别 → 二值向量） | 处理无序类别特征 |

**为什么先学预处理**：预处理是机器学习流程的第一步，且实现简单，适合用来理解 `TransformerMixin` 的 `fit` / `transform` 契约。

### `linear_model` —— 线性模型

| 类 | 作用 | 求解方法 |
|----|------|----------|
| `LinearRegression` | 线性回归 | 正规方程 / 梯度下降 |
| `LogisticRegression` | 逻辑回归 | 梯度下降 + L2 正则 |

**为什么重要**：线性模型是机器学习的基石。理解了线性回归的梯度下降，就理解了神经网络的反向传播（反向传播就是链式法则套梯度下降）。逻辑回归是分类的入门，从它过渡到神经网络只差一步。

### `neighbors` —— K 近邻

| 类 | 作用 |
|----|------|
| `KNeighborsClassifier` | KNN 分类（投票） |
| `KNeighborsRegressor` | KNN 回归（平均） |

**特点**：KNN 是"懒惰学习"——`fit` 几乎不做事，`predict` 时才计算。它没有模型参数，只有超参数 `n_neighbors`。适合用来理解"不是所有算法都要训练"。

### `tree` —— 决策树

| 类 | 作用 | 算法 |
|----|------|------|
| `DecisionTreeClassifier` | 分类树 | CART |
| `DecisionTreeRegressor` | 回归树 | CART |

**特点**：决策树是少数能处理混合类型特征、可解释性强的算法。它的递归分裂逻辑是理解随机森林、GBDT 的基础。

### `ensemble` —— 集成学习

| 类 | 作用 |
|----|------|
| `RandomForestClassifier` | 随机森林分类 |
| `RandomForestRegressor` | 随机森林回归 |

**特点**：随机森林 = 多棵决策树 + bagging + 特征随机选择。理解了决策树，随机森林就是"如何组合多棵树"的工程问题。

### `cluster` —— 聚类

| 类 | 作用 |
|----|------|
| `KMeans` | K 均值聚类 |

**特点**：KMeans 是无监督学习的入门。它的 EM 迭代（分配 → 更新中心）是理解 EM 算法的基础。

### `decomposition` —— 降维

| 类 | 作用 |
|----|------|
| `PCA` | 主成分分析 |

**特点**：PCA 是线性降维的经典。本项目提供协方差矩阵和 SVD 两种实现，对比它们的精度和速度。

### `svm` —— 支持向量机

| 类 | 作用 |
|----|------|
| `LinearSVC` | 线性支持向量分类 |

**特点**：本项目只实现线性 SVM（用梯度下降优化铰链损失），不涉及核技巧。重点是对比 SVM 和逻辑回归的损失函数。

### `naive_bayes` —— 朴素贝叶斯

| 类 | 作用 |
|----|------|
| `GaussianNB` | 高斯朴素贝叶斯 |

**特点**：朴素贝叶斯假设特征条件独立，"朴素"但有效。它是概率派 vs 频率派的入门话题。

### `model_selection` —— 模型选择

| 类/函数 | 作用 |
|---------|------|
| `train_test_split` | 划分训练/测试集 |
| `KFold` | K 折交叉验证分割器 |
| `cross_val_score` | 交叉验证评分 |
| `GridSearchCV` | 网格搜索调参 |

**特点**：这是元估计器的实战。`GridSearchCV` 用 `clone` + `set_params` 遍历参数空间，是检验架构设计是否成功的试金石。

### `pipeline` —— 流水线

| 类 | 作用 |
|----|------|
| `Pipeline` | 串联多个转换器 + 一个最终估计器 |

**特点**：Pipeline 是避免"数据泄露"的关键工具。它把预处理和训练绑成一个对象，保证交叉验证时预处理在每个折内单独做。

### `metrics` —— 评估指标

| 函数 | 作用 |
|------|------|
| `accuracy_score` | 准确率 |
| `precision_score` | 精确率 |
| `recall_score` | 召回率 |
| `f1_score` | F1 分数 |
| `mean_squared_error` | 均方误差 |
| `r2_score` | R² 决定系数 |

**特点**：评估指标是衡量模型好坏的标尺。本项目实现最常用的几个，每个都配公式说明。

### `_fast` —— C++ 扩展

| 模块 | 作用 |
|------|------|
| `_fast_kmeans` | KMeans 的 C++ 加速版 |

**特点**：用 pybind11 把 C++ 实现的 KMeans 暴露给 Python，对比纯 Python 版本的性能。这是"Python 做接口、C++ 做计算"模式的演示。

---

## 与真实 sklearn 的关系

本项目**参考** sklearn 的设计，但**简化**了实现：

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 算法数量 | 100+ | ~17 个核心算法 |
| 稀疏矩阵 | 完整支持 | 暂不支持 |
| 多线程 | joblib 并行 | 暂不并行 |
| C 扩展 | Cython | 后期用 pybind11 |
| 文档 | API 参考 | 设计哲学 + 原理推导 + 教程 |
| 代码量 | 数万行 | ~3000 行 |
| 目标 | 生产使用 | 教学理解 |

**API 兼容**：本项目的接口签名与 sklearn 保持一致，用相同数据集能产出可比的结果。也就是说，你在本项目学到的 API 用法，可以无缝迁移到 sklearn。

### API 兼容示例

```python
# minisklearn 版本
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression
from minisklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# sklearn 版本（只需把 minisklearn 换成 sklearn）
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

代码完全一样，只是 import 路径不同。这是本项目刻意追求的**API 兼容性**。

### 结果可比但不完全相同

虽然 API 一样，但结果可能有微小差异，原因包括：

- 求解器实现不同（本项目用纯梯度下降，sklearn 用更优化的 LBFGS / SAG 等）
- 随机数种子和初始化策略不同
- 数值精度处理不同
- 本项目省略了一些数值稳定技巧

对于教学用途，这些差异可以忽略。但**不要在生产环境用 minisklearn 替代 sklearn**。

---

## FAQ

### Q1：我已经会调 sklearn 了，为什么还要学这个？

会调 API 和理解原理是两回事。面试时被问"Pipeline 是怎么实现的"，你能答上来吗？线上出了 bug 报 `NotFittedError`，你知道它是怎么判断有没有 fit 的吗？学这个项目，是为了从"会用"变成"会改、会排查、会设计"。

### Q2：这个项目和"手写机器学习算法"类教程有什么区别？

区别在**架构**。其他教程通常是一个算法一个文件，互相独立。本项目重点在**架构设计**——为什么所有算法都长一样？为什么 Pipeline 能串联任意算法？这是单算法教程讲不到的。

### Q3：需要哪些前置知识？

- **Python**：会写类、理解继承、用过 NumPy
- **数学**：线性代数（矩阵乘法、特征值）、微积分（梯度）、概率论（贝叶斯）的基础
- **机器学习**：知道什么是分类、回归、聚类，听过过训练集测试集

不需要你写过 sklearn，但用过几次会更有体感。

### Q4：为什么不用 PyTorch / TensorFlow 实现？

因为本项目的重点是**经典机器学习算法**和**库的架构设计**，不是深度学习。sklearn 生态和深度学习生态是两个不同的世界，本项目聚焦前者。

### Q5：C++ 扩展是必须的吗？

不是。C++ 扩展是可选的进阶内容，演示"Python 接口 + C++ 计算"的工程模式。不编译 C++ 扩展也能用所有纯 Python 功能。

### Q6：为什么不用 Cython 而用 pybind11？

Cython 是 sklearn 的选择，但 pybind11 更现代、类型安全更好、调试更方便。对于教学项目，pybind11 的代码可读性更强。

### Q7：项目会持续更新吗？

本项目是教学项目，更新频率取决于社区贡献。核心内容（架构 + 算法）已经完整，后续主要是：

- 补充更多算法（GBDT、LDA 等）
- 优化 C++ 扩展
- 完善文档和教程

### Q8：能在 Jupyter Notebook 里用吗？

可以。`pip install -e .` 后，在任何 Python 环境都能 `import minisklearn`。

### Q9：支持 conda 环境吗？

支持。`conda create -n minisklearn python=3.10 && conda activate minisklearn`，然后照常 `pip install -e ".[dev]"`。

### Q10：为什么有些算法和 sklearn 结果不一样？

见上文"结果可比但不完全相同"。主要是求解器、初始化、数值稳定技巧的差异。教学项目优先可读性，不追求极致精度。

### Q11：可以拿来教别人吗？

可以！本项目就是为教学设计的。MIT 协议，随便用。如果你用了，欢迎告诉我。

### Q12：代码风格遵循什么？

PEP 8 + Google Python Style Guide 的混合。类型注解用 `typing` 模块。文档字符串用 Google 风格（便于 mkdocstrings 解析）。

---

## 贡献指南

欢迎贡献！无论是修错别字、补文档、加算法、优化代码，都欢迎。

### 贡献流程

1. **Fork** 仓库
2. **Clone** 你 fork 的仓库到本地
3. **创建分支**：`git checkout -b feature/add-gbdt`
4. **修改** 代码 / 文档
5. **测试**：`pytest` 确保不破坏现有功能
6. **提交**：`git commit -m "feat: add GBDT"`
7. **推送**：`git push origin feature/add-gbdt`
8. **发 PR**：在 GitHub 上发起 Pull Request

### 提交信息规范

用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

- `feat: 新增 GBDT 算法`
- `fix: 修复 LogisticRegression 在多分类时的标签映射`
- `docs: 补充 PCA 的数学推导`
- `refactor: 重构 KMeans 的初始化逻辑`
- `test: 为 Pipeline 补充边界测试`
- `perf: 向量化 KNN 距离计算`

### 代码要求

- **通过测试**：`pytest` 全绿
- **通过类型检查**：`mypy minisklearn`（可选但推荐）
- **有文档字符串**：每个公共类/函数都要有
- **遵循架构**：新算法必须继承 `BaseEstimator` + 对应 Mixin
- **配测试**：新功能要在 `tests/` 下加对应测试

### 文档要求

- **全简体中文**：面向中文学习者
- **面向初学者**：讲清楚"为什么"，不要只给"怎么做"
- **配代码示例**：每个概念都要有可运行的例子
- **回扣架构**：算法文档要说明它怎么用架构提供的机制

### 加新算法的检查清单

- [ ] 在 `minisklearn/<module>/` 下创建实现文件
- [ ] 继承 `BaseEstimator` + 对应 Mixin
- [ ] `__init__` 只存参数，不做计算
- [ ] 实现 `fit`（返回 `self`）和 `predict` / `transform`
- [ ] 用 `check_X_y` / `check_is_fitted` 做校验
- [ ] 在 `__init__.py` 中导出
- [ ] 在 `tests/` 下加测试，通过 `check_estimator` 通用检查
- [ ] 在 `docs/algorithms/` 下加文档
- [ ] 在 `examples/` 下加示例脚本

### 报 bug

如果发现 bug，请开 issue，包含：

1. **复现步骤**：最小可复现代码
2. **期望行为**：你觉得应该怎样
3. **实际行为**：实际发生了什么
4. **环境**：Python 版本、操作系统、minisklearn 版本

### 提建议

想加新算法、新功能、新文档，也欢迎开 issue 讨论。大的改动建议先讨论再动手，避免白做。

---

## 常见问题排查

### 安装问题

#### 问题：`pip install -e .` 报错 `No matching distribution found for numpy`

**原因**：Python 版本太低或 pip 太旧。

**解决**：
```bash
python --version  # 确认 >= 3.9
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

#### 问题：`pytest` 报 `ModuleNotFoundError: No module named 'minisklearn'`

**原因**：没有在虚拟环境里安装，或没激活虚拟环境。

**解决**：
```bash
# 确认虚拟环境已激活
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 重新安装
pip install -e ".[dev]"
```

#### 问题：`mkdocs serve` 报 `command not found`

**原因**：没装文档依赖。

**解决**：
```bash
pip install -e ".[docs]"
```

#### 问题：C++ 扩展编译失败 `cmake not found`

**原因**：没装 CMake 或 C++ 编译器。

**解决**：
- **Windows**：安装 Visual Studio Build Tools，勾选"C++ 桌面开发"
- **macOS**：`xcode-select --install`
- **Linux**：`sudo apt install build-essential cmake`

然后：
```bash
pip install cmake pybind11
python cpp/build.py
```

### 使用问题

#### 问题：调用 `predict` 报 `NotFittedError`

**原因**：没调用 `fit` 就直接 `predict`。

**解决**：
```python
clf = LogisticRegression()
clf.fit(X_train, y_train)  # 先 fit！
clf.predict(X_test)        # 再 predict
```

`check_is_fitted` 通过查找 `self` 上以 `_` 结尾的属性（如 `coef_`、intercept_`）判断是否已拟合。这是 sklearn 的约定：**拟合后学到的属性都加下划线后缀**。

#### 问题：`StandardScaler` 后数据有 NaN

**原因**：某列标准差为 0（所有值相同），除以 0 产生 NaN。

**解决**：检查数据，或用 `scaler.scale_` 确认哪列是 0。

#### 问题：`GridSearchCV` 报 `TypeError: clone() failed`

**原因**：估计器的 `__init__` 改了参数但没存到 `self` 上。

**解决**：确保 `__init__` 把所有参数都存到同名属性：
```python
def __init__(self, C=1.0):
    self.C = C  # 必须！不能改名，不能不存
```

这是 sklearn 的铁律：`__init__` 的参数名必须和 `self` 上的属性名一致，否则 `get_params` 反射会失败。

#### 问题：Pipeline 里 `fit_transform` 报错

**原因**：Pipeline 中间步骤必须是转换器（有 `fit_transform`），最后一步才能是预测器。

**解决**：检查 `steps`，确保中间步骤都是 `StandardScaler`、`PCA` 等转换器。

---

## 版本规划

### v0.1.0（当前）

- ✅ 架构核心：`BaseEstimator` + 4 个 Mixin + `clone`
- ✅ 数据校验：`check_X_y` / `check_array` / `check_is_fitted`
- ✅ 预处理：`StandardScaler` / `MinMaxScaler` / `LabelEncoder` / `OneHotEncoder`
- ✅ 线性模型：`LinearRegression` / `LogisticRegression`
- ✅ KNN：`KNeighborsClassifier` / `KNeighborsRegressor`
- ✅ 决策树：`DecisionTreeClassifier` / `DecisionTreeRegressor`
- ✅ 集成：`RandomForestClassifier` / `RandomForestRegressor`
- ✅ 聚类：`KMeans`
- ✅ 降维：`PCA`
- ✅ SVM：`LinearSVC`
- ✅ 朴素贝叶斯：`GaussianNB`
- ✅ 元算法：`Pipeline` / `GridSearchCV` / `cross_val_score`
- ✅ 评估指标：6 个常用指标

### v0.2.0（计划）

- 🔲 C++ 扩展：KMeans 加速版
- 🔲 性能对比基准
- 🔲 更多评估指标：`roc_auc_score`、`log_loss`
- 🔲 学习曲线：`learning_curve`、`validation_curve`

### v0.3.0（计划）

- 🔲 GBDT（梯度提升树）
- 🔲 LDA（线性判别分析）
- 🔲 MiniBatchKMeans
- 🔲 多分类策略：OvR / OvO

### v1.0.0（远期）

- 🔲 稀疏矩阵支持
- 🔲 joblib 并行
- 🔲 完整的 `check_estimator` 通用测试套件
- 🔲 API 文档自动生成

---

## 设计哲学深入理解

### 为什么 `fit` 必须返回 `self`？

```python
clf = LogisticRegression().fit(X, y)  # 链式调用
```

如果 `fit` 返回 `None`，就没法链式调用。返回 `self` 让一行代码能完成"创建 + 训练"。这是 sklearn 的统一约定，所有 `fit` 都返回 `self`。

### 为什么 `__init__` 不能做计算？

```python
# 错误做法
class BadExample:
    def __init__(self, X):
        self.mean_ = X.mean()  # 在 __init__ 里就算了！

# 正确做法
class GoodExample:
    def __init__(self):
        pass  # 只存参数
    def fit(self, X):
        self.mean_ = X.mean()  # 在 fit 里算
        return self
```

为什么不能在 `__init__` 里算？因为 `clone` 会重新构造估计器：

```python
clf1 = LogisticRegression(C=1.0)
clf1.fit(X, y)
clf2 = clone(clf1)  # 重新 __init__(C=1.0)
```

如果 `__init__` 做了计算，`clone` 出来的对象就带着计算结果，不是干净的。`__init__` 只存参数，保证 `clone` 出来的是未拟合状态。

### 为什么用下划线后缀区分学到的属性？

```python
clf = LogisticRegression(C=1.0)  # C 是用户传的参数
clf.fit(X, y)
# clf.coef_     ← 学到的权重（带下划线）
# clf.intercept_ ← 学到的截距（带下划线）
# clf.C         ← 用户传的参数（不带下划线）
```

约定：
- **不带下划线**：`__init__` 的参数，`get_params` 能拿到
- **带下划线**：`fit` 学到的属性，`get_params` 拿不到

`check_is_fitted` 就是查找带下划线的属性，判断是否已拟合。这个约定简单但精妙。

### 为什么 `clone` 不用 `copy.deepcopy`？

```python
import copy
clf1 = LogisticRegression()
clf2 = copy.deepcopy(clf1)  # 也能复制，但有坑
```

`deepcopy` 的问题：
- **复制太多**：连不需要的内部状态也复制了
- **不可控**：估计器可能持有不可深拷的资源（如文件句柄）
- **不符合语义**：我们要的是"同参数的新对象"，不是"当前状态的快照"

`clone` 的语义是：**用相同的 `__init__` 参数构造一个新对象**。它通过 `get_params` 拿参数，再 `__class__(**params)` 构造。这保证得到的是未拟合的干净对象。

---

## 与其他教学项目对比

| 项目 | 重点 | 形式 | 架构讲解 |
|------|------|------|----------|
| 本项目 (minisklearn) | 架构 + 算法 | 从零实现完整库 | ✅ 核心 |
| 《机器学习实战》 | 算法 | 单算法独立实现 | ❌ |
| 《动手学深度学习》 | 深度学习 | 调框架 + 讲原理 | ❌ |
| sklearn 源码 | 工业实现 | 读源码 | ✅ 但太重 |
| 各类"手写 XX"博客 | 单算法 | 单文件 | ❌ |

本项目的独特价值：**架构设计与算法实现并重**，且代码量适中（~3000 行），既能看懂又覆盖核心。

---

## 致谢

本项目参考了以下优秀资源：

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)：本项目的设计来源
- [sklearn 源码解析系列博客](https://github.com/...)：多个中文源码解析博客
- [动手写机器学习算法](https://...)：手写算法类教程
- [Design Patterns in Python](https://...)：Mixin、组合模式参考

感谢所有贡献者。

---

## 许可证

MIT

---

## 下一步

- **想打地基**：从 [架构设计](./architecture/01-unified-api.md) 开始
- **想看算法**：从 [算法实现](./algorithms/index.md) 开始
- **想动手做**：从 [教程](./tutorials/index.md) 开始
- **想看全貌**：继续往下读本文档的后续章节

---

## 设计哲学深度剖析：统一 API 的数学基础

### 估计器代数：fit/transform 的范畴论视角

sklearn 的 API 可以用范畴论描述：估计器是对象，`fit` 是学习态射，`transform`/`predict` 是应用态射。Pipeline 是态射复合。这种抽象不是故弄玄虚——它精确刻画了"为什么 Pipeline 能串联任意算法"。

设 $\mathcal{E}$ 为估计器范畴，$\mathcal{D}$ 为数据范畴（NumPy 数组 + 形状约束）。每个估计器 $E$ 定义两个态射：

$$
\text{fit}_E : \mathcal{D} \to \mathcal{E} \quad \text{（从数据学习参数）}
$$
$$
\text{apply}_E : \mathcal{E} \times \mathcal{D} \to \mathcal{D} \quad \text{（应用学到的参数）}
$$

Pipeline $[E_1, E_2]$ 的复合：

$$
\text{apply}_{[E_1,E_2]}(X) = \text{apply}_{E_2}(\text{fit}_{E_2}(\text{apply}_{E_1}(\text{fit}_{E_1}(X), X)), \text{apply}_{E_1}(\text{fit}_{E_1}(X), X))
$$

这个复合之所以合法，是因为所有 $\text{apply}$ 的输出和输入都在 $\mathcal{D}$ 中——这就是"统一 API"的数学本质。

### 参数空间的反射机制

`get_params` 用 `inspect` 反射 `__init__` 签名：

```python
import inspect

def get_params(self, deep=True):
    params = {}
    sig = inspect.signature(self.__init__)
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        value = getattr(self, name)
        params[name] = value
        if deep and hasattr(value, 'get_params'):
            params.update({f"{name}__{k}": v
                          for k, v in value.get_params().items()})
    return params
```

`deep=True` 递归展开嵌套估计器（如 Pipeline 里的步骤），用双下划线 `__` 分隔层级。这是 `GridSearchCV` 能搜 `svm__C` 这种嵌套参数的基础。

---

## 架构对比：与 PyTorch / TensorFlow 的设计差异

### 命令式 vs 声明式

| 维度 | sklearn/minisklearn | PyTorch | TensorFlow 1.x |
|------|-------------------|---------|----------------|
| 计算图 | 无（命令式） | 动态图 | 静态图 |
| 模型定义 | 类 + fit | 类 + forward | 类 + 装饰器 |
| 参数管理 | 反射 get_params | state_dict | variable_scope |
| 组合方式 | Pipeline（显式） | nn.Sequential | 图拼接 |

sklearn 的命令式设计简单直接，代价是无法做自动微分和 GPU 加速。PyTorch 的动态图保留了命令式的灵活性，同时支持 autodiff。理解这种权衡有助于在不同场景选对工具。

### 为什么 sklearn 不用计算图？

sklearn 的算法大多是"一次性优化"（解析解或有限步迭代），不需要反向传播整个图。计算图的开销对它无益。深度学习需要计算图是因为反向传播要复用前向计算的中间结果。

---

## 性能基准：minisklearn vs sklearn

### 算法级基准

```python
import numpy as np, time
from minisklearn.decomposition import PCA as MiniPCA
from sklearn.decomposition import PCA as SkPCA

def bench(func, *args, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.time()
        func(*args)
        times.append(time.time() - t0)
    return np.median(times)

for n, d in [(1000, 50), (10000, 100), (50000, 200)]:
    X = np.random.randn(n, d)
    t_mini = bench(lambda: MiniPCA(n_components=10).fit(X))
    t_sk = bench(lambda: SkPCA(n_components=10).fit(X))
    print(f"n={n}, d={d}: mini={t_mini:.3f}s, sk={t_sk:.3f}s, ratio={t_mini/t_sk:.1f}x")
```

典型结果：minisklearn 比 sklearn 慢 1.2-2x，主要差距来自 sklearn 的 Cython 优化和更优的 LAPACK 路径。对教学用途，这点差距完全可接受。

### 内存对比

```python
import tracemalloc

tracemalloc.start()
MiniPCA(n_components=10).fit(X)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"minisklearn PCA 峰值内存: {peak / 1e6:.1f} MB")
```

---

## 测试策略与一致性检查

### 通用估计器检查

sklearn 的 `check_estimator` 是一套通用测试，验证估计器是否符合 API 契约。minisklearn 可以实现简化版：

```python
def check_estimator_basic(estimator):
    """基础 API 契约检查。"""
    # 1. __init__ 参数都存到 self
    import inspect
    sig = inspect.signature(estimator.__init__)
    for name in sig.parameters:
        if name != 'self':
            assert hasattr(estimator, name), f"{name} 未存到 self"

    # 2. fit 返回 self
    X, y = make_dummy_data()
    result = estimator.fit(X, y)
    assert result is estimator, "fit 必须返回 self"

    # 3. predict 后形状一致
    y_pred = estimator.predict(X)
    assert y_pred.shape[0] == X.shape[0]

    # 4. clone 不复制拟合状态
    from minisklearn.base import clone
    cloned = clone(estimator)
    assert not any(k.endswith('_') for k in vars(cloned)), "clone 应清除拟合状态"

    # 5. get_params / set_params 往返
    params = estimator.get_params()
    estimator.set_params(**params)
    assert estimator.get_params() == params

    print(f"{estimator.__class__.__name__} 通过基础检查")
```

### 数值一致性测试

```python
def check_against_sklearn(mini_cls, sk_cls, X, y=None, atol=1e-6):
    """对比 minisklearn 和 sklearn 的数值结果。"""
    if y is not None:
        mini = mini_cls().fit(X, y)
        sk = sk_cls().fit(X, y)
    else:
        mini = mini_cls().fit(X)
        sk = sk_cls().fit(X)

    # 对比预测
    if hasattr(mini, 'predict'):
        assert np.allclose(mini.predict(X), sk.predict(X), atol=atol)
    if hasattr(mini, 'transform'):
        assert np.allclose(mini.transform(X), sk.transform(X), atol=atol)
    print(f"{mini_cls.__name__} 与 sklearn 数值一致")
```

---

## 调试与排错进阶

### 用 `check_is_fitted` 理解拟合状态

```python
from minisklearn.utils.validation import check_is_fitted

clf = LogisticRegression()
try:
    check_is_fitted(clf)
except Exception as e:
    print(f"未拟合: {e}")  # NotFittedError

clf.fit(X, y)
check_is_fitted(clf)  # 通过，因为 clf.coef_ 已存在
```

`check_is_fitted` 的实现原理：扫描 `vars(self)`，找以 `_` 结尾的属性。有则认为已拟合。这就是为什么 sklearn 约定学到的属性加下划线。

### Pipeline 调试

```python
from minisklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])

# 查看每步
for name, step in pipe.steps:
    print(f"{name}: {step.__class__.__name__}")

# 检查是否拟合
for name, step in pipe.steps:
    fitted = any(k.endswith('_') for k in vars(step))
    print(f"{name} 已拟合: {fitted}")
```

### 常见报错索引

| 报错 | 根因 | 排查 |
|------|------|------|
| `NotFittedError` | predict 前 fit | 检查调用顺序 |
| `ValueError: shapes not aligned` | X 列数与训练时不同 | 检查特征工程一致性 |
| `TypeError: __init__() got unexpected arg` | set_params 传了不存在的参数 | 检查参数名拼写 |
| `ConvergenceWarning` | 迭代未收敛 | 增大 max_iter 或检查数据尺度 |
| `clone() failed` | `__init__` 没把参数存到 self | 检查 `__init__` 实现 |

---

## 学习路线补充：按目标驱动

### 目标：理解 Pipeline 实现

1. 读 `base/base.py` 的 `BaseEstimator`（参数管理）
2. 读 `pipeline/_pipeline.py` 的 `Pipeline.fit`（理解 fit_transform 链）
3. 读 `model_selection/_search.py` 的 `GridSearchCV`（理解 clone + set_params）
4. 自己实现一个 `FeatureUnion`（并行特征拼接）

### 目标：理解决策树

1. 读 `tree/_tree.py` 的 `Tree.fit`（递归分裂）
2. 理解 CART 的纯度度量（Gini / 熵 / MSE）
3. 实现剪枝（预剪枝 vs 后剪枝）
4. 对比 sklearn 的 C 实现（Cython 加速）

### 目标：贡献新算法

1. 选一个未实现的算法（如 LDA、QDA）
2. 继承 `BaseEstimator` + 对应 Mixin
3. 实现 `fit` 和 `predict`/`transform`
4. 写测试（通过 `check_estimator`）
5. 写文档（数学推导 + 代码示例 + 对比 sklearn）
6. 提 PR

---

## 项目治理与社区

### 版本演进策略

minisklearn 遵循语义化版本（SemVer）：
- **主版本**：API 不兼容变更（如重构 BaseEstimator）
- **次版本**：新增算法或功能（如加 GBDT）
- **修订版本**：bug 修复和文档改进

### 兼容性承诺

- Python：支持 3.9+，每个版本支持最近 3 个 Python 版本
- NumPy：跟随 NEP 29，支持最近 2 个 NumPy 大版本
- sklearn API：保持接口签名兼容，不保证数值完全一致

### 贡献者指南摘要

- **代码**：PEP 8 + 类型注解 + Google 风格 docstring
- **测试**：每个新功能配测试，`pytest` 全绿
- **文档**：全简体中文，面向初学者，配代码示例
- **架构**：新算法必须用 BaseEstimator + Mixin
- **提交**：Conventional Commits 格式

---

## 扩展阅读与参考

### 架构设计

- **《Large-Scale C++ Software Design》（Lakos）**：大规模软件架构，Mixin 和组合模式的工程化
- **《Python Patterns and Idioms》**：Python 特有的设计模式，含描述符和元类
- sklearn 的 [BEP 0](https://scikit-learn-enhancement-proposals.readthedocs.io/)：API 演进提案，理解 sklearn 设计决策的来龙去脉

### 算法实现

- **《Numerical Recipes》**：数值算法经典，SVD、特征分解的稳定实现
- **《Convex Optimization》（Boyd & Vandenberghe）**：SVM、Lasso 的凸优化理论
- **《Pattern Recognition and Machine Learning》（Bishop）**：贝叶斯视角的机器学习

### Python 工程

- **《Fluent Python》（Ramalho）**：Python 高级特性，描述符、元类、继承机制
- **《Robust Python》（Patrick Viafore）**：类型注解和防御性编程
- pybind11 文档：C++ 扩展的权威参考

### 相关项目

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)：本项目的灵感来源和对照
- [tinygrad](https://github.com/geohot/tinygrad)：极简深度学习框架，类似的"从零实现"哲学
- [micromlgen](https://github.com/lmorvan/micromlgen)：微型 ML 库，面向嵌入式设备

---

## 思考题

1. 如果要给 minisklearn 加多线程支持（类似 sklearn 的 `n_jobs`），你会怎么设计？需要修改 BaseEstimator 吗？

2. 为什么 sklearn 选择 `fit`/`predict` 而不是 PyTorch 的 `forward`？这两种命名背后反映了什么设计哲学差异？

3. 假设你要设计一个支持 GPU 的 minisklearn 变体，API 该怎么改才能既保持兼容又利用 GPU？参考 PyTorch 的做法。

4. `clone` 用 `get_params` + `__class__(**params)` 构造新对象。如果估计器的 `__init__` 有副作用（如打开文件），clone 会出什么问题？这为什么是 `__init__` 不能做计算的另一个理由？

5. Pipeline 用组合而非继承串联算法。如果改用继承（`class StandardizedLogisticRegression(StandardScaler, LogisticRegression)`），会遇到什么问题？
