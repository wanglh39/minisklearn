# minisklearn —— 从零实现 sklearn

> 理解设计哲学，拆解底层原理，从零手写一个 mini scikit-learn。

本项目不是 sklearn 的替代品，而是一座**拆解机**——把 sklearn 的架构设计和算法实现拆开给你看，讲清楚每一个设计决策的"为什么"。

## 为什么做这个项目？

sklearn 是 Python 机器学习生态的基石。它的伟大不在于算法多（其实很多算法实现得不如专用库快），而在于**架构设计**：

- **统一 API**：所有算法遵循 `fit` / `predict` / `transform` 契约，学一个就会用上百个
- **Mixin 多继承**：用极简的基类组合出分类器、回归器、转换器等不同身份
- **元估计器**：`Pipeline`、`GridSearchCV` 用组合而非继承串联一切
- **参数管理**：`get_params` / `set_params` / `clone` 的反射机制支撑了整个生态

这些设计思想值得每一个 Python 工程师学习。本项目通过从零实现它们，让你真正理解而不是停留在"会用 API"的层面。

## 学习路线

```
架构设计（地基）          算法实现（砌墙）           进阶（装修）
┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│ 统一 API 契约    │    │ preprocessing    │    │ 元估计器      │
│ Mixin 架构      │ →  │ linear_model     │ →  │ Pipeline     │
│ 参数管理        │    │ neighbors (KNN)  │    │ GridSearchCV │
│ 数据约定与校验   │    │ tree / ensemble  │    │ C++ 性能对比  │
│ 一致性测试      │    │ cluster / decomposition │              │
└─────────────────┘    └──────────────────┘    └──────────────┘
```

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行测试
pytest

# 启动文档
pip install -e ".[docs]"
mkdocs serve

# 编译 C++ 扩展 + 性能对比
pip install cmake pybind11 scikit-learn matplotlib
python cpp/build.py
python benchmarks/run_benchmarks.py
```

## 项目结构

```
minisklearn/          # 核心包
  base/               # 基类系统（架构核心）
  utils/              # 数据校验工具
  preprocessing/      # 预处理
  linear_model/       # 线性模型
  ...
docs/                 # 教学文档（GitHub Pages）
  architecture/       # 架构设计 8 讲
  algorithms/         # 算法原理与教程
tests/                # 测试（镜像目录结构）
```

## 文档

完整文档托管在 GitHub Pages：[在线阅读](https://wanglh39.github.io/minisklearn/)

## 许可证

MIT