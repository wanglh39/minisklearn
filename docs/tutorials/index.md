# 教程

> 手把手教程，从零开始使用 minisklearn。

本章节是 minisklearn 的实战教程集合。与架构文档讲"为什么"不同，教程聚焦"怎么做"——每一步都给出可运行代码，跑通后再讲原理。

## 教程列表

本页包含 17 个完整教程，从入门到高级，所有代码均可直接运行。

### 入门

1. [教程一：从零开始的第一步](#教程一从零开始的第一步) —— 环境配置、第一个线性回归和逻辑回归模型
2. [教程二：数据预处理实战](#教程二数据预处理实战) —— StandardScaler / MinMaxScaler / 编码器
3. [教程三：每个算法的使用示例](#教程三每个算法的使用示例) —— 12 个算法的快速用法

### 进阶

4. [教程四：Pipeline + GridSearchCV 实战](#教程四pipeline--gridsearchcv-实战) —— 标准工作流
5. [教程五：交叉验证评估](#教程五交叉验证评估) —— KFold 与 cross_val_score
6. [教程十：从 minisklearn 迁移到 sklearn](#教程十从-minisklearn-迁移到-sklearn) —— API 兼容性

### 高级

7. [教程六：C++ 扩展编译和使用](#教程六c-扩展编译和使用) —— pybind11 加速
8. [教程七：性能对比教程](#教程七性能对比教程) —— 与 sklearn 对比基准
9. [教程十一：深入技术分析](#教程十一深入技术分析理解-fitpredict-背后的机制) —— fit/predict 背后的机制
10. [教程十二：对比实验](#教程十二对比实验亲手比较算法性能) —— 亲手比较算法性能
11. [教程十三：参数调优实战指南](#教程十三参数调优实战指南) —— 各算法关键参数
12. [教程十四：常见错误与调试技巧大全](#教程十四常见错误与调试技巧大全) —— 排错参考
13. [教程十五：实际应用场景实战](#教程十五实际应用场景实战) —— 真实场景
14. [教程十六：思考题与练习](#教程十六思考题与练习) —— 巩固练习
15. [教程十七：扩展阅读与学习路径](#教程十七扩展阅读与学习路径) —— 进阶资源

---

## 教程一：从零开始的第一步

本教程面向完全没用过 minisklearn 的同学，从安装到跑通第一个模型，全程不跳步。

### 1.1 环境准备

#### 1.1.1 确认 Python 版本

minisklearn 需要 Python 3.9 及以上。打开终端（Windows 用 PowerShell，macOS / Linux 用 Terminal），输入：

```bash
python --version
```

应该输出类似 `Python 3.10.8`。如果版本低于 3.9，请先升级 Python：

- **Windows**：从 [python.org](https://python.org) 下载安装
- **macOS**：`brew install python@3.10`
- **Linux**：`sudo apt install python3.10`

#### 1.1.2 创建虚拟环境

虚拟环境能隔离项目依赖，避免污染全局环境。

```bash
# 进入项目目录
cd path/to/sklearn-from-scratch

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

激活后，命令行前面会出现 `(.venv)` 标识。

#### 1.1.3 安装 minisklearn

```bash
# 开发模式安装（推荐，修改源码立即生效）
pip install -e ".[dev]"

# 验证安装
python -c "import minisklearn; print(minisklearn.__version__)"
# 应输出: 0.1.0
```

#### 1.1.4 安装文档依赖（可选）

```bash
pip install -e ".[docs]"
mkdocs serve
# 浏览器打开 http://127.0.0.1:8000
```

#### 1.1.5 安装性能对比依赖（可选）

```bash
pip install -e ".[benchmark]"
# 包含 scikit-learn（用于对比）和 matplotlib（用于画图）
```

### 1.2 第一个模型：线性回归

#### 1.2.1 准备数据

```python
import numpy as np

# 生成模拟数据：y = 2x + 1 + 噪声
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 个样本，1 个特征
y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5  # 加点噪声

print(f"X 形状: {X.shape}")  # (100, 1)
print(f"y 形状: {y.shape}")  # (100,)
print(f"前 5 个 X: {X[:5].ravel()}")
print(f"前 5 个 y: {y[:5]}")
```

#### 1.2.2 训练模型

```python
from minisklearn.linear_model import LinearRegression

# 创建模型
model = LinearRegression()

# 训练
model.fit(X, y)

# 查看学到的参数
print(f"权重 (coef_): {model.coef_}")      # 应接近 2
print(f"截距 (intercept_): {model.intercept_}")  # 应接近 1
```

注意命名约定：
- `coef_`、`intercept_` 带**下划线后缀**，表示是 `fit` 学到的
- 如果没 `fit` 就访问 `coef_`，会报 `NotFittedError`

#### 1.2.3 预测

```python
# 预测新数据
X_new = np.array([[5.0], [10.0]])
y_pred = model.predict(X_new)
print(f"预测: {y_pred}")  # 应接近 [11, 21]

# 评估
from minisklearn.metrics import r2_score, mean_squared_error
y_train_pred = model.predict(X)
print(f"R²: {r2_score(y, y_train_pred):.4f}")        # 越接近 1 越好
print(f"MSE: {mean_squared_error(y, y_train_pred):.4f}")  # 越小越好
```

#### 1.2.4 用 score 方法

```python
# RegressorMixin 提供的 score 方法，返回 R²
print(f"score: {model.score(X, y):.4f}")
```

`score` 方法是 `RegressorMixin` 给的，不用自己写。这就是 Mixin 的好处。

### 1.3 第二个模型：逻辑回归分类

#### 1.3.1 准备分类数据

```python
from minisklearn.model_selection import train_test_split

# 生成两类数据
np.random.seed(42)
X_class0 = np.random.randn(100, 2) + np.array([-2, -2])
X_class1 = np.random.randn(100, 2) + np.array([2, 2])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 100 + [1] * 100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
```

#### 1.3.2 训练和评估

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.metrics import accuracy_score

clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
print(f"score: {clf.score(X_test, y_test):.2%}")  # ClassifierMixin 给的
```

#### 1.3.3 预测概率

```python
proba = clf.predict_proba(X_test[:5])
print(f"前 5 个样本的预测概率:\n{proba}")
# 每行和为 1，第一列是类别 0 的概率，第二列是类别 1 的概率
```

### 1.4 小结

恭喜！你已经完成了 minisklearn 的第一个完整流程：

1. 准备数据（NumPy 数组）
2. 创建模型（`__init__` 传参数）
3. 训练（`fit`）
4. 预测（`predict`）
5. 评估（`score` 或 `metrics` 函数）

这个流程对**所有**算法都一样——这就是统一 API 的威力。

---

## 教程二：数据预处理实战

真实数据通常很"脏"：量纲不同、有类别特征、有缺失值。预处理是把脏数据变成算法能吃的数据的过程。

### 2.1 StandardScaler —— 标准化

#### 2.1.1 为什么需要标准化

```python
import numpy as np

# 假设两个特征：年龄（0-100）和收入（0-1000000）
X = np.array([
    [25, 50000],
    [30, 80000],
    [35, 120000],
    [40, 200000],
])
```

如果不标准化，收入这一列的数值远大于年龄，很多算法（如 KNN、SVM、梯度下降）会被收入主导。标准化让两列都变成均值 0、标准差 1，消除量纲影响。

#### 2.1.2 用法

```python
from minisklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化后:\n{X_scaled}")
print(f"每列均值: {X_scaled.mean(axis=0)}")    # 应接近 [0, 0]
print(f"每列标准差: {X_scaled.std(axis=0)}")   # 应接近 [1, 1]
```

#### 2.1.3 fit 和 transform 分开

```python
scaler = StandardScaler()
scaler.fit(X)           # 只学习，不转换
X_scaled = scaler.transform(X)  # 用学到的参数转换

# 为什么分开？因为训练集和测试集要用同一个 scaler
X_test = np.array([[28, 60000]])
X_test_scaled = scaler.transform(X_test)  # 用训练集的均值/标准差
print(f"测试集标准化: {X_test_scaled}")
```

**关键**：测试集**不能** `fit`，只能 `transform`。否则就是用测试集的统计量，造成数据泄露。

#### 2.1.4 查看学到的参数

```python
print(f"均值 (mean_): {scaler.mean_}")
print(f"标准差 (scale_): {scaler.scale_}")
# transform 公式: X_scaled = (X - mean_) / scale_
```

### 2.2 MinMaxScaler —— 归一化

```python
from minisklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # 默认归一化到 [0, 1]
X_scaled = scaler.fit_transform(X)
print(f"归一化后最小值: {X_scaled.min(axis=0)}")  # [0, 0]
print(f"归一化后最大值: {X_scaled.max(axis=0)}")  # [1, 1]

# 归一化到指定范围
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
print(f"归一化到 [-1, 1]: min={X_scaled.min(axis=0)}, max={X_scaled.max(axis=0)}")
```

#### StandardScaler vs MinMaxScaler 怎么选？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 数据近似正态分布 | StandardScaler | 保留分布形状 |
| 数据有明显边界 | MinMaxScaler | 如像素值 [0, 255] |
| 有离群点 | StandardScaler | MinMax 会被离群点拉扯 |
| 神经网络 | MinMaxScaler | 习惯输入在 [0, 1] 或 [-1, 1] |
| 距离类算法（KNN、SVM） | StandardScaler | 让所有特征等权贡献 |

### 2.3 LabelEncoder —— 标签编码

```python
from minisklearn.preprocessing import LabelEncoder

# 把字符串标签变成整数
labels = ['猫', '狗', '鸟', '猫', '狗']
encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)
print(f"编码后: {encoded}")  # [0, 1, 2, 0, 1]

# 反查
print(f"类别: {encoder.classes_}")  # ['鸟', '狗', '猫']（按字典序）
print(f"反编码: {encoder.inverse_transform([0, 1, 2])}")  # ['鸟', '狗', '猫']
```

### 2.4 OneHotEncoder —— 独热编码

```python
from minisklearn.preprocessing import OneHotEncoder

# 类别特征：颜色
colors = np.array([['红'], ['绿'], ['蓝'], ['红'], ['绿']])
encoder = OneHotEncoder()
onehot = encoder.fit_transform(colors)
print(f"独热编码:\n{onehot}")
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1],
#  [1, 0, 0],
#  [0, 1, 0]]
```

#### LabelEncoder vs OneHotEncoder

| 场景 | 推荐 | 原因 |
|------|------|------|
| 目标标签 | LabelEncoder | 分类器要整数标签 |
| 有序类别（低/中/高） | LabelEncoder | 保留顺序信息 |
| 无序类别（颜色/城市） | OneHotEncoder | 避免引入虚假顺序 |
| 树模型 | LabelEncoder | 树能处理任意整数 |
| 线性模型/SVM | OneHotEncoder | 否则把类别当连续值 |

### 2.5 预处理的统一接口

所有预处理器都遵守：

```python
transformer.fit(X)           # 学习参数
X_new = transformer.transform(X)  # 转换
X_new = transformer.fit_transform(X)  # 一步到位（通常有优化实现）
```

这就是 `TransformerMixin` 的契约。因为接口统一，才能用 Pipeline 串联。

---

## 教程三：每个算法的使用示例

### 3.1 线性回归（LinearRegression）

```python
from minisklearn.linear_model import LinearRegression
import numpy as np

# 多特征回归
X = np.random.randn(200, 3)
true_coef = np.array([1.5, -2.0, 0.5])
y = X @ true_coef + 2.0 + np.random.randn(200) * 0.1

model = LinearRegression()
model.fit(X, y)

print(f"真实权重: {true_coef}, 学到: {model.coef_}")
print(f"真实截距: 2.0, 学到: {model.intercept_}")
print(f"R²: {model.score(X, y):.4f}")
```

### 3.2 逻辑回归（LogisticRegression）

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.metrics import accuracy_score, f1_score

# 二分类
X = np.random.randn(300, 4)
y = (X.sum(axis=1) > 0).astype(int)

clf = LogisticRegression(C=1.0, max_iter=200)
clf.fit(X, y)
y_pred = clf.predict(X)

print(f"准确率: {accuracy_score(y, y_pred):.2%}")
print(f"F1: {f1_score(y, y_pred):.4f}")
print(f"权重: {clf.coef_}")
```

### 3.3 KNN 分类

```python
from minisklearn.neighbors import KNeighborsClassifier

X = np.random.randn(200, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # 圆内/圆外

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print(f"准确率: {knn.score(X, y):.2%}")

# KNN 的特点：fit 几乎不做事，predict 时才计算
```

### 3.4 KNN 回归

```python
from minisklearn.neighbors import KNeighborsRegressor

X = np.sort(np.random.rand(100, 1) * 5, axis=0)
y = np.sin(X.ravel()) + np.random.randn(100) * 0.1

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)
print(f"R²: {knn.score(X, y):.4f}")
```

### 3.5 决策树分类

```python
from minisklearn.tree import DecisionTreeClassifier

X = np.random.randn(300, 4)
y = (X[:, 0] * X[:, 1] > 0).astype(int)  # 非线性决策边界

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, y)
print(f"准确率: {tree.score(X, y):.2%}")
print(f"树深度: {tree.get_depth()}")
print(f"叶子数: {tree.get_n_leaves()}")
```

### 3.6 决策树回归

```python
from minisklearn.tree import DecisionTreeRegressor

X = np.sort(np.random.rand(200, 1) * 5, axis=0)
y = np.sin(X.ravel()) + np.random.randn(200) * 0.1

tree = DecisionTreeRegressor(max_depth=4)
tree.fit(X, y)
print(f"R²: {tree.score(X, y):.4f}")
```

### 3.7 随机森林分类

```python
from minisklearn.ensemble import RandomForestClassifier

X = np.random.randn(500, 6)
y = (X[:, 0] + X[:, 1] > X[:, 2]).astype(int)

rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X, y)
print(f"准确率: {rf.score(X, y):.2%}")
print(f"特征重要性: {rf.feature_importances_}")
```

### 3.8 随机森林回归

```python
from minisklearn.ensemble import RandomForestRegressor

X = np.random.randn(300, 5)
y = X.sum(axis=1) + np.random.randn(300) * 0.5

rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(X, y)
print(f"R²: {rf.score(X, y):.4f}")
```

### 3.9 KMeans 聚类

```python
from minisklearn.cluster import KMeans

# 生成 3 簇数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [-5, 5],
])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print(f"簇中心:\n{kmeans.cluster_centers_}")
print(f"惯性 (inertia_): {kmeans.inertia_:.2f}")  # 越小越好
print(f"前 10 个标签: {kmeans.labels_[:10]}")
```

### 3.10 PCA 降维

```python
from minisklearn.decomposition import PCA

# 生成 5 维数据，但实际只有 2 个主成分
np.random.seed(42)
X = np.random.randn(300, 2)
X = np.hstack([X, X[:, [0]] + X[:, [1]], X[:, [0]] * 2])  # 后 3 列是前 2 列的线性组合

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"降维后形状: {X_pca.shape}")
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.cumsum()}")
```

### 3.11 LinearSVC

```python
from minisklearn.svm import LinearSVC

X = np.random.randn(300, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

svm = LinearSVC(C=1.0)
svm.fit(X, y)
print(f"准确率: {svm.score(X, y):.2%}")
print(f"权重: {svm.coef_}")
```

### 3.12 GaussianNB

```python
from minisklearn.naive_bayes import GaussianNB

X = np.random.randn(300, 3)
y = (X.sum(axis=1) > 0).astype(int)

nb = GaussianNB()
nb.fit(X, y)
print(f"准确率: {nb.score(X, y):.2%}")
print(f"各类别均值: {nb.theta_}")  # 每个特征在每个类别下的均值
```

---

## 教程四：Pipeline + GridSearchCV 实战

这是 minisklearn 架构设计的精华应用。Pipeline 串联流程，GridSearchCV 调参，两者组合是机器学习的标准工作流。

### 4.1 为什么需要 Pipeline

#### 4.1.1 错误做法（数据泄露）

```python
# ❌ 错误！测试集的统计量泄露到了训练过程
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 在全部数据上 fit！
X_train, X_test = train_test_split(X_scaled, ...)
clf.fit(X_train, y_train)
clf.predict(X_test)
```

问题：`scaler.fit_transform(X)` 用了全部数据的均值/标准差，包括测试集。测试集的信息泄露到了训练过程，评估结果会偏乐观。

#### 4.1.2 正确做法（手动）

```python
# ✅ 正确，但繁琐
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只在训练集 fit
X_test_scaled = scaler.transform(X_test)         # 测试集只 transform

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)
clf.predict(X_test_scaled)
```

#### 4.1.3 最佳做法（Pipeline）

```python
# ✅✅ 最佳，简洁且不会泄露
from minisklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)  # 内部自动只在训练集 fit scaler
pipe.predict(X_test)
```

Pipeline 把预处理和训练绑成一个对象，保证 `fit` 时只在训练数据上学习，`predict` 时用学到的参数转换。

### 4.2 Pipeline 详解

#### 4.2.1 基本用法

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),      # 第 1 步：标准化
    ('pca', PCA(n_components=5)),      # 第 2 步：降维
    ('clf', LogisticRegression(C=1.0)),# 第 3 步：分类
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
```

#### 4.2.2 访问中间步骤

```python
# 通过名字访问
print(pipe.named_steps['scaler'].mean_)  # 标准化的均值

# 通过索引访问
print(pipe.steps[0][1].mean_)  # 第 0 步的估计器
```

#### 4.2.3 修改参数

```python
# 修改某一步的参数
pipe.set_params(clf__C=0.5)  # 注意双下划线：步骤名__参数名
pipe.fit(X_train, y_train)
```

#### 4.2.4 Pipeline 的 fit 内部做了什么

```
fit(X, y):
    X1 = scaler.fit_transform(X, y)     # 标准化
    X2 = pca.fit_transform(X1, y)       # 降维
    clf.fit(X2, y)                      # 分类
    return self

predict(X):
    X1 = scaler.transform(X)            # 用训练集的参数
    X2 = pca.transform(X1)
    return clf.predict(X2)
```

关键：`predict` 时只用 `transform`，不会重新 `fit`，所以不会泄露。

### 4.3 GridSearchCV 详解

#### 4.3.1 基本用法

```python
from minisklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

# 参数空间：步骤名__参数名
param_grid = {
    'clf__C': [0.01, 0.1, 1.0, 10.0],
    'clf__max_iter': [100, 200],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)

print(f"最佳参数: {search.best_params_}")
print(f"最佳分数: {search.best_score_:.2%}")
print(f"最佳模型: {search.best_estimator_}")
```

#### 4.3.2 GridSearchCV 内部做了什么

```
for 每个参数组合 in 参数空间:
    for 每个折 in KFold:
        clf = clone(估计器)          # 克隆出干净对象
        clf.set_params(参数组合)     # 设置参数
        clf.fit(训练折数据)
        scores.append(clf.score(验证折数据))
    记录平均分
返回最佳参数对应的模型
```

关键点：
- **`clone`**：每次都要克隆，否则修改同一个对象
- **`set_params`**：通过反射设置参数，支持 `clf__C` 这种嵌套参数
- **交叉验证**：每个参数组合都做 K 折验证，避免过拟合到某一折

#### 4.3.3 搜索多个参数

```python
param_grid = {
    'scaler__with_mean': [True, False],    # StandardScaler 的参数
    'clf__C': [0.01, 0.1, 1.0, 10.0],
    'clf__max_iter': [100, 200, 500],
}

search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=1)
search.fit(X_train, y_train)
print(f"搜索了 {len(search.cv_results_['params'])} 个组合")
```

#### 4.3.4 用 best_estimator_ 预测

```python
# GridSearchCV 已经在 best_params_ 上 refit 了完整训练集
y_pred = search.predict(X_test)  # 直接用最佳模型预测
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2%}")
```

### 4.4 完整工作流

```python
import numpy as np
from minisklearn.model_selection import train_test_split, GridSearchCV
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression
from minisklearn.metrics import accuracy_score, classification_report

# 1. 准备数据
np.random.seed(42)
X = np.random.randn(500, 10)
y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)

# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建 Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

# 4. 定义参数空间
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'clf__max_iter': [100, 200, 500],
}

# 5. 网格搜索 + 交叉验证
search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)

# 6. 查看结果
print(f"最佳参数: {search.best_params_}")
print(f"交叉验证最佳分数: {search.best_score_:.2%}")

# 7. 在测试集上评估
y_pred = search.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2%}")
```

这就是标准的机器学习工作流：**数据 → 划分 → Pipeline → GridSearchCV → 评估**。

---

## 教程五：交叉验证评估

### 5.1 为什么需要交叉验证

`train_test_split` 只划分一次，评估结果依赖划分方式。如果测试集恰好"简单"，结果会偏乐观。K 折交叉验证做 K 次划分，取平均，更稳健。

### 5.2 KFold

```python
from minisklearn.model_selection import KFold

X = np.arange(20).reshape(10, 2)
y = np.arange(10)

kf = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"折 {fold}: 训练 {train_idx}, 验证 {val_idx}")
```

5 折交叉验证：把数据分成 5 份，每次用 4 份训练、1 份验证，循环 5 次。

### 5.3 cross_val_score

```python
from minisklearn.model_selection import cross_val_score
from minisklearn.linear_model import LogisticRegression

X = np.random.randn(200, 5)
y = (X.sum(axis=1) > 0).astype(int)

clf = LogisticRegression()
scores = cross_val_score(clf, X, y, cv=5)
print(f"5 折分数: {scores}")
print(f"平均: {scores.mean():.2%} ± {scores.std():.2%}")
```

`cross_val_score` 内部：
1. 用 `KFold` 划分
2. 每折 `clone` 估计器 → `fit` → `score`
3. 返回每折的分数

### 5.4 选择 K

| K | 优点 | 缺点 |
|---|------|------|
| 5 | 训练集大（80%），常见默认 | - |
| 10 | 评估更稳定 | 训练集小（90%），慢 2 倍 |
| LOO（留一） | 最大化训练数据 | 极慢，方差大 |

一般用 5 或 10。

---

## 教程六：C++ 扩展编译和使用

本教程演示如何编译和使用 C++ 扩展，对比纯 Python 版本的性能。

### 6.1 为什么需要 C++ 扩展

Python 的 NumPy 已经很快（底层是 C），但有些操作 Python 解释器开销大：

- **循环密集型**：如 KMeans 的分配步骤，每次迭代要遍历所有样本
- **内存布局敏感**：NumPy 的广播有时不如手写循环快
- **算法特定优化**：如 SIMD 指令、缓存友好布局

本项目用 KMeans 演示：纯 Python 版本用 NumPy 向量化，C++ 版本用手写循环 + 缓存优化。

### 6.2 准备环境

#### 6.2.1 安装依赖

```bash
pip install cmake pybind11 scikit-learn matplotlib
```

#### 6.2.2 安装 C++ 编译器

- **Windows**：安装 Visual Studio Build Tools，勾选"C++ 桌面开发"
- **macOS**：`xcode-select --install`
- **Linux**：`sudo apt install build-essential`

验证：
```bash
cmake --version  # 应输出 cmake 版本
python -c "import pybind11; print(pybind11.__version__)"
```

### 6.3 编译 C++ 扩展

```bash
# 在项目根目录
python cpp/build.py
```

`build.py` 会：
1. 用 CMake 配置构建
2. 编译 `cpp/src/` 下的 C++ 源码
3. 用 pybind11 生成 Python 绑定
4. 把编译产物放到 `minisklearn/_fast/`

成功后应该看到：
```
[100%] Built target _fast_kmeans
扩展已安装到 minisklearn/_fast/
```

### 6.4 使用 C++ 扩展

```python
import numpy as np
from minisklearn.cluster import KMeans
from minisklearn._fast import _fast_kmeans

# 生成大数据
np.random.seed(42)
X = np.random.randn(10000, 10)

# 纯 Python 版本
import time
t1 = time.time()
kmeans_py = KMeans(n_clusters=10, random_state=42)
kmeans_py.fit(X)
t2 = time.time()
print(f"纯 Python: {t2 - t1:.2f}s")

# C++ 版本
t1 = time.time()
kmeans_cpp = _fast_kmeans.KMeans(n_clusters=10, random_state=42)
kmeans_cpp.fit(X)
t2 = time.time()
print(f"C++ 版本: {t2 - t1:.2f}s")

# 对比结果
print(f"惯性差异: {abs(kmeans_py.inertia_ - kmeans_cpp.inertia_):.4f}")
```

### 6.5 C++ 源码结构

```
cpp/
├── CMakeLists.txt          # CMake 构建配置
├── build.py                # Python 构建脚本
└── src/
    ├── kmeans.cpp          # KMeans 的 C++ 实现
    └── bindings.cpp        # pybind11 绑定代码
```

#### 6.5.1 C++ 实现（简化）

```cpp
// kmeans.cpp（简化版）
#include <vector>
#include <cmath>

void kmeans_fit(const double* X, int n_samples, int n_features,
                int n_clusters, int max_iter, double* centroids) {
    // 1. 初始化中心
    // 2. 迭代：
    //    a. 分配：每个样本找最近中心
    //    b. 更新：重新计算中心
    // 3. 直到收敛或达到 max_iter
}
```

#### 6.5.2 pybind11 绑定

```cpp
// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmeans.cpp"

namespace py = pybind11;

PYBIND11_MODULE(_fast_kmeans, m) {
    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int, int>(), py::arg("n_clusters"), py::arg("random_state") = 42)
        .def("fit", &KMeans::fit)
        .def("predict", &KMeans::predict)
        .def_readwrite("cluster_centers_", &KMeans::centroids)
        .def_readwrite("inertia_", &KMeans::inertia);
}
```

pybind11 把 C++ 类暴露给 Python，让 Python 能像用普通类一样用 C++ 类。

### 6.6 什么时候该写 C++ 扩展

| 场景 | 建议 |
|------|------|
| 原型开发 | 纯 Python |
| 数据量小 | 纯 Python（开销在 Python 调用，不在计算） |
| 数据量大 + 循环密集 | 考虑 C++ |
| 已有 NumPy 向量化 | 先优化向量化，再考虑 C++ |
| 需要并行 | 先试 joblib / multiprocessing |

**不要过早优化**。先写正确的纯 Python，跑 benchmark 确认瓶颈，再针对性用 C++。

---

## 教程七：性能对比教程

### 7.1 与 sklearn 对比

```python
import numpy as np
import time
from minisklearn.cluster import KMeans as MiniKMeans
from sklearn.cluster import KMeans as SklearnKMeans

# 生成数据
np.random.seed(42)
X = np.random.randn(5000, 8)

# minisklearn
t1 = time.time()
km_mini = MiniKMeans(n_clusters=10, random_state=42)
km_mini.fit(X)
t_mini = time.time() - t1

# sklearn
t1 = time.time()
km_sk = SklearnKMeans(n_clusters=10, random_state=42, n_init=1)
km_sk.fit(X)
t_sk = time.time() - t1

print(f"minisklearn: {t_mini:.3f}s, inertia={km_mini.inertia_:.2f}")
print(f"sklearn:     {t_sk:.3f}s, inertia={km_sk.inertia_:.2f}")
print(f"速度比: {t_mini / t_sk:.1f}x")
```

预期结果：sklearn 比 minisklearn 快几倍（sklearn 用了 Cython 和更多优化），但结果（inertia）应该接近。

### 7.2 运行完整 benchmark

```bash
python benchmarks/run_benchmarks.py
```

会对比所有算法在 minisklearn、sklearn、C++ 扩展三者的速度和结果，生成报告。

### 7.3 读懂 benchmark 报告

```
算法              minisklearn    sklearn    C++扩展    结果差异
KMeans (10k×10)   2.3s          0.8s       0.5s       <1%
KNN (10k×5)       1.5s          0.3s       -          <0.1%
...
```

- **速度**：sklearn 通常最快（Cython 优化），C++ 扩展次之，纯 Python 最慢
- **结果差异**：应小于 1%，否则可能是实现 bug

---

## 教程八：常见工作流模式

### 8.1 模式一：分类标准流程

```python
# 1. 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

# 3. 调参
search = GridSearchCV(pipe, {'clf__C': [0.1, 1.0, 10.0]}, cv=5)
search.fit(X_train, y_train)

# 4. 评估
print(search.score(X_test, y_test))
```

### 8.2 模式二：聚类流程

```python
# 1. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)  # fit_predict 一步到位

# 3. 评估（如果有真实标签）
from minisklearn.metrics import adjusted_rand_score
print(adjusted_rand_score(y_true, labels))
```

### 8.3 模式三：降维 + 分类

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

### 8.4 模式四：模型选择

```python
# 对比多个模型
from minisklearn.neighbors import KNeighborsClassifier
from minisklearn.tree import DecisionTreeClassifier
from minisklearn.ensemble import RandomForestClassifier

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Tree': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(n_estimators=50),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2%} ± {scores.std():.2%}")
```

### 8.5 模式五：特征工程 Pipeline

```python
# 假设 X 有数值列和类别列（实际需要 ColumnTransformer，本项目暂未实现）
# 简化版：全部标准化
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=50)),
])
```

---

## 教程九：调试和排查

### 9.1 查看估计器状态

```python
clf = LogisticRegression()
print(clf)  # __repr__，显示所有参数

clf.fit(X, y)
print(clf.coef_)        # 学到的权重
print(clf.n_iter_)      # 迭代次数
```

### 9.2 检查是否拟合

```python
from minisklearn.utils import check_is_fitted

clf = LogisticRegression()
try:
    check_is_fitted(clf)
    print("已拟合")
except:
    print("未拟合")  # 会走到这里
```

### 9.3 调试 Pipeline

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.fit(X, y)

# 查看每一步
print(pipe.named_steps['scaler'].mean_)
print(pipe.named_steps['clf'].coef_)
```

### 9.4 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `NotFittedError` | 没 fit 就 predict | 先调 fit |
| `ValueError: X has NaN` | 数据有缺失值 | 填充或删除缺失值 |
| `ValueError: shapes not aligned` | 训练/测试特征数不一致 | 检查数据形状 |
| `TypeError: clone() failed` | `__init__` 没存参数 | 确保 `self.C = C` |

---

## 教程十：从 minisklearn 迁移到 sklearn

因为 API 兼容，迁移很简单：

```python
# minisklearn 代码
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression
from minisklearn.pipeline import Pipeline

# 改成 sklearn，只需换 import
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 其余代码完全不变
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 迁移检查清单

- [ ] 把所有 `minisklearn` 改成 `sklearn`
- [ ] 检查参数名是否一致（绝大多数一致）
- [ ] 跑一遍测试，对比结果
- [ ] 注意 sklearn 可能有更多参数（如 `n_jobs`、`verbose`）

---

## 下一步

- **想深入架构**：看 [架构设计](../architecture/01-unified-api.md)
- **想看算法原理**：看 [算法实现](../algorithms/index.md)
- **想看完整示例**：看 `examples/` 目录
- **想跑性能对比**：`python benchmarks/run_benchmarks.py`

---

## 教程内容概览

所有教程内容已直接写在本页中，共 17 个教程，涵盖从入门到高级的完整学习路径：

| 教程 | 内容 | 难度 |
|------|------|------|
| 教程一 | 环境配置 + 第一个模型（线性回归 + 逻辑回归） | 入门 |
| 教程二 | 数据预处理实战（标准化 / 归一化 / 编码） | 入门 |
| 教程三 | 12 个算法的使用示例 | 入门 |
| 教程四 | Pipeline + GridSearchCV 标准工作流 | 进阶 |
| 教程五 | 交叉验证评估 | 进阶 |
| 教程六 | C++ 扩展编译和使用 | 高级 |
| 教程七 | 性能对比（与 sklearn 对比基准） | 高级 |
| 教程八 | 常见工作流模式（5 种） | 进阶 |
| 教程九 | 调试和排查 | 进阶 |
| 教程十 | 从 minisklearn 迁移到 sklearn | 进阶 |
| 教程十一 | 深入技术分析：fit/predict 背后的机制 | 高级 |
| 教程十二 | 对比实验：亲手比较算法性能 | 高级 |
| 教程十三 | 参数调优实战指南 | 高级 |
| 教程十四 | 常见错误与调试技巧大全 | 进阶 |
| 教程十五 | 实际应用场景实战 | 高级 |
| 教程十六 | 思考题与练习 | 拓展 |
| 教程十七 | 扩展阅读与学习路径 | 拓展 |

每个教程都包含：
- 完整的可运行代码
- 逐步骤的输出说明
- 常见错误和解决方法
- 与 sklearn 的对比
- 练习题

---

## 学习建议

### 给初学者

1. **先跑通教程一**：建立信心，确认环境没问题
2. **不要跳步**：每一步都自己敲一遍，不要复制粘贴
3. **改参数看效果**：把 `C=1.0` 改成 `C=0.01`，看结果怎么变
4. **遇到错误先读错误信息**：Python 的报错通常很清楚
5. **画图辅助理解**：用 matplotlib 画出数据点和决策边界

### 给有经验者

1. **重点看架构**：教程是应用，架构是原理
2. **对比 sklearn 源码**：看同样的功能 sklearn 怎么实现
3. **看测试文件**：`tests/` 下有大量用法示例
4. **尝试扩展**：加一个新算法，检验对架构的理解

### 给教学者

1. **按教程顺序讲**：从环境配置到第一个模型，循序渐进
2. **每节课配练习**：给一个数据集，让学生用学过的算法做
3. **鼓励读源码**：`minisklearn/` 下的代码都有详细注释
4. **用本项目做对比**：和 sklearn 对比，讲工业级库的考虑

---

## 常见问题

### Q：教程代码在哪里？

本页的所有代码块都可以直接复制运行。17 个教程已全部写在本页中，按顺序阅读即可。

### Q：需要什么数据集？

教程用 NumPy 生成的模拟数据，不需要下载。如果你想用真实数据集，推荐：
- **分类**：鸢尾花（`sklearn.datasets.load_iris()`）
- **回归**：波士顿房价（`sklearn.datasets.load_boston()`）
- **聚类**：手写数字（`sklearn.datasets.load_digits()`）

注意：用 sklearn 的数据集需要 `pip install scikit-learn`。

### Q：教程和架构文档什么关系？

- **架构文档**：讲"为什么"——为什么这么设计
- **教程**：讲"怎么做"——怎么用这套设计

建议交叉阅读：用教程跑通后，去看对应的架构文档理解原理。

### Q：能跳过某些教程吗？

可以。如果你已经会 sklearn，直接看教程四（Pipeline + GridSearchCV）和教程六（C++ 扩展）即可，因为那是本项目的特色。

### Q：教程会更新吗？

会。随着算法增加和文档完善，教程会持续更新。欢迎在 issue 里提你想看的教程主题。

---

## 总结

本教程索引页涵盖了 minisklearn 的完整使用流程：

1. **环境配置**：安装、验证
2. **基础用法**：线性回归、逻辑回归
3. **预处理**：标准化、归一化、编码
4. **所有算法示例**：12 个算法的用法
5. **Pipeline + GridSearchCV**：标准工作流
6. **交叉验证**：稳健评估
7. **C++ 扩展**：性能优化
8. **性能对比**：与 sklearn 对比
9. **工作流模式**：5 种常见模式
10. **调试排查**：常见错误和解决

核心思想：**所有算法都遵循 `fit` / `predict` 契约，学会一个就会用所有**。这是 minisklearn（和 sklearn）最大的设计胜利。

---

## 教程十一：深入技术分析——理解 fit/predict 背后的机制

### 11.1 估计器的三种身份

minisklearn 的每个类通过继承不同的 Mixin 获得不同身份：

```python
class StandardScaler(BaseEstimator, TransformerMixin):  # 转换器
class LogisticRegression(BaseEstimator, ClassifierMixin):  # 分类器
class LinearRegression(BaseEstimator, RegressorMixin):  # 回归器
class KMeans(BaseEstimator, ClusterMixin):  # 聚类器
```

- `BaseEstimator`：提供 `get_params`/`set_params`/`clone`/`__repr__`
- `TransformerMixin`：提供 `fit_transform`（默认 = fit + transform）
- `ClassifierMixin`：提供 `score`（返回准确率）
- `RegressorMixin`：提供 `score`（返回 R²）
- `ClusterMixin`：提供 `fit_predict`

你只需要实现 `fit` 和 `predict`/`transform`，其余的 Mixin 自动给你。

### 11.2 __init__ 的铁律

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100, tol=1e-4):
        self.C = C              # 只存参数，不做计算
        self.max_iter = max_iter
        self.tol = tol
```

**铁律**：`__init__` 只能存参数，不能做任何计算或创建对象。原因：

```python
# clone 的实现
def clone(estimator):
    params = estimator.get_params()  # 从 __init__ 参数取
    new = type(estimator)(**params)  # 重新构造
    return new
```

如果 `__init__` 里做了计算，`clone` 重新构造时会重复计算，甚至出错。

```python
# 错误示例
class BadClassifier(BaseEstimator):
    def __init__(self, k=5):
        self.k = k
        self.model = SomeModel(k)  # 在 __init__ 创建对象！
# clone 时 self.model 会被重新创建，状态丢失
```

### 11.3 带下划线的属性约定

```python
clf = LogisticRegression()
clf.fit(X, y)
print(clf.coef_)        # fit 学到的，带下划线
print(clf.intercept_)   # fit 学到的，带下划线
print(clf.C)            # 用户设的参数，不带下划线
```

约定：
- `coef_`、`mean_`、`classes_`：fit 后才有，是"学习结果"
- `C`、`max_iter`：`__init__` 设的，是"超参数"

`check_is_fitted` 通过查找带下划线的属性判断是否拟合：

```python
def check_is_fitted(estimator, attributes=None):
    if attributes is None:
        attributes = [attr for attr in vars(estimator) if attr.endswith('_')]
    if not attributes:
        raise NotFittedError(f"此 {type(estimator).__name__} 实例还未拟合")
```

### 11.4 fit 返回 self 的原因

```python
clf = LogisticRegression().fit(X, y)  # 链式调用
# 等价于
clf = LogisticRegression()
clf.fit(X, y)
```

`fit` 返回 `self` 允许链式调用，也是 sklearn 的 API 约定。

### 11.5 score 方法来自哪里

```python
# ClassifierMixin 提供
def score(self, X, y):
    return accuracy_score(y, self.predict(X))

# RegressorMixin 提供
def score(self, X, y):
    return r2_score(y, self.predict(X))
```

你不用自己写 `score`，Mixin 自动给你。这就是多继承 Mixin 的好处。

---

## 教程十二：对比实验——亲手比较算法性能

### 12.1 实验一：分类算法大比拼

```python
import numpy as np
import time
from minisklearn.model_selection import cross_val_score, train_test_split
from minisklearn.preprocessing import StandardScaler
from minisklearn.pipeline import Pipeline
from minisklearn.linear_model import LogisticRegression
from minisklearn.neighbors import KNeighborsClassifier
from minisklearn.tree import DecisionTreeClassifier
from minisklearn.ensemble import RandomForestClassifier
from minisklearn.svm import LinearSVC
from minisklearn.naive_bayes import GaussianNB

# 准备数据
rng = np.random.RandomState(42)
X = rng.randn(500, 8)
y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)  # 非线性

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(max_depth=8),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LinearSVC': LinearSVC(max_iter=500),
    'GaussianNB': GaussianNB(),
}

print(f"{'模型':<22} {'CV分数':<12} {'训练时间':<10} {'预测时间':<10} {'测试分数':<10}")
print("-" * 70)
for name, clf in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    
    t0 = time.time()
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    t_cv = time.time() - t0
    
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t_fit = time.time() - t0
    
    t0 = time.time()
    pipe.predict(X_test)
    t_pred = time.time() - t0
    
    test_score = pipe.score(X_test, y_test)
    print(f"{name:<22} {cv_scores.mean():.3f}±{cv_scores.std():.3f}  {t_fit:<10.4f} {t_pred:<10.4f} {test_score:<10.4f}")
```

### 12.2 实验二：数据规模的影响

```python
for n in [100, 500, 1000, 5000]:
    X = rng.randn(n, 10)
    y = (X.sum(axis=1) > 0).astype(int)
    
    for name, clf in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X, y, cv=5)
        print(f"n={n:5d}, {name:<20s}: {scores.mean():.4f}")
    print()
```

观察：
- 小数据：简单模型（NB、LR）更稳，复杂模型（RF）过拟合
- 大数据：复杂模型（RF）优势显现，简单模型欠拟合

### 12.3 实验三：特征数的影响

```python
n, d = 1000, 0
for d in [5, 20, 50, 100]:
    X = rng.randn(1000, d)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 只用前 2 个特征
    
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5)
    print(f"d={d:3d}, KNN: {scores.mean():.4f}")  # 维度灾难，分数下降
```

KNN 在高维下效果变差——所有点距离趋同，"最近邻"失去意义。

---

## 教程十三：参数调优实战指南

### 13.1 调参的标准流程

```python
# 1. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 构建 Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42)),
])

# 3. 粗搜
coarse_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [5, 10, None],
}
grid1 = GridSearchCV(pipe, coarse_grid, cv=5, n_jobs=-1).fit(X_train, y_train)
print(f"粗搜最优: {grid1.best_params_}, 分数: {grid1.best_score_:.4f}")

# 4. 细搜
best_n = grid1.best_params_['clf__n_estimators']
fine_grid = {
    'clf__n_estimators': [best_n - 25, best_n, best_n + 25],
    'clf__max_depth': [grid1.best_params_['clf__max_depth']],
}
grid2 = GridSearchCV(pipe, fine_grid, cv=5).fit(X_train, y_train)
print(f"细搜最优: {grid2.best_params_}, 分数: {grid2.best_score_:.4f}")

# 5. 最终评估
print(f"测试分数: {grid2.score(X_test, y_test):.4f}")
```

### 13.2 各算法的调参经验

#### LogisticRegression

```python
# 唯一关键参数：C（正则强度的倒数）
# C 大 → 弱正则 → 过拟合风险
# C 小 → 强正则 → 欠拟合风险
# 经验：对数尺度搜索
param_grid = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
```

#### KNeighborsClassifier

```python
# 关键参数：n_neighbors
# 经验：K 取 sqrt(n) 附近，用奇数避免平票
import math
k_default = int(math.sqrt(len(y_train)))
if k_default % 2 == 0:
    k_default += 1
param_grid = {'clf__n_neighbors': [k_default-4, k_default-2, k_default, k_default+2, k_default+4]}
```

#### DecisionTreeClassifier

```python
# 关键参数：max_depth
# 经验：从 3 开始，逐步增加
param_grid = {'clf__max_depth': [3, 5, 7, 10, 15, 20, None]}
# None 表示不限制，通常过拟合
```

#### RandomForestClassifier

```python
# 关键参数：n_estimators, max_depth, max_features
param_grid = {
    'clf__n_estimators': [100, 200, 500],      # 树越多越好但越慢
    'clf__max_depth': [5, 10, None],            # None 让树充分生长
    'clf__max_features': ['sqrt', 'log2', None], # 分裂时考虑的特征数
}
# 经验：n_estimators 大方向调，max_features='sqrt' 通常最好
```

#### KMeans

```python
# 关键参数：n_clusters（K）
# 用肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertias.append(km.inertia_)
# 画图找肘部
import matplotlib.pyplot as plt
plt.plot(range(1, 11), inertias, 'o-')
plt.xlabel('K'); plt.ylabel('inertia'); plt.show()
```

### 13.3 调参的陷阱

```python
# 陷阱 1：用测试集调参
grid.fit(X_test, y_test)  # 数据泄露！

# 陷阱 2：参数空间太细（验证集过拟合）
grid = GridSearchCV(clf, {'C': np.linspace(0.001, 1000, 10000)}, cv=5)

# 陷阱 3：CV 分数当真实泛化
print(grid.best_score_)  # 乐观偏误
print(grid.score(X_test, y_test))  # 真实泛化

# 陷阱 4：不看训练分数（无法判断过拟合）
# 用 cross_validate 看训练分数
```

---

## 教程十四：常见错误与调试技巧大全

### 14.1 错误：NotFittedError

```python
clf = LogisticRegression()
clf.predict(X)  # NotFittedError: 还没 fit
# 解决：先 clf.fit(X_train, y_train)
```

### 14.2 错误：数据形状不对

```python
# X 必须是 2D
X = np.random.randn(100)  # 1D，报错
X = X.reshape(-1, 1)      # 改成 2D (100, 1)

# y 必须是 1D
y = np.random.randn(100, 1)  # 2D，可能报错
y = y.ravel()                # 改成 1D
```

### 14.3 错误：数据有缺失值

```python
X = np.array([[1, 2], [3, np.nan], [5, 6]])
LogisticRegression().fit(X, y)  # ValueError: Input contains NaN
# 解决：用 SimpleImputer 填充
from sklearn.impute import SimpleImputer
X = SimpleImputer(strategy='mean').fit_transform(X)
```

### 14.4 错误：训练测试特征数不一致

```python
clf.fit(X_train[:, :5], y_train)  # 用 5 个特征训练
clf.predict(X_test[:, :6])        # 用 6 个特征预测，报错
# 解决：确保特征数一致
```

### 14.5 错误：标签类型不对

```python
y = ['cat', 'dog', 'bird']  # 字符串
# 部分算法要整数标签
from minisklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)  # [0, 1, 2]
```

### 14.6 错误：数据泄露

```python
# 错误：标准化用全部数据
scaler = StandardScaler().fit(X)  # 含测试集
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# 正确：用 Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
cross_val_score(pipe, X, y, cv=5)
```

### 14.7 调试技巧：逐步检查

```python
# 1. 检查数据
print(f"X: {X.shape}, {X.dtype}, NaN={np.isnan(X).any()}")
print(f"y: {y.shape}, 类别={np.unique(y)}, 分布={np.bincount(y)}")

# 2. 检查模型参数
print(clf)  # __repr__ 显示所有参数
print(clf.get_params())

# 3. 检查训练结果
clf.fit(X_train, y_train)
print(f"训练分数: {clf.score(X_train, y_train):.4f}")
print(f"测试分数: {clf.score(X_test, y_test):.4f}")
# 训练远高于测试 → 过拟合

# 4. 检查预测
y_pred = clf.predict(X_test)
print(f"预测分布: {np.bincount(y_pred)}")
print(f"真实分布: {np.bincount(y_test)}")
```

### 14.8 调试技巧：用交叉验证看稳定性

```python
scores = cross_val_score(clf, X, y, cv=10)
print(f"10 折分数: {scores}")
print(f"标准差: {scores.std():.4f}")
# 标准差大 → 模型对数据划分敏感，可能不稳定
```

### 14.9 常见错误速查表

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `NotFittedError` | 没 fit 就 predict | 先调 fit |
| `ValueError: X has NaN` | 数据有缺失 | 用 SimpleImputer |
| `ValueError: shapes not aligned` | 特征数不一致 | 检查数据形状 |
| `ValueError: unknown label type` | 标签非整数 | 用 LabelEncoder |
| `AttributeError: no 'transform'` | Pipeline 中间步非转换器 | 检查步骤类型 |
| 训练分数远高于测试 | 过拟合 | 增加正则、减少深度 |
| 训练和测试都低 | 欠拟合 | 减少正则、增加复杂度 |

---

## 教程十五：实际应用场景实战

### 15.1 场景：鸢尾花分类全流程

```python
import numpy as np
from sklearn.datasets import load_iris
from minisklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.ensemble import RandomForestClassifier
from minisklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Pipeline + 调参
pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))])
grid = GridSearchCV(pipe, {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [3, 5, None]}, cv=5)
grid.fit(X_train, y_train)

# 3. 评估
print(f"最优参数: {grid.best_params_}")
print(f"测试准确率: {grid.score(X_test, y_test):.2%}")
```

### 15.2 场景：回归预测

```python
from minisklearn.linear_model import LinearRegression
from minisklearn.metrics import r2_score, mean_squared_error

# 生成回归数据
X = rng.randn(300, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + 0.5 * X[:, 2] + rng.randn(300) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
pipe.fit(X_train, y_train)
print(f"R²: {pipe.score(X_test, y_test):.4f}")
print(f"权重: {pipe.steps[1][1].coef_}")  # 接近 [2, -3, 0.5, 0, 0]
```

### 15.3 场景：客户分群（聚类）

```python
from minisklearn.cluster import KMeans

# 假设是客户消费数据：[消费频次, 平均消费金额, 最近消费天数]
X = rng.randn(500, 3) * np.array([10, 100, 30]) + np.array([20, 500, 60])

# 标准化后聚类
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 分析各簇
for k in range(4):
    cluster = X[labels == k]
    print(f"簇 {k}: {len(cluster)} 人, 均值={cluster.mean(axis=0).round(1)}")
```

### 15.4 场景：降维可视化

```python
from minisklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_2d = pca.fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(8, 6))
for cls in np.unique(y):
    plt.scatter(X_2d[y==cls, 0], X_2d[y==cls, 1], label=f'类 {cls}')
plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend(); plt.title('PCA 降维可视化'); plt.show()
```

### 15.5 场景：模型选择报告

```python
import pandas as pd

models = {
    'LR': LogisticRegression(max_iter=500),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': LinearSVC(max_iter=500),
    'NB': GaussianNB(),
}

results = []
for name, clf in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    scores = cross_val_score(pipe, X, y, cv=5)
    results.append({
        '模型': name,
        'CV均值': scores.mean(),
        'CV标准差': scores.std(),
        '最小值': scores.min(),
        '最大值': scores.max(),
    })

df = pd.DataFrame(results).sort_values('CV均值', ascending=False)
print(df.to_string(index=False))
```

### 15.6 场景：模型部署

```python
import pickle

# 训练并保存
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# 部署后加载使用
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(X_new)  # 预处理参数都在
```

---

## 教程十六：思考题与练习

### 基础题

1. **简答题**：minisklearn 的所有算法都遵循什么统一接口？这个设计有什么好处？

2. **简答题**：为什么 `__init__` 只能存参数不能做计算？

3. **代码题**：用 LogisticRegression 对随机生成的二分类数据做分类，打印准确率。

4. **操作题**：创建一个 Pipeline，包含 StandardScaler 和 RandomForestClassifier，在任意数据上训练。

### 进阶题

5. **分析题**：StandardScaler 和 MinMaxScaler 各适合什么场景？举例说明。

6. **代码题**：用 GridSearchCV 搜索 RandomForestClassifier 的 `n_estimators` 和 `max_depth`，输出最优参数。

7. **调试题**：下面代码有什么问题？如何修复？
   ```python
   scaler = StandardScaler().fit(X)
   X_scaled = scaler.transform(X)
   X_train, X_test = train_test_split(X_scaled, y, test_size=0.2)
   clf.fit(X_train, y_train)
   ```

8. **设计题**：对于以下场景，你会选什么算法？
   - 文本分类（高维稀疏）
   - 小数据集 + 需要快速基线
   - 非线性关系 + 需要可解释性
   - 不知道标签的分组

### 高级题

9. **实验题**：设计实验验证"KNN 在高维下效果变差"（维度灾难），画出准确率随特征数的变化。

10. **源码题**：阅读 minisklearn 的 BaseEstimator 源码，理解 `get_params` 是如何通过反射从 `__init__` 签名提取参数的。

11. **架构题**：为什么 sklearn 用 Mixin 而非大基类？如果用一个 `Estimator` 基类包含所有功能，会有什么问题？

12. **扩展题**：实现一个自定义转换器 `LogTransformer`（对数据取对数），继承 `BaseEstimator` 和 `TransformerMixin`，确保能用在 Pipeline 中。

---

## 教程十七：扩展阅读与学习路径

### 17.1 官方资源

- [sklearn 官方教程](https://scikit-learn.org/stable/tutorial/index.html)
- [sklearn API 参考](https://scikit-learn.org/stable/modules/classes.html)
- [sklearn 示例库](https://scikit-learn.org/stable/auto_examples/index.html)

### 17.2 推荐书籍

| 书名 | 适合 | 特点 |
|------|------|------|
| 《Hands-On ML with Scikit-Learn》 | 入门 | 实战导向，代码多 |
| 《统计学习方法》（李航） | 进阶 | 数学推导，中文经典 |
| 《The Elements of Statistical Learning》 | 高级 | 理论全面，数学深 |
| 《Pattern Recognition and ML》 | 高级 | 贝叶斯视角 |
| 《Python Machine Learning》 | 入门 | 代码驱动 |

### 17.3 在线课程

- Andrew Ng 的 Machine Learning（Coursera）：经典入门
- fast.ai：实战导向
- 李宏毅机器学习（B站）：中文，理论深

### 17.4 学习路径建议

```
入门：
  1. 跑通本教程的教程一（第一个模型）
  2. 理解 fit/predict 契约
  3. 用 cross_val_score 评估模型

进阶：
  4. 学会 Pipeline 和 GridSearchCV
  5. 理解预处理（标准化、编码）的作用
  6. 对比不同算法的表现

高级：
  7. 阅读架构文档，理解 BaseEstimator 设计
  8. 阅读算法源码，理解实现细节
  9. 尝试实现新算法或 C++ 扩展

专家：
  10. 对比 sklearn 源码，理解工业级考虑
  11. 研究数学推导（损失函数、优化算法）
  12. 贡献代码或写自己的机器学习库
```

### 17.5 相关文档

- [架构设计](../architecture/01-unified-api.md)：理解"为什么这么设计"
- [算法实现](../algorithms/index.md)：每个算法的原理和用法
- [Pipeline 详解](../algorithms/pipeline/index.md)：流水线深入
- [模型选择](../algorithms/model_selection/index.md)：交叉验证和网格搜索

### 17.6 社区资源

- [sklearn GitHub](https://github.com/scikit-learn/scikit-learn)：源码和 issue
- [Stack Overflow](https://stackoverflow.com/questions/tagged/scikit-learn)：问答
- [Cross Validated](https://stats.stackexchange.com/)：统计学理论问答

### 17.7 持续学习建议

1. **每个算法都动手实现一遍**：理解最深的办法
2. **读论文**：从经典论文开始（Random Forests、SVM 等）
3. **参加 Kaggle**：实战练手，看别人的方案
4. **写博客**：教是最好的学
5. **贡献开源**：给 sklearn 或本项目提 PR
