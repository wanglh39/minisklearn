# 算法实现

> 架构是骨架，算法是血肉。本章节逐一实现 sklearn 的核心算法，每个都回扣架构设计。

## 实现路线

### 阶段 2：预处理 + 基础指标

| 算法 | 类型 | 状态 |
|------|------|------|
| `StandardScaler` | Transformer | ✅ 已实现 |
| `MinMaxScaler` | Transformer | ✅ 已实现 |
| `LabelEncoder` | Transformer | ✅ 已实现 |
| `OneHotEncoder` | Transformer | ✅ 已实现 |

### 阶段 3：线性模型

| 算法 | 类型 | 状态 |
|------|------|------|
| `LinearRegression` | Regressor | ✅ 已实现 |
| `LogisticRegression` | Classifier | ✅ 已实现 |

### 阶段 4-8：进阶算法

| 算法 | 类型 | 状态 |
|------|------|------|
| `KNeighborsClassifier` / `KNeighborsRegressor` | Predictor | ✅ 已实现 |
| `DecisionTreeClassifier` / `DecisionTreeRegressor` | Tree | ✅ 已实现 |
| `RandomForestClassifier` / `RandomForestRegressor` | Ensemble | ✅ 已实现 |
| `KMeans` | Clusterer | ✅ 已实现 |
| `PCA` | Transformer | ✅ 已实现 |
| `LinearSVC` | Classifier | ✅ 已实现 |
| `GaussianNB` | Classifier | ✅ 已实现 |

### 阶段 9：元算法

| 算法 | 类型 | 状态 |
|------|------|------|
| `Pipeline` | Meta-Estimator | ✅ 已实现 |
| `GridSearchCV` | Meta-Estimator | ✅ 已实现 |
| `cross_val_score` | 工具函数 | ✅ 已实现 |
| `train_test_split` | 工具函数 | ✅ 已实现 |
| `KFold` | CV Splitter | ✅ 已实现 |

### 阶段 10：C++ 扩展 + 性能对比

| 内容 | 状态 |
|------|------|
| C++ 扩展（pybind11） | 待实现 |
| 三性能对比基准 | 待实现 |

---

## 每个算法的文档结构

每个算法配三篇文档：

- **设计文档**：为什么这么设计（架构层面）
- **原理推导**：数学原理 + 公式推导
- **使用教程**：手把手教程 + 与 sklearn 对比

---

## 算法分类总览

minisklearn 的算法按功能分为五大类，每类遵守不同的接口契约：

### 1. 转换器（Transformer）

**契约**：`fit(X)` → `transform(X)` → `fit_transform(X)`

**作用**：改变数据的表示，不改变样本数量。如标准化、降维、编码。

| 算法 | 模块 | 改变什么 |
|------|------|----------|
| `StandardScaler` | preprocessing | 特征的尺度（均值 0、标准差 1） |
| `MinMaxScaler` | preprocessing | 特征的范围（如 [0, 1]） |
| `LabelEncoder` | preprocessing | 标签的编码（类别 → 整数） |
| `OneHotEncoder` | preprocessing | 类别特征的编码（→ 二值向量） |
| `PCA` | decomposition | 特征的维度（高维 → 低维） |

### 2. 分类器（Classifier）

**契约**：`fit(X, y)` → `predict(X)` → `score(X, y)`（返回准确率）

**作用**：预测离散标签。`y` 是类别。

| 算法 | 模块 | 核心思想 |
|------|------|----------|
| `LogisticRegression` | linear_model | 线性决策边界 + sigmoid |
| `KNeighborsClassifier` | neighbors | 投票：邻居多数属于哪类 |
| `DecisionTreeClassifier` | tree | 树形决策：逐特征分裂 |
| `RandomForestClassifier` | ensemble | 多树投票 + bagging |
| `LinearSVC` | svm | 最大化间隔 + 铰链损失 |
| `GaussianNB` | naive_bayes | 贝叶斯定理 + 独立假设 |

### 3. 回归器（Regressor）

**契约**：`fit(X, y)` → `predict(X)` → `score(X, y)`（返回 R²）

**作用**：预测连续值。`y` 是实数。

| 算法 | 模块 | 核心思想 |
|------|------|----------|
| `LinearRegression` | linear_model | 最小化均方误差 |
| `KNeighborsRegressor` | neighbors | 平均：邻居的平均值 |
| `DecisionTreeRegressor` | tree | 树叶子的均值 |
| `RandomForestRegressor` | ensemble | 多树平均 + bagging |

### 4. 聚类器（Clusterer）

**契约**：`fit(X)` → `predict(X)` → `fit_predict(X)`（无标签，无监督）

**作用**：把样本分成若干组，没有真实标签。

| 算法 | 模块 | 核心思想 |
|------|------|----------|
| `KMeans` | cluster | 最小化样本到中心的距离平方和 |

### 5. 元估计器（Meta-Estimator）

**契约**：包装其他估计器，接口与被包装者一致

**作用**：组合、调参、评估。

| 算法 | 模块 | 作用 |
|------|------|------|
| `Pipeline` | pipeline | 串联多个步骤 |
| `GridSearchCV` | model_selection | 网格搜索调参 |
| `RandomForest*` | ensemble | 组合多棵树 |

---

## 每个算法详解

### StandardScaler —— 标准化

**一句话**：把每个特征变换到均值 0、标准差 1。

**公式**：`z = (x - μ) / σ`

**适用场景**：
- 特征量纲差异大（如年龄 0-100 vs 收入 0-1000000）
- 距离类算法（KNN、SVM、KMeans）——它们对量纲敏感
- 梯度下降类算法——加速收敛
- 假设数据正态分布的算法（如 LDA、GaussianNB）

**不适用**：
- 树模型——树对单调变换不敏感
- 数据有明确边界且想保留（如像素 [0, 255]）

**复杂度**：
- `fit`：O(n × d)，计算每列均值和标准差
- `transform`：O(n × d)，逐元素变换

**参数**：
- `with_mean`：是否减均值（默认 True，稀疏数据要设 False）
- `with_std`：是否除标准差（默认 True）

**学到的属性**（带下划线）：
- `mean_`：每列均值
- `scale_`：每列标准差
- `var_`：每列方差

**与 sklearn 差异**：本项目不支持稀疏矩阵，其余一致。

---

### MinMaxScaler —— 归一化

**一句话**：把每个特征变换到指定范围（默认 [0, 1]）。

**公式**：`z = (x - min) / (max - min) × (max_range - min_range) + min_range`

**适用场景**：
- 数据有明确边界（如像素、百分比）
- 神经网络输入（习惯 [0, 1] 或 [-1, 1]）
- 不假设正态分布的场景

**不适用**：
- 有离群点——会被拉扯，大部分数据挤在一起
- 距离类算法——如果量纲差异大，标准化更合适

**复杂度**：同 StandardScaler，O(n × d)。

**参数**：
- `feature_range`：目标范围，默认 (0, 1)

**学到的属性**：
- `min_`：每列最小值
- `scale_`：缩放因子
- `data_min_`、`data_max_`：原始数据的最小/最大值

---

### LabelEncoder —— 标签编码

**一句话**：把字符串/类别标签编码为 0, 1, 2, ... 的整数。

**公式**：按字典序排序后，第 i 个类别编码为 i。

**适用场景**：
- 分类器的目标标签（算法要整数，用户给字符串）
- 有序类别特征（低/中/高 → 0/1/2）

**不适用**：
- 无序类别特征（颜色/城市）——会引入虚假顺序，用 OneHotEncoder

**复杂度**：
- `fit`：O(n log n)，排序去重
- `transform`：O(n)，查表

**学到的属性**：
- `classes_`：排序后的类别列表

**示例**：
```python
enc = LabelEncoder()
enc.fit(['猫', '狗', '鸟'])
# enc.classes_ = ['鸟', '狗', '猫']  （字典序）
enc.transform(['猫', '狗'])  # [2, 1]
enc.inverse_transform([0, 1, 2])  # ['鸟', '狗', '猫']
```

---

### OneHotEncoder —— 独热编码

**一句话**：把每个类别编码为一个二值向量，只有对应位置是 1，其余是 0。

**公式**：类别 i（共 K 类）→ 长度 K 向量，第 i 位为 1。

**适用场景**：
- 无序类别特征（颜色、城市、性别）
- 线性模型、SVM、神经网络——它们不能直接处理类别

**不适用**：
- 类别数太多——会生成太多稀疏特征
- 树模型——树能直接处理 LabelEncoder 的结果

**复杂度**：
- `fit`：O(n × d)，找出每列的类别
- `transform`：O(n × d × K)，K 是平均类别数

**参数**：
- `handle_unknown`：遇到未知类别怎么处理（'error' 或 'ignore'）

**学到的属性**：
- `categories_`：每列的类别列表

---

### LinearRegression —— 线性回归

**一句话**：用线性函数拟合数据，最小化均方误差。

**模型**：`y = Xw + b`

**损失**：`L = (1/n) Σ (y_i - ŷ_i)²`

**求解方法**：
1. **正规方程**：`w = (X^T X)^{-1} X^T y`，一步到位，但要求矩阵可逆
2. **梯度下降**：`w ← w - η ∇L`，迭代逼近，适合大数据

**适用场景**：
- 目标和特征是线性关系
- 需要可解释性（权重表示特征重要性）
- 作为基线模型

**不适用**：
- 非线性关系（用多项式特征 + 线性回归，或用树/森林）
- 有强离群点（用 RANSAC 或 Huber 回归）

**复杂度**：
- 正规方程：O(n × d² + d³)，d 是特征数
- 梯度下降：O(n × d × iter)，iter 是迭代次数

**参数**：
- `method`：'normal'（正规方程）或 'gd'（梯度下降）
- `learning_rate`：梯度下降的学习率
- `max_iter`：最大迭代次数

**学到的属性**：
- `coef_`：权重向量
- `intercept_`：截距

**与 sklearn 差异**：sklearn 默认用 SVD 求解（数值更稳定），本项目用正规方程 + 梯度下降两种。

---

### LogisticRegression —— 逻辑回归

**一句话**：线性组合 + sigmoid，输出概率，用交叉熵训练。

**模型**：`P(y=1|x) = σ(Xw + b)`，其中 `σ(z) = 1/(1+e^{-z})`

**损失**（交叉熵）：`L = -(1/n) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]` + L2 正则

**求解**：梯度下降。梯度有漂亮的形式：`∇L = X^T (p - y) / n + λw`

**适用场景**：
- 二分类（多分类用 OvR 或 softmax）
- 需要概率输出（不只是标签）
- 需要可解释性
- 作为基线分类器

**不适用**：
- 非线性决策边界（用核 SVM、树、森林）
- 类别严重不平衡（调 class_weight 或用 F1 评估）

**复杂度**：O(n × d × iter)，iter 是迭代次数。

**参数**：
- `C`：正则强度的倒数，越大正则越弱（默认 1.0）
- `max_iter`：最大迭代次数
- `tol`：收敛阈值

**学到的属性**：
- `coef_`：权重向量
- `intercept_`：截距
- `n_iter_`：实际迭代次数

**与 sklearn 差异**：sklearn 默认用 LBFGS 优化器（更快更稳），本项目用纯梯度下降。

---

### KNeighborsClassifier / KNeighborsRegressor —— K 近邻

**一句话**：预测时找 K 个最近邻居，分类投票、回归平均。

**模型**：没有显式模型，"懒惰学习"。

**分类**：`ŷ = mode(邻居的 y)`
**回归**：`ŷ = mean(邻居的 y)`

**距离**：默认欧氏距离 `d(x, x') = √Σ(x_i - x'_i)²`

**适用场景**：
- 决策边界不规则
- 数据量不大（预测慢）
- 对量纲不敏感（但要先标准化！）

**不适用**：
- 数据量大——预测要遍历所有样本，O(n × d) 每次
- 特征数多——维度灾难，高维空间距离都差不多
- 实时要求高——预测慢

**复杂度**：
- `fit`：O(1)（只存数据）或 O(n log n)（建 KD-Tree）
- `predict`：O(n × d) 暴力，或 O(log n) KD-Tree

**参数**：
- `n_neighbors`：K 值
- `weights`：'uniform'（等权）或 'distance'（距离倒数）
- `algorithm`：'brute' 或 'kd_tree'

**学到的属性**：无（懒惰学习）

**与 sklearn 差异**：sklearn 支持 KD-Tree、Ball-Tree 等加速结构，本项目主要用暴力法。

---

### DecisionTreeClassifier / DecisionTreeRegressor —— 决策树

**一句话**：递归地选择最优特征和分裂点，把数据空间划分成区域。

**算法**：CART（Classification and Regression Trees）

**分类分裂准则**：
- **基尼系数**：`Gini = 1 - Σ p_i²`，越小越纯
- **熵**：`Entropy = -Σ p_i log p_i`
- 本项目默认用基尼

**回归分裂准则**：
- **均方误差**：分裂后两边的 MSE 加权和最小

**分裂过程**：
1. 遍历所有特征
2. 对每个特征，遍历所有可能分裂点
3. 计算分裂后的纯度增益
4. 选增益最大的分裂
5. 递归分裂子节点

**适用场景**：
- 非线性关系
- 需要可解释性（树可以可视化）
- 混合类型特征（数值 + 类别）
- 特征选择（树会自动选重要特征）

**不适用**：
- 线性关系——树要很多层才能逼近线性
- 对噪声敏感——单树容易过拟合，用随机森林
- 外推——树不能预测训练范围外的值

**复杂度**：
- `fit`：O(n × d × log n) 平均，O(n × d × n) 最坏
- `predict`：O(log n)，从根走到叶子

**参数**：
- `max_depth`：最大深度
- `min_samples_split`：分裂所需最小样本数
- `min_samples_leaf`：叶子最小样本数
- `criterion`：分裂准则（'gini' / 'entropy'）

**学到的属性**：
- `tree_`：树结构
- `feature_importances_`：特征重要性

---

### RandomForestClassifier / RandomForestRegressor —— 随机森林

**一句话**：训练多棵决策树，分类投票、回归平均。

**两个随机**：
1. **Bagging**：每棵树用有放回抽样的训练集
2. **特征随机**：每次分裂只考虑部分特征

**为什么有效**：
- 单树容易过拟合（方差大）
- 多树平均降低方差
- 特征随机让树多样化，避免所有树长一样

**适用场景**：
- 大多数场景——"当你不知道用什么，就用随机森林"
- 高维数据
- 混合类型特征
- 需要特征重要性

**不适用**：
- 线性关系——用线性模型更高效
- 实时预测——多树预测慢
- 极高维稀疏数据（如文本）——用线性模型 + 正则

**复杂度**：
- `fit`：O(B × n × d × log n)，B 是树数
- `predict`：O(B × log n)

**参数**：
- `n_estimators`：树的数量
- `max_depth`：每棵树的最大深度
- `max_features`：每次分裂考虑的特征数
- `bootstrap`：是否有放回抽样

**学到的属性**：
- `estimators_`：所有树的列表
- `feature_importances_`：特征重要性（所有树平均）

---

### KMeans —— K 均值聚类

**一句话**：把数据分成 K 个簇，每个簇用一个中心代表，最小化样本到中心的距离平方和。

**目标**：`min Σ Σ ||x - μ_k||²`（样本到所属簇中心的距离平方和）

**算法（EM 迭代）**：
1. **初始化**：随机选 K 个样本作为初始中心
2. **E 步（分配）**：每个样本分配到最近中心
3. **M 步（更新）**：每个中心更新为所属样本的均值
4. 重复 2-3 直到中心不再变或达到最大迭代

**适用场景**：
- 簇是球形（各方向方差相近）
- 簇大小相近
- 需要快速聚类

**不适用**：
- 簇非球形（用 DBSCAN）
- 簇大小差异大
- 有离群点（KMeans 很敏感）
- 不知道 K——要先确定 K

**复杂度**：O(n × K × d × iter)

**参数**：
- `n_clusters`：K 值
- `init`：初始化方法（'k-means++' 或 'random'）
- `max_iter`：最大迭代
- `random_state`：随机种子

**学到的属性**：
- `cluster_centers_`：簇中心
- `labels_`：每个样本的簇标签
- `inertia_`：样本到中心的距离平方和（越小越好）

**与 sklearn 差异**：sklearn 默认跑 10 次取最优（`n_init=10`），本项目默认 1 次。

---

### PCA —— 主成分分析

**一句话**：找到方差最大的方向，把数据投影到这些方向上，实现降维。

**思想**：方差大的方向信息多，方差小的方向是噪声。

**算法**：
1. **中心化**：减去均值
2. **计算协方差矩阵**：`C = X^T X / n`
3. **特征值分解**：`C v = λ v`
4. **选前 k 大特征值对应的特征向量**：作为主成分

**或用 SVD**：`X = U Σ V^T`，前 k 列 V 就是主成分。SVD 更数值稳定。

**适用场景**：
- 高维数据可视化（降到 2-3 维）
- 特征压缩
- 去噪（丢掉小方差方向）
- 加速后续算法

**不适用**：
- 非线性结构（用 t-SNE、UMAP）
- 簇结构（PCA 可能合并不同簇）

**复杂度**：O(n × d² + d³)（协方差矩阵 + 特征分解）

**参数**：
- `n_components`：保留的主成分数
- `whiten`：是否白化（让各主成分方差相同）

**学到的属性**：
- `components_`：主成分（特征向量）
- `explained_variance_`：各主成分的方差
- `explained_variance_ratio_`：方差占比
- `mean_`：均值（用于中心化）

---

### LinearSVC —— 线性支持向量分类

**一句话**：找到最大化间隔的超平面，用铰链损失训练。

**模型**：`f(x) = sign(w^T x + b)`

**损失**（铰链损失）：`L = (1/n) Σ max(0, 1 - y_i f(x_i))` + λ||w||²

**与逻辑回归对比**：
- 逻辑回归用对数损失，关心所有样本的概率
- SVM 用铰链损失，只关心间隔边界上的样本（支持向量）
- SVM 的决策边界只由支持向量决定，更鲁棒

**适用场景**：
- 线性可分或近似线性可分
- 高维数据（如文本分类）
- 需要稀疏解（SVM 的 w 通常稀疏）

**不适用**：
- 非线性决策边界（用核 SVM，但本项目未实现）
- 需要概率输出（用逻辑回归）

**复杂度**：O(n × d × iter)

**参数**：
- `C`：正则强度的倒数
- `max_iter`：最大迭代

**学到的属性**：
- `coef_`：权重
- `intercept_`：截距

---

### GaussianNB —— 高斯朴素贝叶斯

**一句话**：用贝叶斯定理 + 特征独立假设 + 高斯似然做分类。

**模型**：`P(y|x) ∝ P(y) Π P(x_i|y)`，其中 `P(x_i|y) ~ N(μ_{i,y}, σ_{i,y}²)`

**"朴素"在哪**：假设特征条件独立 `P(x|y) = Π P(x_i|y)`。这个假设通常不成立，但效果出奇地好。

**适用场景**：
- 特征近似独立
- 数据量小（NB 参数少，不易过拟合）
- 需要快速基线
- 文本分类（用 MultinomialNB，本项目未实现）

**不适用**：
- 特征强相关——独立性假设失效
- 需要精确概率——NB 的概率估计偏差大，但分类决策通常 OK

**复杂度**：
- `fit`：O(n × d)，算每类每特征的均值方差
- `predict`：O(n × d × K)

**参数**：
- `var_smoothing`：方差平滑（避免方差为 0）

**学到的属性**：
- `theta_`：每类每特征的均值
- `var_`：每类每特征的方差
- `class_prior_`：类的先验概率

---

## 算法选择指南

### 按任务类型选

#### 分类任务

```
数据量小？
├── 是 → GaussianNB（最快，参数少）
└── 否 → 
    线性关系？
    ├── 是 → LogisticRegression（可解释，快）
    │       └── 高维稀疏？→ LinearSVC
    └── 否 → 
        数据量中？
        ├── 是 → DecisionTreeClassifier（可解释）
        └── 否 → RandomForestClassifier（通常最好）
```

#### 回归任务

```
线性关系？
├── 是 → LinearRegression（可解释，快）
└── 否 → 
    数据量小？
    ├── 是 → KNeighborsRegressor（简单）
    └── 否 → RandomForestRegressor（通常最好）
```

#### 聚类任务

```
知道 K？
├── 是 → KMeans（球形簇）
└── 否 → （本项目暂无，sklearn 用 DBSCAN / MeanShift）
```

#### 降维任务

```
线性结构？
├── 是 → PCA
└── 否 → （本项目暂无，sklearn 用 t-SNE / UMAP）
```

### 按数据特征选

| 数据特征 | 推荐算法 |
|----------|----------|
| 小数据 + 快速基线 | GaussianNB、LogisticRegression |
| 高维稀疏（文本） | LinearSVC、LogisticRegression |
| 非线性 + 可解释 | DecisionTree |
| 非线性 + 高精度 | RandomForest |
| 混合类型特征 | DecisionTree、RandomForest |
| 有离群点 | RandomForest（比线性鲁棒） |
| 实时预测 | LogisticRegression、LinearSVC（快） |
| 需要概率 | LogisticRegression、GaussianNB |
| 需要特征重要性 | RandomForest、DecisionTree |

### 按评估重点选

| 评估重点 | 推荐 |
|----------|------|
| 准确率 | RandomForest 通常最高 |
| 可解释性 | DecisionTree、LinearRegression |
| 训练速度 | GaussianNB > LogisticRegression > DecisionTree > RandomForest |
| 预测速度 | LogisticRegression > LinearSVC > RandomForest > KNN |
| 内存 | LogisticRegression < DecisionTree < RandomForest < KNN |

---

## 与 sklearn 的 API 对比表

### 通用 API（所有估计器）

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `fit(X, y)` | ✅ | ✅ | ✅ |
| `predict(X)` | ✅ | ✅ | ✅ |
| `score(X, y)` | ✅ | ✅ | ✅ |
| `get_params()` | ✅ | ✅ | ✅ |
| `set_params(**params)` | ✅ | ✅ | ✅ |
| `__repr__()` | ✅ | ✅ | ✅ |

### 转换器 API

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `transform(X)` | ✅ | ✅ | ✅ |
| `fit_transform(X, y)` | ✅ | ✅ | ✅ |
| `inverse_transform(X)` | 部分 | ✅ | ⚠️ |

### 分类器 API

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `predict_proba(X)` | 部分 | ✅ | ⚠️ |
| `predict_log_proba(X)` | ❌ | ✅ | ❌ |
| `decision_function(X)` | ❌ | 部分 | ❌ |
| `classes_` | ✅ | ✅ | ✅ |

### 回归器 API

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `coef_` | ✅ | 部分 | ✅ |
| `intercept_` | ✅ | 部分 | ✅ |

### 聚类器 API

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `fit_predict(X)` | ✅ | ✅ | ✅ |
| `labels_` | ✅ | ✅ | ✅ |
| `cluster_centers_` | ✅ | 部分 | ✅ |

### 元估计器 API

| 方法 | minisklearn | sklearn | 一致？ |
|------|-------------|---------|--------|
| `Pipeline(steps)` | ✅ | ✅ | ✅ |
| `GridSearchCV(est, param_grid)` | ✅ | ✅ | ✅ |
| `best_params_` | ✅ | ✅ | ✅ |
| `best_score_` | ✅ | ✅ | ✅ |
| `cv_results_` | ✅ | ✅ | ✅ |

### 参数对比

#### LogisticRegression

| 参数 | minisklearn | sklearn | 说明 |
|------|-------------|---------|------|
| `C` | ✅ | ✅ | 一致 |
| `max_iter` | ✅ | ✅ | 一致 |
| `tol` | ✅ | ✅ | 一致 |
| `penalty` | 'l2' | 'l2'/'l1'/'elasticnet' | 本项目只支持 L2 |
| `solver` | 'gd' | 'lbfgs'/'sag'/... | 本项目只用梯度下降 |
| `n_jobs` | ❌ | ✅ | 本项目不并行 |

#### KMeans

| 参数 | minisklearn | sklearn | 说明 |
|------|-------------|---------|------|
| `n_clusters` | ✅ | ✅ | 一致 |
| `init` | ✅ | ✅ | 一致 |
| `max_iter` | ✅ | ✅ | 一致 |
| `n_init` | 1 | 10 | sklearn 默认跑 10 次取最优 |
| `algorithm` | 'full' | 'full'/'elkan' | 本项目只有一种 |

---

## 实现状态总览

### 已实现（17 个核心算法 + 5 个工具）

| 模块 | 算法/工具 | 状态 | 完整度 |
|------|-----------|------|--------|
| preprocessing | StandardScaler | ✅ | 95% |
| preprocessing | MinMaxScaler | ✅ | 95% |
| preprocessing | LabelEncoder | ✅ | 95% |
| preprocessing | OneHotEncoder | ✅ | 90% |
| linear_model | LinearRegression | ✅ | 95% |
| linear_model | LogisticRegression | ✅ | 90% |
| neighbors | KNeighborsClassifier | ✅ | 90% |
| neighbors | KNeighborsRegressor | ✅ | 90% |
| tree | DecisionTreeClassifier | ✅ | 90% |
| tree | DecisionTreeRegressor | ✅ | 90% |
| ensemble | RandomForestClassifier | ✅ | 90% |
| ensemble | RandomForestRegressor | ✅ | 90% |
| cluster | KMeans | ✅ | 90% |
| decomposition | PCA | ✅ | 95% |
| svm | LinearSVC | ✅ | 85% |
| naive_bayes | GaussianNB | ✅ | 95% |
| model_selection | train_test_split | ✅ | 95% |
| model_selection | KFold | ✅ | 95% |
| model_selection | cross_val_score | ✅ | 95% |
| model_selection | GridSearchCV | ✅ | 90% |
| pipeline | Pipeline | ✅ | 95% |
| metrics | 6 个指标 | ✅ | 95% |

### 待实现

| 模块 | 算法 | 优先级 | 说明 |
|------|------|--------|------|
| ensemble | GradientBoostingClassifier | 高 | GBDT |
| ensemble | AdaBoostClassifier | 中 | 提升方法 |
| decomposition | TruncatedSVD | 中 | 稀疏数据降维 |
| decomposition | LatentDirichletAllocation | 低 | 主题模型 |
| cluster | DBSCAN | 高 | 密度聚类 |
| cluster | AgglomerativeClustering | 中 | 层次聚类 |
| svm | SVC（核方法） | 中 | 非线性 SVM |
| naive_bayes | MultinomialNB | 中 | 文本分类 |
| naive_bayes | BernoulliNB | 低 | 二值特征 |
| neighbors | RadiusNeighborsClassifier | 低 | 基于半径 |
| linear_model | Ridge | 高 | L2 正则回归 |
| linear_model | Lasso | 高 | L1 正则回归 |
| linear_model | ElasticNet | 中 | L1 + L2 |
| preprocessing | PolynomialFeatures | 高 | 多项式特征 |
| preprocessing | Normalizer | 中 | 样本归一化 |
| preprocessing | Binarizer | 低 | 二值化 |
| model_selection | RandomizedSearchCV | 中 | 随机搜索 |
| model_selection | StratifiedKFold | 高 | 分层 K 折 |
| compose | ColumnTransformer | 高 | 列转换器 |
| metrics | roc_auc_score | 高 | AUC |
| metrics | confusion_matrix | 高 | 混淆矩阵 |
| metrics | classification_report | 高 | 分类报告 |

### 不会实现（超出教学范围）

- 稀疏矩阵完整支持
- joblib 并行
- Cython 加速（用 pybind11 替代）
- 在线学习（partial_fit）
- 多输出
- 复杂的核方法（除线性外）

---

## 学习路线建议

### 按难度排序

#### 入门级（理解 fit/predict 契约）

1. **StandardScaler** —— 最简单的转换器，理解 `fit` + `transform`
2. **LabelEncoder** —— 简单的查表逻辑
3. **LinearRegression** —— 最简单的预测器，理解 `fit` + `predict`
4. **GaussianNB** —— 概率论入门，参数少

#### 进阶级（理解算法核心）

5. **LogisticRegression** —— 梯度下降入门，理解损失函数
6. **KNeighborsClassifier** —— 懒惰学习，理解距离计算
7. **KMeans** —— EM 算法入门，理解迭代优化
8. **PCA** —— 线性代数应用，理解特征值分解

#### 高级（理解递归和组合）

9. **DecisionTreeClassifier** —— 递归分裂，理解树结构
10. **RandomForestClassifier** —— 组合多树，理解 bagging
11. **LinearSVC** —— 优化问题，理解铰链损失

#### 专家级（理解元估计器）

12. **Pipeline** —— 组合模式，理解接口契约
13. **GridSearchCV** —— 反射 + 克隆，理解参数管理

### 按数学背景排序

#### 只要会线性代数

- LinearRegression（矩阵求逆）
- PCA（特征值分解）
- StandardScaler（均值方差）

#### 还要会微积分

- LogisticRegression（梯度下降）
- LinearSVC（梯度下降）

#### 还要会概率论

- GaussianNB（贝叶斯定理）
- LogisticRegression（交叉熵）

#### 还要会信息论

- DecisionTree（熵、基尼）

### 按代码量排序

| 算法 | 核心代码行数 | 难度 |
|------|-------------|------|
| StandardScaler | ~30 | ⭐ |
| LabelEncoder | ~40 | ⭐ |
| MinMaxScaler | ~30 | ⭐ |
| LinearRegression | ~50 | ⭐⭐ |
| GaussianNB | ~60 | ⭐⭐ |
| KMeans | ~80 | ⭐⭐⭐ |
| LogisticRegression | ~100 | ⭐⭐⭐ |
| PCA | ~80 | ⭐⭐⭐ |
| KNeighbors | ~90 | ⭐⭐⭐ |
| LinearSVC | ~100 | ⭐⭐⭐ |
| DecisionTree | ~200 | ⭐⭐⭐⭐ |
| RandomForest | ~150 | ⭐⭐⭐⭐ |
| Pipeline | ~100 | ⭐⭐⭐⭐ |
| GridSearchCV | ~150 | ⭐⭐⭐⭐⭐ |

---

## 算法对比速查表

### 训练速度（从快到慢）

| 算法 | 训练复杂度 | 实测（10k 样本，10 特征） |
|------|-----------|--------------------------|
| GaussianNB | O(n × d) | ~0.01s |
| LinearRegression（正规方程） | O(n × d²) | ~0.05s |
| StandardScaler | O(n × d) | ~0.01s |
| LogisticRegression | O(n × d × iter) | ~0.5s |
| LinearSVC | O(n × d × iter) | ~0.5s |
| KMeans | O(n × K × d × iter) | ~1s |
| PCA | O(n × d² + d³) | ~0.1s |
| KNeighbors（fit） | O(1) | ~0.01s |
| DecisionTree | O(n × d × log n) | ~0.5s |
| RandomForest | O(B × n × d × log n) | ~5s |

### 预测速度（从快到慢）

| 算法 | 预测复杂度 | 实测（1k 样本） |
|------|-----------|----------------|
| LinearRegression | O(n × d) | ~0.001s |
| LogisticRegression | O(n × d) | ~0.001s |
| LinearSVC | O(n × d) | ~0.001s |
| GaussianNB | O(n × d × K) | ~0.005s |
| DecisionTree | O(n × log n) | ~0.01s |
| RandomForest | O(B × log n) | ~0.1s |
| KMeans（predict） | O(n × K × d) | ~0.05s |
| KNeighbors | O(n_train × d) | ~0.5s |

### 内存占用（从少到多）

| 算法 | 存储什么 | 内存 |
|------|----------|------|
| LinearRegression | w, b | O(d) |
| LogisticRegression | w, b | O(d) |
| GaussianNB | μ, σ, prior | O(d × K) |
| KMeans | centers | O(K × d) |
| DecisionTree | 树节点 | O(nodes) |
| RandomForest | B 棵树 | O(B × nodes) |
| KNeighbors | 全部训练数据 | O(n × d) |

---

## 每个算法的回扣架构

本项目的核心思想：**架构是骨架，算法是血肉**。每个算法都回扣架构：

### 预处理算法回扣

```python
class StandardScaler(BaseEstimator, TransformerMixin):
    # 继承 BaseEstimator：获得 get_params/set_params/clone/__repr__
    # 继承 TransformerMixin：获得 fit_transform
    # 自己只需写 fit 和 transform
```

**回扣点**：为什么不用大基类？因为 `fit_transform` 只有转换器需要，分类器不需要。Mixin 让每个类只混入需要的功能。

### 线性模型回扣

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    # 继承 BaseEstimator：参数管理
    # 继承 ClassifierMixin：score = accuracy
    # 自己写 fit（梯度下降）和 predict（sigmoid）
```

**回扣点**：`score` 方法是 Mixin 给的，不用自己写。`GridSearchCV` 能调参，靠的是 `BaseEstimator` 的 `get_params`/`set_params`/`clone`。

### KNN 回扣

```python
class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self._fit_X = check_array(X)  # 只存数据，不训练
        return self
    def predict(self, X):
        check_is_fitted(self)  # 检查是否 fit 过
        # 计算距离，投票
```

**回扣点**：`check_is_fitted` 通过查找 `_fit_X`（带下划线）判断是否拟合。懒惰学习也要遵守契约。

### 决策树回扣

```python
class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)  # 数据校验
        self.tree_ = self._build_tree(X, y)  # 递归建树
        return self
```

**回扣点**：`check_X_y` 统一数据校验。`tree_` 带下划线，表示学到的属性。

### 随机森林回扣

```python
class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, ...):
        self.n_estimators = n_estimators  # 只存参数，不创建树
    def fit(self, X, y):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(...)  # 在 fit 里创建
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)
```

**回扣点**：`__init__` 不创建树（否则 `clone` 会带着树）。树在 `fit` 里创建，这是"组合优于继承"的体现——随机森林**持有**树，不**继承**树。

### KMeans 回扣

```python
class KMeans(BaseEstimator, ClusterMixin):
    # 继承 ClusterMixin：获得 fit_predict
    def fit(self, X):
        X = check_array(X)  # 无监督，没有 y
        # EM 迭代
        return self
```

**回扣点**：聚类器没有 `y`，用 `check_array` 而非 `check_X_y`。`ClusterMixin` 给 `fit_predict`。

### Pipeline 回扣

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # 持有其他估计器
    def fit(self, X, y):
        for step in self.steps[:-1]:
            X = step.fit_transform(X, y)  # 中间步骤转换
        self.steps[-1].fit(X, y)  # 最后一步预测
        return self
```

**回扣点**：Pipeline 不关心 `steps` 里是什么算法，只要遵守 `fit_transform`/`fit` 契约。这是"针对接口编程，而非针对实现"。

### GridSearchCV 回扣

```python
class GridSearchCV(BaseEstimator):
    def fit(self, X, y):
        for params in self._param_grid:
            for train_idx, val_idx in self.cv.split(X):
                est = clone(self.estimator)  # 克隆出干净对象
                est.set_params(**params)  # 设置参数
                est.fit(X[train_idx], y[train_idx])
                score = est.score(X[val_idx], y[val_idx])
            # 记录平均分
        # 用最佳参数 refit 全部数据
```

**回扣点**：`clone` + `set_params` 是元估计器的基石。没有 `BaseEstimator` 的参数管理，`GridSearchCV` 几乎无法实现。

---

## 算法实现的通用模式

每个算法的实现都遵循同一模式：

```python
class XXXClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=default1, param2=default2):
        # 1. 只存参数，不做计算
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        # 2. 数据校验
        X, y = check_X_y(X, y)
        
        # 3. 核心算法（学到的属性带下划线）
        self.coef_ = ...
        self.intercept_ = ...
        
        # 4. 存元信息
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # 5. 返回 self（链式调用）
        return self
    
    def predict(self, X):
        # 6. 检查是否拟合
        check_is_fitted(self)
        
        # 7. 数据校验
        X = check_array(X)
        
        # 8. 用学到的属性预测
        return ...
```

这个模式保证：
- **参数管理**：`__init__` 只存参数，`clone` 能工作
- **数据校验**：统一用 `check_X_y`/`check_array`
- **拟合检查**：`check_is_fitted` 通过下划线属性判断
- **链式调用**：`fit` 返回 `self`
- **Mixin 复用**：`score` 由 Mixin 提供

---

## 数学符号约定

本章节所有算法文档使用统一的数学符号：

| 符号 | 含义 |
|------|------|
| `X` | 特征矩阵，形状 (n_samples, n_features) |
| `y` | 标签向量，形状 (n_samples,) |
| `x_i` | 第 i 个样本（行向量） |
| `x_{ij}` | 第 i 个样本的第 j 个特征 |
| `w` | 权重向量 |
| `b` | 截距 |
| `n` | 样本数 |
| `d` | 特征数 |
| `K` | 类别数 / 簇数 |
| `μ` | 均值 |
| `σ` | 标准差 |
| `Σ` | 求和 |
| `∇` | 梯度 |
| `||·||` | L2 范数 |
| `σ(z)` | sigmoid 函数 1/(1+e^{-z}) |

---

## 下一步

- **想动手用**：看 [教程](../tutorials/index.md)
- **想看架构**：看 [架构设计](../architecture/01-unified-api.md)
- **想看具体算法原理**：看 `docs/algorithms/` 下的各算法文档（待编写）
- **想看源码**：`minisklearn/` 下每个模块都有详细注释

---

## 算法文档编写计划

每个算法将配独立的详细文档：

| 文件 | 算法 | 内容 | 状态 |
|------|------|------|------|
| `preprocessing/standard-scaler.md` | StandardScaler | 设计 + 原理 + 教程 | 待编写 |
| `preprocessing/minmax-scaler.md` | MinMaxScaler | 设计 + 原理 + 教程 | 待编写 |
| `preprocessing/label-encoder.md` | LabelEncoder | 设计 + 原理 + 教程 | 待编写 |
| `preprocessing/onehot-encoder.md` | OneHotEncoder | 设计 + 原理 + 教程 | 待编写 |
| `linear-model/linear-regression.md` | LinearRegression | 设计 + 原理 + 教程 | 待编写 |
| `linear-model/logistic-regression.md` | LogisticRegression | 设计 + 原理 + 教程 | 待编写 |
| `neighbors/knn.md` | KNN | 设计 + 原理 + 教程 | 待编写 |
| `tree/decision-tree.md` | DecisionTree | 设计 + 原理 + 教程 | 待编写 |
| `ensemble/random-forest.md` | RandomForest | 设计 + 原理 + 教程 | 待编写 |
| `cluster/kmeans.md` | KMeans | 设计 + 原理 + 教程 | 待编写 |
| `decomposition/pca.md` | PCA | 设计 + 原理 + 教程 | 待编写 |
| `svm/linear-svc.md` | LinearSVC | 设计 + 原理 + 教程 | 待编写 |
| `naive-bayes/gaussian-nb.md` | GaussianNB | 设计 + 原理 + 教程 | 待编写 |
| `pipeline/pipeline.md` | Pipeline | 设计 + 原理 + 教程 | 待编写 |
| `model-selection/grid-search.md` | GridSearchCV | 设计 + 原理 + 教程 | 待编写 |

每个文档结构：
1. **一句话简介**
2. **设计文档**：架构层面为什么这么设计
3. **原理推导**：数学原理 + 公式推导
4. **实现详解**：逐步骤讲代码
5. **使用教程**：可运行示例
6. **与 sklearn 对比**：API 和结果差异
7. **常见问题**：FAQ
8. **练习题**：巩固理解

---

## 总结

本算法索引页涵盖了 minisklearn 的所有算法：

- **17 个核心算法**：预处理 4 + 线性模型 2 + KNN 2 + 树 2 + 森林 2 + 聚类 1 + 降维 1 + SVM 1 + 朴素贝叶斯 1
- **5 个工具**：train_test_split + KFold + cross_val_score + GridSearchCV + Pipeline
- **6 个评估指标**：accuracy/precision/recall/f1/mse/r2

核心思想：
1. **统一 API**：所有算法遵守 fit/predict/transform 契约
2. **Mixin 组合**：用多继承组合身份，而非大基类
3. **元估计器**：Pipeline/GridSearchCV 用组合包装其他算法
4. **参数管理**：反射机制支撑 clone/set_params/get_params

理解了架构，每个算法只需要写 `fit` 和 `predict`——这就是 sklearn 设计的胜利，也是本项目的核心教学目标。

---

## 深入技术分析：算法的数学统一视角

### 线性模型族的统一形式

minisklearn 的线性模型（LinearRegression、LogisticRegression、LinearSVC）可以用统一的框架理解：

$$
\min_{w, b} \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i)) + \lambda \Omega(w)
$$

其中 $f(x) = w^T x + b$，$L$ 是损失函数，$\Omega$ 是正则项。

| 算法 | 损失 $L$ | 正则 $\Omega$ | 求解 |
|------|----------|---------------|------|
| LinearRegression | $\frac{1}{2}(y - f)^2$ | 无 | 正规方程 / GD |
| LogisticRegression | $\log(1 + e^{-yf})$ | $\frac{1}{2C}\|w\|^2$ | GD |
| LinearSVC | $\max(0, 1 - yf)$ | $\frac{1}{2C}\|w\|^2$ | GD |

这揭示了它们的本质差异：
- **平方损失**对离群点敏感（误差被平方放大）
- **对数损失**关心所有样本的概率（平滑）
- **铰链损失**只关心间隔边界上的样本（稀疏支持向量）

### 树模型与集成的关系

决策树是"贪心递归划分"，随机森林是"bagging + 特征随机"：

```
单棵树：高方差，低偏差（深树能拟合任何训练数据）
随机森林：B 棵树平均 → 方差降为 σ²/B（理想情况）
```

数学上，若 B 棵树完全独立，方差降为 $\sigma^2/B$；但树之间相关（共享数据），实际方差为 $\rho\sigma^2 + (1-\rho)\sigma^2/B$，$\rho$ 是树间相关。特征随机降低 $\rho$，是随机森林有效的关键。

### 聚类与降维的优化视角

KMeans 和 PCA 都是优化问题：

| 算法 | 目标 | 约束 |
|------|------|------|
| KMeans | $\min \sum \|x - \mu_k\|^2$ | 簇中心 $\mu_k$ |
| PCA | $\max \text{Var}(Xw)$ | $\|w\| = 1$ |

KMeans 是非凸的（有局部最优），PCA 是凸的（有全局最优）。这解释了为什么 KMeans 对初始化敏感而 PCA 不敏感。

### 朴素贝叶斯与逻辑回归的关系

两者都用于分类，但假设不同：

- **朴素贝叶斯**：假设特征条件独立 $P(x|y) = \prod P(x_i|y)$，强假设，参数少
- **逻辑回归**：假设 $P(y|x)$ 是 sigmoid 线性形式，弱假设，参数多

当特征确实独立时，朴素贝叶斯更准（方差小）；当特征相关时，逻辑回归更准（偏差小）。数据量大时逻辑回归通常更好。

---

## 对比实验：算法选择的影响

### 实验：不同算法在同一数据集上的表现

```python
import numpy as np
from minisklearn.model_selection import cross_val_score, train_test_split
from minisklearn.linear_model import LogisticRegression, LinearRegression
from minisklearn.neighbors import KNeighborsClassifier
from minisklearn.tree import DecisionTreeClassifier
from minisklearn.ensemble import RandomForestClassifier
from minisklearn.svm import LinearSVC
from minisklearn.naive_bayes import GaussianNB

# 生成非线性可分数据
rng = np.random.RandomState(42)
X = rng.randn(500, 4)
y = (X[:, 0]**2 + X[:, 1]**2 - X[:, 2] > 1).astype(int)  # 非线性边界

models = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
    'LinearSVC': LinearSVC(max_iter=500),
    'GaussianNB': GaussianNB(),
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 预期结果分析

| 算法 | 适合此数据？ | 原因 |
|------|-------------|------|
| LogisticRegression | 差 | 决策边界非线性 |
| KNN | 好 | 能拟合非线性边界 |
| DecisionTree | 好 | 能拟合非线性边界 |
| RandomForest | 最好 | 非线性 + 低方差 |
| LinearSVC | 差 | 线性边界 |
| GaussianNB | 中 | 独立假设部分成立 |

### 实验：数据规模对算法的影响

```python
import time

for n in [100, 1000, 10000]:
    X = rng.randn(n, 10)
    y = (X.sum(axis=1) > 0).astype(int)
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X, y)
        t_fit = time.time() - t0
        t0 = time.time()
        clf.predict(X)
        t_pred = time.time() - t0
        print(f"n={n}, {name}: fit={t_fit:.3f}s, pred={t_pred:.3f}s")
```

观察：
- KNN 的 fit 几乎为 0，但 predict 随 n 线性增长
- RandomForest 的 fit 随 n 和树数增长
- 线性模型的 fit 和 predict 都很快

---

## 参数调优指南：各算法的关键参数

### LogisticRegression

```python
# 关键参数：C（正则强度的倒数）
# C 大 → 弱正则 → 可能过拟合
# C 小 → 强正则 → 可能欠拟合
param_grid = {'C': np.logspace(-3, 3, 7)}  # 对数尺度搜索
```

调参经验：从 `C=1` 开始，用对数尺度搜索 `[0.001, 1000]`。

### KNeighborsClassifier

```python
# 关键参数：n_neighbors（K）
# K 小 → 复杂模型 → 过拟合
# K 大 → 简单模型 → 欠拟合
param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15, 20]}
```

调参经验：K 通常取 $\sqrt{n}$ 附近，奇数避免平票。

### DecisionTreeClassifier

```python
# 关键参数：max_depth
# max_depth 大 → 过拟合
# max_depth 小 → 欠拟合
param_grid = {'max_depth': [3, 5, 7, 10, 15, None]}
```

调参经验：从 `max_depth=3` 开始，逐步增加直到验证分数下降。

### RandomForestClassifier

```python
# 关键参数：n_estimators, max_depth, max_features
param_grid = {
    'n_estimators': [50, 100, 200],    # 树数，越多越好但越慢
    'max_depth': [5, 10, None],         # 树深
    'max_features': ['sqrt', 'log2'],   # 分裂时考虑的特征数
}
```

调参经验：`n_estimators` 越多越好（但收敛），`max_features='sqrt'` 是常用默认。

### KMeans

```python
# 关键参数：n_clusters（K）
# 用肘部法则选 K
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertias.append(km.inertia_)
# 画 inertias vs k，找"肘部"
```

### PCA

```python
# 关键参数：n_components
# 用解释方差比选
pca = PCA().fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comp = np.argmax(cumvar >= 0.95) + 1  # 保留 95% 方差
```

---

## 常见错误与调试技巧

### 错误：忘记标准化

```python
# KNN 对量纲敏感
X = np.array([[1, 1000], [2, 2000], [3, 3000]])  # 第二列数值大
knn = KNeighborsClassifier().fit(X, y)
# 距离被第二列主导，第一列几乎无影响
# 解决：先 StandardScaler
```

### 错误：数据形状不对

```python
# sklearn 要求 X 是 2D，y 是 1D
X = np.random.randn(100)  # 1D，会报错
# 正确
X = np.random.randn(100, 1)  # 2D
```

### 错误：标签类型不对

```python
y = ['cat', 'dog', 'bird']  # 字符串标签
# 大多数算法要整数，用 LabelEncoder
y_encoded = LabelEncoder().fit_transform(y)
```

### 调试技巧：检查数据

```python
print(f"X 形状: {X.shape}, dtype: {X.dtype}")
print(f"y 形状: {y.shape}, 类别: {np.unique(y)}, 分布: {np.bincount(y)}")
print(f"X 有 NaN: {np.isnan(X).any()}, 有 Inf: {np.isinf(X).any()}")
print(f"X 范围: [{X.min():.2f}, {X.max():.2f}]")
```

### 调试技巧：对比训练和测试分数

```python
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"训练: {train_score:.4f}, 测试: {test_score:.4f}")
if train_score - test_score > 0.1:
    print("过拟合！")
elif train_score < 0.6:
    print("欠拟合！")
```

---

## 实际应用场景

### 场景：分类任务全流程

```python
# 1. 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. 标准化 + 模型
pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])

# 3. 调参
grid = GridSearchCV(pipe, {'clf__n_estimators': [50, 100, 200]}, cv=5)
grid.fit(X_train, y_train)

# 4. 评估
print(f"测试准确率: {grid.score(X_test, y_test):.2%}")
```

### 场景：回归任务

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', LinearRegression()),
])
pipe.fit(X_train, y_train)
print(f"R²: {pipe.score(X_test, y_test):.4f}")
```

### 场景：聚类分析

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
# 分析各簇特征
for k in range(3):
    print(f"簇 {k}: 均值={X[labels==k].mean(axis=0)}")
```

### 场景：降维可视化

```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
```

---

## 思考题与练习

### 基础题

1. **简答题**：StandardScaler 和 MinMaxScaler 各适合什么场景？

2. **简答题**：为什么 KNN 预测慢但训练快？

3. **代码题**：用 LogisticRegression 对鸢尾花数据分类，打印准确率。

4. **判断题**：随机森林的树越多越好，没有副作用。（错，计算成本增加）

### 进阶题

5. **分析题**：决策树和随机森林，哪个更容易过拟合？为什么？

6. **代码题**：用肘部法则为 KMeans 选择 K，画出 inertia 随 K 变化的图。

7. **调试题**：KNN 在未标准化的数据上效果差，为什么？如何解决？

8. **设计题**：对于文本分类（高维稀疏），你会选哪些算法？为什么？

### 高级题

9. **推导题**：推导逻辑回归的梯度 $\nabla L = X^T(p - y)/n + \lambda w$。

10. **实验题**：设计实验比较 L1 和 L2 正则的稀疏性差异（虽然本项目只支持 L2）。

11. **源码题**：阅读 DecisionTreeClassifier 源码，找出分裂准则（基尼）是如何计算的。

12. **扩展题**：实现一个简化版的 AdaBoost，用决策树桩作为弱学习器。

---

## 扩展阅读

### 官方文档

- [sklearn 算法选择指南](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [sklearn 各算法 API](https://scikit-learn.org/stable/modules/classes.html)

### 推荐书籍

- 《The Elements of Statistical Learning》：算法的数学基础
- 《Pattern Recognition and Machine Learning》：贝叶斯视角
- 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》：实战
- 《统计学习方法》（李航）：中文经典

### 相关文档

- [Pipeline 详解](./pipeline/index.md)：串联算法
- [模型选择](./model_selection/index.md)：调参和评估
- [教程](../tutorials/index.md)：动手实践

### 进阶主题

- **梯度提升树**（GBDT）：比随机森林更强的集成方法
- **核方法**：非线性 SVM 的基础
- **深度学习**：神经网络的统一框架
- **在线学习**：`partial_fit` 增量更新

### 推荐论文

- "Random Forests"（Breiman, 2001）：随机森林原始论文
- "A Decision-Theoretic Generalization of On-Line Learning"（AdaBoost）
- "Support-Vector Networks"（Cortes & Vapnik, 1995）：SVM 原始论文
