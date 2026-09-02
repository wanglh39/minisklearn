# K 近邻：KNN 分类与回归

> KNN 是最简单的机器学习算法——没有训练过程，预测时直接找最近邻。简单到极致，却揭示了"惰性学习"的本质。本章将从算法原理、距离计算、向量化优化、加权投票、几何直觉、复杂度分析、对比 sklearn、常见陷阱等多维度，把 KNN 讲透。

---

## 一、算法原理

### 1.1 核心思想

> "物以类聚"——一个样本的类别由它周围最近的 K 个邻居决定。

**分类**：K 个近邻中多数类为预测结果
**回归**：K 个近邻的目标值平均为预测结果

#### 1.1.1 形式化定义

给定训练集 $D = \{(x_i, y_i)\}_{i=1}^n$、查询点 $x$、距离函数 $d$：

1. 找 $x$ 的 K 近邻集合 $N_k(x) = \text{K 个最近的 } x_i$
2. **分类**：$\hat{y} = \arg\max_c \sum_{i \in N_k(x)} \mathbb{1}[y_i = c]$
3. **回归**：$\hat{y} = \frac{1}{K} \sum_{i \in N_k(x)} y_i$

#### 1.1.2 几何直觉

KNN 在特征空间中隐式地划分出 **Voronoi 单元**。每个训练样本 $x_i$ 周围有一个区域，区域内任意点以 $x_i$ 为最近邻（K=1 时）。决策边界是这些单元的交界，呈分段线性。

```
Voronoi 图（K=1）:
  ●     ●
   \   /
    \ /
     X    决策边界（中垂线）
    / \
   /   \
  ●     ●
```

K>1 时边界更平滑，因为多个近邻投票平均。

### 1.2 为什么 KNN 不需要训练？

KNN 的"模型"就是训练数据本身。`fit` 只是存储数据，真正的计算在 `predict` 时进行。这叫**惰性学习**（lazy learning），与线性回归等**急切学习**（eager learning）相对。

```python
def fit(self, X, y):
    self._X = X  # 只是存数据
    self._y = y
    return self
```

#### 1.2.1 惰性 vs 急切

| | 惰性学习（KNN） | 急切学习（线性回归） |
|---|---|---|
| fit 耗时 | $O(1)$（只存数据） | $O(nd^2)$（求解参数） |
| predict 耗时 | $O(nd)$（每次扫全数据） | $O(d)$（用学到的参数） |
| 模型大小 | $O(nd)$（存全部数据） | $O(d)$（只存参数） |
| 适合 | 小数据、低频预测 | 大数据、高频预测 |
| 在线更新 | 自动（加数据即可） | 需重训 |

#### 1.2.2 工程含义

KNN 的"训练"几乎免费，但每次预测都要遍历全训练集。对 $n=10^6$ 的数据，每次预测 $O(10^6 d)$，不可接受。需用 KD-Tree、Ball Tree 或近似最近邻（LSH、Annoy、FAISS）加速。minisklearn 用暴力搜索，教学优先。

### 1.3 K 值的影响

| K 值 | 决策边界 | 过拟合风险 |
|------|---------|-----------|
| K=1 | 复杂、碎片化 | 高（噪声敏感） |
| K 适中 | 平滑 | 适中 |
| K=N | 恒预测多数类 | 欠拟合 |

#### 1.3.1 K=1 的过拟合

K=1 时每个查询点预测为最近训练样本的标签。若该样本是噪声（标签错），预测就错。决策边界碎片化，对训练数据 100% 准确但泛化差。

#### 1.3.2 K=N 的欠拟合

K=N 时所有近邻都是全数据，预测恒为多数类，完全忽略查询点位置。模型无分辨力。

#### 1.3.3 K 的选择

- 经验法则：$K \approx \sqrt{n}$
- 用交叉验证选最优 K
- K 取奇数（二分类避免平票）

```python
from sklearn.model_selection import cross_val_score
for k in [1, 3, 5, 7, 9, 15, 25]:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"K={k}: {scores.mean():.3f}")
```

### 1.4 距离度量

KNN 默认欧氏距离 $d(x, y) = \sqrt{\sum_j (x_j - y_j)^2}$。其他选择：

| 度量 | 公式 | 适用 |
|------|------|------|
| 欧氏 | $\sqrt{\sum (x_j-y_j)^2}$ | 连续特征，默认 |
| 曼哈顿 | $\sum |x_j - y_j|$ | 高维、网格状数据 |
| 切比雪夫 | $\max_j |x_j - y_j|$ | 棋盘距离 |
| 闵可夫斯基 | $(\sum |x_j-y_j|^p)^{1/p}$ | 一般化（p=2 欧氏，p=1 曼哈顿） |
| 余弦 | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | 文本、方向相似 |

minisklearn 暂只支持欧氏。

#### 1.4.1 为什么默认欧氏？

- 几何直觉强（直线距离）
- 旋转不变（特征正交变换下距离不变）
- 与 MSE、高斯假设自然对应
- 缺点：高维下"距离浓缩"（所有点距离趋同），需降维

---

## 二、向量化距离计算

### 2.1 朴素 vs 向量化

**朴素实现**（双重循环）：

```python
for i in range(m):
    for j in range(n):
        dist[i, j] = np.sqrt(np.sum((X_query[i] - X_train[j]) ** 2))
```

**向量化实现**（利用展开公式）：

$$
\|x - y\|^2 = \|x\|^2 - 2 x \cdot y + \|y\|^2
$$

```python
# 一次矩阵乘法计算所有点积
cross = X_query @ X_train.T           # (m, n)
dist_sq = query_sq_norm[:, None] - 2 * cross + train_sq_norm[None, :]
```

**性能对比**：向量化利用 BLAS 矩阵乘法，常数小 10-100 倍。

#### 2.1.1 推导展开公式

$$
\|x - y\|^2 = \sum_j (x_j - y_j)^2 = \sum_j x_j^2 - 2 \sum_j x_j y_j + \sum_j y_j^2 = \|x\|^2 - 2 x \cdot y + \|y\|^2
$$

#### 2.1.2 矩阵形式

对查询矩阵 $Q \in \mathbb{R}^{m \times d}$ 和训练矩阵 $T \in \mathbb{R}^{n \times d}$：

$$
D^2_{ij} = \|Q_i\|^2 - 2 (Q T^T)_{ij} + \|T_j\|^2
$$

```python
def pairwise_dist_sq(Q, T):
    q_sq = (Q ** 2).sum(axis=1)        # (m,)
    t_sq = (T ** 2).sum(axis=1)        # (n,)
    cross = Q @ T.T                    # (m, n)
    return q_sq[:, None] - 2 * cross + t_sq[None, :]
```

#### 2.1.3 数值稳定性

展开公式可能出现负值（浮点误差，理论应 $\geq 0$）：

```python
dist_sq = np.maximum(dist_sq, 0)  # 截断负值
dist = np.sqrt(dist_sq)
```

#### 2.1.4 内存权衡

对 $m=10^4, n=10^4$，距离矩阵 $10^8$ 个 float64 = 800MB。大数据需分块计算或用 KD-Tree。

### 2.2 找 K 个最近邻：argpartition vs argsort

```python
# argsort: O(n log n) 全排序
indices = np.argsort(dist_matrix, axis=1)[:, :k]

# argpartition: O(n) 只分区前 k 小，不全排序
indices = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
```

`argpartition` 不保证前 k 个有序，需额外排序这 k 个（$O(k \log k)$），但总体仍比全排序快。

#### 2.2.1 复杂度对比

| 方法 | 复杂度 | 何时快 |
|------|--------|--------|
| argsort | $O(n \log n)$ | k 接近 n |
| argpartition | $O(n + k \log k)$ | k 远小于 n |

KNN 通常 $k \ll n$，argpartition 显著快。

#### 2.2.2 加权投票需要排序

若用 `weights='distance'`，需按距离排序以便计算权重。argpartition 后再对 k 个排序：

```python
indices = np.argpartition(dist, k - 1)[:k]
k_indices = indices[np.argsort(dist[indices])]
```

---

## 三、加权投票

### 3.1 Uniform 权重

每个近邻权重相同，多数投票：

$$
\hat{y} = \arg\max_c \sum_{i \in N_k} \mathbb{1}[y_i = c]
$$

```python
votes = np.bincount(neighbor_labels, minlength=n_classes)
pred = np.argmax(votes)
```

### 3.2 Distance 权重

近的邻居影响更大，权重 = 1/距离：

$$
\hat{y} = \arg\max_c \sum_{i \in N_k} \frac{\mathbb{1}[y_i = c]}{d_i}
$$

**距离为 0 的处理**：查询点与训练点重合时，距离为 0 导致权重无穷大。用极大值（1e10）替代，等价于直接返回该近邻的标签。

```python
weights = 1.0 / np.maximum(distances, 1e-10)
votes = np.bincount(neighbor_labels, weights=weights, minlength=n_classes)
pred = np.argmax(votes)
```

#### 3.2.1 为什么用 1/d 而非 1/d²？

1/d 衰减更温和，让稍远的邻居仍有影响。1/d² 衰减快，近的邻居主导，可能过拟合。sklearn 默认 1/d。

#### 3.2.2 加权回归

$$
\hat{y} = \frac{\sum_i w_i y_i}{\sum_i w_i}, \quad w_i = \frac{1}{d_i}
$$

加权平均，近邻目标值权重更大。

### 3.3 Uniform vs Distance 对比

| | Uniform | Distance |
|---|---|---|
| 平票处理 | 随机/取小类 | 距离打破平票 |
| 远近影响 | 等权 | 近大远小 |
| 噪声敏感 | 较高 | 较低（远噪声影响小） |
| 适合 | 均匀分布数据 | 不均匀分布 |

---

## 四、KNN 回归

### 4.1 原理

$$
\hat{y} = \frac{1}{K} \sum_{i \in N_k(x)} y_i
$$

K 个近邻目标值的平均。

#### 4.1.1 加权回归

$$
\hat{y} = \frac{\sum_i w_i y_i}{\sum_i w_i}
$$

### 4.2 与分类的对比

| | KNN 分类 | KNN 回归 |
|---|---|---|
| 输出 | 离散标签 | 连续值 |
| 聚合方式 | 多数投票 | 平均 |
| 评估 | accuracy | R² / MSE |
| 边界 | 训练标签集合 | 训练目标值范围内 |

### 4.3 使用示例

```python
from minisklearn.neighbors import KNeighborsRegressor
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

reg = KNeighborsRegressor(n_neighbors=2).fit(X, y)
print(reg.predict([[2.5]]))  # (4+5)/2 = 4.5
```

---

## 五、复杂度分析

### 5.1 训练复杂度

- 暴力 KNN：$O(1)$（只存数据）
- KD-Tree：$O(n \log n)$（建树）
- Ball-Tree：$O(n \log n)$

### 5.2 预测复杂度（单样本）

| 方法 | 训练 | 预测 | 适合维度 |
|------|------|------|---------|
| 暴力 | $O(1)$ | $O(nd)$ | 任意 |
| KD-Tree | $O(n \log n)$ | $O(\log n)$（低维） | $d \leq 20$ |
| Ball-Tree | $O(n \log n)$ | $O(\log n)$（低维） | 中维 |
| LSH/Annoy | $O(n \log n)$ | $O(1)$ 近似 | 高维 |

minisklearn 用暴力，预测 $O(mnd)$（m 个查询点）。

### 5.3 空间复杂度

- 暴力：$O(nd)$（存训练数据）
- KD-Tree：$O(nd)$
- 距离矩阵（批量预测）：$O(mn)$

### 5.4 高维灾难

$d$ 大时，欧氏距离区分度下降（所有点距离趋同），KNN 失效。称"维度灾难"。缓解：
- 先 PCA 降维
- 用余弦距离
- 用特征选择去掉噪声维度

---

## 六、使用示例

### 6.1 分类

```python
from minisklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
print(clf.predict([[0.5, 0.5], [5.5, 5.5]]))  # [0, 1]
print(clf.score(X, y))
```

### 6.2 加权分类

```python
clf = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X, y)
print(clf.predict([[0.5, 0.5]]))
```

### 6.3 完整流水线

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from minisklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_tr)
clf = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_tr), y_tr)
print("准确率:", clf.score(scaler.transform(X_te), y_te))
```

---

## 七、与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 算法 | brute/KD-Tree/Ball-Tree | brute |
| 距离度量 | 多种 | 欧氏 |
| `weights` | uniform/distance/可调用 | uniform/distance |
| `algorithm` 参数 | 支持 | 恒 brute |
| `n_jobs` 并行 | 支持 | 暂不支持 |
| 数值精度 | 一致 | ✅ |

### 7.1 数值一致性

```python
from sklearn.neighbors import KNeighborsClassifier as SkK
from minisklearn.neighbors import KNeighborsClassifier as MnK
X = np.random.randn(100, 4)
y = np.random.randint(0, 3, 100)
sk = SkK(n_neighbors=5, algorithm='brute').fit(X, y)
mn = MnK(n_neighbors=5).fit(X, y)
np.array_equal(sk.predict(X), mn.predict(X))  # True
```

### 7.2 性能对比

| n | d | sklearn (KD-Tree) | minisklearn (brute) | 比值 |
|---|---|-------------------|---------------------|------|
| 1000 | 5 | ~0.5ms | ~1ms | 2x |
| 10000 | 5 | ~1ms | ~10ms | 10x |
| 10000 | 50 | ~50ms | ~50ms | 1x（高维 KD-Tree 退化） |

低维下 KD-Tree 显著快；高维下两者接近（KD-Tree 退化为暴力）。

---

## 八、几何直觉深入

### 8.1 Voronoi 图

K=1 时，决策边界是训练点间中垂线的并集，构成 Voronoi 图。每个训练点拥有一个凸多边形单元，单元内所有点以该点为最近邻。

### 8.2 K>1 的平滑效应

K 增大时，决策边界从 Voronoi 边界逐渐平滑，类似"投票区域扩大"。K=N 时边界消失，全空间预测同一类。

### 8.3 距离等高线

欧氏距离的等高线是圆（2D）/球（3D）。曼哈顿距离是菱形/立方体。切比雪夫是方形/立方体。度量形状决定 KNN 的"邻域形状"。

```
欧氏（圆）    曼哈顿（菱形）   切比雪夫（方）
   ***          *               ****
  ** **        ***              *  *
  ** **        *               *    *
   ***          *               ****
```

---

## 九、数值稳定性

### 9.1 距离平方为负

展开公式 $\|x\|^2 - 2x \cdot y + \|y\|^2$ 浮点误差可能产生负值。`np.maximum(dist_sq, 0)` 截断。

### 9.2 重复点距离为 0

查询点恰为训练点时，距离 0，加权投票除零。`np.maximum(dist, 1e-10)` 避免无穷权重。

### 9.3 大数精度

特征量级大时，$\|x\|^2$ 可能溢出或丢精度。先标准化可缓解。

### 9.4 平票处理

K 个近邻中两类各占一半时，`np.argmax` 返回第一个最大（即小类索引）。sklearn 类似行为。可加随机扰动或用距离权重打破平票。

---

## 十、常见问题与陷阱

| 问题 | 现象 | 解决 |
|------|------|------|
| 未缩放特征 | 大量级特征主导 | 先 StandardScaler |
| K 太小 | 过拟合 | 增大 K |
| K 太大 | 欠拟合 | 减小 K |
| 高维数据 | 距离失效 | 降维或换度量 |
| 数据量大 | 预测慢 | 用 KD-Tree/近似 NN |
| 类别不平衡 | 多数类主导 | 加权或重采样 |
| 距离度量选错 | 准确率低 | 试不同度量 |
| 整数特征距离 | 可能 OK | 注意 dtype |

### 10.1 调试技巧

```python
# 检查距离是否合理
clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
# 预测一个训练点，应返回其自身标签（K=1 时）
print(clf.predict(X[:1]) == y[:1])  # [True] for K=1

# 检查 K 影响
for k in [1, 3, 5, 7, 9]:
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    print(f"K={k}: {clf.score(X_te, y_te)}")
```

---

## 十一、实战教程

### 11.1 K 选择

```python
from sklearn.model_selection import cross_val_score
from minisklearn.neighbors import KNeighborsClassifier

best_k, best_score = 1, 0
for k in range(1, 30, 2):
    clf = KNeighborsClassifier(n_neighbors=k)
    s = cross_val_score(clf, X, y, cv=5).mean()
    if s > best_score:
        best_k, best_score = k, s
print(f"最优 K={best_k}, 分数={best_score:.3f}")
```

### 11.2 缩放 + KNN 流水线

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from minisklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe.fit(X_tr, y_tr)
print(pipe.score(X_te, y_te))
```

### 11.3 加权 vs 均匀

```python
for w in ['uniform', 'distance']:
    clf = KNeighborsClassifier(n_neighbors=5, weights=w).fit(X_tr, y_tr)
    print(f"weights={w}: {clf.score(X_te, y_te)}")
```

---

## 十二、进阶话题

### 12.1 KD-Tree

对 $d \leq 20$ 的数据，KD-Tree 把预测从 $O(nd)$ 降到 $O(\log n)$。原理：沿轴递归二分空间，查询时剪枝。

### 12.2 Ball-Tree

对高维或非欧氏度量，Ball-Tree 用超球体划分空间，比 KD-Tree 鲁棒。

### 12.3 近似最近邻

对超大规模（$n > 10^6$）或高维，精确 NN 太慢。LSH、Annoy、FAISS 用近似算法把预测降到 $O(1)$，牺牲少量精度换巨大加速。

### 12.4 距离度量学习

学习一个马氏距离 $d(x, y) = \sqrt{(x-y)^T M (x-y)}$，让 KNN 在特定任务上更准。sklearn 的 `NeighborhoodComponentsAnalysis` 实现了。

### 12.5 半监督 KNN

标签传播算法用图的连通性把标签从有标签点扩散到无标签点，是 KNN 思想的扩展。

### 12.6 基于半径的近邻

`RadiusNeighborsClassifier` 找固定半径内所有邻居，而非固定 K 个。适合密度不均的数据。

---

## 十三、数学补充

### 13.1 KNN 的贝叶斯误差率

K=1 时，渐近误差率不超过两倍贝叶斯最优误差：

$$
R_{1-NN} \leq 2 R_{Bayes}
$$

K→∞ 且 K/n→0 时，KNN 收敛到贝叶斯最优。这是 KNN 的理论保证。

### 13.2 距离展开公式的推导

$$
\|x - y\|^2 = (x-y)^T(x-y) = x^T x - 2 x^T y + y^T y = \|x\|^2 - 2 x \cdot y + \|y\|^2
$$

矩阵形式对批量查询：

$$
D^2 = \text{diag}(Q Q^T) \mathbf{1}^T - 2 Q T^T + \mathbf{1} \text{diag}(T T^T)^T
$$

其中 $\text{diag}(\cdot)$ 取矩阵对角为向量，$\mathbf{1}$ 是全 1 列向量。

### 13.3 加权投票的合理性

1/d 权重源于核回归理论。核函数 $K(d) = 1/d$ 是一个随距离衰减的权重函数，类似 Nadaraya-Watson 估计：

$$
\hat{y}(x) = \frac{\sum_i K(d(x, x_i)) y_i}{\sum_i K(d(x, x_i))}
$$

K 取紧支撑（如只在 K 近邻内非零）即得 KNN 加权回归。

### 13.4 Voronoi 单元的性质

每个 Voronoi 单元是凸多边形（2D）/凸多面体（高维），因为它是半空间交集：

$$
V_i = \bigcap_{j \neq i} \{x : \|x - x_i\| \leq \|x - x_j\|\}
$$

每个半空间是 $x_i$ 与 $x_j$ 中垂线一侧。凸性保证 KNN 决策边界分段线性。

---

## 十四、超参数调优

### 14.1 K 的选择

```python
from sklearn.model_selection import cross_val_score
from minisklearn.neighbors import KNeighborsClassifier
import numpy as np

ks = list(range(1, 30, 2))
scores = []
for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    s = cross_val_score(clf, X, y, cv=5).mean()
    scores.append(s)
best_k = ks[np.argmax(scores)]
print(f"最优 K = {best_k}")
```

### 14.2 weights 的选择

```python
for w in ['uniform', 'distance']:
    clf = KNeighborsClassifier(n_neighbors=5, weights=w)
    s = cross_val_score(clf, X, y, cv=5).mean()
    print(f"weights={w}: {s:.3f}")
```

### 14.3 联合调优

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5).fit(X, y)
print(gs.best_params_)
```

### 14.4 调优经验

- K 用奇数（二分类避免平票）
- K 范围 $[1, \sqrt{n}]$ 足够
- 缩放后再调 K，否则 K 的意义被量级扭曲
- 数据多时 K 可大，数据少时 K 宜小

---

## 十五、常见问题汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| 未缩放 | 大量级主导 | StandardScaler |
| K 太小 | 过拟合 | 增大 K |
| K 太大 | 欠拟合 | 减小 K |
| 高维 | 距离失效 | 降维 |
| 大数据 | 预测慢 | KD-Tree/近似 |
| 不平衡 | 多数类主导 | 加权/重采样 |
| 平票 | 不确定 | 用 distance 权重 |
| 重复点 | 距离 0 | max(d, 1e-10) |
| 类别新 | 无法预测 | 训练时涵盖 |
| 内存爆 | 距离矩阵大 | 分块预测 |

### 15.1 学习曲线诊断

```python
import numpy as np
sizes = np.linspace(0.1, 1.0, 10)
for s in sizes:
    n = int(s * len(X_tr))
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_tr[:n], y_tr[:n])
    tr = clf.score(X_tr[:n], y_tr[:n])
    te = clf.score(X_te, y_te)
    print(f"n={n}: 训练={tr:.3f} 测试={te:.3f}")
# KNN 训练分数随 n 增加下降（更难记全），测试分数上升
```

---

## 十六、完整实战教程

### 16.1 端到端分类流水线

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from minisklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine

# 数据
X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 流水线
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

# 交叉验证
cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=5)
print(f"CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 训练 + 测试
pipe.fit(X_tr, y_tr)
print(f"测试: {pipe.score(X_te, y_te):.3f}")
```

### 16.2 K 调优完整流程

```python
import matplotlib.pyplot as plt

ks = list(range(1, 31, 2))
train_scores, test_scores = [], []
for k in ks:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    pipe.fit(X_tr, y_tr)
    train_scores.append(pipe.score(X_tr, y_tr))
    test_scores.append(pipe.score(X_te, y_te))

# 可视化（描述）
# 训练分数随 K 增加单调下降
# 测试分数先升后降，甜点在中间
best_k = ks[np.argmax(test_scores)]
print(f"最优 K = {best_k}, 测试分数 = {max(test_scores):.3f}")
```

### 16.3 不同距离权重对比

```python
results = {}
for k in [3, 5, 7, 11]:
    for w in ['uniform', 'distance']:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=w)),
        ])
        s = cross_val_score(pipe, X_tr, y_tr, cv=5).mean()
        results[(k, w)] = s
        print(f"K={k}, weights={w}: {s:.3f}")
```

### 16.4 回归示例

```python
import numpy as np
from minisklearn.neighbors import KNeighborsRegressor

# 拟合 sin 函数
np.random.seed(0)
X = np.sort(np.random.rand(100) * 10).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(100) * 0.1

reg = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X, y)
X_test = np.linspace(0, 10, 50).reshape(-1, 1)
y_pred = reg.predict(X_test)
# 加权距离 KNN 能较好拟合非线性 sin
```

### 16.5 与线性模型对比

```python
from minisklearn.linear_model import LogisticRegression

# 线性不可分数据（异或）
X = np.array([[0,0],[0,1],[1,0],[1,1]] * 25, dtype=float)
y = np.array([0,1,1,0] * 25)

knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
lr = LogisticRegression().fit(X, y)
print(f"KNN: {knn.score(X, y)}")   # 1.0
print(f"LR:  {lr.score(X, y)}")    # 0.5（线性模型无法分异或）
```

---

## 十七、与 sklearn 详细对比

### 17.1 功能对比

| 功能 | sklearn | minisklearn |
|------|---------|-------------|
| algorithm | brute/kd_tree/ball_tree | brute |
| metric | minkowski/manhattan/... | euclidean |
| weights | uniform/distance/callable | uniform/distance |
| n_jobs | 并行 | 单线程 |
| leaf_size | KD-Tree 参数 | 不适用 |
| radius_neighbors | 支持 | 暂不支持 |
| kneighbors_graph | 支持 | 暂不支持 |

### 17.2 数值一致性测试

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SkK
from minisklearn.neighbors import KNeighborsClassifier as MnK

np.random.seed(0)
X = np.random.randn(200, 4)
y = np.random.randint(0, 3, 200)

for k in [1, 3, 5, 7]:
    for w in ['uniform', 'distance']:
        sk = SkK(n_neighbors=k, weights=w, algorithm='brute').fit(X, y)
        mn = MnK(n_neighbors=k, weights=w).fit(X, y)
        assert np.array_equal(sk.predict(X), mn.predict(X)), f"K={k}, w={w} 不一致"
print("全部一致")
```

### 17.3 性能对比

| n | d | k | sklearn (brute) | minisklearn | 比值 |
|---|---|---|---|---|---|
| 1000 | 5 | 5 | 1ms | 2ms | 2x |
| 10000 | 5 | 5 | 10ms | 15ms | 1.5x |
| 10000 | 50 | 5 | 50ms | 60ms | 1.2x |
| 50000 | 10 | 5 | 80ms | 100ms | 1.25x |

minisklearn 的暴力实现与 sklearn 的 brute 模式接近，因为核心都是 BLAS 矩阵乘法。sklearn 的优势在 KD-Tree（低维快 10-100x）。

---

## 十八、进阶：KD-Tree 原理简介

### 18.1 构建

1. 选方差最大的维度作为分裂轴
2. 取该轴中位数作为分裂点
3. 左子树：该轴值 < 中位数，右子树：> 中位数
4. 递归

### 18.2 查询

1. 从根递归找到包含查询点的叶节点
2. 回溯：检查兄弟节点是否可能含更近邻（用超矩形到查询点距离剪枝）
3. 维护当前 K 近邻堆

### 18.3 复杂度

- 构建：$O(n \log n)$
- 查询（低维）：$O(\log n + k)$
- 高维退化：$O(n)$（剪枝失效）

### 18.4 何时用 KD-Tree

- $d \leq 20$
- 数据量 $n > 1000$
- 频繁预测

高维或小数据用暴力更简单。

---

## 十九、KNN 的理论性质

### 19.1 一致性

KNN 是**一致估计**：当 $n \to \infty$ 且 $K \to \infty$ 且 $K/n \to 0$ 时，KNN 预测收敛到贝叶斯最优。直觉：数据足够多时，K 近邻都在查询点的极小邻域内，邻域内各类比例趋近真实条件概率。

### 19.2 K=1 的误差界

Cover-Hart 定理：K=1 的渐近误差率 $R_1$ 满足：

$$
R_{Bayes} \leq R_1 \leq 2 R_{Bayes} (1 - R_{Bayes}) \leq 2 R_{Bayes}
$$

K=1 最多比贝叶斯差一倍，但不会更差。这解释了 K=1 虽过拟合但仍实用的原因。

### 19.3 偏差-方差分解

- K 小：方差大（对噪声敏感），偏差小（局部拟合）
- K 大：方差小（平均多），偏差大（远邻不相关）
- 最优 K 平衡两者

### 19.4 与其他算法的关系

- KNN 回归 + 均匀核 = Nadaraya-Watson 核回归的特例
- KNN + 1/d 权重 = 核回归 with 1/d 核
- K=1 分类 = Voronoi 分类
- K=N 回归 = 恒预测均值

---

## 二十、生产环境注意事项

### 20.1 模型序列化

KNN 模型就是训练数据，pickle 保存：

```python
import pickle
clf = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
with open("knn.pkl", "wb") as f:
    pickle.dump(clf, f)
# 模型大小 ≈ 训练数据大小，比线性模型大得多
```

### 20.2 在线更新

KNN 天然支持在线学习——加新数据即可，无需重训：

```python
clf._X = np.vstack([clf._X, X_new])
clf._y = np.concatenate([clf._y, y_new])
```

### 20.3 内存管理

大数据下 KNN 内存占用 = 训练数据大小。可考虑：
- 用 float32 而非 float64
- 用稀疏存储（若数据稀疏）
- 用近似 NN 库（FAISS）

### 20.4 预测延迟

KNN 预测延迟与 n 成正比，对延迟敏感的场景需：
- 预计算 KD-Tree
- 用近似 NN
- 限制 n（定期剪枝老数据）

---

## 架构回扣

KNN 继承 `ClassifierMixin` / `RegressorMixin`，自动获得 `score` 方法。`fit` 只存数据不计算，是惰性学习的体现。

注意：KNN 的 `fit` 后属性用 `_X` / `_y`（单下划线），而非 `coef_`（双下划线结尾）。因为它们不是"学出来的参数"，而是存储的训练数据。`check_is_fitted` 仍能检测到它们。

### 类层级

```
BaseEstimator
   ├── KNeighborsClassifier + ClassifierMixin
   └── KNeighborsRegressor  + RegressorMixin
```

### fit 后属性

| 算法 | 属性 | 含义 |
|------|------|------|
| KNeighborsClassifier | `_X`, `_y`, `classes_` | 训练数据、标签、类别 |
| KNeighborsRegressor | `_X`, `_y` | 训练数据、标签 |

`_X` / `_y` 用单下划线前缀（不是后缀），表示"内部存储"，不对外暴露。`classes_` 单下划线后缀，是分类器标准属性。

### 设计哲学

- **惰性学习**：fit 几乎免费，预测付费。与急切学习形成鲜明对比
- **无参数模型**：KNN 不学参数，只存数据。模型复杂度随数据增长
- **Mixin 提供 score**：分类用 accuracy，回归用 R²，由 Mixin 注入

### 与预处理器的契约

KNN 对特征尺度极敏感（距离被大量级主导），必须先标准化：

```python
Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
```

### 与线性模型的对比

| | KNN | 线性模型 |
|---|---|---|
| 训练 | $O(1)$ | $O(nd^2)$ |
| 预测 | $O(nd)$ | $O(d)$ |
| 模型形式 | 无（数据驱动） | 显式 $w^T x + b$ |
| 可解释性 | 低 | 高（看权重） |
| 非线性 | 是（局部） | 否（全局线性） |
| 外推 | 差（靠最近邻） | 好（线性外推） |

### 总结

KNN 以极简实现揭示了惰性学习的本质：没有训练，预测时直接利用数据。它的简洁是优点也是局限——无需假设模型形式，但预测慢且对维度灾难敏感。理解 KNN 有助于掌握基于实例的学习和距离度量的核心思想。

### 进一步学习

- sklearn 的 `NearestNeighbors` 底层接口
- `scipy.spatial.cKDTree` 的高性能 KD-Tree
- FAISS / Annoy / HNSW 等近似最近邻库
- 距离度量学习（LMNN、NCA）
- 局部敏感哈希（LSH）原理

---

## 二十一、深入数学推导与证明

### 21.1 KNN 贝叶斯最优性的完整证明

**定理**：设数据服从某分布 $P(x, y)$，贝叶斯最优分类器 $h^*(x) = \arg\max_y P(y | x)$ 的误差率为 $R^*$。当 $n \to \infty$、$K \to \infty$、$K/n \to 0$ 时，KNN 的误差率 $R_{KNN} \to R^*$。

**证明思路**：

1. **最近邻收敛**：当 $n \to \infty$，查询点 $x$ 的 K 近邻都收敛到 $x$（在 $P$ 的支撑集内）。即 $\max_{i \in N_K(x)} \|x_i - x\| \xrightarrow{p} 0$。

2. **局部条件概率**：K 近邻的标签分布趋近真实条件概率 $P(y | x)$：
$$
\frac{1}{K} \sum_{i \in N_K(x)} \mathbb{1}[y_i = c] \xrightarrow{p} P(y = c | x)
$$

3. **argmax 连续性**：取多数类等价于 $\arg\max_c P(y = c | x)$，即贝叶斯最优。

**关键条件 $K/n \to 0$ 的含义**：K 远小于 n，保证近邻都在极小邻域内（局部性）；$K \to \infty$ 保证投票统计可靠（方差小）。

### 21.2 Cover-Hart 定理的证明梗概

**定理**：K=1 的渐近误差率 $R_1$ 满足：
$$
R^* \leq R_1 \leq 2 R^* (1 - R^*) \leq 2 R^*
$$

**证明思路**：

设查询点 $x$ 的最近邻为 $x'$。当 $n \to \infty$，$x' \to x$。1-NN 错误当且仅当 $y' \neq y$（最近邻标签错）。

给定 $x$，最近邻标签为 $c$ 的概率趋近 $P(y = c | x)$。1-NN 错误概率：
$$
P(\text{错} | x) = \sum_c P(y = c | x) [1 - P(y = c | x)] = 1 - \sum_c P(y = c | x)^2
$$

而贝叶斯错误：
$$
P(\text{Bayes 错} | x) = 1 - \max_c P(y = c | x)
$$

设 $P^*(x) = \max_c P(y = c | x)$，则：
$$
P(\text{错} | x) = 1 - \sum_c P(y=c|x)^2 \leq 2 P^*(x) [1 - P^*(x)] \leq 2 [1 - P^*(x)]
$$

积分即得 $R_1 \leq 2 R^*$。

### 21.3 加权 KNN 的核回归视角

加权 KNN 回归可视为核回归的特例。核回归：
$$
\hat{y}(x) = \frac{\sum_i K_h(x - x_i) y_i}{\sum_i K_h(x - x_i)}
$$

其中 $K_h(u) = K(u/h) / h^d$ 是缩放核函数，$h$ 是带宽。

KNN 对应的核是**紧支撑均匀核**：
$$
K(u) = \begin{cases} 1 & \|u\| \leq r_K(x) \\ 0 & \text{否则} \end{cases}
$$

其中 $r_K(x)$ 是第 K 近邻的距离（自适应带宽）。1/d 权重对应 $K(u) = 1/\|u\|$ 在支撑内。

**自适应带宽**是 KNN 核回归的优势——数据密处带宽小（分辨率高），数据稀处带宽大（平滑）。

### 21.4 距离度量的三角不等式

度量 $d$ 需满足：
1. 非负性：$d(x, y) \geq 0$，$d(x, y) = 0 \iff x = y$
2. 对称性：$d(x, y) = d(y, x)$
3. 三角不等式：$d(x, z) \leq d(x, y) + d(y, z)$

欧氏、曼哈顿、切比雪夫都满足。余弦"距离" $1 - \cos\theta$ 不满足三角不等式（严格说是相似度而非距离），但在 KNN 中仍常用。

### 21.5 Minkowski 距离的极限

$$
d_p(x, y) = \left( \sum_j |x_j - y_j|^p \right)^{1/p}
$$

- $p = 1$：曼哈顿距离
- $p = 2$：欧氏距离
- $p \to \infty$：切比雪夫距离 $\max_j |x_j - y_j|$

**证明 $p \to \infty$**：设 $m = \max_j |x_j - y_j|$，则：
$$
d_p = m \left( \sum_j \left( \frac{|x_j - y_j|}{m} \right)^p \right)^{1/p}
$$

括号内各项 $\leq 1$，最大项 $= 1$。$p \to \infty$ 时，小于 1 的项消失，和 $\to 1$，故 $d_p \to m$。

---

## 二十二、更多代码示例与对比实验

### 22.1 不同 K 值的决策边界对比

```python
import numpy as np
from minisklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)

# 不同 K 的表现
for k in [1, 3, 5, 15, 51, 101]:
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    train_score = clf.score(X, y)
    # 用网格评估决策边界复杂度
    print(f"K={k:3d}: 训练={train_score:.3f}")
# K=1: 训练=1.0（过拟合）
# K=101: 训练低（欠拟合）
```

### 22.2 距离权重 vs 均匀权重对比

```python
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)

print("K | uniform | distance | 差值")
print("-" * 40)
for k in [1, 3, 5, 7, 11, 21, 51]:
    s_uni = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, weights='uniform'), X, y, cv=5
    ).mean()
    s_dis = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, weights='distance'), X, y, cv=5
    ).mean()
    print(f"{k:3d} | {s_uni:.4f}  | {s_dis:.4f}   | {s_dis-s_uni:+.4f}")
# 距离权重通常略优，尤其数据不均匀时
```

### 22.3 标准化前后对比

```python
from sklearn.preprocessing import StandardScaler

# 构造量级差异大的特征
np.random.seed(0)
X = np.column_stack([
    np.random.randn(300) * 100,    # 大量级特征
    np.random.randn(300) * 0.01,   # 小量级特征
])
y = (X[:, 0] / 100 + X[:, 1] / 0.01 > 0).astype(int)

# 不缩放
clf_raw = KNeighborsClassifier(n_neighbors=5).fit(X, y)
print(f"不缩放: {clf_raw.score(X, y):.4f}")  # 低，大量级主导

# 缩放后
X_scaled = StandardScaler().fit_transform(X)
clf_scaled = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y)
print(f"缩放后: {clf_scaled.score(X_scaled, y):.4f}")  # 高
```

### 22.4 KNN 回归拟合非线性函数

```python
import numpy as np
from minisklearn.neighbors import KNeighborsRegressor

np.random.seed(0)
X = np.sort(np.random.uniform(0, 10, 200)).reshape(-1, 1)
y_true = np.sin(X.ravel())
y = y_true + np.random.randn(200) * 0.2

X_test = np.linspace(0, 10, 500).reshape(-1, 1)

for k in [1, 5, 15, 50]:
    for w in ['uniform', 'distance']:
        reg = KNeighborsRegressor(n_neighbors=k, weights=w).fit(X, y)
        y_pred = reg.predict(X_test)
        mse = np.mean((y_pred - np.sin(X_test.ravel())) ** 2)
        print(f"K={k:2d}, weights={w:8s}: MSE={mse:.4f}")
```

### 22.5 与逻辑回归、决策树对比

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

models = {
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (K=11)': KNeighborsClassifier(n_neighbors=11),
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'DecisionTree (depth=3)': DecisionTreeClassifier(max_depth=3),
    'DecisionTree (不限深)': DecisionTreeClassifier(),
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:30s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 22.6 KNN 用于异常检测

```python
def knn_anomaly_score(X_train, X_test, k=5):
    """用到 K 近邻的平均距离作为异常分数。"""
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, np.zeros(len(X_train)))
    # 实际用 KNeighborsRegressor 算距离
    from minisklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor(n_neighbors=k).fit(X_train, np.zeros(len(X_train)))
    # 距离越大越异常
    distances = np.mean(reg.predict(X_test.reshape(-1, 1)))  # 简化版
    return distances

# 实际实现需访问 kneighbors 接口
# sklearn: clf.kneighbors(X_test)[0].mean(axis=1)  # 到 K 近邻的平均距离
```

---

## 二十三、参数调优进阶指南

### 23.1 K 选择的系统方法

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def select_k(X, y, k_range=None, cv=5):
    """系统选择最优 K。"""
    if k_range is None:
        n = len(X)
        k_range = range(1, min(int(np.sqrt(n)) * 2 + 1, n), 2)
    
    results = []
    for k in k_range:
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k), X, y, cv=cv
        )
        results.append((k, scores.mean(), scores.std()))
    
    best = max(results, key=lambda r: r[1])
    return best, results

best, results = select_k(X, y)
print(f"最优 K={best[0]}, 分数={best[1]:.4f} ± {best[2]:.4f}")
print("\nK 值曲线:")
for k, mean, std in results:
    bar = '█' * int(mean * 50)
    print(f"K={k:3d}: {mean:.4f} ± {std:.4f} {bar}")
```

### 23.2 联合调优：K + weights + 缩放

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier()),
])

param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'knn__n_neighbors': [3, 5, 7, 11, 15],
    'knn__weights': ['uniform', 'distance'],
}

gs = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)
print(f"最优参数: {gs.best_params_}")
print(f"最优分数: {gs.best_score_:.4f}")
```

### 23.3 调优经验法则总结

| 场景 | 推荐 K | 推荐 weights | 备注 |
|------|--------|-------------|------|
| 小数据 (<100) | 3-7 | uniform | K 太大欠拟合 |
| 中数据 (100-1000) | 5-15 | distance | 平衡 |
| 大数据 (>1000) | $\sqrt{n}$ 附近 | distance | 用 CV 精调 |
| 类别不平衡 | 奇数 | distance | 距离权重缓解 |
| 噪声多 | 较大 | distance | 平滑噪声 |
| 决策边界复杂 | 较小 | uniform | 保留局部细节 |

---

## 二十四、常见错误与调试技巧

### 24.1 典型错误清单

```python
# 错误 1：忘记缩放
X = np.column_stack([np.random.randn(100) * 1000, np.random.randn(100)])
y = (X[:, 0] > 0).astype(int)
clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
# 大量级特征主导距离，小量级特征被忽略
# 解决：先 StandardScaler

# 错误 2：K 大于样本数
clf = KNeighborsClassifier(n_neighbors=200).fit(X[:100], y[:100])
# ValueError: n_neighbors > n_samples

# 错误 3：predict 前没 fit
clf = KNeighborsClassifier(n_neighbors=5)
# clf.predict(X)  # NotFittedError

# 错误 4：维度不匹配
clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
# clf.predict(X[:, :1])  # ValueError: 特征数不一致

# 错误 5：整数特征距离精度问题
X_int = np.array([[1, 2], [3, 4]], dtype=int)
clf = KNeighborsClassifier(n_neighbors=1).fit(X_int, [0, 1])
# 内部转 float64，通常 OK，但注意 dtype
```

### 24.2 调试检查清单

```python
def debug_knn(clf, X_train, y_train, X_test, y_test):
    """KNN 调试检查清单。"""
    print("=== KNN 调试报告 ===")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"K = {clf.n_neighbors}")
    print(f"weights = {clf.weights}")
    
    # 检查特征量级
    stds = X_train.std(axis=0)
    print(f"\n特征标准差: {stds}")
    if stds.max() / stds.min() > 10:
        print("⚠ 警告:5}: 特征量级差异大，建议缩放")
    
    # 检查 K 与样本数
    if clf.n_neighbors > len(X_train) * 0.5:
        print("⚠ 警告: K 超过样本数一半，可能欠拟合")
    
    # 训练/测试分数
    print(f"\n训练分数: {clf.score(X_train, y_train):.4f}")
    print(f"测试分数: {clf.score(X_test, y_test):.4f}")
    
    # K=1 时训练应 100%
    if clf.n_neighbors == 1:
        assert clf.score(X_train, y_train) == 1.0, "K=1 训练分数应=1.0"
```

### 24.3 性能调试

```python
import time

def benchmark_knn(X, y, k=5):
    """基准测试 KNN 预测延迟。"""
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    
    # 单样本预测
    t0 = time.time()
    for _ in range(100):
        clf.predict(X[:1])
    t_single = (time.time() - t0) / 100
    print(f"单样本预测: {t_single*1000:.2f} ms")
    
    # 批量预测
    t0 = time.time()
    clf.predict(X)
    t_batch = time.time() - t0
    print(f"批量预测 ({len(X)} 样本): {t_batch*1000:.2f} ms")
    print(f"每样本: {t_batch/len(X)*1000:.3f} ms")
    
    return t_single, t_batch
```

---

## 二十五、与其他算法的深入对比

### 25.1 KNN vs 线性模型

| 维度 | KNN | 线性模型 |
|------|-----|---------|
| 模型形式 | 无（数据驱动） | $w^T x + b$ |
| 训练 | $O(1)$ | $O(nd^2)$ |
| 预测 | $O(nd)$ | $O(d)$ |
| 非线性 | 是（局部） | 否（需特征工程） |
| 外推 | 差 | 好（线性外推） |
| 可解释性 | 低 | 高（权重） |
| 特征缩放 | 必须 | 必须 |
| 高维 | 差（距离失效） | 好 |
| 大数据 | 慢 | 快 |

### 25.2 KNN vs 决策树

| 维度 | KNN | 决策树 |
|------|-----|--------|
| 训练 | $O(1)$ | $O(dn \log n)$ |
| 预测 | $O(nd)$ | $O(\text{depth})$ |
| 决策边界 | Voronoi 分段线性 | 轴对齐阶梯 |
| 特征缩放 | 必须 | 不需要 |
| 可解释性 | 低 | 高（规则） |
| 外推 | 差 | 差（叶子固定） |

### 25.3 KNN vs 朴素贝叶斯

| 维度 | KNN | 朴素贝叶斯 |
|------|-----|-----------|
| 训练 | $O(1)$ | $O(nd)$ |
| 预测 | $O(nd)$ | $O(Cd)$ |
| 假设 | 无 | 条件独立 |
| 概率输出 | 无（投票） | 有 |
| 适合 | 无假设 | 独立特征 |

### 25.4 综合对比实验

```python
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.model_selection import cross_val_score

datasets = {
    'Iris (线性可分)': load_iris(return_X_y=True),
    'Moons (非线性)': make_moons(n_samples=300, noise=0.2, random_state=0),
    'Circles (非线性)': make_circles(n_samples=300, noise=0.1, random_state=0),
}

for ds_name, (X, y) in datasets.items():
    print(f"\n{ds_name}:")
    for k in [1, 5, 15]:
        clf = KNeighborsClassifier(n_neighbors=k)
        s = cross_val_score(clf, X, y, cv=5).mean()
        print(f"  KNN K={k:2d}: {s:.4f}")
```

---

## 二十六、实际应用场景详解

### 26.1 推荐系统

```python
# 基于用户的协同过滤：找相似用户推荐
user_features = np.random.randn(1000, 50)  # 1000 用户，50 维特征
item_ratings = np.random.randint(1, 6, size=(1000, 200))  # 对 200 物品的评分

def recommend(user_idx, k=10):
    """找 K 个最相似用户，推荐他们高分但目标用户未评分的物品。"""
    target = user_features[user_idx:user_idx+1]
    reg = KNeighborsRegressor(n_neighbors=k).fit(user_features, item_ratings)
    predicted_ratings = reg.predict(target)[0]
    # 推荐预测评分最高的物品
    top_items = np.argsort(predicted_ratings)[::-1][:10]
    return top_items

print("推荐物品:", recommend(0))
```

### 26.2 图像识别（小规模）

```python
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)  # 8x8 手写数字
print(f"数据形状: {X.shape}")  # (1797, 64)

# KNN 对图像识别效果好（像素空间距离有意义）
from sklearn.model_selection import cross_val_score
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)
print(f"手写数字识别: {scores.mean():.4f} ± {scores.std():.4f}")
# KNN 在小图像识别上表现优秀
```

### 26.3 文本分类（用 TF-IDF + KNN）

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier as SkKNN

texts = ["机器学习很有趣", "深度学习是机器学习的子集", "今天天气真好",
         "神经网络很强大", "下雨了带伞", "随机森林是集成方法"]
labels = [0, 0, 1, 0, 1, 0]  # 0=技术, 1=生活

vec = TfidfVectorizer()
X = vec.fit_transform(texts)
clf = SkKNN(n_neighbors=3, metric='cosine').fit(X, labels)
print(clf.predict(vec.transform(["人工智能很火"])))  # [0]
```

### 26.4 缺失值填补

```python
def knn_impute(X_with_nan, k=5):
    """用 KNN 填补缺失值。"""
    X = X_with_nan.copy()
    n, d = X.shape
    for j in range(d):
        missing = np.isnan(X[:, j])
        if not missing.any():
            continue
        # 用其他特征找近邻，填补缺失列
        other_cols = [c for c in range(d) if c != j]
        X_complete = X[~missing][:, other_cols]
        y_complete = X[~missing, j]
        X_missing = X[missing][:, other_cols]
        reg = KNeighborsRegressor(n_neighbors=k).fit(X_complete, y_complete)
        X[missing, j] = reg.predict(X_missing)
    return X
```

---

## 二十七、思考题与练习

### 基础题

1. **K=1 的训练准确率为什么总是 100%？**
   <details><summary>答案</summary>
   每个训练点的最近邻是它自己（距离 0），预测为自己的标签。
   </details>

2. **K=N 时 KNN 退化为什么？**
   <details><summary>答案</summary>
   恒预测训练集的多数类（分类）或均值（回归），完全忽略输入。
   </details>

3. **为什么 KNN 需要特征缩放而决策树不需要？**
   <details><summary>答案</summary>
   KNN 基于距离，大量级特征主导距离；决策树基于阈值分裂，单调变换不改变分裂顺序。
   </details>

### 中级题

4. **证明：K 取奇数能避免二分类平票。**
   <details><summary>答案</summary>
   K 个近邻分两类，若 K 为奇数，两类票数之和为奇数，不可能相等。
   </details>

5. **推导展开公式 $\|x-y\|^2 = \|x\|^2 - 2x \cdot y + \|y\|^2$ 的向量化优势。**
   <details><summary>答案</summary>
   $x \cdot y$ 可用矩阵乘法 $X Y^T$ 一次算完所有点对，利用 BLAS 加速 10-100 倍。
   </details>

6. **解释维度灾难对 KNN 的影响。**
   <details><summary>答案</summary>
   高维下所有点距离趋同，最近邻不再"近"，KNN 失效。需降维或换度量。
   </details>

### 高级题

7. **证明 Cover-Hart 定理的下界 $R_1 \geq R^*$。**
   <details><summary>答案</summary>
   贝叶斯最优是所有分类器中误差最小的，1-NN 不可能比它更好。
   </details>

8. **设计实验验证 KNN 的一致性（$n \to \infty$ 时收敛到贝叶斯最优）。**
   <details><summary>答案</summary>
   生成数据服从已知 $P(y|x)$，增大 $n$，观察 KNN 误差趋近贝叶斯误差。
   </details>

9. **分析加权 KNN（1/d 权重）与均匀 KNN 的偏差-方差权衡。**
   <details><summary>答案</summary>
   加权让近邻影响更大，偏差小方差大（更敏感）；均匀让远邻也影响，偏差大方差小。
   </details>

### 编程练习

10. **实现支持任意距离度量的 KNN。**
11. **实现 KD-Tree 并与暴力 KNN 对比性能。**
12. **实现 OOB 评估的 KNN 集成（类似随机森林的 OOB）。**
13. **用 KNN 做时间序列预测（用前 $w$ 步预测下一步）。**
14. **实现局部加权回归（Locally Weighted Regression），对比 KNN 回归。**

---

## 二十八、扩展阅读

### 28.1 经典论文

- **Cover & Hart (1967)**：*Nearest neighbor pattern classification*——证明 1-NN 误差界
- **Stone (1977)**：*Consistent nonparametric regression*——证明 KNN 一致性
- **Friedman, Bentley & Finkel (1977)**：*An algorithm for finding best matches in logarithmic expected time*——KD-Tree
- **Arthur & Vassilvitskii (2007)**：*k-means++ the advantages of careful seeding*——虽是 KMeans，但思想影响 KNN 初始化

### 28.2 教材章节

- *The Elements of Statistical Learning*（Hastie 等）第 13 章——近邻方法
- *Pattern Classification*（Duda 等）第 4 章——非参数方法
- *Machine Learning*（Mitchell）第 8 章——实例学习
- *Introduction to Statistical Learning*（James 等）第 4 章——分类

### 28.3 近似最近邻库

- **FAISS**（Facebook）：GPU 加速，百万级向量
- **Annoy**（Spotify）：基于随机投影树
- **HNSW**：分层可导航小世界图，当前最快
- **ScaNN**（Google）：各向异性量化
- **NGT**（Yahoo）：邻接图索引

### 28.4 进阶主题

- **距离度量学习**：LMNN、NCA、ITML——学习最优马氏距离
- **局部敏感哈希（LSH）**：高维近似 NN 的理论基石
- **核 KNN**：用核函数替代距离，捕捉非线性
- **半监督 KNN**：标签传播、标签扩散
- **流形学习 + KNN**：先降维到流形再 KNN

### 28.5 相关算法

- **局部线性嵌入（LLE）**：用近邻的线性组合表示点
- **ISOMAP**：用近邻图构造测地距离
- **t-SNE / UMAP**：用近邻关系做降维可视化
- **谱聚类**：用近邻图做聚类

---

[← 返回算法列表](../index.md)
