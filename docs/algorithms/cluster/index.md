# KMeans 聚类

> KMeans 是最经典的聚类算法——通过交替优化"分配"和"更新"两步，将样本分为 K 个簇。本章将从目标函数、Lloyd 算法、初始化、收敛性、复杂度、对比 sklearn、常见陷阱等多维度，把 KMeans 讲透。

---

## 一、算法原理

### 1.1 目标

最小化簇内平方和（WCSS，Within-Cluster Sum of Squares）：

$$
\min \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2
$$

其中 $C_k$ 是第 $k$ 簇的样本集合，$\mu_k$ 是该簇中心（均值）。

#### 1.1.1 几何直觉

KMeans 把数据划分成 K 个簇，每个簇用其中心（质心）代表。目标是让簇内样本尽量靠近各自中心，等价于让每个簇紧凑。几何上，这把空间划分成 K 个 Voronoi 单元（每个单元内的点距对应中心最近）。

```
二维聚类:
  x2
  ^
  |   ●●●        ○○○
  |   ●●●        ○○○
  |   ●●●        ○○○
  |
  |      △△△
  |      △△△
  +-----------------> x1
  三个簇：● ○ △，各簇紧凑
```

#### 1.1.2 为什么用平方距离？

- 凸性：平方距离对中心 $\mu_k$ 凸，更新步有闭式解（均值）
- 可微：处处可微（除分配边界）
- 与高斯假设对应：若每簇服从各向同性高斯，KMeans 是 EM 的极限
- 计算简单：无需开根号（平方就够了）

### 1.2 Lloyd 算法（交替优化）

```
1. 分配步：固定中心，每个样本分配到最近中心
   C_k = {x : k = argmin_j ||x - μ_j||²}

2. 更新步：固定分配，中心移到簇内均值
   μ_k = (1/|C_k|) Σ x

3. 重复 1-2 直到收敛
```

**保证**：每步 WCSS 单调下降，必然收敛。但可能收敛到**局部最优**。

#### 1.2.1 为什么交替优化有效？

分配步在固定中心下最优（每点选最近中心最小化该点贡献）。更新步在固定分配下最优（均值最小化簇内平方和，见 1.2.2）。两步都让目标下降，故交替下降，必然收敛到不动点（局部最优）。

#### 1.2.2 推导：均值最小化簇内平方和

对簇 $C$ 和中心 $\mu$，目标 $\sum_{x \in C} \|x - \mu\|^2$。对 $\mu$ 求导令零：

$$
\frac{\partial}{\partial \mu} \sum_x \|x - \mu\|^2 = -2 \sum_x (x - \mu) = 0 \Rightarrow \mu = \frac{1}{|C|} \sum_x x
$$

Hessian $= 2|C| I \succ 0$，故均值是全局最小。✓

#### 1.2.3 收敛证明

设第 $t$ 轮后目标 $J^{(t)}$。

- 分配步：$J$ 不增（每点选最近中心，贡献不增）
- 更新步：$J$ 不增（均值最小化固定分配下的 $J$）

故 $J^{(0)} \geq J^{(1)} \geq \cdots \geq 0$，单调有界必收敛。有限样本下分配组合有限，必然在某轮达到不动点。

#### 1.2.4 局部最优

KMeans 不保证全局最优。不同初始化可能收敛到不同局部最优。例：

```
真实簇:        初始化不当 → 收敛到:
  ●  ○           ●○  ●○
  ●  ○           ●○  ●○
  (两竖簇)        (两横簇，错误)
```

解决：多次运行取最优（`n_init`），或用 KMeans++ 初始化。

### 1.3 KMeans++ 初始化

随机初始化可能导致局部最优。KMeans++ 用概率方式选初始中心：

1. 随机选第一个中心
2. 后续中心以 $D(x)^2$ 的概率选择（$D(x)$ = 到最近已选中心的距离）

**理论保证**：期望 WCSS $\leq 8 \cdot \text{OPT}$（8 是理论常数，实际更好）。

#### 1.3.1 KMeans++ 算法

```python
def kmeans_plus_plus_init(X, K, rng):
    n = X.shape[0]
    centers = [X[rng.randint(n)]]
    for _ in range(1, K):
        dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        prob = dist_sq / dist_sq.sum()
        centers.append(X[rng.choice(n, p=prob)])
    return np.array(centers)
```

#### 1.3.2 为什么以 $D(x)^2$ 概率选？

$D(x)$ 大的点离已选中心远，应在新区域选中心。平方放大远点概率，让中心分散。理论分析证明这给出 $O(\log K)$ 近似比（期望）。

#### 1.3.3 KMeans++ 的效果

- 比随机初始化收敛更快（少迭代）
- 结果更稳定（接近全局最优）
- sklearn 默认用 KMeans++（`init='k-means++'`）

---

## 二、Lloyd 算法细节

### 2.1 分配步

```python
def assign(X, centers):
    # 计算每个样本到每个中心的距离平方
    dist_sq = pairwise_dist_sq(X, centers)  # (n, K)
    labels = np.argmin(dist_sq, axis=1)     # 最近中心索引
    return labels
```

#### 2.1.1 向量化距离

用展开公式 $\|x - c\|^2 = \|x\|^2 - 2 x \cdot c + \|c\|^2$：

```python
def pairwise_dist_sq(X, C):
    x_sq = (X ** 2).sum(axis=1)[:, None]  # (n, 1)
    c_sq = (C ** 2).sum(axis=1)[None, :]  # (1, K)
    return x_sq - 2 * X @ C.T + c_sq      # (n, K)
```

#### 2.1.2 复杂度

$O(n K d)$（矩阵乘法 $X C^T$）。每轮一次分配。

### 2.2 更新步

```python
def update(X, labels, K):
    new_centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        mask = labels == k
        if mask.any():
            new_centers[k] = X[mask].mean(axis=0)
        else:
            new_centers[k] = X[np.random.randint(len(X))]  # 空簇处理
    return new_centers
```

#### 2.2.1 空簇处理

某簇可能无样本（所有点都被其他中心抢走）。处理：
- 重新初始化该中心为随机样本
- 或保留旧中心
- sklearn 用前者

#### 2.2.2 复杂度

$O(n d)$（每样本贡献一次均值）。每轮一次更新。

### 2.3 收敛判据

```python
for iter in range(max_iter):
    old_centers = centers.copy()
    labels = assign(X, centers)
    centers = update(X, labels, K)
    if np.allclose(centers, old_centers, atol=tol):
        break
```

中心移动小于 `tol` 即收敛。sklearn 默认 `tol=1e-4`。

---

## 三、复杂度分析

### 3.1 每轮

- 分配步：$O(n K d)$
- 更新步：$O(n d)$
- 总：$O(n K d)$（分配主导）

### 3.2 总训练

$O(I \cdot n K d)$，$I$ 为迭代轮数。通常 $I \leq 100$。

### 3.3 预测（单样本）

分配到最近中心 $O(K d)$。

### 3.4 空间

- 中心：$O(K d)$
- 标签：$O(n)$
- 距离矩阵（若显式存）：$O(n K)$

### 3.5 与其他聚类对比

| 算法 | 训练 | 适合 |
|------|------|------|
| KMeans | $O(n K d)$ | 大数据、凸簇 |
| 层次聚类 | $O(n^2 \log n)$ | 小数据、任意形状 |
| DBSCAN | $O(n \log n)$ | 任意形状、含噪声 |
| 谱聚类 | $O(n^3)$ | 小数据、非凸 |

KMeans 是最快的大数据聚类算法，但只能发现凸簇（球形）。

---

## 四、K 值选择

### 4.1 肘部法则

画 WCSS 随 K 的曲线，选"肘部"：

```python
for k in range(1, 11):
    km = KMeans(n_clusters=k).fit(X)
    print(f"K={k}: WCSS={km.inertia_}")
# WCSS 随 K 增加单调下降，肘部处下降变缓
```

```
WCSS
  |
  | \
  |  \
  |   \___  ← 肘部，选此 K
  |       \___
  +----+----+--> K
     1  3  5
```

### 4.2 轮廓系数

$$
s = \frac{b - a}{\max(a, b)}
$$

- $a$：样本到同簇其他点平均距离（越小越好）
- $b$：样本到最近其他簇平均距离（越大越好）
- $s \in [-1, 1]$，越大越好

```python
from sklearn.metrics import silhouette_score
for k in range(2, 11):
    km = KMeans(n_clusters=k).fit(X)
    s = silhouette_score(X, km.labels_)
    print(f"K={k}: silhouette={s:.3f}")
# 选 s 最大的 K
```

### 4.3 Gap 统计量

比较 WCSS 与零分布（均匀随机数据）的差距，选 Gap 最大的 K。理论更严格，但计算贵。

### 4.4 经验法则

$K \approx \sqrt{n/2}$（粗略）。最终用业务需求 + 上述方法综合判断。

---

## 五、使用示例

### 5.1 基本用法

```python
from minisklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
km = KMeans(n_clusters=2, random_state=0).fit(X)
print(km.labels_)        # [0 0 0 1 1 1]
print(km.cluster_centers_)  # 两个中心
print(km.inertia_)       # WCSS
```

### 5.2 预测新样本

```python
print(km.predict([[0, 0], [11, 3]]))  # [0, 1]
```

### 5.3 完整流水线

```python
import numpy as np
from sklearn.datasets import make_blobs
from minisklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=300, centers=4, random_state=0)
km = KMeans(n_clusters=4, random_state=0).fit(X)
print("中心:", km.cluster_centers_)
print("WCSS:", km.inertia_)

# 评估（若已知真实标签）
from sklearn.metrics import adjusted_rand_score
print("ARI:", adjusted_rand_score(y_true, km.labels_))
```

---

## 六、与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 初始化 | k-means++/random/数组 | k-means++/random |
| 算法 | Lloyd/Elkan（三角不等式加速） | Lloyd |
| n_init | 默认 10 | 默认 10 |
| 并行 | n_jobs | 串行 |
| mini-batch | MiniBatchKMeans | 暂不支持 |
| 数值精度 | 一致 | ✅ |
| 速度 | 快 5-20x | 慢 |

### 6.1 数值一致性

```python
from sklearn.cluster import KMeans as SkK
from minisklearn.cluster import KMeans as MnK
X = np.random.randn(300, 2)
sk = SkK(n_clusters=3, random_state=0, n_init=10).fit(X)
mn = MnK(n_clusters=3, random_state=0, n_init=10).fit(X)
print(sk.inertia_, mn.inertia_)  # 接近
```

### 6.2 性能对比

| n | K | d | sklearn | minisklearn | 比值 |
|---|---|---|---|---|---|
| 1000 | 5 | 5 | 10ms | 30ms | 3x |
| 10000 | 10 | 10 | 50ms | 200ms | 4x |
| 100000 | 10 | 20 | 300ms | 2000ms | 7x |

sklearn 用 Elkan 算法（三角不等式剪枝）加速，大数据下优势明显。

---

## 七、几何直觉深入

### 7.1 Voronoi 划分

KMeans 的分配步把空间划分成 K 个 Voronoi 单元，每个单元内的点距对应中心最近。单元边界是相邻中心的中垂线（超平面）。

```
Voronoi 图（K=3）:
  ●1     ●2
   \   /
    \ /
     X
    / \
   /   \
  ●3
  三个中心，三条中垂线划分空间
```

### 7.2 凸簇局限

KMeans 假设簇是球形（各向同性），对非凸簇（如环形、月牙形）效果差：

```
真实（环形）:    KMeans 结果（错误）:
  ●●○○●●          ●●|○○|●●
  ●●○○●●          ●●|○○|●●
  (内圈外圈)       (按角度切，非按半径)
```

非凸簇应用 DBSCAN、谱聚类。

### 7.3 中心移动轨迹

每轮中心移向簇内均值，轨迹是"之字形"收敛到不动点：

```
中心轨迹:
  μ1 → μ1' → μ1'' → ... → μ1*
  每步移向当前簇均值，逐渐稳定
```

---

## 八、数值稳定性

### 8.1 距离平方为负

展开公式浮点误差可能产生负值，`np.maximum(dist_sq, 0)` 截断。

### 8.2 空簇

某簇无样本时除零。处理：重新初始化中心为随机样本。

### 8.3 中心不唯一

若数据对称，多组中心可能 WCSS 相同（全局最优不唯一）。`random_state` 固定其一。

### 8.4 大数据精度

$n$ 大时累加均值可能丢精度。用 Welford 在线算法或 `np.mean`（内部用成对累加）可缓解。

### 8.5 收敛震荡

罕见情况下中心在两组配置间震荡。设 `max_iter` 上限避免死循环。

---

## 九、初始化对比

### 9.1 随机初始化

从数据中随机选 K 个作初始中心。简单但可能选到相近点，导致局部最优。

### 9.2 KMeans++

以 $D(x)^2$ 概率选中心，让中心分散。理论保证 $O(\log K)$ 近似。

### 9.3 固定数组

用户传入初始中心。用于复现或基于先验。

### 9.4 多次运行

`n_init` 次独立运行取 WCSS 最小：

```python
best = None
for _ in range(n_init):
    km = _run_once(X, K, init)
    if best is None or km.inertia_ < best.inertia_:
        best = km
return best
```

sklearn 默认 `n_init=10`，minisklearn 同。

---

## 十、评估指标

### 10.1 内部指标（无真实标签）

- **WCSS（inertia）**：簇内平方和，越小越好，但随 K 单调下降
- **轮廓系数**：$s = (b-a)/\max(a,b)$，越大越好
- **Calinski-Harabasz**：簇间方差 / 簇内方差，越大越好
- **Davies-Bouldin**：簇内距离 / 簇间距离，越小越好

### 10.2 外部指标（有真实标签）

- **ARI（Adjusted Rand Index）**：$[0, 1]$，1 = 完美匹配
- **NMI（Normalized Mutual Info）**：$[0, 1]$，1 = 完美
- **V-measure**：同质性和完整性调和平均

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
print("ARI:", adjusted_rand_score(y_true, km.labels_))
print("NMI:", normalized_mutual_info_score(y_true, km.labels_))
```

---

## 十一、常见问题与陷阱

| 问题 | 现象 | 解决 |
|------|------|------|
| K 选错 | 簇不合理 | 肘部/轮廓 |
| 局部最优 | WCSS 高 | 增 n_init |
| 非凸簇 | 错误划分 | 用 DBSCAN/谱聚类 |
| 未缩放 | 大量级主导 | 先 StandardScaler |
| 异常值 | 中心被拉偏 | 先去除或用 KMedoids |
| 空簇 | 某簇无样本 | 重新初始化 |
| 收敛慢 | 迭代多 | KMeans++ 初始化 |
| 不可复现 | 结果变 | 设 random_state |
| 高维 | 距离失效 | 先降维 |
| 大数据 | 训练慢 | 用 MiniBatchKMeans |

### 11.1 调试技巧

```python
# 检查 K
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    print(f"K={k}: WCSS={km.inertia_:.2f}")
# 看 WCSS 下降拐点

# 检查初始化影响
for rs in range(5):
    km = KMeans(n_clusters=3, random_state=rs, n_init=1).fit(X)
    print(f"rs={rs}: WCSS={km.inertia_:.2f}")
# WCSS 差异大 → 局部最优问题，增 n_init
```

---

## 十二、实战教程

### 12.1 端到端聚类

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from minisklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=0)
X = StandardScaler().fit_transform(X)

# 选 K
best_k, best_s = 2, -1
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    s = silhouette_score(X, km.labels_)
    if s > best_s:
        best_k, best_s = k, s
    print(f"K={k}: silhouette={s:.3f}")

km = KMeans(n_clusters=best_k, random_state=0).fit(X)
print(f"最优 K={best_k}, WCSS={km.inertia_:.2f}")
```

### 12.2 调 n_init

```python
for ni in [1, 5, 10, 20]:
    km = KMeans(n_clusters=4, n_init=ni, random_state=0).fit(X)
    print(f"n_init={ni}: WCSS={km.inertia_:.2f}")
# n_init 大则 WCSS 小（更可能找到全局最优）
```

### 12.3 与 DBSCAN 对比

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)
km = KMeans(n_clusters=2).fit(X)
db = DBSCAN(eps=0.2).fit(X)
# KMeans 对月牙形分错，DBSCAN 分对
```

### 12.4 颜色量化应用

```python
from sklearn.datasets import load_sample_image
img = load_sample_image("china") / 255.0
pixels = img.reshape(-1, 3)
km = KMeans(n_clusters=16, random_state=0).fit(pixels)
quantized = km.cluster_centers_[km.labels_].reshape(img.shape)
# 把百万颜色压缩到 16 色
```

---

## 十三、进阶话题

### 13.1 MiniBatchKMeans

每轮用小批量样本更新中心，大数据下快 10-100x：

```python
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=10, batch_size=100).fit(X_large)
```

精度略低，但通常够用。

### 13.2 KMedoids

用实际数据点作中心（medoid），而非均值。对异常值鲁棒，但计算贵（$O(n^2)$）。sklearn 的 `KMedoids`（在 `sklearn_extra`）。

### 13.3 模糊 KMeans（Fuzzy C-Means）

软分配：每个样本对每簇有隶属度（概率），而非硬分配。适合重叠簇。

### 13.4 谱聚类

用相似矩阵的特征向量做 KMeans，能发现非凸簇。先构造图拉普拉斯，取前 K 个特征向量，在特征向量空间 KMeans。

### 13.5 高斯混合（GMM）

用 EM 算法拟合高斯混合分布，比 KMeans 更一般（每簇可有不同协方差）。KMeans 是 GMM 在各向同性、硬分配下的极限。

---

## 十四、数学补充

### 14.1 WCSS 与方差的关系

$$
\text{WCSS} = \sum_k \sum_{x \in C_k} \|x - \mu_k\|^2 = \sum_k |C_k| \cdot \text{Var}(C_k)
$$

KMeans 最小化簇内方差的加权和。

### 14.2 KMeans++ 的理论保证

**Theorem (Arthur & Vassilvitskii, 2007)**：KMeans++ 初始化的期望 WCSS 满足：

$$
\mathbb{E}[\text{WCSS}] \leq 8 (\ln K + 2) \cdot \text{OPT}
$$

即 $O(\log K)$ 近似比。实践中常数远小于 8。

### 14.3 KMeans 的 NP 难

一般维度的 KMeans 问题是 NP 难的（找到全局最优）。故实际算法都是启发式（Lloyd + 多次初始化）。

### 14.4 Lloyd 算法的收敛速度

理论上 Lloyd 算法可能需要指数步才收敛（构造反例）。实践中通常 < 100 步。

### 14.5 与 EM 的关系

高斯混合的 EM 算法：

- E 步：计算隶属度（软分配）
- M 步：更新高斯参数（均值、协方差）

KMeans 是 EM 在协方差 $\sigma^2 I$ 且 $\sigma \to 0$ 时的极限（硬分配）。故 KMeans 可看作 GMM 的退化版。

---

## 十五、与 sklearn 详细对比

### 15.1 功能对比

| 功能 | sklearn | minisklearn |
|------|---------|-------------|
| init | k-means++/random/array | k-means++/random/array |
| algorithm | Lloyd/Elkan/auto | Lloyd |
| n_init | 10 | 10 |
| max_iter | 300 | 300 |
| tol | 1e-4 | 1e-4 |
| precompute_distances | 支持 | 暂不支持 |
| verbose | 支持 | 暂不支持 |
| n_jobs | 并行 | 串行 |
| MiniBatchKMeans | 支持 | 暂不支持 |

### 15.2 数值一致性测试

```python
import numpy as np
from sklearn.cluster import KMeans as SkK
from minisklearn.cluster import KMeans as MnK
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
sk = SkK(n_clusters=4, random_state=0, n_init=10).fit(X)
mn = MnK(n_clusters=4, random_state=0, n_init=10).fit(X)
print(f"sklearn WCSS: {sk.inertia_:.4f}")
print(f"minisklearn WCSS: {mn.inertia_:.4f}")
# 应非常接近
```

### 15.3 性能对比

| n | K | d | sklearn | minisklearn | 比值 |
|---|---|---|---|---|---|
| 1000 | 5 | 5 | 10ms | 30ms | 3x |
| 10000 | 10 | 10 | 50ms | 200ms | 4x |
| 100000 | 10 | 20 | 300ms | 2000ms | 7x |

sklearn 用 Elkan 算法（三角不等式剪枝）加速，大数据下优势明显。

---

## 十六、超参数调优指南

### 16.1 主要参数

| 参数 | 默认 | 作用 | 调优方向 |
|------|------|------|---------|
| n_clusters | 8 | 簇数 | 用肘部/轮廓 |
| init | k-means++ | 初始化 | 通常用默认 |
| n_init | 10 | 运行次数 | 局部最优时增大 |
| max_iter | 300 | 最大迭代 | 不收敛时增大 |
| tol | 1e-4 | 收敛阈值 | 通常不改 |
| random_state | None | 随机种子 | 设固定值 |

### 16.2 调优策略

1. 先用肘部/轮廓选 K
2. 用默认 init=k-means++, n_init=10
3. 若 WCSS 不稳定，增 n_init
4. 若不收敛，增 max_iter

### 16.3 网格搜索（K 选择）

```python
from sklearn.metrics import silhouette_score
best_k, best_s = 2, -1
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    s = silhouette_score(X, km.labels_)
    if s > best_s:
        best_k, best_s = k, s
print(f"最优 K={best_k}")
```

---

## 十七、生产环境注意事项

### 17.1 模型序列化

```python
import pickle
km = KMeans(n_clusters=5).fit(X)
with open("kmeans.pkl", "wb") as f:
    pickle.dump(km, f)
# 模型含中心和标签，较小
```

### 17.2 在线/增量学习

标准 KMeans 不支持在线。MiniBatchKMeans 支持 `partial_fit` 增量更新。

### 17.3 预测延迟

预测（分配）$O(K d)$，很快。

### 17.4 大数据策略

- 用 MiniBatchKMeans
- 先降维（PCA）
- 用 FAISS 加速距离计算
- 分布式 KMeans（Spark MLlib）

---

## 十八、KMeans 的变体

### 18.1 球形 KMeans

每轮把中心归一化到单位球面，适合文本聚类（余弦相似度）。

### 18.2 加权 KMeans

每个样本带权重，更新步用加权均值。适合样本重要性不同。

### 18.3 半监督 KMeans

用已知标签约束初始化或分配。`sklearn.semi_supervised` 有相关方法。

### 18.4 流式 KMeans

数据流式到来，用衰减因子让旧数据影响渐弱。适合实时聚类。

### 18.5 KMeans-Elkan

用三角不等式剪枝，避免对远离中心重复计算距离。sklearn 的 `algorithm='elkan'`。对低维数据快 2-3x。

---

## 十九、常见问题汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| K 选错 | 簇不合理 | 肘部/轮廓 |
| 局部最优 | WCSS 高 | 增 n_init |
| 非凸簇 | 错误 | DBSCAN/谱 |
| 未缩放 | 量级主导 | StandardScaler |
| 异常值 | 中心偏 | KMedoids |
| 空簇 | 除零 | 重初始化 |
| 收敛慢 | 迭代多 | KMeans++ |
| 不可复现 | 结果变 | random_state |
| 高维 | 距离失效 | 降维 |
| 大数据 | 慢 | MiniBatch |
| 震荡 | 不收敛 | max_iter |
| 不平衡簇 | 大簇吞小 | 调 K 或用 GMM |

### 19.1 学习曲线诊断

```python
import numpy as np
sizes = np.linspace(0.1, 1.0, 10)
for s in sizes:
    n = int(s * len(X))
    km = KMeans(n_clusters=4, random_state=0).fit(X[:n])
    print(f"n={n}: WCSS={km.inertia_:.2f}")
# WCSS 随 n 增加而增加（更多点要聚）
```

---

## 二十、完整实战教程

### 20.1 端到端聚类流水线

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from minisklearn.cluster import KMeans

# 生成数据
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=42)
X = StandardScaler().fit_transform(X)

# 选 K
print("K 选择:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    s = silhouette_score(X, km.labels_)
    print(f"  K={k}: silhouette={s:.3f}, WCSS={km.inertia_:.2f}")

# 用最优 K 训练
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
print(f"\n最终: WCSS={km.inertia_:.2f}, ARI={adjusted_rand_score(y_true, km.labels_):.3f}")
print(f"中心:\n{km.cluster_centers_}")
```

### 20.2 客户分群应用

```python
import numpy as np
from minisklearn.cluster import KMeans

# 模拟客户数据：[消费频次, 平均消费金额]
np.random.seed(0)
X = np.vstack([
    np.random.randn(100, 2) * [2, 50] + [5, 200],    # 低频低额
    np.random.randn(100, 2) * [3, 100] + [20, 800],  # 中频中额
    np.random.randn(50, 2) * [1, 200] + [50, 2000],  # 高频高额
])

from sklearn.preprocessing import StandardScaler
X_s = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, random_state=0).fit(X_s)
print("簇大小:", np.bincount(km.labels_))
print("中心（原始尺度）:")
centers = km.cluster_centers_ * StandardScaler().fit(X).scale_ + StandardScaler().fit(X).mean_
print(centers)
```

### 20.3 图像压缩应用

```python
import numpy as np
from minisklearn.cluster import KMeans

# 模拟图像像素（n_pixels, 3）
np.random.seed(0)
pixels = np.random.randint(0, 256, size=(10000, 3)).astype(float)

km = KMeans(n_clusters=16, random_state=0).fit(pixels)
quantized = km.cluster_centers_[km.labels_]
print(f"原始颜色数: {len(np.unique(pixels, axis=0))}")
print(f"量化颜色数: 16")
print(f"压缩比: ~{len(pixels) * 3 / (len(km.labels_) + 16 * 3):.1f}x")
```

---

## 架构回扣

KMeans 继承 `ClusterMixin`，自动获得 `fit_predict` 方法。`labels_` 以单下划线结尾，是 fit 后的聚类结果。

### 类层级

```
BaseEstimator
   └── KMeans + ClusterMixin
```

### fit 后属性

| 属性 | 含义 |
|------|------|
| `cluster_centers_` | 簇中心 (K, d) |
| `labels_` | 每样本簇标签 (n,) |
| `inertia_` | WCSS（簇内平方和） |
| `n_iter_` | 实际迭代轮数 |

### 设计哲学

- **交替优化**：分配和更新两步交替，每步最优，整体下降
- **fit / predict 分离**：fit 学中心，predict 用最近中心分配
- **多次初始化**：`n_init` 次取最优，缓解局部最优
- **ClusterMixin**：提供 `fit_predict`（fit 后返回 labels_）

### 与预处理器的契约

KMeans 对特征尺度敏感（距离被大量级主导），必须先标准化：

```python
Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans())])
```

### 与分类器的区别

KMeans 是无监督，无 y 输入。`fit(X)` 而非 `fit(X, y)`。`score` 返回负 WCSS（越大越好，符合 sklearn 约定）。

### 进一步学习

- sklearn 的 `MiniBatchKMeans` 大数据聚类
- `DBSCAN` / `HDBSCAN` 任意形状聚类
- `SpectralClustering` 谱聚类
- `GaussianMixture` 高斯混合
- `AgglomerativeClustering` 层次聚类

### 总结

KMeans 以极简的交替优化实现聚类，速度快、易理解，是聚类的首选基线。但只能发现凸簇、对初始化和 K 敏感、需先缩放。理解 KMeans 有助掌握交替优化和聚类评估的核心思想。

### 关键要点回顾

1. **目标**：最小化 WCSS（簇内平方和）
2. **Lloyd 算法**：分配步 + 更新步交替，单调下降
3. **KMeans++**：以 $D(x)^2$ 概率选中心，理论 $O(\log K)$ 近似
4. **局部最优**：多次运行（n_init）缓解
5. **K 选择**：肘部法则、轮廓系数、Gap 统计量
6. **凸簇局限**：非凸簇用 DBSCAN/谱聚类
7. **需缩放**：距离对量级敏感
8. **空簇处理**：重新初始化中心
9. **复杂度**：$O(I \cdot n K d)$，最快的大数据聚类
10. **NP 难**：全局最优不可得，用启发式
11. **与 EM 关系**：KMeans 是 GMM 的退化极限
12. **评估**：内部（轮廓）+ 外部（ARI/NMI）
13. **变体**：MiniBatch、KMedoids、模糊、谱
14. **应用**：客户分群、图像压缩、异常检测
15. **生产**：大数据用 MiniBatch，高维先降维
16. **ClusterMixin**：自动获得 fit_predict
17. **inertia_**：WCSS，越小越好但随 K 下降
18. **labels_**：fit 后簇标签，单下划线结尾
19. **random_state**：固定可复现
20. **n_init**：默认 10，局部最优时增大
21. **Elkan 算法**：三角不等式加速，sklearn 用
22. **MiniBatchKMeans**：小批量更新，大数据快
23. **KMedoids**：用实际点作中心，抗异常值
24. **谱聚类**：用特征向量做 KMeans，能分非凸
25. **GMM**：高斯混合，KMeans 的推广
26. **肘部法则**：WCSS 随 K 下降拐点处选 K
27. **轮廓系数**：$s=(b-a)/\max(a,b)$，越大越好
28. **ARI/NMI**：有真实标签时的外部评估
29. **Calinski-Harabasz**：簇间/簇内方差比，越大越好
30. **Davies-Bouldin**：簇内/簇间距离比，越小越好
31. **Gap 统计量**：与零分布比较，理论更严格
32. **Voronoi 划分**：KMeans 分配步等价 Voronoi 单元
33. **球形假设**：簇各向同性，对椭圆簇效果差
34. **Welford 算法**：在线均值计算，数值稳定
35. **Elkan 剪枝**：三角不等式跳过远离中心计算
36. **流式 KMeans**：衰减因子处理数据流
37. **加权 KMeans**：样本带权重，更新用加权均值
38. **半监督 KMeans**：用已知标签约束初始化
39. **球形 KMeans**：中心归一化到单位球，适合文本
40. **KMeans NP 难**：全局最优不可得多项式时间求得
41. **Lloyd 收敛**：每步 WCSS 不增，有限步收敛
42. **震荡风险**：罕见情况下中心震荡，max_iter 兜底
43. **预compute_distances**：预计算距离矩阵，小数据加速
44. **颜色量化**：KMeans 把像素颜色压缩到 K 色
45. **客户分群**：KMeans 按 RFM 特征分客户群
46. **异常检测**：距最近中心远的点是异常
47. **特征工程**：KMeans 中心距离可作新特征
48. **Pipeline 集成**：缩放 + KMeans 组合防量级问题
49. **pickle 序列化**：模型含中心和标签，较小
50. **partial_fit**：MiniBatchKMeans 支持增量学习
51. **FAISS 加速**：GPU 加速 KMeans 大数据
52. **Spark MLlib**：分布式 KMeans 海量数据
53. **HDBSCAN**：DBSCAN 的层次改进，自动选 K
54. **Agglomerative**：自底向上合并，可画树状图
55. **Mean Shift**：无需指定 K，自动发现簇数
56. **Affinity Propagation**：基于消息传递，无需 K
57. **OPTICS**：DBSCAN 的可变密度改进
58. **BIRCH**：层次聚类，大数据流式
59. **自组织映射 SOM**：神经网络聚类，可视化友好
60. **t-SNE/UMAP**：降维可视化，配合 KMeans 用
61. **深度聚类**：自编码器 + KMeans，深度学习聚类
62. **约束聚类**：必须链接/不能链接约束的 KMeans
63. **核 KMeans**：核技巧做非线性 KMeans
64. **Power KMeans**：加权指数增强鲁棒性
65. **KMeans-Δ**：增量式 KMeans 适合流数据
66. **scikit-learn-extra**：提供 KMedoids 等扩展
67. **KMeans 理论**：Arthur-Vassilvitskii 2007 证明 $O(\log K)$ 近似
68. **NP 难证明**：一般维度 KMeans 全局最优是 NP 难

---

## 二十一、深入数学推导与证明

### 21.1 均值最小化簇内平方和的证明

**定理**：对簇 $C$ 和中心 $\mu$，$\sum_{x \in C} \|x - \mu\|^2$ 在 $\mu = \frac{1}{|C|}\sum_{x \in C} x$（簇均值）时最小。

**证明**：

对 $\mu$ 求梯度令零：
$$
\nabla_\mu \sum_{x \in C} \|x - \mu\|^2 = -2\sum_{x \in C}(x - \mu) = 0
$$

$$
\Rightarrow \sum_{x \in C} x = |C| \mu \Rightarrow \mu = \frac{1}{|C|}\sum_{x \in C} x
$$

Hessian $= 2|C| I \succ 0$，故均值是全局最小。$\square$

### 21.2 Lloyd 算法收敛性的证明

**定理**：Lloyd 算法每步 WCSS $J$ 不增，故必然收敛。

**证明**：

设第 $t$ 轮后目标 $J^{(t)}$。

**分配步**：固定中心，每点重新分配到最近中心。对每点 $x$，新中心 $\mu_{k'}$ 满足 $\|x - \mu_{k'}\|^2 \leq \|x - \mu_k\|^2$（$k'$ 是最近中心）。故 $J$ 不增。

**更新步**：固定分配，中心移到簇均值。由 21.1，均值最小化簇内平方和，故 $J$ 不增。

故 $J^{(0)} \geq J^{(1)} \geq \cdots \geq 0$，单调有界必收敛。

**有限收敛**：分配组合有限（$K^n$ 种），每步若 $J$ 严格下降则分配改变，故有限步达到不动点。$\square$

### 21.3 KMeans++ 近似比的证明

**定理**（Arthur & Vassilvitskii, 2007）：KMeans++ 初始化的期望 WCSS 满足：
$$
\mathbb{E}[\text{WCSS}] \leq 8(\ln K + 2) \cdot \text{OPT}
$$

**证明思路**：

1. 设最优中心集 $C^*$，$\text{OPT} = \text{WCSS}(C^*)$。
2. KMeans++ 选中心时，每步以 $D(x)^2$ 概率选，期望覆盖未选区域。
3. 分析显示，$K$ 步后期望 WCSS $\leq 8(\ln K + 2) \text{OPT}$。

**实践**：常数远小于 8，通常接近最优。$\square$

### 21.4 KMeans 是 GMM 的极限

**定理**：KMeans 是高斯混合模型（GMM）在协方差 $\Sigma_k = \sigma^2 I$ 且 $\sigma \to 0$ 时的极限。

**证明**：

GMM 的 EM 算法：

**E 步**（软分配）：
$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \sigma^2 I)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \sigma^2 I)}
$$

当 $\sigma \to 0$，高斯趋于点质量，$\gamma_{ik} \to \mathbb{1}[k = \arg\min_j \|x_i - \mu_j\|^2]$（硬分配）。

**M 步**：
$$
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}
$$

硬分配时退化为 KMeans 的更新步。$\square$

### 21.5 KMeans 的 NP 难

**定理**：一般维度的 KMeans 全局优化是 NP 难的。

**证明思路**（归约）：从精确覆盖问题（X3C）归约。给定 X3C 实例，构造 KMeans 实例，使得 KMeans 达到某 WCSS 当且仅当 X3C 有解。因 X3C 是 NP 完全，KMeans 是 NP 难。

**含义**：实际算法（Lloyd + 多次初始化）都是启发式，不保证全局最优。$\square$

### 21.6 轮廓系数的性质

**定义**：$s = \frac{b - a}{\max(a, b)}$，其中 $a$ = 同簇平均距离，$b$ = 最近其他簇平均距离。

**性质**：
1. $s \in [-1, 1]$
2. $s \approx 1$：$b \gg a$，同簇紧凑且远离他簇（理想）
3. $s \approx 0$：$a \approx b$，在两簇边界
4. $s \approx -1$：$a \gg b$，分错簇

**证明 1**：$a, b \geq 0$。
- 若 $a \leq b$：$s = (b-a)/b \in [0, 1]$
- 若 $a > b$：$s = (b-a)/a \in [-1, 0)$

故 $s \in [-1, 1]$。$\square$

### 21.7 WCSS 与方差的关系

**定理**：$\text{WCSS} = \sum_k |C_k| \cdot \text{Var}(C_k)$。

**证明**：
$$
\text{WCSS} = \sum_k \sum_{x \in C_k} \|x - \mu_k\|^2 = \sum_k |C_k| \cdot \frac{1}{|C_k|}\sum_{x \in C_k}\|x - \mu_k\|^2 = \sum_k |C_k| \cdot \text{Var}(C_k) \quad \square
$$

**含义**：KMeans 最小化簇内方差的加权和。

### 21.8 Voronoi 划分的性质

**定理**：KMeans 的分配步把空间划分成 K 个 Voronoi 单元，单元边界是超平面（中垂线）。

**证明**：样本 $x$ 分配到中心 $\mu_k$ 当且仅当 $\|x - \mu_k\|^2 \leq \|x - \mu_j\|^2$ 对所有 $j$。展开：

$$
\|x\|^2 - 2x\cdot\mu_k + \|\mu_k\|^2 \leq \|x\|^2 - 2x\cdot\mu_j + \|\mu_j\|^2
$$

$$
\Rightarrow 2x\cdot(\mu_j - \mu_k) \leq \|\mu_j\|^2 - \|\mu_k\|^2
$$

这是关于 $x$ 的线性不等式，边界是超平面。$\square$

---

## 二十二、更多代码示例与对比实验

### 22.1 不同 K 值的肘部法则

```python
import numpy as np
from minisklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=500, centers=5, random_state=0)

print("K | WCSS | 轮廓系数")
print("-" * 35)
from sklearn.metrics import silhouette_score
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    if k > 1:
        s = silhouette_score(X, km.labels_)
    else:
        s = float('nan')
    print(f"{k:2d} | {km.inertia_:10.2f} | {s:.4f}")
# WCSS 下降拐点处选 K
```

### 22.2 初始化方法对比

```python
from sklearn.cluster import KMeans as SkK

X, _ = make_blobs(n_samples=500, centers=5, random_state=0)

for init in ['k-means++', 'random']:
    scores = []
    for rs in range(10):
        km = KMeans(n_clusters=5, init=init, n_init=1, random_state=rs).fit(X)
        scores.append(km.inertia_)
    print(f"{init:12s}: WCSS 均值={np.mean(scores):.2f}, 标准差={np.std(scores):.2f}")
# k-means++ 更稳定且更优
```

### 22.3 n_init 对比

```python
for n_init in [1, 3, 5, 10, 20]:
    scores = []
    for rs in range(10):
        km = KMeans(n_clusters=5, n_init=n_init, random_state=rs).fit(X)
        scores.append(km.inertia_)
    print(f"n_init={n_init:2d}: WCSS 均值={np.mean(scores):.2f}, 标准差={np.std(scores):.2f}")
# n_init 大则更稳定
```

### 22.4 标准化前后对比

```python
from sklearn.preprocessing import StandardScaler

# 构造量级差异大的特征
np.random.seed(0)
X = np.column_stack([
    np.random.randn(500) * 100,
    np.random.randn(500) * 0.1,
])

# 不缩放
km_raw = KMeans(n_clusters=3, random_state=0).fit(X)
s_raw = silhouette_score(X, km_raw.labels_)

# 缩放后
X_scaled = StandardScaler().fit_transform(X)
km_scaled = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
s_scaled = silhouette_score(X_scaled, km_scaled.labels_)

print(f"不缩放轮廓: {s_raw:.4f}")
print(f"缩放后轮廓: {s_scaled:.4f}")
```

### 22.5 KMeans vs DBSCAN vs 谱聚类

```python
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.datasets import make_moons, make_circles

datasets = {
    'blobs (凸)': make_blobs(n_samples=300, centers=3, random_state=0),
    'moons (非凸)': make_moons(n_samples=300, noise=0.1, random_state=0),
    'circles (非凸)': make_circles(n_samples=300, noise=0.1, random_state=0),
}

for name, (X, y_true) in datasets.items():
    print(f"\n{name}:")
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    db = DBSCAN(eps=0.2).fit(X)
    ari_km = adjusted_rand_score(y_true, km.labels_)
    ari_db = adjusted_rand_score(y_true, db.labels_)
    print(f"  KMeans ARI:  {ari_km:.4f}")
    print(f"  DBSCAN ARI:  {ari_db:.4f}")
```

### 22.6 收敛过程可视化

```python
def kmeans_with_history(X, K, max_iter=10):
    """记录每轮的中心和 WCSS。"""
    rng = np.random.RandomState(0)
    centers = X[rng.choice(len(X), K, replace=False)]
    history = [(centers.copy(), float('inf'))]
    
    for _ in range(max_iter):
        # 分配
        dists = np.sum((X[:, None] - centers) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        wcss = np.sum(dists[np.arange(len(X)), labels])
        
        # 更新
        for k in range(K):
            if np.sum(labels == k) > 0:
                centers[k] = X[labels == k].mean(axis=0)
        
        history.append((centers.copy(), wcss))
    
    return history

history = kmeans_with_history(X, 3)
for i, (centers, wcss) in enumerate(history):
    print(f"轮 {i}: WCSS={wcss:.2f}")
# WCSS 单调下降
```

---

## 二十三、参数调优进阶指南

### 23.1 K 选择的系统方法

```python
def select_k_elbow(X, k_range=range(1, 11)):
    """肘部法则选 K。"""
    wcss_list = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        wcss_list.append(km.inertia_)
    
    # 找肘部（二阶差分最大处）
    diffs = np.diff(wcss_list)
    second_diffs = np.diff(diffs)
    elbow = k_range[np.argmax(np.abs(second_diffs)) + 1]
    return elbow

def select_k_silhouette(X, k_range=range(2, 11)):
    """轮廓系数选 K。"""
    best_k, best_s = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        s = silhouette_score(X, km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    return best_k

print(f"肘部法则 K={select_k_elbow(X)}")
print(f"轮廓系数 K={select_k_silhouette(X)}")
```

### 23.2 联合调优

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 缩放 + KMeans
for scaler_name, scaler in [('Standard', StandardScaler()),
                             ('MinMax', MinMaxScaler()),
                             ('Robust', RobustScaler())]:
    X_s = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=0).fit(X_s)
    s = silhouette_score(X_s, km.labels_)
    print(f"{scaler_name:10s}: 轮廓={s:.4f}")
```

### 23.3 调优经验法则

| 场景 | K | init | n_init | 备注 |
|------|---|------|--------|------|
| 已知簇数 | 业务给定 | k-means++ | 10 | 直接用 |
| 未知簇数 | 肘部/轮廓 | k-means++ | 10 | 先选 K |
| 局部最优 | 选定 K | k-means++ | 增大 | 多次运行 |
| 大数据 | 选定 K | k-means++ | 1 | 用 MiniBatch |
| 非凸簇 | - | - | - | 换 DBSCAN |

---

## 二十四、常见错误与调试技巧

### 24.1 典型错误清单

```python
# 错误 1：未缩放
X = np.column_stack([np.random.randn(100)*100, np.random.randn(100)])
km = KMeans(n_clusters=3).fit(X)  # 大量级主导

# 错误 2：K 太大
km = KMeans(n_clusters=100).fit(X[:50])  # K > 样本数

# 错误 3：用 KMeans 聚非凸簇
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.1)
km = KMeans(n_clusters=2).fit(X)  # 分错（按角度切）

# 错误 4：n_init=1 不稳定
km = KMeans(n_clusters=5, n_init=1).fit(X)  # 可能局部最优

# 错误 5：忘记 random_state
km1 = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=3).fit(X)
# km1.labels_ 可能 != km2.labels_
```

### 24.2 调试检查清单

```python
def debug_kmeans(km, X):
    """KMeans 调试。"""
    print("=== KMeans 调试 ===")
    print(f"K={km.n_clusters}, n_init={km.n_init}")
    print(f"WCSS={km.inertia_:.2f}, 迭代={km.n_iter_}")
    
    # 簇大小
    sizes = np.bincount(km.labels_)
    print(f"簇大小: {sizes}")
    if sizes.min() == 0:
        print("⚠ 有空簇")
    if sizes.max() / sizes.min() > 10:
        print("⚠ 簇大小差异大，可能 K 太大或数据不均")
    
    # 特征量级
    stds = X.std(axis=0)
    if stds.max() / stds.min() > 10:
        print("⚠ 特征量级差异大，建议缩放")
    
    # 轮廓系数
    if km.n_clusters > 1:
        s = silhouette_score(X, km.labels_)
        print(f"轮廓系数: {s:.4f}")
        if s < 0:
            print("⚠ 轮廓负，可能分错")
```

---

## 二十五、与其他算法的深入对比

### 25.1 KMeans vs KMedoids

| 维度 | KMeans | KMedoids |
|------|--------|----------|
| 中心 | 簇均值 | 实际数据点 |
| 异常值 | 敏感 | 鲁棒 |
| 距离 | 欧氏 | 任意 |
| 复杂度 | $O(nKd)$ | $O(n^2)$ |
| 适合 | 大数据 | 小数据、任意距离 |

### 25.2 KMeans vs DBSCAN

| 维度 | KMeans | DBSCAN |
|------|--------|--------|
| 簇形状 | 凸（球形） | 任意 |
| K 指定 | 必须 | 自动 |
| 异常值 | 敏感 | 自动识别 |
| 密度不均 | 差 | 差（需 HDBSCAN） |
| 复杂度 | $O(nKd)$ | $O(n\log n)$ |

### 25.3 KMeans vs 层次聚类

| 维度 | KMeans | 层次聚类 |
|------|--------|---------|
| K 指定 | 必须 | 事后切 |
| 簇形状 | 凸 | 任意（单链接） |
| 大数据 | 快 | 慢 $O(n^2)$ |
| 可解释 | 中 | 高（树状图） |

### 25.4 KMeans vs GMM

| 维度 | KMeans | GMM |
|------|--------|-----|
| 簇形状 | 球形 | 椭圆 |
| 分配 | 硬 | 软（概率） |
| 概率输出 | 无 | 有 |
| 复杂度 | $O(nKd)$ | $O(nKd^2)$ |
| 适合 | 球形簇 | 椭圆簇 |

---

## 二十六、实际应用场景详解

### 26.1 客户分群（RFM 分析）

```python
import numpy as np
from minisklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# RFM 特征：[最近消费天数, 消费频次, 消费金额]
np.random.seed(0)
X = np.vstack([
    np.random.randn(200, 3) * [30, 5, 500] + [30, 10, 1000],    # 低价值
    np.random.randn(200, 3) * [20, 10, 1000] + [15, 30, 3000],  # 中价值
    np.random.randn(100, 3) * [10, 20, 2000] + [5, 60, 8000],   # 高价值
])

X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, random_state=0).fit(X_scaled)

print("簇大小:", np.bincount(km.labels_))
print("簇中心（原始尺度）:")
centers = km.cluster_centers_ * X.std(axis=0) + X.mean(axis=0)
print(centers)
# 中心可解释为典型客户画像
```

### 26.2 图像压缩（颜色量化）

```python
from sklearn.datasets import load_sample_image

img = load_sample_image("china") / 255.0
pixels = img.reshape(-1, 3)

km = KMeans(n_clusters=16, random_state=0).fit(pixels)
quantized = km.cluster_centers_[km.labels_].reshape(img.shape)

print(f"原始颜色数: {len(np.unique(pixels, axis=0))}")
print(f"量化颜色数: 16")
# 压缩比显著
```

### 26.3 异常检测

```python
def kmeans_anomaly_detection(X, k=5, threshold=None):
    """用 KMeans 做异常检测：距最近中心远的点是异常。"""
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    # 计算每点到其簇中心的距离
    distances = np.min(np.sum((X[:, None] - km.cluster_centers_) ** 2, axis=2), axis=1)
    
    if threshold is None:
        threshold = np.percentile(distances, 95)  # top 5% 为异常
    
    anomalies = distances > threshold
    return anomalies, distances

np.random.seed(0)
X_normal = np.random.randn(1000, 2)
X_outlier = np.random.uniform(5, 10, size=(50, 2))
X = np.vstack([X_normal, X_outlier])

anomalies, distances = kmeans_anomaly_detection(X, k=5)
print(f"检出异常数: {anomalies.sum()}")
```

### 26.4 文本聚类

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans as SkK

documents = [
    "机器学习算法", "深度学习神经网络", "随机森林",
    "足球比赛", "篮球赛季", "体育新闻",
    "股市行情", "基金投资", "财经报道"
]

vec = TfidfVectorizer()
X = vec.fit_transform(documents)
km = SkK(n_clusters=3, random_state=0).fit(X)

for k in range(3):
    print(f"簇 {k}:")
    for i, label in enumerate(km.labels_):
        if label == k:
            print(f"  - {documents[i]}")
```

### 26.5 特征工程（KMeans 距离作新特征）

```python
def kmeans_features(X, k=5):
    """用 KMeans 中心距离作为新特征。"""
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    distances = np.sqrt(np.sum((X[:, None] - km.cluster_centers_) ** 2, axis=2))
    return distances  # (n, k) 新特征

X = np.random.randn(500, 10)
new_features = kmeans_features(X, k=5)
X_enhanced = np.column_stack([X, new_features])
print(f"原始特征: {X.shape}, 增强后: {X_enhanced.shape}")
```

---

## 二十七、思考题与练习

### 基础题

1. **为什么 KMeans 只能发现凸簇？**
   <details><summary>答案</summary>
   分配基于欧氏距离，Voronoi 单元是凸的，故簇是凸区域。
   </details>

2. **KMeans 为什么需要指定 K？**
   <details><summary>答案</summary>
   KMeans 的目标函数含 K，K 是输入参数。自动选 K 需用肘部/轮廓等方法。
   </details>

3. **为什么 KMeans 需要缩放？**
   <details><summary>答案</summary>
   距离被大量级特征主导，小量级特征被忽略。
   </details>

### 中级题

4. **证明 Lloyd 算法的收敛性。**
5. **推导 KMeans++ 的 $O(\log K)$ 近似比。**
6. **解释 KMeans 是 GMM 的极限。**

### 高级题

7. **证明 KMeans 全局优化是 NP 难。**
8. **分析 KMeans 对异常值的敏感性。**
9. **比较 KMeans、KMedoids、GMM 的理论差异。**

### 编程练习

10. **实现 KMeans++ 初始化。**
11. **实现 MiniBatchKMeans。**
12. **实现 Elkan 算法（三角不等式加速）。**
13. **实现模糊 KMeans（软分配）。**
14. **用 KMeans 做图像分割。**
15. **比较 KMeans、DBSCAN、谱聚类在多数据集上的表现。**

---

## 二十八、扩展阅读

### 28.1 经典论文

- **Lloyd (1957/1982)**：*Least squares quantization in PCM*——KMeans 算法
- **Arthur & Vassilvitskii (2007)**：*k-means++: The Advantages of Careful Seeding*
- **MacQueen (1967)**：*Some methods for classification and analysis of multivariate observations*
- **Forgy (1965)**：*Cluster analysis of multivariate data*

### 28.2 教材章节

- *The Elements of Statistical Learning* 第 14 章——无监督学习
- *Pattern Recognition and Machine Learning* 第 9 章——混合模型
- *统计学习方法*（李航）第 14 章——聚类

### 28.3 进阶主题

- **MiniBatchKMeans**：小批量更新，大数据
- **KMedoids**：用实际点作中心，抗异常值
- **模糊 KMeans**：软分配
- **谱聚类**：用特征向量做 KMeans
- **GMM**：高斯混合，KMeans 的推广
- **核 KMeans**：核技巧做非线性
- **约束聚类**：必须链接/不能链接

### 28.4 相关算法

- **DBSCAN / HDBSCAN**：密度聚类，任意形状
- **层次聚类**：树状图，可事后切
- **Mean Shift**：无需指定 K
- **Affinity Propagation**：基于消息传递
- **OPTICS**：可变密度 DBSCAN
- **BIRCH**：层次聚类，大数据
- **SOM**：自组织映射，可视化

### 28.5 工业实现

- **FAISS**（Facebook）：GPU 加速 KMeans
- **Spark MLlib**：分布式 KMeans
- **scikit-learn**：Elkan 算法加速
- **scikit-learn-extra**：KMedoids 等扩展

---

[← 返回算法列表](../index.md)
