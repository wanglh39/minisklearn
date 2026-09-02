# PCA 主成分分析

> PCA 通过 SVD 分解找到数据方差最大的方向，实现线性降维。它是无监督降维的"Hello World"——既能在压缩数据时尽量少丢信息，又能把高维数据可视化到二维平面上。

---

## 一、算法原理

### 1.1 核心思想

找到数据方差最大的方向，将高维数据投影到低维子空间。

为什么是"方差最大"？直觉上，方差大的方向上数据分布得更"开"，信息量更丰富；方差小的方向上数据挤成一团，几乎不携带可区分信息。PCA 的目标就是保留那些"看得见差别"的方向，丢掉那些"看起来都一样"的方向。

更正式地说，PCA 寻找一个低维子空间，使得原数据在该子空间上的投影与原数据之间的**重构误差最小**。可以证明：**最小重构误差等价于最大投影方差**，二者是同一问题的两种表述。

设中心化后的数据矩阵为 $X' \in \mathbb{R}^{n \times d}$（$n$ 个样本、$d$ 个特征），我们要找一个单位向量 $w$（$\|w\|=1$），使投影 $X'w$ 的方差最大：

$$
\max_{w} \; \mathrm{Var}(X'w) = \max_{w} \; \frac{1}{n-1} w^T X'^T X' w \quad \text{s.t.} \quad w^T w = 1
$$

记协方差矩阵 $C = \dfrac{X'^T X'}{n-1}$，问题变成：

$$
\max_{w} \; w^T C w \quad \text{s.t.} \quad w^T w = 1
$$

用拉格朗日乘子法，构造 $L(w, \lambda) = w^T C w - \lambda(w^T w - 1)$，对 $w$ 求导并令其为 0：

$$
\frac{\partial L}{\partial w} = 2 C w - 2 \lambda w = 0 \;\Rightarrow\; C w = \lambda w
$$

这正是**特征值方程**！所以最优 $w$ 是 $C$ 的特征向量，目标值 $w^T C w = \lambda$。要使方差最大，就选**最大特征值对应的特征向量**。第二主成分要与第一主成分正交，所以选第二大特征值对应的特征向量，以此类推。

### 1.2 SVD 实现

直接对协方差矩阵 $C$ 做特征分解在理论上是对的，但实践中 sklearn 与 minisklearn 都用 SVD 实现：

$$
X' = U S V^T \quad \text{（中心化后 SVD 分解）}
$$

其中：
- $U \in \mathbb{R}^{n \times r}$ 是左奇异向量矩阵，列向量正交
- $S \in \mathbb{R}^{r \times r}$ 是对角阵，对角元素为奇异值 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r \geq 0$
- $V \in \mathbb{R}^{d \times r}$ 是右奇异向量矩阵，列向量正交
- $r = \mathrm{rank}(X')$

- **主成分** = $V$ 的前 $k$ 列（右奇异向量），即 `components_ = V[:, :k].T`
- **降维**：$X_{\text{new}} = X' V_{:,k}$，把 $n \times d$ 的数据投影到 $n \times k$
- **解释方差**：$\lambda_k = S_k^2 / (n-1)$，因为 $\sigma_k^2 / (n-1)$ 正是协方差矩阵的特征值
- **解释方差比**：$\text{ratio}_k = \lambda_k / \sum_j \lambda_j$，表示第 $k$ 个主成分占总方差的比例

为什么 $V$ 就是主成分？推导如下：

$$
C = \frac{X'^T X'}{n-1} = \frac{(U S V^T)^T (U S V^T)}{n-1} = \frac{V S U^T U S V^T}{n-1} = \frac{V S^2 V^T}{n-1}
$$

这里用到了 $U^T U = I$（左奇异向量正交）。对比特征分解 $C = V \Lambda V^T$，立即得到 $\Lambda = S^2 / (n-1)$。所以 $V$ 的列就是 $C$ 的特征向量，$S^2/(n-1)$ 的对角元素就是特征值。

### 1.3 为什么用 SVD 而非协方差矩阵？

协方差矩阵 $C = X'^T X' / (n-1) = V S^2 V^T / (n-1)$

- **SVD 直接给出 $V$ 和 $S$**，无需显式计算 $X'^T X'$（避免精度损失）
- **SVD 对数值条件更鲁棒**：$X'^T X'$ 会把条件数平方。若 $X'$ 的条件数是 $\kappa$，则 $X'^T X'$ 的条件数是 $\kappa^2$。当 $\kappa = 10^6$ 时，$X'^T X'$ 的条件数变成 $10^{12}$，双精度浮点数（约 15 位有效数字）会损失 12 位，几乎只剩 3 位有效数字
- **SVD 有成熟的 LAPACK 实现**（如 `dgesdd`），数值稳定且速度快
- **内存效率**：当 $n \gg d$ 或 $d \gg n$ 时，可以用截断 SVD（` randomized_svd`）只算前 $k$ 个奇异向量，避免存储完整的 $U$

举个数值例子：假设 $X'$ 有一个奇异值 $\sigma_1 = 10^6$，另一个 $\sigma_2 = 1$。条件数 $\kappa(X') = 10^6$，对 SVD 来说完全没问题。但 $X'^T X'$ 的特征值是 $10^{12}$ 和 $1$，条件数 $10^{12}$，在双精度下 $\sigma_2^2 = 1$ 可能被 $10^{12}$ 的舍入误差淹没，导致第二个主成分算不准。

### 1.4 白化

白化使各主成分方差为 1：$X_{\text{whiten}} = X_{\text{new}} / \sqrt{\lambda_k}$

更完整地，白化变换是：

$$
X_{\text{whiten}} = X' V \Lambda^{-1/2} = U S V^T V \Lambda^{-1/2} = U S \Lambda^{-1/2} = U \sqrt{n-1}
$$

白化后数据的协方差矩阵是单位阵 $I$，即各维度不相关且方差都是 1。白化常用于：
- **ICA（独立成分分析）** 的预处理步骤
- **深度学习的输入归一化**，让各特征尺度一致
- **k-means 聚类** 前的去相关，让欧氏距离更合理

### 1.5 几何直觉

PCA 可以理解为对数据云做"最佳拟合椭球"：

1. **中心化**：把数据云的质心平移到原点
2. **找长轴**：第一主成分是椭球最长的轴方向（方差最大）
3. **找次长轴**：第二主成分是与第一主成分正交的方向中，方差最大的
4. **以此类推**：直到找出 $d$ 个正交方向
5. **降维**：只保留前 $k$ 个最长的轴，把数据投影到这 $k$ 个轴张成的子空间

可视化描述：想象一个三维数据云像一根被压扁的雪茄，长方向方差大，两个短方向方差小。PCA 会先找到雪茄的长轴（第一主成分），然后把数据投影到这个方向上，三维数据就压缩成一维，且保留了大部分"形状信息"。

如果数据云是个各方向差不多长的球，PCA 就无能为力——所有方向方差都差不多，丢掉任何方向都会损失大量信息。这种情况下应该考虑非线性降维（如 t-SNE、UMAP）。

### 1.6 中心化的必要性

PCA 一定要先中心化（减去均值），原因有二：

1. **方差定义**：方差是相对于均值计算的，$\mathrm{Var}(X) = E[(X - \mu)^2]$。如果不中心化，$X'^T X'$ 不是协方差矩阵，而是二阶矩矩阵，方向会偏向远离原点的方向
2. **SVD 的几何意义**：SVD 找的是"过原点"的最佳低维子空间。如果数据质心不在原点，过原点的子空间显然不是最佳拟合

sklearn 的 PCA 在 `fit` 时会计算并保存 `mean_`，在 `transform` 时先减去 `mean_` 再投影，在 `inverse_transform` 时加回 `mean_`。

```python
def fit(self, X, y=None):
    self.mean_ = X.mean(axis=0)
    X_centered = X - self.mean_
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    self.components_ = Vt[:self.n_components]          # 形状 (k, d)
    self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
    return self

def transform(self, X):
    return (X - self.mean_) @ self.components_.T       # 形状 (n, k)

def inverse_transform(self, X_transformed):
    return X_transformed @ self.components_ + self.mean_
```

### 1.7 解释方差与选择 k

怎么决定保留几个主成分？看**累计解释方差比**：

```python
import numpy as np
from minisklearn.decomposition import PCA

pca = PCA(n_components=min(X.shape)).fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(cumvar)
# [0.62, 0.85, 0.92, 0.96, 0.98, 0.99, 1.0]
#          ↑ 保留 2 个就解释了 85% 方差
```

常用阈值：
- **95%**：最常用，信息损失可接受
- **99%**：要求高保真
- **"mle"**：用 MLE 自动估计 k（sklearn 支持）
- **肘部法则**：画碎石图（scree plot），在方差比下降明显变缓处截断

---

## 二、数学推导详解

### 2.1 从最小重构误差出发

另一种等价表述：PCA 寻找 $k$ 维子空间，使数据到该子空间的投影与原数据的平方误差最小。

设子空间由正交基 $W = [w_1, \dots, w_k]$ 张成，数据点 $x_i$ 的投影是 $W W^T x_i$（假设已中心化）。重构误差：

$$
J(W) = \sum_{i=1}^n \|x_i - W W^T x_i\|^2
$$

展开：

$$
J(W) = \sum_i \|x_i\|^2 - \sum_i \|W^T x_i\|^2 = \mathrm{const} - \mathrm{tr}(W^T X'^T X' W)
$$

最小化 $J(W)$ 等价于最大化 $\mathrm{tr}(W^T X'^T X' W) = \mathrm{tr}(W^T (n-1) C W)$，即最大化投影方差之和。这正是 1.1 节的优化问题。

### 2.2 拉格朗日乘子法求解

对 $\max \, \mathrm{tr}(W^T C W)$ s.t. $W^T W = I_k$，构造拉格朗日函数：

$$
L(W, \Lambda) = \mathrm{tr}(W^T C W) - \mathrm{tr}(\Lambda^T (W^T W - I))
$$

其中 $\Lambda$ 是 $k \times k$ 的拉格朗日乘子矩阵。对 $W$ 求导（用到 $\frac{\partial}{\partial W}\mathrm{tr}(W^T C W) = 2 C W$ 和 $\frac{\partial}{\partial W}\mathrm{tr}(\Lambda^T W^T W) = 2 W \Lambda$）：

$$
\frac{\partial L}{\partial W} = 2 C W - 2 W \Lambda = 0 \;\Rightarrow\; C W = W \Lambda
$$

这说明 $W$ 的列向量是 $C$ 的特征向量，$\Lambda$ 是对角阵（可对角化），对角元素是对应特征值。要最大化 $\mathrm{tr}(W^T C W) = \mathrm{tr}(\Lambda)$，就选最大的 $k$ 个特征值。

### 2.3 SVD 与特征分解的等价性

设 $X' = U S V^T$，则：

$$
X' X'^T = U S V^T V S U^T = U S^2 U^T \quad \text{（左奇异向量是 } X'X'^T \text{ 的特征向量）}
$$
$$
X'^T X' = V S U^T U S V^T = V S^2 V^T \quad \text{（右奇异向量是 } X'^T X' \text{ 的特征向量）}
$$

所以：
- $V$ 的列 = $X'^T X'$ 的特征向量 = 协方差矩阵 $C$ 的特征向量 = **主成分方向**
- $U$ 的列 = $X' X'^T$ 的特征向量 = 数据在主成分方向上的归一化坐标
- $S^2 / (n-1)$ 的对角元素 = $C$ 的特征值 = **解释方差**

### 2.4 降维与重构的矩阵形式

降维（编码）：

$$
Z = X' V_k \in \mathbb{R}^{n \times k}
$$

其中 $V_k = V[:, :k]$ 是前 $k$ 个右奇异向量。$Z$ 的行就是每个样本在主成分坐标系下的 $k$ 维表示。

重构（解码）：

$$
\hat{X}' = Z V_k^T = X' V_k V_k^T \in \mathbb{R}^{n \times d}
$$

$V_k V_k^T$ 是到 $V_k$ 列空间的正交投影矩阵。$\hat{X}'$ 是 $X'$ 在 $k$ 维主成分子空间上的正交投影。

加回均值得到原始空间的重构：

$$
\hat{X} = \hat{X}' + \mathbf{1} \mu^T = X' V_k V_k^T + \mathbf{1} \mu^T
$$

重构误差的平方和：

$$
\sum_i \|x_i - \hat{x}_i\|^2 = \sum_{j=k+1}^{d} \lambda_j (n-1) = \sum_{j=k+1}^{d} \sigma_j^2
$$

即丢弃的奇异值的平方和。这就是为什么"保留大方差方向 = 最小重构误差"。

---

## 三、实现细节

### 3.1 完整实现

```python
import numpy as np
from ..base import BaseEstimator, TransformerMixin

class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # 1. 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 2. SVD 分解
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # U: (n, min(n,d)), S: (min(n,d),), Vt: (min(n,d), d)

        # 3. 确定保留的主成分数
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        # 4. 保存主成分（每行是一个主成分方向）
        self.components_ = Vt[:n_components]              # (k, d)
        self singular_values_ = S[:n_components]          # (k,)

        # 5. 解释方差
        total_var = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ = (S[:n_components] ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # 6. 白化预处理
        if self.whiten:
            self.whiten_scale_ = np.sqrt(self.explained_variance_)
        else:
            self.whiten_scale_ = np.ones(n_components)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        Z = X_centered @ self.components_.T               # (n, k)
        if self.whiten:
            Z = Z / self.whiten_scale_
        return Z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.whiten:
            X = X * self.whiten_scale_
        return X @ self.components_ + self.mean_

    def get_covariance(self):
        """返回估计的协方差矩阵 (d, d)。"""
        return self.components_.T @ np.diag(self.explained_variance_) @ self.components_
```

### 3.2 fit_transform 的优化

`fit_transform` 可以直接用 SVD 的左奇异向量，避免重复计算：

```python
def fit_transform(self, X, y=None):
    X = np.asarray(X, dtype=np.float64)
    self.mean_ = X.mean(axis=0)
    X_centered = X - self.mean_
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 直接得到投影：X_centered @ V = U S
    Z = U[:, :self.n_components] * S[:self.n_components]  # (n, k)
    # ... 保存 components_ 等 ...
    return Z
```

因为 $X' V = U S V^T V = U S$，左奇异向量乘奇异值就是投影结果。这比先 `fit` 再 `transform` 少一次矩阵乘法。

### 3.3 explained_variance_ratio_ 的计算

```python
total_var = (S ** 2).sum() / (n_samples - 1)
explained_variance_ratio = (S[:k] ** 2) / (n_samples - 1) / total_var
                          = (S[:k] ** 2) / (S ** 2).sum()
```

注意分母是**所有**奇异值的平方和，不是前 $k$ 个。这样 ratio 之和才 ≤ 1，且能反映"保留了多少信息"。

---

## 四、使用示例

### 4.1 基础降维

```python
import numpy as np
from minisklearn.decomposition import PCA

# 生成 100 个 5 维数据
rng = np.random.RandomState(42)
X = rng.randn(100, 5)

# 降到 2 维
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)
print(X_new.shape)                              # (100, 2)
print(pca.explained_variance_ratio_)            # [0.27, 0.24, ...]
print(pca.explained_variance_ratio_.sum())      # 0.51（保留了 51% 方差）
```

### 4.2 选择保留多少主成分

```python
# 先算所有主成分
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print(cumvar)
# [0.27, 0.51, 0.74, 0.89, 1.0]

# 找到累计方差 ≥ 95% 的最小 k
k = np.searchsorted(cumvar, 0.95) + 1
print(f"保留 {k} 个主成分即可解释 95% 方差")

# 用这个 k 重新降维
pca = PCA(n_components=k).fit(X)
```

### 4.3 可视化高维数据

```python
import matplotlib.pyplot as plt
from minisklearn.decomposition import PCA
from minisklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)                # 4 维
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)                       # 降到 2 维

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("PCA 降维后的鸢尾花数据")
plt.show()
```

### 4.4 重构与误差

```python
pca = PCA(n_components=2).fit(X)
X_new = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_new)

reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"重构 MSE: {reconstruction_error:.4f}")
# 理论值 = sum(丢弃的 explained_variance) = 1 - sum(保留的 ratio)
```

### 4.5 白化

```python
pca = PCA(n_components=2, whiten=True)
X_whiten = pca.fit_transform(X)
print(np.cov(X_whiten, rowvar=False))
# [[1. 0.]      ← 各主成分方差为 1
#  [0. 1.]]     ← 且不相关
```

### 4.6 完整可运行示例

```python
import numpy as np
from minisklearn.decomposition import PCA
from minisklearn.preprocessing import StandardScaler
from minisklearn.datasets import load_iris

# 1. 加载数据
X, y = load_iris(return_X_y=True)
print(f"原始数据形状: {X.shape}")                # (150, 4)

# 2. 标准化（PCA 对尺度敏感，建议先标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"降维后形状: {X_pca.shape}")              # (150, 2)
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计: {pca.explained_variance_ratio_.sum():.2%}")

# 4. 重构
X_reconstructed = pca.inverse_transform(X_pca)
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"重构 MSE: {mse:.4f}")

# 5. 查看主成分方向
print("主成分方向 (每行是一个主成分):")
print(pca.components_)
```

### 4.7 错误示例

```python
# 错误 1：忘记中心化（手动实现时）
# 直接对 X 做 SVD 而不减均值，主成分会偏向远离原点的方向
X = np.array([[100, 100], [101, 101], [102, 102]])
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(Vt)  # 方向可能不对，因为数据质心在 (101, 101) 而非原点

# 正确做法
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
print(Vt)  # 这才是主成分方向

# 错误 2：用 transform 之前没 fit
pca = PCA(n_components=2)
# pca.transform(X)  # ← 报错：AttributeError: 'PCA' object has no attribute 'mean_'

# 错误 3：n_components 大于数据秩
X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])  # 秩为 1
pca = PCA(n_components=2).fit(X)
# 不会报错，但 explained_variance_[1] 接近 0，第二个主成分无意义
print(pca.explained_variance_)  # [大, ~0]
```

### 4.8 对比示例：标准化 vs 不标准化

```python
# 不标准化：尺度大的特征主导主成分
X = np.column_stack([np.random.randn(100) * 100,    # 特征 1 尺度 100
                     np.random.randn(100)])          # 特征 2 尺度 1
pca = PCA(n_components=1).fit(X)
print(pca.components_)  # [~1, ~0]，主成分几乎全是特征 1

# 标准化后：两个特征同等重要
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=1).fit(X_scaled)
print(pca.components_)  # [~0.7, ~0.7]，两个特征都有贡献
```

---

## 五、与 sklearn 对比

### 5.1 API 一致性

| 特性 | minisklearn PCA | sklearn PCA |
|------|----------------|-------------|
| `n_components` | int 或 None | int、float、'mle'、None |
| `whiten` | bool | bool 或 'arbitrary_variance' |
| `fit` / `transform` | ✓ | ✓ |
| `fit_transform` | ✓ | ✓ |
| `inverse_transform` | ✓ | ✓ |
| `components_` | (k, d) | (k, d) |
| `explained_variance_` | ✓ | ✓ |
| `explained_variance_ratio_` | ✓ | ✓ |
| `singular_values_` | ✓ | ✓ |
| `mean_` | ✓ | ✓ |
| `n_components_` | ✓ | ✓ |
| `noise_variance_` | ✗ | ✓ |
| `score` / `score_samples` | ✗ | ✓（对数似然） |
| 随机 SVD | ✗ | `svd_solver='randomized'` |
| 增量 PCA | ✗ | `IncrementalPCA` |
| 稀疏 PCA | ✗ | `SparsePCA` |

### 5.2 n_components 的灵活指定

sklearn 支持用 float 指定"保留的方差比"：

```python
# sklearn：自动选择 k 使得累计方差 ≥ 95%
pca = PCA(n_components=0.95).fit(X)
print(pca.n_components_)  # 自动算出 k=3

# minisklearn 需要手动算
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
k = np.searchsorted(cumvar, 0.95) + 1
pca = PCA(n_components=k).fit(X)
```

### 5.3 性能对比

```python
import time
import numpy as np

X = np.random.randn(10000, 100)

# minisklearn
from minisklearn.decomposition import PCA as MiniPCA
t0 = time.time()
pca_mini = MiniPCA(n_components=10).fit(X)
t_mini = time.time() - t0

# sklearn
from sklearn.decomposition import PCA as SkPCA
t0 = time.time()
pca_sk = SkPCA(n_components=10).fit(X)
t_sk = time.time() - t0

print(f"minisklearn: {t_mini:.3f}s")
print(f"sklearn:     {t_sk:.3f}s")
# sklearn 更快，因为用了更优的 LAPACK 路径和可能的随机 SVD
```

### 5.4 数值结果对比

```python
X = np.random.RandomState(0).randn(100, 5)

pca_mini = MiniPCA(n_components=2).fit(X)
pca_sk = SkPCA(n_components=2).fit(X)

# 主成分方向可能差一个正负号（SVD 的符号不确定性）
print(np.abs(pca_mini.components_) - np.abs(pca_sk.components_))
# 接近 0，说明绝对值一致

# 解释方差应该完全一致
print(pca_mini.explained_variance_ - pca_sk.explained_variance_)
# 接近 0
```

### 5.5 SVD 的符号不确定性

SVD 分解不唯一：如果 $U, S, V$ 是一个 SVD，那么 $-U, S, -V$ 也是。所以不同实现得到的 `components_` 可能差一个正负号。sklearn 用 `svd_flip` 约定符号（让 $U$ 每列绝对值最大的元素为正），minisklearn 可以照做：

```python
def svd_flip(U, Vt):
    """约定 SVD 的符号：让 U 每列绝对值最大的元素为正。"""
    signs = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]
    return U, Vt
```

---

## 六、复杂度分析

### 6.1 时间复杂度

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 中心化 | $O(nd)$ | 减均值 |
| SVD 分解 | $O(\min(nd^2, n^2 d))$ | 全 SVD |
| 截断 SVD（前 $k$） | $O(ndk)$ | 随机化 SVD |
| transform | $O(nk)$ | 矩阵乘法 |
| inverse_transform | $O(nd)$ | 矩阵乘法 |

对于 $n \gg d$（样本多于特征），全 SVD 是 $O(nd^2)$；对于 $d \gg n$，是 $O(n^2 d)$。

当 $k \ll \min(n, d)$ 时，随机化 SVD（如 sklearn 的 `svd_solver='randomized'`）只需 $O(ndk)$，远快于全 SVD。

### 6.2 空间复杂度

| 存储 | 大小 |
|------|------|
| `mean_` | $O(d)$ |
| `components_` | $O(kd)$ |
| `explained_variance_` | $O(k)$ |
| SVD 中间结果 | $O(\min(n, d)^2)$ |

### 6.3 实测耗时

```python
import numpy as np, time
from minisklearn.decomposition import PCA

for n, d in [(1000, 50), (10000, 100), (10000, 500)]:
    X = np.random.randn(n, d)
    t0 = time.time()
    PCA(n_components=10).fit(X)
    print(f"n={n}, d={d}: {time.time()-t0:.3f}s")
```

---

## 七、数值稳定性

### 7.1 中心化后的 SVD vs 协方差矩阵特征分解

```python
import numpy as np

# 构造条件数大的矩阵
rng = np.random.RandomState(0)
U_, _ = np.linalg.qr(rng.randn(5, 5))
V_, _ = np.linalg.qr(rng.randn(5, 5))
S = np.diag([1e6, 1e3, 1, 1e-3, 1e-6])
X = U_ @ S @ V_.T * 100

# 方法 1：SVD（推荐）
X_c = X - X.mean(axis=0)
_, S_svd, Vt_svd = np.linalg.svd(X_c, full_matrices=False)
var_svd = (S_svd ** 2) / (X.shape[0] - 1)

# 方法 2：协方差矩阵特征分解
C = np.cov(X, rowvar=False)
eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]

print("SVD:    ", var_svd)
print("eigend: ", eigvals)
# 大特征值一致，小特征值 SVD 更准
```

### 7.2 处理零方差方向

如果某特征是常数（方差为 0），中心化后该列全为 0，对应奇异值为 0。SVD 能正确处理，但要注意：
- `explained_variance_ratio_` 中会有 0
- 白化时除以 $\sqrt{0}$ 会得到 inf，需要加 epsilon

```python
# 安全的白化
eps = 1e-8
whiten_scale = np.sqrt(explained_variance + eps)
```

### 7.3 重复特征值

如果协方差矩阵有重复特征值，对应的特征向量不唯一（任何正交旋转都是解）。这意味着主成分方向在重复特征值子空间内可以任意旋转。实践中这会导致：
- 不同实现得到不同的 `components_`
- 但 `explained_variance_` 和降维后的几何结构（子空间本身）是唯一的

### 7.4 浮点数误差

```python
# 理论上 X @ components_.T @ components_ 应该等于 X（在主成分子空间内）
# 但浮点数会有误差
X = np.random.randn(100, 5)
pca = PCA(n_components=5).fit(X)
X_proj = pca.inverse_transform(pca.transform(X))
print(np.allclose(X, X_proj))  # True，误差在 1e-15 量级
```

---

## 八、常见问题与陷阱

### 8.1 PCA 对尺度敏感

PCA 找的是方差最大的方向，如果特征尺度不同，尺度大的特征会主导主成分：

```python
X = np.column_stack([np.random.randn(100) * 1000,   # 尺度 1000
                     np.random.randn(100)])           # 尺度 1
pca = PCA(n_components=1).fit(X)
print(pca.components_)  # [~1, ~0]，主成分几乎全是特征 1
```

**解决**：先标准化 `StandardScaler`，再做 PCA。这等价于对**相关矩阵**做 PCA，而非协方差矩阵。

### 8.2 PCA 是线性的

PCA 只能找线性子空间。如果数据有非线性结构（如瑞士卷形），PCA 会失败：

```python
# 瑞士卷数据
t = np.random.uniform(0, 1, 500) * 2 * np.pi
h = np.random.uniform(-1, 1, 500)
X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])  # 3 维瑞士卷

pca = PCA(n_components=2).fit(X)
# PCA 找到的 2 维平面不能很好地"展开"瑞士卷
# 应该用流形学习：sklearn.manifold.LocallyLinearEmbedding
```

### 8.3 PCA 不考虑类别标签

PCA 是无监督的，降维时不用 $y$。如果目标是分类，可能降维后类别反而混在一起。这时考虑 **LDA（线性判别分析）**，它用类别信息找最有判别力的方向。

### 8.4 n_components 太大

```python
X = np.random.randn(100, 5)
pca = PCA(n_components=10).fit(X)  # 5 维数据降到 10 维？
# sklearn 会报错或截断到 5
# minisklearn 需要检查：n_components <= min(n_samples, n_features)
```

### 8.5 解释方差比不等于信息量

方差大不等于信息多。对于分类任务，方差大的方向可能不含判别信息。一个反例：

```python
# 特征 1 方差大但与标签无关
# 特征 2 方差小但完美区分标签
X = np.column_stack([np.random.randn(200) * 10,
                     np.where(np.random.randn(200) > 0, 1, -1) * 0.1])
y = (X[:, 1] > 0).astype(int)

pca = PCA(n_components=1).fit(X)
print(pca.components_)  # [~1, ~0]，PCA 选了方差大但无用的特征 1
```

### 8.6 训练集和测试集要用同一个 PCA

```python
# 正确
pca = PCA(n_components=10).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)   # 用训练集的 PCA

# 错误：测试集重新 fit
pca_test = PCA(n_components=10).fit(X_test)  # ← 数据泄露 + 维度不一致
X_test_pca = pca_test.transform(X_test)
```

用 Pipeline 可以避免这个错误：

```python
from minisklearn.pipeline import Pipeline
pipe = Pipeline([('pca', PCA(n_components=10)),
                 ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)       # PCA 只在训练集上 fit
pipe.predict(X_test)             # PCA 用训练集参数 transform 测试集
```

### 8.7 explained_variance_ratio_ 之和可能小于 1

当 `n_components < min(n_samples, n_features)` 时，`explained_variance_ratio_` 只包含前 $k$ 个，之和小于 1。这是正常的，表示保留了部分方差。

---

## 九、实际使用教程

### 9.1 标准降维流程

```python
import numpy as np
from minisklearn.preprocessing import StandardScaler
from minisklearn.decomposition import PCA
from minisklearn.linear_model import LogisticRegression
from minisklearn.pipeline import Pipeline
from minisklearn.model_selection import cross_val_score

# 1. 加载数据（假设 X, y 已就绪）
# 2. 构建流水线
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),    # 保留 95% 方差（需手动算 k）
    ('clf', LogisticRegression()),
])

# 3. 交叉验证评估
scores = cross_val_score(pipe, X, y, cv=5)
print(f"准确率: {scores.mean():.3f} ± {scores.std():.3f}")

# 4. 训练最终模型
pipe.fit(X, y)
```

### 9.2 用 PCA 加速训练

当特征维度很高时，先降维可以大幅加速后续模型训练：

```python
# 原始：10000 维
X_train.shape  # (1000, 10000)

# 降到 100 维
pca = PCA(n_components=100).fit(X_train)
X_train_pca = pca.transform(X_train)  # (1000, 100)
X_test_pca = pca.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_pca, y_train)  # 比在 10000 维上快得多
```

### 9.3 用 PCA 去噪

降维再重构可以去掉小方差方向上的噪声：

```python
# 加噪声的图像
X_noisy = X_clean + np.random.randn(*X_clean.shape) * 0.5

# PCA 去噪：保留主要成分，丢弃小方差（噪声）成分
pca = PCA(n_components=50).fit(X_noisy)
X_denoised = pca.inverse_transform(pca.transform(X_noisy))

# 评估
mse_noisy = np.mean((X_noisy - X_clean) ** 2)
mse_denoised = np.mean((X_denoised - X_clean) ** 2)
print(f"去噪前 MSE: {mse_noisy:.4f}")
print(f"去噪后 MSE: {mse_denoised:.4f}")  # 应该更小
```

### 9.4 用 PCA 做特征工程

主成分本身可以作为新特征：

```python
pca = PCA(n_components=10).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 用主成分作为新特征训练模型
clf = LogisticRegression().fit(X_train_pca, y_train)
# 主成分之间不相关，对线性模型友好
```

### 9.5 可视化碎石图

```python
import matplotlib.pyplot as plt

pca = PCA().fit(X)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'r-o')
plt.xlabel('主成分编号')
plt.ylabel('解释方差比 / 累计')
plt.axhline(0.95, color='g', linestyle='--', label='95%')
plt.legend()
plt.title('碎石图')
plt.show()
```

### 9.6 高维数据的随机 SVD

当 $d$ 很大（如 10000）且 $k$ 很小（如 10）时，全 SVD 太慢。sklearn 的随机 SVD 只算前 $k$ 个：

```python
# sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='randomized', random_state=42).fit(X)
# 复杂度 O(ndk) 而非 O(nd^2)

# minisklearn 暂不支持，可以用截断 SVD 库
from scipy.sparse.linalg import svds
U, S, Vt = svds(X_centered, k=10)
```

---

## 十、变体与扩展

### 10.1 增量 PCA（IncrementalPCA）

当数据太大放不进内存时，可以分批 fit：

```python
# sklearn
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=10, batch_size=100)
for batch in np.array_split(X, 10):  # 分 10 批
    ipca.partial_fit(batch)
X_new = ipca.transform(X)
```

### 10.2 稀疏 PCA（SparsePCA）

普通 PCA 的主成分是所有特征的线性组合，系数通常都非零。稀疏 PCA 加 $L_1$ 正则让系数稀疏，便于解释：

```python
# sklearn
from sklearn.decomposition import SparsePCA
spca = SparsePCA(n_components=5, alpha=1).fit(X)
print(spca.components_)  # 很多系数为 0
```

### 10.3 核 PCA（KernelPCA）

用核技巧做非线性 PCA：

```python
# sklearn
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10).fit(X)
X_new = kpca.transform(X)  # 非线性降维
```

### 10.4 截断 SVD（TruncatedSVD）

不中心化的 PCA，常用于文本数据（LSA 潜在语义分析）：

```python
# sklearn
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100).fit(X_tfidf)  # 不中心化
X_lsa = svd.transform(X_tfidf)
```

---

## 十一、架构回扣

### 11.1 TransformerMixin 与 fit_transform

PCA 继承 `TransformerMixin`，自动获得 `fit_transform`。`TransformerMixin` 的默认实现是：

```python
class TransformerMixin:
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

PCA 可以覆盖这个方法用 SVD 的左奇异向量直接得到投影，避免重复计算（见 3.2 节）。

### 11.2 双下划线属性的语义

`components_`、`mean_`、`explained_variance_` 等以单下划线结尾的属性是 **fit 后学出的参数**，区别于 `n_components`、`whiten` 等 `__init__` 参数。这是 sklearn 的命名约定：
- `__init__` 参数：用户传入的**超参数**，不随 fit 改变
- `xxx_`（下划线结尾）：fit 后学出的**学习参数**，predict/transform 时用

`clone` 函数只复制 `__init__` 参数，不复制 `xxx_` 属性，得到"未训练的同参数副本"。

### 11.3 与 BaseEstimator 的协作

```python
class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
```

- `BaseEstimator` 提供 `get_params`/`set_params`（通过反射 `__init__` 签名）
- `BaseEstimator` 提供 `__repr__`（如 `PCA(n_components=2, whiten=True)`）
- `TransformerMixin` 提供 `fit_transform`
- MRO（方法解析顺序）：`PCA → TransformerMixin → BaseEstimator → object`

### 11.4 在 Pipeline 中的角色

PCA 作为转换器，可以放在 Pipeline 的非最后位置：

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),       # 转换器
    ('clf', LogisticRegression()),       # 最后一步：估计器
])
```

Pipeline 的 `fit` 会对前两步调用 `fit_transform`，对最后一步调用 `fit`。PCA 的 `fit_transform` 学到 `mean_`、`components_` 并返回降维后的数据。

### 11.5 与 GridSearchCV 的协作

PCA 的 `n_components` 可以网格搜索：

```python
from minisklearn.model_selection import GridSearchCV

grid = GridSearchCV(pipe, {
    'pca__n_components': [5, 10, 20, 50],
    'clf__C': [0.1, 1, 10],
}, cv=5)
grid.fit(X, y)
print(grid.best_params_)  # {'pca__n_components': 20, 'clf__C': 1}
```

`pca__n_components` 通过 Pipeline 的 `set_params` 路由到 `PCA.set_params(n_components=...)`。

### 11.6 估计器检查

sklearn 有 `check_estimator` 验证估计器是否符合 API 规范。PCA 应该通过：
- `fit` 返回 self
- `transform` 返回正确形状
- `fit_transform` 等价于 `fit` + `transform`
- `inverse_transform(transform(X))` 近似恢复 X
- `get_params`/`set_params` 往返一致
- `clone` 不复制拟合状态

---

## 十二、进阶话题

### 12.1 PCA 与 KLT 变换

PCA 本质上是 Karhunen-Loève 变换（KLT）的离散版本。KLT 是信号处理中"最优变换"——在均方误差意义下，用 $k$ 个系数重构信号，KLT 的误差最小。PCA 就是数据集上的 KLT。

### 12.2 PCA 与因子分析

因子分析模型：$x = \mu + W z + \epsilon$，其中 $z$ 是隐因子，$\epsilon$ 是噪声。PCA 可以看作因子分析的特例（$\epsilon$ 各向同性且方差趋于 0）。但因子分析更一般，允许 $\epsilon$ 各维度方差不同。

### 12.3 PCA 的概率解释

可以给 PCA 一个概率模型：$x | z \sim \mathcal{N}(W z + \mu, \sigma^2 I)$，$z \sim \mathcal{N}(0, I)$。最大似然解的 $W$ 就是主成分方向（差一个旋转）。这叫概率 PCA（PPCA），是因子分析的特例。

### 12.4 PCA 与自编码器

线性自编码器（无激活函数）的最优解就是 PCA 子空间。自编码器最小化 $\|x - W_2 W_1 x\|^2$，最优 $W_1$ 的行空间就是主成分子空间。深度自编码器可以看作非线性 PCA 的推广。

### 12.5 robust PCA

普通 PCA 对离群点敏感（一个大离群点可以完全改变主成分方向）。Robust PCA 把数据分解为低秩 + 稀疏：$X = L + S$，$L$ 是低秩（主成分结构），$S$ 是稀疏（离群点）。用凸优化求解：

```python
# sklearn 没有直接实现，可以用 spc 库或手动实现
# from rpca import r_pca
# L, S = r_pca(X).fit()
```

---

## 十三、总结

| 要点 | 内容 |
|------|------|
| 核心思想 | 找方差最大的正交方向，投影降维 |
| 数学基础 | 协方差矩阵特征分解 / SVD |
| 实现 | 中心化 + SVD + 取前 k 个右奇异向量 |
| 复杂度 | $O(nd^2)$（全 SVD），$O(ndk)$（截断 SVD） |
| 数值稳定 | 用 SVD 而非协方差矩阵特征分解 |
| 常见陷阱 | 忘记标准化、忘记中心化、训练测试用不同 PCA |
| 与 sklearn | API 一致，sklearn 功能更全（随机 SVD、mle、score） |
| 适用场景 | 线性降维、可视化、去噪、特征工程、加速训练 |
| 不适用 | 非线性结构、分类任务（考虑 LDA）、尺度敏感未标准化 |

---

## 十四、更深入的数学推导与证明

### 14.1 PCA 的最优性严格证明

**定理**：设 $X' \in \mathbb{R}^{n \times d}$ 已中心化，$C = X'^T X' / (n-1)$ 为协方差矩阵。在所有秩 $k$ 的正交投影矩阵 $P$（$P^T = P$，$P^2 = P$，$\text{rank}(P) = k$）中，$P = V_k V_k^T$ 最小化重构误差 $\|X' - X'P\|_F^2$，其中 $V_k$ 是 $C$ 前 $k$ 大特征值对应的特征向量。

**证明**：

$$
\|X' - X'P\|_F^2 = \text{tr}((X' - X'P)^T(X' - X'P)) = \text{tr}(X'^T X') - \text{tr}(P^T X'^T X' P)
$$

第一项是常数。最小化误差等价于最大化 $\text{tr}(P^T X'^T X' P) = (n-1)\text{tr}(P^T C P)$。

设 $P = U U^T$，$U \in \mathbb{R}^{d \times k}$，$U^T U = I_k$（正交基）。由 Ky Fan 定理：

$$
\max_{U^T U = I_k} \text{tr}(U^T C U) = \sum_{i=1}^k \lambda_i
$$

等号当 $U$ 的列是前 $k$ 大特征值对应的特征向量时成立。$\square$

### 14.2 解释方差比的统计意义

**命题**：$\text{ratio}_k = \lambda_k / \sum_j \lambda_j$ 等于"第 $k$ 个主成分方向上的方差占总方差的比例"。

**证明**：总方差 $\text{tr}(C) = \sum_j \text{Var}(X_j) = \sum_j \lambda_j$（迹等于特征值之和）。第 $k$ 主成分 $z_k = X' v_k$ 的方差 $\text{Var}(z_k) = v_k^T C v_k = \lambda_k$。故比例为 $\lambda_k / \text{tr}(C)$。$\square$

### 14.3 白化后的协方差为单位阵

**命题**：白化变换 $Z = X' V \Lambda^{-1/2}$ 后，$\text{Cov}(Z) = I$。

**证明**：

$$
\text{Cov}(Z) = \Lambda^{-1/2} V^T C V \Lambda^{-1/2} = \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I
$$

其中 $V^T C V = \Lambda$（特征分解）。$\square$

### 14.4 PCA 与 KLT 的等价性

Karhunen-Loève 变换（KLT）对随机向量 $x$ 用其协方差矩阵的特征向量做展开 $x = \sum_k z_k v_k$。PCA 是 KLT 在有限样本上的经验版本：用样本协方差矩阵代替总体协方差矩阵。当 $n \to \infty$ 时，样本协方差矩阵收敛到总体协方差矩阵，PCA 收敛到 KLT。

### 14.5 降维后数据的协方差

**命题**：降维后 $Z = X' V_k$ 的协方差矩阵为 $\text{diag}(\lambda_1, \dots, \lambda_k)$。

**证明**：

$$
\text{Cov}(Z) = \frac{1}{n-1} Z^T Z = \frac{1}{n-1} V_k^T X'^T X' V_k = V_k^T C V_k = \Lambda_k
$$

降维后各主成分不相关（协方差对角），方差为对应特征值。这是 PCA"去相关"性质的来源。$\square$

### 14.6 PCA 的旋转不变性

**命题**：对数据先做正交变换 $Q$（$Q^T Q = I$）再做 PCA，主成分方向是 $Q^T v_k$，解释方差不变。

**证明**：变换后协方差 $C' = Q^T C Q$。若 $C v = \lambda v$，则 $C' (Q^T v) = Q^T C Q Q^T v = \lambda (Q^T v)$。故 $C'$ 的特征向量是 $Q^T v$，特征值（解释方差）不变。$\square$

推论：PCA 对数据的旋转不敏感（主成分跟着旋转），但对尺度敏感（非均匀缩放改变协方差矩阵特征值）。

---

## 十五、更多代码示例与对比实验

### 15.1 PCA vs LDA：有监督 vs 无监督降维

```python
import numpy as np
from minisklearn.decomposition import PCA

# 构造数据：类别信息在方差小的方向上
rng = np.random.RandomState(0)
n = 200
X = np.column_stack([rng.randn(n) * 10,                      # 方差大，与类别无关
                     rng.choice([-1, 1], n)])                 # 方差小，完美区分类别
y = (X[:, 1] > 0).astype(int)

# PCA 选方差大的方向（无用）
pca = PCA(n_components=1).fit(X)
print("PCA 主成分:", pca.components_[0].round(3))  # [~1, 0]，选了无用方向

# LDA 会选方差小但有判别力的方向（需 sklearn）
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
```

### 15.2 不同 n_components 的重构误差

```python
import numpy as np
from minisklearn.decomposition import PCA

X = np.random.RandomState(42).randn(200, 10)
for k in range(1, 11):
    pca = PCA(n_components=k).fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_rec) ** 2)
    var_kept = pca.explained_variance_ratio_.sum()
    print(f"k={k:2d}: 重构MSE={mse:.4f}, 保留方差={var_kept:.2%}")
```

### 15.3 PCA 去噪实验

```python
import numpy as np
from minisklearn.decomposition import PCA

rng = np.random.RandomState(0)
# 生成低秩信号 + 噪声
true_dim = 5
X_clean = rng.randn(300, 50) @ rng.randn(50, true_dim)  # 秩 5
X_noisy = X_clean + rng.randn(*X_clean.shape) * 5

for k in [5, 10, 20, 50]:
    pca = PCA(n_components=k).fit(X_noisy)
    X_denoised = pca.inverse_transform(pca.transform(X_noisy))
    mse = np.mean((X_denoised - X_clean) ** 2)
    print(f"k={k:2d}: 去噪后MSE={mse:.4f}")
# k=5 附近最优，k 太大保留噪声，k 太小丢信号
```

### 15.4 标准化对 PCA 的影响

```python
import numpy as np
from minisklearn.decomposition import PCA
from minisklearn.preprocessing import StandardScaler

X = np.column_stack([np.random.randn(200) * 100,   # 大尺度
                     np.random.randn(200),
                     np.random.randn(200) * 10])

print("不标准化:")
pca = PCA(n_components=2).fit(X)
print("  解释方差比:", pca.explained_variance_ratio_.round(3))
print("  主成分:", pca.components_.round(3))

print("标准化后:")
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)
print("  解释方差比:", pca.explained_variance_ratio_.round(3))
print("  主成分:", pca.components_.round(3))
```

### 15.5 PCA 用于异常检测

```python
import numpy as np
from minisklearn.decomposition import PCA

rng = np.random.RandomState(0)
X = rng.randn(200, 5)
X[0] = [10, 10, 10, 10, 10]  # 异常点

pca = PCA(n_components=3).fit(X)
X_rec = pca.inverse_transform(pca.transform(X))
reconstruction_error = np.sum((X - X_rec) ** 2, axis=1)

print("重构误差最大的 5 个点:", np.argsort(reconstruction_error)[-5:])
print("异常点（索引0）误差:", reconstruction_error[0].round(2))
# 异常点不能被低维子空间很好重构，误差最大
```

### 15.6 增量 PCA 模拟

```python
import numpy as np
from minisklearn.decomposition import PCA

# 模拟增量 PCA：分批 fit 后平均（简化版，不严格）
def incremental_pca(X, k, batch_size):
    n = X.shape[0]
    components_acc = np.zeros((k, X.shape[1]))
    n_batches = 0
    for i in range(0, n, batch_size):
        batch = X[i:i+batch_size]
        if len(batch) < k:
            continue
        pca = PCA(n_components=k).fit(batch)
        components_acc += np.abs(pca.components_)  # 取绝对值避免符号问题
        n_batches += 1
    return components_acc / n_batches

X = np.random.randn(1000, 20)
avg_components = incremental_pca(X, k=5, batch_size=100)
print("增量估计的平均主成分:", avg_components.round(3))
```

---

## 十六、参数调优指南

### 16.1 选择 n_components 的方法

| 方法 | 适用场景 | 代码 |
|------|---------|------|
| 固定 k | 已知目标维度 | `PCA(n_components=2)` |
| 累计方差阈值 | 保留大部分信息 | 手动算 `searchsorted(cumvar, 0.95)` |
| 碎石图肘部 | 直观判断 | 画 `explained_variance_ratio_` 找拐点 |
| MLE | 数据驱动 | sklearn 的 `n_components='mle'` |
| 交叉验证 | 下游任务驱动 | GridSearchCV 搜 `pca__n_components` |

### 16.2 用交叉验证选 n_components

```python
from minisklearn.model_selection import GridSearchCV
from minisklearn.pipeline import Pipeline
from minisklearn.linear_model import LogisticRegression
from minisklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA()),
                 ('clf', LogisticRegression())])

grid = GridSearchCV(pipe, {'pca__n_components': [2, 5, 10, 20, 50]}, cv=5)
grid.fit(X, y)
print(f"最优 n_components: {grid.best_params_['pca__n_components']}")
```

### 16.3 何时用白化

```python
# 白化适合：下游算法假设特征不相关且等方差
# 如 KMeans（欧氏距离在各方向等权）、ICA 预处理
pca = PCA(n_components=10, whiten=True).fit(X)
X_whiten = pca.transform(X)
print("白化后协方差:", np.cov(X_whiten, rowvar=False).round(3))
# 应接近单位阵
```

---

## 十七、常见错误与调试技巧

### 17.1 忘记标准化

```python
# 症状：主成分几乎只反映尺度最大的特征
pca = PCA(n_components=2).fit(X)
print(pca.components_)
# 若某行接近 [1, 0, 0, ...]，说明尺度大的特征主导

# 调试：检查各特征方差
print("各特征方差:", X.var(axis=0))
# 若方差差异 > 100x，必须先标准化
```

### 17.2 训练测试用不同 PCA

```python
# 错误：测试集重新 fit
pca_test = PCA(n_components=10).fit(X_test)  # ❌ 数据泄露
X_test_pca = pca_test.transform(X_test)

# 正确：用训练集的 PCA
pca = PCA(n_components=10).fit(X_train)
X_test_pca = pca.transform(X_test)  # ✅

# 调试：检查 transform 后形状和统计量
assert X_test_pca.shape[1] == 10
```

### 17.3 n_components 超过数据秩

```python
X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])  # 秩 1
pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_)  # [大, ~0]
# 第二主成分方差接近 0，无意义
# 调试：检查 explained_variance_ 是否有接近 0 的
```

### 17.4 SVD 符号不确定性

```python
# 不同实现/不同随机种子下，components_ 可能差正负号
pca1 = PCA(n_components=2).fit(X)
pca2 = PCA(n_components=2).fit(X)
# 可能 pca1.components_ != pca2.components_
# 但 np.abs(pca1.components_) == np.abs(pca2.components_)
# 调试：对比绝对值或对比 explained_variance_
```

### 17.5 调试检查清单

```python
def debug_pca(pca, X):
    """PCA 调试工具。"""
    print(f"原始维度: {X.shape[1]}, 降到: {pca.n_components_}")
    print(f"解释方差比: {pca.explained_variance_ratio_.round(3)}")
    print(f"累计: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"主成分正交性: {np.allclose(pca.components_ @ pca.components_.T, np.eye(pca.n_components_), atol=1e-6)}")
    X_rec = pca.inverse_transform(pca.transform(X))
    print(f"重构误差: {np.mean((X - X_rec)**2):.4f}")
    print(f"理论误差: {1 - pca.explained_variance_ratio_.sum():.4f}")
```

---

## 十八、与其他降维方法对比

### 18.1 PCA vs LDA

| 维度 | PCA | LDA |
|------|-----|-----|
| 监督性 | 无监督 | 有监督 |
| 目标 | 最大方差 | 最大类间方差/类内方差 |
| 降维上限 | $\min(n, d)$ | $K-1$（类别数-1） |
| 适用 | 可视化、去噪、压缩 | 分类预处理 |

### 18.2 PCA vs t-SNE vs UMAP

| 维度 | PCA | t-SNE | UMAP |
|------|-----|-------|------|
| 线性/非线性 | 线性 | 非线性 | 非线性 |
| 保留全局结构 | 是 | 否（局部） | 部分 |
| 适合可视化 | 一般 | 优 | 优 |
| 可 transform 新数据 | 是 | 否（需重新 fit） | 是 |
| 复杂度 | $O(nd^2)$ | $O(n^2)$ | $O(n \log n)$ |
| 可逆 | 是（inverse_transform） | 否 | 否 |

### 18.3 PCA vs 自编码器

```python
# 线性自编码器等价于 PCA
# 自编码器：min ||x - W2 @ W1 @ x||^2
# 最优 W1 的行空间 = PCA 主成分子空间

# 非线性自编码器（有激活函数）是 PCA 的非线性推广
# 能捕获 PCA 找不到的非线性结构（如瑞士卷）
```

### 18.4 PCA vs Random Projection

随机投影降维 $O(ndk)$，比 PCA 快，但不保证保留方差最大的方向。Johnson-Lindenstrauss 引理保证随机投影近似保留点间距离，适合对距离敏感的任务（如 KNN）且不需要最优重构。

---

## 十九、实际应用场景

### 19.1 人脸识别：特征脸

```python
import numpy as np
from minisklearn.decomposition import PCA

# 假设 faces 是 (n_faces, height*width) 的矩阵
faces = np.random.randn(100, 100*100)  # 100 张 100x100 人脸
pca = PCA(n_components=50).fit(faces)
# pca.components_ 的每行 reshape 成 (100, 100) 就是"特征脸"
# eigenface = pca.components_[i].reshape(100, 100)
print(f"50 个特征脸解释了 {pca.explained_variance_ratio_.sum():.1%} 方差")
```

### 19.2 金融：股票收益率降维

```python
# 100 只股票的日收益率，降到几个主成分（市场因子、行业因子）
returns = np.random.randn(252, 100) * 0.02  # 252 交易日，100 只股票
pca = PCA(n_components=10).fit(returns)
print("前 10 个主成分解释方差:", pca.explained_variance_ratio_.sum())
# 第 1 主成分通常是"市场因子"，解释 30-50% 方差
```

### 19.3 文本：潜在语义分析

```python
# TF-IDF 矩阵降维，发现潜在主题
# 用 TruncatedSVD（不中心化的 PCA）
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=100).fit(X_tfidf)
# X_lsa = svd.transform(X_tfidf)
# svd.components_ 的每行是一个"主题"（词的线性组合）
```

### 19.4 数据压缩

```python
# 高维数据压缩存储
X = np.random.randn(10000, 500)
pca = PCA(n_components=50).fit(X)
X_compressed = pca.transform(X)  # (10000, 50)，压缩 10x
# 存 X_compressed + pca.components_ + pca.mean_
# 重构：X ≈ X_compressed @ pca.components_ + pca.mean_
```

---

## 二十、思考题与练习

### 基础题

1. **手算**：对 $X = [[1, 2], [3, 4], [5, 6]]$，手动中心化并计算协方差矩阵。

2. **证明**：证明 PCA 主成分正交，即 $v_i^T v_j = 0$（$i \neq j$）。

3. **判断**：PCA 保留 95% 方差意味着重构误差是总方差的 5% 吗？证明你的结论。

### 进阶题

4. **实现**：用协方差矩阵特征分解（`np.linalg.eigh`）实现 PCA，对比 SVD 版本的数值精度。

5. **分析**：对瑞士卷数据，PCA 降到 2 维会损失什么结构？为什么 t-SNE 能更好地保留？

6. **实验**：生成各向同性数据（协方差 $\approx I$），PCA 降维后解释方差比应如何分布？验证你的预测。

7. **推导**：设 $X$ 服从 $\mathcal{N}(0, \Sigma)$，PCA 主成分是 $\Sigma$ 的特征向量。若 $\Sigma = \text{diag}(\lambda_1, \dots, \lambda_d)$（已对角），主成分是什么？

### 思考题

8. PCA 假设数据近似在低维线性子空间上。如果数据在低维**流形**上（非线性），PCA 会怎样？哪些算法能处理这种情况？

9. `explained_variance_ratio_` 大就一定对下游任务好吗？构造一个反例：PCA 选了方差大但无判别信息的方向。

10. 增量 PCA（IncrementalPCA）为什么不能简单地"分批 fit 后平均主成分"？正确的增量更新需要维护哪些统计量？（提示：运行均值、运行协方差）

---

## 二十一、扩展阅读

### 书籍

- **《The Elements of Statistical Learning》** 第 14.5 节：PCA 的理论
- **《Pattern Recognition and Machine Learning》** 第 12 章：PCA 的概率视角（PPCA）
- **《Numerical Linear Algebra》（Trefethen & Bau）**：SVD 的数值理论

### 论文

- **"A Tutorial on Principal Component Analysis" (Shlens, 2014)**：PCA 最清晰的教程
- **"Randomized Algorithms for Matrices and Data" (Mahoney)**：随机 SVD
- **"Robust Principal Component Analysis?" (Candès et al., 2011)**：鲁棒 PCA

### 在线资源

- sklearn PCA 文档：https://scikit-learn.org/stable/modules/decomposition.html#pca
- "PCA from Scratch" 系列：多种语言的 PCA 实现
- 3Blue1Brown 的线性代数视频：SVD 的几何直觉

### 相关算法

- `IncrementalPCA`：增量 PCA，大数据
- `SparsePCA`：稀疏 PCA，可解释性
- `KernelPCA`：核 PCA，非线性
- `TruncatedSVD`：截断 SVD，文本 LSA
- `FactorAnalysis`：因子分析
- `FastICA`：独立成分分析
- `NMF`：非负矩阵分解

---

[← 返回算法列表](../index.md)
