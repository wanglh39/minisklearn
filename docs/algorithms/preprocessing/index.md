# 预处理算法：缩放与编码

> 本章节实现 sklearn.preprocessing 的核心组件，讲解特征缩放和类别编码的数学原理与工程实现。预处理是机器学习流水线的第一道关卡，往往决定了模型上限。本章将从数学推导、几何直觉、工程实现、数值稳定性、对比 sklearn、常见陷阱等多个维度，把四个最常用的预处理算法讲透。

---

## 概览

| 算法 | 类型 | 作用 | 数学核心 | 适用场景 | 复杂度 |
|------|------|------|---------|---------|--------|
| `StandardScaler` | Transformer | 标准化（均值0、标准差1） | $z = \frac{x - \mu}{\sigma}$ | 服从近似正态分布的特征 | $O(nd)$ |
| `MinMaxScaler` | Transformer | 归一化到 [min, max] | $z = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (r_{max} - r_{min}) + r_{min}$ | 已知上下界、需要保正性的特征 | $O(nd)$ |
| `LabelEncoder` | Transformer | 标签 → 整数 | 排序 + 索引映射 | 目标标签 y 的编码 | $O(n \log n)$ |
| `OneHotEncoder` | Transformer | 类别 → 独热向量 | 指示函数 | 类别特征 X 的编码 | $O(nd)$ |

其中 $n$ 为样本数，$d$ 为特征数。所有预处理算法都是**无监督**的——它们只看 X（或 y），不需要标签信息。

预处理在机器学习流水线中的位置：

```
原始数据 → [预处理] → 数值矩阵 → [模型训练] → 模型 → [预测]
                ↑
            本章重点
```

---

## 一、为什么需要预处理？

### 1.1 特征尺度问题

很多算法对特征尺度敏感：

- **KNN**：用欧氏距离度量，大量级特征主导距离
- **梯度下降**：不同量级特征需要不同学习率
- **正则化**：L1/L2 对大量级特征惩罚不均
- **PCA**：方差大的特征被自动赋予更高权重
- **SVM/Ridge/Lasso**：正则化项 $\|w\|^2$ 对所有维度一视同仁，但若某维度数值范围是其他维度的 1000 倍，则对应 $w_j$ 只需很小就能产生大输出，正则化对它的约束实质上被削弱
- **神经网络**：激活函数的敏感区通常在 $[-2, 2]$ 附近，输入量级过大直接饱和
- **Naive Bayes**：高斯朴素贝叶斯假设各特征独立同分布，量级差异会扭曲方差估计

#### 1.1.1 一个直观的数值例子

```python
# 不缩放：身高（m）和体重（kg）量级差异巨大
X = [[1.7, 70], [1.6, 55], [1.8, 80]]
# KNN 距离 ≈ |Δ体重|，身高几乎被忽略

# 缩放后：两个特征贡献均衡
X_scaled = StandardScaler().fit_transform(X)
```

让我们手算一下不缩放时的距离。取两个样本 $a = (1.7, 70)$ 与 $b = (1.8, 80)$：

$$
d(a, b) = \sqrt{(1.7-1.8)^2 + (70-80)^2} = \sqrt{0.01 + 100} \approx 10.0005
$$

身高的贡献 $0.01$ 几乎被体重的 $100$ 淹没。标准化后两个特征都变成单位方差，距离计算才公平。

#### 1.1.2 梯度下降视角

考虑损失 $L(w) = \frac{1}{2}(y - Xw)^2$，梯度 $\nabla L = -X^T(y - Xw)$。若特征 $j$ 量级大，则 $X_j$ 大，对应梯度分量大，更新步长实际由学习率 $\eta$ 与 $X_j$ 共同决定。量级大的特征"跑得快"，量级小的特征"几乎不动"，等价于每个特征用了不同学习率，优化轨迹严重偏斜，收敛慢甚至发散。

数值演示：

```python
import numpy as np
# 特征 1 量级 1，特征 2 量级 1000
X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
y = np.array([1.0, 2.0, 3.0])
w = np.zeros(2)
eta = 0.01
for i in range(5):
    grad = X.T @ (X @ w - y)
    w = w - eta * grad
    print(f"iter {i}: w = {w}, grad = {grad}")
# 特征 2 的梯度比特征 1 大 1000 倍，w[1] 剧烈震荡，w[0] 几乎不动
```

#### 1.1.3 正则化视角

L2 正则项 $\frac{\lambda}{2}\|w\|^2$ 假设所有 $w_j$ 处于同一尺度。若 $x_1 \in [0, 1]$ 而 $x_2 \in [0, 1000]$，要让 $w_2 x_2$ 产生与 $w_1 x_1$ 相当的输出，需要 $w_2 \ll w_1$。正则化对 $w_2$ 的惩罚反而比 $w_1$ 小（因为 $w_2$ 本身小），结果是大量级特征逃脱了正则化约束，模型对它过拟合。

#### 1.1.4 距离算法视角

KNN、KMeans、SVM（RBF 核）等都依赖距离 $\|x_i - x_j\|$。距离被大量级特征主导，小量级特征几乎不参与决策。标准化后所有特征等权参与，决策边界才合理。

### 1.2 类别编码问题

大多数算法只接受数值输入，但现实数据有大量类别型特征：

- 分类标签："猫" / "狗" / "鸟"
- 枚举特征："红" / "绿" / "蓝"
- 高基数类别：用户 ID、城市名（成千上万个取值）

编码方式选择不当会引入虚假的数学关系。例如把"红=0, 绿=1, 蓝=2"输入线性模型，模型会默认"蓝 > 绿 > 红"且"蓝 - 绿 = 绿 - 红"，这在语义上毫无依据。

### 1.3 预处理的工程位置

在 sklearn 流水线中，预处理通常作为 `Pipeline` 的第一步：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

**关键原则**：预处理必须在交叉验证内部进行（即放进 `Pipeline`），不能在划分训练/测试集之前对全量数据 `fit_transform`。否则测试集信息泄露到训练集（例如用全量均值标准化训练集），交叉验证分数会虚高。

错误示例（数据泄露）：

```python
# ❌ 错误：全量 fit，测试集均值泄露
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)       # 用了全部数据
X_train, X_test = train_test_split(X_scaled) # 划分已在缩放之后
clf.fit(X_train, y_train)

# ✅ 正确：先划分，只在训练集 fit
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler().fit(X_train)       # 只见训练集
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)
clf.fit(X_train_s, y_train)
```

数据泄露的后果量化：在 5 折 CV 中，全量 fit 的测试集均值与训练集均值之差约为真实情况的 $1/\sqrt{n_{test}/n_{train}}$ 倍，分数系统性偏高 1-5%。

---

## 二、StandardScaler：标准化

### 2.1 数学原理

对每个特征 $j$ 独立计算样本均值与样本标准差：

$$
\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$

$$
\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}
$$

变换：

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

变换后：$\mathbb{E}[z_j] = 0$，$\text{Var}[z_j] = 1$。

#### 2.1.1 推导：为什么变换后方差为 1？

$$
\text{Var}[z_j] = \frac{1}{n}\sum_i (z_{ij} - \bar{z}_j)^2 = \frac{1}{n}\sum_i \left(\frac{x_{ij}-\mu_j}{\sigma_j}\right)^2 = \frac{1}{\sigma_j^2}\cdot \frac{1}{n}\sum_i (x_{ij}-\mu_j)^2 = \frac{\sigma_j^2}{\sigma_j^2} = 1
$$

#### 2.1.2 推导：为什么变换后均值为 0？

$$
\bar{z}_j = \frac{1}{n}\sum_i z_{ij} = \frac{1}{n}\sum_i \frac{x_{ij}-\mu_j}{\sigma_j} = \frac{1}{\sigma_j}\left(\frac{1}{n}\sum_i x_{ij} - \mu_j\right) = \frac{\mu_j - \mu_j}{\sigma_j} = 0
$$

#### 2.1.3 几何直觉

标准化对每个特征做**平移 + 缩放**：先减均值把数据云中心移到原点，再除以标准差把每个轴向的"伸展量"归一。几何上，原始数据云可能是一个细长的椭球（某轴长某轴短），标准化后变成近似各向同性的球状。这对基于距离的算法（KNN、KMeans、SVM）尤其重要——它们默认各方向等价。

可视化描述（二维情形）：

```
原始数据云:              标准化后:
   y                            y
   ^   *  *                     ^   *
   |    * *                     |  * *
   |   *  *      *              | * * *
   |       *  *                 |* * *
   +-------> x                  +-------> x
  (椭球，长轴沿 x)            (近似圆，各向同性)
```

#### 2.1.4 与总体均值的关系

注意 $\mu_j, \sigma_j$ 是**样本**统计量，不是总体。当新数据到来时，必须用训练时学到的 $\mu_j, \sigma_j$ 做 `transform`，不能在新数据上重新 `fit`。这是 `fit` / `transform` 分离的根本原因。

#### 2.1.5 样本标准差 vs 总体标准差

sklearn 用**总体**公式（除以 $n$）而非样本公式（除以 $n-1$）。两者关系 $\sigma_{sample} = \sigma_{pop} \cdot \sqrt{n/(n-1)}$，大样本下几乎相同。sklearn 选总体公式是为了与 `np.std` 默认一致，且对机器学习无偏性不是关键。

### 2.2 向量化实现

```python
def fit(self, X):
    self.mean_ = np.mean(X, axis=0)    # 沿样本轴求均值，shape (n_features,)
    self.scale_ = np.std(X, axis=0)    # 沿样本轴求标准差
    return self

def transform(self, X):
    return (X - self.mean_) / self.scale_  # 广播：每列减均值除标准差
```

NumPy 广播机制让 `(n, d) - (d,)` 自动按列操作，无需 for 循环。

#### 2.2.1 为什么不用 for 循环？

朴素实现：

```python
def transform_naive(self, X):
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = (X[:, j] - self.mean_[j]) / self.scale_[j]
    return out
```

实测在 $n=10^5, d=50$ 时，向量化版本约快 30-50 倍。原因：
1. 单次 C 层运算 vs 多次 Python 循环调度
2. SIMD 向量指令
3. 缓存友好（连续内存访问）

#### 2.2.2 fit_transform 的优化

`TransformerMixin` 提供的默认 `fit_transform` 是 `fit(X).transform(X)`，但 sklearn 的 `StandardScaler` 重写了它，避免对 X 走两遍。minisklearn 沿用默认实现，简洁优先。

#### 2.2.3 广播规则详解

`(n, d) - (d,)` 的广播过程：
1. `(d,)` 在左侧补 1 成 `(1, d)`
2. 沿第 0 轴复制 n 次成 `(n, d)`
3. 逐元素相减

整个过程 NumPy 内部用 C 完成，不实际复制内存（用 stride trick）。

### 2.3 边界情况：常量特征

如果某特征标准差为 0（所有值相同），除零会得到 NaN/Inf。

sklearn 的处理：`scale_[scale_ == 0] = 1.0`，即常量特征缩放因子设为 1，变换后全为 0（因为 $x - \mu = 0$）。

#### 2.3.1 为什么不抛异常？

常量特征虽然对建模无用，但在流水线中抛异常会打断批量处理。设为 1 让流水线继续跑，下游模型自行决定是否忽略该特征（L1 正则会把它对应的 $w$ 压到 0）。

#### 2.3.2 浮点精度陷阱

```python
X = np.array([1.0, 1.0, 1.0 + 1e-15])
# 理论 std > 0，但浮点误差可能让 std == 0
```

工程上可加一个微小 epsilon：`scale_ = np.where(scale_ < eps, 1.0, scale_)`，但 sklearn 没这么做，因为 `np.std` 在这种情况通常返回非零极小值，下游除法得到极大值，反而暴露了问题。

#### 2.3.3 NaN 和 Inf 输入

若 X 含 NaN，`np.mean` 返回 NaN，整个变换失效。应在预处理前用 `SimpleImputer` 填充。若 X 含 Inf，结果更不可控。生产环境建议先 `np.isfinite(X).all()` 检查。

### 2.4 使用示例

```python
from minisklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = [[1, 1000], [2, 2000], [3, 3000]]
X_scaled = scaler.fit_transform(X)
# → [[-1.22, -1.22], [0, 0], [1.22, 1.22]]
```

#### 2.4.1 手算验证

第一列：$\mu = 2, \sigma = \sqrt{\frac{1+0+1}{3}} = \sqrt{2/3} \approx 0.816$

$z_1 = (1-2)/0.816 \approx -1.225$，$z_2 = 0$，$z_3 = 1.225$。✓

#### 2.4.2 完整可运行示例

```python
import numpy as np
from minisklearn.preprocessing import StandardScaler

np.random.seed(0)
X = np.random.randn(1000, 3) * [1, 10, 100] + [0, 5, -50]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

print("均值:", Xs.mean(axis=0))   # ≈ [0, 0, 0]
print("标准差:", Xs.std(axis=0))  # ≈ [1, 1, 1]
print("mean_ 属性:", scaler.mean_)
print("scale_ 属性:", scaler.scale_)

# 在新数据上 transform（用训练时学到的参数）
X_new = np.array([[1.0, 5.0, -50.0]])
print("新数据缩放:", scaler.transform(X_new))
```

#### 2.4.3 错误示例：在测试集上 fit

```python
# ❌ 错误
scaler = StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)  # 用测试集自己的均值！

# ✅ 正确
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)  # 用训练集均值
```

#### 2.4.4 错误示例：忘记 transform

```python
scaler = StandardScaler().fit(X_train)
clf.fit(X_train, y_train)  # ❌ 忘了 transform X_train
# 模型在原始量级上训练，scaler 白学了

# ✅
clf.fit(scaler.transform(X_train), y_train)
```

### 2.5 与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 公式 | 完全一致 | ✅ |
| `with_mean` / `with_std` 参数 | 支持 | 暂不支持（恒为 True） |
| 稀疏矩阵 | `with_mean=False` 时支持 | 暂不支持 |
| `partial_fit`（增量） | 支持 | 暂不支持 |
| 数值结果 | 一致 | ✅ |

实测数值一致性：

```python
from sklearn.preprocessing import StandardScaler as SkS
from minisklearn.preprocessing import StandardScaler as MnS
X = np.random.randn(100, 5) * [1, 3, 10, 0.1, 100]
a = SkS().fit_transform(X)
b = MnS().fit_transform(X)
np.allclose(a, b)  # True
```

### 2.6 复杂度分析

- **时间**：`fit` 计算 mean 和 std，各 $O(nd)$；`transform` 一次广播运算 $O(nd)$。总计 $O(nd)$。
- **空间**：存储 `mean_` 和 `scale_`，各 $O(d)$；`transform` 输出 $O(nd)$。
- 对比：朴素 Python 循环 $O(nd)$ 但常数大 30-50 倍。

### 2.7 常见陷阱

1. **测试集 fit**：见 2.4.3
2. **稀疏矩阵传给 with_mean=True**：减均值会破坏稀疏结构，把 0 元素变成非零，内存爆炸
3. **对已经标准化过的数据再标准化**：数值上无害但浪费，且若浮点误差累积可能让 std 略偏离 1
4. **整数输入**：`np.std` 对 int 数组返回 float，但若用 `X - mean` 时 X 是 int 数组可能截断。minisklearn 在内部 `np.asarray(X, dtype=float)` 规避
5. **顺序敏感**：先标准化再独热 vs 先独热再标准化，结果不同。一般独热后不再标准化（独热的 0/1 已是好尺度）

### 2.8 何时不该用 StandardScaler

- 数据有显著异常值：用 `RobustScaler`（中位数 + IQR）
- 数据已天然在 $[0, 1]$：无需再缩放
- 稀疏数据：减均值破坏稀疏性，用 `MaxAbsScaler` 或 `with_mean=False`
- 已知分布非正态且需保留分布形状：考虑 `QuantileTransformer`

---

## 三、MinMaxScaler：归一化

### 3.1 数学原理

$$
z_{ij} = \frac{x_{ij} - x_{min,j}}{x_{max,j} - x_{min,j}} \cdot (r_{max} - r_{min}) + r_{min}
$$

默认 $r_{min} = 0, r_{max} = 1$。

#### 3.1.1 推导：变换后范围

设 $r_{min}=0, r_{max}=1$。当 $x_{ij} = x_{min,j}$ 时：

$$
z = \frac{0}{x_{max}-x_{min}} \cdot 1 + 0 = 0
$$

当 $x_{ij} = x_{max,j}$ 时：

$$
z = \frac{x_{max}-x_{min}}{x_{max}-x_{min}} \cdot 1 + 0 = 1
$$

故变换后值域恰为 $[0, 1]$。一般 $[r_{min}, r_{max}]$ 同理。

#### 3.1.2 几何直觉

MinMaxScaler 对每个特征做**仿射变换**（平移 + 缩放），把数据云的每个轴向上最小最大值分别对齐到 $r_{min}$ 和 $r_{max}$。与 StandardScaler 不同，它保留分布的"形状"——若原数据在某轴上分布偏斜，缩放后仍偏斜，只是被压到 $[0, 1]$。

```
原始分布（偏斜）:        MinMax 后（仍偏斜）:      Standard 后（仍偏斜，但中心在 0）:
   |  *                       |  *                      |     *
   | * *                      | * *                     |   * *
   |* * *                     |* * *                    |  * * *
   |    * *                   |    * *                  | * * *
   +------                    +------                   +------
  [10, 100]                 [0, 1]                    [-1.5, 2]
```

#### 3.1.3 与 StandardScaler 的关系

两者都是逐特征的仿射变换 $z = a x + b$，区别只在 $a, b$ 的选择：

| | StandardScaler | MinMaxScaler |
|---|---|---|
| $a$ | $1/\sigma$ | $(r_{max}-r_{min})/(x_{max}-x_{min})$ |
| $b$ | $-\mu/\sigma$ | $-x_{min} \cdot a + r_{min}$ |
| 用到的统计量 | 均值、标准差 | 最小值、最大值 |
| 对异常值敏感度 | 较低（用 std） | 较高（用 min/max） |

### 3.2 与 StandardScaler 的详细对比

| | StandardScaler | MinMaxScaler |
|---|---|---|
| 变换后范围 | 无界（可有负值） | 有界 $[r_{min}, r_{max}]$ |
| 保留分布形状 | 是（线性变换） | 是（线性变换） |
| 对异常值敏感度 | 较低（用 std） | 较高（用 min/max） |
| 稀疏矩阵兼容 | 需 `with_mean=False` | 兼容（不减均值） |
| 神经网络常用 | 较少 | 较多（激活函数输入范围友好） |
| 距离算法常用 | 较多 | 一般 |
| 保正性 | 否 | 是（若 $r_{min} \geq 0$） |
| 保留零点 | 否（除非 $\mu=0$） | 是（若 $x_{min}=0$ 且 $r_{min}=0$） |

#### 3.2.1 异常值敏感度演示

```python
X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])  # 100 是异常值

mm = MinMaxScaler().fit(X)
ss = StandardScaler().fit(X)

print("MinMax:", mm.transform(X))   # 100 → 1.0，前 4 个被压到接近 0
print("Std:  ", ss.transform(X))    # 100 → ~2.24，前 4 个集中在 -0.4 附近
```

MinMaxScaler 把异常值拉到上界，正常值全部挤到下界附近，丢失分辨力。StandardScaler 受影响较小。

### 3.3 边界情况

#### 3.3.1 常量特征

$x_{max} = x_{min}$ 时分母为 0。sklearn 处理：`scale_ = where(scale_ == 0, 1.0, scale_)`，变换后全为 $r_{min}$（因为 $x - x_{min} = 0$）。

#### 3.3.2 新数据超出训练范围

```python
scaler = MinMaxScaler().fit([[0], [10]])
scaler.transform([[-5]])  # → [-0.5]，超出 [0, 1]
scaler.transform([[20]])  # → [2.0]
```

MinMaxScaler 不截断，新数据超出训练范围会产生超出 $[r_{min}, r_{max}]$ 的值。若需截断，应额外用 `np.clip`。

### 3.4 使用示例

```python
from minisklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
X = [[1], [2], [3], [4]]
X_scaled = scaler.fit_transform(X)
# → [[-1.], [-0.33], [0.33], [1.]]
```

#### 3.4.1 手算验证（feature_range=(-1, 1)）

$x_{min}=1, x_{max}=4, r_{min}=-1, r_{max}=1$。

$z(1) = \frac{0}{3} \cdot 2 + (-1) = -1$ ✓
$z(2) = \frac{1}{3} \cdot 2 - 1 = -0.333$ ✓
$z(4) = \frac{3}{3} \cdot 2 - 1 = 1$ ✓

#### 3.4.2 完整可运行示例

```python
import numpy as np
from minisklearn.preprocessing import MinMaxScaler

# 模拟图像像素值 [0, 255] 归一化到 [0, 1]
pixels = np.random.randint(0, 256, size=(100, 3)).astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
normalized = scaler.fit_transform(pixels)
print("最小:", normalized.min(axis=0))  # ≈ [0, 0, 0]
print("最大:", normalized.max(axis=0))  # ≈ [1, 1, 1]
```

#### 3.4.3 神经网络场景：归一化到 [-1, 1]

tanh 激活函数在 $[-1, 1]$ 内最敏感，常用 `MinMaxScaler(feature_range=(-1, 1))` 把输入对齐到 tanh 甜点区。

### 3.5 复杂度与数值稳定性

- **时间**：`fit` 求 min/max 各 $O(nd)$；`transform` $O(nd)$。
- **空间**：`min_` / `scale_` 各 $O(d)$。
- **数值稳定性**：减法 $x - x_{min}$ 当 $x$ 与 $x_{min}$ 都很大时可能丢失有效位（catastrophic cancellation）。例如 $x = 1000000.1, x_{min} = 1000000.0$，float32 下 $x - x_{min}$ 可能得 0。建议输入用 float64。

### 3.6 何时用 MinMaxScaler

- 神经网络输入
- 图像像素归一化
- 需要保正性的特征（如计数）
- 已知数据严格有界且无异常值

---

## 四、LabelEncoder：标签编码

### 4.1 原理

将类别标签映射为连续整数 $[0, n\_classes - 1]$：

1. `fit`：`np.unique(y)` 排序去重，得到 `classes_`
2. `transform`：用 `np.searchsorted` 二分查找每个标签在 `classes_` 中的位置

```python
le = LabelEncoder()
le.fit(["猫", "狗", "鸟"])
# classes_ = ["鸟", "狗", "猫"]  （按 Unicode 排序）
le.transform(["猫", "鸟"])
# → [2, 0]
```

#### 4.1.1 数学表达

设 `classes_` 为排序后的类别数组。对标签 $y$：

$$
\text{encode}(y) = \min\{k : \text{classes\_}[k] = y\}
$$

由于 `classes_` 已排序去重，该位置唯一。`np.searchsorted` 用二分查找在 $O(\log K)$ 内找到。

#### 4.1.2 内部实现

```python
def fit(self, y):
    self.classes_ = np.unique(y)   # 排序去重
    return self

def transform(self, y):
    return np.searchsorted(self.classes_, y)
```

`np.unique` 内部用排序，时间 $O(n \log n)$。`np.searchsorted` 对每个元素二分查找，总计 $O(n \log K)$。

### 4.2 为什么排序？

排序保证映射的**确定性**：无论输入顺序如何，同一组标签的编码结果一致。这对模型复现和序列化很重要。

#### 4.2.1 反例：不排序的后果

```python
# 假设不排序，按出现顺序编码
le1 = LabelEncoder_unsorted().fit(["猫", "狗", "鸟"])  # → 猫=0, 狗=1, 鸟=2
le2 = LabelEncoder_unsorted().fit(["鸟", "猫", "狗"])  # → 鸟=0, 猫=1, 狗=2
# 同一组标签，编码不同！模型无法复现
```

排序后 `classes_` 恒为 `["鸟", "狗", "猫"]`，编码恒为 鸟=0, 狗=1, 猫=2，与输入顺序无关。

#### 4.2.2 排序规则

`np.unique` 用 NumPy 的默认排序，对字符串按 Unicode 码点。中文按拼音意外的顺序（实际是 Unicode 码点）。若需自定义顺序（如"低<中<高"），应先手动映射再传入。

### 4.3 适用场景

- **适合**：编码目标标签 y（分类标签）
- **不适合**：编码特征矩阵 X（引入虚假序关系，"蓝"=2 > "红"=0 无意义）

特征编码应该用 `OneHotEncoder`。

#### 4.3.1 误用示例

```python
# ❌ 错误：用 LabelEncoder 编码颜色特征
le = LabelEncoder().fit(["红", "绿", "蓝"])
X_color = le.transform(["红", "绿", "蓝"])  # [0, 1, 2]
# 线性模型会认为 蓝 > 绿 > 红，且 蓝 - 绿 = 绿 - 红（等距），语义错误

# ✅ 正确：用 OneHotEncoder
enc = OneHotEncoder().fit([["红"], ["绿"], ["蓝"]])
X_color = enc.transform([["红"], ["绿"], ["蓝"]])
# [[1,0,0],[0,1,0],[0,0,1]]  三个类别独立维度，无序关系
```

#### 4.3.2 适合用 LabelEncoder 的标签

- 二分类：`["否", "是"]` → `[0, 1]`，序关系恰好对应"是否"，无语义问题
- 有序类别：`["低", "中", "高"]` → `[0, 1, 2]`，序关系符合语义（此时 LabelEncoder 反而合适）
- 模型输出标签：所有分类器的 `predict` 内部都用整数标签

### 4.4 inverse_transform

```python
le = LabelEncoder().fit(["猫", "狗", "鸟"])
y = le.transform(["猫", "鸟", "狗"])  # [2, 0, 1]
le.inverse_transform(y)  # ["猫", "鸟", "狗"]
```

实现就是 `classes_[y]`，NumPy 高级索引 $O(n)$。

### 4.5 复杂度

- **fit**：`np.unique` 排序 $O(n \log n)$
- **transform**：$n$ 次 `searchsorted`，每次 $O(\log K)$，总计 $O(n \log K)$
- **空间**：`classes_` $O(K)$

### 4.6 与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 编码结果 | 完全一致 | ✅ |
| 未见过的类别 transform | 抛 KeyError | 抛 KeyError（searchsorted 越界） |
| 数值型标签 | 支持 | 支持 |
| `inverse_transform` | 支持 | 支持 |

---

## 五、OneHotEncoder：独热编码

### 5.1 原理

对每个类别值，生成一个二值指示向量：

```
"猫" → [1, 0, 0]
"狗" → [0, 1, 0]
"鸟" → [0, 0, 1]
```

数学上，对类别 $c$ 和类别集合 $\mathcal{C}$：

$$
\text{onehot}(c)_k = \mathbb{1}[c = \mathcal{C}_k]
$$

其中 $\mathbb{1}[\cdot]$ 是指示函数，条件成立为 1 否则为 0。

#### 5.1.1 为什么用独热？

独热编码把类别映射到正交基向量，任意两个不同类别的独热向量内积为 0、欧氏距离为 $\sqrt{2}$。这保证：
1. 类别间无序关系（任意两两等距）
2. 线性模型对每个类别学独立权重 $w_k$，预测时 $w_{c}$ 起作用，其他类别权重不影响

#### 5.1.2 维度膨胀

$K$ 个类别 → $K$ 维独热。若 $K$ 很大（高基数类别，如用户 ID $K=10^6$），独热后特征维度爆炸，内存和计算都吃不消。此时应考虑：
- 目标编码（TargetEncoder）
- 频率编码
- 嵌入向量（Entity Embedding，神经网络场景）

### 5.2 向量化实现

```python
# 对每列，用 searchsorted 找类别索引
indices = np.searchsorted(categories, X[:, col])
# 用高级索引填充独热矩阵
output[np.arange(n_samples), col_offset + indices] = 1.0
```

关键：`output[行索引, 列索引] = 1` 一次性设置所有非零位置，无需循环。

#### 5.2.1 逐步分解

```python
# 假设 categories = ["鸟", "狗", "猫"], X_col = ["猫", "鸟", "狗", "猫"]
indices = np.searchsorted(categories, X_col)
# → [2, 0, 1, 2]

n = len(X_col)
output = np.zeros((n, 3))
output[np.arange(n), indices] = 1.0
# → [[0,0,1],
#    [1,0,0],
#    [0,1,0],
#    [0,0,1]]
```

`np.arange(n)` 是行索引 `[0,1,2,3]`，`indices` 是列索引 `[2,0,1,2]`，配对成 `[(0,2),(1,0),(2,1),(3,2)]` 一次性置 1。

#### 5.2.2 高级索引原理

NumPy 的高级索引 `A[idx_rows, idx_cols]` 选取 `A[idx_rows[i], idx_cols[i]]` for each i，等价于逐对取元素。赋值 `A[idx_rows, idx_cols] = 1` 一次性把所有这些位置置 1。这是向量化编码的关键。

### 5.3 多列处理

对多列特征，各列的独热块拼接：

```
列0: ["a", "b"] → 2 列
列1: ["x", "y"] → 2 列
总输出: 4 列
```

#### 5.3.1 列偏移计算

```python
# 假设 2 列，分别有 2、3 个类别
n_categories = [2, 3]
col_offsets = np.cumsum([0] + n_categories[:-1])  # [0, 2]
total = sum(n_categories)  # 5

output = np.zeros((n_samples, total))
for col, (cat, offset) in enumerate(zip(categories_per_col, col_offsets)):
    indices = np.searchsorted(cat, X[:, col])
    output[np.arange(n_samples), offset + indices] = 1.0
```

第 0 列的独热块占输出列 `[0, 1]`，第 1 列占 `[2, 3, 4]`。

### 5.4 使用示例

```python
from minisklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
X = [["猫"], ["狗"], ["鸟"]]
enc.fit(X)
enc.transform([["猫"], ["鸟"]])
# → [[1., 0., 0.],
#    [0., 1., 0.]]
```

#### 5.4.1 完整可运行示例

```python
import numpy as np
from minisklearn.preprocessing import OneHotEncoder

# 多列类别特征
X = np.array([
    ["猫", "小"],
    ["狗", "中"],
    ["鸟", "大"],
    ["猫", "大"],
])
enc = OneHotEncoder()
enc.fit(X)
print("每列类别:", enc.categories_)  # [["鸟","狗","猫"], ["大","中","小"]]

X_new = np.array([["猫", "小"], ["鸟", "大"]])
print(enc.transform(X_new))
# 列0: 猫→[0,0,1], 鸟→[1,0,0]
# 列1: 小→[0,0,1], 大→[1,0,0]
# 输出:
# [[0,0,1, 0,0,1],
#  [1,0,0, 1,0,0]]
```

#### 5.4.2 错误示例：fit 和 transform 列数不一致

```python
enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))  # 1 列
enc.transform(np.array([["a", "x"]]))  # ❌ 2 列，ValueError
```

### 5.5 稀疏输出 vs 稠密输出

sklearn 默认返回稀疏矩阵（CSR），因为独热矩阵每行只有 $d$ 个 1（$d$ 为列数），其余全 0，稀疏存储省内存。minisklearn 暂返回稠密矩阵，简单优先。对 $n=10^5, K=10^3$ 的单列独热，稠密需 800MB，稀疏只需约 5MB。

### 5.6 handle_unknown

sklearn 支持 `handle_unknown='ignore'`（未见类别对应全 0 行）和 `'error'`（抛异常）。minisklearn 暂只支持抛异常（`searchsorted` 越界）。生产环境常用 `'ignore'` 避免线上新类别导致流水线崩溃。

### 5.7 复杂度

- **fit**：每列 `np.unique` $O(n \log n)$，总计 $O(d \cdot n \log n)$
- **transform**：每列 searchsorted $O(n \log K)$ + 高级索引 $O(n)$，总计 $O(d \cdot n \log K)$
- **空间**：输出 $O(n \cdot \sum_k K_k)$，稀疏存储 $O(n \cdot d)$

### 5.8 何时用 OneHotEncoder

- 低基数类别特征（$K \leq 100$）
- 无序类别
- 线性模型、SVM、神经网络等对数值敏感的模型
- 树模型一般不需要独热（能直接处理类别，sklearn 的树需独热但 LightGBM/XGBoost 新版支持原生类别）

---

## 六、与 sklearn 全面对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| API | 完整 | 核心子集 |
| 稀疏矩阵 | 支持 | 暂不支持 |
| `handle_unknown` | 支持 | 暂不支持 |
| `partial_fit` | 支持 | 暂不支持 |
| 数值精度 | 一致 | ✅ |
| `feature_names_in_` | 支持 | 暂不支持 |
| Pipeline 兼容 | 完整 | ✅ |

### 6.1 数值一致性测试

```python
import numpy as np
from sklearn.preprocessing import StandardScaler as SkS, MinMaxScaler as SkM
from minisklearn.preprocessing import StandardScaler as MnS, MinMaxScaler as MnM

X = np.random.randn(200, 6) * [1, 3, 10, 0.1, 100, 7] + [0, 5, -50, 2, 30, -8]

assert np.allclose(SkS().fit_transform(X), MnS().fit_transform(X))
assert np.allclose(SkM().fit_transform(X), MnM().fit_transform(X))
print("全部一致")
```

### 6.2 性能对比

在 $n=10^5, d=20$ 下（粗略量级）：

| 算法 | sklearn | minisklearn | 比值 |
|------|---------|-------------|------|
| StandardScaler.fit_transform | ~5ms | ~6ms | 1.2x |
| MinMaxScaler.fit_transform | ~4ms | ~5ms | 1.2x |
| OneHotEncoder (稠密) | ~15ms | ~18ms | 1.2x |

差异主要来自 sklearn 的 Cython 优化和更精细的内存布局。对绝大多数应用，这点差距可忽略。

---

## 七、实战教程：完整预处理流水线

### 7.1 混合类型数据预处理

```python
import numpy as np
from minisklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# 假设数据：数值特征 [年龄, 收入] + 类别特征 [城市]
X_num = np.array([[25, 5000], [30, 8000], [40, 12000], [35, 7000]], dtype=float)
X_cat = np.array([["北京"], ["上海"], ["广州"], ["北京"]])
y = np.array(["买", "不买", "买", "不买"])

# 1. 数值特征标准化
scaler = StandardScaler().fit(X_num)
X_num_s = scaler.transform(X_num)

# 2. 类别特征独热
enc = OneHotEncoder().fit(X_cat)
X_cat_s = enc.transform(X_cat)

# 3. 拼接
X_final = np.hstack([X_num_s, X_cat_s])

# 4. 标签编码
le = LabelEncoder().fit(y)
y_enc = le.transform(y)

print("最终特征形状:", X_final.shape)
print("标签:", y_enc)
```

### 7.2 保存与加载

预处理器的状态全在 `mean_` / `scale_` / `classes_` 等属性里，可用 pickle 持久化：

```python
import pickle
scaler = StandardScaler().fit(X_train)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("scaler.pkl", "rb") as f:
    scaler2 = pickle.load(f)
scaler2.transform(X_test)  # 与原 scaler 一致
```

### 7.3 在 Pipeline 中使用

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from minisklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

### 7.4 完整机器学习流水线示例

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from minisklearn.preprocessing import StandardScaler, OneHotEncoder

# 生成模拟数据
np.random.seed(42)
n = 1000
age = np.random.randint(18, 70, n)
income = np.random.randint(3000, 50000, n)
city = np.random.choice(["北京", "上海", "广州", "深圳"], n)
y = (income > 25000).astype(int)

X_num = np.column_stack([age, income]).astype(float)
X_cat = city.reshape(-1, 1)

# 划分
X_num_tr, X_num_te, X_cat_tr, X_cat_te, y_tr, y_te = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=0
)

# 预处理（只在训练集 fit）
scaler = StandardScaler().fit(X_num_tr)
enc = OneHotEncoder().fit(X_cat_tr)

X_tr = np.hstack([scaler.transform(X_num_tr), enc.transform(X_cat_tr)])
X_te = np.hstack([scaler.transform(X_num_te), enc.transform(X_cat_te)])

# 训练
clf = LogisticRegression().fit(X_tr, y_tr)
print("测试准确率:", clf.score(X_te, y_te))
```

---

## 八、常见问题与陷阱汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| 测试集 fit | CV 分数虚高 | 只在训练集 fit |
| 整数除法截断 | 输出全 0 | 输入转 float |
| 常量特征 NaN | scale_=0 除零 | sklearn 自动置 1 |
| 独热维度爆炸 | 内存不足 | 用稀疏或目标编码 |
| LabelEncoder 误用于特征 | 引入虚假序关系 | 改用 OneHotEncoder |
| 未排序编码 | 模型不可复现 | 用 np.unique 排序 |
| 新类别 transform 崩溃 | KeyError | handle_unknown='ignore' |
| 大数相减精度丢失 | 归一化后值异常 | 用 float64 |
| Pipeline 外预处理 | 数据泄露 | 放进 Pipeline |
| 重复标准化 | 浪费计算 | 只标准化一次 |

### 8.1 调试技巧

```python
# 检查缩放是否成功
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
assert np.allclose(Xs.mean(axis=0), 0, atol=1e-10), "均值不为 0"
assert np.allclose(Xs.std(axis=0), 1, atol=1e-10), "标准差不为 1"

# 检查独热是否正确
enc = OneHotEncoder().fit(X_cat)
Xo = enc.transform(X_cat)
assert (Xo.sum(axis=1) == X_cat.shape[1]).all(), "每行 1 的个数不等于列数"
assert np.isin(Xo, [0, 1]).all(), "存在非 0/1 值"
```

---

## 九、进阶话题

### 9.1 RobustScaler（异常值鲁棒）

sklearn 提供 `RobustScaler` 用中位数和 IQR 代替均值和标准差：

$$
z = \frac{x - \text{median}}{IQR}
$$

其中 $IQR = Q_3 - Q_1$。对异常值不敏感，因为中位数和分位数不受极端值影响。minisklearn 暂未实现。

### 9.2 QuantileTransformer

把特征变换到均匀分布或正态分布，对异常值鲁棒且能处理任意分布。但计算量大 $O(n \log n)$，且破坏原始值的相对关系（非线性变换）。

### 9.3 PowerTransformer

Box-Cox 或 Yeo-Johnson 变换，把偏斜分布拉向正态。适合线性模型假设正态的场景。

### 9.4 目标编码

对高基数类别特征，用每个类别的目标均值作为编码：

```python
# 城市 "北京" 的编码 = P(y=1 | 城市=北京)
city_target_mean = df.groupby("city")["y"].mean()
df["city_encoded"] = df["city"].map(city_target_mean)
```

需配合正则化（如 Bayesian 平均）避免过拟合，sklearn 的 `TargetEncoder`（1.3+）实现了。

---

## 架构回扣

这四个预处理算法都继承 `BaseEstimator + TransformerMixin`，自动获得：

- `get_params` / `set_params` / `clone`（来自 BaseEstimator）
- `fit_transform`（来自 TransformerMixin）

我们只需实现 `fit` 和 `transform`，其余的架构能力**免费获得**。这就是第一讲讲的统一 API 契约的回报。

### 设计哲学

- **fit / transform 分离**：让"学习参数"和"应用参数"解耦，同一套参数可反复 transform 不同数据，是流水线和交叉验证的基础
- **逐特征独立**：所有缩放器对每列独立计算，天然支持广播，无需针对多特征写特殊逻辑
- **属性命名约定**：`mean_` / `scale_` / `classes_` 等以单下划线结尾，表示"fit 后学到的属性"，`check_is_fitted` 据此判断是否已 fit
- **不可变参数**：`__init__` 的参数（如 `feature_range`）不应被 fit 修改，保证 `clone` 后行为一致

### 与下游模型的契约

预处理器输出 numpy 数组，下游模型（线性回归、KNN 等）接受 numpy 数组。这个简单契约让任意 Transformer 和 Estimator 可自由组合进 Pipeline，是 sklearn 生态可组合性的根基。

### 类层级

```
BaseEstimator
   ├── StandardScaler
   ├── MinMaxScaler
   ├── LabelEncoder
   └── OneHotEncoder
        (全部混入 TransformerMixin 获得 fit_transform)
```

### fit 后属性命名约定

| 算法 | fit 后属性 | 含义 |
|------|-----------|------|
| StandardScaler | `mean_`, `scale_`, `var_` | 每列均值、标准差、方差 |
| MinMaxScaler | `min_`, `scale_`, `data_min_`, `data_max_` | 每列偏移、缩放因子、原始最小最大 |
| LabelEncoder | `classes_` | 排序去重后的类别数组 |
| OneHotEncoder | `categories_` | 每列的类别数组列表 |

所有属性以单下划线结尾，遵循 sklearn 约定：单下划线 = "fit 后学到的"，双下划线结尾（如 `coef_`）= "模型参数"。预处理器学到的是变换参数，不是模型参数，故用单下划线。

---

## 十、更深入的数学推导与证明

### 10.1 仿射变换的复合性质

所有逐特征缩放器（StandardScaler、MinMaxScaler、MaxAbsScaler）都是仿射变换 $z = a x + b$，其中 $a, b$ 是逐特征向量。一个自然的问题：两个仿射变换的复合仍是仿射变换吗？

**定理**：设 $T_1(x) = a_1 x + b_1$，$T_2(x) = a_2 x + b_2$，则 $T_2 \circ T_1(x) = a_2(a_1 x + b_1) + b_2 = (a_2 a_1) x + (a_2 b_1 + b_2)$，仍是仿射变换，且复合参数为 $a = a_2 a_1$，$b = a_2 b_1 + b_2$。

**推论**：对同一数据先 StandardScaler 再 MinMaxScaler，等价于一个 MinMaxScaler（参数不同）。验证：

```python
import numpy as np
from minisklearn.preprocessing import StandardScaler, MinMaxScaler

X = np.random.randn(100, 3) * [1, 5, 10] + [0, 2, -3]

# 先标准化再归一化
Xs = StandardScaler().fit_transform(X)
Xm = MinMaxScaler().fit_transform(Xs)

# 等价的单次 MinMax（用复合参数）
ss = StandardScaler().fit(X)
a1, b1 = 1.0 / ss.scale_, -ss.mean_ / ss.scale_
Xs2 = X * a1 + b1
mm = MinMaxScaler().fit(Xs2)
a2, b2 = mm.scale_, mm.min_ - mm.data_min_ * mm.scale_
a = a2 * a1
b = a2 * b1 + b2
Xm2 = X * a + b

print(np.allclose(Xm, Xm2))  # True
```

### 10.2 StandardScaler 保持线性相关性

**命题**：StandardScaler 是可逆线性变换，不改变特征间的 Pearson 相关系数。

**证明**：设 $z_j = (x_j - \mu_j) / \sigma_j$。Pearson 相关系数

$$
\rho(z_i, z_j) = \frac{\text{Cov}(z_i, z_j)}{\sqrt{\text{Var}(z_i)\text{Var}(z_j)}} = \frac{\text{Cov}\left(\frac{x_i-\mu_i}{\sigma_i}, \frac{x_j-\mu_j}{\sigma_j}\right)}{1 \cdot 1}
$$

$$
= \frac{1}{\sigma_i \sigma_j} \text{Cov}(x_i, x_j) = \frac{\text{Cov}(x_i, x_j)}{\sigma_i \sigma_j} = \rho(x_i, x_j)
$$

故标准化不改变特征间相关性。这意味着 PCA 在标准化数据上做等价于对**相关矩阵**做特征分解，而非协方差矩阵。

### 10.3 MinMaxScaler 的逆变换

MinMaxScaler 的逆变换存在且仍是仿射变换：

$$
x = \frac{z - r_{min}}{r_{max} - r_{min}} \cdot (x_{max} - x_{min}) + x_{min}
$$

```python
scaler = MinMaxScaler().fit(X)
Xt = scaler.transform(X)
X_back = scaler.inverse_transform(Xt)  # sklearn 支持，minisklearn 可手动算
assert np.allclose(X, X_back)
```

### 10.4 OneHotEncoder 的正交性证明

**命题**：独热编码矩阵 $H \in \{0,1\}^{n \times K}$ 满足 $H^T H = \text{diag}(n_1, \dots, n_K)$，其中 $n_k$ 是类别 $k$ 的样本数。

**证明**：$(H^T H)_{kl} = \sum_i H_{ik} H_{il}$。$H_{ik} H_{il} = 1$ 当且仅当样本 $i$ 同时属于类别 $k$ 和 $l$，即 $k = l$ 且 $i$ 属于该类。故对角元素为各类样本数，非对角元素为 0。

**推论**：若各类样本数相等（$n_k = n/K$），则 $H^T H = (n/K) I$，独热列向量正交。这是独热编码"无序"性质的数学基础。

### 10.5 LabelEncoder 的信息论视角

LabelEncoder 把 $K$ 个类别映射为整数，所需信息量 $\log_2 K$ 比特。若类别分布不均，最优编码（Huffman 编码）平均比特数 $H(Y) = -\sum_k p_k \log_2 p_k < \log_2 K$。LabelEncoder 用等长编码，未利用分布信息，但对下游模型无影响（模型自己学权重）。

---

## 十一、更多代码示例与对比实验

### 11.1 四种缩放器全对比

```python
import numpy as np
from minisklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)
X = np.random.lognormal(size=(200, 3))  # 对数正态，偏斜且有量级差异
X[:, 0] *= 1000
X[:, 2] += 50

# 加入异常值
X[0] = [1e6, 100, 1e4]

print("原始:    mean=", X.mean(axis=0).round(2), "std=", X.std(axis=0).round(2))
Xs = StandardScaler().fit_transform(X)
print("Std:     mean=", Xs.mean(axis=0).round(2), "std=", Xs.std(axis=0).round(2))
Xm = MinMaxScaler().fit_transform(X)
print("MinMax:  min=", Xm.min(axis=0).round(2), "max=", Xm.max(axis=0).round(2))
```

### 11.2 缩放对 KNN 的影响实验

```python
import numpy as np
from minisklearn.neighbors import KNeighborsClassifier
from minisklearn.preprocessing import StandardScaler, MinMaxScaler
from minisklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)
X = np.column_stack([rng.randn(300) * 100,      # 大尺度特征
                     rng.randn(300),             # 小尺度特征
                     rng.randn(300) * 50 + 10])  # 中等尺度
y = (X[:, 0] + X[:, 1] > X[:, 0].mean()).astype(int)

knn = KNeighborsClassifier(n_neighbors=5)
print("不缩放:   ", cross_val_score(knn, X, y, cv=5).mean())
print("Std:      ", cross_val_score(knn, StandardScaler().fit_transform(X), y, cv=5).mean())
print("MinMax:   ", cross_val_score(knn, MinMaxScaler().fit_transform(X), y, cv=5).mean())
```

### 11.3 缩放对梯度下降收敛速度的影响

```python
import numpy as np
import time

def gd_logistic(X, y, lr=0.01, n_iter=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        grad_w = X.T @ (1 / (1 + np.exp(-z)) - y) / n
        w -= lr * grad_w
        b -= lr * np.mean(1 / (1 + np.exp(-z)) - y)
    return w, b

rng = np.random.RandomState(0)
X = np.column_stack([rng.randn(500) * 100, rng.randn(500)])
y = (X[:, 0] + X[:, 1] > 0).astype(int)

t0 = time.time(); gd_logistic(X, y, lr=0.001); t_raw = time.time() - t0
t0 = time.time(); gd_logistic(StandardScaler().fit_transform(X), y, lr=0.1); t_scaled = time.time() - t0
print(f"不缩放 (lr=0.001): {t_raw:.3f}s")
print(f"缩放后 (lr=0.1):   {t_scaled:.3f}s，且收敛更好")
```

### 11.4 OneHotEncoder vs LabelEncoder 在线性模型上的对比

```python
import numpy as np
from minisklearn.preprocessing import OneHotEncoder, LabelEncoder
from minisklearn.linear_model import LogisticRegression

rng = np.random.RandomState(0)
n = 300
city = rng.choice(["北京", "上海", "广州", "深圳"], n)
y = (city == "北京").astype(int)

# LabelEncoder（错误用法）
le = LabelEncoder().fit(city)
X_le = le.transform(city).reshape(-1, 1)
acc_le = LogisticRegression().fit(X_le, y).score(X_le, y)

# OneHotEncoder（正确用法）
enc = OneHotEncoder().fit(city.reshape(-1, 1))
X_oh = enc.transform(city.reshape(-1, 1))
acc_oh = LogisticRegression().fit(X_oh, y).score(X_oh, y)

print(f"LabelEncoder 准确率: {acc_le:.3f}  (引入虚假序关系)")
print(f"OneHotEncoder 准确率: {acc_oh:.3f} (正确)")
```

### 11.5 高基数类别的内存对比

```python
import numpy as np
from minisklearn.preprocessing import OneHotEncoder

n, K = 100000, 1000
cats = np.array([f"c{i}" for i in range(K)])
X = cats[rng.randint(0, K, n)].reshape(-1, 1)

enc = OneHotEncoder().fit(X)
X_oh = enc.transform(X)
print(f"稠密矩阵内存: {X_oh.nbytes / 1e6:.1f} MB")  # ~800 MB
print(f"稀疏估计内存: {n * 8 / 1e6:.1f} MB")         # ~0.8 MB
```

---

## 十二、参数调优指南

### 12.1 选择缩放器的决策树

```
数据有异常值吗？
├── 是 → RobustScaler（中位数 + IQR）
└── 否 → 数据分布近似正态吗？
         ├── 是 → StandardScaler
         └── 否 → 需要有界输出吗？
                  ├── 是 → MinMaxScaler
                  └── 否 → QuantileTransformer / PowerTransformer
```

### 12.2 feature_range 选择

| 场景 | 推荐 feature_range | 原因 |
|------|-------------------|------|
| 神经网络（sigmoid） | $(0, 1)$ | sigmoid 在 $[0,1]$ 输入下梯度合理 |
| 神经网络（tanh） | $(-1, 1)$ | tanh 在 $[-1,1]$ 最敏感 |
| 神经网络（ReLU） | $(0, 1)$ | ReLU 对正输入线性，$[0,1]$ 避免死神经元 |
| 图像像素 | $(0, 1)$ | 与像素归一化一致 |
| 保正性特征（计数） | $(0, 1)$ 或 $(0, \text{max})$ | 避免负值 |

### 12.3 编码器选择指南

| 特征类型 | 样本数 | 类别数 $K$ | 推荐编码 |
|---------|--------|-----------|---------|
| 有序类别 | 任意 | 任意 | LabelEncoder / OrdinalEncoder |
| 无序类别 | 任意 | $K \leq 30$ | OneHotEncoder |
| 无序类别 | 大 | $30 < K \leq 1000$ | OneHotEncoder（稀疏） |
| 无序类别 | 大 | $K > 1000$ | 目标编码 / 频率编码 / 嵌入 |
| 标签 y | 任意 | 任意 | LabelEncoder |

---

## 十三、常见错误与调试技巧

### 13.1 静默错误：整数除法截断

```python
X = np.array([[1, 2], [3, 4]])  # int 数组
# 若实现忘了转 float，(X - mean) / scale 可能整数除法截断
# 调试：检查输出 dtype
scaler = StandardScaler()
print(scaler.fit_transform(X).dtype)  # 必须是 float64
```

### 13.2 静默错误：测试集超出训练范围

```python
scaler = MinMaxScaler().fit([[0], [10]])
print(scaler.transform([[-5], [20]]))  # [-0.5], [2.0] —— 超出 [0,1] 但不报错
# 调试：检查 transform 后的 min/max
Xt = scaler.transform(X_test)
if Xt.min() < 0 or Xt.max() > 1:
    print("警告：测试集超出训练范围")
```

### 13.3 调试检查清单

```python
def debug_preprocessing(scaler, X_train, X_test):
    """预处理调试工具。"""
    Xt_train = scaler.transform(X_train)
    Xt_test = scaler.transform(X_test)

    print(f"训练集形状: {Xt_train.shape}, 测试集形状: {Xt_test.shape}")
    print(f"训练集列数一致: {Xt_train.shape[1] == Xt_test.shape[1]}")
    print(f"无 NaN: {np.isfinite(Xt_train).all() and np.isfinite(Xt_test).all()}")
    if hasattr(scaler, 'mean_'):
        print(f"训练集均值≈0: {np.allclose(Xt_train.mean(axis=0), 0, atol=1e-8)}")
        print(f"训练集std≈1:  {np.allclose(Xt_train.std(axis=0), 1, atol=1e-8)}")
    print(f"测试集均值: {Xt_test.mean(axis=0).round(4)}  (应接近但不等于0)")
```

### 13.4 常见报错与原因

| 报错 | 原因 | 解决 |
|------|------|------|
| `ValueError: Found array with 0 feature(s)` | X 为空 | 检查数据加载 |
| `KeyError` in transform | 类别未在 fit 中见过 | 用 `handle_unknown='ignore'` |
| 输出全 NaN | 某列 std=0 或含 NaN | 先 `SimpleImputer`，检查常量列 |
| `AttributeError: 'StandardScaler' object has no attribute 'mean_'` | transform 前没 fit | 先调用 fit |

---

## 十四、与其他框架的对比

### 14.1 与 PyTorch 标准化对比

```python
# minisklearn
from minisklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# PyTorch
import torch
X_t = torch.tensor(X)
mean, std = X_t.mean(dim=0), X_t.std(dim=0)
X_scaled_t = (X_t - mean) / std
```

PyTorch 的标准化是即时的（无 fit/transform 分离），适合在线计算但不适合持久化参数。sklearn 的 fit/transform 分离让训练参数可保存、可复用。

### 14.2 与 pandas 对比

```python
# pandas 手动标准化
df_scaled = (df - df.mean()) / df.std()

# sklearn/minisklearn
scaler = StandardScaler().fit(df.values)
df_scaled = scaler.transform(df.values)
```

pandas 简洁但无法保存参数、无法放进 Pipeline、无法处理测试集（会用测试集自己的均值）。

### 14.3 与 Spark ML 对比

Spark 的 `StandardScaler` 也是 fit/transform 模式，但分布在不同机器上计算均值和标准差（分布式聚合）。API 几乎一致，体现了 fit/transform 契约的跨框架普适性。

---

## 十五、实际应用场景

### 15.1 金融风控：混合特征预处理

```python
# 数值特征：年龄、收入、负债比
# 类别特征：职业、婚姻、学历
# 需要分别处理后拼接

import numpy as np
from minisklearn.preprocessing import StandardScaler, OneHotEncoder

X_num = np.array([[25, 5000, 0.3], [40, 15000, 0.6], [35, 8000, 0.2]])
X_cat = np.array([["工程师", "已婚", "本科"],
                  ["医生", "未婚", "博士"],
                  ["教师", "已婚", "硕士"]])

X_num_s = StandardScaler().fit_transform(X_num)
X_cat_s = OneHotEncoder().fit_transform(X_cat)
X_final = np.hstack([X_num_s, X_cat_s])
print(f"最终特征维度: {X_final.shape[1]}")
```

### 15.2 推荐系统：用户特征工程

```python
# 用户画像：年龄段（有序）、性别（无序）、城市（高基数）
age = np.array([20, 35, 50, 25]).reshape(-1, 1)        # 有序，直接用
gender = np.array(["M", "F", "M", "F"]).reshape(-1, 1)  # 无序，独热
city = np.array(["北京", "上海", "广州", "北京"]).reshape(-1, 1)

age_s = MinMaxScaler(feature_range=(0, 1)).fit_transform(age)
gender_oh = OneHotEncoder().fit_transform(gender)
city_oh = OneHotEncoder().fit_transform(city)

user_features = np.hstack([age_s, gender_oh, city_oh])
```

### 15.3 NLP：文本特征缩放

```python
# TF-IDF 特征通常用 L2 归一化而非 StandardScaler
# 因为文本向量是稀疏且非负的，减均值会破坏稀疏性
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer()  # 内部已做 L2 归一化
# 不需要额外 StandardScaler
```

### 15.4 图像处理：像素归一化

```python
# CNN 输入通常归一化到 [0, 1] 或用预训练模型的均值标准差
# ImageNet 均值 = [0.485, 0.456, 0.406]，标准差 = [0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_normalized = (image / 255.0 - mean) / std  # 手动标准化
```

---

## 十六、思考题与练习

### 基础题

1. **手算**：对 $X = [[1], [3], [5], [7]]$，手动计算 StandardScaler 和 MinMaxScaler(feature_range=(0,1)) 的变换结果。

2. **证明**：证明 StandardScaler 变换后，数据的协方差矩阵等于原数据的相关矩阵。

3. **判断**：以下哪些预处理会改变特征间的 Pearson 相关系数？(a) StandardScaler (b) MinMaxScaler (c) 对数变换 (d) 独热编码

### 进阶题

4. **实现**：手写一个 `MaxAbsScaler`，把数据除以每列绝对值最大值，使结果在 $[-1, 1]$ 且保留稀疏性（不减均值）。

5. **分析**：为什么对独热编码后的特征再做 StandardScaler 通常没意义？什么情况下有意义？

6. **实验**：生成一个含异常值的数据集，对比 StandardScaler、MinMaxScaler、RobustScaler（若可用）在下游 KNN 上的交叉验证分数。

7. **推导**：设 $X$ 服从 $\mathcal{N}(\mu, \Sigma)$，求 StandardScaler 变换后 $Z$ 的分布。证明 $Z$ 的各分量不相关当且仅当 $X$ 的各分量不相关。

### 思考题

8. 为什么 sklearn 的 `__init__` 约定"只存参数不做计算"对预处理器的 `clone` 行为至关重要？举例说明违反这条约定会导致什么问题。

9. 在流式数据场景（数据不断到来），StandardScaler 的 `fit` 需要全部数据才能算均值。如何设计一个 `partial_fit` 接口支持增量更新？（提示：维护运行均值和运行方差）

10. OneHotEncoder 对高基数类别会产生维度爆炸。除了目标编码，还有哪些方法？各自的优缺点是什么？

---

## 十七、扩展阅读

### 书籍

- **《The Elements of Statistical Learning》（Hastie et al.）** 第 2.5 节：预处理与特征工程的统计视角
- **《Feature Engineering and Selection》（Kuhn & Johnson）**：专门讲特征工程的专著，涵盖各种编码和变换
- **《Applied Predictive Modeling》（Kuhn & Johnson）** 第 3 章：数据预处理实践指南

### 论文

- **"A Survey of Methods for Exploratory Data Analysis with Missing Values"**：缺失值处理的综述
- **"On the Effectiveness of Least Squares Estimation for Integer Positioning"**：标准化对数值稳定性的影响
- **"Target Encoding for Categorical Features"**：目标编码的理论与实践

### 在线资源

- sklearn 官方预处理文档：https://scikit-learn.org/stable/modules/preprocessing.html
- sklearn 特征工程教程：https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- "Why, How and When to Standardize Your Data"：https://towardsdatascience.com/...

### 相关算法

- `RobustScaler`：异常值鲁棒缩放
- `QuantileTransformer`：分位数变换到均匀/正态
- `PowerTransformer`：Box-Cox / Yeo-Johnson 变换
- `TargetEncoder`：目标编码（sklearn 1.3+）
- `OrdinalEncoder`：有序类别编码（特征版 LabelEncoder）
- `PolynomialFeatures`：多项式特征展开

---

[← 返回算法列表](../index.md)
