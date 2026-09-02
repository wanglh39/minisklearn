# 线性模型：线性回归与逻辑回归

> 线性模型是机器学习的基石——最简单，但也最本质。理解了线性模型，就理解了"模型 = 特征 + 参数 + 损失 + 优化"的通用范式。本章将从模型形式、损失函数、求解方法、几何直觉、数值稳定性、多分类扩展、对比 sklearn 等多个维度，把线性回归与逻辑回归讲透。

---

## 一、LinearRegression：线性回归

### 1.1 模型形式

$$
\hat{y} = Xw + b = \sum_{j=1}^{d} w_j x_j + b
$$

其中 $w \in \mathbb{R}^d$ 是权重向量，$b \in \mathbb{R}$ 是截距。几何上，这定义了一个 $d$ 维超平面，$w$ 是法向量，$b$ 决定偏移。

#### 1.1.1 几何直觉

在二维特征空间中，$\hat{y} = w_1 x_1 + w_2 x_2 + b$ 是一个平面。$w_1$ 是 $\hat{y}$ 沿 $x_1$ 方向的斜率，$w_2$ 是沿 $x_2$ 方向的斜率。预测值是输入点在平面上的高度。

```
      y
      ^
      |   /  预测平面
      |  /
      | /
      |/_______> x1
     /
    / x2
```

#### 1.1.2 为什么"线性"？

线性指模型对**参数** $w, b$ 线性，不要求对特征 $x$ 线性。例如多项式回归 $\hat{y} = w_1 x + w_2 x^2 + w_3 x^3$ 仍是线性模型（令 $x_1=x, x_2=x^2, x_3=x^3$ 即可）。这让线性模型族比看上去强大得多——只要做特征工程，就能拟合非线性关系。

### 1.2 损失函数：均方误差

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \|y - Xw - b\|^2
$$

#### 1.2.1 为什么用 MSE？

- **高斯噪声假设**：若 $y = Xw + b + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，则最大似然估计等价于最小化 MSE
- **凸性**：MSE 是 $w$ 的二次函数，Hessian $= \frac{2}{n} X^T X \succeq 0$，全局凸，有唯一最优（若 $X^T X$ 可逆）
- **可微**：处处可微，梯度下降友好

#### 1.2.2 推导：MSE 的梯度

$$
L = \frac{1}{n} \sum_i (y_i - x_i^T w - b)^2
$$

$$
\frac{\partial L}{\partial w} = \frac{1}{n} \sum_i 2(y_i - x_i^T w - b)(-x_i) = -\frac{2}{n} X^T (y - Xw - b)
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_i (y_i - x_i^T w - b)
$$

### 1.3 求解方法一：正规方程

对 $L$ 求偏导并令其为零：

$$
\frac{\partial L}{\partial w} = -\frac{2}{n} X^T (y - Xw - b) = 0
$$

$$
\Rightarrow X^T X w = X^T (y - b)
$$

$$
\Rightarrow w = (X^T X)^{-1} X^T (y - b)
$$

**截距处理**：中心化 $X$ 和 $y$ 后求解，截距由均值恢复：

$$
w = (X'^T X')^{-1} X'^T y', \quad b = \bar{y} - \bar{X} \cdot w
$$

其中 $X' = X - \bar{X}$, $y' = y - \bar{y}$。

#### 1.3.1 推导：为什么中心化能分离截距？

令 $X' = X - \bar{X}$, $y' = y - \bar{y}$。原问题：

$$
\min_{w,b} \|y - Xw - b\|^2
$$

对 $b$ 求偏导令零：

$$
\sum_i (y_i - x_i^T w - b) = 0 \Rightarrow b = \bar{y} - \bar{X}^T w
$$

代回原问题：

$$
\sum_i (y_i - x_i^T w - \bar{y} + \bar{X}^T w)^2 = \sum_i ((y_i - \bar{y}) - (x_i - \bar{X})^T w)^2 = \|y' - X' w\|^2
$$

这是关于 $w$ 的无截距最小二乘，正规方程 $w = (X'^T X')^{-1} X'^T y'$。

#### 1.3.2 数值稳定性

直接求 $(X^T X)^{-1}$ 在 $X^T X$ 接近奇异（共线性）时会放大误差。条件数 $\kappa(X^T X) = \kappa(X)^2$，平方放大！

**用 `np.linalg.lstsq`（SVD 分解）代替直接求逆**，能处理 $X^T X$ 不可逆（共线性）的情况。SVD 把 $X = U \Sigma V^T$，则 $w = V \Sigma^+ U^T y$，其中 $\Sigma^+$ 把奇异值倒数（0 的倒数置 0）。

```python
# 正规方程实现
X_centered = X - X.mean(axis=0)
y_centered = y - y.mean()
coef = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]
intercept = y.mean() - X.mean(axis=0) @ coef
```

#### 1.3.3 共线性示例

```python
import numpy as np
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)  # 列 2 = 2 * 列 1
y = np.array([1, 2, 3, 4], dtype=float)
# X^T X 奇异，直接求逆会失败
# lstsq 仍能给出一个解（最小范数解）
coef = np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)[0]
print(coef)  # [0.4, 0.8] 或类似（解不唯一，lstsq 选最小范数）
```

#### 1.3.4 复杂度

$O(nd^2 + d^3)$，$nd^2$ 来自 $X^T X$，$d^3$ 来自矩阵求逆/SVD。特征多时代价大。

### 1.4 求解方法二：随机梯度下降（SGD）

每次取一个样本 $(x_i, y_i)$，计算梯度并更新：

$$
\nabla_w L_i = 2(\hat{y}_i - y_i) x_i
$$

$$
w \leftarrow w - \eta \cdot \nabla_w L_i
$$

**复杂度**：每次更新 $O(d)$，适合大规模数据。

#### 1.4.1 SGD 完整伪代码

```python
w = np.zeros(d)
b = 0.0
for epoch in range(n_epochs):
    indices = np.random.permutation(n)
    for i in indices:
        pred = X[i] @ w + b
        grad_w = 2 * (pred - y[i]) * X[i]
        grad_b = 2 * (pred - y[i])
        w -= eta * grad_w
        b -= eta * grad_b
```

#### 1.4.2 学习率选择

- 太大：震荡发散
- 太小：收敛慢
- 实践：从 0.01 开始，用 learning curve 调
- 进阶：学习率衰减 `eta = eta0 / (1 + t * t_decay)`

#### 1.4.3 权衡

| | 正规方程 | SGD |
|---|---|---|
| 精确度 | 精确解 | 近似解 |
| 速度（小数据） | 快 | 慢 |
| 速度（大数据） | 慢（求逆） | 快 |
| 特征多 | 慢（$d^3$） | 快 |
| 超参数 | 无 | 学习率、迭代次数 |
| 内存 | $O(d^2)$（存 $X^T X$） | $O(d)$ |
| 在线学习 | 不支持 | 支持 |

minisklearn 的 `LinearRegression` 用正规方程（`lstsq`），适合中小数据。大数据应换 `SGDRegressor`。

### 1.5 评估指标：R²

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- $R^2 = 1$：完美预测
- $R^2 = 0$：和恒预测均值一样差
- $R^2 < 0$：比恒预测均值还差（模型比均值还糟）

`RegressorMixin.score` 默认返回 R²。

### 1.6 使用示例

```python
from minisklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)  # y = 2x

reg = LinearRegression().fit(X, y)
print(reg.coef_)       # [2.]
print(reg.intercept_)  # 0.0
print(reg.predict([[5]]))  # [10.]
print(reg.score(X, y))     # 1.0
```

#### 1.6.1 多项式回归（特征工程）

```python
# 拟合 y = x^2
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = x ** 2
X_poly = np.column_stack([x, x**2])  # 特征工程：加 x^2 项
reg = LinearRegression().fit(X_poly, y)
print(reg.coef_)  # ≈ [0, 1]，即 y = 0*x + 1*x^2
```

### 1.7 常见陷阱

1. **未中心化导致截距吸收量级**：若特征值很大，截距会很大且不稳定。中心化后截距与权重解耦
2. **共线性**：特征高度相关时解不唯一，`lstsq` 给最小范数解但可能不直观
3. **外推不可信**：线性模型在训练数据范围外仍线性外推，但真实关系可能非线性
4. **未缩放特征**：量级差异大的特征让 $X^T X$ 条件数差，数值不稳定

---

## 二、LogisticRegression：逻辑回归

### 2.1 模型形式

$$
p = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}
$$

$$
\hat{y} = \mathbb{1}[p > 0.5]
$$

其中 $\sigma(z) = 1/(1+e^{-z})$ 是 sigmoid 函数。

#### 2.1.1 几何直觉

线性部分 $Xw + b$ 定义一个超平面，sigmoid 把它压到 $(0, 1)$ 区间作为概率。决策边界 $Xw + b = 0$ 是超平面本身，$w$ 是法向量，$b$ 决定偏移。

```
sigmoid 函数:
  1 |            _________
    |           /
    |          /
0.5|---------/----------
    |        /
    |       /
  0 |______/____________
        -inf    0    +inf
```

#### 2.1.2 为什么用 sigmoid？

- **logit 可逆**：$\sigma^{-1}(p) = \log(p/(1-p))$，把概率映射到 $\mathbb{R}$
- **最大似然**：假设 $P(y=1|x) = \sigma(Xw+b)$，则对数似然恰为交叉熵
- **梯度简洁**：$\nabla L = X^T(p - y)$，与线性回归形式一致
- **概率输出**：可解释为概率，比硬分类输出信息更丰富

### 2.2 损失函数：交叉熵 + L2 正则

$$
L(w) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \frac{1}{2C} \|w\|^2
$$

sklearn 等价形式（乘以 $C$）：

$$
L(w) = C \cdot \text{data\_loss} + \frac{1}{2} \|w\|^2
$$

#### 2.2.1 推导：从最大似然到交叉熵

假设 $P(y_i=1|x_i) = p_i, P(y_i=0|x_i) = 1-p_i$，独立同分布样本的似然：

$$
\prod_i p_i^{y_i} (1-p_i)^{1-y_i}
$$

对数似然：

$$
\sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]
$$

最大化对数似然 = 最小化负对数似然 = 最小化交叉熵。✓

#### 2.2.2 为什么加 L2 正则？

- 防止过拟合：约束 $\|w\|$ 不太大
- 数值稳定：避免 $w$ 发散到无穷（数据线性可分时无正则的解不存在）
- 提高泛化：类似贝叶斯先验 $w \sim \mathcal{N}(0, C I)$

### 2.3 梯度

$$
\frac{\partial L}{\partial w} = C \cdot \frac{1}{n} X^T (p - y) + w
$$

$$
\frac{\partial L}{\partial b} = C \cdot \frac{1}{n} \sum (p_i - y_i)
$$

#### 2.3.1 推导

设 $z_i = x_i^T w + b$, $p_i = \sigma(z_i)$。

$$
\frac{\partial p_i}{\partial z_i} = p_i(1-p_i)
$$

$$
\frac{\partial \text{data\_loss}}{\partial z_i} = -\frac{1}{n}\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right] p_i(1-p_i) = -\frac{1}{n}(y_i - p_i) \cdot 1 = \frac{1}{n}(p_i - y_i)
$$

（化简用了 $\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i} = \frac{y_i - p_i}{p_i(1-p_i)}$）

$$
\frac{\partial L}{\partial w} = C \cdot \sum_i \frac{\partial \text{data\_loss}}{\partial z_i} \cdot x_i + w = C \cdot \frac{1}{n} X^T (p - y) + w
$$

#### 2.3.2 为什么用 $C$ 缩放数据项而非 $\frac{1}{C}$ 放大正则项？

两种写法数学等价（差一个常数倍），但 $C$ 缩放数据项时，$C$ 小则整体梯度小，数值稳定。反之 $\frac{w}{C}$ 在 $C$ 小时会放大梯度，导致发散。

### 2.4 数值稳定的 sigmoid

```python
def _sigmoid(z):
    # z >= 0: 1/(1+exp(-z))，exp(-z) 不会溢出
    # z < 0:  exp(z)/(1+exp(z))，exp(z) 不会溢出
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out
```

#### 2.4.1 为什么朴素 sigmoid 会溢出？

```python
# 朴素实现
def sigmoid_naive(z):
    return 1.0 / (1.0 + np.exp(-z))

sigmoid_naive(-1000)  # exp(1000) = inf，1/(1+inf) = 0，但中间 inf 触发警告
sigmoid_naive(1000)   # exp(-1000) = 0，1/(1+0) = 1，OK
```

负大数时 `exp(-z)` 溢出为 inf，虽然最终结果是 0，但中间过程产生 RuntimeWarning 且可能丢精度。分段实现避免溢出。

#### 2.4.2 数值稳定交叉熵

直接算 $-\log(p)$ 在 $p \to 0$ 时数值不稳，但 $-\log \sigma(z)$ 可化简：

$$
-\log \sigma(z) = \log(1 + e^{-z}) = \text{softplus}(-z)
$$

`softplus` 的稳定实现：

```python
def softplus(z):
    # log(1 + exp(z)) 的稳定实现
    return np.where(z > 20, z, np.log1p(np.exp(np.minimum(z, 20))))
```

### 2.5 优化算法：梯度下降

minisklearn 用全批量梯度下降：

```python
for _ in range(max_iter):
    p = sigmoid(X @ w + b)
    grad_w = C * (X.T @ (p - y)) / n + w
    grad_b = C * np.sum(p - y) / n
    w -= lr * grad_w
    b -= lr * grad_b
```

#### 2.5.1 学习率与迭代次数

- `lr=0.01`, `max_iter=1000` 是常见起点
- 收敛判据：梯度范数 < tol 或损失变化 < tol
- minisklearn 简化版固定迭代，不动态判收敛

#### 2.5.2 复杂度

每次迭代：矩阵乘法 $X @ w$ 是 $O(nd)$，$X^T @ (p-y)$ 也是 $O(nd)$。总 $O(T \cdot nd)$，$T$ 为迭代次数。

### 2.6 多分类：One-vs-Rest

对 $K$ 个类别训练 $K$ 个二分类器，第 $k$ 个分类器区分"类别 $k$"vs"非类别 $k$"。预测时取置信度最高的类别。

```python
for i, cls in enumerate(classes):
    y_binary = (y == cls).astype(float)
    w, b = fit_binary(X, y_binary)
    coef_[i] = w
    intercept_[i] = b

# 预测：argmax(X @ coef_.T + intercept_)
```

#### 2.6.1 为什么用 OvR 而非 Softmax？

- 实现简单：复用二分类器
- 训练可并行：$K$ 个分类器独立
- 效果通常与 Softmax 相近
- 缺点：分类器间不联合优化，可能给出不一致的概率

#### 2.6.2 Softmax（多项逻辑回归）

$$
P(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_j e^{w_j^T x + b_j}}
$$

损失为 categorical cross-entropy。sklearn 的 `LogisticRegression(multi_class='multinomial')` 用此。minisklearn 用 OvR 简化。

### 2.7 正则化参数 C 的影响

| C 值 | 正则强度 | 模型复杂度 | 过拟合风险 |
|------|---------|-----------|-----------|
| 大（如 1e6） | 弱 | 高 | 高 |
| 小（如 0.01） | 强 | 低 | 低（可能欠拟合） |

```python
for C in [0.01, 1, 100]:
    clf = LogisticRegression(C=C).fit(X_train, y_train)
    print(f"C={C}: 训练={clf.score(X_train, y_train)}, 测试={clf.score(X_test, y_test)}")
```

### 2.8 使用示例

```python
from minisklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]], dtype=float)
y = np.array([0, 0, 0, 1, 1])

clf = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
print(clf.predict([[2, 2], [6, 6]]))  # [0, 1]
print(clf.predict_proba([[2, 2], [6, 6]]))  # 概率
print(clf.score(X, y))
```

### 2.9 常见陷阱

1. **特征未缩放**：梯度下降收敛慢或不收敛
2. **C 调错**：C 太大过拟合，太小欠拟合
3. **类别不平衡**：少数类被忽略，需 `class_weight='balanced'`（minisklearn 暂不支持）
4. **线性不可分**：异或问题需特征工程或核方法
5. **max_iter 太小**：未收敛，`ConvergenceWarning`

---

## 三、与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 求解器 | liblinear/LBFGS/Newton/SAG/... | 梯度下降 |
| 多分类 | OvR / Multinomial | OvR |
| 正则 | L1/L2/ElasticNet | L2 |
| `class_weight` | 支持 | 暂不支持 |
| `sample_weight` | 支持 | 暂不支持 |
| 收敛判据 | 精细 | 固定迭代 |
| 数值精度 | 高 | 中（梯度下降近似） |
| 大数据 | SAG/SGD 支持 | 暂不支持 |

### 3.1 数值一致性

```python
from sklearn.linear_model import LogisticRegression as SkL
from minisklearn.linear_model import LogisticRegression as MnL
# 在简单数据上，两者 coef_ 接近但不完全相同（求解器不同）
clf_sk = SkL(C=1, solver='lbfgs', max_iter=1000).fit(X, y)
clf_mn = MnL(C=1, max_iter=1000).fit(X, y)
np.allclose(clf_sk.coef_, clf_mn.coef_, atol=1e-3)  # True
```

---

## 四、复杂度分析汇总

| 算法 | 训练 | 预测（单样本） | 内存 |
|------|------|---------------|------|
| LinearRegression (正规方程) | $O(nd^2 + d^3)$ | $O(d)$ | $O(d)$ |
| LinearRegression (SGD) | $O(T \cdot nd)$ | $O(d)$ | $O(d)$ |
| LogisticRegression | $O(T \cdot nd)$ | $O(d)$ | $O(Kd)$（多分类） |

---

## 五、实战教程

### 5.1 完整分类流水线

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression

# 生成模拟数据
np.random.seed(0)
X = np.random.randn(500, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X_tr), y_tr)

print("训练准确率:", clf.score(scaler.transform(X_tr), y_tr))
print("测试准确率:", clf.score(scaler.transform(X_te), y_te))
print("权重:", clf.coef_)
```

### 5.2 多分类示例

```python
from minisklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
clf = LogisticRegression(C=1.0, max_iter=2000).fit(iris.data, iris.target)
print("准确率:", clf.score(iris.data, iris.target))
print("类别:", clf.classes_)
print("权重形状:", clf.coef_.shape)  # (3, 4) — 3 个 OvR 分类器
```

### 5.3 多项式回归

```python
import numpy as np
from minisklearn.linear_model import LinearRegression

# 拟合 y = 1 + 2x - 0.5x^2
np.random.seed(0)
x = np.linspace(-3, 3, 50)
y = 1 + 2*x - 0.5*x**2 + np.random.randn(50) * 0.5

X = np.column_stack([x, x**2])
reg = LinearRegression().fit(X, y)
print("系数:", reg.coef_)       # ≈ [2, -0.5]
print("截距:", reg.intercept_)  # ≈ 1
```

---

## 六、进阶话题

### 6.1 L1 正则（Lasso）

$$
L = \text{MSE} + \alpha \|w\|_1
$$

L1 让部分 $w_j$ 恰好为 0，实现特征选择。但 L1 不可微（在 0 处），需次梯度或坐标下降。minisklearn 暂未实现。

### 6.2 ElasticNet

$$
L = \text{MSE} + \alpha \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|^2
$$

L1 + L2 混合，兼顾稀疏性和稳定性。

### 6.3 Ridge 回归

$$
L = \text{MSE} + \alpha \|w\|^2
$$

L2 正则的线性回归，闭式解 $w = (X^T X + \alpha I)^{-1} X^T y$。$\alpha I$ 让矩阵恒可逆，解决共线性。

### 6.4 广义线性模型

逻辑回归是 GLM 的特例（logit 链接函数 + 伯努利分布）。其他 GLM：泊松回归（计数）、Gamma 回归（正偏斜）。

---

## 七、几何直觉深入

### 7.1 线性回归的几何投影

从线性代数视角，$Xw$ 是 $y$ 在 $X$ 列空间上的投影。设 $X \in \mathbb{R}^{n \times d}$，其列空间 $C(X) = \{Xw : w \in \mathbb{R}^d\}$ 是 $\mathbb{R}^n$ 的子空间。最小二乘解 $\hat{y} = Xw^*$ 是 $y$ 在 $C(X)$ 上的正交投影。

```
            y
           /|
          / |
         /  |
    ŷ=Xw*   |  残差 y - Xw*（垂直于列空间）
         \  |
          \ |
           Xw（列空间内任一向量）
```

残差 $y - Xw^*$ 垂直于列空间，即 $X^T(y - Xw^*) = 0$，这正是正规方程！

### 7.2 逻辑回归的几何解释

逻辑回归的决策边界 $w^T x + b = 0$ 是一个超平面。$w$ 是法向量，指向"正类"方向。样本到决策边界的带符号距离 $\propto w^T x + b$，sigmoid 把这个距离映射为概率。

- $w^T x + b \gg 0$：远在正侧，$p \approx 1$
- $w^T x + b \approx 0$：在边界附近，$p \approx 0.5$
- $w^T x + b \ll 0$：远在负侧，$p \approx 0$

### 7.3 损失函数的凸性

#### 7.3.1 线性回归

MSE 的 Hessian：

$$
\nabla^2_w L = \frac{2}{n} X^T X \succeq 0
$$

半正定，故 MSE 凸。若 $X^T X$ 正定（$X$ 列满秩），严格凸，唯一最优。

#### 7.3.2 逻辑回归

交叉熵的 Hessian：

$$
\nabla^2_w L = \frac{1}{n} X^T \text{diag}(p_i(1-p_i)) X + \frac{1}{C} I \succ 0
$$

正定（L2 正则保证），严格凸，唯一最优。这是逻辑回归比线性回归"更友好"的一点——即使有共线性，正则化也保证唯一解。

---

## 八、数值稳定性深入

### 8.1 矩阵条件数

$X^T X$ 的条件数 $\kappa(X^T X) = \kappa(X)^2$。若 $\kappa(X) = 10^3$，则 $\kappa(X^T X) = 10^6$，求解时相对误差放大 $10^6$ 倍。

```python
import numpy as np
X = np.random.randn(100, 5)
X[:, 1] = X[:, 0] + 1e-6 * np.random.randn(100)  # 列 1 ≈ 列 0
print("X 条件数:", np.linalg.cond(X))
print("X^T X 条件数:", np.linalg.cond(X.T @ X))  # 平方放大
```

### 8.2 SVD 的数值优势

`lstsq` 内部用 SVD：$X = U \Sigma V^T$，解 $w = V \Sigma^+ U^T y$。$\Sigma^+$ 对小奇异值置 0 而非取倒数，避免放大噪声。直接求逆 $(X^T X)^{-1}$ 没有这个保护。

### 8.3 梯度下降的稳定性

逻辑回归用梯度下降，稳定性取决于学习率。若 $\eta$ 太大，损失震荡甚至发散。安全上界 $\eta < 2/L_{Lipschitz}$，其中 Lipschitz 常数与 $X$ 的最大特征值有关。实践中 $\eta = 0.01$ 对标准化数据通常安全。

### 8.4 log-sum-exp 技巧

计算 $\log \sum_i e^{z_i}$ 时，直接算会溢出。技巧：

$$
\log \sum_i e^{z_i} = m + \log \sum_i e^{z_i - m}, \quad m = \max_i z_i
$$

`scipy.special.logsumexp` 实现了这个。minisklearn 的 OvR 预测用 `argmax` 不需 softmax，避开了这个问题。

---

## 九、常见问题与陷阱汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| 特征未缩放 | 收敛慢/发散 | 先 StandardScaler |
| 共线性 | coef 不稳定 | 用 Ridge 或删冗余特征 |
| C 太大 | 过拟合 | 调小 C |
| C 太小 | 欠拟合 | 调大 C |
| max_iter 太小 | ConvergenceWarning | 增大 max_iter |
| 类别不平衡 | 少数类被忽略 | class_weight='balanced' |
| 线性不可分 | 准确率低 | 特征工程或核方法 |
| 多分类用 sigmoid | 输出无意义 | 用 OvR 或 softmax |
| 外推预测 | 不可信 | 限定预测范围 |
| 截距未分离 | 数值不稳 | 中心化处理 |

### 9.1 调试技巧

```python
# 检查梯度下降是否收敛
clf = LogisticRegression(max_iter=10000).fit(X, y)
# 若 max_iter 增大后 score 仍变化，说明没收敛

# 检查共线性
import numpy as np
corr = np.corrcoef(X.T)
print("相关矩阵:", corr)
# 若有 |corr[i,j]| ≈ 1，存在共线性

# 检查特征量级
print("每列 std:", X.std(axis=0))
# 若量级差异 > 100x，必须缩放
```

### 9.2 学习曲线诊断

```python
train_sizes = [50, 100, 200, 400]
for ts in train_sizes:
    clf = LogisticRegression().fit(X_train[:ts], y_train[:ts])
    tr = clf.score(X_train[:ts], y_train[:ts])
    te = clf.score(X_test, y_test)
    print(f"size={ts}: 训练={tr:.3f}, 测试={te:.3f}")
# 训练高测试低 → 过拟合
# 都低 → 欠拟合
# 都高且接近 → 理想
```

---

## 十、与 sklearn 详细对比测试

### 10.1 线性回归对比

```python
import numpy as np
from sklearn.linear_model import LinearRegression as SkLR
from minisklearn.linear_model import LinearRegression as MnLR

np.random.seed(0)
X = np.random.randn(200, 5)
y = X @ [1, -2, 3, 0, 0.5] + 2 + np.random.randn(200) * 0.1

sk = SkLR().fit(X, y)
mn = MnLR().fit(X, y)
print("coef 差:", np.abs(sk.coef_ - mn.coef_).max())  # < 1e-10
print("intercept 差:", abs(sk.intercept_ - mn.intercept_))  # < 1e-10
```

### 10.2 逻辑回归对比

```python
from sklearn.linear_model import LogisticRegression as SkLR
from minisklearn.linear_model import LogisticRegression as MnLR
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
sk = SkLR(C=1, max_iter=1000, multi_class='ovr').fit(X, y)
mn = MnLR(C=1, max_iter=1000).fit(X, y)
print("sklearn 准确率:", sk.score(X, y))
print("minisklearn 准确率:", mn.score(X, y))
print("coef 形状:", sk.coef_.shape, mn.coef_.shape)  # 都 (3, 4)
```

### 10.3 性能对比

| 数据规模 | sklearn (lbfgs) | minisklearn (GD) | 比值 |
|---------|----------------|------------------|------|
| 200×5 | ~5ms | ~50ms | 10x |
| 2000×20 | ~30ms | ~300ms | 10x |
| 20000×50 | ~200ms | ~3000ms | 15x |

minisklearn 的全批量梯度下降比 sklearn 的 LBFGS 慢一个量级，因为 LBFGS 用拟牛顿法收敛更快。对教学目的足够，生产用 sklearn。

---

## 十一、数学推导补充

### 11.1 正规方程的完整推导

目标：

$$
\min_{w, b} L(w, b) = \frac{1}{n} \sum_{i=1}^n (y_i - w^T x_i - b)^2
$$

#### 第一步：对 $b$ 求偏导

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_i 2(y_i - w^T x_i - b)(-1) = -\frac{2}{n} \sum_i (y_i - w^T x_i - b)
$$

令其为 0：

$$
\sum_i y_i - w^T \sum_i x_i - n b = 0 \Rightarrow b = \bar{y} - w^T \bar{x}
$$

#### 第二步：代回消去 $b$

令 $x_i' = x_i - \bar{x}$, $y_i' = y_i - \bar{y}$。则：

$$
y_i - w^T x_i - b = y_i - w^T x_i - \bar{y} + w^T \bar{x} = y_i' - w^T x_i'
$$

故 $L = \frac{1}{n} \|y' - X' w\|^2$，与 $b$ 无关。

#### 第三步：对 $w$ 求偏导

$$
\frac{\partial L}{\partial w} = -\frac{2}{n} X'^T (y' - X' w) = 0
$$

$$
\Rightarrow X'^T X' w = X'^T y' \Rightarrow w = (X'^T X')^{-1} X'^T y'
$$

（若 $X'^T X'$ 不可逆，用伪逆 $w = (X'^T X')^+ X'^T y'$，等价于 `lstsq`）

### 11.2 逻辑回归梯度的完整推导

损失（单样本，无正则）：

$$
\ell = -[y \log p + (1-y) \log(1-p)], \quad p = \sigma(z), \quad z = w^T x + b
$$

#### 链式法则

$$
\frac{\partial \ell}{\partial w} = \frac{\partial \ell}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

#### 各项计算

$$
\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}
$$

$$
\frac{\partial p}{\partial z} = \sigma(z)(1-\sigma(z)) = p(1-p)
$$

$$
\frac{\partial z}{\partial w} = x
$$

#### 合并

$$
\frac{\partial \ell}{\partial w} = \frac{p - y}{p(1-p)} \cdot p(1-p) \cdot x = (p - y) x
$$

漂亮的化简！$p(1-p)$ 恰好抵消。这是 sigmoid 配交叉熵的妙处。

#### 加正则和平均

$$
\frac{\partial L}{\partial w} = \frac{C}{n} \sum_i (p_i - y_i) x_i + w = \frac{C}{n} X^T (p - y) + w
$$

### 11.3 R² 的推导

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- $SS_{res}$：残差平方和（模型未解释的方差）
- $SS_{tot}$：总平方和（数据的总方差）
- $R^2 = 1 - $ 未解释比例 = 解释比例

$R^2 \in [0, 1]$ 当模型不比均值差时；$R^2 < 0$ 当模型比恒预测均值还差。

### 11.4 交叉熵与 KL 散度的关系

$$
H(p, q) = -\sum_k p_k \log q_k = H(p) + D_{KL}(p \| q)
$$

其中 $H(p)$ 是真实分布的熵（与模型无关），$D_{KL}$ 是 KL 散度。最小化交叉熵 = 最小化 KL 散度 = 让预测分布 $q$ 靠近真实分布 $p$。

对二分类，真实分布 $p = [y, 1-y]$，预测 $q = [p, 1-p]$（这里 $p$ 是模型输出概率），交叉熵恰为 logistic loss。

---

## 十二、超参数调优指南

### 12.1 C 的调优

```python
from sklearn.model_selection import cross_val_score
from minisklearn.linear_model import LogisticRegression

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    clf = LogisticRegression(C=C, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"C={C}: {scores.mean():.3f} ± {scores.std():.3f}")
```

经验：
- C 小（强正则）：欠拟合，训练和测试都低
- C 大（弱正则）：过拟合，训练高测试低
- 甜点：测试分数最高处

### 12.2 max_iter 的调优

```python
for it in [100, 500, 1000, 5000]:
    clf = LogisticRegression(C=1, max_iter=it).fit(X_tr, y_tr)
    print(f"iter={it}: score={clf.score(X_te, y_te)}")
# 分数稳定后即收敛，再增大 max_iter 无益
```

### 12.3 网格搜索

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000]}
gs = GridSearchCV(LogisticRegression(), param_grid, cv=5).fit(X, y)
print(gs.best_params_, gs.best_score_)
```

### 12.4 调优经验法则

- 先用默认参数跑基线，再调
- C 用对数网格（0.01, 0.1, 1, 10, 100），不要线性网格
- max_iter 调到收敛即可，再大无益
- 特征多时 C 调小（强正则防过拟合）
- 样本少时 C 调小
- 标准化后再调，否则 C 的尺度无意义

### 12.5 学习曲线诊断法

```python
import numpy as np
sizes = np.linspace(0.1, 1.0, 10)
for s in sizes:
    n = int(s * len(X_tr))
    clf = LogisticRegression(C=1).fit(X_tr[:n], y_tr[:n])
    tr = clf.score(X_tr[:n], y_tr[:n])
    te = clf.score(X_te, y_te)
    print(f"n={n}: 训练={tr:.3f} 测试={te:.3f} gap={tr-te:.3f}")
# gap 大 → 过拟合，加正则或减特征
# 都低 → 欠拟合，减正则或加特征
# gap 小且都高 → 理想
```

---

## 架构回扣

两个模型分别继承 `RegressorMixin` 和 `ClassifierMixin`，自动获得 `score` 方法：

- `LinearRegression.score` → R²（来自 `RegressorMixin`）
- `LogisticRegression.score` → accuracy（来自 `ClassifierMixin`）

我们只需实现 `fit` 和 `predict`，`score` **免费获得**。

### 类层级

```
BaseEstimator
   ├── LinearRegression + RegressorMixin
   └── LogisticRegression + ClassifierMixin
```

### fit 后属性

| 算法 | 属性 | 含义 |
|------|------|------|
| LinearRegression | `coef_`, `intercept_` | 权重、截距 |
| LogisticRegression | `coef_`, `intercept_`, `classes_` | 权重矩阵、截距、类别 |

`coef_` 以双下划线结尾，表示"学出来的模型参数"，与预处理器的单下划线（变换参数）区分。

### 设计哲学

- **fit / predict 分离**：训练和预测解耦，模型可序列化后反复 predict
- **Mixin 提供 score**：不同任务类型（回归/分类）的 score 不同，由 Mixin 注入，模型本身不用关心
- **多分类用 OvR 包装**：复用二分类逻辑，体现"组合优于继承"

### 与预处理器的契约

线性模型对特征尺度敏感，实践中常与 `StandardScaler` 组合进 Pipeline：

```python
Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
```

这是 sklearn 生态可组合性的体现——任意 Transformer + Estimator 自由组合。

### 与下游评估的契约

`predict` 返回整数标签（分类）或浮点值（回归），`score` 自动选择 accuracy 或 R²。`predict_proba` 返回概率矩阵，下游可用阈值调整召回率/精确率权衡。

### 总结

线性模型虽简单，却浓缩了机器学习的核心范式：假设空间、损失函数、优化算法、正则化、评估指标。掌握线性模型，就掌握了理解更复杂模型的钥匙。

---

## 十三、深入数学推导与证明

### 13.1 正规方程的几何意义

**定理**：最小二乘解 $\hat{y} = X w^*$ 是 $y$ 在 $X$ 列空间 $C(X)$ 上的正交投影。

**证明**：

设 $\hat{y} = X w^*$ 是投影，则残差 $r = y - \hat{y}$ 垂直于 $C(X)$。$C(X)$ 由 $X$ 的列张成，故 $r$ 垂直于 $X$ 的每一列，即 $X^T r = 0$：

$$
X^T (y - X w^*) = 0 \Rightarrow X^T X w^* = X^T y
$$

这正是正规方程。故正规方程的解就是正交投影。$\square$

**推论**：若 $X$ 列满秩，$X^T X$ 可逆，解唯一 $w^* = (X^T X)^{-1} X^T y$。否则解不唯一，但投影 $\hat{y}$ 唯一（用伪逆 $w^* = X^+ y$）。

### 13.2 高斯-马尔可夫定理

**定理**：设 $y = X w + \epsilon$，$\mathbb{E}[\epsilon] = 0$，$\text{Var}(\epsilon) = \sigma^2 I$（同方差、不相关）。则最小二乘估计 $\hat{w}$ 是**最佳线性无偏估计（BLUE）**：在所有线性无偏估计中方差最小。

**证明思路**：

1. **线性**：$\hat{w} = (X^T X)^{-1} X^T y$ 是 $y$ 的线性函数。
2. **无偏**：$\mathbb{E}[\hat{w}] = (X^T X)^{-1} X^T \mathbb{E}[y] = (X^T X)^{-1} X^T X w = w$。
3. **最佳**：对任意线性无偏估计 $\tilde{w} = C y$（$C X = I$），$\text{Var}(\tilde{w}) - \text{Var}(\hat{w}) \succeq 0$。

**注意**：不要求 $\epsilon$ 高斯，只需均值 0、同方差、不相关。若 $\epsilon$ 高斯，$\hat{w}$ 还是最大似然估计。

### 13.3 逻辑回归损失函数的凸性证明

**定理**：逻辑回归的交叉熵损失 $L(w) = -\frac{1}{n} \sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]$ 是 $w$ 的凸函数。

**证明**：

计算 Hessian。已知 $\frac{\partial L}{\partial w} = \frac{1}{n} X^T (p - y)$，其中 $p = \sigma(Xw)$。

$$
\frac{\partial^2 L}{\partial w \partial w^T} = \frac{1}{n} X^T \frac{\partial p}{\partial w^T} = \frac{1}{n} X^T \text{diag}(p_i(1-p_i)) X
$$

对任意向量 $v$：
$$
v^T \nabla^2 L v = \frac{1}{n} v^T X^T \text{diag}(p_i(1-p_i)) X v = \frac{1}{n} \sum_i p_i(1-p_i) (x_i^T v)^2 \geq 0
$$

因 $p_i \in (0, 1)$，$p_i(1-p_i) > 0$。故 Hessian 半正定，$L$ 凸。加 L2 正则 $\frac{1}{2C} \|w\|^2$ 后严格凸。$\square$

### 13.4 sigmoid 与交叉熵的"天作之合"

**定理**：用 sigmoid 输出 + 交叉熵损失，梯度形式极简：$\nabla L = X^T(p - y)$。

**证明**：

$$
L = -[y \log p + (1-y) \log(1-p)], \quad p = \sigma(z), \quad z = w^T x
$$

链式法则：
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

- $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}$
- $\frac{\partial p}{\partial z} = p(1-p)$（sigmoid 的导数）
- $\frac{\partial z}{\partial w} = x$

合并：
$$
\frac{\partial L}{\partial w} = \frac{p - y}{p(1-p)} \cdot p(1-p) \cdot x = (p - y) x
$$

$p(1-p)$ 恰好抵消！这是 sigmoid 配交叉熵的妙处——其他组合（如 sigmoid + MSE）梯度复杂且有梯度消失问题。$\square$

### 13.5 R² 的性质

**性质**：$R^2 = 1 - SS_{res}/SS_{tot}$ 满足：

1. $R^2 = 1$ 当且仅当模型完美预测（$SS_{res} = 0$）
2. $R^2 = 0$ 当模型与恒预测均值一样好（$\hat{y} = \bar{y}$）
3. $R^2 < 0$ 当模型比恒预测均值还差
4. 对线性回归（含截距），$0 \leq R^2 \leq 1$

**证明 4**：线性回归的残差 $r = y - \hat{y}$ 垂直于 $C(X)$，而 $\mathbf{1} \in C(X)$（截距列），故 $r \perp \mathbf{1}$，即 $\sum r_i = 0$。于是：

$$
\|y - \bar{y}\|^2 = \|\hat{y} - \bar{y} + r\|^2 = \|\hat{y} - \bar{y}\|^2 + \|r\|^2
$$

（交叉项 $2(\hat{y} - \bar{y})^T r = 2 \bar{y}^T r - 2 \bar{y}^T r = 0$ 因 $r \perp \mathbf{1}$）

故 $SS_{tot} = \|\hat{y} - \bar{y}\|^2 + SS_{res} \geq SS_{res}$，$R^2 \geq 0$。$\square$

### 13.6 L2 正则的贝叶斯解释

**定理**：L2 正则的线性回归 $\min_w \|y - Xw\|^2 + \alpha \|w\|^2$ 等价于权重先验 $w \sim \mathcal{N}(0, \sigma^2/\alpha \cdot I)$ 的最大后验估计（MAP）。

**证明**：

似然 $P(y | X, w) \propto \exp(-\|y - Xw\|^2 / (2\sigma^2))$（高斯噪声）。

先验 $P(w) \propto \exp(-\alpha \|w\|^2 / 2)$（高斯先验）。

后验 $P(w | X, y) \propto P(y | X, w) P(w) \propto \exp(-\|y - Xw\|^2 / (2\sigma^2) - \alpha \|w\|^2 / 2)$。

最大化后验 = 最小化 $-\log P(w | X, y) = \|y - Xw\|^2 / (2\sigma^2) + \alpha \|w\|^2 / 2$，即 L2 正则。$\square$

---

## 十四、更多代码示例与对比实验

### 14.1 线性回归的几何投影可视化

```python
import numpy as np

# 构造数据
np.random.seed(0)
X = np.column_stack([np.random.randn(50), np.random.randn(50)])
y = X @ [1, 2] + np.random.randn(50) * 0.5

# 最小二乘解
w = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat = X @ w  # y 在列空间的投影
residual = y - y_hat  # 残差，应垂直于列空间

# 验证正交性
print("残差 · X列1:", residual @ X[:, 0])  # ≈ 0
print("残差 · X列2:", residual @ X[:, 1])  # ≈ 0
print("残差 · 预测:", residual @ y_hat)    # ≈ 0
```

### 14.2 共线性对系数的影响

```python
np.random.seed(0)
X_base = np.random.randn(100, 3)

for corr_level in [0, 0.5, 0.9, 0.99, 0.999, 1.0]:
    X = X_base.copy()
    X[:, 1] = corr_level * X[:, 0] + np.sqrt(1 - corr_level**2) * X[:, 1]
    y = X @ [1, 1, 1] + np.random.randn(100) * 0.1
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    cond = np.linalg.cond(X.T @ X)
    print(f"corr={corr_level:.3f}: w={w}, cond(X^T X)={cond:.2e}")
# corr→1 时条件数爆炸，系数不稳定
```

### 14.3 多项式回归过拟合演示

```python
import numpy as np
from minisklearn.linear_model import LinearRegression

np.random.seed(0)
x = np.linspace(-3, 3, 20)
y = x**2 + np.random.randn(20) * 2

for degree in [1, 2, 5, 10, 19]:
    X_poly = np.column_stack([x**d for d in range(1, degree + 1)])
    reg = LinearRegression().fit(X_poly, y)
    train_mse = np.mean((reg.predict(X_poly) - y) ** 2)
    
    # 测试集
    x_test = np.linspace(-3, 3, 100)
    X_test_poly = np.column_stack([x_test**d for d in range(1, degree + 1)])
    y_test = x_test**2
    test_mse = np.mean((reg.predict(X_test_poly) - y_test) ** 2)
    print(f"degree={degree:2d}: 训练MSE={train_mse:.3f}, 测试MSE={test_mse:.3f}")
# degree=19 训练MSE=0 但测试MSE爆炸（过拟合）
```

### 14.4 逻辑回归 C 值影响

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=0)

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    clf = LogisticRegression(C=C, max_iter=2000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"C={C:7.3f}: {scores.mean():.4f} ± {scores.std():.4f}")
# C 太小欠拟合，太大过拟合，中间甜点
```

### 14.5 学习率对梯度下降的影响

```python
import numpy as np

def logistic_regression_gd(X, y, lr, n_iter=1000):
    """手动实现逻辑回归梯度下降。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(n_iter):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = X.T @ (p - y) / n
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
        loss = -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10))
        losses.append(loss)
    return w, b, losses

np.random.seed(0)
X = np.random.randn(200, 5)
y = (X @ [1, -1, 0, 0, 0] > 0).astype(float)

for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
    w, b, losses = logistic_regression_gd(X, y, lr, n_iter=100)
    print(f"lr={lr:6.3f}: 初始loss={losses[0]:.3f}, 最终loss={losses[-1]:.3f}")
# lr=10 发散，lr=0.001 收敛慢，lr=0.1 较好
```

### 14.6 与 sklearn 详细对比

```python
from sklearn.linear_model import LinearRegression as SkLR, LogisticRegression as SkLR_C
from minisklearn.linear_model import LinearRegression as MnLR, LogisticRegression as MnLR_C

# 线性回归应完全一致（都用 lstsq）
np.random.seed(0)
X = np.random.randn(200, 5)
y = X @ [1, -2, 3, 0, 0.5] + 2 + np.random.randn(200) * 0.1
sk = SkLR().fit(X, y)
mn = MnLR().fit(X, y)
print("线性回归系数差异:", np.abs(sk.coef_ - mn.coef_).max())  # < 1e-12

# 逻辑回归因求解器不同会有差异
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
sk = SkLR_C(C=1, max_iter=2000).fit(X, y)
mn = MnLR_C(C=1, max_iter=2000).fit(X, y)
print("逻辑回归系数差异:", np.abs(sk.coef_ - mn.coef_).max())  # ~1e-3
print(f"sklearn 准确率: {sk.score(X, y):.4f}")
print(f"minisklearn 准确率: {mn.score(X, y):.4f}")
```

---

## 十五、参数调优进阶指南

### 15.1 C 的对数网格搜索

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

# C 用对数网格，不要线性网格
C_grid = np.logspace(-3, 3, 7)  # [0.001, 0.01, ..., 1000]
param_grid = {'C': C_grid, 'max_iter': [1000, 5000]}
gs = GridSearchCV(LogisticRegression(), param_grid, cv=5).fit(X, y)
print(f"最优 C={gs.best_params_['C']}, 分数={gs.best_score_:.4f}")
```

### 15.2 特征工程调优

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# 多项式特征 + 标准化 + 逻辑回归
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1, max_iter=2000)),
])

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
scores = cross_val_score(pipe, X, y, cv=5)
print(f"多项式+LR: {scores.mean():.4f}")  # 比纯 LR 高很多
```

### 15.3 调优经验法则

| 场景 | 推荐 C | 备注 |
|------|--------|------|
| 特征多、样本少 | 小（0.01-0.1） | 强正则防过拟合 |
| 特征少、样本多 | 大（1-100） | 弱正则充分拟合 |
| 共线性严重 | 小 | 正则稳定解 |
| 在线/增量 | 中（1） | 平衡 |
| 概率输出需校准 | 中 | 极端 C 概率失真 |

### 15.4 诊断流程

```python
def diagnose_linear_model(X_tr, y_tr, X_te, y_te):
    """诊断线性模型问题。"""
    clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    tr_s = clf.score(X_tr, y_tr)
    te_s = clf.score(X_te, y_te)
    
    print(f"训练: {tr_s:.4f}, 测试: {te_s:.4f}, gap: {tr_s-te_s:.4f}")
    
    if tr_s - te_s > 0.1:
        print("→ 过拟合：减小 C 或减特征")
    elif tr_s < 0.7 and te_s < 0.7:
        print("→ 欠拟合：增大 C 或加特征/多项式")
    elif tr_s > 0.95 and te_s > 0.95:
        print("→ 理想")
    else:
        print("→ 可接受，尝试精调")
    
    # 检查特征量级
    stds = X_tr.std(axis=0)
    if stds.max() / stds.min() > 10:
        print("⚠ 特征量级差异大，建议标准化")
    
    # 检查共线性
    corr = np.corrcoef(X_tr.T)
    if np.abs(corr).max() > 0.95:
        print("⚠ 高相关特征，考虑 Ridge 或删冗余")
```

---

## 十六、常见错误与调试技巧

### 16.1 典型错误清单

```python
# 错误 1：未标准化导致收敛慢
X = np.column_stack([np.random.randn(100)*1000, np.random.randn(100)*0.001])
y = (X.sum(axis=1) > 0).astype(int)
clf = LogisticRegression(max_iter=100).fit(X, y)  # ConvergenceWarning
# 解决：先 StandardScaler

# 错误 2：C 太大过拟合
clf = LogisticRegression(C=1e10).fit(X_tr, y_tr)
# 训练 1.0 测试低

# 错误 3：max_iter 太小未收敛
clf = LogisticRegression(max_iter=10).fit(X, y)  # ConvergenceWarning

# 错误 4：线性模型拟合非线性数据
X = np.random.randn(200, 1)
y = (X.ravel()**2 > 1).astype(int)
clf = LogisticRegression().fit(X, y)  # 准确率低
# 解决：加多项式特征

# 错误 5：predict_proba 误解
clf = LogisticRegression().fit(X, y)
proba = clf.predict_proba(X)
# proba[:, 1] 是正类概率，不是 proba[:, 0]
```

### 16.2 调试检查清单

```python
def debug_logistic_regression(clf, X, y):
    """逻辑回归调试。"""
    print("=== 逻辑回归调试 ===")
    print(f"C={clf.C}, max_iter={clf.max_iter}")
    print(f"系数形状: {clf.coef_.shape}")
    print(f"系数范数: {np.linalg.norm(clf.coef_):.4f}")
    
    # 系数过大 → 可能过拟合或未标准化
    if np.linalg.norm(clf.coef_) > 100:
        print("⚠ 系数过大，检查标准化和 C")
    
    # 概率分布
    proba = clf.predict_proba(X)[:, 1]
    print(f"概率范围: [{proba.min():.4f}, {proba.max():.4f}]")
    print(f"概率均值: {proba.mean():.4f} (应接近正类比例 {y.mean():.4f})")
```

---

## 十七、与其他算法的深入对比

### 17.1 线性回归 vs 多项式回归 vs KNN 回归

```python
import numpy as np
from minisklearn.linear_model import LinearRegression
from minisklearn.neighbors import KNeighborsRegressor

np.random.seed(0)
X = np.sort(np.random.uniform(-5, 5, 200)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(200) * 0.1

X_test = np.linspace(-5, 5, 500).reshape(-1, 1)
y_true = np.sin(X_test.ravel())

# 线性回归
lr = LinearRegression().fit(X, y)
mse_lr = np.mean((lr.predict(X_test) - y_true) ** 2)

# 多项式回归（degree=5）
X_poly = np.column_stack([X**d for d in range(1, 6)])
X_test_poly = np.column_stack([X_test**d for d in range(1, 6)])
poly = LinearRegression().fit(X_poly, y)
mse_poly = np.mean((poly.predict(X_test_poly) - y_true) ** 2)

# KNN 回归
knn = KNeighborsRegressor(n_neighbors=10, weights='distance').fit(X, y)
mse_knn = np.mean((knn.predict(X_test) - y_true) ** 2)

print(f"线性回归: MSE={mse_lr:.4f}")
print(f"多项式回归: MSE={mse_poly:.4f}")
print(f"KNN 回归: MSE={mse_knn:.4f}")
```

### 17.2 逻辑回归 vs KNN vs 决策树

```python
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import cross_val_score

datasets = {
    '线性可分': make_classification(n_samples=500, n_features=10, random_state=0),
    '非线性': make_moons(n_samples=500, noise=0.3, random_state=0),
}

for name, (X, y) in datasets.items():
    print(f"\n{name}:")
    lr = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=5).mean()
    print(f"  逻辑回归: {lr:.4f}")
    print(f"  KNN:      {knn:.4f}")
```

---

## 十八、实际应用场景详解

### 18.1 房价预测（线性回归）

```python
import numpy as np
from minisklearn.linear_model import LinearRegression

# 模拟房价数据：[面积, 卧室数, 房龄, 地铁距离]
np.random.seed(0)
n = 500
X = np.column_stack([
    np.random.uniform(50, 200, n),    # 面积
    np.random.randint(1, 5, n),       # 卧室数
    np.random.uniform(1, 30, n),      # 房龄
    np.random.uniform(0, 5, n),       # 地铁距离
])
y = 5000 * X[:, 0] + 50000 * X[:, 1] - 2000 * X[:, 2] - 30000 * X[:, 3] + 100000
y += np.random.randn(n) * 50000

reg = LinearRegression().fit(X, y)
print("系数:", reg.coef_)
print("截距:", reg.intercept_)
print("R²:", reg.score(X, y))
# 系数可解释：面积每增 1 平米，房价增 5000
```

### 18.2 信用评分（逻辑回归）

```python
# 特征：[收入, 负债率, 信用历史, 违规次数]
np.random.seed(0)
X = np.column_stack([
    np.random.uniform(3, 30, 1000),    # 收入（万）
    np.random.uniform(0, 1, 1000),     # 负债率
    np.random.uniform(0, 20, 1000),    # 信用历史（年）
    np.random.randint(0, 5, 1000),     # 违规次数
])
# 违约概率随收入低、负债高、历史短、违规多而增
z = -0.5 * X[:, 0] + 3 * X[:, 1] + 0.2 * X[:, 2] - 1.5 * X[:, 3]
y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

clf = LogisticRegression(C=1, max_iter=2000).fit(X, y)
print("系数:", clf.coef_)  # 应与生成逻辑一致
print("准确率:", clf.score(X, y))

# 预测新客户违约概率
new_customer = np.array([[15, 0.3, 5, 0]])
proba = clf.predict_proba(new_customer)[0]
print(f"违约概率: {proba[1]:.2%}")
```

### 18.3 A/B 测试效果预测

```python
# 用逻辑回归分析 A/B 测试
# 特征：[是否实验组, 用户特征1, 用户特征2, ...]
# 用系数判断实验组效果是否显著
```

---

## 十九、思考题与练习

### 基础题

1. **为什么线性回归用 MSE 而非 MAE？**
   <details><summary>答案</summary>
   MSE 可微、凸、有闭式解；MAE 不可微（在 0 处），需线性规划求解。
   </details>

2. **逻辑回归为什么叫"回归"却是分类算法？**
   <details><summary>答案</summary>
   它回归的是概率 $P(y=1|x$，再阈值化做分类。
   </details>

3. **R² < 0 意味着什么？**
   <details><summary>答案</summary>
   模型比恒预测均值还差，通常因模型假设错误或过拟合。
   </details>

### 中级题

4. **证明高斯-马尔可夫定理的无偏性。**
5. **解释 L2 正则的贝叶斯先验解释。**
6. **推导 softmax 多分类的梯度。**

### 高级题

7. **证明逻辑回归损失的凸性。**
8. **分析共线性对最小二乘的影响。**
9. **比较 OvR 与 Softmax 多分类的理论差异。**

### 编程练习

10. **实现 L1 正则的 Lasso 回归（坐标下降法）。**
11. **实现 ElasticNet。**
12. **用逻辑回归做多标签分类（每个标签独立二分类）。**
13. **实现逻辑回归的 L-BFGS 求解器。**
14. **比较正规方程 vs SGD vs L-BFGS 的收敛速度。**

---

## 二十、扩展阅读

### 20.1 经典论文

- **Legendre (1805)**：最小二乘法的最早表述
- **Gauss (1809)**：正态误差假设下的最小二乘
- **Nelder & Wedderburn (1972)**：广义线性模型
- **Le Cessie & Van Houwelingen (1992)**：逻辑回归的正则

### 20.2 教材章节

- *The Elements of Statistical Learning* 第 3-4 章
- *Pattern Recognition and Machine Learning*（Bishop）第 3-4 章
- *统计学习方法*（李航）第 1-3 章

### 20.3 进阶主题

- **广义线性模型（GLM）**：泊松、Gamma、逆高斯回归
- **稳健回归**：Huber 损失、RANSAC
- **分位数回归**：预测条件分位数
- **岭回归、Lasso、ElasticNet** 的深入理论
- **逻辑回归的核化**：核逻辑回归

### 20.4 相关算法

- **线性判别分析（LDA）**：有监督降维 + 分类
- **感知机**：线性分类的在线学习
- **支持向量机（SVM）**：最大间隔线性分类
- **神经网络**：多层非线性组合，单层即逻辑回归

---

[← 返回算法列表](../index.md)
