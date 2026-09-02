# 线性支持向量机（LinearSVC）

> SVM 寻找最大间隔超平面，用 hinge loss 的梯度下降求解。它代表了"几何直觉 + 优化算法"的完美结合——从"离最近的点最远"这一朴素想法出发，推导出一套优雅的凸优化框架。

---

## 一、原理

### 1.1 最大间隔

SVM 的核心直觉：在两类点之间画一条直线（超平面），让这条线离两边最近的点都尽可能远。这条"最宽走廊"的中线就是最优分类面，走廊的宽度叫**间隔（margin）**。

为什么最大间隔好？直觉上，间隔越大，对噪声和未来新数据的容忍度越高，泛化能力越强。这有 PAC 学习理论的支持：间隔 $\gamma$ 越大，VC 维相关的泛化界越紧。

形式化：超平面 $w \cdot x + b = 0$，分类决策 $\hat{y} = \mathrm{sign}(w \cdot x + b)$。点 $x_i$ 到超平面的几何距离是 $\dfrac{|w \cdot x_i + b|}{\|w\|}$。要求所有点正确分类且距离至少 $\gamma$：

$$
y_i(w \cdot x_i + b) \geq \gamma \|w\| \quad \forall i
$$

由于 $w, b$ 可以任意缩放（超平面不变），固定 $\gamma \|w\| = 1$（归一化约束），则最大化间隔 $\gamma = \dfrac{1}{\|w\|}$ 等价于最小化 $\|w\|$。标准形式：

$$
\max \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1
$$

或等价地：

$$
\min \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1
$$

这是个**凸二次规划**（目标函数凸，约束线性），有唯一全局最优解。

### 1.2 软间隔（hinge loss）

现实数据往往不可分（噪声、重叠），硬间隔约束 $y_i(w \cdot x_i + b) \geq 1$ 太严格。引入松弛变量 $\xi_i \geq 0$ 允许违反约束，但惩罚违反量：

$$
\min \frac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0
$$

消去 $\xi_i$（取 $\xi_i = \max(0, 1 - y_i(w \cdot x_i + b))$）得到无约束形式：

$$
\min \frac{1}{2}\|w\|^2 + C \sum \max(0, 1 - y_i(w \cdot x_i + b))
$$

其中 $\max(0, 1 - z)$ 就是 **hinge loss**（合页损失）。$C$ 是正则化强度的倒数：$C$ 大 → 少容忍违反 → 间隔窄 → 可能过拟合；$C$ 小 → 多容忍违反 → 间隔宽 → 可能欠拟合。

hinge loss 的形状：

```
loss
  │
3 │      ╱
  │      ╱
2 │      ╱
  │      ╱
1 │──────╱          ← hinge loss: max(0, 1-z)
  │      │
0 │──────┴────────
  │      1
  └──────────────── z = y·(w·x+b)
```

- $z \geq 1$（间隔满足）：loss = 0，无梯度
- $z < 1$（违反间隔）：loss = 1 - z，梯度 = -1
- $z = 1$ 处不可导，用次梯度

### 1.3 次梯度下降

违反间隔的样本（$y_i(w \cdot x_i + b) < 1$）贡献梯度：

$$
\nabla_w = w - C \sum_{i \in \text{violated}} y_i x_i
$$

完整推导：目标 $L = \dfrac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i(w \cdot x_i + b))$。

对 $w$ 求导（次梯度）：

$$
\frac{\partial L}{\partial w} = w + C \sum_i \frac{\partial \max(0, 1 - y_i(w \cdot x_i + b))}{\partial w}
$$

hinge loss 的次梯度：

$$
\frac{\partial \max(0, 1 - z)}{\partial z} = \begin{cases} 0 & z > 1 \\ [-1, 0] & z = 1 \\ -1 & z < 1 \end{cases}
$$

取 $z = 1$ 处的次梯度为 0（或 -1，都行），则：

$$
\frac{\partial L}{\partial w} = w - C \sum_{i: y_i(w \cdot x_i + b) < 1} y_i x_i
$$

对 $b$：

$$
\frac{\partial L}{\partial b} = -C \sum_{i: y_i(w \cdot x_i + b) < 1} y_i
$$

梯度下降更新：

$$
w \leftarrow w - \eta \nabla_w = w - \eta \left( w - C \sum_{i \in \text{violated}} y_i x_i \right)
$$
$$
b \leftarrow b - \eta \nabla_b = b + \eta C \sum_{i \in \text{violated}} y_i
$$

### 1.4 多分类：OvR 策略

二分类 SVM 推广到 $K$ 类用 **One-vs-Rest**：训练 $K$ 个二分类器，第 $k$ 个把类别 $k$ 当正类、其余当负类。预测时取决策值最大的类：

$$
\hat{y} = \arg\max_k (w_k \cdot x + b_k)
$$

OvR 训练 $K$ 个模型，简单高效。另一种是 OvO（One-vs-One），训练 $K(K-1)/2$ 个模型，每个区分两类。OvO 更准但更贵，sklearn 的 SVC 默认用 OvO，LinearSVC 默认用 OvR。

### 1.5 几何直觉

SVM 的几何画面：

1. **硬间隔**：两类点之间有一条最宽的"无人走廊"，走廊边界上的点叫**支持向量**（它们"撑起"了间隔）。其他点对解无影响——移除非支持向量不改变超平面。
2. **软间隔**：允许一些点进入走廊甚至跨到对面，但每个越界点按越界程度受惩罚。支持向量包括：走廊边界上的点、走廊内的点、跨界的点。
3. **$C$ 的作用**：$C \to \infty$ 退化为硬间隔（不容忍违反）；$C \to 0$ 间隔很宽但可能很多点违反（欠拟合）。

为什么叫"支持向量"？因为最优 $w = \sum_i \alpha_i y_i x_i$，只有 $\alpha_i > 0$ 的点（支持向量）对 $w$ 有贡献，其他点的 $\alpha_i = 0$。解只由少数支持向量决定，这是 SVM 稀疏性的来源，也是 SVM 在预测时快的原因。

---

## 二、数学推导详解

### 2.1 拉格朗日对偶

硬间隔 SVM 的拉格朗日函数：

$$
L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i [y_i(w \cdot x_i + b) - 1]
$$

KKT 条件：

$$
\frac{\partial L}{\partial w} = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i
$$
$$
\frac{\partial L}{\partial b} = 0 \Rightarrow \sum_i \alpha_i y_i = 0
$$

代入得到对偶问题：

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i \cdot x_j \quad \text{s.t.} \quad \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0
$$

对偶问题只涉及内积 $x_i \cdot x_j$，这是**核技巧**的入口：把内积换成核函数 $K(x_i, x_j)$ 就得到非线性 SVM。

### 2.2 软间隔的对偶

软间隔 SVM 的对偶：

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i \cdot x_j \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \; \sum_i \alpha_i y_i = 0
$$

与硬间隔唯一区别：$\alpha_i$ 有上界 $C$。KKT 互补条件 $\alpha_i [y_i(w \cdot x_i + b) - 1 + \xi_i] = 0$ 给出：
- $\alpha_i = 0$：非支持向量，正确分类且在间隔外
- $0 < \alpha_i < C$：在间隔边界上（$\xi_i = 0$）
- $\alpha_i = C$：越界点（$\xi_i > 0$）

### 2.3 从原始到无约束

minisklearn 的 LinearSVC 不解对偶 QP，而是直接在原始空间用梯度下降优化无约束的 hinge loss 形式：

$$
\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i(w \cdot x_i + b))
$$

这避开了 QP 求解器，实现简单，但对大规模线性问题效率不如对偶方法（如 sklearn 的 LinearSVC 用 liblinear 解对偶）。

### 2.4 次梯度的收敛性

hinge loss 不可导，但它是凸的 Lipschitz 函数，次梯度下降仍收敛。收敛速率：
- 凸函数：$O(1/\sqrt{T})$
- 强凸（$\|w\|^2$ 项）：$O(1/T)$

实践中加学习率衰减 $\eta_t = \eta_0 / (1 + t \cdot \eta_0)$ 可以保证收敛。

---

## 三、实现细节

### 3.1 完整实现

```python
import numpy as np
from ..base import BaseEstimator, ClassifierMixin

class LinearSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.01, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            # 二分类
            y_bin = np.where(y == self.classes_[1], 1, -1)
            self._fit_binary(X, y_bin)
        else:
            # 多分类：OvR
            self._ovr_coefs = []
            self._ovr_intercepts = []
            for cls in self.classes_:
                y_bin = np.where(y == cls, 1, -1)
                w, b = self._fit_binary_return(X, y_bin)
                self._ovr_coefs.append(w)
                self._ovr_intercepts.append(b)
        return self

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            margins = y * (X @ self.coef_ + self.intercept_)
            violated = margins < 1                              # 违反间隔的样本

            grad_w = self.coef_ - self.C * (X[violated] * y[violated, None]).sum(axis=0)
            grad_b = -self.C * y[violated].sum()

            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

            if np.linalg.norm(grad_w) < self.tol:
                break

    def _fit_binary_return(self, X, y):
        """与 _fit_binary 相同，但返回 w, b 而非存到 self。"""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.max_iter):
            margins = y * (X @ w + b)
            violated = margins < 1
            grad_w = w - self.C * (X[violated] * y[violated, None]).sum(axis=0)
            grad_b = -self.C * y[violated].sum()
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if np.linalg.norm(grad_w) < self.tol:
                break
        return w, b

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            return X @ self.coef_ + self.intercept_
        else:
            return np.array([X @ w + b
                             for w, b in zip(self._ovr_coefs, self._ovr_intercepts)]).T

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            return self.classes_[np.argmax(scores, axis=1)]
```

### 3.2 向量化解释

梯度计算的核心是这一行：

```python
grad_w = self.coef_ - self.C * (X[violated] * y[violated, None]).sum(axis=0)
```

拆解：
- `X[violated]`：选出违反间隔的样本，形状 $(m, d)$，$m$ 是违反数
- `y[violated, None]`：把 $(m,)$ 变成 $(m, 1)$，便于广播
- `X[violated] * y[violated, None]`：每行乘以对应的 $y_i$，形状 $(m, d)$
- `.sum(axis=0)`：对所有违反样本求和，形状 $(d,)$

这等价于 $\sum_{i \in \text{violated}} y_i x_i$，但用 NumPy 向量化，比 Python 循环快 100 倍。

### 3.3 学习率衰减

固定学习率可能振荡，加衰减更稳定：

```python
for t in range(self.max_iter):
    eta = self.learning_rate / (1 + t * self.learning_rate)
    # ... 用 eta 更新 ...
```

### 3.4 偏置项的处理

有些实现把 $b$ 并入 $w$，给 $X$ 加一列 1：

```python
X_aug = np.column_stack([X, np.ones(n_samples)])  # (n, d+1)
w_aug = np.zeros(n_features + 1)                   # (d+1,)
# 训练 w_aug，最后 b = w_aug[-1], w = w_aug[:-1]
```

这样代码更统一，但 $b$ 不该有正则化（$\|w\|^2$ 不含 $b$），所以分开处理更正确。

---

## 四、使用示例

### 4.1 基础二分类

```python
import numpy as np
from minisklearn.svm import LinearSVC

# 生成线性可分数据
rng = np.random.RandomState(42)
X = np.vstack([rng.randn(50, 2) + [2, 2],
               rng.randn(50, 2) + [-2, -2]])
y = np.array([0] * 50 + [1] * 50)

clf = LinearSVC(C=1.0, max_iter=1000, learning_rate=0.01)
clf.fit(X, y)
print(clf.coef_)           # [w1, w2]
print(clf.intercept_)      # b
print(clf.score(X, y))     # 准确率
```

### 4.2 多分类

```python
# 三类数据
X = np.vstack([rng.randn(50, 2) + [3, 0],
               rng.randn(50, 2) + [-3, 0],
               rng.randn(50, 2) + [0, 3]])
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

clf = LinearSVC(C=1.0).fit(X, y)
print(clf.predict(X[:5]))  # [0 0 0 0 0]
print(clf.score(X, y))
```

### 4.3 调 C

```python
for C in [0.01, 0.1, 1, 10, 100]:
    clf = LinearSVC(C=C).fit(X_train, y_train)
    print(f"C={C}: train={clf.score(X_train, y_train):.3f}, test={clf.score(X_test, y_test):.3f}")
# C 小：间隔宽，可能欠拟合
# C 大：间隔窄，可能过拟合
```

### 4.4 完整可运行示例

```python
import numpy as np
from minisklearn.svm import LinearSVC
from minisklearn.preprocessing import StandardScaler
from minisklearn.model_selection import train_test_split, cross_val_score
from minisklearn.pipeline import Pipeline

# 1. 生成数据
rng = np.random.RandomState(0)
X = np.vstack([rng.randn(100, 2) + [2, 2],
               rng.randn(100, 2) + [-2, -2]])
y = np.array([0] * 100 + [1] * 100)

# 2. 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 流水线（标准化 + SVM）
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=2000)),
])

# 4. 交叉验证
scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV 准确率: {scores.mean():.3f} ± {scores.std():.3f}")

# 5. 训练最终模型
pipe.fit(X_train, y_train)
print(f"测试准确率: {pipe.score(X_test, y_test):.3f}")
```

### 4.5 错误示例

```python
# 错误 1：标签不是 ±1
y = np.array([0, 1])  # hinge loss 假设 y ∈ {-1, +1}
# 实现内部要转换：y_bin = np.where(y == classes[1], 1, -1)

# 错误 2：学习率太大，发散
clf = LinearSVC(C=1.0, learning_rate=10.0).fit(X, y)
print(clf.coef_)  # nan 或 inf

# 错误 3：max_iter 太少，没收敛
clf = LinearSVC(max_iter=10).fit(X, y)
print(clf.score(X, y))  # 准确率低

# 错误 4：忘记标准化
# SVM 对尺度敏感，大尺度特征主导决策面
X = np.column_stack([rng.randn(100) * 1000, rng.randn(100)])
clf = LinearSVC().fit(X, y)
# 决策面几乎只看特征 1
```

### 4.6 对比示例：不同 C

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, C in zip(axes, [0.01, 1, 100]):
    clf = LinearSVC(C=C).fit(X, y)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    # 画决策面 w·x + b = 0
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yy = -(clf.coef_[0] * xx + clf.intercept_) / clf.coef_[1]
    ax.plot(xx, yy, 'r-')
    ax.set_title(f"C={C}")
plt.show()
```

---

## 五、与 sklearn 对比

### 5.1 API 一致性

| 特性 | minisklearn LinearSVC | sklearn LinearSVC |
|------|----------------------|-------------------|
| `C` | ✓ | ✓ |
| `max_iter` | ✓ | ✓ |
| `learning_rate` | ✓（梯度下降） | ✗（用 liblinear，无 lr） |
| `tol` | ✓ | ✓ |
| `fit` / `predict` | ✓ | ✓ |
| `decision_function` | ✓ | ✓ |
| `coef_` / `intercept_` | ✓ | ✓ |
| `classes_` | ✓ | ✓ |
| `support_vectors_` | ✗ | ✓（liblinear 不返回，SVC 才有） |
| `n_support_` | ✗ | ✓ |
| `dual_coef_` | ✗ | ✓ |
| 多分类策略 | OvR | OvR / crammer_singer |
| 求解器 | 次梯度下降 | liblinear（坐标下降） |
| 核支持 | ✗ | ✗（用 SVC） |

### 5.2 求解器差异

```python
# sklearn 用 liblinear（对偶坐标下降），收敛快且准
from sklearn.svm import LinearSVC as SkLinearSVC
clf_sk = SkLinearSVC(C=1.0).fit(X, y)

# minisklearn 用原始次梯度下降，简单但慢
from minisklearn.svm import LinearSVC
clf_mini = LinearSVC(C=1.0, max_iter=5000).fit(X, y)

# 系数应该接近
print(clf_sk.coef_, clf_mini.coef_)  # 方向一致，幅度可能略差
```

### 5.3 与 SVC 的区别

sklearn 有两个线性 SVM：
- `LinearSVC`：专门优化线性核，快，不支持核
- `SVC(kernel='linear')`：通用 SVM，支持核，但线性情况下比 LinearSVC 慢

```python
# 大规模线性数据，用 LinearSVC
clf = LinearSVC().fit(X_large, y_large)  # 快

# 非线性数据，用 SVC + RBF
clf = SVC(kernel='rbf').fit(X, y)  # 慢但能处理非线性
```

minisklearn 只实现 LinearSVC，不实现核 SVM（核 SVM 需要解 QP，实现复杂）。

### 5.4 数值结果对比

```python
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

clf_mini = LinearSVC(C=1.0, max_iter=10000).fit(X, y)
clf_sk = SkLinearSVC(C=1.0).fit(X, y)

print(f"minisklearn 准确率: {clf_mini.score(X, y):.4f}")
print(f"sklearn 准确率:     {clf_sk.score(X, y):.4f}")
# 应该接近
```

---

## 六、复杂度分析

### 6.1 训练复杂度

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 原始次梯度下降 | $O(T \cdot n \cdot d)$ | $O(d)$ |
| 对偶坐标下降（liblinear） | $O(T \cdot n \cdot d)$ 但常数小 | $O(n)$ |
| 核 SVM（SMO） | $O(n^2 \cdot d)$ ~ $O(n^3)$ | $O(n^2)$ |

$T$ 是迭代次数，$n$ 样本数，$d$ 特征数。

线性 SVM 的优势：$O(nd)$ 每次迭代，与 $n$ 线性（而非平方），适合大规模数据。

### 6.2 预测复杂度

| 方法 | 时间 | 空间 |
|------|------|------|
| 线性 SVM | $O(d)$ | $O(d)$ |
| 核 SVM | $O(s \cdot d)$，$s$ 支持向量数 | $O(s \cdot d)$ |

线性 SVM 预测只需一次内积，极快。核 SVM 预测要算 $s$ 个核函数，慢。

### 6.3 实测

```python
import numpy as np, time
from minisklearn.svm import LinearSVC

for n in [1000, 10000, 100000]:
    X = np.random.randn(n, 50)
    y = (X[:, 0] > 0).astype(int)
    t0 = time.time()
    LinearSVC(max_iter=100).fit(X, y)
    print(f"n={n}: {time.time()-t0:.3f}s")
```

---

## 七、数值稳定性

### 7.1 学习率选择

学习率太大 → 发散；太小 → 收敛慢。经验法则：$\eta_0 \approx 1 / \|X\|_{\text{op}}$（谱范数），或简单试 $0.001$、$0.01$、$0.1$。

```python
# 检查是否发散
clf = LinearSVC(learning_rate=1.0).fit(X, y)
if np.any(np.isnan(clf.coef_)) or np.any(np.isinf(clf.coef_)):
    print("发散了，减小 learning_rate")
```

### 7.2 特征标准化

SVM 对特征尺度敏感（hinge loss 里 $w \cdot x$ 的尺度影响间隔）。建议先标准化：

```python
pipe = Pipeline([('scaler', StandardScaler()),
                 ('svm', LinearSVC())])
```

### 7.3 大 C 的数值问题

$C$ 很大时，hinge loss 项主导，梯度大，容易发散。减小学习率或用对偶求解器。

### 7.4 标签编码

实现内部把标签转成 ±1，但用户标签可能是 0/1 或字符串。`classes_` 保存原始标签，`predict` 时映射回去。

---

## 八、常见问题与陷阱

### 8.1 线性不可分数据

```python
# XOR 问题：线性 SVM 不行
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = LinearSVC().fit(X, y)
print(clf.score(X, y))  # 0.5，瞎猜水平

# 解决：用核 SVM（sklearn.svm.SVC(kernel='rbf')）或加特征
X_feat = np.column_stack([X, X[:, 0] * X[:, 1]])  # 加交互项
clf = LinearSVC().fit(X_feat, y)
print(clf.score(X_feat, y))  # 1.0
```

### 8.2 不平衡数据

```python
# 90% 正样本，10% 负样本
X = np.vstack([rng.randn(900, 2), rng.randn(100, 2) + [3, 3]])
y = np.array([0] * 900 + [1] * 100)
clf = LinearSVC().fit(X, y)
print(clf.score(X, y))  # 0.9，但全预测 0 也有 0.9

# 解决：用 class_weight（sklearn 支持），或重采样
```

### 8.3 max_iter 不够

```python
clf = LinearSVC(max_iter=100).fit(X, y)  # 可能没收敛
# sklearn 会警告 "ConvergenceWarning: Liblinear failed to converge"
# 检查：看梯度范数是否足够小
```

### 8.4 多分类的决策值

OvR 下 $K$ 个二分类器各给一个决策值，取最大。但各分类器的决策值尺度可能不同（不同 $C$、不同正负样本比），直接比较有偏差。这是 OvR 的已知问题，OvO 更稳健。

### 8.5 predict 的阈值

```python
# decision_function 返回 w·x + b，predict 用 >= 0 判正
# 但可以自定义阈值
scores = clf.decision_function(X)
y_pred = (scores > 0.5).astype(int)  # 更保守的正类判定
```

---

## 九、实际使用教程

### 9.1 文本分类

线性 SVM 是文本分类的强 baseline（TF-IDF + LinearSVC）：

```python
from minisklearn.feature_extraction.text import TfidfVectorizer
from minisklearn.svm import LinearSVC
from minisklearn.pipeline import Pipeline

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC(C=1.0)),
])
pipe.fit(texts_train, labels_train)
pred = pipe.predict(texts_test)
```

### 9.2 大规模数据

```python
# 线性 SVM 适合大规模，比核 SVM 快得多
clf = LinearSVC(C=1.0, max_iter=1000).fit(X_large, y_large)
# 百万样本也能几分钟训完
```

### 9.3 调参

```python
from minisklearn.model_selection import GridSearchCV

grid = GridSearchCV(LinearSVC(), {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 5000],
}, cv=5)
grid.fit(X, y)
print(grid.best_params_)
```

### 9.4 可视化决策面

```python
import matplotlib.pyplot as plt

clf = LinearSVC().fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y)

# 决策面 w·x + b = 0
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yy = -(clf.coef_[0] * xx + clf.intercept_) / clf.coef_[1]
plt.plot(xx, yy, 'r-', label='决策面')

# 间隔边界 w·x + b = ±1
yy_up = -(clf.coef_[0] * xx + clf.intercept_ - 1) / clf.coef_[1]
yy_dn = -(clf.coef_[0] * xx + clf.intercept_ + 1) / clf.coef_[1]
plt.plot(xx, yy_up, 'k--', label='间隔')
plt.plot(xx, yy_dn, 'k--')
plt.legend()
plt.show()
```

---

## 十、变体与扩展

### 10.1 核 SVM

把内积 $x_i \cdot x_j$ 换成核 $K(x_i, x_j)$，得到非线性 SVM：

```python
# sklearn
from sklearn.svm import SVC
clf = SVC(kernel='rbf', gamma=0.1).fit(X, y)  # RBF 核
clf = SVC(kernel='poly', degree=3).fit(X, y)  # 多项式核
```

常用核：
- **线性**：$K(x, z) = x \cdot z$
- **多项式**：$K(x, z) = (x \cdot z + c)^d$
- **RBF**：$K(x, z) = \exp(-\gamma \|x - z\|^2)$
- **Sigmoid**：$K(x, z) = \tanh(\alpha x \cdot z + c)$

### 10.2 SVR（支持向量回归）

用 $\epsilon$-不敏感损失替代 hinge loss：

$$
L = \frac{1}{2}\|w\|^2 + C \sum \max(0, |y_i - \hat{y}_i| - \epsilon)
$$

### 10.3 One-Class SVM

无监督异常检测，找包含大部分数据的最小体积区域。

### 10.4 LinearSVR / LinearSVC 的优化

sklearn 的 LinearSVC 用 liblinear（对偶坐标下降），比原始梯度下降快且准。minisklearn 用原始梯度下降，简单但慢。生产环境用 sklearn。

---

## 十一、架构回扣

### 11.1 ClassifierMixin

LinearSVC 继承 `ClassifierMixin`，自动获得 `score`（计算准确率）：

```python
class ClassifierMixin:
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

### 11.2 双下划线属性

`coef_`、`intercept_`、`classes_` 是 fit 后学出的参数。`C`、`max_iter`、`learning_rate` 是 `__init__` 超参数。

### 11.3 OvR 的元估计器本质

OvR 多分类本质是元估计器：内部训练 $K$ 个二分类器。minisklearn 把它内联到 LinearSVC，sklearn 有独立的 `OneVsRestClassifier` 元估计器：

```python
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LinearSVC()).fit(X, y)  # 显式 OvR
```

### 11.4 在 Pipeline 中

```python
pipe = Pipeline([('scaler', StandardScaler()),
                 ('svm', LinearSVC())])
# GridSearchCV 可以搜 svm__C
```

### 11.5 decision_function 的作用

`decision_function` 返回原始决策值（到超平面的有符号距离 × $\|w\|$），用于：
- 自定义阈值分类
- 排序（如搜索相关性）
- 可视化

`predict` 是 `decision_function` + 阈值（0）的封装。

---

## 十二、进阶话题

### 12.1 SVM 的 VC 维

间隔 $\gamma$ 在半径 $R$ 的球内时，VC 维 $h \leq \lceil R^2 / \gamma^2 \rceil + 1$。间隔越大，VC 维越低，泛化界越紧。这是 SVM 泛化能力的理论保证。

### 12.2 hinge loss vs logistic loss

| 损失 | 形状 | 性质 |
|------|------|------|
| hinge | $\max(0, 1-y\hat{y})$ | 稀疏支持向量，对噪声鲁棒 |
| logistic | $\log(1 + e^{-y\hat{y}})$ | 概率输出，光滑 |
| 0-1 | $\mathbb{1}[y\hat{y} < 0]$ | 理想但不可优化 |

hinge loss 是 0-1 loss 的凸上界，比 logistic loss 更"激进"（一旦间隔满足就不再优化），这解释了 SVM 的稀疏性。

### 12.3 SMO 算法

核 SVM 用 SMO（Sequential Minimal Optimization）解对偶 QP。每次优化两个 $\alpha$ 变量，解析求解，循环至收敛。sklearn 的 SVC 用 libsvm 实现的 SMO。

### 12.4 L1-SVM

把 $\|w\|^2$ 换成 $\|w\|_1$ 得到 L1-SVM，产生稀疏 $w$（特征选择）：

```python
# sklearn
clf = LinearSVC(penalty='l1', dual=False).fit(X, y)
```

### 12.5 与 LogisticRegression 的关系

两者都是线性分类器，损失不同：
- SVM：hinge loss + L2 正则
- LR：logistic loss + L2 正则

实践中效果常接近，SVM 对离群点更鲁棒（hinge loss 线性增长，logistic 指数增长）。

---

## 十三、SMO 算法详解（核 SVM 的求解器）

虽然 minisklearn 的 LinearSVC 不解核 SVM，但理解 SMO 有助于看清 SVM 的全貌。

### 13.1 对偶问题

核 SVM 的对偶：

$$
\max_\alpha \; W(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$
$$
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \; \sum_i \alpha_i y_i = 0
$$

这是个 box 约束下的凸 QP。通用 QP 求解器（如 interior point）复杂度 $O(n^3)$，对大 $n$ 不可行。SMO 把问题分解：每次只优化两个 $\alpha$ 变量，其余固定。

### 13.2 为什么是两个？

约束 $\sum_i \alpha_i y_i = 0$ 要求 $\alpha$ 的变动必须保持加权和为 0。如果只改一个 $\alpha_i$，$\Delta \alpha_i y_i = 0 \Rightarrow \Delta \alpha_i = 0$，没法动。至少改两个：$\Delta \alpha_1 y_1 + \Delta \alpha_2 y_2 = 0$，即 $\Delta \alpha_2 = -\Delta \alpha_1 y_1 / y_2$。

### 13.3 两变量子问题的解析解

固定其余 $\alpha$，优化 $\alpha_1, \alpha_2$。代入约束后 $W$ 变成 $\alpha_2$ 的一元二次函数，有解析解：

$$
\alpha_2^{new,unc} = \alpha_2^{old} + \frac{y_2(E_1 - E_2)}{\eta}
$$

其中 $E_i = f(x_i) - y_i$ 是预测误差，$\eta = 2 K(x_1, x_2) - K(x_1, x_1) - K(x_2, x_2)$。

再裁剪到 box 约束 $[L, H]$：

$$
\alpha_2^{new} = \begin{cases} H & \alpha_2^{new,unc} > H \\ \alpha_2^{new,unc} & L \leq \alpha_2^{new,unc} \leq H \\ L & \alpha_2^{new,unc} < L \end{cases}
$$

然后 $\alpha_1^{new} = \alpha_1^{old} + y_1 y_2 (\alpha_2^{old} - \alpha_2^{new})$。

### 13.4 SMO 主循环

```python
def smo(X, y, C, K, max_iter, tol):
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    for _ in range(max_iter):
        num_changed = 0
        for i in range(n):
            E_i = f(X[i]) - y[i]   # f 用当前 alpha, b 计算
            if (y[i] * E_i < -tol and alpha[i] < C) or \
               (y[i] * E_i > tol and alpha[i] > 0):
                j = select_j(i, n)               # 启发式选 j
                E_j = f(X[j]) - y[j]
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                eta = 2 * K(X[i], X[j]) - K(X[i], X[i]) - K(X[j], X[j])
                alpha[j] += y[j] * (E_i - E_j) / eta
                # 裁剪到 [L, H] ...
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                # 更新 b ...
                num_changed += 1
        if num_changed == 0:
            break
    return alpha, b
```

### 13.5 启发式选择

SMO 的关键在于选哪两个变量。启发式：选违反 KKT 条件最严重的 $i$，再选使 $|E_i - E_j|$ 最大的 $j$。这大幅加速收敛。libsvm 的实现就是这套启发式。

---

## 十四、核方法详解

### 14.1 核技巧的原理

核技巧：不显式映射到高维空间，而在原空间用核函数计算内积。

**Mercer 定理**：$K$ 是合法核函数当且仅当对应的核矩阵半正定。

常用核的 Mercers 条件：
- RBF 核 $K(x, z) = e^{-\gamma\|x-z\|^2}$：对任意 $\gamma > 0$ 合法
- 多项式核 $K(x, z) = (x \cdot z + c)^d$：$c \geq 0, d \in \mathbb{N}$ 时合法

### 14.2 RBF 核的隐式映射

RBF 核对应无限维特征空间。证明：展开 $e^{-\gamma\|x-z\|^2} = e^{-\gamma\|x\|^2} e^{-\gamma\|z\|^2} e^{2\gamma x \cdot z}$，把 $e^{2\gamma x \cdot z}$ 泰勒展开得到无穷级数，每项对应一个特征维度。所以 RBF 核 SVM 能在无限维空间找线性分隔面，但计算量与原维度相同。

### 14.3 核矩阵的存储

核 SVM 要存 $n \times n$ 核矩阵，$O(n^2)$ 内存。$n = 10^5$ 时核矩阵 80GB，不可行。所以核 SVM 只适合中小数据（$n < 10^4$）。大规模数据用线性 SVM 或核近似（如 Nystroem、Random Fourier Features）。

### 14.4 核 SVM 的预测

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b
$$

预测一个新点要算 $s$ 个核函数（$s$ 支持向量数），$O(s \cdot d)$。支持向量多时预测慢。

---

## 十五、更多代码示例

### 15.1 手动实现次梯度下降（教学版）

```python
import numpy as np

def svm_sgd(X, y, C=1.0, eta=0.01, n_iter=1000):
    """最简的 SVM 次梯度下降（教学用）。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        margins = y * (X @ w + b)
        mask = margins < 1                       # 违反间隔
        grad_w = w - C * (X[mask] * y[mask, None]).sum(axis=0)
        grad_b = -C * y[mask].sum()
        w -= eta * grad_w
        b -= eta * grad_b
    return w, b

# 测试
rng = np.random.RandomState(0)
X = np.vstack([rng.randn(50, 2) + [2, 2], rng.randn(50, 2) + [-2, -2]])
y = np.array([-1] * 50 + [1] * 50)
w, b = svm_sgd(X, y)
print(f"w = {w}, b = {b}")
print(f"训练准确率: {np.mean(np.sign(X @ w + b) == y):.2%}")
```

### 15.2 逐样本更新（在线学习）

```python
def svm_sgd_online(X, y, C=1.0, eta=0.01, n_epochs=10):
    """逐样本更新，适合流式数据。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_epochs):
        for i in range(n):
            margin = y[i] * (X[i] @ w + b)
            if margin < 1:
                w -= eta * (w - C * y[i] * X[i])
                b -= eta * (-C * y[i])
            else:
                w -= eta * w                  # 只有正则项
    return w, b
```

### 15.3 Pegasos 算法

Pegasos 用 $1/t$ 学习率，有理论收敛保证：

```python
def pegasos(X, y, lam=0.01, n_epochs=10):
    """Pegasos: Primal Estimated sub-GrAdient SOlver for SVM."""
    n, d = X.shape
    w = np.zeros(d)
    t = 1
    for _ in range(n_epochs):
        for i in range(n):
            eta = 1.0 / (lam * t)
            if y[i] * (X[i] @ w) < 1:
                w = (1 - eta * lam) * w + eta * y[i] * X[i]
            else:
                w = (1 - eta * lam) * w
            t += 1
    return w
```

### 15.4 检查 KKT 条件（验证解的正确性）

```python
def check_kkt(X, y, alpha, C, K):
    """验证 KKT 条件（用于核 SVM 解的检查）。"""
    n = len(y)
    w = (alpha * y) @ X                         # w = sum alpha_i y_i x_i
    for i in range(n):
        margin = y[i] * (w @ X[i])
        if alpha[i] > 1e-8:                     # 支持向量
            assert margin <= 1 + 1e-3, f"样本 {i} 违反 KKT"
        if alpha[i] < C - 1e-8:                 # 非越界
            assert margin >= 1 - 1e-3, f"样本 {i} 违反 KKT"
    print("KKT 条件满足")
```

### 15.5 多分类 OvR 的完整实现

```python
class LinearSVC_OvR:
    def __init__(self, C=1.0):
        self.C = C
        self.classifiers_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_bin = np.where(y == cls, 1, -1)
            clf = LinearSVC(C=self.C).fit(X, y_bin)
            self.classifiers_.append(clf)
        return self

    def predict(self, X):
        scores = np.column_stack([clf.decision_function(X)
                                  for clf in self.classifiers_])
        return self.classes_[np.argmax(scores, axis=1)]
```

### 15.6 与 LogisticRegression 对比

```python
from minisklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = LinearSVC(C=1.0).fit(X_train, y_train)
lr = LogisticRegression(C=1.0).fit(X_train, y_train)

print(f"SVM 测试准确率: {svm.score(X_test, y_test):.4f}")
print(f"LR  测试准确率: {lr.score(X_test, y_test):.4f}")
# 通常接近，SVM 略好或相当
```

### 15.7 特征重要性

```python
clf = LinearSVC().fit(X, y)
importance = np.abs(clf.coef_)
print("特征重要性（按 |w_i| 排序）:")
for i in np.argsort(importance)[::-1]:
    print(f"  特征 {i}: {importance[i]:.4f}")
```

---

## 十六、损失函数对比

### 16.1 各种损失的梯度

| 损失 | $L(y, \hat{y})$ | $\partial L / \partial \hat{y}$ |
|------|----------------|-------------------------------|
| Hinge | $\max(0, 1 - y\hat{y})$ | $-y \cdot \mathbb{1}[y\hat{y} < 1]$ |
| Logistic | $\log(1 + e^{-y\hat{y}})$ | $-y \cdot \sigma(-y\hat{y})$ |
| 0-1 | $\mathbb{1}[y\hat{y} < 0]$ | 0 或不可导 |
| Squared Hinge | $\max(0, 1 - y\hat{y})^2$ | $-2y(1 - y\hat{y}) \cdot \mathbb{1}[y\hat{y} < 1]$ |
| Modified Huber | 见 sklearn 文档 | 光滑，对噪声鲁棒 |

### 16.2 损失形状对比

```
loss
  │
3 │      hinge     logistic
  │       ╱          ╱
2 │      ╱          ╱
  │     ╱          ╱
1 │────╱         ╱
  │    │        ╱
0 │────┴──────╱────────
  │    1
  └──────────────── y·ŷ
```

hinge loss 在 $y\hat{y} \geq 1$ 时梯度为 0（不再优化），logistic loss 永远有梯度。这解释了 SVM 的稀疏性（间隔满足的样本不贡献梯度）。

### 16.3 选择建议

- **SVM (hinge)**：大规模线性、文本分类、对稀疏性有需求
- **LR (logistic)**：需要概率输出、可解释性
- **Squared hinge**：对噪声更敏感但更光滑
- **Modified Huber**：对噪声鲁棒且光滑

---

## 十七、总结

| 要点 | 内容 |
|------|------|
| 核心思想 | 最大间隔超平面 |
| 数学基础 | 凸 QP / hinge loss 次梯度下降 |
| 实现 | 次梯度下降，OvR 多分类 |
| 复杂度 | 训练 $O(Tnd)$，预测 $O(d)$ |
| 数值稳定 | 标准化特征，调学习率 |
| 常见陷阱 | 线性不可分、不平衡、未标准化 |
| 与 sklearn | API 一致，求解器不同（梯度下降 vs liblinear） |
| 适用场景 | 大规模线性分类、文本分类 |
| 不适用 | 非线性（用核 SVM 或神经网络） |
| 核 SVM | 用 SMO 解对偶，$O(n^2)$~$O(n^3)$，适合中小数据 |
| 与 LR 关系 | 都是线性分类器，损失不同，效果常接近 |

---

## 十八、更深入的数学推导与证明

### 18.1 间隔的几何意义严格推导

**命题**：点 $x_0$ 到超平面 $H = \{x : w \cdot x + b = 0\}$ 的欧氏距离为 $|w \cdot x_0 + b| / \|w\|$。

**证明**：$x_0$ 到 $H$ 的最近点 $x^*$ 沿法向量方向，$x^* = x_0 - t w$（$t$ 待定）。$x^* \in H$ 故 $w \cdot x^* + b = 0$：

$$
w \cdot (x_0 - t w) + b = 0 \Rightarrow t = \frac{w \cdot x_0 + b}{\|w\|^2}
$$

距离 $\|x_0 - x^*\| = |t| \|w\| = \frac{|w \cdot x_0 + b|}{\|w\|}$。$\square$

**推论**：间隔（两类最近点到超平面的距离之和）为 $2 / \|w\|$（在约束 $y_i(w \cdot x_i + b) \geq 1$ 下）。最大化间隔等价于最小化 $\|w\|$。

### 18.2 强对偶性（Slater 条件）

**定理**：软间隔 SVM 的原始问题和对偶问题强对偶成立（最优值相等）。

**证明**：原始问题是凸 QP（凸目标 + 线性约束）。Slater 条件要求存在严格可行点：取 $\xi_i = 2$（对所有 $i$），$w = 0$，$b = 0$，则 $y_i(0) \geq 1 - 2 = -1$ 严格成立，$\xi_i > 0$ 严格成立。Slater 条件满足，强对偶成立。$\square$

### 18.3 KKT 互补条件的完整推导

软间隔 SVM 的 KKT 条件：

$$
\alpha_i [y_i(w \cdot x_i + b) - 1 + \xi_i] = 0 \quad \text{(互补松弛)}
$$
$$
\mu_i \xi_i = 0 \quad \text{($\mu_i$ 是 $\xi_i \geq 0$ 的乘子)}
$$
$$
\alpha_i + \mu_i = C \quad \text{(对 $\xi_i$ 的驻点条件)}
$$

分类讨论：

1. **$\alpha_i = 0$**：$\mu_i = C > 0$，故 $\xi_i = 0$，$y_i(w \cdot x_i + b) \geq 1$。样本在间隔外，正确分类，非支持向量。

2. **$0 < \alpha_i < C$**：$\mu_i > 0$，$\xi_i = 0$，$y_i(w \cdot x_i + b) = 1$。样本恰在间隔边界上，是**自由支持向量**。

3. **$\alpha_i = C$**：$\mu_i = 0$，$\xi_i \geq 0$，$y_i(w \cdot x_i + b) = 1 - \xi_i \leq 1$。样本在间隔内或越界，是**越界支持向量**。

这给出了支持向量的完整刻画：$\alpha_i > 0$ 的样本就是支持向量。

### 18.4 hinge loss 是 0-1 loss 的凸上界

**命题**：$\max(0, 1 - y\hat{y}) \geq \mathbb{1}[y\hat{y} < 0]$。

**证明**：分两种情况：

- $y\hat{y} \geq 1$：左边 $= 0$，右边 $= 0$（$y\hat{y} > 0$），成立。
- $y\hat{y} < 1$：左边 $= 1 - y\hat{y} > 0$。若 $y\hat{y} \geq 0$，右边 $= 0 \leq$ 左边；若 $y\hat{y} < 0$，右边 $= 1$，左边 $= 1 - y\hat{y} > 1 \geq$ 右边。$\square$

凸性：$\max(0, 1-z)$ 是两个凸函数（$0$ 和 $1-z$）的逐点最大，故凸。

### 18.5 SVM 的泛化界

**定理**（间隔界）：设数据在半径 $R$ 的球内，间隔 $\gamma$，则 VC 维 $h \leq \lceil R^2 / \gamma^2 \rceil + 1$。泛化误差以高概率不超过 $\sqrt{(h \ln(2n/h) + \ln(4/\delta)) / n}$。

**意义**：间隔越大 → VC 维越低 → 泛化界越紧。这是 SVM"最大间隔"哲学的理论依据。注意这个界与**维度 $d$ 无关**——即使特征空间无限维（RBF 核），只要间隔大，泛化仍有保证。

### 18.6 RBF 核对应无限维空间

**命题**：RBF 核 $K(x, z) = e^{-\gamma\|x-z\|^2}$ 对应无限维特征映射。

**证明**：

$$
e^{-\gamma\|x-z\|^2} = e^{-\gamma\|x\|^2} e^{-\gamma\|z\|^2} e^{2\gamma x \cdot z}
$$

对 $e^{2\gamma x \cdot z}$ 泰勒展开：

$$
e^{2\gamma x \cdot z} = \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!} (x \cdot z)^k = \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!} \sum_{|\alpha|=k} \binom{k}{\alpha} (x^\alpha)(z^\alpha)
$$

每项对应一个特征维度，无穷多项 → 无限维。特征映射 $\phi(x)$ 的第 $\alpha$ 分量为 $\sqrt{\frac{(2\gamma)^{|\alpha|}}{\alpha!}} e^{-\gamma\|x\|^2} x^\alpha$。$\square$

---

## 十九、更多代码示例与对比实验

### 19.1 不同 C 值的决策边界对比

```python
import numpy as np
import matplotlib.pyplot as plt
from minisklearn.svm import LinearSVC

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(50, 2) + [2, 2], rng.randn(50, 2) + [-2, -2]])
y = np.array([0]*50 + [1]*50)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, C in zip(axes, [0.01, 0.1, 1, 100]):
    clf = LinearSVC(C=C, max_iter=5000).fit(X, y)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yy = -(clf.coef_[0] * xx + clf.intercept_) / clf.coef_[1]
    ax.plot(xx, yy, 'k-')
    # 间隔边界
    margin = 1 / np.linalg.norm(clf.coef_)
    yy_up = yy + margin / abs(clf.coef_[1])
    yy_dn = yy - margin / abs(clf.coef_[1])
    ax.plot(xx, yy_up, 'k--', alpha=0.5)
    ax.plot(xx, yy_dn, 'k--', alpha=0.5)
    ax.set_title(f"C={C}, 间隔={2*np.linalg.norm(clf.coef_)**-1:.2f}")
plt.tight_layout()
plt.show()
```

### 19.2 SVM vs LogisticRegression：不同噪声水平

```python
import numpy as np
from minisklearn.svm import LinearSVC
from minisklearn.linear_model import LogisticRegression

rng = np.random.RandomState(0)
for noise in [0.5, 2.0, 5.0]:
    X = np.vstack([rng.randn(100, 2) * noise + [3, 3],
                   rng.randn(100, 2) * noise + [-3, -3]])
    y = np.array([0]*100 + [1]*100)

    svm = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
    lr = LogisticRegression(C=1.0).fit(X, y)
    print(f"noise={noise}: SVM={svm.score(X, y):.3f}, LR={lr.score(X, y):.3f}")
# 噪声大时 SVM 更鲁棒（hinge loss 线性增长，logistic 指数增长）
```

### 19.3 SVM vs LogisticRegression：离群点

```python
import numpy as np
from minisklearn.svm import LinearSVC
from minisklearn.linear_model import LogisticRegression

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(50, 2) + [2, 2], rng.randn(50, 2) + [-2, -2]])
y = np.array([0]*50 + [1]*50)

# 加极端离群点
X = np.vstack([X, [[10, -10], [12, -12]]])
y = np.concatenate([y, [0, 0]])

svm = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
lr = LogisticRegression(C=1.0).fit(X, y)
print(f"有离群点: SVM={svm.score(X[:100], y[:100]):.3f}, LR={lr.score(X[:100], y[:100]):.3f}")
# SVM 对离群点更鲁棒
```

### 19.4 学习率对收敛的影响

```python
import numpy as np
from minisklearn.svm import LinearSVC

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(100, 2) + [2, 2], rng.randn(100, 2) + [-2, -2]])
y = np.array([0]*100 + [1]*100)

for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
    clf = LinearSVC(C=1.0, learning_rate=lr, max_iter=1000).fit(X, y)
    if np.any(np.isnan(clf.coef_)):
        print(f"lr={lr}: 发散")
    else:
        print(f"lr={lr}: 准确率={clf.score(X, y):.3f}, ||w||={np.linalg.norm(clf.coef_):.3f}")
```

### 19.5 多分类 OvR 决策面可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from minisklearn.svm import LinearSVC

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(50, 2) + [3, 0],
               rng.randn(50, 2) + [-3, 0],
               rng.randn(50, 2) + [0, 3]])
y = np.array([0]*50 + [1]*50 + [2]*50)

clf = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')

# 画 OvR 的 3 个决策面
xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
Z = clf.predict(grid).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5, 1.5], colors='k', linestyles='--')
plt.title("OvR 多分类决策面")
plt.show()
```

### 19.6 特征标准化前后对比

```python
import numpy as np
from minisklearn.svm import LinearSVC
from minisklearn.preprocessing import StandardScaler

rng = np.random.RandomState(0)
X = np.column_stack([rng.randn(200) * 100, rng.randn(200)])
y = (X[:, 0] / 100 + X[:, 1] > 0).astype(int)

print("不标准化:")
clf = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
print(f"  准确率={clf.score(X, y):.3f}, coef={clf.coef_.round(3)}")

print("标准化后:")
Xs = StandardScaler().fit_transform(X)
clf = LinearSVC(C=1.0, max_iter=5000).fit(Xs, y)
print(f"  准确率={clf.score(Xs, y):.3f}, coef={clf.coef_.round(3)}")
# 标准化后两个特征系数更均衡
```

---

## 二十、参数调优指南

### 20.1 C 的选择策略

| C 值 | 行为 | 适用场景 |
|------|------|---------|
| $C \to 0$ | 间隔宽，多违反，欠拟合 | 噪声大、想强正则 |
| $C \sim 1$ | 平衡 | 默认起点 |
| $C \to \infty$ | 间隔窄，少违反，过拟合 | 干净数据、硬间隔近似 |

**调参流程**：

```python
from minisklearn.model_selection import GridSearchCV, cross_val_score
from minisklearn.svm import LinearSVC

# 对数网格搜索
C_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for C in C_grid:
    scores = cross_val_score(LinearSVC(C=C, max_iter=5000), X, y, cv=5)
    print(f"C={C:7.3f}: {scores.mean():.3f} ± {scores.std():.3f}")
# 选 CV 均值最高且方差小的 C
```

### 20.2 学习率调参

```python
# 经验法则：lr ≈ 1 / (C * ||X||^2_max)
import numpy as np
lr_init = 1.0 / (1.0 * np.max(np.sum(X**2, axis=1)))
print(f"建议初始学习率: {lr_init:.4f}")

# 若发散，减半；若收敛慢，加倍
```

### 20.3 max_iter 调参

```python
# 监控梯度范数判断收敛
clf = LinearSVC(C=1.0, max_iter=10000, tol=1e-6).fit(X, y)
# 若警告未收敛，增大 max_iter 或放宽 tol
```

---

## 二十一、常见错误与调试技巧

### 21.1 学习率过大发散

```python
clf = LinearSVC(learning_rate=10.0).fit(X, y)
if np.any(np.isnan(clf.coef_)) or np.any(np.isinf(clf.coef_)):
    print("发散！减小 learning_rate")
# 调试：监控训练过程中 coef_ 是否爆炸
```

### 21.2 线性不可分数据

```python
# XOR 问题
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
clf = LinearSVC().fit(X, y)
print(f"XOR 准确率: {clf.score(X, y)}")  # 0.5，线性 SVM 无法处理

# 解决：加非线性特征 或 用核 SVM
X_feat = np.column_stack([X, X[:, 0] * X[:, 1]])  # 加交互项
clf = LinearSVC().fit(X_feat, y)
print(f"加特征后: {clf.score(X_feat, y)}")  # 1.0
```

### 21.3 不平衡数据

```python
# 90% 正样本
rng = np.random.RandomState(0)
X = np.vstack([rng.randn(900, 2), rng.randn(100, 2) + [3, 3]])
y = np.array([0]*900 + [1]*100)
clf = LinearSVC().fit(X, y)
print(f"准确率: {clf.score(X, y)}")  # 0.9，但全预测 0 也 0.9
# 调试：看分类报告而非仅准确率
# 解决：class_weight='balanced'（sklearn）或重采样
```

### 21.4 调试检查清单

```python
def debug_svm(clf, X, y):
    """SVM 调试工具。"""
    print(f"类别: {clf.classes_}")
    print(f"coef_: {clf.coef_}, intercept_: {clf.intercept_}")
    print(f"||w||: {np.linalg.norm(clf.coef_):.4f}")
    print(f"间隔: {2 / np.linalg.norm(clf.coef_):.4f}")

    scores = clf.decision_function(X)
    margins = np.abs(scores)
    print(f"最小间隔样本的 margin: {margins.min():.4f}")

    y_pred = clf.predict(X)
    print(f"训练准确率: {np.mean(y_pred == y):.4f}")

    # 检查是否所有样本都满足间隔约束
    y_bin = np.where(y == clf.classes_[1], 1, -1)
    satisfied = (y_bin * scores >= 1 - 1e-3).mean()
    print(f"满足间隔约束比例: {satisfied:.2%}")
```

### 21.5 常见报错索引

| 报错 | 原因 | 解决 |
|------|------|------|
| coef_ 全 NaN | 学习率过大发散 | 减小 learning_rate |
| 准确率 0.5 | 线性不可分 | 加特征 / 用核 / 换算法 |
| ConvergenceWarning | max_iter 不够 | 增大 max_iter |
| 准确率虚高 | 不平衡数据 | 看 precision/recall |

---

## 二十二、与其他分类器对比

### 22.1 SVM vs LogisticRegression vs RandomForest

```python
import numpy as np
from minisklearn.svm import LinearSVC
from minisklearn.linear_model import LogisticRegression
# from minisklearn.ensemble import RandomForestClassifier

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(100, 2) + [2, 2], rng.randn(100, 2) + [-2, -2]])
y = np.array([0]*100 + [1]*100)

svm = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
lr = LogisticRegression(C=1.0).fit(X, y)
print(f"SVM:  {svm.score(X, y):.3f}")
print(f"LR:   {lr.score(X, y):.3f}")
```

| 维度 | LinearSVC | LogisticRegression | RandomForest |
|------|-----------|-------------------|--------------|
| 损失 | hinge | logistic | 不纯度（Gini） |
| 线性/非线性 | 线性 | 线性 | 非线性 |
| 概率输出 | 否 | 是 | 是 |
| 特征重要性 | $\|w_i\|$ | $\|w_i\|$ | 内置 |
| 大数据 | 快 | 快 | 中 |
| 可解释性 | 中 | 高 | 中 |

### 22.2 LinearSVC vs SVC(kernel='linear')

```python
# sklearn 有两个线性 SVM
# LinearSVC: 优化线性核，用 liblinear，快
# SVC(kernel='linear'): 通用 SVM，用 libsvm，慢但支持核

# 大规模线性数据：用 LinearSVC
# 需要核：用 SVC(kernel='rbf')
# minisklearn 只实现 LinearSVC（梯度下降版）
```

### 22.3 SVM vs 神经网络

```python
# 单层神经网络 + sigmoid 激活 ≈ LogisticRegression
# 单层神经网络 + hinge loss ≈ LinearSVC
# 多层神经网络 = 非线性 SVM 的推广（但用 SGD 而非 QP）

# SVM 优势：凸优化（全局最优）、理论保证、小数据好
# 神经网络优势：大数据、复杂非线性、可扩展
```

---

## 二十三、实际应用场景

### 23.1 文本分类（经典应用）

```python
# 线性 SVM 是文本分类的强 baseline
# TF-IDF + LinearSVC 在 20_newsgroups 上常达 90%+ 准确率

from minisklearn.svm import LinearSVC
from minisklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer

# pipe = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=50000)),
#     ('svm', LinearSVC(C=1.0)),
# ])
# pipe.fit(texts_train, labels_train)
# 线性 SVM 适合高维稀疏特征（文本），比核 SVM 快得多
```

### 23.2 大规模线性分类

```python
import numpy as np
from minisklearn.svm import LinearSVC

# 百万样本、千维特征
rng = np.random.RandomState(0)
X = rng.randn(100000, 100)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

import time
t0 = time.time()
clf = LinearSVC(C=1.0, max_iter=100).fit(X, y)
print(f"10万样本训练: {time.time()-t0:.1f}s, 准确率: {clf.score(X, y):.3f}")
# 线性 SVM 与 n 线性，适合大规模
```

### 23.3 异常检测（One-Class SVM 思路）

```python
# 用 SVM 的决策值做异常评分
import numpy as np
from minisklearn.svm import LinearSVC

rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = (X[:, 0] > 0).astype(int)
clf = LinearSVC().fit(X, y)

# 新点的决策值离 0 越远越确信，接近 0 越不确定
X_new = np.array([[5, 5], [0, 0], [-5, -5]])
scores = clf.decision_function(X_new)
print("决策值:", scores)  # [大正, ~0, 大负]
```

### 23.4 排序（搜索相关性）

```python
# 用 SVM 的决策值做文档排序（而非硬分类）
# decision_function 大 = 更相关
# 这是学习排序（learning to rank）的线性版
scores = clf.decision_function(X_test)
ranked_indices = np.argsort(scores)[::-1]  # 按相关性降序
```

---

## 二十四、思考题与练习

### 基础题

1. **手算**：对 $X = [[1, 1], [-1, -1]]$，$y = [1, -1]$，手算硬间隔 SVM 的 $w, b$ 和间隔。

2. **证明**：证明 hinge loss $\max(0, 1-z)$ 是凸函数但非严格凸。

3. **判断**：以下哪些会改变 SVM 的决策面？(a) 特征标准化 (b) 乘以正常数 (c) 特征正交旋转 (d) 改变 C

### 进阶题

4. **实现**：手写 Pegasos 算法（用 $1/t$ 学习率），对比固定学习率版本的收敛速度。

5. **分析**：为什么 SVM 对离群点比 LogisticRegression 更鲁棒？从损失函数增长速度角度分析。

6. **实验**：对 XOR 数据，分别用 (a) LinearSVC (b) LinearSVC + 二次特征 (c) LinearSVC + RBF 特征近似，对比准确率。

7. **推导**：设数据线性可分，证明硬间隔 SVM 的解是唯一的（提示：凸 QP 的严格凸目标）。

### 思考题

8. SVM 的"支持向量"性质（解只由少数样本决定）带来哪些工程优势？预测时复杂度与什么有关？

9. 核 SVM 把数据隐式映射到高维（甚至无限维）空间，但计算量与原维度相同。这个"免费午餐"的代价是什么？（提示：$O(n^2)$ 核矩阵存储和计算）

10. 深度学习用 SGD + 反向传播优化非凸目标，SVM 用凸优化找全局最优。为什么深度学习不用凸优化？SVM 的凸性在什么假设下成立？

---

## 二十五、扩展阅读

### 书籍

- **《Learning with Kernels》（Schölkopf & Smola）**：核方法的权威著作
- **《Optimization for Machine Learning》（Sra et al.）**：SVM 的优化理论
- **《Understanding Machine Learning》（Shalev-Shwartz & Ben-David）**：SVM 的泛化理论

### 论文

- **"A Tutorial on Support Vector Machines for Pattern Recognition" (Burges, 1998)**：SVM 经典教程
- **"Pegasos: Primal Estimated sub-GrAdient SOlver for SVM" (Shalev-Shwartz et al., 2007)**：Pegasos 算法
- **"Sequential Minimal Optimization" (Platt, 1998)**：SMO 算法原文

### 在线资源

- sklearn SVM 文档：https://scikit-learn.org/stable/modules/svm.html
- "SVM from Scratch" 系列：多种实现的 SVM
- LIBSVM 主页：https://www.csie.ntu.edu.tw/~cjlin/libsvm/

### 相关算法

- `SVC`：核 SVM（支持 RBF、多项式等核）
- `SVR`：支持向量回归
- `OneClassSVM`：异常检测
- `NuSVC`：用 $\nu$ 参数控制支持向量比例
- `LinearSVR`：线性支持向量回归
- `SGDClassifier(loss='hinge')`：随机梯度下降 SVM，超大规模

---

[← 返回算法列表](../index.md)
