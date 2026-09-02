# 高斯朴素贝叶斯（GaussianNB）

> 基于贝叶斯定理 + 条件独立假设，用高斯分布建模每个特征。朴素贝叶斯是机器学习里"最朴素却最有效"的算法——假设天真得离谱，实践中却出奇地好用，尤其在文本分类和高维稀疏数据上。

---

## 一、原理

### 1.1 贝叶斯定理 + 条件独立

给定特征 $x = (x_1, \dots, x_d)$ 和类别 $y$，要预测 $P(y | x)$。贝叶斯定理：

$$
P(y | x) = \frac{P(x | y) P(y)}{P(x)} \propto P(x | y) P(y)
$$

$P(x | y)$ 是似然，$P(y)$ 是先验，$P(x)$ 是证据（归一化常数，与 $y$ 无关，可忽略）。

**条件独立假设**（"朴素"之处）：假设在给定类别 $y$ 后，各特征独立：

$$
P(x | y) = \prod_{j=1}^d P(x_j | y)
$$

代入：

$$
P(y | x) \propto P(y) \prod_j P(x_j | y)
$$

取对数（避免下溢，且加法比乘法快）：

$$
\hat{y} = \arg\max_y \left[ \log P(y) + \sum_j \log P(x_j | y) \right]
$$

### 1.2 高斯假设

对连续特征，假设 $P(x_j | y = c)$ 服从高斯分布：

$$
P(x_j | y=c) = \mathcal{N}(x_j | \mu_{cj}, \sigma^2_{cj}) = \frac{1}{\sqrt{2\pi \sigma^2_{cj}}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2 \sigma^2_{cj}} \right)
$$

fit 阶段对每个类别 $c$、每个特征 $j$ 计算 $\mu_{cj}$ 和 $\sigma^2_{cj}$：

$$
\mu_{cj} = \frac{1}{n_c} \sum_{i: y_i = c} x_{ij}, \quad \sigma^2_{cj} = \frac{1}{n_c} \sum_{i: y_i = c} (x_{ij} - \mu_{cj})^2
$$

其中 $n_c$ 是类别 $c$ 的样本数。先验 $P(y = c) = n_c / n$。

predict 阶段对每个样本算各类别的对数后验：

$$
\log P(y = c | x) \propto \log P(y=c) + \sum_j \left[ -\frac{1}{2}\log(2\pi \sigma^2_{cj}) - \frac{(x_j - \mu_{cj})^2}{2 \sigma^2_{cj}} \right]
$$

取最大的 $c$ 作为预测。

### 1.3 为什么"朴素"？

假设特征条件独立（$P(x|y) = \prod P(x_j|y)$），现实中几乎不成立，但实践效果出奇地好。原因：

1. **分类只需排序，不需精确概率**：即使独立假设使概率估计偏差大，只要 argmax 排序正确，分类就正确。
2. **偏差-方差权衡**：强独立假设是高偏差，但参数少（每类每特征两个参数）是低方差。高维数据下方差是主要矛盾，朴素贝叶斯用偏差换方差，反而占优。
3. **乘法变加法**：独立假设让似然分解为各特征贡献之和，可解释性强（每个特征独立贡献）。
4. **抗过拟合**：参数极少，几乎不过拟合。

### 1.4 几何直觉

高斯朴素贝叶斯的决策边界是**二次的**。展开对数后验：

$$
\log P(y=c|x) \propto \log P(c) - \frac{1}{2}\sum_j \left[ \log \sigma^2_{cj} + \frac{(x_j - \mu_{cj})^2}{\sigma^2_{cj}} \right] + \text{const}
$$

展开 $(x_j - \mu_{cj})^2 / \sigma^2_{cj} = x_j^2/\sigma^2_{cj} - 2\mu_{cj} x_j/\sigma^2_{cj} + \mu_{cj}^2/\sigma^2_{cj}$，包含 $x_j^2$ 项，所以决策边界是二次曲面。

特例：当所有类别各特征方差相同（$\sigma^2_{cj} = \sigma^2_j$，与 $c$ 无关）时，$x_j^2$ 项在比较中消去，决策边界退化为**线性**。这叫"同方差假设下的线性朴素贝叶斯"。

可视化：在二维平面上，高斯朴素贝叶斯用椭圆等高线建模每类数据（椭圆轴与坐标轴对齐，因为独立假设）。决策边界是两族椭圆的等高线交点，是双曲线或抛物线。

### 1.5 与 LDA/QDA 的关系

- **LDA（线性判别分析）**：假设每类高斯，且所有类共享协方差矩阵 $\Sigma$（不限于对角），决策边界线性。
- **QDA（二次判别分析）**：每类有自己的协方差矩阵 $\Sigma_c$，决策边界二次。
- **高斯朴素贝叶斯**：QDA 的特例，每类协方差矩阵是对角阵 $\mathrm{diag}(\sigma^2_{c1}, \dots, \sigma^2_{cd})$。

所以高斯朴素贝叶斯是 QDA 的简化（对角协方差），QDA 是 LDA 的推广（异协方差）。

---

## 二、数学推导详解

### 2.1 从贝叶斯定理到对数后验

完整推导链：

$$
\hat{y} = \arg\max_y P(y | x) = \arg\max_y \frac{P(x|y) P(y)}{P(x)}
$$

$P(x)$ 与 $y$ 无关：

$$
= \arg\max_y P(x|y) P(y)
$$

独立假设：

$$
= \arg\max_y P(y) \prod_j P(x_j | y)
$$

取对数（单调变换不改变 argmax）：

$$
= \arg\max_y \left[ \log P(y) + \sum_j \log P(x_j | y) \right]
$$

代入高斯：

$$
\log P(x_j | y=c) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log \sigma^2_{cj} - \frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}
$$

$-\frac{1}{2}\log(2\pi)$ 与 $c$ 无关，可忽略：

$$
\log P(y=c | x) \propto \log P(c) - \frac{1}{2}\sum_j \left[ \log \sigma^2_{cj} + \frac{(x_j - \mu_{cj})^2}{\sigma^2_{cj}} \right]
$$

### 2.2 参数估计的 MLE

对类别 $c$ 的样本 $\{x_i : y_i = c\}$，高斯分布的 MLE：

$$
\hat{\mu}_{cj} = \frac{1}{n_c} \sum_{i: y_i = c} x_{ij}
$$

$$
\hat{\sigma}^2_{cj} = \frac{1}{n_c} \sum_{i: y_i = c} (x_{ij} - \hat{\mu}_{cj})^2
$$

这是有偏估计（除以 $n_c$ 而非 $n_c - 1$）。sklearn 默认用 MLE（`var_smoothing` 加在方差上防止除零）。

先验的 MLE：

$$
\hat{P}(y = c) = \frac{n_c}{n}
$$

### 2.3 var_smoothing 的作用

方差可能为 0（某特征在某类是常数），导致除零。sklearn 加一个小的平滑项：

$$
\sigma^2_{cj} \leftarrow \sigma^2_{cj} + \epsilon \cdot \max_j \sigma^2_{cj}
$$

其中 $\epsilon = 10^{-9}$（默认 `var_smoothing`）。这保证方差非零，且对大方差特征影响极小。

### 2.4 预测的概率校准

朴素贝叶斯的概率输出通常**过度自信**（接近 0 或 1），因为独立假设使似然被多次相乘，概率被"过度集中"。如果要可靠概率，应该用 `CalibratedClassifierCV` 做 isotonic 回归校准。

但 argmax 分类不受影响，所以分类任务可以直接用。

---

## 三、实现细节

### 3.1 完整实现

```python
import numpy as np
from ..base import BaseEstimator, ClassifierMixin

class GaussianNB(ClassifierMixin, BaseEstimator):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        self.class_count_ = np.zeros(len(self.classes_))
        self.class_prior_ = np.zeros(len(self.classes_))
        self.theta_ = np.zeros((len(self.classes_), n_features))      # 均值
        self.var_ = np.zeros((len(self.classes_), n_features))        # 方差

        epsilon = self.var_smoothing * X.var(axis=0).max()             # 平滑项

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.class_count_[idx] = len(X_c)
            self.class_prior_[idx] = len(X_c) / n_samples
            self.theta_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0) + epsilon                 # 加平滑

        return self

    def _joint_log_likelihood(self, X):
        """计算每个样本对每个类别的对数后验（未归一化）。"""
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[idx])
            # log N(x | mu, sigma^2) = -0.5*log(2*pi*sigma^2) - (x-mu)^2/(2*sigma^2)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var_[idx]))
            log_likelihood -= 0.5 * np.sum((X - self.theta_[idx]) ** 2 / self.var_[idx], axis=1)
            jll[:, idx] = log_prior + log_likelihood
        return jll

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        # log-sum-exp 归一化，避免溢出
        log_norm = np.log(np.sum(np.exp(jll - jll.max(axis=1, keepdims=True)), axis=1)) + jll.max(axis=1)
        return np.exp(jll - log_norm[:, None])

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        log_norm = np.log(np.sum(np.exp(jll - jll.max(axis=1, keepdims=True)), axis=1)) + jll.max(axis=1)
        return jll - log_norm[:, None]
```

### 3.2 向量化优化

`_joint_log_likelihood` 可以完全向量化，避免类别循环：

```python
def _joint_log_likelihood_vectorized(self, X):
    # theta_: (C, d), var_: (C, d), class_prior_: (C,)
    # X: (n, d)
    n, d = X.shape
    C = len(self.classes_)

    # (n, C, d) = (n, 1, d) - (1, C, d)
    diff = X[:, None, :] - self.theta_[None, :, :]

    # (n, C) = sum over d of (diff^2 / var)
    quad = np.sum(diff ** 2 / self.var_[None, :, :], axis=2)           # (n, C)

    # (C,) 常数项
    log_det = np.sum(np.log(2 * np.pi * self.var_), axis=1)            # (C,)

    jll = np.log(self.class_prior_)[None, :] - 0.5 * log_det[None, :] - 0.5 * quad
    return jll
```

但向量化会创建 $(n, C, d)$ 的中间数组，内存 $O(nCd)$。类别少时用向量化，类别多时用循环。

### 3.3 log-sum-exp 技巧

`predict_proba` 要算 $\dfrac{e^{l_c}}{\sum_{c'} e^{l_{c'}}}$，直接算会溢出（$l_c$ 可能很大或很小）。log-sum-exp 技巧：

$$
\log \sum_c e^{l_c} = m + \log \sum_c e^{l_c - m}, \quad m = \max_c l_c
$$

减去最大值后 $e^{l_c - m} \leq 1$，不会溢出。这是数值稳定的关键。

```python
# 不稳定
proba = np.exp(jll) / np.exp(jll).sum(axis=1, keepdims=True)  # 可能溢出

# 稳定
m = jll.max(axis=1, keepdims=True)
log_norm = m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True))
proba = np.exp(jll - log_norm)
```

### 3.4 在线更新

朴素贝叶斯支持 `partial_fit`，用 Welford 算法在线更新均值和方差：

```python
def partial_fit(self, X, y, classes=None):
    if not hasattr(self, 'classes_'):
        self.classes_ = classes
    # 用 Welford 更新 theta_, var_, class_count_
    # ...
```

这适合流式数据（数据太大放不进内存）。

---

## 四、使用示例

### 4.1 基础分类

```python
import numpy as np
from minisklearn.naive_bayes import GaussianNB

rng = np.random.RandomState(42)
X = np.vstack([rng.randn(100, 2) + [2, 2],
               rng.randn(100, 2) + [-2, -2]])
y = np.array([0] * 100 + [1] * 100)

clf = GaussianNB().fit(X, y)
print(clf.predict(X[:5]))          # [0 0 0 0 0]
print(clf.predict_proba(X[:5]))    # 概率
print(clf.score(X, y))             # 准确率
```

### 4.2 查看学到的参数

```python
clf = GaussianNB().fit(X, y)
print("类别先验:", clf.class_prior_)      # [0.5, 0.5]
print("各类均值:", clf.theta_)            # [[2, 2], [-2, -2]]
print("各类方差:", clf.var_)              # [[1, 1], [1, 1]]
```

### 4.3 多分类

```python
X = np.vstack([rng.randn(50, 2) + [3, 0],
               rng.randn(50, 2) + [-3, 0],
               rng.randn(50, 2) + [0, 3]])
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

clf = GaussianNB().fit(X, y)
print(clf.predict(X[:3]))           # [0 0 0]
print(clf.predict(X[50:53]))        # [1 1 1]
print(clf.predict(X[100:103]))      # [2 2 2]
```

### 4.4 概率输出

```python
proba = clf.predict_proba(X)
print(proba[:5])
# [[0.99 0.01 0.  ]
#  [0.95 0.04 0.01]
#  ...
# ]
print(proba.sum(axis=1))            # [1. 1. 1. ...] 每行和为 1
```

### 4.5 完整可运行示例

```python
import numpy as np
from minisklearn.naive_bayes import GaussianNB
from minisklearn.model_selection import train_test_split, cross_val_score
from minisklearn.datasets import load_iris

# 1. 加载数据
X, y = load_iris(return_X_y=True)
print(f"数据形状: {X.shape}, 类别: {np.unique(y)}")

# 2. 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 训练
clf = GaussianNB().fit(X_train, y_train)

# 4. 评估
print(f"训练准确率: {clf.score(X_train, y_train):.4f}")
print(f"测试准确率: {clf.score(X_test, y_test):.4f}")

# 5. 交叉验证
scores = cross_val_score(GaussianNB(), X, y, cv=5)
print(f"CV 准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. 查看参数
print("各类均值:")
print(clf.theta_)
print("各类方差:")
print(clf.var_)
```

### 4.6 错误示例

```python
# 错误 1：特征有零方差
X = np.array([[1, 2], [1, 3], [1, 4]])  # 特征 1 恒为 1
y = np.array([0, 0, 1])
clf = GaussianNB().fit(X, y)
# 没 var_smoothing 会除零；有 smoothing 则正常
print(clf.var_)  # 第一列是 epsilon，非零

# 错误 2：predict 前没 fit
clf = GaussianNB()
# clf.predict(X)  # 报错：AttributeError: 'GaussianNB' has no attribute 'theta_'

# 错误 3：测试时遇到训练时没见过的类别
# GaussianNB 假设类别在 fit 时都见过，predict 时新类别无法处理
# （但 predict 的 X 可以有新特征值，只是类别集合固定）
```

### 4.7 对比示例：独立 vs 相关特征

```python
# 独立特征：朴素贝叶斯准
X_indep = rng.randn(200, 2)
y_indep = (X_indep[:, 0] + X_indep[:, 1] > 0).astype(int)
print("独立特征:", GaussianNB().fit(X_indep, y_indep).score(X_indep, y_indep))

# 强相关特征：朴素贝叶斯仍可用但概率失真
X_corr = np.column_stack([X_indep[:, 0], X_indep[:, 0] + 0.01 * rng.randn(200)])
y_corr = (X_corr[:, 0] + X_corr[:, 1] > 0).astype(int)
print("相关特征:", GaussianNB().fit(X_corr, y_corr).score(X_corr, y_corr))
# 分类可能仍准，但 predict_proba 会过度自信
```

---

## 五、与 sklearn 对比

### 5.1 API 一致性

| 特性 | minisklearn GaussianNB | sklearn GaussianNB |
|------|----------------------|-------------------|
| `var_smoothing` | ✓ | ✓ |
| `fit` / `partial_fit` | ✓（fit） | ✓ |
| `predict` / `predict_proba` | ✓ | ✓ |
| `predict_log_proba` | ✓ | ✓ |
| `score` | ✓ | ✓ |
| `theta_`（均值） | ✓ | ✓ |
| `var_`（方差） | ✓ | ✓ |
| `class_prior_` | ✓ | ✓ |
| `class_count_` | ✓ | ✓ |
| `epsilon_` | ✗ | ✓ |

### 5.2 数值结果对比

```python
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from minisklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
clf_mini = GaussianNB().fit(X, y)
clf_sk = SkGaussianNB().fit(X, y)

print("均值差异:", np.abs(clf_mini.theta_ - clf_sk.theta_).max())   # ~0
print("方差差异:", np.abs(clf_mini.var_ - clf_sk.var_).max())       # ~0
print("预测一致:", np.all(clf_mini.predict(X) == clf_sk.predict(X))) # True
```

### 5.3 其他朴素贝叶斯变体

sklearn 还有：
- **MultinomialNB**：多项式分布，文本分类（词频特征）
- **BernoulliNB**：伯努利分布，二值特征
- **ComplementNB**：补集朴素贝叶斯，对不平衡数据更好
- **CategoricalNB**：类别分布，离散特征

minisklearn 只实现 GaussianNB（连续特征）。

```python
# sklearn 文本分类
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
pipe = Pipeline([('vec', CountVectorizer()), ('clf', MultinomialNB())])
pipe.fit(texts, labels)
```

---

## 六、复杂度分析

### 6.1 训练复杂度

| 步骤 | 时间 | 空间 |
|------|------|------|
| 按类分组 | $O(n)$ | $O(n)$ |
| 计算均值方差 | $O(nd)$ | $O(Cd)$ |
| 总计 | $O(nd)$ | $O(Cd)$ |

$C$ 类别数，$n$ 样本数，$d$ 特征数。朴素贝叶斯训练是**线性**的，极快。

### 6.2 预测复杂度

| 步骤 | 时间 |
|------|------|
| 算对数后验 | $O(nCd)$ |
| argmax | $O(nC)$ |
| 总计 | $O(nCd)$ |

预测也是线性的。

### 6.3 与其他分类器对比

| 分类器 | 训练 | 预测 |
|--------|------|------|
| GaussianNB | $O(nd)$ | $O(nCd)$ |
| LogisticRegression | $O(Tnd)$ | $O(nd)$ |
| LinearSVC | $O(Tnd)$ | $O(nd)$ |
| KNN | $O(1)$（惰性） | $O(nd)$（暴力） |

朴素贝叶斯训练最快（无迭代），预测也快。

### 6.4 实测

```python
import numpy as np, time
from minisklearn.naive_bayes import GaussianNB

for n, d in [(1000, 10), (10000, 100), (100000, 50)]:
    X = np.random.randn(n, d)
    y = (X[:, 0] > 0).astype(int)
    t0 = time.time()
    GaussianNB().fit(X, y)
    print(f"n={n}, d={d}: {time.time()-t0:.3f}s")
```

---

## 七、数值稳定性

### 7.1 var_smoothing

方差为 0 时除零。`var_smoothing` 加一个相对最大方差的 epsilon：

```python
epsilon = var_smoothing * X.var(axis=0).max()  # 1e-9 * max_var
var_ += epsilon
```

这保证方差非零，且对大方差特征影响极小（相对误差 $10^{-9}$）。

### 7.2 log-sum-exp

`predict_proba` 用 log-sum-exp 避免溢出（见 3.3 节）。

### 7.3 下溢

直接算 $\prod_j P(x_j | y)$ 会下溢（很多小数相乘趋于 0）。取对数变加法：

$$
\log \prod_j P(x_j | y) = \sum_j \log P(x_j | y)
$$

这是为什么实现里全程用对数。

### 7.4 大特征值

$(x_j - \mu_{cj})^2 / \sigma^2_{cj}$ 当 $x_j$ 远离 $\mu_{cj}$ 时很大，对数后验很负。这是正常的（远离该类中心，后验小）。但如果方差极小，这个项会爆炸，导致该类概率下溢到 0。var_smoothing 缓解这个问题。

---

## 八、常见问题与陷阱

### 8.1 独立假设失效

特征强相关时，朴素贝叶斯概率失真（过度自信）。但分类可能仍准（argmax 鲁棒）。

```python
# 完美相关特征
X = np.column_stack([rng.randn(100), rng.randn(100)])  # 独立
X_corr = np.column_stack([X[:, 0], X[:, 0]])            # 完全相关
# 后者方差估计会很小，概率过度自信
```

**解决**：特征选择去相关，或用 QDA/LDA（建模协方差）。

### 8.2 零方差特征

某特征在某类是常数，方差为 0。var_smoothing 处理，但若该特征对分类重要，平滑会引入偏差。

### 8.3 离群点

高斯分布对离群点敏感（一个离群点大幅拉偏均值和方差）。Robust 变体用中位数和 MAD 代替。

### 8.4 不平衡数据

先验 $P(y=c) = n_c/n$ 反映不平衡。如果不想让先验影响预测，设 `class_prior=` 为均匀。

```python
clf = GaussianNB(priors=[0.5, 0.5]).fit(X, y)  # sklearn 支持
```

### 8.5 概率未校准

朴素贝叶斯概率通常过度自信（接近 0/1）。要可靠概率需校准：

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(GaussianNB(), cv=5).fit(X, y)
```

### 8.6 连续特征的非高斯分布

如果特征明显非高斯（如重尾、多峰），高斯假设不准。可以：
- 变换特征（log、Box-Cox）使其接近高斯
- 用 KernelDensity 估计非参数分布
- 离散化后用 MultinomialNB/CategoricalNB

### 8.7 测试时遇到训练集未见的类别

GaussianNB 的类别在 fit 时固定。如果测试数据有新类别标签，predict 会出错（但通常测试数据只有 X 没有 y，所以不涉及）。

---

## 九、实际使用教程

### 9.1 标准分类流程

```python
from minisklearn.naive_bayes import GaussianNB
from minisklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = GaussianNB().fit(X_train, y_train)
print(f"测试准确率: {clf.score(X_test, y_test):.4f}")

scores = cross_val_score(GaussianNB(), X, y, cv=5)
print(f"CV: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 9.2 基线模型

朴素贝叶斯常作基线：训练快、参数少、效果不差。先跑朴素贝叶斯，再试更复杂模型，看提升是否值得。

```python
# 基线
nb_score = cross_val_score(GaussianNB(), X, y, cv=5).mean()
# 复杂模型
lr_score = cross_val_score(LogisticRegression(), X, y, cv=5).mean()
print(f"NB:  {nb_score:.4f}")
print(f"LR:  {lr_score:.4f}")
# 如果 LR 只高 1%，NB 的简单性可能更值得
```

### 9.3 文本分类（用 sklearn 的 MultinomialNB）

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
pipe.fit(texts_train, labels_train)
pred = pipe.predict(texts_test)
```

### 9.4 增量学习

```python
# sklearn
clf = GaussianNB()
for batch in batches:
    clf.partial_fit(batch_X, batch_y, classes=np.unique(y_all))
```

### 9.5 可视化决策边界

```python
import matplotlib.pyplot as plt

clf = GaussianNB().fit(X, y)
xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100),
                     np.linspace(X[:,1].min(), X[:,1].max(), 100))
Z = clf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```

---

## 十、变体与扩展

### 10.1 MultinomialNB（多项式朴素贝叶斯）

对计数特征（如词频）：

$$
P(x_j | y=c) = \frac{x_j + \alpha}{N_c + \alpha d}
$$

$\alpha$ 是拉普拉斯平滑（$\alpha=1$ 是加一平滑）。适合文本分类。

### 10.2 BernoulliNB（伯努利朴素贝叶斯）

对二值特征（词出现/不出现）：

$$
P(x_j | y=c) = p_{cj}^{x_j} (1 - p_{cj})^{1 - x_j}
$$

### 10.3 ComplementNB

用补集类建模，对不平衡数据更好。sklearn 实现。

### 10.4 半监督朴素贝叶斯

用 EM 在未标注数据上迭代：E 步用当前模型预测伪标签，M 步用伪标签更新模型。

### 10.5 核朴素贝叶斯

用核密度估计替代高斯假设，处理非高斯特征。

---

## 十一、架构回扣

### 11.1 ClassifierMixin

GaussianNB 继承 `ClassifierMixin`，自动获得 `score`（准确率）。

### 11.2 双下划线属性

`theta_`、`var_`、`class_prior_`、`class_count_` 是 fit 后学出的参数。`var_smoothing` 是 `__init__` 超参数。

### 11.3 在 Pipeline 中

```python
pipe = Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())])
# 注意：标准化对 GaussianNB 不必要（高斯假设对线性变换不变）
# 但若特征非高斯，标准化后可能更接近高斯
```

### 11.4 与 GridSearchCV

```python
grid = GridSearchCV(GaussianNB(), {'var_smoothing': [1e-9, 1e-7, 1e-5]}, cv=5)
grid.fit(X, y)
```

### 11.5 partial_fit 的在线学习接口

`partial_fit` 让 GaussianNB 支持流式数据，这是 sklearn 在线学习 API 的一部分。minisklearn 可选实现。

---

## 十二、进阶话题

### 12.1 朴素贝叶斯的误差分析

朴素贝叶斯的误差来自两部分：
1. **偏差**：独立假设使模型类受限
2. **方差**：参数估计的随机性

在高维低样本场景，方差主导，朴素贝叶斯的强假设反而降低方差，表现好。这是"偏差-方差权衡"的经典案例。

### 12.2 与 LogisticRegression 的关系

在特定条件下（两类、同方差高斯），朴素贝叶斯的决策边界与 LogisticRegression 相同。但朴素贝叶斯假设特征条件独立，LR 不假设，所以 LR 通常更准（但训练慢）。

### 12.3 朴素贝叶斯的最大熵解释

可以证明：在给定特征边缘分布和类别-特征联合分布的约束下，朴素贝叶斯分布是最大熵分布。这给了"朴素"假设一个信息论解释。

### 12.4 概率校准

朴素贝叶斯概率过度自信的原因：独立假设使似然被多次相乘，即使每个 $P(x_j|y)$ 偏差不大，乘起来偏差放大。校准方法：
- **Platt scaling**：用 LR 拟合 $\Pr[y=1 | f(x)] = \sigma(A f(x) + B)$
- **Isotonic 回归**：非参数单调校准

### 12.5 AODE（Averaged One-Dependence Estimators）

放宽独立假设到"一依赖"：假设所有特征依赖一个公共特征，对所有可能公共特征平均。比朴素贝叶斯准，仍快。

---

## 十三、更多数学推导

### 13.1 决策边界的二次性

展开两类（$c=0, 1$）的对数后验差：

$$
\log \frac{P(y=1|x)}{P(y=0|x)} = \log \frac{P(y=1)}{P(y=0)} + \sum_j \log \frac{P(x_j|y=1)}{P(x_j|y=0)}
$$

代入高斯：

$$
= \log \frac{P(y=1)}{P(y=0)} + \sum_j \left[ -\frac{1}{2}\log\frac{\sigma^2_{1j}}{\sigma^2_{0j}} - \frac{(x_j - \mu_{1j})^2}{2\sigma^2_{1j}} + \frac{(x_j - \mu_{0j})^2}{2\sigma^2_{0j}} \right]
$$

展开平方项：

$$
\frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}} = \frac{x_j^2}{2\sigma^2_{cj}} - \frac{\mu_{cj} x_j}{\sigma^2_{cj}} + \frac{\mu_{cj}^2}{2\sigma^2_{cj}}
$$

合并 $x_j^2$ 项：

$$
\sum_j \frac{x_j^2}{2} \left( \frac{1}{\sigma^2_{0j}} - \frac{1}{\sigma^2_{1j}} \right)
$$

如果 $\sigma^2_{0j} \neq \sigma^2_{1j}$，这是 $x_j^2$ 项，决策边界是二次的。如果 $\sigma^2_{0j} = \sigma^2_{1j}$（同方差），$x_j^2$ 项消去，决策边界线性。

### 13.2 同方差下的线性形式

设 $\sigma^2_{0j} = \sigma^2_{1j} = \sigma^2_j$，对数后验差：

$$
\log \frac{P(y=1|x)}{P(y=0|x)} = \log \frac{P(y=1)}{P(y=0)} + \sum_j \left[ \frac{\mu_{1j} - \mu_{0j}}{\sigma^2_j} x_j - \frac{\mu_{1j}^2 - \mu_{0j}^2}{2\sigma^2_j} \right]
$$

这是 $w \cdot x + b$ 的形式，其中：

$$
w_j = \frac{\mu_{1j} - \mu_{0j}}{\sigma^2_j}, \quad b = \log \frac{P(y=1)}{P(y=0)} - \sum_j \frac{\mu_{1j}^2 - \mu_{0j}^2}{2\sigma^2_j}
$$

这正是 LDA 的形式（LDA 还允许 $\sigma^2_j$ 之间相关，即非对角协方差）。

### 13.3 期望风险

朴素贝叶斯的期望风险（0-1 loss）：

$$
R = \mathbb{E}[\mathbb{1}[\hat{y}(X) \neq Y]]
$$

在独立假设下可以分解分析。当假设正确时（数据真的条件独立），朴素贝叶斯是贝叶斯最优分类器。假设错误时，风险增加，但增加量有界（取决于特征间的依赖强度）。

### 13.4 朴素贝叶斯的可分解性

对数后验 $\log P(y=c|x) = \log P(c) + \sum_j \log P(x_j|c)$ 是各特征贡献之和。这意味着：
- 可以算每个特征对分类的贡献（可解释性）
- 可以处理缺失特征（缺的特征不贡献，其余照算）
- 可以增量添加特征（无需重训，只算新特征的 $\mu, \sigma^2$）

---

## 十四、更多代码示例

### 14.1 手动实现（教学版）

```python
import numpy as np

class GaussianNBScratch:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(y)
            self.means_[c] = Xc.mean(axis=0)
            self.vars_[c] = Xc.var(axis=0) + 1e-9
        return self

    def predict(self, X):
        preds = []
        for x in X:
            best_c, best_ll = None, -np.inf
            for c in self.classes_:
                ll = np.log(self.priors_[c])
                ll += -0.5 * np.sum(np.log(2 * np.pi * self.vars_[c]))
                ll += -0.5 * np.sum((x - self.means_[c]) ** 2 / self.vars_[c])
                if ll > best_ll:
                    best_c, best_ll = c, ll
            preds.append(best_c)
        return np.array(preds)
```

### 14.2 特征贡献分析

```python
def feature_contribution(clf, x, cls):
    """计算每个特征对类别 cls 的对数后验贡献。"""
    contributions = []
    for j in range(len(x)):
        log_p = -0.5 * np.log(2 * np.pi * clf.var_[cls, j])
        log_p -= 0.5 * (x[j] - clf.theta_[cls, j]) ** 2 / clf.var_[cls, j]
        contributions.append(log_p)
    return np.array(contributions)

clf = GaussianNB().fit(X, y)
contrib = feature_contribution(clf, X[0], 0)
print("各特征对类别 0 的贡献:", contrib)
print("总贡献:", contrib.sum() + np.log(clf.class_prior_[0]))
```

### 14.3 处理缺失值

```python
def predict_with_missing(clf, x, missing_mask):
    """x 中 missing_mask=True 的特征缺失，跳过其贡献。"""
    best_c, best_ll = None, -np.inf
    for idx, c in enumerate(clf.classes_):
        ll = np.log(clf.class_prior_[idx])
        for j in range(len(x)):
            if missing_mask[j]:
                continue  # 缺失特征不贡献
            ll += -0.5 * np.log(2 * np.pi * clf.var_[idx, j])
            ll -= 0.5 * (x[j] - clf.theta_[idx, j]) ** 2 / clf.var_[idx, j]
        if ll > best_ll:
            best_c, best_ll = c, ll
    return best_c
```

### 14.4 与其他分类器对比

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.svm import LinearSVC

for name, clf in [('NB', GaussianNB()),
                  ('LR', LogisticRegression()),
                  ('SVM', LinearSVC())]:
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 14.5 校准概率（用 sklearn）

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

calibrated = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
proba = calibrated.predict_proba(X_test)
# 校准后的概率更可靠（Brier score 更低）
```

### 14.6 在线学习示例

```python
# sklearn 的 partial_fit
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
classes = np.unique(y_all)

# 模拟流式数据
for batch_X, batch_y in stream_batches:
    clf.partial_fit(batch_X, batch_y, classes=classes)
    if batch_idx % 100 == 0:
        print(f"批次 {batch_idx}: {clf.score(X_test, y_test):.4f}")
```

---

## 十五、应用场景详解

### 15.1 垃圾邮件分类

经典应用：邮件特征（词频、是否含特定词等），分类垃圾/正常。用 MultinomialNB 或 BernoulliNB。

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = ["免费赚钱", "明天开会", "中奖了点击领取", "项目报告"]
labels = [1, 0, 1, 0]  # 1=垃圾

vec = CountVectorizer()
X = vec.fit_transform(emails)
clf = MultinomialNB().fit(X, labels)
print(clf.predict(vec.transform(["免费中奖"])))  # [1]
```

### 15.2 情感分析

```python
texts = ["这部电影太棒了", "垃圾电影", "推荐", "难看"]
labels = [1, 0, 1, 0]  # 1=正面
# 同上，用 MultinomialNB + TfidfVectorizer
```

### 15.3 医疗诊断

连续特征（血压、血糖等），用 GaussianNB：

```python
# 特征：血压、血糖、年龄、BMI
X = np.array([[120, 90, 45, 25], [140, 110, 60, 30], ...])
y = np.array([0, 1, ...])  # 0=健康, 1=患病
clf = GaussianNB().fit(X, y)
```

### 15.4 实时推荐

朴素贝叶斯快，适合实时场景：

```python
# 用户特征 → 推荐类别
clf = GaussianNB().fit(user_features, click_labels)
# 实时预测
recommendation = clf.predict(new_user_features)
```

---

## 十六、总结

| 要点 | 内容 |
|------|------|
| 核心思想 | 贝叶斯定理 + 条件独立假设 |
| 数学基础 | 高斯似然 + 对数后验 argmax |
| 实现 | 按类算均值方差，对数后验预测 |
| 复杂度 | 训练 $O(nd)$，预测 $O(nCd)$，极快 |
| 数值稳定 | var_smoothing + log-sum-exp |
| 常见陷阱 | 独立假设失效、概率未校准、零方差 |
| 与 sklearn | API 一致，sklearn 有更多变体 |
| 适用场景 | 基线模型、高维稀疏、文本分类、在线学习 |
| 不适用 | 强相关特征、需精确概率、非高斯特征 |
| 变体 | MultinomialNB、BernoulliNB、ComplementNB |
| 决策边界 | 一般二次，同方差时退化为线性 |
| 可解释性 | 各特征独立贡献，可分析、可处理缺失值 |

---

## 十七、深入数学推导与证明

### 17.1 贝叶斯定理的完整推导

**贝叶斯定理**：
$$
P(y | x) = \frac{P(x | y) P(y)}{P(x)}
$$

**证明**：由条件概率定义：
$$
P(y | x) = \frac{P(x, y)}{P(x)}, \quad P(x | y) = \frac{P(x, y)}{P(y)}
$$

故 $P(x, y) = P(y | x) P(x) = P(x | y) P(y)$，即 $P(y | x) = \frac{P(x | y) P(y)}{P(x)}$。$\square$

### 17.2 条件独立假设的数学表述

**定义**：特征 $x_1, \ldots, x_d$ 在给定 $y$ 时条件独立，若：
$$
P(x_1, \ldots, x_d | y) = \prod_{j=1}^d P(x_j | y)
$$

等价于：对任意 $j$，$P(x_j | y, x_{-j}) = P(x_j | y)$（给定 $y$ 后，其他特征不提供关于 $x_j$ 的额外信息）。

### 17.3 对数后验的完整推导

**目标**：$\hat{y} = \arg\max_y P(y | x)$。

**推导链**：

1. 贝叶斯定理：$P(y | x) = \frac{P(x | y) P(y)}{P(x)}$
2. $P(x)$ 与 $y$ 无关：$\arg\max_y P(y | x) = \arg\max_y P(x | y) P(y)$
3. 条件独立：$P(x | y) = \prod_j P(x_j | y)$
4. 取对数（单调）：$\arg\max_y \left[ \log P(y) + \sum_j \log P(x_j | y) \right]$
5. 高斯似然：$\log P(x_j | y=c) = -\frac{1}{2}\log(2\pi \sigma^2_{cj}) - \frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}$
6. 忽略与 $c$ 无关的常数项：

$$
\log P(y=c | x) \propto \log P(c) - \frac{1}{2}\sum_j \left[ \log \sigma^2_{cj} + \frac{(x_j - \mu_{cj})^2}{\sigma^2_{cj}} \right] \quad \square
$$

### 17.4 决策边界二次性的证明

**定理**：高斯朴素贝叶斯的决策边界是二次曲面。当各类方差相同时退化为线性。

**证明**：两类 $c=0, 1$ 的决策边界为 $\log \frac{P(y=1|x)}{P(y=0|x)} = 0$。

$$
\log \frac{P(y=1|x)}{P(y=0|x)} = \log \frac{P(y=1)}{P(y=0)} + \sum_j \left[ -\frac{1}{2}\log\frac{\sigma^2_{1j}}{\sigma^2_{0j}} - \frac{(x_j - \mu_{1j})^2}{2\sigma^2_{1j}} + \frac{(x_j - \mu_{0j})^2}{2\sigma^2_{0j}} \right]
$$

展开平方项，$x_j^2$ 的系数为 $\frac{1}{2}\left(\frac{1}{\sigma^2_{0j}} - \frac{1}{\sigma^2_{1j}}\right)$。

- 若 $\sigma^2_{0j} \neq \sigma^2_{1j}$：有 $x_j^2$ 项，边界二次。
- 若 $\sigma^2_{0j} = \sigma^2_{1j}$：$x_j^2$ 项消去，边界线性。$\square$

### 17.5 MLE 估计的推导

**定理**：高斯分布参数的 MLE 为 $\hat{\mu} = \bar{x}$，$\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$。

**证明**：对数似然：
$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2
$$

对 $\mu$ 求导令零：
$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \Rightarrow \hat{\mu} = \bar{x}
$$

对 $\sigma^2$ 求导令零：
$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_i (x_i - \mu)^2 = 0 \Rightarrow \hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2 \quad \square
$$

**注意**：这是有偏估计（除以 $n$ 而非 $n-1$），$E[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$。

### 17.6 朴素贝叶斯与逻辑回归的关系

**定理**：在两类、同方差高斯假设下，朴素贝叶斯的决策边界与逻辑回归相同。

**证明**：由 17.4，同方差时决策边界为 $w \cdot x + b = 0$，其中：
$$
w_j = \frac{\mu_{1j} - \mu_{0j}}{\sigma^2_j}, \quad b = \log\frac{P(y=1)}{P(y=0)} - \sum_j \frac{\mu_{1j}^2 - \mu_{0j}^2}{2\sigma^2_j}
$$

这正是逻辑回归的形式。但朴素贝叶斯假设特征条件独立，逻辑回归不假设，故逻辑回归更一般（可处理相关特征）。$\square$

### 17.7 朴素贝叶斯的最大熵解释

**定理**：在给定特征边缘分布 $\{P(x_j)\}$ 和类别-特征联合分布 $\{P(y, x_j)\}$ 的约束下，朴素贝叶斯分布 $P(y) \prod_j P(x_j | y)$ 是最大熵分布。

**含义**：在信息不足时，朴素贝叶斯做最"保守"的假设（最大熵），不引入额外依赖。这给了"朴素"假设一个信息论解释。

---

## 十八、更多代码示例与对比实验

### 18.1 独立 vs 相关特征对比

```python
import numpy as np
from minisklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

np.random.seed(0)
n = 500

# 独立特征：朴素贝叶斯假设成立
X_indep = np.random.randn(n, 5)
y_indep = (X_indep.sum(axis=1) > 0).astype(int)

# 强相关特征：假设违反
X_corr = np.random.randn(n, 1)
X_corr = np.column_stack([X_corr, X_corr + 0.01 * np.random.randn(n, 1)])
y_corr = (X_corr.sum(axis=1) > 0).astype(int)

for name, X, y in [('独立特征', X_indep, y_indep), ('相关特征', X_corr, y_corr)]:
    scores = cross_val_score(GaussianNB(), X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 18.2 不同 var_smoothing 对比

```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

for vs in [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]:
    clf = GaussianNB(var_smoothing=vs)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"var_smoothing={vs:.0e}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 18.3 概率校准对比

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.metrics import brier_score_loss

X, y = load_iris(return_X_y=True)
y_binary = (y == 0).astype(int)

# 未校准
clf_raw = SkGaussianNB().fit(X, y_binary)
proba_raw = clf_raw.predict_proba(X)[:, 1]
brier_raw = brier_score_loss(y_binary, proba_raw)

# 校准后
clf_cal = CalibratedClassifierCV(SkGaussianNB(), cv=5).fit(X, y_binary)
proba_cal = clf_cal.predict_proba(X)[:, 1]
brier_cal = brier_score_loss(y_binary, proba_cal)

print(f"未校准 Brier: {brier_raw:.4f}")
print(f"校准后 Brier: {brier_cal:.4f}")
# 校准后 Brier 分数更低（概率更可靠）
```

### 18.4 与逻辑回归、KNN 对比

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

models = {
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 18.5 特征贡献分析

```python
clf = GaussianNB().fit(X, y)

def feature_contribution(clf, x, cls_idx):
    """计算每个特征对类别 cls 的对数后验贡献。"""
    contributions = np.zeros(len(x))
    for j in range(len(x)):
        log_p = -0.5 * np.log(2 * np.pi * clf.var_[cls_idx, j])
        log_p -= 0.5 * (x[j] - clf.theta_[cls_idx, j]) ** 2 / clf.var_[cls_idx, j]
        contributions[j] = log_p
    return contributions

x = X[0]
for c in range(len(clf.classes_)):
    contrib = feature_contribution(clf, x, c)
    print(f"类别 {c} 各特征贡献: {contrib}")
    print(f"  总贡献: {contrib.sum() + np.log(clf.class_prior_[c]):.4f}")
```

### 18.6 处理缺失值

```python
def predict_with_missing(clf, x, missing_mask):
    """跳过缺失特征的贡献。"""
    best_c, best_ll = None, -np.inf
    for idx in range(len(clf.classes_)):
        ll = np.log(clf.class_prior_[idx])
        for j in range(len(x)):
            if missing_mask[j]:
                continue
            ll += -0.5 * np.log(2 * np.pi * clf.var_[idx, j])
            ll -= 0.5 * (x[j] - clf.theta_[idx, j]) ** 2 / clf.var_[idx, j]
        if ll > best_ll:
            best_c, best_ll = idx, ll
    return clf.classes_[best_c]

clf = GaussianNB().fit(X, y)
x = X[0].copy()
x[2] = np.nan  # 特征 2 缺失
pred = predict_with_missing(clf, x, missing_mask=np.isnan(x))
print(f"含缺失值的预测: {pred}")
```

---

## 十九、参数调优进阶指南

### 19.1 var_smoothing 调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'var_smoothing': [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]}
gs = GridSearchCV(GaussianNB(), param_grid, cv=5).fit(X, y)
print(f"最优 var_smoothing: {gs.best_params_['var_smoothing']}")
print(f"最优分数: {gs.best_score_:.4f}")
```

### 19.2 先验设置

```python
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

# 自定义先验（不平衡数据时有用）
clf = SkGaussianNB(priors=[0.3, 0.7]).fit(X, y_binary)
# 默认先验 = 类别频率，自定义可纠正不平衡
```

### 19.3 调优经验

| 场景 | var_smoothing | priors | 备注 |
|------|---------------|--------|------|
| 默认 | 1e-9 | 类频率 | 通常够 |
| 零方差特征 | 增大 | 类频率 | 防除零 |
| 不平衡数据 | 1e-9 | 均匀 | 纠正先验偏置 |
| 概率需校准 | 1e-9 | 类频率 | 配合 CalibratedClassifierCV |

---

## 二十、常见错误与调试技巧

### 20.1 典型错误清单

```python
# 错误 1：用 GaussianNB 处理离散特征
X_discrete = np.random.randint(0, 5, size=(100, 3))
y = np.random.randint(0, 2, 100)
# GaussianNB 假设连续高斯，对离散特征不准
# 解决：用 MultinomialNB 或 CategoricalNB

# 错误 2：零方差未处理
X = np.array([[1, 2], [1, 3], [1, 4]])  # 特征 0 恒为 1
# 没 var_smoothing 会除零

# 错误 3：概率过度自信
clf = GaussianNB().fit(X, y)
proba = clf.predict_proba(X)
# 概率接近 0/1，不可直接用于决策阈值
# 解决：CalibratedClassifierCV 校准

# 错误 4：强相关特征
X_corr = np.column_stack([np.random.randn(100), np.random.randn(100)])
X_corr[:, 1] = X_corr[:, 0]  # 完全相关
# 独立假设严重违反，概率失真
```

### 20.2 调试检查清单

```python
def debug_gaussian_nb(clf, X, y):
    """GaussianNB 调试。"""
    print("=== GaussianNB 调试 ===")
    print(f"类别数: {len(clf.classes_)}")
    print(f"先验: {clf.class_prior_}")
    
    # 检查方差
    min_var = clf.var_.min()
    if min_var < 1e-8:
        print(f"⚠ 最小方差 {min_var:.2e}，可能零方差特征")
    
    # 检查概率分布
    proba = clf.predict_proba(X)
    max_proba = proba.max(axis=1)
    print(f"最大概率均值: {max_proba.mean():.4f}")
    if max_proba.mean() > 0.95:
        print("⚠ 概率过度自信，建议校准")
    
    print(f"准确率: {clf.score(X, y):.4f}")
```

---

## 二十一、与其他算法的深入对比

### 21.1 GaussianNB vs MultinomialNB vs BernoulliNB

| 变体 | 数据类型 | 似然 | 典型应用 |
|------|---------|------|---------|
| GaussianNB | 连续 | 高斯 | 医疗、传感器 |
| MultinomialNB | 计数 | 多项式 | 文本分类（词频） |
| BernoulliNB | 二值 | 伯努利 | 文本（词出现/不出现） |
| ComplementNB | 计数 | 补集 | 不平衡文本分类 |
| CategoricalNB | 类别 | 类别分布 | 离散特征 |

### 21.2 GaussianNB vs LDA vs QDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

X, y = load_iris(return_X_y=True)

models = {
    'GaussianNB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:15s}: {scores.mean():.4f} ± {scores.std():.4f}")
# GaussianNB 是 QDA 的对角协方差特例
```

### 21.3 训练速度对比

```python
import time

np.random.seed(0)
X = np.random.randn(10000, 50)
y = np.random.randint(0, 5, 10000)

for name, clf in [('GaussianNB', GaussianNB()),
                  ('LogisticRegression', LogisticRegression(max_iter=1000)),
                  ('KNN', KNeighborsClassifier(n_neighbors=5))]:
    t0 = time.time()
    clf.fit(X, y)
    t_fit = time.time() - t0
    print(f"{name:20s}: 训练 {t_fit:.3f}s")
# GaussianNB 最快（无迭代）
```

---

## 二十二、实际应用场景详解

### 22.1 垃圾邮件分类（MultinomialNB）

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = [
    "免费赚钱机会点击链接", "明天下午开会讨论项目", "恭喜中奖领取奖金",
    "项目报告已上传请查看", "限时优惠打折促销", "周末团建活动通知",
    "信用卡办理低息贷款", "代码评审请回复", "投资理财高回报", "需求文档已更新"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=垃圾

vec = CountVectorizer()
X = vec.fit_transform(emails)
clf = MultinomialNB().fit(X, labels)

test_emails = ["免费中奖点击", "代码更新请查看"]
print(clf.predict(vec.transform(test_emails)))  # [1, 0]
```

### 22.2 情感分析

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

texts = ["这部电影太棒了强烈推荐", "垃圾电影浪费时间", "剧情精彩演技在线",
         "难看无聊不推荐", "年度最佳影片", "烂片别看"]
labels = [1, 0, 1, 0, 1, 0]  # 1=正面

pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
pipe.fit(texts, labels)
print(pipe.predict(["好看推荐"]))  # [1]
```

### 22.3 医疗诊断

```python
import numpy as np

# 特征：[血压, 血糖, 年龄, BMI, 心率]
np.random.seed(0)
n = 1000
X = np.column_stack([
    np.random.normal(120, 20, n),    # 血压
    np.random.normal(90, 15, n),     # 血糖
    np.random.uniform(20, 80, n),    # 年龄
    np.random.normal(25, 5, n),      # BMI
    np.random.normal(75, 10, n),     # 心率
])
# 患病概率随各指标异常而增
z = 0.02*(X[:,0]-120) + 0.03*(X[:,1]-90) + 0.02*(X[:,2]-50) + 0.1*(X[:,3]-25)
y = (1/(1+np.exp(-z)) > 0.5).astype(int)

clf = GaussianNB().fit(X, y)
print(f"诊断准确率: {clf.score(X, y):.4f}")

# 新患者诊断
new_patient = np.array([[145, 110, 65, 30, 85]])
proba = clf.predict_proba(new_patient)[0]
print(f"患病概率: {proba[1]:.2%}")
```

### 22.4 文档分类

```python
# 多类文档分类（科技/体育/财经/娱乐）
documents = ["AI技术突破", "足球比赛结果", "股市行情分析", "明星绯闻",
             "机器学习算法", "篮球赛季", "基金收益", "电影上映"]
categories = [0, 1, 2, 3, 0, 1, 2, 3]

pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
pipe.fit(documents, categories)
print(pipe.predict(["深度学习模型"]))  # [0] 科技
```

---

## 二十三、思考题与练习

### 基础题

1. **为什么朴素贝叶斯叫"朴素"？**
   <details><summary>答案</summary>
   假设特征条件独立，这假设"天真"得离谱，但实践效果好。
   </details>

2. **为什么用对数而非原始概率？**
   <details><summary>答案</summary>
   避免下溢（多个小概率相乘趋零），且加法比乘法快。
   </details>

3. **var_smoothing 的作用是什么？**
   <details><summary>答案</summary>
   防止零方差除零，加一个相对最大方差的微小项。
   </details>

### 中级题

4. **证明高斯朴素贝叶斯决策边界的二次性。**
5. **解释为什么朴素贝叶斯概率过度自信。**
6. **推导 MLE 估计 $\hat{\mu}, \hat{\sigma}^2$。**

### 高级题

7. **证明同方差下朴素贝叶斯等价于逻辑回归。**
8. **分析朴素贝叶斯的偏差-方差权衡。**
9. **证明朴素贝叶斯的最大熵解释。**

### 编程练习

10. **实现 MultinomialNB（文本分类）。**
11. **实现 BernoulliNB（二值特征）。**
12. **实现 Welford 在线更新（partial_fit）。**
13. **用朴素贝叶斯做多标签分类。**
14. **比较 GaussianNB、LDA、QDA 在不同数据上的表现。**

---

## 二十四、扩展阅读

### 24.1 经典论文

- **Maron (1961)**：*Automatic Indexing: An Experimental Inquiry*——最早朴素贝叶斯文本分类
- **Lewis (1998)**：*Naive (Bayes) at Forty*——独立假设的惊人效果
- **Rennie et al. (2003)**：*Tackling the Poor Assumptions of Naive Bayes Text Classifiers*——ComplementNB

### 24.2 教材章节

- *The Elements of Statistical Learning* 第 6 章——贝叶斯分类
- *Pattern Classification*（Duda 等）第 2 章——贝叶斯决策理论
- *统计学习方法*（李航）第 4 章——朴素贝叶斯

### 24.3 进阶主题

- **半监督朴素贝叶斯**：EM 算法利用未标注数据
- **结构化朴素贝叶斯**：树形依赖（TAN）
- **AODE**：平均一依赖估计器
- **核朴素贝叶斯**：核密度替代高斯
- **深度朴素贝叶斯**：结合神经网络

### 24.4 相关算法

- **贝叶斯网络**：一般化依赖结构
- **LDA / QDA**：放宽独立假设
- **逻辑回归**：不假设独立，更准但慢
- **最大熵模型**：朴素贝叶斯的理论基础

---

[← 返回算法列表](../index.md)
