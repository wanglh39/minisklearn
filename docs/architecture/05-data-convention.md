# 第五讲：数据约定与校验

> **核心问题**：为什么 `X` 是 `(n_samples, n_features)` 而不是反过来？`check_array` / `check_X_y` 如何把错误前置到入口？数据约定看似琐碎，却是 sklearn 能够把"几十种算法、上百种组合"装进同一套框架的物理基础。

---

## 1. 维度约定：`(n_samples, n_features)`

sklearn 规定特征矩阵 `X` 的 shape 永远是 `(n_samples, n_features)`——**样本在行，特征在列**。这是整个框架最朴素但最重要的约定，所有算法、所有元估计器、所有校验函数都默认它成立。

```python
import numpy as np

X = np.array([
    [1.7, 70, 0],   # 样本 0：身高 1.7m，体重 70kg，性别女
    [1.8, 80, 1],   # 样本 1：身高 1.8m，体重 80kg，性别男
    [1.6, 55, 0],   # 样本 2：身高 1.6m，体重 55kg，性别女
])
# X.shape = (3, 3) → 3 个样本，3 个特征
# X[0]    → 第 0 个样本的所有特征
# X[:, 0] → 所有样本的第 0 个特征（身高）
```

理解这个约定，可以从三个角度切入：

- **行**：一个样本的"完整画像"，对应数据库里的一行记录。
- **列**：一个特征在所有样本上的取值，对应 pandas 的一个 Series。
- **元素 `X[i, j]`**：第 `i` 个样本的第 `j` 个特征值。

### 1.1 一个直观的对照表

| 概念       | 在 `X` 中的位置       | NumPy 写法       | pandas 对应       |
|------------|------------------------|-------------------|---------------------|
| 第 i 个样本 | 第 i 行                | `X[i, :]` 或 `X[i]` | `df.iloc[i]`        |
| 第 j 个特征 | 第 j 列                | `X[:, j]`          | `df.iloc[:, j]`     |
| 全部样本数 | `X.shape[0]`           | `len(X)`           | `len(df)`           |
| 全部特征数 | `X.shape[1]`           | `X.shape[1]`       | `df.shape[1]`       |
| 单个值     | `X[i, j]`              | `X[i, j]`          | `df.iloc[i, j]`     |

### 1.2 为什么样本在行？

#### 1.2.1 与 NumPy 惯例一致

NumPy 的几乎所有聚合操作都默认 `axis=0`，也就是"沿样本方向压缩"。如果样本在行，那么"对每个特征求均值"就是最自然的写法：

```python
X.mean(axis=0)   # → shape (n_features,)，每个特征的均值
X.std(axis=0)    # → 每个特征的标准差
X.min(axis=0)    # → 每个特征的最小值
X.max(axis=0)    # → 每个特征的最大值
```

如果反过来把样本放在列，就要处处写 `axis=1`，既反直觉又容易写错。

#### 1.2.2 内存布局友好

NumPy 默认是 C-order（行主序），同一行的元素在内存中是连续的。当我们对**一个样本**做内积 `w @ x_i` 时，`x_i = X[i]` 是一段连续内存，CPU 缓存命中率高，向量化计算（SIMD）能充分发挥。

```python
# 一次取一个样本做预测（常见于在线推理）
x_i = X[i]            # 连续内存，缓存友好
y_i = w @ x_i + b     # 高效内积
```

如果样本在列，`X[:, i]` 是跨步取值（strided access），缓存命中率差，性能下降。

#### 1.2.3 与数学记号一致

线性代数教材里，数据矩阵几乎一律写作 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 是样本数、$d$ 是特征数。线性模型的预测公式：

$$
\hat{y} = X w + b
$$

这里 $w \in \mathbb{R}^{d}$、$\hat{y} \in \mathbb{R}^{n}$，形状 `(n, d) @ (d,) → (n,)` 自然对齐。如果反过来写成 `(d, n)`，每次写公式都要转置，徒增心智负担。

#### 1.2.4 与 pandas / 数据库一致

pandas DataFrame 的 `df.iloc[i]` 取第 i 行，数据库 `SELECT * FROM t LIMIT 1` 返回一行。sklearn 直接接受 DataFrame 作为输入，并保持"行=样本"的语义：

```python
import pandas as pd

df = pd.DataFrame({
    "height": [1.7, 1.8, 1.6],
    "weight": [70, 80, 55],
    "sex":    [0, 1, 0],
})
clf.fit(df, y)         # sklearn 内部把 df 转成 ndarray，行=样本
clf.predict(df.iloc[:2])  # 预测前两个样本
```

### 1.3 与图像约定的对比

图像库（如 PyTorch）常用 `(C, H, W)` 或 `(N, C, H, W)`——**通道在前**。这与 sklearn 的 `(N, D)` 不冲突，因为 sklearn 处理的是**表格数据**，不处理图像。

| 领域         | 约定                | 第一维含义 | 原因                       |
|--------------|----------------------|------------|----------------------------|
| sklearn      | `(n_samples, n_features)` | 样本       | 表格数据、批量推理         |
| PyTorch CNN  | `(N, C, H, W)`       | batch      | 卷积核沿通道维做内积       |
| TensorFlow   | `(N, H, W, C)`       | batch      | 与 HWIO 卷积核形状对齐     |
| OpenCV       | `(H, W, C)`          | 行（y 坐标）| 图像坐标系原点在左上角     |

如果你要把图像喂给 sklearn（例如做像素级分类），需要先 flatten 成 `(n_samples, n_features)`：

```python
# (N, C, H, W) → (N, C*H*W)
X_flat = X_images.reshape(X_images.shape[0], -1)
clf.fit(X_flat, y)
```

### 1.4 常见错误示例

#### 1.4.1 把单个样本当成一维数组

```python
# 错误：单个样本应该是 (1, n_features)，不是 (n_features,)
x = np.array([1.7, 70, 0])        # shape (3,)，sklearn 会误解为 3 个样本、1 个特征
clf.predict(x)                    # 行为不可预期

# 正确：升维
x = np.array([[1.7, 70, 0]])      # shape (1, 3)
clf.predict(x)                    # → shape (1,)
```

`check_array` 默认 `ensure_2d=True`，会把一维数组**重解释为列向量**（n 个样本、1 个特征），而不是一行。这是初学者最常踩的坑。

#### 1.4.2 把特征当成样本

```python
# 想做"5 个特征、1000 个样本"，却写反了
X = np.random.randn(5, 1000)      # sklearn 会理解为 5 个样本、1000 个特征
clf.fit(X, y)                     # 如果 y 长度是 1000，会报样本数不一致
```

#### 1.4.3 混淆 pandas 的行和列

```python
# 错误：把特征当成行
X = df.T                          # 转置后 shape 变成 (n_features, n_samples)
clf.fit(X, y)                     # 样本数对不上
```

### 1.5 历史背景

sklearn 的维度约定继承自 SciPy 和早期的 MATLAB 科学计算生态。在 MATLAB 中，数据矩阵也是 `samples × features`（MATLAB 索引从 1 开始，但维度顺序一致）。R 的 `matrix` 默认是**列主序**，但 `data.frame` 仍然是"行=观测、列=变量"，语义与 sklearn 一致。

早期机器学习库（如 Weka、Orange）的数据格式五花八门，sklearn 选择与 NumPy 对齐，使得"数据在 NumPy / pandas / sklearn 之间无缝流转"成为可能，这是它后来居上的重要原因之一。

### 1.6 思考题

1. 如果让你重新设计一个 ML 框架，你会把样本放在行还是列？为什么？
2. `X.mean(axis=0)` 和 `X.mean(axis=1)` 分别返回什么形状？哪个更常用于特征标准化？
3. 为什么 `X[i]` 比 `X[:, i]` 快？用 `%timeit` 实测一下。
4. 如果输入是 `(n_samples, 1)` 的单特征矩阵，和 `(n_samples,)` 的一维数组，sklearn 内部会如何区别处理？

---

## 2. 标签约定

```python
# 单输出分类 / 回归
y = np.array([0, 1, 1, 0])        # shape (n_samples,)，二分类
y = np.array([1.5, 2.3, 3.1])     # shape (n_samples,)，回归
y = np.array(["cat", "dog", "cat"])  # 字符串标签也合法

# 多输出
y = np.array([[0, 1.5], [1, 2.3]])  # shape (n_samples, n_outputs)
```

`y` 通常是**一维**的，多输出时才二维。这个约定比 `X` 的约定更微妙，因为 `y` 的维度既承载"样本数"，又承载"输出个数"。

### 2.1 单输出：一维

绝大多数监督学习任务（二分类、多分类、回归）的标签是一维数组：

```python
y_class = np.array([0, 1, 1, 0, 1])          # 二分类
y_multi = np.array([0, 1, 2, 1, 0])          # 多分类
y_reg   = np.array([2.3, 4.1, 5.0, 3.7])     # 回归
y_str   = np.array(["spam", "ham", "spam"])   # 字符串标签
```

一维的好处：

- 形状 `(n_samples,)` 直接对应样本数，不需要 `y.shape[0]`。
- 与 NumPy 的索引习惯一致：`y[i]` 是第 i 个样本的标签。
- 与 pandas Series 一致：`df["target"].values` 就是 `(n_samples,)`。

### 2.2 多输出：二维

当每个样本有多个目标时（例如同时预测身高和体重），`y` 升为二维：

```python
# 100 个样本，每个样本预测 2 个目标
y = np.random.randn(100, 2)
reg.fit(X, y)
reg.predict(X[:5]).shape   # → (5, 2)
```

sklearn 用 `MultiOutputRegressor` 和 `MultiOutputClassifier` 包装单输出估计器来处理多输出：

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

multi_reg = MultiOutputRegressor(LinearRegression())
multi_reg.fit(X, y)            # y shape (n_samples, n_outputs)
multi_reg.predict(X[:5]).shape # → (5, n_outputs)
```

### 2.3 标签编码

sklearn 内部用 `LabelEncoder` 把字符串标签转成整数：

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(["cat", "dog", "cat", "fish"])
# → array([0, 1, 0, 2])
le.classes_   # → array(['cat', 'dog', 'fish'])
le.inverse_transform([0, 1, 2])   # → array(['cat', 'dog', 'fish'])
```

分类器内部会自动调用类似机制，所以你可以直接传字符串标签：

```python
clf.fit(X, ["spam", "ham", "spam"])
clf.classes_   # → array(['ham', 'spam'])，按字典序
```

### 2.4 连续 vs 离散的判定

sklearn 通过 dtype 判定 `y` 是分类还是回归：

- 整数 dtype（`int64` 等）→ 分类
- 浮点 dtype（`float64`）→ 回归
- 字符串 / object dtype → 分类

```python
y_int = np.array([0, 1, 1, 0])        # → 分类
y_float = np.array([0.0, 1.0, 1.0])   # → 回归（即使值都是整数）
y_str = np.array(["a", "b", "a"])     # → 分类
```

这个判定有时会让人困惑：

```python
# 误把整数编码的回归目标当成分类
y = np.array([1, 2, 3, 4])   # dtype int64
# 如果你本意是回归（预测连续评分），应该转成 float
y = y.astype(float)
```

### 2.5 常见陷阱

#### 2.5.1 标签 dtype 暗中影响算法

```python
from sklearn.linear_model import LogisticRegression, LinearRegression

y = np.array([1, 2, 3, 4])
# LogisticRegression 会把 y 当成 4 分类
# LinearRegression 会把 y 当成连续目标
```

#### 2.5.2 多输出写成 list of list 而不是 ndarray

```python
y = [[0, 1.5], [1, 2.3]]      # list of list
clf.fit(X, y)                 # 内部会转成 ndarray，但容易出 dtype 陷阱
y = np.array([[0, 1.5], [1, 2.3]])  # 显式 ndarray 更安全
```

#### 2.5.3 标签中有 NaN

```python
y = np.array([1.0, 2.0, np.nan, 3.0])
clf.fit(X, y)   # 大多数分类器会报错，回归器可能静默产出 NaN
```

### 2.6 思考题

1. 为什么 sklearn 不强制 `y` 是整数？这样设计有什么好处和坏处？
2. `y.shape == (n,)` 和 `y.shape == (n, 1)` 在 sklearn 里会被区别对待吗？
3. 如果标签是 `["cat", "dog", None]`，会发生什么？
4. 多输出回归和多标签分类是一回事吗？它们的 `y` 形状分别是什么？

---

## 3. `check_array`：入口校验

机器学习代码最常见的崩溃不是算法错误，而是数据错误。sklearn 的解法是**入口校验**：在 `fit` / `predict` / `transform` 的最开头把数据"洗"成合法形态，遇到非法数据立刻报清晰错误。

### 3.1 设计动机

#### 3.1.1 错误前置原则

考虑没有校验时，一个含 NaN 的 `X` 喂给 LogisticRegression：

```python
clf.fit(X_with_nan, y)
# → numpy.linalg.LinAlgError: SVD did not converge
```

这个错误来自底层 SVD，对用户毫无帮助——他不知道是 NaN 导致的，也不知道 NaN 在哪一行。

加上校验后：

```python
clf.fit(X_with_nan, y)
# → ValueError: Input contains NaN at row 2, column 1
```

错误**前置**到 `fit` 的第一行，信息清晰、定位精确。这就是"防御性编程"在 sklearn 里的体现。

#### 3.1.2 防御性编程的三个层次

| 层次       | 做法                         | sklearn 中的体现                |
|------------|------------------------------|----------------------------------|
| 入口校验   | 在边界检查输入               | `check_array` / `check_X_y`      |
| 不变式维护 | 在操作后检查中间状态         | `assert` 在算法内部              |
| 异常传播   | 让错误显式抛出而非静默吞掉   | `raise ValueError` 而非 `return None` |

sklearn 主要依赖第一层，因为机器学习的中间状态太复杂，逐个断言不现实；而入口校验投入产出比最高。

### 3.2 简化实现

```python
import numpy as np

def check_array(array, dtype=None, ensure_2d=True, force_all_finite=True,
                copy=False, ensure_min_samples=1, ensure_min_features=1):
    """校验并清洗输入数组，返回一个干净的 ndarray。"""
    # 1. 接受 list / tuple / DataFrame → 转 ndarray
    array = np.asarray(array, dtype=dtype)

    # 2. 检查 NaN / Inf
    if force_all_finite:
        if not np.all(np.isfinite(array)):
            bad = np.argwhere(~np.isfinite(array))
            raise ValueError(
                f"输入包含 NaN 或 inf，第一个非法元素在 {tuple(bad[0])}"
            )

    # 3. 维度检查
    if ensure_2d:
        if array.ndim == 1:
            array = array.reshape(-1, 1)   # 一维 → 列向量（n 个样本、1 个特征）
        elif array.ndim != 2:
            raise ValueError(
                f"期望二维数组，得到 {array.ndim} 维，shape={array.shape}"
            )

    # 4. 最小样本数 / 特征数
    if ensure_2d:
        n_samples, n_features = array.shape
        if n_samples < ensure_min_samples:
            raise ValueError(f"样本数 {n_samples} 少于最小要求 {ensure_min_samples}")
        if n_features < ensure_min_features:
            raise ValueError(f"特征数 {n_features} 少于最小要求 {ensure_min_features}")

    # 5. 是否复制
    if copy:
        array = array.copy()

    return array
```

### 3.3 各参数详解

| 参数                  | 默认      | 作用                                   |
|-----------------------|-----------|----------------------------------------|
| `dtype`               | None      | 强制转换的 dtype                       |
| `ensure_2d`           | True      | 是否强制二维                           |
| `force_all_finite`    | True      | 是否禁止 NaN / Inf                     |
| `copy`                | False     | 是否强制复制（避免修改原数组）         |
| `ensure_min_samples`  | 1         | 最少样本数                             |
| `ensure_min_features` | 1         | 最少特征数                             |
| `accept_sparse`       | False     | 是否接受稀疏矩阵                       |
| `order`               | None      | 强制内存顺序（'C' / 'F'）              |

#### 3.3.1 `dtype` 的常见用法

```python
# 把整数特征转成浮点（很多算法要求浮点）
X = check_array(X, dtype=np.float64)

# 把标签转成整数
y = check_array(y, dtype=np.int64, ensure_2d=False)
```

#### 3.3.2 `ensure_2d=False` 的场景

```python
# 一维标签
y = check_array(y, ensure_2d=False)

# 一维系数
coef = check_array(coef, ensure_2d=False)
```

#### 3.3.3 `force_all_finite=False` 的场景

某些算法天然支持缺失值（如 `SimpleImputer`、XGBoost），需要关掉这个检查：

```python
X = check_array(X, force_all_finite=False)   # 允许 NaN
```

新版 sklearn 把这个参数拆成 `force_all_finite="allow-nan"`，区分"允许 NaN 但不允许 Inf"和"两者都允许"。

#### 3.3.4 `accept_sparse` 的场景

文本数据通常是高维稀疏的，用稀疏矩阵能省大量内存：

```python
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X_dense)
X = check_array(X_sparse, accept_sparse=True)   # 保留稀疏格式
```

### 3.4 校验的价值：对比示例

#### 3.4.1 没有校验

```python
def fit_naive(X, y):
    # 直接做矩阵运算，不做任何检查
    return np.linalg.inv(X.T @ X) @ X.T @ y

X_bad = np.array([[1, 2], [1, 2], [np.nan, 3]])
fit_naive(X_bad, y)
# → numpy.linalg.LinAlgError: SVD did not converge
# 用户：??? 哪里出了问题？
```

#### 3.4.2 有校验

```python
def fit_checked(X, y):
    X = check_array(X, dtype=np.float64)
    y = check_array(y, ensure_2d=False)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X 样本数 {X.shape[0]} != y 样本数 {y.shape[0]}")
    return np.linalg.inv(X.T @ X) @ X.T @ y

fit_checked(X_bad, y)
# → ValueError: 输入包含 NaN 或 inf，第一个非法元素在 (2, 0)
# 用户：哦，第 2 行第 0 列有 NaN，我去处理一下。
```

### 3.5 性能权衡

校验不是免费的——`np.isfinite` 要扫描整个数组。对于大数据集，这个开销可能显著。

```python
import timeit

X_large = np.random.randn(100000, 100)

# 不校验
t1 = timeit.timeit(lambda: X_large @ X_large.T, number=10)

# 校验
t2 = timeit.timeit(lambda: (check_array(X_large), X_large @ X_large.T)[1], number=10)

# 校验开销约 5%~10%
```

sklearn 的应对：

1. **`assume_finite` 配置**：用户确认数据干净时可以跳过 NaN 检查。
2. **校验只做一次**：`fit` 校验后，内部算法不再重复校验。
3. **延迟校验**：某些校验（如稀疏矩阵的内部结构）只在真正用到时才做。

```python
from sklearn import config_context

with config_context(assume_finite=True):
    clf.fit(X_large, y)   # 跳过 NaN 检查，加速
```

### 3.6 自定义校验

你可以写自己的校验函数，复用 `check_array` 的逻辑：

```python
def check_positive_semidefinite(K):
    """校验核矩阵是半正定的。"""
    K = check_array(K, dtype=np.float64, ensure_2d=True)
    eigvals = np.linalg.eigvalsh(K)
    if np.min(eigvals) < -1e-10:
        raise ValueError(f"矩阵不是半正定，最小特征值 {np.min(eigvals)}")
    return K
```

### 3.7 校验失败时的报错艺术

好的报错信息应该包含三要素：

1. **是什么**：违反了哪条约束。
2. **在哪里**：哪个位置的数据有问题。
3. **怎么办**：建议的修复方向。

```python
# 差回的报错
raise ValueError("invalid input")

# 好的报错
raise ValueError(
    f"期望二维数组，得到 {array.ndim} 维数组，shape={array.shape}。"
    f"如果你传入的是单个样本，请用 x.reshape(1, -1) 升维。"
)
```

sklearn 的报错信息经过多年打磨，几乎每条都包含修复建议，这是它"用户友好"口碑的重要来源。

### 3.8 思考题

1. 为什么 `check_array` 默认 `ensure_2d=True`？如果默认 False 会出什么问题？
2. `force_all_finite=True` 会让哪些算法无法工作？举两个例子。
3. 校验开销是 O(n) 的，为什么不在第一次校验后设个 flag 跳过后续校验？
4. 写一个 `check_categorical_array`，要求输入是字符串或整数 dtype，且类别数不超过 100。

---

## 4. `check_X_y`：样本数一致性

```python
def check_X_y(X, y, accept_sparse=False, y_numeric=False,
              multi_output=False):
    """同时校验 X 和 y，并检查样本数一致。"""
    X = check_array(X, accept_sparse=accept_sparse)
    y = check_array(y, ensure_2d=False, dtype=None)

    if y_numeric and not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"y 必须是数值类型，得到 {y.dtype}")

    if not multi_output:
        y = y.ravel()   # 强制一维

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X 的样本数 {X.shape[0]} 与 y 的样本数 {y.shape[0]} 不一致"
        )
    return X, y
```

样本数不一致是最常见的用户错误，`check_X_y` 把它前置到 `fit` 入口。

### 4.1 各种不一致场景

#### 4.1.1 X 和 y 长度不同

```python
X = np.random.randn(100, 5)
y = np.random.randn(99)   # 少一个
clf.fit(X, y)
# → ValueError: X 的样本数 100 与 y 的样本数 99 不一致
```

#### 4.1.2 用错 DataFrame 的列

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
X = df[["a", "b"]].values
y = df["a"].values   # 误把特征当标签
clf.fit(X, y)        # 样本数一致但语义错误，校验无法发现
```

校验只能发现**结构**错误，发现不了**语义**错误。后者要靠人工审查。

#### 4.1.3 多输出 y 形状错

```python
X = np.random.randn(100, 5)
y = np.random.randn(100, 2, 3)   # 三维
clf.fit(X, y)
# → ValueError: 期望一维或二维 y，得到 3 维
```

### 4.2 多输出场景

```python
X = np.random.randn(100, 5)
y = np.random.randn(100, 3)   # 3 个输出
multi_reg.fit(X, y)            # 内部用 check_X_y(X, y, multi_output=True)
```

### 4.3 权重向量的一致性

`sample_weight` 也要和 `X` 样本数一致：

```python
def check_sample_weight(sample_weight, X):
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.shape != (X.shape[0],):
        raise ValueError(
            f"sample_weight 长度 {sample_weight.shape[0]} != X 样本数 {X.shape[0]}"
        )
    if np.any(sample_weight < 0):
        raise ValueError("sample_weight 不能有负值")
    return sample_weight
```

### 4.4 思考题

1. 为什么 `check_X_y` 不检查 `X` 和 `y` 的 dtype 是否兼容？
2. 如果 `y` 是 `(n, 1)` 的二维数组，`check_X_y` 会怎么处理？
3. 写一个 `check_X_y_weight`，同时校验 `X`、`y`、`sample_weight` 三者样本数一致。

---

## 5. `check_is_fitted`：防止未训练就预测

```python
def check_is_fitted(estimator, attributes=None, msg=None):
    """检查估计器是否已经 fit 过。"""
    if attributes is None:
        attributes = [attr for attr in vars(estimator) if attr.endswith("_")]
    if not all(hasattr(estimator, attr) for attr in attributes):
        raise NotFittedError(msg or "请先调用 fit() 再做预测")
```

在 `predict` / `transform` / `score` 开头调用：

```python
def predict(self, X):
    check_is_fitted(self, ['coef_', 'intercept_'])   # 没拟合就报错
    X = check_array(X)
    return X @ self.coef_ + self.intercept_
```

### 5.1 没有这个检查会怎样

```python
clf = LogisticRegression()
clf.predict(X)
# 没有 check_is_fitted：
# → AttributeError: 'LogisticRegression' object has no attribute 'coef_'
# 用户：coef_ 是什么？我哪里没做？

# 有 check_is_fitted：
# → NotFittedError: 请先调用 fit() 训练模型，再调用 predict()。
# 用户：哦，忘了 fit。
```

`AttributeError` 是 Python 内部错误，对用户不友好；`NotFittedError` 是 sklearn 主动抛出，信息明确。

### 5.2 拟合标记的选择

#### 5.2.1 用学出的属性

```python
check_is_fitted(self, 'coef_')   # 单个属性
check_is_fitted(self, ['coef_', 'intercept_'])   # 多个属性
```

#### 5.2.2 用专门的标记属性

某些估计器没有明显的"学出属性"（例如 KNN 把训练集全存下来），可以用专门标记：

```python
class KNNClassifier:
    def fit(self, X, y):
        self._fit_X = X
        self._fit_y = y
        return self

    def predict(self, X):
        check_is_fitted(self, '_fit_X')
        ...
```

#### 5.2.3 自动推断（新版 sklearn）

新版 sklearn 支持不传 `attributes`，自动找所有以 `_` 结尾的属性：

```python
check_is_fitted(self)   # 自动检查所有 *_ 属性
```

### 5.3 常见陷阱

#### 5.3.1 在 `__init__` 里设默认值

```python
class BadLR:
    def __init__(self):
        self.coef_ = None   # 坏：__init__ 里就有 coef_，check_is_fitted 永远通过

    def predict(self, X):
        check_is_fitted(self, 'coef_')   # 永远不报错
        return X @ self.coef_            # 但 coef_ 是 None，崩溃
```

正确做法是 `coef_` 只在 `fit` 里赋值，`__init__` 不碰它。

#### 5.3.2 用非下划线属性做标记

```python
check_is_fitted(self, 'coef')   # 没有下划线，约定上不算"学出属性"
```

虽然能工作，但违反命名约定，会让 `clone` 误以为是超参数。

### 5.4 思考题

1. 为什么 `check_is_fitted` 默认检查以 `_` 结尾的属性？这和下划线约定有什么关系？
2. 如果一个估计器的 `fit` 是 no-op（例如 `PassthroughTransformer`），怎么实现 `check_is_fitted`？
3. `NotFittedError` 继承自 `ValueError` 还是 `RuntimeError`？为什么？

---

## 6. 下划线约定：学出的参数

| 命名                  | 含义             | 何时存在       | 是否被 `clone` 保留 |
|-----------------------|------------------|----------------|----------------------|
| `self.C`              | 超参数           | `__init__` 后  | 是                   |
| `self.coef_`          | 学出的参数       | `fit` 后       | 否                   |
| `self.classes_`       | 学出的类别       | `fit` 后       | 否                   |
| `self.n_features_in_` | 输入特征数       | `fit` 后       | 否                   |
| `self._fit_X`         | 内部训练数据     | `fit` 后       | 否                   |

下划线结尾 = "这是 `fit` 学出来的，不是用户传入的"。这个约定让 `clone` 知道哪些属性该丢弃。

### 6.1 命名规则的精确定义

- **超参数**：在 `__init__` 里赋值的属性，**不以 `_` 结尾**。例如 `self.C = C`、`self.alpha = alpha`。
- **学出属性**：在 `fit` 里赋值的属性，**以 `_` 结尾**。例如 `self.coef_ = ...`、`self.classes_ = ...`。
- **私有属性**：以 `_` 开头（不强求结尾），纯内部使用。例如 `self._fit_X`、`self._cache`。

```python
class LogisticRegression:
    def __init__(self, C=1.0, penalty='l2'):
        self.C = C               # 超参数，无下划线
        self.penalty = penalty   # 超参数，无下划线

    def fit(self, X, y):
        self.coef_ = ...         # 学出属性，下划线结尾
        self.classes_ = ...      # 学出属性，下划线结尾
        self.n_features_in_ = X.shape[1]   # 学出属性
        return self
```

### 6.2 与 `clone` 的关系

`clone` 创建一个"干净"的估计器：保留超参数，丢弃学出属性。它通过 `get_params` 拿到超参数，再 `__init__` 一个新实例：

```python
def clone(estimator):
    params = estimator.get_params()   # 只拿超参数（无下划线）
    new = type(estimator)(**params)   # 重新 __init__
    return new                        # 没有任何学出属性
```

所以下划线约定是 `clone` 能工作的**前提**——如果 `coef_` 没有下划线，`get_params` 会把它当超参数，`clone` 后会保留它，破坏语义。

### 6.3 常见学出属性列表

| 属性                  | 含义                       | 出现的估计器                |
|-----------------------|----------------------------|------------------------------|
| `coef_`               | 系数 / 权重                | 线性模型                     |
| `intercept_`          | 截距                       | 线性模型                     |
| `classes_`            | 类别标签                   | 分类器                       |
| `n_features_in_`      | 输入特征数                 | 所有估计器                   |
| `feature_names_in_`   | 输入特征名                 | 接受 DataFrame 的估计器      |
| `labels_`             | 聚类标签                   | 聚类器                       |
| `cluster_centers_`    | 聚类中心                   | KMeans                       |
| `components_`         | 主成分 / 字典原子           | PCA / DictionaryLearning     |
| `explained_variance_` | 解释方差                   | PCA                          |
| `support_`            | 选中的特征 mask            | SelectKBest / SVM            |
| `n_iter_`             | 实际迭代次数               | 迭代算法                     |

### 6.4 `n_features_in_` 的一致性检查

`fit` 时记录特征数，`predict` 时检查是否一致：

```python
def fit(self, X, y):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]   # 记住训练时的特征数
    ...

def predict(self, X):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
        raise ValueError(
            f"X 有 {X.shape[1]} 个特征，但模型是用 {self.n_features_in_} 个特征训练的"
        )
    ...
```

这能防止"训练用 5 个特征，预测给 4 个"这类错误。

### 6.5 思考题

1. 为什么 `n_features_in_` 要在 `fit` 里记录，而不是 `__init__`？
2. `clone` 后的估计器有 `coef_` 吗？为什么？
3. 如果一个属性既是超参数又是学出属性（例如 `random_state` 被替换成实际种子），该怎么命名？
4. `feature_names_in_` 是什么？为什么需要它？

---

## 7. 数据类型约定

### 7.1 dtype 处理

sklearn 内部统一用 `float64` 做浮点运算，用 `int64` 做整数标签。`check_array` 的 `dtype` 参数负责转换：

```python
X = check_array(X, dtype=np.float64)   # 强制 float64
y = check_array(y, dtype=np.int64, ensure_2d=False)   # 强制 int64
```

转换的代价是数据复制：

```python
X_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
X_float = check_array(X_int, dtype=np.float64)   # 复制 + 转换
X_float is X_int   # False
```

如果数据本来就是 `float64`，`check_array` 不会复制：

```python
X = np.random.randn(100, 5)   # 默认 float64
X2 = check_array(X, dtype=np.float64)
X2 is X   # True（同一对象）
```

### 7.2 浮点 vs 整数

| 场景               | 推荐 dtype  | 原因                         |
|--------------------|-------------|------------------------------|
| 特征矩阵 `X`       | float64     | 算法统一、精度足够           |
| 分类标签 `y`       | int64       | 类别索引                     |
| 回归标签 `y`       | float64     | 连续值                       |
| 样本权重           | float64     | 非负浮点                     |
| 系数 `coef_`       | float64     | 学出的浮点                   |

### 7.3 稀疏矩阵

文本数据 one-hot 后是高维稀疏的，用 `scipy.sparse` 能省 90%+ 内存：

```python
from scipy.sparse import csr_matrix

X_dense = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]])
X_sparse = csr_matrix(X_dense)
print(X_sparse.memory_usage())   # 远小于 X_dense.nbytes
```

sklearn 大部分估计器接受稀疏输入：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer()
X_sparse = vec.fit_transform(["hello world", "foo bar"])   # csr_matrix
clf = LogisticRegression()
clf.fit(X_sparse, [0, 1])   # 直接接受稀疏
```

但不是所有算法都支持稀疏：

```python
from sklearn.tree import DecisionTreeClassifier
# DecisionTreeClassifier 支持稀疏

from sklearn.naive_bayes import GaussianNB
# GaussianNB 不支持稀疏，会报错
clf.fit(X_sparse, y)
# → TypeError: A sparse matrix was passed, but dense data is required
```

### 7.4 思考题

1. 为什么 sklearn 不用 `float32` 做默认？省一半内存不是更好吗？
2. 稀疏矩阵的 `nnz` 是什么？它和内存占用什么关系？
3. `csr_matrix` 和 `csc_matrix` 有什么区别？sklearn 默认用哪个？

---

## 8. 内存布局与性能

### 8.1 C-order vs F-order

NumPy 数组有两种内存顺序：

- **C-order**（行主序）：同一行连续，`X[i]` 快。NumPy 默认。
- **F-order**（列主序）：同一列连续，`X[:, i]` 快。Fortran / MATLAB 默认。

```python
X_C = np.random.randn(1000, 1000)                    # C-order
X_F = np.asfortranarray(np.random.randn(1000, 1000)) # F-order

# 矩阵乘法在不同顺序下性能不同
%timeit X_C @ X_C.T      # C-order 更快
%timeit X_F @ X_F.T      # F-order 可能慢 2x
```

sklearn 默认要求 C-order，但有些算法（如线性代数密集运算）在 F-order 下更快，会内部转换。

### 8.2 连续内存的重要性

```python
X = np.random.randn(10000, 100)
X_strided = X[::2]   # 跨步取偶数行，不是连续内存

%timeit X.sum(axis=0)            # 快
%timeit X_strided.sum(axis=0)    # 慢 2-3x
```

`check_array` 的 `order='C'` 参数会强制转成连续内存：

```python
X = check_array(X_strided, order='C')   # 复制成连续
```

### 8.3 思考题

1. 为什么 C-order 对 sklearn 的"批量样本"模式更友好？
2. `np.ascontiguousarray` 做了什么？什么时候会复制？
3. 转置 `X.T` 会改变内存顺序吗？`X.T` 是连续的吗？

---

## 9. 与其他框架深度对比

### 9.1 PyTorch

PyTorch 的数据约定：

- 张量 shape `(N, C, H, W)` 用于图像，`(N, D)` 用于表格。
- 不做入口校验，传错形状会在 forward 里崩。
- 没有下划线约定，模型属性自由命名。

```python
import torch

X = torch.randn(100, 5)   # (N, D)，和 sklearn 一致
y = torch.randint(0, 2, (100,))

model = torch.nn.Linear(5, 1)
model.fit(X, y)   # 不存在！PyTorch 没有统一的 fit
```

PyTorch 把训练循环留给用户，所以不需要"统一 API 契约"，也不需要入口校验——用户自己负责。

### 9.2 TensorFlow / Keras

Keras 有类似的 `fit` / `predict` API，但数据约定不同：

- 输入通常是 `tf.Tensor` 或 `np.ndarray`。
- 标签 `y` 的形状约定不如 sklearn 严格。
- 没有 `clone` / `get_params` 机制，超参数管理靠构造函数。

```python
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
model.fit(X, y, epochs=5)   # Keras 的 fit
# 没有 model.get_params()，也没有 clone(model)
```

### 9.3 R 的 caret / tidymodels

R 生态有类似的"统一 API"理念：

```r
library(caret)
model <- train(y ~ ., data = df, method = "rf")
predict(model, newdata)
```

但 R 的数据约定是 `data.frame`，不是矩阵；标签通过公式 `y ~ .` 指定，而不是分开传 `X, y`。

### 9.4 Julia 的 MLJ

MLJ 是 Julia 的 ML 框架，借鉴了 sklearn 的很多理念：

```julia
using MLJ
model = LogisticClassifier()
mach = machine(model, X, y)
fit!(mach)
predict(mach, X_new)
```

它也有 `fit` / `predict` 统一 API，但数据约定用 Julia 的 `Tables.jl`，更灵活。

### 9.5 对比表

| 框架          | 数据形状约定         | 入口校验 | 下划线约定 | clone 机制 |
|---------------|------------------------|----------|------------|------------|
| sklearn       | `(n_samples, n_features)` | 强       | 有         | 有         |
| PyTorch       | 自由                   | 弱       | 无         | 无         |
| Keras         | 自由                   | 弱       | 无         | 无         |
| R caret       | data.frame            | 中       | 无         | 无         |
| Julia MLJ     | Tables.jl             | 中       | 有         | 有         |

sklearn 是**约定最严格**的，也是**通用工具最丰富**的——这两者互为因果。

---

## 10. 常见陷阱与 FAQ

### 10.1 `X` 是 DataFrame 时列名丢失

```python
df = pd.DataFrame({"height": [1.7, 1.8], "weight": [70, 80]})
clf.fit(df, y)
# 内部转成 ndarray，列名丢失
# 新版 sklearn 会把列名存到 clf.feature_names_in_
```

### 10.2 `y` 是 Series 时索引不对齐

```python
y = pd.Series([0, 1, 0], index=[10, 20, 30])
clf.fit(X, y)   # sklearn 内部用 y.values，忽略索引
# 如果 X 是 DataFrame 且索引不同，可能出问题
```

### 10.3 整数特征被当成类别

某些估计器（如 LightGBM）会把整数 dtype 当成类别，sklearn 一般不会，但要注意：

```python
X = np.array([[1, 2], [3, 4]])   # int
clf.fit(X, y)   # sklearn 内部转成 float，不会当类别
```

### 10.4 复制 vs 视图

```python
X = np.random.randn(100, 5)
X2 = check_array(X)         # 可能返回视图（不复制）
X2[0, 0] = 999              # 可能改到原数组！
```

要安全就传 `copy=True`：

```python
X2 = check_array(X, copy=True)   # 保证独立
```

### 10.5 FAQ

**Q: 为什么 sklearn 不接受 list of list？**
A: 接受，内部会转成 ndarray。但每次转换有开销，建议直接传 ndarray。

**Q: 为什么 `predict` 也要校验 `X`？**
A: 因为 `predict` 是新入口，用户可能传错数据。校验把错误前置到 `predict` 而不是算法内部。

**Q: 能不能关掉所有校验换性能？**
A: 可以用 `config_context(assume_finite=True)` 跳过 NaN 检查，但维度 / dtype 检查不能关——它们是正确性的基础。

---

## 11. 数据预处理的约定

sklearn 的预处理模块（`preprocessing`）也遵循统一的数据约定，使得预处理可以无缝串到 Pipeline 里。

### 11.1 标准化与归一化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: 减均值除标准差
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 等价于 (X - X.mean(axis=0)) / X.std(axis=0)

# MinMaxScaler: 缩放到 [0, 1]
mm = MinMaxScaler()
X_mm = mm.fit_transform(X)

# RobustScaler: 用中位数和分位数，对离群点鲁棒
rs = RobustScaler()
X_rs = rs.fit_transform(X)
```

它们都遵循 `TransformerMixin` 契约：`fit` 学出 `mean_` / `scale_` 等属性，`transform` 应用变换，`inverse_transform` 反变换。

### 11.2 编码器

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# OneHotEncoder: 类别特征 → one-hot
ohe = OneHotEncoder(sparse_output=False)
X_ohe = ohe.fit_transform([["cat"], ["dog"], ["fish"]])
# → array([[1., 0., 0.],
#           [0., 1., 0.],
#           [0., 0., 1.]])

# OrdinalEncoder: 类别特征 → 整数
oe = OrdinalEncoder()
X_oe = oe.fit_transform([["cat"], ["dog"], ["fish"]])
# → array([[0.], [1.], [2.]])
```

### 11.3 预处理与 Pipeline 的协作

预处理器的统一约定让它能直接放进 Pipeline：

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipe.fit(X, y)   # scaler.fit_transform → clf.fit
pipe.predict(X_test)   # scaler.transform → clf.predict
```

如果预处理器不遵循 `(n_samples, n_features)` 约定，Pipeline 就没法这样组合。

### 11.4 思考题

1. `StandardScaler.fit_transform` 和 `fit` + `transform` 有什么区别？为什么有时前者更快？
2. `OneHotEncoder` 默认输出稀疏矩阵，为什么？怎么改成密集？
3. 如果训练集和测试集的类别不同，`OneHotEncoder.transform` 会怎么处理？

---

## 12. 序列化与持久化的约定

sklearn 估计器用 pickle 序列化，但有一些约定需要注意。

### 12.1 pickle 的基本用法

```python
import pickle

# 保存
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# 加载
with open("model.pkl", "rb") as f:
    clf_loaded = pickle.load(f)

clf_loaded.predict(X_test)   # 和原模型一致
```

### 12.2 版本兼容性

sklearn 不保证跨大版本的 pickle 兼容：

```python
# 用 0.20 训练的模型，0.24 加载可能报错
# 建议用同一版本训练和推理
```

更安全的做法是保存超参数 + 训练数据，重新训练：

```python
params = clf.get_params()
# 保存 params 和数据
# 加载后重新 fit
clf_new = LogisticRegression(**params)
clf_new.fit(X_train, y_train)
```

### 12.3 joblib 的优势

sklearn 推荐用 `joblib` 代替 pickle，对含大 ndarray 的估计器更高效：

```python
from joblib import dump, load

dump(clf, "model.joblib")
clf_loaded = load("model.joblib")
```

joblib 用压缩 + 内存映射，对大模型加载更快。

### 12.4 思考题

1. 为什么 pickle 不跨版本兼容？哪些因素导致不兼容？
2. `clone(clf)` 和 `pickle.loads(pickle.dumps(clf))` 有什么区别？
3. 保存模型时，要不要同时保存 sklearn 版本？怎么保存？

---

## 13. 多输出与多标签的约定

### 13.1 多输出（multi-output）

每个样本有多个连续目标：

```python
from sklearn.multioutput import MultiOutputRegressor

X = np.random.randn(100, 5)
y = np.random.randn(100, 3)   # 3 个连续目标
multi = MultiOutputRegressor(LinearRegression())
multi.fit(X, y)
multi.predict(X[:5]).shape   # (5, 3)
```

### 13.2 多标签（multi-label）

每个样本可属于多个类别：

```python
from sklearn.multioutput import MultiOutputClassifier

y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])   # 3 个样本，3 个标签
multi_clf = MultiOutputClassifier(LogisticRegression())
multi_clf.fit(X, y)
multi_clf.predict(X[:2])   # shape (2, 3)，每行是 0/1 数组
```

### 13.3 形状约定总结

| 任务          | `y` 形状                | dtype    | 例子                |
|---------------|--------------------------|----------|----------------------|
| 二分类        | `(n_samples,)`           | int/str  | `[0, 1, 1, 0]`       |
| 多分类        | `(n_samples,)`           | int/str  | `[0, 1, 2, 1]`       |
| 回归          | `(n_samples,)`           | float    | `[2.3, 4.1, 5.0]`    |
| 多输出回归    | `(n_samples, n_outputs)` | float    | `[[1.5, 2.3], ...]`  |
| 多标签分类    | `(n_samples, n_labels)`  | int 0/1  | `[[1, 0, 1], ...]`   |

### 13.4 思考题

1. 多输出和多标签的 `y` 形状一样，sklearn 怎么区分？
2. `MultiOutputRegressor` 内部是怎么工作的？它创建了几个子估计器？
3. 为什么不直接让所有估计器支持多输出，而要用包装器？

---

## 14. 实战：写一个带完整校验的估计器

把这一讲的所有约定用起来：

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardizedLinearRegression(BaseEstimator, RegressorMixin):
    """先标准化 X 再做线性回归。"""

    def __init__(self, alpha=0.0):
        self.alpha = alpha   # 超参数，无下划线

    def fit(self, X, y):
        # 1. 入口校验
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X 样本数 {X.shape[0]} != y 样本数 {y.shape[0]}"
            )

        # 2. 记录输入特征数
        self.n_features_in_ = X.shape[1]

        # 3. 标准化（学出属性）
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X_centered = (X - self.mean_) / self.std_

        # 4. 解正规方程
        n = X.shape[0]
        XtX = X_centered.T @ X_centered + n * self.alpha * np.eye(self.n_features_in_)
        Xty = X_centered.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)
        self.intercept_ = y.mean() - (self.coef_ * self.mean_ / self.std_).sum()
        return self

    def predict(self, X):
        # 1. 检查已拟合
        check_is_fitted(self, ['coef_', 'mean_', 'std_'])

        # 2. 入口校验 + 特征数检查
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X 有 {X.shape[1]} 个特征，但模型是用 {self.n_features_in_} 个特征训练的"
            )

        # 3. 标准化 + 预测
        X_centered = (X - self.mean_) / self.std_
        return X_centered @ self.coef_ + self.intercept_
```

这个例子体现了：

- 维度约定 `(n_samples, n_features)`。
- 标签约定 `y` 一维。
- `check_array` 入口校验。
- `check_is_fitted` 防未训练。
- 下划线约定：`coef_` / `mean_` / `std_` / `n_features_in_`。
- `n_features_in_` 一致性检查。

---

## 12. 小结

| 约定          | 内容                                  | 理由                         |
|---------------|----------------------------------------|------------------------------|
| `X.shape`     | `(n_samples, n_features)`              | NumPy 惯例、内存布局、数学记号 |
| `y.shape`     | `(n_samples,)` 或 `(n_samples, n_outputs)` | 简洁、与 pandas 一致          |
| 入口校验      | `check_array` / `check_X_y`            | 错误前置、信息清晰           |
| 下划线结尾    | `coef_` / `classes_`                   | 区分超参数与学出参数         |
| `n_features_in_` | fit 时记录，predict 时检查          | 防止特征数不匹配             |
| dtype         | float64 for X, int64 for class y      | 算法统一、精度足够           |

**核心洞察**：sklearn 的数据约定是一套**防御性编程**体系——通过在入口处校验，把数据错误前置到最清晰的位置，避免晦涩的深层报错。这套约定看似琐碎，却是"统一 API 契约"能落地的物理基础：只有所有算法假设同样的数据形态，才能用同一套校验、同一套元估计器、同一套测试套件。

---

## 15. 校验的进阶模式

### 15.1 延迟校验

对于已知干净的数据（例如刚从 `StandardScaler` 出来的 `X`），重复校验是浪费。sklearn 内部有些路径会跳过：

```python
# Pipeline 内部，前一步的输出已知是 float64、无 NaN
# 后一步可以跳过部分校验
```

### 15.2 增量校验

对于流式数据，可以只校验新到的批次：

```python
class StreamingChecker:
    def __init__(self):
        self.n_features_in_ = None

    def check_batch(self, X_batch):
        X_batch = check_array(X_batch, ensure_2d=True)
        if self.n_features_in_ is None:
            self.n_features_in_ = X_batch.shape[1]
        elif X_batch.shape[1] != self.n_features_in_:
            raise ValueError("特征数变了")
        return X_batch
```

### 15.3 校验与类型提示的结合

新版 sklearn 开始用类型提示补充运行时校验：

```python
from typing import ArrayLike

def fit(self, X: ArrayLike, y: ArrayLike) -> "Self":
    X = check_array(X)   # 运行时校验
    ...
```

类型提示给静态检查工具（mypy）用，运行时校验给实际执行用，两者互补。

### 15.4 校验失败时的恢复策略

校验报错后，用户可能想要"自动修复"而不是放弃。sklearn 一般不自动修复（避免静默错误），但你可以写包装器：

```python
def auto_fix_and_fit(clf, X, y):
    try:
        return clf.fit(X, y)
    except ValueError as e:
        if "NaN" in str(e):
            from sklearn.impute import SimpleImputer
            X = SimpleImputer().fit_transform(X)
            return clf.fit(X, y)
        raise
```

这种"自动修复"要谨慎，容易掩盖真正的问题。

---

## 16. 练习

### 15.1 基础练习

1. 写一个 `check_positive_array`，要求输入非负、二维、float64。
2. 实现一个 `MeanEstimator`，`fit` 计算每个特征均值，`predict` 永远返回均值。要求带完整校验和 `check_is_fitted`。
3. 比较 `check_array(np.array([[1, 2], [3, 4]]), copy=True)` 和 `copy=False` 的内存地址，验证是否复制。
4. 构造一个含 NaN 的 `X`，分别用有校验和无校验的 `fit` 处理，对比报错信息。
5. 写一个测试，验证 `clf.predict(X_wrong_features)` 会报 `n_features_in_` 不匹配错误。

### 15.2 进阶练习

6. 实现一个 `check_pairwise_distance_matrix(D)`，校验输入是方阵、对称、对角线为 0、非负。
7. 写一个 `MemoryEfficientScaler`，对大数据分批标准化，不一次性载入内存。
8. 实现一个 `RobustLinearRegression`，对 NaN 用中位数填充后再回归，要求在 `fit` 里完成填充。
9. 比较 `check_array` 在 `accept_sparse=True` 和 `False` 下对同一稀疏矩阵的行为。
10. 写一个 benchmark，测量 `check_array` 在不同数据规模下的开销，画出开销占比曲线。

### 15.3 思考题

11. 如果让你设计一个支持 GPU 张量的 sklearn 变体，数据约定要怎么改？校验要怎么改？
12. 为什么 sklearn 不接受 `pandas.DataFrame` 的字符串列直接进线性模型？应该怎么处理？
13. `check_array` 的 `ensure_min_samples=1` 默认值合理吗？什么场景下应该要求更多？
14. 如果 `X` 是内存映射的 `np.memmap`，`check_array` 会复制吗？怎么避免复制？
15. sklearn 的数据约定有哪些地方和深度学习框架冲突？如果要做"sklearn + PyTorch"混合，怎么调和？

---

## 上一讲 / 下一讲

[← 第四讲：元估计器模式](04-meta-estimator.md) ｜  [第六讲：一致性测试机制 →](06-consistency-testing.md）
