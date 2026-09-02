# 模型选择：交叉验证与网格搜索

> 模型选择回答两个问题：**这个模型好不好？**（交叉验证）**哪个参数最好？**（网格搜索）。它们是元算法的典型应用——不自己学习数据，而是驱动其他估计器学习。

---

## 一、train_test_split：训练/测试集划分

### 1.1 为什么需要划分？

如果用全部数据训练，再用同一批数据评估，评估分数会**乐观偏误**——模型可能只是记住了数据，而非学到了规律。

划分方案：

```
全部数据 ──→ 训练集（fit）  ──→ 学到参数
          └→ 测试集（score） ──→ 评估泛化能力
```

测试集在训练时**不可见**，模拟"未来遇到的新数据"。

更深的道理：监督学习的目标是泛化——在新数据上表现好。训练集上的误差叫**训练误差**，测试集上的叫**测试误差**（泛化误差的估计）。当模型容量足够大时，可以记住训练集使训练误差为 0，但测试误差可能很高（过拟合）。划分测试集让我们能估计真实的泛化能力。

### 1.2 实现

```python
def train_test_split(*arrays, test_size=0.25, random_state=None,
                      shuffle=True, stratify=None):
    n_samples = arrays[0].shape[0]
    rng = check_random_state(random_state)

    if shuffle:
        indices = rng.permutation(n_samples)  # 随机打乱
    else:
        indices = np.arange(n_samples)         # 原序

    n_test = int(n_samples * test_size)
    train_indices = indices[:n_samples - n_test]
    test_indices = indices[n_samples - n_test:]

    # 对每个输入数组按索引切分
    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    return result
```

逐行解析：
- `*arrays`：可变参数，允许同时切分 X 和 y（甚至更多数组）
- `check_random_state`：把 int/None 转成 `np.random.RandomState`，保证可复现
- `rng.permutation`：生成打乱的索引，不复制数据
- `int(n_samples * test_size)`：向下取整，训练集多一个样本

### 1.3 分层抽样（stratify）

当类别不平衡时（如 90% 正样本、10% 负样本），随机划分可能导致测试集没有负样本。分层抽样保证训练集和测试集的类别比例与原数据一致：

```python
# 按类别比例划分
for cls in classes:
    cls_indices = np.where(y == cls)[0]
    rng.shuffle(cls_indices)
    n_cls_train = int(len(cls_indices) * n_train / len(y))
    train_indices.extend(cls_indices[:n_cls_train])
```

为什么重要？不平衡数据下，随机划分可能让少数类在测试集消失，无法评估对少数类的识别能力。分层抽样保证每类都有代表。

```python
# 不分层：可能测试集没负样本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(np.bincount(y_test))  # 可能 [200, 0]

# 分层：测试集类别比例与原数据一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(np.bincount(y_test))  # [180, 20]，比例 9:1
```

### 1.4 使用示例

```python
import numpy as np
from minisklearn.model_selection import train_test_split

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# 基础划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)  # (80, 5) (20, 5)

# 不打乱（时间序列数据）
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
# 前 80% 训练，后 20% 测试

# 分层
y_imbalanced = np.array([0] * 90 + [1] * 10)
X_train, X_test, y_train, y_test = train_test_split(X, y_imbalanced, test_size=0.2, stratify=y_imbalanced)
print(np.bincount(y_train), np.bincount(y_test))  # [72 8] [18 2]
```

### 1.5 常见陷阱

```python
# 错误 1：划分前没打乱（数据有序）
X = np.vstack([class0_data, class1_data])  # 前 500 是类 0，后 500 是类 1
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
# 测试集全是类 1！必须 shuffle=True

# 错误 2：用不同 random_state 切 X 和 y
X_train, X_test = train_test_split(X, random_state=1)
y_train, y_test = train_test_split(y, random_state=2)  # ← 索引不对齐！
# 正确：一次切两个
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 错误 3：测试集太小，评估方差大
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
# 只有 1 个测试样本，分数非 0 即 1，没意义
```

### 1.6 test_size 的选择

| 数据量 | 推荐 test_size | 说明 |
|--------|---------------|------|
| < 100 | 0.2 ~ 0.3 | 测试集要够多评估 |
| 100 ~ 1000 | 0.2 | 常用 |
| 1000 ~ 10000 | 0.2 | 常用 |
| > 100000 | 0.1 或更小 | 训练集够大，测试集 1% 也够 |

经验法则：测试集至少 100-1000 个样本，且覆盖各类别。

---

## 二、KFold：K 折交叉验证

### 2.1 原理

单次划分的评估方差大（运气好分数高，运气差分数低）。K 折交叉验证把数据分成 K 份，每次用 K-1 份训练、1 份测试，循环 K 次取平均：

```
K=5 时：

第1折：[测试] [训练] [训练] [训练] [训练]
第2折：[训练] [测试] [训练] [训练] [训练]
第3折：[训练] [训练] [测试] [训练] [训练]
第4折：[训练] [训练] [训练] [测试] [训练]
第5折：[训练] [训练] [训练] [训练] [测试]

最终分数 = mean(5 次分数)
```

**优点**：每个样本都参与过测试，评估更稳定。

**为什么有效**：单次划分只用了部分数据测试，受随机性影响大。K 折让每个样本都当过一次测试样本，平均后方差降低约 $\sqrt{K}$ 倍。同时每次训练用了 $(K-1)/K$ 的数据，比 holdout（固定划分）用了更多训练数据，模型训练更充分。

**K 的选择**：

| K | 特点 |
|---|------|
| 5 | 常用，偏差/方差平衡 |
| 10 | 常用，方差更小 |
| n（留一法） | 方差最小但计算最贵 |

偏差-方差权衡：
- K 小（如 2）：训练集小，模型欠训练，评估偏差大
- K 大（如 n）：训练集接近全量，偏差小；但各折训练集高度重叠，方差大
- K=5 或 10 是经验上的甜点

### 2.2 实现

```python
class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)
        indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)

        # 各折大小可能不等（n 不整除 K 时）
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1  # 前几折多分一个

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
```

**细节**：当 $n$ 不整除 $K$ 时，前 $n \mod K$ 折各多分一个样本，保证不遗漏。例如 $n=103, K=5$：折大小 $[21, 21, 21, 20, 20]$，$103 = 21 \times 3 + 20 \times 2$。

### 2.3 生成器模式

`split` 用 `yield` 返回生成器，而非一次性返回列表。好处是**惰性求值**——K 很大时不需要同时存储所有折的索引。

```python
# 生成器：惰性
for train_idx, test_idx in KFold(n_splits=1000).split(X):
    # 每次只生成一折，内存省
    pass

# 列表：一次性
all_folds = list(KFold(n_splits=1000).split(X))  # 1000 折全在内存
```

### 2.4 使用示例

```python
from minisklearn.model_selection import KFold
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"折 {fold}: 训练 {len(train_idx)}, 测试 {len(test_idx)}")
# 折 0: 训练 80, 测试 20
# 折 1: 训练 80, 测试 20
# ...
```

### 2.5 变体

**StratifiedKFold**：分层 K 折，每折的类别比例与原数据一致。分类任务必用。

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):  # 注意要传 y
    print(np.bincount(y[train_idx]), np.bincount(y[test_idx]))
```

**RepeatedKFold**：重复 K 折多次（不同随机种子），进一步降低方差。

**LeaveOneOut**：K=n 的特例，每折留一个样本测试。方差小但计算贵，且测试集只 1 个样本使分数波动大。

**GroupKFold**：按组分折，同组样本要么全在训练集要么全在测试集。用于避免数据泄露（如同一患者的多个样本不能跨训练/测试）。

**TimeSeriesSplit**：时间序列交叉验证，训练集只能用过去数据，测试集是未来数据：

```
折 1: 训练 [1]       测试 [2]
折 2: 训练 [1,2]     测试 [3]
折 3: 训练 [1,2,3]   测试 [4]
```

---

## 三、cross_val_score：交叉验证评分

### 3.1 原理

`cross_val_score` 是 KFold 的便捷封装：自动划分、克隆估计器、训练、评分、返回分数数组。

```python
def cross_val_score(estimator, X, y, cv=5):
    scores = []
    for train_idx, test_idx in cv.split(X):
        est = clone(estimator)          # 关键：克隆干净副本
        est.fit(X[train_idx], y[train_idx])
        score = est.score(X[test_idx], y[test_idx])
        scores.append(score)
    return np.array(scores)
```

### 3.2 为什么必须 clone？

```python
# 错误写法（不 clone）
for train_idx, test_idx in cv.split(X):
    estimator.fit(X[train_idx], y[train_idx])  # ← 污染了原对象！
    scores.append(estimator.score(X[test_idx], y[test_idx]))
```

不 clone 的话，`estimator` 在第一折 fit 后就有了参数，第二折 fit 会覆盖，但如果中间有异常，原对象就处于半训练状态。clone 保证了**原估计器不被修改**，每次评估都是独立的。

更深的原因：元估计器（如 GridSearchCV）会反复驱动同一基础估计器。如果不 clone，前一次 fit 的状态会污染后一次，导致评估不独立、结果不可复现。

### 3.3 使用示例

```python
from minisklearn.model_selection import cross_val_score
from minisklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print(f"各折分数: {scores}")
print(f"平均: {scores.mean():.4f} ± {scores.std():.4f}")

# 用不同评分
from sklearn.metrics import make_scorer, f1_score
scores = cross_val_score(LogisticRegression(), X, y, cv=5,
                         scoring=make_scorer(f1_score, average='macro'))
```

### 3.4 cv 参数的灵活指定

```python
# int：默认 KFold
cross_val_score(clf, X, y, cv=5)

# 交叉验证对象：自定义
cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))

# 显式索引列表
cv_indices = [(train_idx, test_idx) for train_idx, test_idx in KFold(5).split(X)]
cross_val_score(clf, X, y, cv=cv_indices)
```

### 3.5 cross_validate（更全的版本）

sklearn 的 `cross_validate` 返回训练分数、多评分、耗时：

```python
from sklearn.model_selection import cross_validate
results = cross_validate(clf, X, y, cv=5,
                         scoring=['accuracy', 'f1'],
                         return_train_score=True)
print(results['test_accuracy'])   # 测试准确率
print(results['train_accuracy'])  # 训练准确率（看是否过拟合）
print(results['fit_time'])        # 每折训练耗时
```

---

## 四、GridSearchCV：网格搜索 + 交叉验证

### 4.1 原理

网格搜索枚举参数空间的所有组合（笛卡尔积），对每个组合用交叉验证评估，选平均分最高的：

```python
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}
# 笛卡尔积 → 3 × 2 = 6 个组合
```

```
组合 1: C=0.1, max_iter=100 → CV 分数 0.82
组合 2: C=0.1, max_iter=200 → CV 分数 0.83
组合 3: C=1,   max_iter=100 → CV 分数 0.88
组合 4: C=1,   max_iter=200 → CV 分数 0.89  ← 最优
组合 5: C=10,  max_iter=100 → CV 分数 0.85
组合 6: C=10,  max_iter=200 → CV 分数 0.86

best_params_ = {'C': 1, 'max_iter': 200}
best_score_  = 0.89
```

### 4.2 完整流程

```python
class GridSearchCV(BaseEstimator):
    def fit(self, X, y):
        for param_combo in product(*param_values):  # 枚举笛卡尔积
            params = dict(zip(param_names, param_combo))

            scores = []
            for train_idx, test_idx in cv.split(X):
                est = clone(self.estimator)      # 克隆基础估计器
                est.set_params(**params)          # 设置当前参数组合
                est.fit(X[train_idx], y[train_idx])
                scores.append(est.score(X[test_idx], y[test_idx]))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        # 用最优参数在全量数据上重训
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**best_params)
        self.best_estimator_.fit(X, y)
```

### 4.3 三个关键设计

**（1）clone × 2**：每折 clone 一次（防止折间污染），最优参数再 clone 一次（在全量数据上重训）。

**（2）set_params 而非 __init__**：用 `clone` + `set_params` 而非 `type(est)(**params)`，因为 `clone` 保留了用户在 `__init__` 中设置的未搜索参数。

**（3）全量重训**：找到最优参数后，用**全量数据**（而非某一折的训练集）重新 fit，得到最终模型。这是交叉验证的标淮流程——CV 只用于选参数，最终模型要用尽可能多的数据训练。

### 4.4 搜索 Pipeline 嵌套参数

GridSearchCV 最强大的能力是搜索 Pipeline 内部任意步骤的参数：

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

grid = GridSearchCV(pipe, {
    'clf__C': [0.1, 1, 10],        # ← 嵌套命名
    'clf__max_iter': [100, 200],
}, cv=5)
grid.fit(X, y)
```

`clf__C` 会被 `Pipeline.set_params` 路由到 `LogisticRegression.C`。这依赖 Pipeline 覆盖了 `get_params`/`set_params` 来展平嵌套参数命名。

### 4.5 使用示例

```python
from minisklearn.model_selection import GridSearchCV
from minisklearn.svm import LinearSVC

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [500, 1000, 2000],
}

grid = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)

print(f"最优参数: {grid.best_params_}")
print(f"最优分数: {grid.best_score_:.4f}")
print(f"测试分数: {grid.best_estimator_.score(X_test, y_test):.4f}")

# 查看所有结果
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
print(results[['params', 'mean_test_score', 'std_test_score']])
```

### 4.6 cv_results_ 详解

sklearn 的 GridSearchCV 暴露 `cv_results_`，包含每个参数组合的详细结果：

```python
grid.cv_results_['mean_test_score']   # 各组合的平均 CV 分数
grid.cv_results_['std_test_score']    # 标准差
grid.cv_results_['params']            # 参数组合
grid.cv_results_['rank_test_score']   # 排名
```

### 4.7 复杂度

| 参数 | 值 |
|------|-----|
| 参数组合数 | $\prod_i |param_i|$（笛卡尔积） |
| CV 折数 | $K$ |
| 总训练次数 | $\prod_i |param_i| \times K$ |
| 单次训练复杂度 | 取决于估计器 |

例如 3 个参数各 5 个值，5 折 CV：$5 \times 5 \times 5 \times 5 = 625$ 次训练。GridSearchCV 很快变贵。

### 4.8 替代：RandomizedSearchCV

参数空间大时，随机搜索更高效：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

grid = RandomizedSearchCV(
    LogisticRegression(),
    {'C': loguniform(1e-3, 1e3)},  # 对数均匀分布
    n_iter=20,  # 只试 20 个随机组合
    cv=5,
)
```

随机搜索的优势：在相同预算下，能探索更多"重要"参数（对结果影响大的参数），而网格搜索在"不重要"参数上浪费很多组合。

---

## 五、架构回扣

### 5.1 元估计器模式

`GridSearchCV` 和 `Pipeline` 都是**元估计器**——它们包装其他估计器，自身也是估计器：

```
GridSearchCV(estimator=LogisticRegression(), ...)
    → 有 fit / predict / score → 是估计器
    → 内部驱动 LogisticRegression → 是元估计器
```

这意味着 GridSearchCV 可以嵌套：

```python
# Pipeline 里套 GridSearchCV？或 GridSearchCV 里套 Pipeline？
# 都可以！因为它们都是 BaseEstimator。
grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
    {'clf__C': [0.1, 1, 10]},
)
```

### 5.2 clone 是元估计器的基石

整个 model_selection 模块的核心依赖是 `clone`：

| 函数 | clone 的作用 |
|------|-------------|
| `cross_val_score` | 每折 clone，防止原估计器被污染 |
| `GridSearchCV.fit` | 每个参数组合 × 每折 clone，保证独立评估 |

没有 clone，元估计器就无法安全地反复驱动基础估计器。

### 5.3 与架构设计的联系

- **统一 API**：GridSearchCV 有 `fit`/`predict`/`score`，用法和普通估计器一样
- **参数管理**：GridSearchCV 通过 `get_params`/`set_params` 管理自身参数（estimator、param_grid 等）
- **Mixin 设计**：GridSearchCV 继承 `BaseEstimator`，自动获得 `__repr__`、`clone` 支持

---

## 六、数学原理：为什么交叉验证能估计泛化误差

### 6.1 泛化误差的估计

真实泛化误差 $R(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[L(f(x), y)]$，其中 $\mathcal{D}$ 是真实数据分布。我们只有有限样本，无法直接算。

**训练误差**（apparent error）：$\hat{R}_{train}(f) = \frac{1}{n}\sum_i L(f(x_i), y_i)$。这是乐观偏误的（$f$ 是在同样数据上学的）。

**测试误差**：$\hat{R}_{test}(f) = \frac{1}{m}\sum_{j} L(f(x_j), y_j)$，测试集独立于训练集。这是 $R(f)$ 的无偏估计。

### 6.2 K 折 CV 的估计

K 折 CV 估计：

$$
\hat{R}_{CV} = \frac{1}{K}\sum_{k=1}^K \frac{1}{|T_k|}\sum_{i \in T_k} L(f_{-k}(x_i), y_i)
$$

其中 $f_{-k}$ 是用除第 $k$ 折外数据训练的模型，$T_k$ 是第 $k$ 折测试集。

**偏差**：每次训练用 $(K-1)/K$ 的数据，比全量少，所以 $f_{-k}$ 比全量训练的 $f$ 略差，$\hat{R}_{CV}$ 略偏大（悲观偏误）。K 越大偏差越小。

**方差**：各折训练集高度重叠（共享 $K-2$ 份），$f_{-k}$ 之间相关，使 $\hat{R}_{CV}$ 方差大。K 越大方差越大。

**LOO（K=n）**：偏差最小（训练用 $n-1$ 个样本），但方差最大（$n$ 个模型几乎相同）。

### 6.3 .632 估计

LOO 的悲观偏误可以用 .632 bootstrap 修正：

$$
\hat{R}_{.632} = 0.368 \hat{R}_{apparent} + 0.632 \hat{R}_{boot}
$$

sklearn 没直接实现，但了解原理有助于解释 CV 分数与测试分数的差异。

---

## 七、实现细节

### 7.1 check_random_state

```python
def check_random_state(seed):
    if seed is None:
        return np.random.mtrand._rand  # 全局随机状态
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError
```

这把 int/None/RandomState 统一成 RandomState，保证可复现。

### 7.2 笛卡尔积的生成

```python
from itertools import product

param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}
param_names = list(param_grid.keys())
param_values = list(param_grid.values())

for combo in product(*param_values):
    params = dict(zip(param_names, combo))
    print(params)
# {'C': 0.1, 'max_iter': 100}
# {'C': 0.1, 'max_iter': 200}
# ...
```

### 7.3 评分的处理

```python
# 默认用估计器的 score 方法
score = est.score(X_test, y_test)

# 自定义评分
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, average='macro')
score = scorer(est, X_test, y_test)
```

### 7.4 并行化

```python
# sklearn 用 joblib 并行
grid = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)  # 用所有 CPU
```

每折/每个参数组合独立，天然可并行。minisklearn 可选实现。

---

## 八、使用示例

### 8.1 完整调参流程

```python
import numpy as np
from minisklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from minisklearn.svm import LinearSVC
from minisklearn.preprocessing import StandardScaler
from minisklearn.pipeline import Pipeline

# 1. 加载数据
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 构建流水线
pipe = Pipeline([('scaler', StandardScaler()), ('svm', LinearSVC())])

# 3. 网格搜索
grid = GridSearchCV(pipe, {'svm__C': [0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(X_train, y_train)

# 4. 评估
print(f"最优参数: {grid.best_params_}")
print(f"CV 分数: {grid.best_score_:.4f}")
print(f"测试分数: {grid.score(X_test, y_test):.4f}")
```

### 8.2 嵌套交叉验证

要无偏估计调参后的泛化能力，用嵌套 CV：外层评估，内层调参。

```python
outer_cv = KFold(n_splits=5)
inner_cv = KFold(n_splits=3)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X):
    grid = GridSearchCV(clf, param_grid, cv=inner_cv)
    grid.fit(X[train_idx], y[train_idx])
    score = grid.score(X[test_idx], y[test_idx])
    nested_scores.append(score)

print(f"嵌套 CV: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
```

非嵌套 CV（普通 GridSearchCV 的 best_score_）会乐观偏误，因为用了同一份数据选参数和评估。

### 8.3 多评分

```python
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

scoring = {
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro'),
}
grid = GridSearchCV(clf, param_grid, cv=5, scoring=scoring, refit='f1')
grid.fit(X, y)
```

### 8.4 错误示例

```python
# 错误 1：用测试集调参
grid.fit(X_test, y_test)  # ← 数据泄露！测试集参与了选参数
# 正确：grid.fit(X_train, y_train)，然后用 X_test 评估

# 错误 2：CV 分数当真实泛化
print(grid.best_score_)  # 这是 CV 分数，乐观偏误
# 真实泛化要用独立测试集
print(grid.score(X_test, y_test))

# 错误 3：参数空间太粗
grid = GridSearchCV(clf, {'C': [0.1, 10]}, cv=5)  # 跳过了 1
# 最优可能在 C=1，但没搜到

# 错误 4：参数空间太细（过拟合验证集）
grid = GridSearchCV(clf, {'C': np.linspace(0.1, 10, 1000)}, cv=5)
# 搜 1000 个参数，best_score_ 会乐观偏误（验证集过拟合）
```

---

## 九、与 sklearn 对比

### 9.1 API 一致性

| 特性 | minisklearn | sklearn |
|------|------------|---------|
| `train_test_split` | ✓ | ✓ |
| `KFold` | ✓ | ✓ |
| `StratifiedKFold` | ✗ | ✓ |
| `cross_val_score` | ✓ | ✓ |
| `cross_validate` | ✗ | ✓ |
| `GridSearchCV` | ✓ | ✓ |
| `RandomizedSearchCV` | ✗ | ✓ |
| `cv_results_` | ✗ | ✓ |
| `n_jobs` 并行 | ✗ | ✓ |
| `HalvingGridSearchCV` | ✗ | ✓（连续减半） |

### 9.2 功能差异

```python
# sklearn 的 GridSearchCV 功能更全
grid = GridSearchCV(
    clf, param_grid,
    cv=5,
    scoring='f1_macro',     # 多评分
    n_jobs=-1,              # 并行
    verbose=2,              # 日志
    refit=True,             # 自动用最优参数重训
    return_train_score=True,
)
```

minisklearn 实现核心功能，sklearn 是生产级。

---

## 十、复杂度分析

### 10.1 GridSearchCV

| 项 | 复杂度 |
|----|--------|
| 参数组合 | $P = \prod_i |param_i|$ |
| CV 折数 | $K$ |
| 总训练次数 | $P \times K$ |
| 总复杂度 | $O(P \times K \times T_{fit})$ |

### 10.2 cross_val_score

$O(K \times T_{fit})$，K 折各训练一次。

### 10.3 实测

```python
import numpy as np, time
from minisklearn.model_selection import GridSearchCV
from minisklearn.svm import LinearSVC

X, y = np.random.randn(1000, 20), np.random.randint(0, 2, 1000)
for P in [10, 100, 1000]:
    param_grid = {'C': np.logspace(-3, 3, P)}
    t0 = time.time()
    GridSearchCV(LinearSVC(max_iter=100), param_grid, cv=5).fit(X, y)
    print(f"P={P}: {time.time()-t0:.3f}s")
```

---

## 十一、常见问题与陷阱

### 11.1 数据泄露

```python
# 错误：标准化用全部数据，再划分
scaler = StandardScaler().fit(X)          # ← 用了全部数据
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)  # 测试集统计信息泄露

# 正确：先划分，再在训练集上 fit
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 最安全：用 Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
cross_val_score(pipe, X, y, cv=5)  # Pipeline 保证 fit/transform 不泄露
```

### 11.2 用测试集调参

```python
# 错误
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid.fit(X_test, y_test)  # ← 测试集调参，泄露

# 正确
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
```

### 11.3 CV 分数乐观偏误

`grid.best_score_` 是 CV 分数，用了训练数据选参数，乐观偏误。真实泛化要用独立测试集。

### 11.4 K 太大或太小

- K 太小（如 2）：训练集小，偏差大
- K 太大（如 LOO）：方差大，计算贵
- 经验：5 或 10

### 11.5 不分层导致少数类消失

```python
# 不平衡数据，不分层
scores = cross_val_score(clf, X, y, cv=KFold(5))  # 某折可能没少数类
# 用 StratifiedKFold
scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5))
```

### 11.6 参数空间设计

- 对数尺度搜正则化参数（C、alpha）：`np.logspace(-3, 3, 7)`
- 线性尺度搜迭代数：`[100, 200, 500, 1000]`
- 不要太细（验证集过拟合）

---

## 十二、进阶话题

### 12.1 学习曲线与验证曲线

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
# 画训练/测试分数随训练量变化，判断是否过拟合/欠拟合
```

### 12.2 validation_curve

```python
from sklearn.model_selection import validation_curve
param_range = np.logspace(-3, 3, 7)
train_scores, test_scores = validation_curve(
    clf, X, y, param_name='C', param_range=param_range, cv=5)
# 画分数随某参数变化，看参数影响
```

### 12.3 贝叶斯优化

网格/随机搜索不利用历史评估信息。贝叶斯优化用高斯过程建模参数-分数关系，智能选下一个参数：

```python
from skopt import BayesSearchCV
grid = BayesSearchCV(clf, {'C': (1e-3, 1e3, 'log-uniform')}, n_iter=20, cv=5)
```

### 12.4 HalvingGridSearchCV

sklearn 的连续减半搜索：先粗筛（少资源），再精筛（多资源），比全网格快很多。

### 12.5 Optuna

更现代的超参数优化框架，支持剪枝、多目标：

```python
import optuna
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    clf = LogisticRegression(C=C)
    return cross_val_score(clf, X, y, cv=5).mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 十四、交叉验证的统计学视角

### 14.1 CV 估计的方差分解

K 折 CV 估计 $\hat{R}_{CV}$ 的方差可以分解为两部分：

$$
\mathrm{Var}(\hat{R}_{CV}) = \frac{1}{K^2}\left[ \sum_k \mathrm{Var}(\hat{R}_k) + 2\sum_{k < l} \mathrm{Cov}(\hat{R}_k, \hat{R}_l) \right]
$$

各折训练集重叠使 $\mathrm{Cov}(\hat{R}_k, \hat{R}_l) > 0$，这是 K 大时方差大的根源。LOO 极端情况：所有折训练集只差一个样本，协方差接近方差，总方差几乎不随 K 减小。

### 14.2 偏差-方差权衡的量化

设全量训练的误差为 $R_n$，K 折训练（用 $(K-1)/K \cdot n$ 样本）的误差为 $R_{(K-1)n/K}$。学习曲线 $R_m$ 随 $m$ 递减，所以 $R_{(K-1)n/K} > R_n$，CV 估计偏大。

偏差约：

$$
\text{bias} \approx R_{(K-1)n/K} - R_n \approx \frac{a}{(K-1)n/K} - \frac{a}{n} = \frac{a}{n}\left(\frac{K}{K-1} - 1\right) = \frac{a}{n(K-1)}
$$

K 越大偏差越小，K=5 时偏差约 $a/(4n)$，K=10 时约 $a/(9n)$。

### 14.3 何时 CV 失效

- **数据有时间结构**：随机打乱破坏时间顺序，用 TimeSeriesSplit
- **数据有组结构**：同组样本跨训练/测试会泄露，用 GroupKFold
- **数据极少**：CV 方差大，考虑 bootstrap
- **类别极不平衡**：普通 KFold 可能让某折没少数类，用 StratifiedKFold

---

## 十五、更多代码示例

### 15.1 手动实现 K 折

```python
import numpy as np

def kfold_indices(n, k, shuffle=False, random_state=None):
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    folds = []
    cur = 0
    for fs in fold_sizes:
        folds.append(idx[cur:cur+fs])
        cur += fs
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test

for train, test in kfold_indices(10, 3, shuffle=True, random_state=0):
    print(f"train={train}, test={test}")
```

### 15.2 手动实现 cross_val_score

```python
from minisklearn.base import clone

def cross_val_score_scratch(estimator, X, y, cv=5):
    scores = []
    n = len(y)
    for train_idx, test_idx in kfold_indices(n, cv, shuffle=True):
        est = clone(estimator)  # 关键：克隆
        est.fit(X[train_idx], y[train_idx])
        scores.append(est.score(X[test_idx], y[test_idx]))
    return np.array(scores)
```

### 15.3 手动实现 GridSearchCV

```python
from itertools import product

def grid_search_scratch(estimator, param_grid, X, y, cv=5):
    best_score, best_params = -np.inf, None
    names = list(param_grid.keys())
    for combo in product(*param_grid.values()):
        params = dict(zip(names, combo))
        scores = []
        for train_idx, test_idx in kfold_indices(len(y), cv, shuffle=True):
            est = clone(estimator)
            est.set_params(**params)
            est.fit(X[train_idx], y[train_idx])
            scores.append(est.score(X[test_idx], y[test_idx]))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    # 全量重训
    best_est = clone(estimator)
    best_est.set_params(**best_params)
    best_est.fit(X, y)
    return best_est, best_params, best_score
```

### 15.4 分层 K 折手动实现

```python
def stratified_kfold_indices(y, k, shuffle=False, random_state=None):
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    if shuffle:
        for c in classes:
            rng.shuffle(class_indices[c])
    # 每类按 k 折切分，再合并
    folds = [[] for _ in range(k)]
    for c in classes:
        cls_idx = class_indices[c]
        for i, idx in enumerate(cls_idx):
            folds[i % k].append(idx)
    folds = [np.array(f) for f in folds]
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test
```

### 15.5 可视化 CV 划分

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_cv(cv, X, y):
    n = len(y)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        for i in range(n):
            if i in test:
                ax.scatter(fold, i, c='red', marker='s')
            else:
                ax.scatter(fold, i, c='blue', marker='s')
    ax.set_xlabel('折')
    ax.set_ylabel('样本')
    ax.set_title('CV 划分可视化')
    plt.show()

from sklearn.model_selection import KFold, StratifiedKFold
plot_cv(KFold(5), X, y)
plot_cv(StratifiedKFold(5), X, y)
```

### 15.6 比较不同 CV 策略

```python
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit

for name, cv in [('KFold(5)', KFold(5)),
                 ('StratifiedKFold(5)', StratifiedKFold(5)),
                 ('LOO', LeaveOneOut()),
                 ('ShuffleSplit', ShuffleSplit(n_splits=10, test_size=0.2))]:
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 十六、实际使用教程

### 16.1 标准调参流程

```python
# 1. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 粗搜
coarse_grid = {'C': [0.001, 0.1, 1, 10, 1000]}
grid1 = GridSearchCV(clf, coarse_grid, cv=5).fit(X_train, y_train)
print(f"粗搜最优: {grid1.best_params_}")

# 3. 细搜（在粗搜最优附近）
best_C = grid1.best_params_['C']
fine_grid = {'C': np.linspace(best_C / 10, best_C * 10, 10)}
grid2 = GridSearchCV(clf, fine_grid, cv=5).fit(X_train, y_train)
print(f"细搜最优: {grid2.best_params_}")

# 4. 评估
print(f"测试分数: {grid2.score(X_test, y_test):.4f}")
```

### 16.2 用 Pipeline 防泄露

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])
# GridSearchCV 搜 Pipeline 参数，CV 时每折内部 fit/transform 不泄露
grid = GridSearchCV(pipe, {'clf__C': [0.1, 1, 10]}, cv=5)
grid.fit(X, y)
```

### 16.3 评估多个模型

```python
models = {
    'LR': LogisticRegression(),
    'SVM': LinearSVC(),
    'NB': GaussianNB(),
}
for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 16.4 报告生成

```python
import pandas as pd

results = []
for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    results.append({'model': name, 'mean': scores.mean(), 'std': scores.std()})
df = pd.DataFrame(results).sort_values('mean', ascending=False)
print(df)
```

---

## 十七、总结

| 要点 | 内容 |
|------|------|
| train_test_split | 简单划分，测试集不可见于训练 |
| KFold | K 折循环，每个样本当过测试，方差小 |
| cross_val_score | KFold 便捷封装，必须 clone |
| GridSearchCV | 笛卡尔积搜索 + CV，全量重训 |
| clone | 元估计器基石，防止污染 |
| 嵌套参数 | `step__param` 路由到 Pipeline 内部 |
| 数据泄露 | Pipeline 防泄露，测试集不参与调参 |
| 复杂度 | $O(P \times K \times T_{fit})$ |
| 与 sklearn | 核心一致，sklearn 功能更全 |
| 进阶 | 嵌套 CV、贝叶斯优化、Halving |
| 偏差-方差 | K 大偏差小方差大，K=5/10 平衡 |
| 分层 | 不平衡数据必用 StratifiedKFold |

---

## 十八、深入技术分析：交叉验证的偏差-方差分解

### 18.1 CV 估计的偏差来源

K 折 CV 估计 $\hat{R}_{CV}$ 对真实泛化误差 $R(f)$ 的偏差主要来自训练集大小：

$$
\text{bias}(\hat{R}_{CV}) = R_{(K-1)n/K} - R_n \approx \frac{a}{n(K-1)}
$$

其中 $R_m$ 是用 $m$ 个样本训练的模型的误差，$a$ 是学习曲线的系数。这意味着：

- K=2：偏差约 $a/n$（训练集只有一半，欠训练严重）
- K=5：偏差约 $a/(4n)$
- K=10：偏差约 $a/(9n)$
- K=n（LOO）：偏差约 0（训练集只少一个样本）

### 18.2 CV 估计的方差来源

方差来自各折训练集的重叠：

$$
\text{Var}(\hat{R}_{CV}) = \frac{1}{K^2}\left[ \sum_k \text{Var}(\hat{R}_k) + 2\sum_{k<l} \text{Cov}(\hat{R}_k, \hat{R}_l) \right]
$$

各折训练集共享 $K-2$ 份数据，协方差为正。K 越大，重叠越多，协方差越大，总方差越大。LOO 极端情况：所有折训练集只差一个样本，方差几乎不随 K 减小。

### 18.3 偏差-方差权衡曲线

```
K:    2 ──── 5 ──── 10 ──── n (LOO)
偏差:  高      中       低       最低
方差:  低      中       高       最高
总误差:高      最低     较高     高
```

K=5 或 10 是经验上的甜点。

### 18.4 修正方法：.632 和 .632+ bootstrap

LOO 的悲观偏误可用 bootstrap 修正：

$$
\hat{R}_{.632} = 0.368 \hat{R}_{\text{apparent}} + 0.632 \hat{R}_{\text{boot}}
$$

0.632 来自 bootstrap 中每个样本出现在训练集的概率 $1 - (1-1/n)^n \approx 1 - e^{-1} \approx 0.632$。

### 18.5 重复 K 折降低方差

```python
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
# 50 次划分，方差更小，但计算贵 10 倍
scores = cross_val_score(clf, X, y, cv=cv)
```

每次用不同随机种子划分，平均后方差降低约 $\sqrt{n\_repeats}$ 倍。

---

## 十九、对比实验：不同 CV 策略

### 19.1 实验设计

```python
import numpy as np
from minisklearn.linear_model import LogisticRegression
from minisklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, ShuffleSplit, RepeatedKFold

rng = np.random.RandomState(42)
X = rng.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

clf = LogisticRegression()
cv_strategies = {
    'KFold(5)': KFold(5, shuffle=True, random_state=42),
    'KFold(10)': KFold(10, shuffle=True, random_state=42),
    'StratifiedKFold(5)': StratifiedKFold(5, shuffle=True, random_state=42),
    'ShuffleSplit(20)': ShuffleSplit(n_splits=20, test_size=0.2, random_state=42),
    'RepeatedKFold(5x3)': RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
    'LOO': LeaveOneOut(),
}

for name, cv in cv_strategies.items():
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f} (n={len(scores)})")
```

### 19.2 预期结果分析

| 策略 | 偏差 | 方差 | 计算量 | 适用 |
|------|------|------|--------|------|
| KFold(5) | 中 | 中 | 5x | 通用 |
| KFold(10) | 低 | 较高 | 10x | 需要低偏差 |
| StratifiedKFold | 中 | 中 | 5x | 不平衡数据 |
| ShuffleSplit(20) | 中 | 低 | 20x | 需要稳定估计 |
| RepeatedKFold(5x3) | 中 | 低 | 15x | 需要低方差 |
| LOO | 最低 | 最高 | 200x | 小数据 |

### 19.3 不平衡数据的分层效果

```python
# 90% 类 0，10% 类 1
y_imbalanced = np.array([0] * 180 + [1] * 20)
X_imb = rng.randn(200, 10)

# 不分层：某折可能没有类 1
scores_kf = cross_val_score(clf, X_imb, y_imbalanced, cv=KFold(5, shuffle=True, random_state=0))
# 分层：每折都有类 1
scores_skf = cross_val_score(clf, X_imb, y_imbalanced, cv=StratifiedKFold(5, shuffle=True, random_state=0))

print(f"KFold: {scores_kf.mean():.4f} ± {scores_kf.std():.4f}")
print(f"StratifiedKFold: {scores_skf.mean():.4f} ± {scores_skf.std():.4f}")
```

---

## 二十、参数调优指南：GridSearchCV 的搜索策略

### 20.1 参数空间设计原则

```python
# 原则 1：正则参数用对数尺度
param_grid = {'C': np.logspace(-3, 3, 7)}  # [0.001, 0.01, ..., 1000]

# 原则 2：迭代数用线性尺度
param_grid = {'max_iter': [100, 200, 500, 1000, 2000]}

# 原则 3：树深度用小范围
param_grid = {'max_depth': [3, 5, 7, 10, 15, None]}  # None 表示不限制

# 原则 4：先粗后细
coarse = {'C': [0.01, 1, 100]}
fine = {'C': np.linspace(0.1, 10, 10)}
```

### 20.2 两阶段搜索

```python
# 阶段 1：粗搜
coarse_grid = {'C': [0.001, 0.1, 1, 10, 1000], 'max_iter': [100, 500, 2000]}
grid1 = GridSearchCV(clf, coarse_grid, cv=5).fit(X_train, y_train)
print(f"粗搜最优: {grid1.best_params_}, 分数: {grid1.best_score_:.4f}")

# 阶段 2：在粗搜最优附近细搜
best_C = grid1.best_params_['C']
fine_grid = {
    'C': [best_C/3, best_C, best_C*3],
    'max_iter': [grid1.best_params_['max_iter']],
}
grid2 = GridSearchCV(clf, fine_grid, cv=5).fit(X_train, y_train)
print(f"细搜最优: {grid2.best_params_}, 分数: {grid2.best_score_:.4f}")
```

### 20.3 搜索不同类型的参数

```python
# 搜索算法类型
from minisklearn.linear_model import LogisticRegression
from minisklearn.svm import LinearSVC
from minisklearn.naive_bayes import GaussianNB

models = {
    'LR': (LogisticRegression(), {'C': [0.1, 1, 10]}),
    'SVM': (LinearSVC(), {'C': [0.1, 1, 10]}),
    'NB': (GaussianNB(), {'var_smoothing': [1e-9, 1e-7, 1e-5]}),
}

results = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=5).fit(X_train, y_train)
    results[name] = (grid.best_score_, grid.best_params_)
    
for name, (score, params) in results.items():
    print(f"{name}: {score:.4f} with {params}")
```

### 20.4 避免验证集过拟合

参数空间太细会导致"验证集过拟合"——best_score_ 乐观偏误：

```python
# 危险：搜 10000 个参数组合
grid = GridSearchCV(clf, {'C': np.linspace(0.001, 1000, 10000)}, cv=5)
grid.fit(X_train, y_train)
# best_score_ 会偏高，因为从 10000 个里选最好的，运气成分大
```

**解决**：用嵌套 CV 评估真实泛化能力。

### 20.5 复杂度估算表

| 参数维度 | 每维候选数 | 组合数 | × 5 折 | 训练次数 |
|----------|-----------|--------|--------|----------|
| 1 | 5 | 5 | 25 | 25 |
| 2 | 5 | 25 | 125 | 125 |
| 3 | 5 | 125 | 625 | 625 |
| 3 | 10 | 1000 | 5000 | 5000 |
| 4 | 10 | 10000 | 50000 | 50000 |

组合爆炸很快，3 维以上考虑 RandomizedSearchCV 或贝叶斯优化。

---

## 二十一、常见错误与调试技巧

### 21.1 错误：用测试集调参

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid.fit(X_test, y_test)  # 数据泄露！
```

**正确**：`grid.fit(X_train, y_train)`，用 `X_test` 只做最终评估。

### 21.2 错误：CV 分数当真实泛化

```python
grid = GridSearchCV(clf, param_grid, cv=5).fit(X_train, y_train)
print(grid.best_score_)  # 乐观偏误！用了训练数据选参数
print(grid.score(X_test, y_test))  # 真实泛化
```

### 21.3 错误：不分层导致少数类消失

```python
scores = cross_val_score(clf, X, y, cv=KFold(5))  # 不平衡数据某折可能没少数类
scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5))  # 正确
```

### 21.4 错误：参数空间太粗或太细

```python
# 太粗：跳过最优
GridSearchCV(clf, {'C': [0.1, 10]}, cv=5)  # 最优可能在 1

# 太细：验证集过拟合
GridSearchCV(clf, {'C': np.linspace(0.1, 10, 1000)}, cv=5)
```

### 21.5 调试技巧：查看所有 CV 结果

```python
import pandas as pd
grid = GridSearchCV(clf, param_grid, cv=5).fit(X, y)
results = pd.DataFrame(grid.cv_results_)
print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
      .sort_values('rank_test_score'))
```

### 21.6 调试技巧：可视化参数影响

```python
import matplotlib.pyplot as plt

C_range = np.logspace(-3, 3, 20)
train_scores, test_scores = validation_curve(
    clf, X, y, param_name='C', param_range=C_range, cv=5
)
plt.semilogx(C_range, train_scores.mean(axis=1), label='train')
plt.semilogx(C_range, test_scores.mean(axis=1), label='test')
plt.legend(); plt.xlabel('C'); plt.ylabel('accuracy')
plt.show()
```

### 21.7 调试技巧：学习曲线判断过拟合

```python
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
# train 远高于 test → 过拟合
# 两者都低 → 欠拟合
# 两者接近且高 → 恰好
```

---

## 二十二、实际应用场景

### 22.1 场景：模型选择

```python
models = {
    'LR': LogisticRegression(),
    'SVM': LinearSVC(),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(),
}
for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
# 选 cross_val_score 最高的模型
```

### 22.2 场景：超参数调优报告

```python
grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True).fit(X, y)
results = pd.DataFrame(grid.cv_results_)
# 生成报告：参数 → 训练分数 → 验证分数 → 是否过拟合
results['overfit'] = results['mean_train_score'] - results['mean_test_score']
print(results[['params', 'mean_train_score', 'mean_test_score', 'overfit']])
```

### 22.3 场景：嵌套 CV 评估调参后的模型

```python
outer_cv = KFold(5, shuffle=True, random_state=42)
inner_cv = KFold(3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X):
    grid = GridSearchCV(clf, param_grid, cv=inner_cv)
    grid.fit(X[train_idx], y[train_idx])
    nested_scores.append(grid.score(X[test_idx], y[test_idx]))

print(f"嵌套 CV: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
# 这是调参后模型的无偏估计
```

### 22.4 场景：时间序列 CV

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # 训练集都在测试集之前（时间上）
    assert train_idx.max() < test_idx.min()
```

### 22.5 场景：分组 CV（避免泄露）

```python
from sklearn.model_selection import GroupKFold
# 同一患者的数据不能跨训练/测试
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=patient_ids):
    # 同一 patient_id 只在一个集合
    pass
```

---

## 二十三、思考题与练习

### 基础题

1. **简答题**：`cross_val_score` 为什么每折都要 `clone` 估计器？不 clone 会怎样？

2. **简答题**：`GridSearchCV.fit` 最后为什么要在全量数据上重训？

3. **代码题**：用 `KFold(n_splits=5, shuffle=True)` 对逻辑回归做 5 折 CV，打印每折分数和平均分。

4. **计算题**：参数空间 `{'C': [0.1, 1, 10], 'max_iter': [100, 200, 500]}`，5 折 CV，总共训练多少次？

### 进阶题

5. **分析题**：K=5 和 K=10 的 CV，哪个偏差大？哪个方差大？为什么？

6. **代码题**：实现嵌套 CV，外层 5 折评估，内层 3 折调参，输出调参后模型的无偏泛化估计。

7. **调试题**：下面代码有什么问题？
   ```python
   scaler = StandardScaler().fit(X)
   X_scaled = scaler.transform(X)
   scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
   ```

8. **设计题**：如何用 GridSearchCV 比较多个模型？给出代码框架。

### 高级题

9. **推导题**：推导 K 折 CV 估计的偏差约为 $a/(n(K-1))$，其中 $a$ 是学习曲线系数。

10. **实验题**：设计实验验证"参数空间太细会导致验证集过拟合"——随候选数增加，best_score_ 如何变化？

11. **源码题**：阅读 minisklearn 的 GridSearchCV 源码，找出 `clone` 被调用了几次，分别在什么时机。

12. **扩展题**：实现一个简化版的 `RandomizedSearchCV`，从参数分布中随机采样 `n_iter` 个组合，用 CV 评估。

---

## 二十四、扩展阅读

### 24.1 官方文档

- [sklearn 交叉验证文档](https://scikit-learn.org/stable/modules/cross_validation.html)
- [sklearn 网格搜索文档](https://scikit-learn.org/stable/modules/grid_search.html)
- [sklearn CV 策略对比](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)

### 24.2 推荐书籍

- 《The Elements of Statistical Learning》第 7 章：交叉验证的理论基础
- 《Pattern Recognition and Machine Learning》第 3 章：模型选择
- 《Applied Predictive Modeling》第 4 章：超参数调优实践

### 24.3 相关算法

- [Pipeline 详解](../pipeline/index.md)：防数据泄露的搭档
- [KFold 源码](../../../minisklearn/model_selection/_split.py)：划分实现
- [GridSearchCV 源码](../../../minisklearn/model_selection/_search.py)：搜索实现

### 24.4 进阶主题

- **贝叶斯优化**：skopt、Optuna、Hyperopt
- **HalvingGridSearchCV**：连续减半搜索，sklearn 1.0+
- **Successive Halving**：资源分配策略
- **多目标优化**：同时优化多个指标

### 24.5 推荐论文

- "A Survey of Cross-Validation Procedures for Model Selection"（Bergstra & Bengio）
- "Algorithms for Hyper-Parameter Optimization"（Bergstra et al.）
- "On Over-fitting in Model Selection"（Varma & Simon）

---

[← 返回算法列表](../index.md)
