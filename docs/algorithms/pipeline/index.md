# Pipeline：估计器流水线

> Pipeline 是 sklearn 最常用的元估计器——它把数据预处理和模型训练串成一条流水线，保证 predict 时的预处理参数和 fit 时完全一致，杜绝数据泄露。

---

## 一、为什么需要 Pipeline？

### 1.1 没有 Pipeline 时的痛点

```python
# 手动串联
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled = scaler.transform(X_test)          # 只 transform

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
```

问题：

1. **容易数据泄露**：误写 `scaler.fit_transform(X_test)` 就完了——测试集的统计信息泄露进了预处理参数。
2. **代码冗余**：每个步骤都要手动传递数据。
3. **无法网格搜索预处理参数**：GridSearchCV 只能搜索一个估计器的参数，无法同时搜索 scaler 和 clf 的参数。
4. **无法序列化为一个整体**：scaler 和 clf 要分别 pickle。

### 1.2 Pipeline 的解决方案

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)    # 自动串联 fit_transform
y_pred = pipe.predict(X_test) # 自动串联 transform + predict
```

一条流水线，一个对象，一次 fit，一次 predict。

### 1.3 数据泄露的严重性

数据泄露不只是"理论问题"，它会让你的模型在测试集上表现虚高，上线后大跌。经典场景：

```python
# 错误：标准化用全部数据
scaler = StandardScaler().fit(X)  # 用了 X（含测试集）
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)
# 测试集的均值/方差影响了 scaler，模型"偷看"了测试集分布

# 错误：特征选择用全部数据
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10).fit(X, y)  # 用了全部 y
X_selected = selector.transform(X)
# 测试集的标签参与了特征选择，泄露严重
```

Pipeline 在 CV 中自动避免这些：每折只在训练部分 fit 转换器，测试部分只 transform。

---

## 二、核心设计：fit 链 vs predict 链

### 2.1 fit 链

```
fit(X, y):
    X → scaler.fit_transform(X, y) → X₁   # 学习缩放参数 + 转换
    X₁ → clf.fit(X₁, y)                    # 学习模型参数
```

每步调用 `fit_transform`（转换器同时学习参数和转换数据）。

### 2.2 predict 链

```
predict(X):
    X → scaler.transform(X) → X₁   # 只用已学参数转换
    X₁ → clf.predict(X₁)           # 预测
```

每步调用 `transform`（只用 fit 时学到的参数，不重新学习）。

### 2.3 为什么这个区别很重要？

```
                    fit 时                    predict 时
StandardScaler   fit_transform(X)          transform(X)
                 → 学 mean_、scale_        → 用 mean_、scale_
                 → 返回 (X - mean_) / scale_  → 返回 (X - mean_) / scale_

LogisticRegression  fit(X, y)              predict(X)
                     → 学 coef_、intercept_  → 用 coef_、intercept_
```

**保证一致性**：predict 用的 `mean_`、`scale_` 是 fit 时学到的，不会用测试集重新计算。这就是 Pipeline 防数据泄露的机制——不是靠纪律，而是靠**架构**。

### 2.4 实现

```python
def fit(self, X, y=None):
    for name, step in self.steps[:-1]:
        X = step.fit_transform(X, y)   # 转换器：fit + transform
    self.steps[-1][1].fit(X, y)        # 最后一步：只 fit
    return self

def predict(self, X):
    for name, step in self.steps[:-1]:
        X = step.transform(X)          # 转换器：只 transform
    return self.steps[-1][1].predict(X)  # 最后一步：predict
```

### 2.5 fit_transform 的优化

`TransformerMixin` 默认 `fit_transform = fit; transform`，但有些转换器覆盖它以避免重复计算。例如 PCA：

```python
class PCA:
    def fit_transform(self, X):
        # SVD 直接给出投影，不用再 transform
        U, S, Vt = np.linalg.svd(X - X.mean(axis=0))
        return U[:, :k] * S[:k]  # 直接返回投影，省一次矩阵乘法
```

Pipeline 调用 `fit_transform` 时自动用这个优化。

### 2.6 fit_predict 的特殊情况

如果最后一步有 `fit_predict`（如 KMeans），`Pipeline.fit_predict` 会用它。但通常 predict 链用 transform + predict。

---

## 三、嵌套参数管理

### 3.1 问题：BaseEstimator 的默认实现不够用

`BaseEstimator.get_params` 通过反射从 `__init__` 签名提取参数名。Pipeline 的 `__init__` 只有一个参数 `steps`：

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps
```

所以默认 `get_params` 只返回 `{'steps': [...]}`，不返回 `clf__C` 等嵌套参数。`set_params(clf__C=10)` 也会报错"无效参数 clf"。

### 3.2 解决方案：覆盖 get_params / set_params

```python
def get_params(self, deep=True):
    out = {"steps": self.steps}
    if deep:
        for name, step in self.steps:
            if hasattr(step, "get_params"):
                for sub_key, sub_value in step.get_params(deep=True).items():
                    out[f"{name}__{sub_key}"] = sub_value
    return out
```

遍历每个步骤，把内部估计器的参数展平为 `stepname__param` 格式：

```
Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1))])
→ get_params(deep=True) 返回:
  {
    'steps': [...],
    'scaler__with_mean': True,
    'scaler__with_std': True,
    'clf__C': 1.0,
    'clf__max_iter': 100,
    ...
  }
```

`set_params` 则反向操作——把 `clf__C=10` 路由到 `clf` 步骤的 `set_params(C=10)`：

```python
def set_params(self, **params):
    for key, value in params.items():
        if "__" in key:
            step_name, sub_param = key.split("__", 1)
            step_dict[step_name].set_params(**{sub_param: value})
        else:
            setattr(self, key, value)
    return self
```

### 3.3 为什么用双下划线 `__`？

- 单下划线 `_` 在 Python 中有"受保护属性"的语义
- 双下划线 `__` 在 Python 中有名称改写（name mangling）的语义，但在这里只是字符串分隔符
- `__` 不会与任何合法的 Python 标识符冲突（参数名不会包含 `__`）
- sklearn 的 SLEP003 约定了这个命名规则

### 3.4 这使得 GridSearchCV 可以搜索 Pipeline 参数

```python
grid = GridSearchCV(pipe, {
    'scaler__with_std': [True, False],
    'clf__C': [0.1, 1, 10],
}, cv=5)
```

GridSearchCV 调用 `pipe.set_params(**params)`，Pipeline 的覆盖版 `set_params` 把嵌套参数路由到正确的步骤。这是 Pipeline + GridSearchCV 协作的核心机制。

### 3.5 多层嵌套

Pipeline 可以套 Pipeline，参数名用多层 `__`：

```python
pipe = Pipeline([
    ('preprocess', Pipeline([('scaler', StandardScaler()), ('pca', PCA())])),
    ('clf', LogisticRegression()),
])
# 参数：preprocess__scaler__with_std, preprocess__pca__n_components, clf__C
```

`set_params` 递归处理：`preprocess__scaler__with_std` → `preprocess.set_params(scaler__with_std=...)` → `scaler.set_params(with_std=...)`。

---

## 四、clone 的特殊处理

### 4.1 默认 clone 的问题

`clone` 函数对普通估计器的处理是：取 `__init__` 参数 → 重新构造。对 Pipeline：

```python
# clone 的默认逻辑
param_names = ['steps']
params = {'steps': clone(self.steps)}  # steps 是 list → deepcopy
new_object = Pipeline(steps=deepcopy(self.steps))
```

`deepcopy(self.steps)` 会深拷贝整个列表，包括每个估计器对象。但 deepcopy 会复制**拟合状态**（`coef_`、`mean_` 等），而 clone 的语义是"得到未训练的同参数副本"。

### 4.2 覆盖 __sklearn_clone__

```python
def __sklearn_clone__(self):
    new_steps = [(name, clone(step)) for name, step in self.steps]
    return Pipeline(steps=new_steps)
```

对每个步骤的估计器调用 `clone`（而非 `deepcopy`），得到未训练的同参数副本。`clone` 函数会检测子类是否覆盖了 `__sklearn_clone__`，如果覆盖了就调用它：

```python
def clone(estimator, *, safe=True):
    if not isinstance(estimator, BaseEstimator):
        return copy.deepcopy(estimator)

    # 子类覆盖了 __sklearn_clone__ → 用自定义克隆
    if type(estimator).__sklearn_clone__ is not BaseEstimator.__sklearn_clone__:
        return estimator.__sklearn_clone__()

    # 默认克隆逻辑（反射 __init__ 参数）
    ...
```

### 4.3 clone 的语义

clone 的核心语义：**得到一个未训练的同参数副本**。

- 复制 `__init__` 参数（超参数）
- 不复制 `xxx_` 属性（学习参数）
- 递归处理嵌套估计器（Pipeline 的各步骤）

这保证 GridSearchCV 每次评估都从干净状态开始，不被前一次 fit 污染。

---

## 五、使用示例

### 5.1 分类流水线

```python
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, learning_rate=0.5)),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

### 5.2 回归流水线

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', LinearRegression()),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

### 5.3 Pipeline + GridSearchCV

```python
from minisklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

grid = GridSearchCV(pipe, {
    'clf__C': [0.1, 1, 10],
    'clf__max_iter': [100, 200, 500],
}, cv=5)
grid.fit(X, y)
print(grid.best_params_)  # {'clf__C': 1, 'clf__max_iter': 200}
```

### 5.4 多步流水线

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # 缺失值填充
    ('scaler', StandardScaler()),                   # 标准化
    ('pca', PCA(n_components=10)),                  # 降维
    ('clf', LogisticRegression()),                  # 分类
])
pipe.fit(X_train, y_train)
```

### 5.5 完整可运行示例

```python
import numpy as np
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.decomposition import PCA
from minisklearn.linear_model import LogisticRegression
from minisklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# 1. 生成数据
rng = np.random.RandomState(42)
X = rng.randn(500, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 流水线
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression(max_iter=500)),
])

# 4. 交叉验证
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"CV: {scores.mean():.4f} ± {scores.std():.4f}")

# 5. 网格搜索
grid = GridSearchCV(pipe, {
    'pca__n_components': [5, 10, 15],
    'clf__C': [0.1, 1, 10],
}, cv=5)
grid.fit(X_train, y_train)
print(f"最优参数: {grid.best_params_}")
print(f"测试分数: {grid.score(X_test, y_test):.4f}")
```

### 5.6 错误示例

```python
# 错误 1：最后一步不是估计器
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
# pipe.predict(X) 报错：最后一步没有 predict

# 错误 2：中间步骤不是转换器
pipe = Pipeline([('clf1', LogisticRegression()), ('clf2', LogisticRegression())])
# clf1 没有 transform，fit 时报错

# 错误 3：步骤名重复
pipe = Pipeline([('step', StandardScaler()), ('step', LogisticRegression())])
# 报错：步骤名必须唯一

# 错误 4：用 fit_transform 当 transform
pipe.fit(X_train, y_train)
# pipe.transform(X_test) 报错：最后一步不是转换器，没有 transform
# 用 pipe.predict(X_test)
```

### 5.7 对比示例：手动 vs Pipeline

```python
# 手动（易错）
scaler = StandardScaler().fit(X_train)
pca = PCA(n_components=10).fit(scaler.transform(X_train))
clf = LogisticRegression().fit(pca.transform(scaler.transform(X_train)), y_train)
y_pred = clf.predict(pca.transform(scaler.transform(X_test)))  # 容易写错顺序

# Pipeline（安全）
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(10)), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)  # 自动按正确顺序 transform
```

---

## 六、与 sklearn 对比

### 6.1 API 一致性

| 特性 | minisklearn Pipeline | sklearn Pipeline |
|------|---------------------|------------------|
| `steps` | ✓ | ✓ |
| `fit` / `predict` / `transform` | ✓ | ✓ |
| `fit_transform` | ✓ | ✓ |
| `inverse_transform` | ✗ | ✓ |
| `get_params` / `set_params` | ✓（覆盖） | ✓（覆盖） |
| `__sklearn_clone__` | ✓（覆盖） | ✓（覆盖） |
| `named_steps` | ✗ | ✓（字典访问） |
| `[:-1]` 切片 | ✗ | ✓ |
| `FeatureUnion` 并行 | ✗ | ✓ |
| `ColumnTransformer` 列变换 | ✗ | ✓ |
| `make_pipeline` 便捷构造 | ✗ | ✓ |

### 6.2 named_steps 的便利

```python
# sklearn：用名字访问步骤
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
print(pipe.named_steps['scaler'].mean_)  # 直接访问

# minisklearn：用 steps 列表
print(pipe.steps[0][1].mean_)
```

### 6.3 make_pipeline

```python
# sklearn：自动生成步骤名（类名小写）
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), PCA(10), LogisticRegression())
# 等价于 Pipeline([('standardscaler', StandardScaler()), ...])

# minisklearn：必须显式命名
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(10)), ('clf', LogisticRegression())])
```

### 6.4 FeatureUnion（并行变换）

sklearn 的 FeatureUnion 把多个变换器的输出拼接（而非串联）：

```python
from sklearn.pipeline import FeatureUnion
features = FeatureUnion([
    ('pca', PCA(5)),
    ('svd', TruncatedSVD(5)),
])
# 输出是 PCA 和 SVD 结果的拼接，10 维
```

minisklearn 不实现，但理解原理有助于设计复杂流水线。

### 6.5 ColumnTransformer（按列变换）

对不同列应用不同变换：

```python
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),      # 数值列标准化
    ('cat', OneHotEncoder(), ['gender', 'city']),       # 类别列独热
])
pipe = Pipeline([('pre', preprocessor), ('clf', LogisticRegression())])
```

这是处理混合类型数据的标准模式。

---

## 七、复杂度分析

### 7.1 fit 复杂度

$$
T_{fit}^{pipe} = \sum_{i=1}^{n-1} T_{fit\_transform}^{(i)} + T_{fit}^{(n)}
$$

各步顺序执行，总时间是各步之和。

### 7.2 predict 复杂度

$$
T_{predict}^{pipe} = \sum_{i=1}^{n-1} T_{transform}^{(i)} + T_{predict}^{(n)}
$$

### 7.3 内存

各步的中间结果 $X_1, X_2, \dots$ 顺序覆盖，峰值内存是最大中间结果 + 各步参数。

### 7.4 实测

```python
import numpy as np, time
from minisklearn.pipeline import Pipeline

X = np.random.randn(10000, 100)
y = (X[:, 0] > 0).astype(int)
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(50)), ('clf', LogisticRegression())])
t0 = time.time()
pipe.fit(X, y)
print(f"fit: {time.time()-t0:.3f}s")
t0 = time.time()
pipe.predict(X)
print(f"predict: {time.time()-t0:.3f}s")
```

---

## 八、常见问题与陷阱

### 8.1 步骤名重复

```python
pipe = Pipeline([('step', StandardScaler()), ('step', PCA())])
# 报错：步骤名必须唯一（set_params 路由歧义）
```

### 8.2 最后一步不是估计器

```python
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pipe.predict(X)  # 报错：PCA 没有 predict
# 如果要 transform 链，用 pipe.transform(X)
```

### 8.3 中间步骤不是转换器

```python
pipe = Pipeline([('clf', LogisticRegression()), ('scaler', StandardScaler())])
pipe.fit(X, y)  # 报错：LogisticRegression 没有 transform
```

### 8.4 忘记 fit 就 predict

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
# pipe.predict(X)  # 报错：没 fit，scaler 没 mean_
```

### 8.5 CV 中的数据泄露（Pipeline 防的就是这个）

```python
# 不用 Pipeline，手动预处理 + CV → 泄露
scaler = StandardScaler().fit(X)  # 用全部数据
X_scaled = scaler.transform(X)
cross_val_score(LogisticRegression(), X_scaled, y)  # 每折的"测试"数据已被全量 scaler 处理 → 泄露

# 用 Pipeline → 不泄露
cross_val_score(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]), X, y)
# 每折只在训练部分 fit scaler
```

### 8.6 步骤顺序错误

```python
# 错误：先 PCA 再标准化
pipe = Pipeline([('pca', PCA()), ('scaler', StandardScaler()), ('clf', LogisticRegression())])
# PCA 对尺度敏感，应该先标准化

# 正确
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('clf', LogisticRegression())])
```

### 8.7 transform 与 predict 混用

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.fit(X, y)
# pipe.transform(X)  # 报错：最后一步 clf 没有 transform
# 要获取中间结果，用 pipe[:-1].transform(X)（sklearn）
```

---

## 九、实际使用教程

### 9.1 标准分类流水线

```python
from minisklearn.pipeline import Pipeline
from minisklearn.preprocessing import StandardScaler
from minisklearn.linear_model import LogisticRegression
from minisklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
grid = GridSearchCV(pipe, {'clf__C': [0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
print(f"测试分数: {grid.score(X_test, y_test):.4f}")
```

### 9.2 完整数据科学流水线

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),    # 缺失值
    ('scaler', StandardScaler()),                     # 标准化
    ('pca', PCA(n_components=0.95)),                  # 降维（保留 95% 方差）
    ('clf', LogisticRegression(max_iter=1000)),       # 分类
])

param_grid = {
    'imputer__strategy': ['mean', 'median'],
    'pca__n_components': [10, 20, 30],
    'clf__C': [0.1, 1, 10],
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

### 9.3 保存和加载

```python
import pickle
# 训练
pipe.fit(X_train, y_train)
# 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f)
# 加载
with open('model.pkl', 'rb') as f:
    pipe_loaded = pickle.load(f)
pipe_loaded.predict(X_new)  # 直接用，预处理参数都在
```

### 9.4 检查中间结果

```python
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(2)), ('clf', LogisticRegression())])
pipe.fit(X, y)

# 查看各步参数
print(pipe.steps[0][1].mean_)   # scaler 的均值
print(pipe.steps[1][1].components_)  # PCA 的主成分
print(pipe.steps[2][1].coef_)   # clf 的系数

# 获取中间变换结果（sklearn）
X_scaled = pipe[:-1].transform(X)  # 去掉最后一步，transform
```

### 9.5 可视化流水线

```python
from sklearn import set_config
set_config(display='diagram')  # Jupyter 中显示流水线图
pipe  # 显示 HTML 图
```

---

## 十、变体与扩展

### 10.1 FeatureUnion（并行）

```python
from sklearn.pipeline import FeatureUnion
union = FeatureUnion([
    ('pca', PCA(5)),
    ('pca2', PCA(10)),
])
# 输出：PCA(5) 的 5 维 + PCA(10) 的 10 维 = 15 维
```

### 10.2 ColumnTransformer（按列）

```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', StandardScaler(), [0, 1, 2]),      # 前 3 列标准化
    ('cat', OneHotEncoder(), [3, 4]),           # 后 2 列独热
])
pipe = Pipeline([('ct', ct), ('clf', LogisticRegression())])
```

### 10.3 TransformedTargetRegressor（变换标签）

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LogTransformer
reg = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=LogTransformer(),  # 对 y 取对数
)
# fit 时对 log(y) 回归，predict 时返回 exp(pred)
```

### 10.4 make_union / make_column_transformer

便捷构造函数，自动命名。

---

## 十一、架构回扣

### 11.1 元估计器架构

Pipeline 是第 4 讲[元估计器设计](../../architecture/04-meta-estimator.md)的典型实现：

- **包装其他估计器**：`steps` 列表中的每个元素都是估计器
- **自身也是估计器**：有 `fit`/`predict`/`score`，可以替代基础估计器出现在任何地方
- **参数展平**：通过覆盖 `get_params`/`set_params` 支持嵌套参数命名

### 11.2 与 BaseEstimator 的协作

| BaseEstimator 能力 | Pipeline 如何使用 |
|--------------------|--------------------|
| `get_params`/`set_params` | 覆盖以支持嵌套参数 |
| `clone` | 覆盖 `__sklearn_clone__` 以正确克隆各步骤 |
| `__repr__` | 继承默认实现（`get_params(deep=False)` 返回 `{'steps': ...}`） |

### 11.3 鸭子类型的力量

Pipeline 不检查步骤类型，只检查方法是否存在：

```python
def _validate_steps(self):
    for name, step in self.steps[:-1]:
        if not hasattr(step, "transform"):  # 有 transform 就是转换器
            raise TypeError(...)
    if not hasattr(self.steps[-1][1], "fit"):  # 有 fit 就是估计器
        raise TypeError(...)
```

这得益于第 1 讲[统一 API](../../architecture/01-unified-api.md)的鸭子类型设计——不看出身（继承关系），只看能力（方法存在）。

### 11.4 与 GridSearchCV 的协作

Pipeline + GridSearchCV 是 sklearn 的黄金组合：

```python
grid = GridSearchCV(pipe, {'clf__C': [0.1, 1, 10]}, cv=5)
# GridSearchCV 调用 pipe.set_params(clf__C=...)
# Pipeline 的 set_params 路由到 clf.set_params(C=...)
# CV 每折 clone(pipe)，pipe.__sklearn_clone__ 克隆各步骤
# 每折 fit 时 scaler 只在训练部分 fit，防泄露
```

### 11.5 与 cross_val_score 的协作

```python
cross_val_score(pipe, X, y, cv=5)
# 每折 clone(pipe) → 新 Pipeline，各步骤 clone
# fit(X_train, y_train) → scaler.fit_transform(X_train) → clf.fit(...)
# score(X_test, y_test) → scaler.transform(X_test) → clf.predict → accuracy
```

---

## 十二、进阶话题

### 12.1 Pipeline 的代数性质

Pipeline 可以看作函数复合：$f = f_n \circ f_{n-1} \circ \dots \circ f_1$。fit 学的是各 $f_i$ 的参数，predict 是复合函数求值。

### 12.2 可逆变换与 inverse_transform

如果所有步骤都可逆（如 StandardScaler、PCA），Pipeline 可以 `inverse_transform`：

```python
X_new = pipe.transform(X)
X_reconstructed = pipe.inverse_transform(X_new)  # 近似 X
```

### 12.3 缓存转换器

重复计算（如 CV 中多次 fit 同一转换器）可以缓存：

```python
from joblib import Memory
memory = Memory(location='cachedir')
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())],
                memory=memory)
```

### 12.4 Pipeline 的可序列化

Pipeline 是一个对象，可以整体 pickle。这解决了"分别保存 scaler 和 clf"的痛点：

```python
pickle.dump(pipe, f)  # 一次保存所有步骤
```

### 12.5 自定义步骤

任何有 `fit`/`transform` 的类都能做步骤：

```python
class MyTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X ** 2  # 平方特征

pipe = Pipeline([('sq', MyTransformer()), ('clf', LogisticRegression())])
```

---

## 十三、更多代码示例

### 13.1 手动实现 Pipeline（教学版）

```python
class SimplePipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def predict(self, X):
        for name, step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1][1].predict(X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

### 13.2 调试中间步骤

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)

# 检查 scaler 学到什么
scaler = pipe.steps[0][1]
print(f"均值: {scaler.mean_}")
print(f"标准差: {scaler.scale_}")

# 检查 clf 学到什么
clf = pipe.steps[1][1]
print(f"系数: {clf.coef_}")

# 手动走一遍 predict 链
X_test_scaled = scaler.transform(X_test)
y_pred_manual = clf.predict(X_test_scaled)
y_pred_pipe = pipe.predict(X_test)
assert np.all(y_pred_manual == y_pred_pipe)
```

### 13.3 动态修改步骤

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
# 替换步骤
pipe.steps[1] = ('clf', LinearSVC())  # 换成 SVM
pipe.fit(X_train, y_train)
```

### 13.4 条件流水线

```python
def make_pipeline(use_pca=True):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(10)))
    steps.append(('clf', LogisticRegression()))
    return Pipeline(steps)
```

### 13.5 对比有/无 Pipeline 的 CV

```python
# 无 Pipeline：泄露
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
scores_leak = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)

# 有 Pipeline：不泄露
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
scores_safe = cross_val_score(pipe, X, y, cv=5)

print(f"泄露: {scores_leak.mean():.4f}")  # 乐观偏误
print(f"安全: {scores_safe.mean():.4f}")  # 真实估计
```

---

## 十五、Pipeline 的数学视角

### 15.1 流水线作为函数复合

设第 $i$ 步的变换为 $f_i$（fit 后学到的参数固定），Pipeline 的整体变换是复合：

$$
f_{\text{pipe}} = f_n \circ f_{n-1} \circ \dots \circ f_1
$$

predict 是 $f_{\text{pipe}}(x)$。fit 是学各 $f_i$ 的参数：$f_1$ 在原始数据上学，$f_2$ 在 $f_1(X)$ 上学，以此类推。

### 15.2 链式法则与梯度反传

如果所有步骤可微，Pipeline 的梯度可以用链式法则算（类似神经网络的反向传播）：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_1}{\partial x}
$$

sklearn 的 Pipeline 不做自动微分，但理解这有助于把 Pipeline 看作"数据处理网络"。

### 15.3 交换律与顺序

Pipeline 步骤一般**不可交换**：先标准化再 PCA ≠ 先 PCA 再标准化。因为 PCA 对尺度敏感，标准化改变尺度，顺序影响结果。

少数可交换的情况：同类的标准化（如两个 StandardScaler 等价于一个）。

### 15.4 幂等性

有些变换幂等：$f(f(x)) = f(x)$。如 OneHotEncoder（独热编码两次结果相同）。幂等步骤重复无副作用，但浪费计算。

---

## 十六、更多使用场景

### 16.1 文本分类流水线

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
grid = GridSearchCV(pipe, {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.1, 1, 10],
}, cv=5)
grid.fit(texts, labels)
```

### 16.2 数值与类别混合

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols),
])
pipe = Pipeline([('pre', preprocessor), ('clf', LogisticRegression())])
```

### 16.3 特征工程流水线

```python
from sklearn.preprocessing import PolynomialFeatures
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),  # 交互特征
    ('clf', LogisticRegression()),
])
```

### 16.4 模型选择流水线

```python
from sklearn.model_selection import cross_val_score
pipes = {
    'lr': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
    'svm': Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC())]),
    'nb': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
}
for name, pipe in pipes.items():
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 16.5 增量学习流水线

```python
# sklearn 的 partial_fit
pipe = Pipeline([('scaler', StandardScaler()), ('clf', SGDClassifier())])
for batch_X, batch_y in batches:
    pipe.partial_fit(batch_X, batch_y, classes=np.unique(y))
```

---

## 十七、调试与诊断

### 17.1 检查中间数据形状

```python
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(10)), ('clf', LogisticRegression())])
pipe.fit(X, y)

# 逐步检查
X1 = pipe.steps[0][1].transform(X)
print(f"标准化后: {X1.shape}")
X2 = pipe.steps[1][1].transform(X1)
print(f"PCA 后: {X2.shape}")
```

### 17.2 检查参数是否正确设置

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=5))])
print(pipe.get_params())
# 确认 clf__C = 5

pipe.set_params(clf__C=10)
print(pipe.get_params()['clf__C'])  # 10
```

### 17.3 验证防泄露

```python
# 手动模拟 CV 一折
X_train, X_test = X[:80], X[80:]
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)

# scaler 应该只用 X_train 的统计量
assert np.allclose(pipe.steps[0][1].mean_, X_train.mean(axis=0))
# 不应该包含 X_test 的信息
```

### 17.4 性能分析

```python
import time
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(50)), ('clf', LogisticRegression(max_iter=1000))])
for name, step in pipe.steps:
    t0 = time.time()
    # 单独计时...
    print(f"{name}: {time.time()-t0:.3f}s")
```

---

## 十八、设计哲学

### 18.1 为什么用元组列表而非 *args

```python
# sklearn 选择
Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
# 而非
Pipeline(StandardScaler(), LogisticRegression())
```

原因：步骤名要唯一（set_params 路由需要），显式命名比自动生成（类名小写）更可控、更稳定（类名改了不影响参数名）。

### 18.2 为什么中间步用 fit_transform 而非 fit + transform

`fit_transform` 让转换器有机会优化（如 PCA 直接用 SVD 结果），避免重复计算。默认实现是 `fit; transform`，但覆盖可以省一次矩阵乘法。

### 18.3 为什么最后一步特殊

最后一步是估计器（有 predict），中间步是转换器（有 transform）。这个区分让 Pipeline 既能 fit+predict（分类/回归），又能 fit+transform（降维链）。如果最后一步也是转换器，Pipeline 整体是转换器，可以再嵌套。

### 18.4 为什么不用继承图区分转换器/估计器

sklearn 用 Mixin（TransformerMixin、ClassifierMixin）标记能力，但不强制继承。Pipeline 用鸭子类型（hasattr transform/predict）检查，更灵活。这符合"鸭子类型优于继承"的 Python 哲学。

---

## 十九、总结

| 要点 | 内容 |
|------|------|
| 核心作用 | 串联步骤，防数据泄露，统一接口 |
| fit 链 | 中间步 fit_transform，最后步 fit |
| predict 链 | 中间步 transform，最后步 predict |
| 嵌套参数 | `step__param`，覆盖 get_params/set_params |
| clone | 覆盖 `__sklearn_clone__`，递归 clone 各步骤 |
| 防泄露 | 架构保证，不靠纪律 |
| 与 GridSearchCV | 黄金组合，搜任意步骤参数 |
| 与 cross_val_score | 每折 clone，独立评估 |
| 复杂度 | 各步顺序执行，时间求和 |
| 与 sklearn | 核心一致，sklearn 有 FeatureUnion/ColumnTransformer |
| 适用 | 任何多步数据处理流程 |
| 陷阱 | 步骤名重复、顺序错误、混用 transform/predict |
| 数学视角 | 函数复合，链式法则 |
| 设计哲学 | 显式命名、鸭子类型、fit_transform 优化 |

---

## 二十、深入技术分析：fit_transform 的语义陷阱

### 20.1 fit_transform ≠ fit + transform

很多人以为 `fit_transform` 就是 `fit` 然后 `transform`，这在默认实现下成立，但转换器可以覆盖 `fit_transform` 做完全不同的事：

```python
class TransformerMixin:
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)  # 默认实现
```

但 PCA 覆盖了它：

```python
class PCA(TransformerMixin):
    def fit_transform(self, X, y=None):
        # 直接用 SVD 的 U*S，省一次矩阵乘法
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        return U[:, :self.n_components] * S[:self.n_components]
```

**陷阱**：如果你手动调用 `fit` 再 `transform`，结果数值上相同，但多一次矩阵乘法。Pipeline 调用 `fit_transform` 自动享受这个优化。

### 20.2 fit_transform 的契约要求

覆盖 `fit_transform` 必须满足：

1. **语义等价**：`fit_transform(X, y)` 的返回值必须等于 `fit(X, y).transform(X)`（数值上）
2. **状态一致**：调用后 `self` 的状态必须和 `fit(X, y)` 后相同（`mean_`、`components_` 等都要设置）
3. **参数一致**：`fit_transform` 接受的参数必须和 `fit` 一致

违反这些会导致 Pipeline 行为异常：

```python
class BadTransformer:
    def fit(self, X, y=None):
        self.param_ = X.mean()
        return self
    def transform(self, X):
        return X - self.param_
    def fit_transform(self, X, y=None):
        # 坏：没设置 self.param_，fit_transform 后 transform 会报错
        return X - X.mean()

t = BadTransformer()
out = t.fit_transform(X)  # 似乎 OK
t.transform(X)  # 报错：没有 param_
```

### 20.3 Pipeline 对 fit_transform 的调用时机

```python
def fit(self, X, y=None):
    for name, step in self.steps[:-1]:
        X = step.fit_transform(X, y)   # 中间步：fit_transform
    self.steps[-1][1].fit(X, y)        # 最后步：fit
```

中间步骤用 `fit_transform`，最后一步用 `fit`。如果最后一步是转换器（整个 Pipeline 是转换器），Pipeline 自己的 `fit_transform` 会优化：

```python
def fit_transform(self, X, y=None):
    for name, step in self.steps:
        X = step.fit_transform(X, y)   # 全部用 fit_transform
    return X
```

### 20.4 y 在转换链中的传递

有些转换器需要 `y`（如 `SelectKBest` 做特征选择时要看标签）：

```python
class SelectKBest:
    def fit(self, X, y):
        self.scores_ = compute_scores(X, y)  # 用 y 算特征得分
        return self
    def transform(self, X):
        return X[:, self.top_k_indices_]
```

Pipeline 把 `y` 一路传给所有中间步骤：

```python
for name, step in self.steps[:-1]:
    X = step.fit_transform(X, y)  # y 传给每个步骤
```

这允许"有监督的预处理"——如用标签做特征选择。但要注意：这会让数据泄露风险更大，必须保证 `y` 只来自训练折。

---

## 二十一、对比实验：Pipeline vs 手动串联

### 21.1 实验设计

我们对比四种方式在相同数据上的表现和代码复杂度：

```python
import numpy as np, time
from minisklearn.preprocessing import StandardScaler
from minisklearn.decomposition import PCA
from minisklearn.linear_model import LogisticRegression
from minisklearn.pipeline import Pipeline
from minisklearn.model_selection import cross_val_score

rng = np.random.RandomState(42)
X = rng.randn(800, 30)
y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)

# 方式 A：手动串联（易泄露）
scaler = StandardScaler().fit(X)
X1 = scaler.transform(X)
pca = PCA(n_components=10).fit(X1)
X2 = pca.transform(X1)
clf = LogisticRegression().fit(X2, y)
score_A = clf.score(X2, y)  # 用训练数据评估，乐观偏误

# 方式 B：手动串联 + 正确的 CV（繁琐）
def manual_cv(X, y, k=5):
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, Xte = X[train], X[test]
        ytr = y[train]
        s = StandardScaler().fit(Xtr)
        Xtr1 = s.transform(Xtr); Xte1 = s.transform(Xte)
        p = PCA(10).fit(Xtr1)
        Xtr2 = p.transform(Xtr1); Xte2 = p.transform(Xte1)
        c = LogisticRegression().fit(Xtr2, ytr)
        scores.append(c.score(Xte2, y[test]))
    return np.array(scores)
scores_B = manual_cv(X, y)

# 方式 C：Pipeline + cross_val_score（推荐）
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression()),
])
scores_C = cross_val_score(pipe, X, y, cv=5)

print(f"方式 A（泄露）: {score_A:.4f}")
print(f"方式 B（手动 CV）: {scores_B.mean():.4f} ± {scores_B.std():.4f}")
print(f"方式 C（Pipeline）: {scores_C.mean():.4f} ± {scores_C.std():.4f}")
```

预期：方式 A 分数最高（泄露），方式 B 和 C 接近（C 更简洁）。

### 21.2 代码量对比

| 方式 | 代码行数 | 泄露风险 | 可维护性 |
|------|----------|----------|----------|
| 手动串联 | ~15 | 高 | 差 |
| 手动 CV | ~20 | 中 | 差 |
| Pipeline | ~5 | 无 | 好 |

### 21.3 性能对比

```python
import time

# Pipeline
t0 = time.time()
cross_val_score(pipe, X, y, cv=5)
t_pipe = time.time() - t0

# 手动 CV
t0 = time.time()
manual_cv(X, y)
t_manual = time.time() - t0

print(f"Pipeline: {t_pipe:.3f}s")
print(f"手动 CV: {t_manual:.3f}s")
```

Pipeline 通常略快，因为内部用了 `fit_transform` 优化。

---

## 二十二、参数调优指南：Pipeline 的嵌套搜索

### 22.1 搜索预处理参数

```python
from minisklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression(max_iter=500)),
])

param_grid = {
    'pca__n_components': [5, 10, 15, 20, 25],   # 降维维度
    'clf__C': [0.01, 0.1, 1, 10, 100],          # 正则强度
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)
print(f"最优: {grid.best_params_}")
```

### 22.2 搜索是否需要某步

用 `None` 表示跳过该步骤：

```python
param_grid = {
    'pca': [PCA(5), PCA(10), None],  # None 表示不做 PCA
    'clf__C': [0.1, 1, 10],
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)
# 如果最优是 None，说明 PCA 没帮助
```

### 22.3 搜索不同的转换器

```python
from minisklearn.preprocessing import MinMaxScaler

param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],  # 试两种缩放
    'clf__C': [0.1, 1, 10],
}
grid = GridSearchCV(pipe, param_grid, cv=5)
```

### 22.4 调参顺序建议

1. **先固定预处理**，调模型参数（如 `clf__C`）
2. **再调预处理参数**（如 `pca__n_components`）
3. **最后联合微调**

```python
# 第一步：粗调 clf__C
grid1 = GridSearchCV(pipe, {'clf__C': [0.01, 0.1, 1, 10, 100]}, cv=5).fit(X, y)
best_C = grid1.best_params_['clf__C']

# 第二步：固定 C，调 PCA
pipe.set_params(clf__C=best_C)
grid2 = GridSearchCV(pipe, {'pca__n_components': [5, 10, 15, 20]}, cv=5).fit(X, y)
best_n = grid2.best_params_['pca__n_components']

# 第三步：联合微调
pipe.set_params(pca__n_components=best_n)
grid3 = GridSearchCV(pipe, {'clf__C': [best_C/3, best_C, best_C*3]}, cv=5).fit(X, y)
```

### 22.5 参数搜索的复杂度估算

```
参数组合数 = Π |每个参数的候选数|
总训练次数 = 参数组合数 × K 折
```

| pca__n_components | clf__C | 组合 | × 5 折 | 训练次数 |
|-------------------|--------|------|--------|----------|
| 5 | 5 | 25 | ×5 | 125 |
| 10 | 10 | 100 | ×5 | 500 |
| 20 | 20 | 400 | ×5 | 2000 |

组合爆炸很快，先用粗网格，再细搜。

---

## 二十三、常见错误与调试技巧

### 23.1 错误：步骤名重复

```python
Pipeline([('step', StandardScaler()), ('step', LogisticRegression())])
# ValueError: 步骤名必须唯一
```

**调试**：步骤名是 `set_params` 路由的 key，重复会导致路由歧义。用语义化名字：`'scaler'`、`'clf'`。

### 23.2 错误：中间步没有 transform

```python
Pipeline([('clf1', LogisticRegression()), ('clf2', LogisticRegression())])
# AttributeError: 'LogisticRegression' object has no attribute 'transform'
```

**调试**：中间步必须是转换器（有 `transform`）。要串联两个分类器，用 VotingClassifier 或 Stacking。

### 23.3 错误：最后步没有 predict

```python
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pipe.predict(X)
# AttributeError: 'PCA' object has no attribute 'predict'
```

**调试**：最后步是转换器时，用 `pipe.transform(X)` 而非 `predict`。

### 23.4 错误：fit 前就 predict

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe.predict(X)
# NotFittedError: 这个 Pipeline 实例还没拟合
```

**调试**：先 `pipe.fit(X_train, y_train)`。

### 23.5 调试技巧：逐步检查中间输出

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)

# 检查每步的输出形状
X1 = pipe.steps[0][1].transform(X_train)
print(f"标准化后: {X1.shape}, 均值={X1.mean(axis=0).round(3)}")
X2 = pipe.steps[1][1].transform(X1)
print(f"PCA 后: {X2.shape}, 方差比={pipe.steps[1][1].explained_variance_ratio_}")
y_pred = pipe.steps[2][1].predict(X2)
print(f"预测: {y_pred[:10]}")

# 验证与 pipe.predict 一致
assert np.all(y_pred == pipe.predict(X_train))
```

### 23.6 调试技巧：检查参数是否正确设置

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=5))])
params = pipe.get_params(deep=True)
print(params['clf__C'])  # 应为 5

pipe.set_params(clf__C=10, scaler__with_std=False)
print(pipe.get_params()['clf__C'])           # 10
print(pipe.get_params()['scaler__with_std'])  # False
```

### 23.7 调试技巧：验证防泄露

```python
X_train, X_test = X[:80], X[80:]
pipe.fit(X_train, y_train)
# scaler 的均值应该等于 X_train 的均值，不含 X_test
assert np.allclose(pipe.steps[0][1].mean_, X_train.mean(axis=0))
```

### 23.8 错误：步骤顺序不合理

```python
# 错误：先 PCA 再标准化
Pipeline([('pca', PCA()), ('scaler', StandardScaler()), ('clf', LogisticRegression())])
# PCA 对尺度敏感，结果差
```

**经验顺序**：缺失值填充 → 标准化 → 特征选择/降维 → 模型

---

## 二十四、实际应用场景

### 24.1 场景：文本分类

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', MultinomialNB()),
])
pipe.fit(texts_train, labels_train)
```

### 24.2 场景：表格数据（混合类型）

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income', 'score']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'city']),
])
pipe = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
```

### 24.3 场景：高维数据降维后分类

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # 保留 95% 方差
    ('clf', LogisticRegression()),
])
```

### 24.4 场景：缺失值处理

```python
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier()),
])
```

### 24.5 场景：模型部署（序列化）

```python
import pickle
pipe.fit(X_train, y_train)
with open('pipeline_model.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# 部署后
with open('pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)
predictions = model.predict(X_new)  # 预处理参数都在
```

### 24.6 场景：特征工程

```python
from sklearn.preprocessing import PolynomialFeatures

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])
# 自动生成交互特征，再标准化，再分类
```

---

## 二十五、思考题与练习

### 基础题

1. **简答题**：Pipeline 的 `fit` 方法和 `predict` 方法分别调用中间步骤的什么方法？为什么不同？

2. **简答题**：为什么 Pipeline 的步骤名必须唯一？如果允许重复会出什么问题？

3. **代码题**：写一个 Pipeline，包含 `StandardScaler` → `PCA(n_components=5)` → `LogisticRegression()`，在鸢尾花数据上做 5 折交叉验证。

4. **判断题**：`Pipeline.fit_transform` 一定等于 `Pipeline.fit` 后 `Pipeline.transform`。（提示：考虑最后一步）

### 进阶题

5. **分析题**：下面两种顺序哪个对？为什么？
   ```python
   # A
   Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('clf', LR())])
   # B
   Pipeline([('pca', PCA()), ('scaler', StandardScaler()), ('clf', LR())])
   ```

6. **代码题**：用 GridSearchCV 搜索 Pipeline 中 `pca__n_components` 取 `[5, 10, 15, None]`（None 表示跳过 PCA），找出最优配置。

7. **调试题**：下面代码哪里错了？
   ```python
   pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
   scaler = StandardScaler().fit(X)
   X_scaled = scaler.transform(X)
   pipe.fit(X_scaled, y)  # 这里有问题
   ```

8. **设计题**：如果要实现一个"条件步骤"——根据数据自动决定是否标准化——你会怎么设计？能在 Pipeline 框架内实现吗？

### 高级题

9. **源码题**：阅读 minisklearn 的 Pipeline 源码，找出 `get_params` 是如何递归处理多层嵌套 Pipeline 的。

10. **性能题**：Pipeline 中间步骤的 `fit_transform` 如果不优化（用默认的 `fit; transform`），PCA 步骤会多多少计算？设计实验测量。

11. **架构题**：为什么 Pipeline 用元组列表 `[(name, estimator), ...]` 而非字典 `{name: estimator}`？从步骤顺序和参数命名角度分析。

12. **扩展题**：实现一个 `MyPipeline`，支持"并行步骤"（类似 FeatureUnion），输出是多个分支的拼接。给出 `fit` 和 `transform` 的实现。

---

## 二十六、扩展阅读

### 26.1 官方文档

- [sklearn Pipeline 文档](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [sklearn FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)

### 26.2 设计文档

- sklearn 的 SLEP003：嵌套参数命名规范
- sklearn 的 SLEP007：估计器检查机制
- 本项目架构文档：[元估计器设计](../../architecture/04-meta-estimator.md)

### 26.3 相关算法

- [GridSearchCV 详解](../model_selection/index.md)：Pipeline 的最佳搭档
- [StandardScaler](../preprocessing/index.md)：最常用的第一步
- [PCA](../decomposition/index.md)：常用的降维步骤

### 26.4 进阶主题

- **sklearn 的 `set_config(display='diagram')`**：在 Jupyter 中可视化 Pipeline
- **`joblib.Memory` 缓存**：缓存中间步骤的计算结果
- **`TransformedTargetRegressor`**：对回归目标 y 做变换
- **Pipeline 与深度学习**：Pipeline 的函数复合视角与神经网络的前向传播类比

### 26.5 推荐论文

- "A Survey of Cross-Validation Procedures for Model Selection"（交叉验证与 Pipeline 的统计基础）
- "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation"（嵌套 CV 的必要性）

---

[← 返回算法列表](../index.md)
