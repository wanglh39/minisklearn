# 第四讲：元估计器模式

> **核心问题**：`Pipeline` 凭什么能串联任意算法？`GridSearchCV` 如何包装基础估计器？为什么 sklearn 用组合而非继承做元估计器？

---

## 1. 什么是元估计器？

**元估计器**（Meta-Estimator）是"以估计器为参数的估计器"——它包装其他估计器，组合出更复杂的行为。

```python
# Pipeline 是元估计器：包装了 StandardScaler + LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

# GridSearchCV 是元估计器：包装了 LogisticRegression
grid = GridSearchCV(
    LogisticRegression(),
    param_grid={'C': [0.1, 1, 10]},
)

# RandomForest 是元估计器：包装了 DecisionTree
rf = RandomForestClassifier(n_estimators=100)
```

元估计器的核心特征：**`__init__` 接收其他估计器作为参数**。

### 1.1 元估计器的本质

"元"（Meta）在这里的意思是"高阶"——元估计器是"估计器的估计器"，就像"元编程"是"编程的编程"。

```python
# 普通估计器：参数是数值/字符串
clf = LogisticRegression(C=1.0, penalty='l2')
# C 和 penalty 是数值/字符串

# 元估计器：参数是估计器
pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
# steps 里的元素是估计器
```

元估计器把估计器当"一等公民"——可以传参、可以组合、可以返回。这是函数式编程的"高阶函数"思想在 OOP 里的体现。

### 1.2 元估计器的分类

sklearn 的元估计器大致分五类：

| 类型 | 例子 | 包装什么 | 做什么 |
|------|------|---------|--------|
| **流水线** | `Pipeline` | 有序的步骤序列 | 串联执行 |
| **调参** | `GridSearchCV` | 一个基础估计器 | 搜索最优参数 |
| **集成** | `RandomForest` | 多个同类型估计器 | 投票/平均 |
| **多输出** | `MultiOutputClassifier` | 一个基础估计器 | 复制多份处理多输出 |
| **特征选择** | `RFE` | 一个基础估计器 | 用估计器选特征 |

它们的共同模式：`__init__` 接收估计器，`fit` / `predict` 委托给包装的估计器。

### 1.3 元估计器的威力

元估计器的威力在于**组合**——把简单估计器组合成复杂行为：

```python
# 简单估计器
scaler = StandardScaler()
pca = PCA(n_components=10)
clf = LogisticRegression()

# 组合成流水线
pipe = Pipeline([('scaler', scaler), ('pca', pca), ('clf', clf)])

# 再包装成网格搜索
grid = GridSearchCV(pipe, param_grid={
    'pca__n_components': [5, 10, 20],
    'clf__C': [0.1, 1, 10],
})

# 再包装成集成
bagging = BaggingClassifier(grid, n_estimators=10)
```

这种"无限嵌套"的能力，是 sklearn 统一 API 的最大红利。

---

## 2. 组合优于继承

为什么 `Pipeline` 不继承 `StandardScaler` 和 `LogisticRegression`？

```python
# ❌ 坏设计：多继承
class Pipeline(StandardScaler, LogisticRegression):
    ...
```

问题：

1. **动态组合**：`Pipeline` 的步骤是运行时传入的，不是编译时固定的
2. **菱形继承**：两者都继承 `BaseEstimator`，MRO 复杂
3. **职责混乱**：`Pipeline` 不是"是一种 StandardScaler"，而是"组合了若干步骤"

正确做法——组合：

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # 组合：持有步骤的引用
```

`Pipeline` **是一个 `BaseEstimator`**（有 `get_params` / `clone`），**组合了**若干步骤估计器。

### 2.1 动态组合的必要性

`Pipeline` 的步骤是运行时传入的，无法用继承表达：

```python
# 步骤是运行时决定的
if data_type == 'image':
    steps = [('scaler', StandardScaler()), ('pca', PCA()), ('clf', SVC())]
elif data_type == 'text':
    steps = [('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())]
else:
    steps = [('scaler', StandardScaler()), ('clf', LogisticRegression())]

pipe = Pipeline(steps)  # 动态组合
```

如果用继承，`Pipeline` 的类层次要在编译时固定，无法根据运行时条件组合。

### 2.2 菱形继承问题

```python
# 假设用继承
class Pipeline(StandardScaler, PCA, LogisticRegression): ...

# StandardScaler 继承 BaseEstimator
# PCA 继承 BaseEstimator
# LogisticRegression 继承 BaseEstimator
# 菱形继承：Pipeline → StandardScaler → BaseEstimator
#            Pipeline → PCA → BaseEstimator
#            Pipeline → LogisticRegression → BaseEstimator
```

三个父类都继承 `BaseEstimator`，菱形继承让 MRO 复杂，`get_params` 等方法的行为难以预测。

组合避开菱形——`Pipeline` 只继承 `BaseEstimator`，步骤是持有的引用，不参与继承。

### 2.3 职责的正确表达

```python
# is-a：Pipeline 是一个 BaseEstimator（有 get_params/clone）
class Pipeline(BaseEstimator): ...

# has-a：Pipeline 有若干步骤
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # has-a
```

`Pipeline` 和步骤的关系是 has-a（有），不是 is-a（是）——`Pipeline` 不是"是一种 StandardScaler"，而是"包含若干步骤"。用组合表达 has-a，正确。

---

## 3. `Pipeline` 的核心：fit 链 vs predict 链

`Pipeline` 最精妙的设计是 `fit` 和 `predict` 走不同的链路：

```python
class Pipeline(BaseEstimator):
    def fit(self, X, y=None):
        # fit 链：每一步 fit 后 transform，把结果传给下一步
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X, y)  # 转换器：fit + transform
        # 最后一步只 fit 不 transform
        self.steps[-1][1].fit(X, y)
        return self

    def predict(self, X):
        # predict 链：每一步只 transform，最后一步 predict
        for name, step in self.steps[:-1]:
            X = step.transform(X)  # 转换器：只 transform
        return self.steps[-1][1].predict(X)  # 最后一步 predict
```

**关键区别**：

- `fit` 时：转换器要 `fit_transform`（学习参数 + 转换数据）
- `predict` 时：转换器只 `transform`（用已学参数转换）

这保证了 `predict` 时的转换用的是 `fit` 时学到的参数，不会数据泄露。

### 3.1 fit 链的详细过程

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),   # 转换器
    ('pca', PCA(n_components=10)),  # 转换器
    ('clf', LogisticRegression()),  # 预测器
])

pipe.fit(X, y)

# 内部执行：
# 1. X_scaled = scaler.fit_transform(X, y)
#    - scaler 学习 X 的均值、标准差
#    - X_scaled = (X - mean) / std
# 2. X_pca = pca.fit_transform(X_scaled, y)
#    - pca 学习 X_scaled 的主成分
#    - X_pca = X_scaled @ pca.components_
# 3. clf.fit(X_pca, y)
#    - clf 学习 X_pca → y 的映射
```

每一步都 `fit_transform`：学习参数 + 转换数据，把转换结果传给下一步。

### 3.2 predict 链的详细过程

```python
y_pred = pipe.predict(X_test)

# 内部执行：
# 1. X_scaled = scaler.transform(X_test)
#    - 用 fit 时学的 mean、std 转换 X_test
#    - 不学习新参数！
# 2. X_pca = pca.transform(X_scaled)
#    - 用 fit 时学的 components_ 转换 X_scaled
#    - 不学习新参数！
# 3. y_pred = clf.predict(X_pca)
#    - 用 fit 时学的 coef_ 预测
```

每一步只 `transform`：用 `fit` 时学的参数转换，不学习新参数。

### 3.3 为什么 fit 和 predict 要走不同链路

如果 `predict` 也 `fit_transform`，会出两个问题：

```python
# ❌ 假设 predict 也 fit_transform（错误）
def predict(self, X):
    for name, step in self.steps[:-1]:
        X = step.fit_transform(X)  # 错！重新 fit 了
    return self.steps[-1][1].predict(X)

# 问题 1：数据泄露
# predict 时用测试数据重新 fit scaler，测试数据的分布影响了转换
# 这就是"数据泄露"——测试信息泄漏到了预测

# 问题 2：结果不稳定
# 每次 predict 都重新 fit，结果会变（如 PCA 的符号不确定）
```

`fit` 时学习，`predict` 时只用学到的参数——这是机器学习的基本原则。`Pipeline` 的 fit 链 vs predict 链设计，正是这一原则的体现。

### 3.4 `fit_transform` vs `fit` + `transform`

`Pipeline.fit` 用 `fit_transform` 而非 `fit` + `transform`：

```python
# ✅ 用 fit_transform
X = step.fit_transform(X, y)

# 等价但可能更慢
step.fit(X, y)
X = step.transform(X)
```

为什么用 `fit_transform`？因为有些算法可以优化：

```python
# PCA 的 fit_transform：fit 时已经算了转换结果，直接返回
class PCA(TransformerMixin):
    def fit_transform(self, X, y=None):
        U, S, V = np.linalg.svd(X)  # SVD 分解
        self.components_ = V
        return U * S  # 直接返回，省了一次矩阵乘法

    def transform(self, X):
        return X @ self.components_.T  # 还要算一次矩阵乘法
```

`fit_transform` 让 PCA 省一次矩阵乘法，对大数据集显著加速。

### 3.5 `Pipeline` 的其他方法

除了 `fit` 和 `predict`，`Pipeline` 还有：

```python
# fit_predict：fit 后 predict
y_pred = pipe.fit_predict(X, y)  # 聚类用

# fit_transform：fit 后 transform（如果最后一步是转换器）
X_new = pipe.fit_transform(X, y)

# predict_proba：最后一步的 predict_proba
y_proba = pipe.predict_proba(X)

# score：fit 后 score
score = pipe.score(X, y)

# inverse_transform：反向转换（如果所有步骤都支持）
X_orig = pipe.inverse_transform(X_new)
```

这些方法都遵循同样的"链路"模式——前几步 transform，最后一步调对应方法。

---

## 4. 嵌套参数命名：`step__param`

元估计器的参数用 `步骤名__参数名` 访问：

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0))])

pipe.get_params()
# {
#   'scaler': StandardScaler(),
#   'scaler__with_mean': True,
#   'scaler__with_std': True,
#   'clf': LogisticRegression(C=1.0),
#   'clf__C': 1.0,
#   'clf__max_iter': 100,
# }

pipe.set_params(clf__C=10.0)  # 设置嵌套参数
```

这个命名规则让 `GridSearchCV` 可以搜索 Pipeline 内部任意步骤的参数：

```python
GridSearchCV(pipe, param_grid={'clf__C': [0.1, 1, 10]})
```

### 4.1 `__` 分隔符的选择

为什么用 `__`（双下划线）而非 `.`（点）或 `_`（单下划线）？

| 分隔符 | 例子 | 问题 |
|--------|------|------|
| `.` | `clf.C` | Python 属性访问冲突 |
| `_` | `clf_C` | 和单下划线属性混淆 |
| `__` | `clf__C` | 无冲突，清晰 |

```python
# __ 不会和 Python 语法冲突
pipe.get_params()['clf__C']  # 字典键，OK
pipe.set_params(clf__C=10.0)  # kwargs 键，OK

# . 会冲突
pipe.set_params(clf.C=10.0)  # 语法错误：clf.C 不是合法标识符
```

`__` 是经过深思熟虑的选择——避免语法冲突，又清晰表达嵌套。

### 4.2 多层嵌套

```python
# 多层嵌套
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GridSearchCV(
        LogisticRegression(),
        param_grid={'C': [0.1, 1, 10]},
    )),
])

# 访问最内层的 C
pipe.get_params()
# {
#   'scaler': ...,
#   'scaler__with_mean': True,
#   'clf': GridSearchCV(...),
#   'clf__estimator': LogisticRegression(),
#   'clf__estimator__C': 1.0,  # 三层嵌套！
#   'clf__param_grid': {'C': [0.1, 1, 10]},
#   ...
# }

pipe.set_params(clf__estimator__C=5.0)  # 设置三层嵌套参数
```

`clf__estimator__C` 表示 `pipe.clf.estimator.C`——三层嵌套用 `__` 串联。

### 4.3 嵌套参数的实现

```python
def get_params(self, deep=True):
    out = super().get_params(deep=False)  # 顶层参数
    if deep:
        for name, step in self.steps:
            out[name] = step  # 步骤本身
            for sub_key, sub_value in step.get_params().items():
                out[f"{name}__{sub_key}"] = sub_value  # 嵌套展平
    return out

def set_params(self, **params):
    # 分离顶层和嵌套
    nested = {}
    for key, value in params.items():
        if "__" in key:
            step, sub = key.split("__", 1)
            nested.setdefault(step, {})[sub] = value
        else:
            setattr(self, key, value)
    # 递归 set_params
    for step, sub_params in nested.items():
        self.named_steps[step].set_params(**sub_params)
    return self
```

`__` 分隔左边是步骤名，右边是子参数名。递归处理多层嵌套。

---

## 5. `GridSearchCV`：clone + 交叉验证

`GridSearchCV` 是元估计器的集大成者：

```python
class GridSearchCV(BaseEstimator):
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator = estimator   # 被包装的基础估计器
        self.param_grid = param_grid
        self.cv = cv

    def fit(self, X, y):
        best_score = -inf
        for params in self._iter_param_combinations():
            scores = []
            for train_idx, val_idx in self._split(X, y):
                # clone 出干净副本，设置参数，fit，评分
                clf = clone(self.estimator)
                clf.set_params(**params)
                clf.fit(X[train_idx], y[train_idx])
                scores.append(clf.score(X[val_idx], y[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params

        # 用最优参数在全量数据上重训
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)
```

注意 `clone` 的使用：每个参数组合都 `clone` 一个干净副本，防止状态污染。

### 5.1 `GridSearchCV.fit` 的详细流程

```python
grid = GridSearchCV(
    LogisticRegression(),
    param_grid={'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
    cv=5,
)
grid.fit(X, y)

# 内部流程：
# 1. 生成参数组合
#    [(C=0.1, penalty='l1'), (C=0.1, penalty='l2'),
#     (C=1, penalty='l1'), (C=1, penalty='l2'),
#     (C=10, penalty='l1'), (C=10, penalty='l2')]
#    共 3 × 2 = 6 种组合

# 2. 对每种组合，5 折交叉验证
#    for params in combinations:
#        for fold in range(5):
#            clf = clone(estimator)  # 干净副本
#            clf.set_params(**params)
#            clf.fit(X_train_fold, y_train_fold)
#            score = clf.score(X_val_fold, y_val_fold)
#        mean_score = mean(5 个 score)

# 3. 选最优
#    best_params = argmax(mean_score)

# 4. 用最优参数在全量数据重训
#    best_estimator = clone(estimator).set_params(**best_params)
#    best_estimator.fit(X, y)
```

总共训练 6 × 5 + 1 = 31 次模型。`clone` 保证每次从干净状态开始。

### 5.2 `clone` 在 `GridSearchCV` 中的关键作用

```python
# ❌ 不用 clone，复用同一个 estimator
class BadGridSearch:
    def fit(self, X, y):
        for params in self.param_grid:
            self.estimator.set_params(**params)  # 复用！
            self.estimator.fit(X_train, y_train)
            # 问题：self.estimator 已经 fit 了
            # 下次循环 set_params 后，coef_ 还在
            # 如果 set_params 不清 coef_，状态污染
```

`clone` 保证每次循环都是干净对象：

```python
# ✅ 用 clone
for params in self.param_grid:
    clf = clone(self.estimator)  # 干净的，没有 coef_
    clf.set_params(**params)
    clf.fit(X_train, y_train)  # 从干净状态开始 fit
```

### 5.3 `GridSearchCV` 的属性

`fit` 后，`GridSearchCV` 暴露：

```python
grid.fit(X, y)

print(grid.best_params_)      # {'C': 1, 'penalty': 'l2'}
print(grid.best_score_)       # 0.85（最优交叉验证分数）
print(grid.best_estimator_)   # LogisticRegression(C=1, penalty='l2')
print(grid.cv_results_)       # 所有组合的详细结果
print(grid.n_splits_)         # 5

# predict 用 best_estimator_
y_pred = grid.predict(X_test)
```

`best_estimator_` 是用最优参数在全量数据上重训的模型，`predict` 委托给它。

### 5.4 `GridSearchCV` 嵌套 `Pipeline`

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

grid = GridSearchCV(pipe, param_grid={
    'scaler__with_mean': [True, False],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
})

grid.fit(X, y)
# 搜索 scaler 的 with_mean 和 clf 的 C、penalty
# 共 2 × 3 × 2 = 12 种组合
```

`GridSearchCV` 能搜索 `Pipeline` 内部任意步骤的参数，靠的就是 `step__param` 命名。

---

## 6. 元估计器的分类

| 类型 | 例子 | 包装什么 |
|------|------|---------|
| **流水线** | `Pipeline` | 有序的步骤序列 |
| **调参** | `GridSearchCV` / `RandomizedSearchCV` | 一个基础估计器 |
| **集成** | `RandomForest` / `AdaBoost` | 多个同类型估计器 |
| **多输出** | `MultiOutputClassifier` | 一个基础估计器（复制多份） |
| **特征选择** | `RFE` / `SelectFromModel` | 一个基础估计器（用作选择器） |

它们的共同模式：`__init__` 接收估计器，`fit` / `predict` 委托给包装的估计器。

### 6.1 流水线类

```python
# Pipeline：有序步骤
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression()),
])

# FeatureUnion：并行特征合并
union = FeatureUnion([
    ('pca', PCA()),
    ('svd', TruncatedSVD()),
])
# PCA 和 SVD 并行，结果拼接
```

`Pipeline` 是串行，`FeatureUnion` 是并行。两者都是组合多个估计器。

### 6.2 调参类

```python
# GridSearchCV：网格搜索
grid = GridSearchCV(clf, param_grid={'C': [0.1, 1, 10]})

# RandomizedSearchCV：随机搜索
random = RandomizedSearchCV(clf, param_distributions={'C': uniform(0, 10)}, n_iter=20)
```

`GridSearchCV` 穷举所有组合，`RandomizedSearchCV` 随机采样。都包装一个基础估计器。

### 6.3 集成类

```python
# RandomForest：bagging 集成
rf = RandomForestClassifier(n_estimators=100)
# 内部 clone 100 棵决策树

# AdaBoost：boosting 集成
ada = AdaBoostClassifier(n_estimators=50)
# 内部 clone 50 个弱分类器，加权组合

# VotingClassifier：投票集成
voting = VotingClassifier([
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svm', SVC()),
])
# 组合不同类型的分类器
```

集成类包装多个估计器，用投票/平均/boosting 组合预测。

### 6.4 多输出类

```python
# MultiOutputClassifier：多标签分类
multi = MultiOutputClassifier(LogisticRegression())
multi.fit(X, Y)  # Y 是 (n_samples, n_outputs)
# 内部为每个输出 clone 一个 LogisticRegression
```

`MultiOutputClassifier` 把一个估计器复制多份，每份处理一个输出。

### 6.5 特征选择类

```python
# RFE：递归特征消除
rfe = RFE(LogisticRegression(), n_features_to_select=10)
# 用 LogisticRegression 的 coef_ 选特征

# SelectFromModel：基于模型选择
sfm = SelectFromModel(LogisticRegression())
# 用 LogisticRegression 的 coef_ 选特征
```

特征选择类包装一个估计器，用其 `coef_` 或 `feature_importances_` 选特征。

---

## 7. 元估计器的实现模式

### 7.1 模式 1：委托

最简单的元估计器——委托给包装的估计器：

```python
class MyMetaEstimator(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)  # clone 一份
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)  # 委托
```

### 7.2 模式 2：预处理 + 委托

```python
class MyMetaEstimator(BaseEstimator):
    def __init__(self, estimator, preprocess=True):
        self.estimator = estimator
        self.preprocess = preprocess

    def fit(self, X, y):
        if self.preprocess:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        if self.preprocess:
            X = self.scaler_.transform(X)
        return self.estimator_.predict(X)
```

### 7.3 模式 3：集成多个

```python
class MyEnsemble(BaseEstimator):
    def __init__(self, estimator, n_estimators=10):
        self.estimator = estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # 每次 clone + 采样 + fit
            est = clone(self.estimator)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            est.fit(X_sample, y_sample)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        # 投票
        predictions = [est.predict(X) for est in self.estimators_]
        return mode(predictions, axis=0).mode
```

### 7.4 模式 4：迭代优化

```python
class MyBoosting(BaseEstimator):
    def __init__(self, estimator, n_estimators=50):
        self.estimator = estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators_ = []
        y_residual = y.copy()
        for _ in range(self.n_estimators):
            est = clone(self.estimator)
            est.fit(X, y_residual)  # 拟合残差
            self.estimators_.append(est)
            y_residual -= est.predict(X)  # 更新残差
        return self

    def predict(self, X):
        return sum(est.predict(X) for est in self.estimators_)
```

---

## 8. 元估计器的嵌套

元估计器可以无限嵌套：

```python
# 嵌套：GridSearchCV(Pipeline([..., GridSearchCV(...)]))
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', GridSearchCV(
        LogisticRegression(),
        param_grid={'C': [0.1, 1, 10]},
    )),
])

# 甚至
outer_grid = GridSearchCV(
    pipe,
    param_grid={
        'pca__n_components': [5, 10, 20],
        'clf__C': [0.1, 1, 10],  # 内层 GridSearchCV 的 C
    },
)
```

这种"无限嵌套"的能力，是 sklearn 统一 API 的最大威力——任何估计器都能被元估计器包装，包括元估计器本身。

### 8.1 嵌套的参数访问

```python
# 三层嵌套
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GridSearchCV(
        LogisticRegression(),
        param_grid={'C': [0.1, 1, 10]},
    )),
])

# 访问最内层
pipe.get_params()
# {
#   'scaler': StandardScaler(),
#   'scaler__with_mean': True,
#   'clf': GridSearchCV(...),
#   'clf__estimator': LogisticRegression(),
#   'clf__estimator__C': 1.0,  # 三层
#   'clf__param_grid': {'C': [0.1, 1, 10]},
#   ...
# }

pipe.set_params(clf__estimator__C=5.0)  # 设置最内层
```

`clf__estimator__C` 串联三层——`Pipeline.clf.estimator.C`。

---

## 9. 与其他框架的对比

### 9.1 sklearn vs PyTorch 的 nn.Sequential

```python
# PyTorch 的 Sequential：类似 Pipeline
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

# 但 PyTorch 的 forward 统一走前向，没有 fit/predict 分链
model(x)  # 所有层都 forward
```

PyTorch 的 `Sequential` 串联层，但所有层都走 `forward`——没有 sklearn 的 fit 链 vs predict 链区分。因为 PyTorch 的层没有"学习"和"推理"的分离——`forward` 既用于训练也用于推理。

### 9.2 sklearn vs Keras 的 Sequential

```python
# Keras 的 Sequential
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(5),
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100)
```

Keras 的 `Sequential` 串联层，`fit` 训练所有层。和 sklearn 的 `Pipeline` 类似，但 Keras 的 `fit` 是迭代训练，sklearn 的 `fit` 是一次性。

### 9.3 sklearn vs HuggingFace 的 Pipeline

```python
# HuggingFace 的 Pipeline
from transformers import pipeline
nlp = pipeline("sentiment-analysis")
nlp("I love this!")
# [{'label': 'POSITIVE', 'score': 0.99}]
```

HuggingFace 的 `pipeline` 是高层封装，内部组合了 tokenizer + model + postprocessor。和 sklearn 的 `Pipeline` 类似，但 HuggingFace 的更专用（针对 NLP）。

---

## 10. 常见问题和陷阱

### 10.1 陷阱 1：忘了 clone

```python
# ❌ 不 clone，复用 estimator
class BadMeta(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y):
        self.estimator.fit(X, y)  # 直接 fit，改了原 estimator！
        return self

base = LogisticRegression()
meta = BadMeta(base)
meta.fit(X, y)
# base 现在已经 fit 了！用户可能不知道
```

正确做法：`clone` 后 fit。

### 10.2 陷阱 2：fit 链和 predict 链不一致

```python
# ❌ fit 时 transform，predict 时忘了
class BadPipeline(BaseEstimator):
    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)
        self.clf_ = LogisticRegression().fit(X, y)
        return self
    def predict(self, X):
        # 忘了 scaler_.transform(X)！
        return self.clf_.predict(X)  # 错！用未转换的 X 预测
```

### 10.3 陷阱 3：数据泄露

```python
# ❌ 在 fit 前用全量数据 fit 转换器
class BadMeta(BaseEstimator):
    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)  # 用全量 X
        # 然后交叉验证
        for train_idx, val_idx in cv.split(X):
            X_train = self.scaler_.transform(X[train_idx])
            # 错！scaler_ 用了全量 X（含 val）fit，数据泄露
```

正确做法：在每折内 fit 转换器，或用 `Pipeline` + `cross_val_score`。

### 10.4 陷阱 4：嵌套参数名写错

```python
# ❌ 用单下划线
grid = GridSearchCV(pipe, param_grid={'clf_C': [0.1, 1, 10]})  # 错！
# 应该是 clf__C（双下划线）

# ✅ 双下划线
grid = GridSearchCV(pipe, param_grid={'clf__C': [0.1, 1, 10]})
```

### 10.5 陷阱 5：步骤名重复

```python
# ❌ 步骤名重复
pipe = Pipeline([
    ('clf', LogisticRegression()),
    ('clf', SVC()),  # 重名！
])
# ValueError: Steps names must be unique
```

---

## 11. 实际使用模式

### 11.1 模式 1：标准流水线

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 11.2 模式 2：流水线 + 网格搜索

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
grid = GridSearchCV(pipe, param_grid={
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
})
grid.fit(X_train, y_train)
```

### 11.3 模式 3：集成

```python
voting = VotingClassifier([
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svm', SVC()),
])
voting.fit(X_train, y_train)
```

### 11.4 模式 4：特征选择 + 分类

```python
pipe = Pipeline([
    ('select', SelectKBest(k=10)),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

### 11.5 模式 5：多输出

```python
multi = MultiOutputClassifier(RandomForestClassifier())
multi.fit(X_train, Y_train)  # Y_train 是 (n_samples, n_outputs)
Y_pred = multi.predict(X_test)
```

---

## 12. 思考题和练习

### 12.1 思考题

1. 为什么 `Pipeline` 用组合而非继承？如果用继承会出什么问题？
2. `Pipeline.fit` 用 `fit_transform`，`predict` 用 `transform`。为什么不能都用 `transform`？
3. `GridSearchCV` 每次 `clone`，如果用 `deepcopy` 会出什么问题？
4. 嵌套参数用 `__`（双下划线）分隔，为什么不用 `.`（点）？
5. 元估计器可以无限嵌套，这种"统一性"的代价是什么？

### 12.2 练习

1. 实现一个简单的 `MyPipeline`，支持 fit 链和 predict 链。
2. 实现一个 `MyGridSearch`，用 `clone` + 交叉验证搜索参数。
3. 实现一个 `MyEnsemble`，用 bagging 组合多个估计器。

---

## 13. 深入：元估计器的实现细节

### 13.1 `Pipeline` 的完整实现思路

让我们更详细地看 `Pipeline` 的实现：

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    @property
    def named_steps(self):
        """用名字访问步骤的字典"""
        return dict(self.steps)

    def _validate_steps(self):
        """校验步骤"""
        names, estimators = zip(*self.steps)
        # 步骤名唯一
        if len(set(names)) != len(names):
            raise ValueError("Step names must be unique")
        # 步骤名不能含 __
        for name in names:
            if '__' in name:
                raise ValueError(f"Step name '{name}' contains '__'")
        # 最后一步必须是估计器（非 None）
        if estimators[-1] is None:
            raise ValueError("Last step must be an estimator")

    def fit(self, X, y=None):
        self._validate_steps()
        # fit 链
        for name, step in self.steps[:-1]:
            if step is not None:
                X = step.fit_transform(X, y)
        # 最后一步
        self.steps[-1][1].fit(X, y)
        return self

    def predict(self, X):
        # predict 链
        for name, step in self.steps[:-1]:
            if step is not None:
                X = step.transform(X)
        return self.steps[-1][1].predict(X)
```

注意几个细节：

1. **`named_steps` 属性**：用名字访问步骤，方便调试
2. **`_validate_steps`**：校验步骤名唯一、不含 `__`、最后一步是估计器
3. **`None` 步骤**：`('passthrough', None)` 表示跳过该步骤

### 13.2 `FeatureUnion`：并行特征合并

`FeatureUnion` 和 `Pipeline` 对应——`Pipeline` 串行，`FeatureUnion` 并行：

```python
from sklearn.pipeline import FeatureUnion

union = FeatureUnion([
    ('pca', PCA(n_components=5)),
    ('svd', TruncatedSVD(n_components=5)),
])

X_new = union.fit_transform(X)  # shape (n_samples, 10)
# PCA 输出 5 维 + SVD 输出 5 维 = 10 维
```

实现思路：

```python
class FeatureUnion(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None):
        # 并行：每个转换器独立 fit_transform
        results = [trans.fit_transform(X, y) for name, trans in self.transformer_list]
        # 横向拼接
        return np.hstack(results)

    def transform(self, X):
        # 并行：每个转换器独立 transform
        results = [trans.transform(X) for name, trans in self.transformer_list]
        return np.hstack(results)
```

`FeatureUnion` 把多个转换器的输出**横向拼接**，而 `Pipeline` 把数据**串行传递**。

### 13.3 `GridSearchCV` 的并行化

`GridSearchCV` 可以并行搜索：

```python
grid = GridSearchCV(
    LogisticRegression(),
    param_grid={'C': [0.1, 1, 10]},
    cv=5,
    n_jobs=-1,  # 并行，-1 表示用所有 CPU
)
```

实现思路：

```python
from joblib import Parallel, delayed

class GridSearchCV(BaseEstimator):
    def fit(self, X, y):
        # 生成所有 (params, fold) 组合
        tasks = [(params, fold) for params in self._iter() for fold in self._splits(X, y)]

        # 并行执行
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_and_score)(params, fold, X, y)
            for params, fold in tasks
        )

        # 聚合结果
        ...
```

`joblib.Parallel` 让网格搜索并行化，大幅加速。

---

## 14. 元估计器的设计模式总结

### 14.1 模式 1：包装器（Wrapper）

最简单的元估计器——包装一个估计器，添加行为：

```python
class Wrapper(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)
```

`GridSearchCV`、`RFE` 等都是包装器。

### 14.2 模式 2：流水线（Pipeline）

串联多个估计器，数据依次流过：

```python
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps
```

`Pipeline`、`FeatureUnion` 是流水线。

### 14.3 模式 3：集成（Ensemble）

组合多个估计器，聚合预测：

```python
class Ensemble(BaseEstimator):
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators_ = [clone(self.base_estimator).fit(...) for _ in range(self.n_estimators)]
```

`RandomForest`、`AdaBoost`、`VotingClassifier` 是集成。

### 14.4 模式 4：多输出（Multi-Output）

复制估计器处理多输出：

```python
class MultiOutput(BaseEstimator):
    def fit(self, X, Y):
        self.estimators_ = [clone(self.estimator).fit(X, Y[:, i]) for i in range(Y.shape[1])]
```

`MultiOutputClassifier`、`MultiOutputRegressor` 是多输出。

---

## 15. 元估计器的嵌套深度

元估计器可以无限嵌套，但实际中要注意：

### 15.1 嵌套的参数爆炸

```python
# 三层嵌套
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', GridSearchCV(
        LogisticRegression(),
        param_grid={'C': [0.1, 1, 10]},
        cv=5,
    )),
])

outer_grid = GridSearchCV(pipe, param_grid={
    'pca__n_components': [5, 10, 20],
}, cv=3)

# 总训练次数：
# 外层 GridSearchCV: 3 个 pca__n_components × 3 折 = 9 次
# 每次内层 GridSearchCV: 3 个 C × 5 折 = 15 次
# 总共: 9 × 15 = 135 次训练
```

嵌套越深，训练次数爆炸增长。实际中要控制嵌套深度。

### 15.2 嵌套的可读性

```python
# 一层嵌套：可读
GridSearchCV(LogisticRegression(), param_grid={'C': [0.1, 1, 10]})

# 两层嵌套：还行
GridSearchCV(Pipeline([('clf', LogisticRegression())]), param_grid={'clf__C': [0.1, 1, 10]})

# 三层嵌套：难读
GridSearchCV(Pipeline([('clf', GridSearchCV(LogisticRegression(), ...))]), ...)
```

嵌套太深，代码可读性下降。建议把深层嵌套拆成变量：

```python
# 拆成变量，可读
inner_grid = GridSearchCV(LogisticRegression(), param_grid={'C': [0.1, 1, 10]})
pipe = Pipeline([('scaler', StandardScaler()), ('clf', inner_grid)])
outer_grid = GridSearchCV(pipe, param_grid={'pca__n_components': [5, 10, 20]})
```

---

## 16. 与其他生态的元估计器对比

### 16.1 PyTorch 的 `nn.Sequential`

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)
```

类似 `Pipeline`，但所有层都走 `forward`，没有 fit 链 vs predict 链区分。

### 16.2 Keras 的 `Sequential`

```python
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(5),
])
```

类似 `Pipeline`，但 `fit` 是迭代训练，不是一次性。

### 16.3 HuggingFace 的 `pipeline`

```python
nlp = pipeline("sentiment-analysis", model="bert-base-uncased")
```

高层封装，内部组合 tokenizer + model + postprocessor。比 sklearn 的 `Pipeline` 更专用。

### 16.4 对比表

| 框架 | 元估计器 | fit/predict 区分 | 嵌套支持 |
|------|---------|-----------------|---------|
| sklearn | `Pipeline` | 有（fit 链 vs predict 链） | 无限嵌套 |
| PyTorch | `nn.Sequential` | 无（都走 forward） | 无限嵌套 |
| Keras | `Sequential` | 无（fit 迭代） | 有限 |
| HuggingFace | `pipeline` | 有（train vs infer） | 不支持 |

sklearn 的 `Pipeline` 独特之处在于 fit 链 vs predict 链的区分——这源于传统 ML 的"学习"和"推理"分离。

### 16.5 嵌套能力的本质对比

不同框架的"嵌套"能力差异，根源在于抽象层次：

- **sklearn**：估计器是一等公民，可任意组合、嵌套、传递
- **PyTorch**：`Module` 是一等公民，`Sequential` 组合 `Module`
- **Keras**：`Layer` 是一等公民，`Sequential` 组合 `Layer`
- **HuggingFace**：`Pipeline` 是高层封装，不支持再嵌套

sklearn 的嵌套能力最强，因为它的抽象（估计器）最通用——任何有 `fit`/`predict` 的对象都是估计器。

### 16.6 为什么 sklearn 的嵌套特别强大

sklearn 嵌套强大的三个原因：

1. **统一 API**：所有估计器接口一致，元估计器能包装任何估计器
2. **参数管理**：`get_params`/`set_params` 支持嵌套参数，`GridSearchCV` 能搜内部参数
3. **clone 机制**：`clone` 递归复制嵌套结构，状态隔离

这三者缺一不可——统一 API 让包装可能，参数管理让搜索可能，clone 让状态隔离可能。

---

## 17. 小结

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| 关系 | 组合而非继承 | 动态组合、避免菱形继承 |
| 参数命名 | `step__param` | 支持嵌套搜索 |
| fit vs predict | 不同链路 | 防止数据泄露 |
| clone 的使用 | 每次搜索 clone | 防止状态污染 |

**核心洞察**：元估计器是"估计器的估计器"——它把估计器当参数，用组合而非继承构建更复杂的行为。这种"一切皆估计器"的统一性，让 sklearn 可以无限嵌套：`GridSearchCV(Pipeline([('scaler', ...), ('clf', ...)]))`。

### 17.1 本讲要点回顾

1. **元估计器包装估计器**：`__init__` 接收估计器作为参数。
2. **组合优于继承**：动态组合、避免菱形继承、正确表达 has-a。
3. **fit 链 vs predict 链**：fit 时学习+转换，predict 时只转换，防泄露。
4. **`step__param` 嵌套命名**：支持 `GridSearchCV` 搜索内部参数。
5. **`clone` 防状态污染**：每次搜索 clone 干净副本。
6. **无限嵌套**：元估计器可包装元估计器，统一 API 的威力。
7. **五大类型**：流水线、调参、集成、多输出、特征选择。

### 17.2 思考延伸

- 元估计器的"无限嵌套"是统一 API 的最大红利，但也带来参数爆炸和可读性挑战
- 组合而非继承是元估计器的核心设计，正确表达 has-a 关系
- fit 链 vs predict 链的区分是 sklearn 独有，源于传统 ML 的学习/推理分离
- `clone` 递归复制是状态隔离的关键，没有它嵌套就会共享状态
- `step__param` 命名规则是嵌套参数搜索的基础，`__` 分隔符经过深思熟虑
- 元估计器是 sklearn 生态繁荣的核心，让简单算法组合出复杂能力

---

## 上一讲 / 下一讲


[← 第三讲：参数管理机制](03-parameter-management.md) ｜  [第五讲：数据约定与校验 →](05-data-convention.md）
