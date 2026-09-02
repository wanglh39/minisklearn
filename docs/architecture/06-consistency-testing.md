# 第六讲：一致性测试机制

> **核心问题**：为什么 sklearn 能写一个测试套件测所有算法？`check_estimator` 的设计思想是什么？为什么"能写通用测试"本身就是架构成熟度的标志？

---

## 1. 统一契约的回报：通用测试套件

因为所有估计器遵循同一 API 契约（`fit` / `predict` / `transform` / `get_params` / `clone`），sklearn 可以写**一套测试**检查所有估计器：

```python
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression

# 一行代码测试一个估计器的所有契约
check_estimator(LogisticRegression)
```

这一行代码背后是**几十项检查**，覆盖了 API 契约的方方面面。

### 1.1 它测试了什么

`check_estimator` 实际测试的内容（部分）：

- `fit` 后 `predict` 能正常工作
- `get_params` / `set_params` 互逆
- `clone` 产生干净副本
- `__repr__` 不崩溃
- 多维 / 一维输入处理
- `fit` 两次结果一致（幂等性）
- `predict` 不改状态（纯查询）
- `set_params` 部分更新正确
- 多次 `fit` 覆盖之前的结果
- `score` 返回合理范围
- `predict_proba` 和为 1（分类器）
- `decision_function` 形状正确
- 稀疏矩阵输入处理
- DataFrame 输入处理
- 标签字符串 / 整数都能处理
- `n_features_in_` 正确记录
- ...等几十项检查

### 1.2 一行代码测几十项的价值

假设 sklearn 有 100 个估计器，每项契约测试要 10 行代码。

- **没有通用测试**：100 × 30 项 × 10 行 = 30000 行测试代码。
- **有通用测试**：30 项 × 10 行 + 100 × 1 行 = 400 行测试代码。

更重要的是，**新增一个估计器只需要 1 行**：

```python
check_estimator(MyNewEstimator)
```

这 1 行自动覆盖 30 项契约，开发者只需专注算法本身的测试。

### 1.3 历史背景

sklearn 早期（0.10 之前）每个估计器有自己的测试文件，重复代码很多。0.10 引入 `check_estimator` 后，测试代码量骤减，新增估计器的门槛也降低。这是 sklearn 能快速积累上百种算法的重要原因之一。

这个理念后来被很多框架借鉴：

- imbalanced-learn 的 `check_estimator` 适配版。
- scikit-learn-contrib 项目都要求通过 `check_estimator`。
- Julia MLJ 也有类似的 `fit` / `predict` 契约测试。

### 1.4 思考题

1. 如果 sklearn 没有统一 API，`check_estimator` 还能存在吗？
2. `check_estimator` 测的是"契约"还是"算法正确性"？两者区别是什么？
3. 为什么 imbalanced-learn 的估计器不能直接用 sklearn 的 `check_estimator`？

---

## 2. 为什么这是架构成熟度的标志？

能写通用测试套件，说明 API 契约**足够统一**。反过来，如果每个算法的接口都不一样，你就得为每个算法写专门的测试。

这是"约定优于配置"的最高回报：**约定让通用工具成为可能**。

### 2.1 约定 vs 配置的回报曲线

| 约定程度 | 通用工具 | 测试代码 | 新增成本 | 例子            |
|----------|----------|----------|----------|------------------|
| 无约定   | 不可能   | 极多     | 极高     | 早期 Weka       |
| 弱约定   | 少量     | 多       | 高       | Keras 早期      |
| 强约定   | 丰富     | 少       | 低       | sklearn         |
| 过强约定 | 丰富但僵 | 少       | 低但僵   | （理想中不存在）|

sklearn 处在"强约定但不过僵"的甜点：约定足够强以支撑通用工具，又足够灵活以容纳各种算法。

### 2.2 通用工具的连锁反应

统一契约不仅让通用测试成为可能，还连锁地让以下工具成为可能：

- **Pipeline**：任意串联若干估计器。
- **GridSearchCV**：任意搜索任意估计器的超参数。
- **cross_val_score**：任意交叉验证任意估计器。
- **clone**：任意复制任意估计器。
- **pickle**：任意序列化任意估计器。

这些工具都假设"输入是个 sklearn 估计器"，而不关心具体是哪种。`check_estimator` 是这个生态的**守门人**——通过它的估计器自动获得所有通用工具的支持。

### 2.3 与其他框架对比

| 框架     | 统一契约 | 通用测试 | 通用工具 | 新增算法成本 |
|----------|----------|----------|----------|--------------|
| sklearn  | 强       | 有       | 丰富     | 低           |
| PyTorch  | 弱       | 无       | 少       | 高（写训练循环）|
| Keras    | 中       | 少       | 中       | 中           |
| R caret  | 中       | 少       | 中       | 中           |
| Weka     | 中       | 有       | 中       | 中           |

### 2.4 思考题

1. "约定优于配置"在 sklearn 里体现在哪里？有没有"配置"的部分？
2. 如果一个新算法天然不符合 `fit` / `predict` 契约（例如强化学习），该怎么融入 sklearn？
3. 通用工具的"连锁反应"有没有反面——约定带来的限制？

---

## 3. 自己实现：mini `check_estimator`

```python
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification

def check_estimator(EstimatorClass):
    """通用估计器测试套件。"""
    est = EstimatorClass()

    # 测试 1：get_params / set_params 互逆
    params = est.get_params()
    est.set_params(**params)
    assert est.get_params() == params, "get_params/set_params 不互逆"

    # 测试 2：clone 产生干净副本
    est_clone = clone(est)
    assert type(est_clone) == type(est), "clone 类型变了"
    assert est_clone.get_params() == est.get_params(), "clone 参数变了"

    # 测试 3：fit 后 predict 不崩溃
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y_pred.shape[0] == X.shape[0], "predict 输出长度不对"

    # 测试 4：predict 不改状态
    coef_before = est.coef_.copy()
    est.predict(X)
    assert np.allclose(est.coef_, coef_before), "predict 改了状态"

    # 测试 5：fit 幂等（同样数据 fit 两次结果一致）
    est.fit(X, y)
    coef_1 = est.coef_.copy()
    est.fit(X, y)
    coef_2 = est.coef_.copy()
    assert np.allclose(coef_1, coef_2), "fit 不幂等"
```

### 3.1 各项测试详解

#### 3.1.1 get_params / set_params 互逆

```python
params = est.get_params()
est.set_params(**params)
assert est.get_params() == params
```

这测试的是参数管理的**自洽性**：拿出来的参数能塞回去，且塞回去后拿出来的还是一样的。

失败示例：

```python
class BrokenEstimator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_params(self):
        return {"alpha": self.alpha * 2}   # 坏：拿出来的不是原值

    def set_params(self, **params):
        self.alpha = params["alpha"]
        return self

# 测试失败：get_params 返回 2.0，set_params(2.0) 后 get_params 返回 4.0
```

#### 3.1.2 clone 产生干净副本

```python
est_clone = clone(est)
assert type(est_clone) == type(est)
assert est_clone.get_params() == est.get_params()
```

`clone` 要产生一个**未训练**的副本，超参数相同但无学出属性。

失败示例：

```python
class BrokenEstimator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None   # 坏：__init__ 里就有学出属性

# clone 后 coef_ 还在，不是"干净"副本
```

#### 3.1.3 fit 后 predict 不崩溃

```python
est.fit(X, y)
y_pred = est.predict(X)
assert y_pred.shape[0] == X.shape[0]
```

最基本的契约：能 fit，能 predict，输出长度对。

失败示例：

```python
class BrokenEstimator:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X) - 1)   # 坏：少一个

# 测试失败：输出长度 != X.shape[0]
```

#### 3.1.4 predict 不改状态

```python
coef_before = est.coef_.copy()
est.predict(X)
assert np.allclose(est.coef_, coef_before)
```

`predict` 应该是**纯查询**，不改模型状态。这保证多次 predict 结果一致。

失败示例：

```python
class BrokenEstimator:
    def predict(self, X):
        self.coef_ += 0.001   # 坏：predict 改了 coef_
        return X @ self.coef_
```

#### 3.1.5 fit 幂等

```python
est.fit(X, y)
coef_1 = est.coef_.copy()
est.fit(X, y)
coef_2 = est.coef_.copy()
assert np.allclose(coef_1, coef_2)
```

同样数据 fit 两次，结果应该一样。这排除了"fit 累加状态"的 bug。

失败示例：

```python
class BrokenEstimator:
    def fit(self, X, y):
        if not hasattr(self, 'coef_'):
            self.coef_ = np.zeros(X.shape[1])
        self.coef_ += np.linalg.lstsq(X, y, rcond=None)[0]   # 坏：累加
        return self
```

### 3.2 测试的组织

真实的 `check_estimator` 把每项测试拆成独立函数，用 pytest 的参数化机制对每个估计器跑一遍：

```python
import pytest

@pytest.mark.parametrize("EstimatorClass", [
    LogisticRegression, LinearRegression, DecisionTreeClassifier, ...
])
def test_get_params_set_params_inverse(EstimatorClass):
    est = EstimatorClass()
    params = est.get_params()
    est.set_params(**params)
    assert est.get_params() == params

@pytest.mark.parametrize("EstimatorClass", [...])
def test_clone_clean(EstimatorClass):
    ...

@pytest.mark.parametrize("EstimatorClass", [...])
def test_fit_predict(EstimatorClass):
    ...
```

这样每个失败能精确定位到"哪个估计器违反了哪条契约"。

### 3.3 思考题

1. 为什么把每项测试拆成独立函数，而不是一个大函数？
2. `check_estimator` 用 `make_classification` 生成测试数据，为什么不用真实数据？
3. 如果一个估计器天然不幂等（例如随机算法不设 random_state），怎么通过测试？

---

## 4. check_estimator 测试项详解

### 4.1 get_params / set_params 互逆

```python
def test_get_params_set_params_inverse(est):
    params = est.get_params()
    est.set_params(**params)
    assert est.get_params() == params
```

变体：部分设置后其他参数不变。

```python
def test_set_params_partial(est):
    original = est.get_params()
    est.set_params(alpha=999)
    new = est.get_params()
    assert new["alpha"] == 999
    for k in original:
        if k != "alpha":
            assert new[k] == original[k]
```

### 4.2 clone 干净副本

```python
def test_clone_clean(est):
    est_clone = clone(est)
    assert type(est_clone) == type(est)
    assert est_clone.get_params() == est.get_params()
    # clone 后不应该有学出属性
    for attr in vars(est_clone):
        assert not attr.endswith("_"), f"clone 后有学出属性 {attr}"
```

### 4.3 fit 后 predict

```python
def test_fit_predict(est, X, y):
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y_pred.shape[0] == X.shape[0]
    # 分类器：预测值在类别集合里
    if hasattr(est, 'classes_'):
        assert set(np.unique(y_pred)).issubset(set(est.classes_))
```

### 4.4 predict 不改状态

```python
def test_predict_no_mutation(est, X, y):
    est.fit(X, y)
    state_before = {k: v.copy() if hasattr(v, 'copy') else v
                    for k, v in vars(est).items()}
    est.predict(X)
    for k, v in vars(est).items():
        if k in state_before:
            assert np.allclose(v, state_before[k]), f"predict 改了 {k}"
```

### 4.5 fit 幂等

```python
def test_fit_idempotent(est, X, y):
    est.fit(X, y)
    state_1 = {k: v.copy() if hasattr(v, 'copy') else v
               for k, v in vars(est).items()}
    est.fit(X, y)
    state_2 = {k: v.copy() if hasattr(v, 'copy') else v
               for k, v in vars(est).items()}
    for k in state_1:
        assert np.allclose(state_1[k], state_2[k]), f"fit 不幂等于 {k}"
```

### 4.6 多维输入

```python
def test_fit_predict_2d(est, X_2d, y):
    est.fit(X_2d, y)
    est.predict(X_2d)   # 不应崩溃

def test_fit_predict_1d_feature(est, X_1d_feature, y):
    # X shape (n_samples, 1)
    est.fit(X_1d_feature, y)
    est.predict(X_1d_feature)
```

### 4.7 一维输入（单样本）

```python
def test_predict_single_sample(est, X, y):
    est.fit(X, y)
    x_single = X[:1]   # shape (1, n_features)
    y_pred = est.predict(x_single)
    assert y_pred.shape == (1,)
```

### 4.8 repr 不崩溃

```python
def test_repr(est):
    s = repr(est)
    assert isinstance(s, str)
    # repr 应该能 eval 回来（理想情况）
    # assert eval(s) == est  # 太严格，不强制
```

### 4.9 set_params 部分更新

```python
def test_set_params_partial_update(est):
    original = est.get_params()
    est.set_params(C=999)
    assert est.C == 999
    # 其他参数不变
    for k, v in original.items():
        if k != "C":
            assert est.get_params()[k] == v
```

### 4.10 多次 fit 覆盖

```python
def test_refit_overwrites(est, X1, y1, X2, y2):
    est.fit(X1, y1)
    coef_1 = est.coef_.copy()
    est.fit(X2, y2)   # 用新数据重新 fit
    coef_2 = est.coef_.copy()
    # 第二次 fit 应该完全覆盖第一次，不是累加
    assert not np.allclose(coef_1, coef_2)  # 一般不同
```

### 4.11 proba 和为 1（分类器）

```python
def test_predict_proba_sums_to_one(clf, X, y):
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
```

### 4.12 score 合理范围

```python
def test_score_range(clf, X, y):
    clf.fit(X, y)
    score = clf.score(X, y)
    # 分类器：accuracy 在 [0, 1]
    # 回归器：R² 可能为负，但不应是 nan/inf
    assert np.isfinite(score)
```

### 4.13 稀疏矩阵输入

```python
def test_sparse_input(est, X_sparse, y):
    est.fit(X_sparse, y)
    est.predict(X_sparse)   # 不应崩溃
```

### 4.14 DataFrame 输入

```python
def test_dataframe_input(est, df, y):
    est.fit(df, y)
    est.predict(df)
    # feature_names_in_ 应该记录列名
    assert hasattr(est, 'feature_names_in_')
```

### 4.15 标签类型

```python
def test_string_labels(clf, X, y_str):
    clf.fit(X, y_str)
    y_pred = clf.predict(X)
    assert set(np.unique(y_pred)).issubset(set(y_str))

def test_int_labels(clf, X, y_int):
    clf.fit(X, y_int)
    y_pred = clf.predict(X)
    assert set(np.unique(y_pred)).issubset(set(y_int))
```

### 4.16 n_features_in_ 一致性

```python
def test_n_features_in_consistency(est, X, y):
    est.fit(X, y)
    assert est.n_features_in_ == X.shape[1]
    # predict 时特征数不一致应报错
    X_wrong = X[:, :-1]
    with pytest.raises(ValueError):
        est.predict(X_wrong)
```

### 4.17 思考题

1. 为什么 `test_predict_no_mutation` 要检查**所有**属性，而不只是 `coef_`？
2. `test_fit_idempotent` 对随机算法（无 random_state）会失败吗？怎么处理？
3. `test_repr` 不要求 `eval(repr(est)) == est`，为什么？什么情况下能做到？

---

## 5. 参数化测试与 pytest

### 5.1 pytest 的参数化

```python
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ALL_ESTIMATORS = [
    LogisticRegression,
    LinearRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
]

@pytest.mark.parametrize("EstimatorClass", ALL_ESTIMATORS)
def test_check_estimator(EstimatorClass):
    check_estimator(EstimatorClass)
```

每个估计器对每项测试跑一遍，失败时 pytest 报告"哪个估计器哪项失败"。

### 5.2 跳过特定测试

某些估计器天然不满足某些契约，需要跳过：

```python
@pytest.mark.parametrize("EstimatorClass", ALL_ESTIMATORS)
def test_sparse_input(EstimatorClass):
    est = EstimatorClass()
    if not supports_sparse(est):
        pytest.skip(f"{EstimatorClass.__name__} 不支持稀疏输入")
    check_sparse(est)
```

sklearn 用 `@pytest.mark.parametrize` + `pytest.skip` 精细控制。

### 5.3 标记预期失败

```python
@pytest.mark.xfail(reason="已知 bug，待修复")
def test_known_issue(est):
    ...
```

`xfail` 标记"预期失败"，测试失败不算 regression，测试通过反而提醒"bug 修了"。

### 5.4 思考题

1. 参数化测试和循环测试有什么区别？为什么用参数化？
2. `pytest.skip` 和 `pytest.xfail` 有什么区别？各自适用什么场景？
3. 怎么只对特定估计器跑特定测试？

---

## 6. 测试的分层

sklearn 的测试分三层：

### 6.1 契约测试（contract tests）

`check_estimator` 测的是 API 契约，不关心算法正确性。所有估计器都跑。

```python
check_estimator(LogisticRegression)   # 契约测试
```

### 6.2 算法测试（algorithm tests）

每个算法有自己的正确性测试，例如 LogisticRegression 要和 statsmodels 对比：

```python
def test_logistic_regression_vs_statsmodels():
    clf = LogisticRegression().fit(X, y)
    # 对比系数
    assert np.allclose(clf.coef_, statsmodels_coef, atol=1e-4)
```

### 6.3 集成测试（integration tests）

测试估计器在 Pipeline / GridSearchCV 等元估计器中的行为：

```python
def test_logistic_in_pipeline():
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression())])
    pipe.fit(X, y)
    pipe.predict(X)
```

### 6.4 三层的关系

| 层次       | 测什么               | 谁来跑         | 例子                |
|------------|----------------------|----------------|----------------------|
| 契约测试   | API 契约             | 所有估计器     | `check_estimator`   |
| 算法测试   | 算法正确性           | 单个估计器     | 对比 statsmodels    |
| 集成测试   | 组合行为             | 估计器 + 元估计器 | Pipeline 端到端     |

契约测试是**第一道防线**——通不过契约测试，就不用谈算法正确性。

### 6.5 思考题

1. 契约测试和算法测试的边界在哪？`predict_proba` 和为 1 算哪个？
2. 为什么集成测试不也做成通用的？
3. 如果一个估计器通过了契约测试但算法测试失败，说明什么？

---

## 7. 常见失败模式

### 7.1 `__init__` 做了工作

```python
class Bad:
    def __init__(self, C=1.0):
        self.C = C
        self._validate_C()   # 坏：__init__ 里做工作

# clone 后 _validate_C 又跑一遍，可能副作用
```

修复：把 `__validate_C` 移到 `fit` 里。

### 7.2 学出属性没加下划线

```python
class Bad:
    def fit(self, X, y):
        self.coef = ...   # 坏：没有下划线
```

`get_params` 会把 `coef` 当超参数，`clone` 后还在，破坏语义。

### 7.3 predict 改了状态

```python
class Bad:
    def predict(self, X):
        self.n_calls_ += 1   # 坏：predict 改状态
        return X @ self.coef_
```

`test_predict_no_mutation` 会失败。

### 7.4 fit 不幂等

```python
class Bad:
    def fit(self, X, y):
        if hasattr(self, 'coef_'):
            self.coef_ += solve(X, y)   # 坏：累加
        else:
            self.coef_ = solve(X, y)
```

`test_fit_idempotent` 会失败。

### 7.5 random_state 没生效

```python
class Bad:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState()   # 坏：没用 self.random_state
        self.subset = rng.choice(len(X), 10)
```

两次 fit 结果不同，`test_fit_idempotent` 会失败。

### 7.6 不接受 2D 输入

```python
class Bad:
    def fit(self, X, y):
        if X.ndim != 1:   # 坏：只接受 1D
            raise ValueError
```

`test_fit_predict_2d` 会失败。

### 7.7 思考题

1. 这些失败模式里，哪个最常见？哪个最难发现？
2. `random_state` 没生效为什么会被 `test_fit_idempotent` 抓到？
3. 写一个故意违反每条契约的"坏估计器"，验证 `check_estimator` 能抓到。

---

## 8. 自定义估计器测试

### 8.1 注册到 check_estimator

```python
from sklearn.utils.estimator_checks import check_estimator

class MyEstimator(BaseEstimator, ClassifierMixin):
    ...

check_estimator(MyEstimator)   # 跑通用契约测试
```

### 8.2 跳过不适用的测试

```python
from sklearn.utils.estimator_checks import check_estimator

# 某些测试不适用
check_estimator(MyEstimator, expected_failed_checks={
    "check_sample_weights_invariance": "我的估计器不支持 sample_weight",
})
```

### 8.3 自定义契约测试

如果你的估计器有额外契约（例如"必须支持稀疏"），写自己的测试：

```python
def test_my_estimator_sparse():
    est = MyEstimator()
    X_sparse = csr_matrix(X_dense)
    est.fit(X_sparse, y)
    est.predict(X_sparse)
```

### 8.4 算法正确性测试

```python
def test_my_estimator_correctness():
    est = MyEstimator()
    est.fit(X_train, y_train)
    # 对比已知正确答案
    assert np.allclose(est.coef_, true_coef, atol=1e-4)
```

### 8.5 思考题

1. `expected_failed_checks` 和 `pytest.skip` 有什么区别？
2. 自定义契约测试应该放在哪？和 `check_estimator` 什么关系？
3. 怎么测试一个估计器在 Pipeline 里的行为？

---

## 9. 性能测试与基准

### 9.1 时间复杂度测试

```python
def test_logistic_regression_time_complexity():
    sizes = [100, 1000, 10000]
    times = []
    for n in sizes:
        X, y = make_classification(n_samples=n)
        t = timeit.timeit(lambda: LogisticRegression().fit(X, y), number=5)
        times.append(t)
    # 验证时间增长不超过 O(n^1.5)
    ratio = times[2] / times[0]
    assert ratio < (10000 / 100) ** 1.5
```

### 9.2 内存占用测试

```python
def test_kmeans_memory():
    import tracemalloc
    tracemalloc.start()
    KMeans(n_clusters=10).fit(X_large)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100 * 1024 * 1024   # 不超过 100MB
```

### 9.3 数值稳定性测试

```python
def test_logistic_regression_stability():
    # 极大特征值
    X = np.array([[1e10, 0], [0, 1e10], [-1e10, 0], [0, -1e10]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression().fit(X, y)
    assert np.isfinite(clf.coef_).all()
```

### 9.4 思考题

1. 性能测试和正确性测试的优先级哪个更高？
2. 数值稳定性测试为什么要用极端输入？
3. 怎么写一个"回归测试"防止性能退化？

---

## 10. 与其他框架测试对比

### 10.1 PyTorch

PyTorch 没有统一的 `check_estimator`，每个模型自己写测试：

```python
class TestLinear(unittest.TestCase):
    def test_forward(self):
        model = nn.Linear(10, 5)
        x = torch.randn(100, 10)
        y = model(x)
        assert y.shape == (100, 5)
```

因为 PyTorch 没有"统一 API 契约"，也就没有通用测试。

### 10.2 TensorFlow / Keras

Keras 有部分统一测试，但不如 sklearn 系统化：

```python
def test_dense_layer():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=1)
```

### 10.3 R testthat

R 的 `testthat` 包做单元测试，但没有"通用契约测试"理念：

```r
test_that("lm works", {
  model <- lm(y ~ x, data = df)
  expect_equal(length(model$coefficients), 2)
})
```

### 10.4 对比表

| 框架     | 通用契约测试 | 测试组织       | 参数化测试 | 例子                |
|----------|--------------|----------------|------------|----------------------|
| sklearn  | 有           | pytest 参数化  | 强         | `check_estimator`   |
| PyTorch  | 无           | unittest       | 弱         | 每个模型自己写      |
| Keras    | 部分         | pytest         | 中         | layer 测试          |
| R        | 无           | testthat       | 弱         | 每个模型自己写      |

### 10.5 思考题

1. 为什么 PyTorch 没有类似 `check_estimator` 的机制？
2. 如果要给 PyTorch 设计一个 `check_module`，会检查哪些契约？
3. sklearn 的测试理念能移植到非 ML 框架吗？举一个例子。

---

## 11. 测试的哲学

### 11.1 测试即文档

`check_estimator` 的测试项列表，本身就是 API 契约的文档。读测试代码就能知道"估计器应该满足什么"。

### 11.2 测试即设计

写 `check_estimator` 的过程，逼迫 sklearn 团队明确"到底什么是估计器"。这个明确化反过来指导新算法的设计。

### 11.3 测试即契约 enforcement

没有 `check_estimator`，API 契约只是纸上的文字。有了它，契约是**可执行的**——违反就报错。

### 11.4 测试的经济学

| 投入                 | 回报                           |
|----------------------|--------------------------------|
| 写契约测试（一次性） | 所有估计器自动检查（持续）     |
| 写通用工具（一次性） | 所有估计器自动支持（持续）     |
| 维护契约（持续）     | 生态健康（持续）               |

sklearn 的测试投入是**一次投入、持续回报**的典型。

### 11.5 测试的反模式

#### 11.5.1 测试依赖执行顺序

```python
# 坏：test_2 依赖 test_1 的副作用
def test_1():
    clf.fit(X, y)

def test_2():
    clf.predict(X)   # 依赖 test_1 已经 fit
```

每个测试应该独立，不依赖其他测试的执行。

#### 11.5.2 测试依赖外部数据

```python
# 坏：依赖本地文件
def test_fit():
    X = np.load("/home/user/data/X.npy")   # 别人跑不了
```

应该用合成数据或包内测试数据。

#### 11.5.3 测试不清理状态

```python
# 坏：改了全局配置不恢复
def test_config():
    sklearn.set_config(assume_finite=True)
    # 没恢复，影响后续测试
```

用 `with config_context(...)` 或 `pytest` fixture 确保清理。

### 11.6 思考题

1. "测试即文档"和"文档即测试"是一回事吗？
2. 如果契约测试太严格，会不会限制算法创新？
3. 怎么平衡"测试覆盖率高"和"测试维护成本"？
4. 测试依赖执行顺序会有什么后果？怎么检测？
5. 怎么用 pytest fixture 保证测试独立性和清理？

---

## 12. 实战：给自定义估计器写测试

```python
import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

class MeanRegressor(BaseEstimator, RegressorMixin):
    """永远预测训练集均值的回归器。"""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = y.mean()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'mean_')
        X = check_array(X, dtype=np.float64)
        return np.full(X.shape[0], self.mean_)


# 1. 跑通用契约测试
check_estimator(MeanRegressor)


# 2. 算法正确性测试
def test_mean_regressor_correctness():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    reg = MeanRegressor().fit(X, y)
    assert reg.mean_ == pytest.approx(y.mean())
    y_pred = reg.predict(X)
    assert np.all(y_pred == y.mean())


# 3. 边界测试
def test_mean_regressor_single_sample():
    X = np.array([[1.0, 2.0]])
    y = np.array([5.0])
    reg = MeanRegressor().fit(X, y)
    assert reg.predict(X)[0] == 5.0


# 4. 异常测试
def test_mean_regressor_nan_y():
    X = np.random.randn(10, 3)
    y = np.array([1, 2, np.nan, 3, 4, 5, 6, 7, 8, 9])
    reg = MeanRegressor()
    # 应该能处理或报清晰错误
    with pytest.raises((ValueError, FloatingPointError)):
        reg.fit(X, y)
```

### 12.1 思考题

1. `MeanRegressor` 通过了 `check_estimator` 吗？哪些测试可能失败？
2. 给 `MeanRegressor` 加 `transform` 方法，让它也通过 `TransformerMixin` 的契约测试。
3. 写一个 `MedianRegressor`（预测中位数），对比它的测试和 `MeanRegressor`。

---

## 13. 常见问题与陷阱

### 13.1 check_estimator 报错怎么办

```python
check_estimator(MyEstimator)
# → AssertionError: fit 不幂等
```

按报错信息定位：

1. 看是哪项测试失败。
2. 读该项测试的代码，理解它检查什么。
3. 修估计器，重跑。

### 13.2 某项测试不适用怎么办

```python
check_estimator(MyEstimator, expected_failed_checks={
    "check_sample_weights_invariance": "不支持 sample_weight",
})
```

### 13.3 随机算法怎么通过幂等测试

```python
class RandomEstimator:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)   # 用 self.random_state
        self.subset_ = rng.choice(len(X), 10)
        return self

# 设了 random_state 后，fit 幂等
```

### 13.4 测试太慢怎么办

```python
# 用小数据集测试
check_estimator(MyEstimator, n_samples=50, n_features=5)
```

### 13.5 测试与生态兼容性

第三方估计器（imbalanced-learn、scikit-learn-extra 等）要进入 sklearn 生态，必须通过 `check_estimator`。这是生态的**质量门槛**。

```python
# imbalanced-learn 的估计器
from imblearn.over_sampling import SMOTE
check_estimator(SMOTE)   # 可能部分失败，因为 SMOTE 改变样本数
```

imbalanced-learn 的估计器天然违反某些 sklearn 契约（例如 SMOTE 的 `fit_resample` 不是标准 `fit_transform`），所以有自己的 `check_estimator` 变体。

### 13.6 思考题

1. `check_estimator` 报"predict 改了状态"，怎么定位是哪个属性被改了？
2. 一个估计器在 0.20 通过测试，0.24 不通过，可能是什么原因？
3. 怎么只跑 `check_estimator` 的某一项测试？
4. 第三方库怎么声明"我兼容 sklearn 的哪些契约"？
5. 如果一个估计器天然违反某条契约（例如 SMOTE），应该改契约还是改估计器？

---

## 14. 进阶：契约测试的元层次

### 14.1 测试测试器

`check_estimator` 本身也需要测试——验证它能正确地"抓出"违反契约的估计器：

```python
def test_check_estimator_catches_violations():
    class Bad:
        def fit(self, X, y):
            self.coef_ = np.zeros(X.shape[1])
            return self
        def predict(self, X):
            self.coef_ += 1   # 坏：改状态
            return X @ self.coef_

    with pytest.raises(AssertionError):
        check_estimator(Bad)
```

### 14.2 契约的演化

sklearn 的契约在演化，`check_estimator` 也跟着演化：

- 0.20：加入 `n_features_in_` 检查。
- 0.22：加入 `feature_names_in_` 检查。
- 0.24：加入更多稀疏矩阵检查。

每次契约加严，所有估计器都要更新通过新测试。这是 sklearn 维护成本的大头。

### 14.3 契约测试与 deprecation

```python
# 旧契约：fit_transform 接受 y
# 新契约：fit_transform 不接受 y（SLEP011）

# 过渡期：旧估计器仍能工作，但有 deprecation warning
# 新估计器必须遵守新契约
```

### 14.4 思考题

1. "测试测试器"听起来很元，它的价值是什么？
2. 契约加严时，怎么平衡"向后兼容"和"契约统一"？
3. 如果让你给 sklearn 加一条新契约，你会加什么？怎么更新 `check_estimator`？

---

## 15. 测试驱动开发与 sklearn

### 15.1 TDD 流程在 sklearn 的应用

写一个新估计器时，推荐 TDD 流程：

1. **先写契约测试**：`check_estimator(MyEstimator)`，此时全部失败（估计器还没写）。
2. **写最小实现**：让 `fit` / `predict` 能跑通，契约测试逐项变绿。
3. **写算法测试**：对比已知正确答案。
4. **优化算法**：保持测试绿色。

```python
# 步骤 1：先写测试
class MyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    # fit / predict 还没写

check_estimator(MyRegressor)   # 全部失败

# 步骤 2：写最小实现
class MyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        X = check_array(X)
        self.mean_ = y.mean()
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.full(X.shape[0], self.mean_)

check_estimator(MyRegressor)   # 大部分通过
```

### 15.2 契约先行的价值

先写契约测试的好处：

- **明确接口**：写测试时必须想清楚 `fit` / `predict` 的签名和返回。
- **即时反馈**：每加一个方法，立刻知道是否满足契约。
- **防止偏离**：写算法时容易"顺手"违反契约，测试能立刻发现。

### 15.3 测试与重构

有了契约测试，重构时敢于大改内部实现：

```python
# 原实现用 for 循环
def fit(self, X, y):
    for i in range(len(X)):
        ...

# 重构成向量化
def fit(self, X, y):
    ...

# 只要 check_estimator 还通过，契约就没破
```

### 15.4 思考题

1. TDD 在 ML 里的适用性和传统软件开发有什么不同？
2. 契约测试绿了但算法测试红，说明什么？反过来呢？
3. 重构时如果契约测试通过但精度下降，测试能发现吗？

---

## 16. 持续集成与测试自动化

### 16.1 CI 流水线

sklearn 的 CI 流水线（简化）：

```yaml
# .github/workflows/test.yml
jobs:
  test:
    steps:
      - run: pytest sklearn/tests/ -k "check_estimator"
      - run: pytest sklearn/tests/test_linear_model.py
      - run: pytest sklearn/tests/test_pipeline.py
```

每次 PR 都跑全套测试，保证不破坏现有契约。

### 16.2 测试矩阵

sklearn 在多个 Python 版本 × 多个依赖版本上测试：

```yaml
strategy:
  matrix:
    python: [3.8, 3.9, 3.10, 3.11]
    numpy: [1.20, 1.21, 1.22]
    scipy: [1.7, 1.8, 1.9]
```

这能发现版本兼容性问题。

### 16.3 覆盖率检查

```bash
pytest --cov=sklearn --cov-report=html
```

sklearn 要求覆盖率 > 90%，新代码覆盖率 > 95%。

### 16.4 思考题

1. 为什么要在多个版本组合上测试？只测最新版不行吗？
2. 覆盖率高不等于测试好，举一个反例。
3. CI 跑一次 30 分钟，怎么加速？

### 16.5 测试的常见误区

#### 16.5.1 测实现而非测契约

```python
# 坏：测实现细节（内部用了 SVD）
def test_logistic_uses_svd():
    assert clf._solver == 'svd'

# 好：测契约（预测正确）
def test_logistic_predicts_correctly():
    assert accuracy > 0.9
```

测实现会让重构困难——换求解器就挂。

#### 16.5.2 测 happy path 而忽略边界

```python
# 只测正常输入
def test_fit_normal():
    clf.fit(X_normal, y_normal)

# 忘了测：空数组、单样本、全相同标签、NaN、Inf、超大值...
```

契约测试的价值之一就是自动覆盖各种边界。

#### 16.5.3 测过拟合

```python
# 坏：在训练集上测高精度
def test_high_accuracy():
    clf.fit(X, y)
    assert clf.score(X, y) > 0.99   # 训练精度，没意义

# 好：用交叉验证
def test_generalization():
    score = cross_val_score(clf, X, y, cv=5).mean()
    assert score > 0.7
```

#### 16.5.4 固定随机种子但写死结果

```python
# 坏：写死数值，换版本就挂
def test_coef():
    assert clf.coef_[0] == 0.123456789

# 好：用近似比较
def test_coef():
    assert clf.coef_[0] == pytest.approx(0.123, rel=1e-3)
```

### 16.6 测试与文档的关系

好的测试是活的文档：

```python
def test_clone_removes_fitted_attributes():
    """clone 后的估计器不应该有 fit 学出的属性。

    这对应文档里"clone 创建干净副本"的约定。
    """
    clf = LogisticRegression().fit(X, y)
    clf_clone = clone(clf)
    assert not hasattr(clf_clone, 'coef_')
```

读测试名字和 docstring 就能理解契约。

### 16.7 思考题

1. "测实现而非测契约"会带来什么长期问题？
2. 怎么写一个"测过拟合"的反例测试？
3. 测试的 docstring 应该写什么？和函数 docstring 有什么不同？

---

## 17. 小结

| 要点             | 内容                                       |
|------------------|--------------------------------------------|
| 通用测试         | 一套测试测所有估计器                       |
| 前提             | 统一 API 契约                              |
| 回报             | 新增估计器自动获得所有通用工具支持         |
| 测试项           | 几十项，覆盖 fit/predict/clone/params       |
| 组织             | pytest 参数化，每项独立函数                |
| 分层             | 契约测试 / 算法测试 / 集成测试             |

**核心洞察**：通用测试套件是统一 API 契约的**回报**——契约越统一，通用工具越强大。`check_estimator` 能测所有估计器，正是因为所有估计器都遵守同一套约定。这把"约定优于配置"的回报具象化：一次投入（写契约 + 写通用测试），持续回报（新增算法自动获得测试 + 通用工具支持）。

### 17.1 一致性测试的局限

`check_estimator` 不是万能的：

- **不测算法正确性**：通过契约测试不代表算法对。一个永远返回 0 的估计器可能通过大部分契约测试。
- **不测性能**：契约测试不关心 fit 要跑多久。
- **不测泛化**：契约测试用合成数据，不反映真实数据表现。
- **不测可解释性**：契约测试不检查 `feature_importances_` 是否合理。

所以契约测试只是第一道防线，还需要算法测试、性能测试、集成测试补充。

### 17.2 一致性测试的未来

sklearn 团队在探索：

- **更严格的契约**：例如要求所有估计器支持 `feature_names_in_`。
- **类型提示**：用 mypy 做静态契约检查，补充运行时测试。
- **属性测试**：用 hypothesis 生成随机输入，发现更多边界 bug。
- **GPU 支持**：如果 sklearn 支持 GPU，契约测试要扩展到 GPU 张量。

### 17.3 思考题

1. 一个"永远返回 0"的估计器能通过多少项契约测试？这说明了什么？
2. 怎么用 hypothesis 给 `check_estimator` 加属性测试？
3. 如果 sklearn 要支持 GPU，契约测试要加哪些项？

---

## 16. 练习

### 16.1 基础练习

1. 运行 `check_estimator(LogisticRegression)`，观察它跑了哪些测试。
2. 写一个故意违反"predict 不改状态"的估计器，验证 `check_estimator` 能抓到。
3. 给 `MeanRegressor` 加 `score` 方法，验证 `test_score_range` 通过。
4. 用 pytest 参数化机制对 5 个 sklearn 估计器跑 `check_estimator`。

### 16.2 进阶练习

5. 实现一个 mini `check_transformer`，专门测 `TransformerMixin` 的契约（`fit_transform` == `fit` + `transform`）。
6. 写一个 `check_pipeline`，测 Pipeline 对任意估计器组合的行为。
7. 给 `MeanRegressor` 写算法正确性测试，对比它和 `DummyRegressor` 的预测结果。
8. 写一个性能基准测试，比较 `MeanRegressor` 和 `LinearRegression` 在不同数据规模下的 fit 时间。

### 16.3 思考题

9. 如果 sklearn 没有 `check_estimator`，新增一个估计器要做多少额外工作？
10. `check_estimator` 能保证算法正确吗？如果不能，它保证的是什么？
11. 设计一个"契约测试套件测试套件"——怎么验证 `check_estimator` 本身是正确的？
12. 如果让你给一个非 ML 库（例如 requests）设计类似的"通用契约测试"，会是什么样？
13. 契约测试和类型系统（mypy / pyright）有什么互补关系？各自能抓什么 bug？
14. `check_estimator` 在 sklearn 0.20 和 0.24 之间多了哪些测试项？为什么加？
15. 如果一个第三方库声称"sklearn 兼容"，怎么验证？`check_estimator` 够吗？

---

## 上一讲 / 下一讲

[← 第五讲：数据约定与校验](05-data-convention.md) ｜  [第七讲：全局配置与演进 →](07-config-and-evolution.md）
