# 第三讲：参数管理机制

> **核心问题**：为什么 `__init__` 只存参数不做任何事？`get_params` / `set_params` 如何用反射自动工作？`clone` 为什么不用 `copy.deepcopy`？

---

## 1. 最反直觉的约定：`__init__` 不做事

sklearn 有一条硬性约定（SLEP009），违反它会导致 `clone`、`GridSearchCV` 等全部失效：

> **`__init__` 只能把参数原样存到 `self` 上，不允许做任何计算、校验、转换。**

```python
# ✅ 正确
class LogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter

# ❌ 错误：做了类型转换
class LogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100):
        self.C = float(C)          # 错！做了转换
        self.max_iter = int(max_iter)  # 错！

# ❌ 错误：初始化了非参数状态
class LogisticRegression(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
        self._cache = {}           # 错！初始化了非参数状态
```

### 为什么这么严格？

三个理由：

**理由 1：`clone` 的需要**

`clone(estimator)` 的实现是：

```python
def clone(estimator):
    params = estimator.get_params()  # 取出 __init__ 参数
    return type(estimator)(**params)  # 用参数重新构造
```

如果 `__init__` 做了 `self.C = float(C)`，而用户传了 `C="1.0"`（字符串），那么：

- 原对象：`self.C = 1.0`（float，因为 `__init__` 转了）
- `get_params()` 返回 `{'C': 1.0}`
- `clone` 重建：`LogisticRegression(C=1.0)` → `self.C = float(1.0) = 1.0`

看起来没问题？但如果 `__init__` 有副作用（如读文件、建缓存），`clone` 就会重复这些副作用，行为不可控。

**理由 2：可序列化**

只存参数的对象状态可由参数完全描述，`pickle` 和 `inspect` 都更可靠。

**理由 3：一致性**

所有估计器遵循同一约定，`get_params` 的反射实现才能统一工作（见下节）。

### 1.1 理由 1 的深入分析：`clone` 失效的具体场景

让我们看一个 `__init__` 做了转换导致 `clone` 出问题的具体例子：

```python
# ❌ __init__ 做了转换
class BadAlgo(BaseEstimator):
    def __init__(self, penalty='l2'):
        if penalty == 'l2':
            self.penalty = 'l2'           # 转换：存的是 'l2'
            self._penalty_fn = lambda x: x**2  # 还建了函数！
        elif penalty == 'l1':
            self.penalty = 'l1'
            self._penalty_fn = lambda x: abs(x)

algo = BadAlgo(penalty='l2')
# algo.penalty = 'l2'
# algo._penalty_fn = <function>

new_algo = clone(algo)
# clone 内部：BadAlgo(penalty='l2')
# __init__ 又建了一次 _penalty_fn
# new_algo._penalty_fn 是新建的函数，不是原来的
```

问题：

1. `_penalty_fn` 是 `__init__` 建的，`get_params` 取不到它
2. `clone` 重建时，`__init__` 又建了一次，新函数和原函数不是同一个对象
3. 如果 `_penalty_fn` 持有状态（如计数器），状态丢失

这就是为什么 `__init__` 不能做"创建对象"的事——`clone` 会重复创建，行为不可控。

### 1.2 理由 2 的深入分析：`pickle` 失效

```python
# ❌ __init__ 建了不可 pickle 的对象
class BadAlgo(BaseEstimator):
    def __init__(self):
        self._db_connection = connect_to_db()  # 数据库连接

algo = BadAlgo()
pickle.dumps(algo)  # 可能失败：数据库连接不可 pickle
```

如果 `__init__` 只存参数，`pickle` 只存参数，重建时用参数重新 `__init__`，行为可控。

### 1.3 理由 3 的深入分析：`get_params` 反射失效

`get_params` 用反射从 `__init__` 签名提取参数名，然后从 `self` 取值。如果 `__init__` 改了参数名：

```python
# ❌ __init__ 改了参数名
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        self._C = C  # 存成 _C，不是 C

algo = BadAlgo(C=1.0)
algo.get_params()
# 反射从 __init__ 签名取 'C'
# 然后 getattr(algo, 'C') → AttributeError！
# 因为实际存的是 _C
```

`get_params` 假设 `__init__` 的参数名和 `self` 上的属性名一致。如果 `__init__` 改名，反射就失效。

### 校验放哪？

如果 `__init__` 不能校验，参数校验放哪？放 `fit`：

```python
class LogisticRegression(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C  # 原样存储

    def fit(self, X, y):
        if self.C <= 0:
            raise ValueError("C 必须为正数")
        # ... 真正的计算
```

这是 sklearn 的约定：**校验在 `fit` 里做，不在 `__init__` 里做**。

### 1.4 校验放 `fit` 的好处

1. **`clone` 安全**：`clone` 不调 `fit`，不会触发校验
2. **错误延迟到使用时**：用户构造时不报错，`fit` 时才报错，允许"先构造后配置"
3. **统一位置**：所有校验都在 `fit`，易于查找

```python
# 允许先构造后配置
clf = LogisticRegression(C=-1)  # 不报错
clf.C = 1.0                     # 改成正数
clf.fit(X, y)                   # 现在校验，通过
```

如果 `__init__` 校验，上面就不行——构造时就报错了。

### 1.5 什么时候可以例外？

极少数情况下，`__init__` 可以做轻微计算，但必须保证：

1. 计算是**幂等**的：多次调用结果一样
2. 计算不创建**外部资源**：不连数据库、不读文件
3. 计算结果**可 pickle**：能被序列化

```python
# 勉强可以：把列表转成 tuple（幂等、无外部资源、可 pickle）
class MyAlgo(BaseEstimator):
    def __init__(self, layers=(10, 20)):
        self.layers = tuple(layers)  # 转成 tuple
```

但即使这种情况，sklearn 也建议在 `fit` 里做。保守一点不会错。

---

## 2. `get_params`：反射的魔法

`get_params` 不需要你手写——它通过反射自动从 `__init__` 签名提取参数名：

```python
import inspect

def _get_param_names(cls):
    """从 __init__ 签名提取参数名"""
    signature = inspect.signature(cls.__init__)
    return [
        name for name, param in signature.parameters.items()
        if name != "self"
        and param.kind != param.VAR_POSITIONAL   # 排除 *args
        and param.kind != param.VAR_KEYWORD       # 排除 **kwargs
    ]
```

然后 `get_params` 用这些名字从 `self` 取值：

```python
def get_params(self, deep=True):
    out = {}
    for key in self._get_param_names():
        value = getattr(self, key)
        out[key] = value
        if deep and hasattr(value, "get_params"):
            # 嵌套估计器：展平为 key__subkey
            for sub_key, sub_value in value.get_params().items():
                out[f"{key}__{sub_key}"] = sub_value
    return out
```

### 为什么用反射而不是手写？

1. **DRY**：上百个估计器，每个都手写 `get_params` 会产生大量重复代码
2. **一致性**：反射保证 `get_params` 与 `__init__` 永远同步，不会漏参数
3. **约定优于配置**：只要你遵守"`__init__` 只存同名属性"的约定，`get_params` 自动工作

### 2.1 反射的具体工作过程

让我们跟踪 `get_params` 的执行：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100, penalty='l2'):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty

clf = LogisticRegression(C=2.0, max_iter=200)

# 1. _get_param_names 从 __init__ 签名提取
#    signature: (self, C=1.0, max_iter=100, penalty='l2')
#    参数名: ['C', 'max_iter', 'penalty']

# 2. get_params 从 self 取值
#    getattr(clf, 'C') = 2.0
#    getattr(clf, 'max_iter') = 200
#    getattr(clf, 'penalty') = 'l2'

# 3. 返回
clf.get_params()
# {'C': 2.0, 'max_iter': 200, 'penalty': 'l2'}
```

整个过程完全自动，不用手写一行 `get_params` 代码。

### 2.2 反射的 `inspect` 模块

`get_params` 用的 `inspect` 模块是 Python 的反射工具：

```python
import inspect

class Foo:
    def __init__(self, a, b=2, *args, c=3, **kwargs):
        pass

sig = inspect.signature(Foo.__init__)
for name, param in sig.parameters.items():
    print(f"{name}: kind={param.kind}, default={param.default}")

# a: kind=POSITIONAL_OR_KEYWORD, default=<class 'inspect._empty'>
# b: kind=POSITIONAL_OR_KEYWORD, default=2
# args: kind=VAR_POSITIONAL, default=<class 'inspect._empty'>
# c: kind=KEYWORD_ONLY, default=3
# kwargs: kind=VAR_KEYWORD, default=<class 'inspect._empty'>
```

`_get_param_names` 排除 `*args` 和 `**kwargs`，只取命名参数：

```python
def _get_param_names(cls):
    signature = inspect.signature(cls.__init__)
    return [
        name for name, param in signature.parameters.items()
        if name != "self"
        and param.kind != param.VAR_POSITIONAL   # 排除 *args
        and param.kind != param.VAR_KEYWORD       # 排除 **kwargs
    ]
```

为什么排除 `*args` 和 `**kwargs`？因为它们不是"命名参数"，无法用 `getattr(self, name)` 取值。

### 2.3 反射的局限性

反射依赖 `__init__` 签名，所以：

1. **`__init__` 必须有明确签名**：不能用 `**kwargs` 收参数
2. **参数名和属性名必须一致**：`__init__(self, C=1.0)` 必须 `self.C = C`
3. **不能用 `setattr` 动态设置**：反射看不到动态属性

```python
# ❌ 用 **kwargs，反射失效
class BadAlgo(BaseEstimator):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

algo = BadAlgo(C=1.0)
algo.get_params()  # {}，反射看不到参数！
```

这是为什么 sklearn 估计器的 `__init__` 总是明确列出所有参数——反射需要明确签名。

### `deep=True` 的嵌套展平

当参数本身是估计器时，`deep=True` 会递归展平：

```python
pipe = Pipeline([('clf', LogisticRegression(C=2.0))])
pipe.get_params(deep=True)
# {
#   'steps': [('clf', LogisticRegression(C=2.0))],
#   'clf': LogisticRegression(C=2.0),
#   'clf__C': 2.0,              # 嵌套展平！
#   'clf__max_iter': 100,
#   ...
# }
```

`clf__C` 这种命名是 `GridSearchCV` 能用 `{'clf__C': [1, 2, 3]}` 搜索嵌套参数的基础。

### 2.4 `deep=True` vs `deep=False`

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=2.0))])

# deep=True：递归展平
pipe.get_params(deep=True)
# {
#   'steps': [...],
#   'scaler': StandardScaler(),
#   'scaler__with_mean': True,
#   'scaler__with_std': True,
#   'clf': LogisticRegression(C=2.0),
#   'clf__C': 2.0,
#   'clf__max_iter': 100,
#   'clf__penalty': 'l2',
#   'memory': None,
#   'verbose': False,
# }

# deep=False：只取顶层
pipe.get_params(deep=False)
# {
#   'steps': [...],
#   'memory': None,
#   'verbose': False,
# }
```

`deep=True` 把嵌套参数展平成 `step__param` 形式，`deep=False` 只取顶层。`GridSearchCV` 用 `deep=True` 拿到所有可搜索参数。

### 2.5 嵌套展平的递归实现

```python
def get_params(self, deep=True):
    out = {}
    for key in self._get_param_names():
        value = getattr(self, key)
        out[key] = value
        if deep and hasattr(value, "get_params"):
            # value 是估计器，递归展平
            for sub_key, sub_value in value.get_params().items():
                out[f"{key}__{sub_key}"] = sub_value
    return out
```

递归过程：

1. 取 `pipe` 的参数：`steps`, `memory`, `verbose`
2. `steps` 是列表，没有 `get_params`，不递归
3. 但 `pipe` 有特殊处理：把 `steps` 里的估计器也展平
4. 对每个步骤估计器（`scaler`, `clf`），递归调 `get_params`
5. 用 `step__param` 命名展平

实际 sklearn 的 `Pipeline.get_params` 比上面更复杂，因为要处理 `steps` 列表。但思路一致——递归展平嵌套估计器的参数。

---

## 3. `set_params`：嵌套参数的设置

`set_params` 支持嵌套命名：

```python
pipe.set_params(clf__C=3.0)
# 等价于
pipe.named_steps['clf'].C = 3.0
```

实现：

```python
def set_params(self, **params):
    nested_params = {}
    for key, value in params.items():
        if "__" in key:
            step, sub = key.split("__", 1)
            nested_params.setdefault(step, {})[sub] = value
        else:
            setattr(self, key, value)
    for step, sub_params in nested_params.items():
        getattr(self, step).set_params(**sub_params)
    return self
```

返回 `self` 是为了链式调用：`est.set_params(a=1).set_params(b=2)`。

### 3.1 `set_params` 的执行过程

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0))])

pipe.set_params(clf__C=10.0, scaler__with_mean=False)

# 内部执行：
# 1. 分离嵌套参数
#    nested_params = {'clf': {'C': 10.0}, 'scaler': {'with_mean': False}}
# 2. 对每个嵌套，递归 set_params
#    pipe.named_steps['clf'].set_params(C=10.0)  # clf.C = 10.0
#    pipe.named_steps['scaler'].set_params(with_mean=False)  # scaler.with_mean = False
```

### 3.2 `set_params` vs 直接赋值

```python
# set_params：支持嵌套
pipe.set_params(clf__C=10.0)

# 直接赋值：不支持嵌套
pipe.C  # AttributeError，pipe 没有 C
pipe.named_steps['clf'].C = 10.0  # 要手动导航
```

`set_params` 的价值在于支持嵌套命名，让 `GridSearchCV` 能用 `clf__C` 设置 Pipeline 内部参数。

### 3.3 `set_params` 的链式调用

```python
# 链式调用
clf = LogisticRegression().set_params(C=1.0).set_params(max_iter=200)
# 等价于
clf = LogisticRegression()
clf.C = 1.0
clf.max_iter = 200
```

`set_params` 返回 `self`，支持链式。但在生产代码中，链式可读性差，不推荐。

### 3.4 `set_params` 的错误处理

```python
# 设置不存在的参数
clf = LogisticRegression()
clf.set_params(nonexistent=1.0)
# ValueError: Invalid parameter 'nonexistent' for estimator LogisticRegression

# 设置嵌套参数时，步骤不存在
pipe.set_params(nonexistent__C=1.0)
# ValueError: Invalid parameter 'nonexistent' for estimator Pipeline
```

sklearn 会检查参数名是否合法，给出清晰错误信息。

---

## 4. `clone`：为什么不用 `deepcopy`？

```python
from sklearn.base import clone
from copy import deepcopy

clf = LogisticRegression(C=1.0).fit(X, y)  # 已训练

# clone：得到未训练的同参数副本
new_clf = clone(clf)
# new_clf.C == 1.0，但没有 coef_（未训练）

# deepcopy：复制全部状态（包括训练结果）
copy_clf = deepcopy(clf)
# copy_clf.C == 1.0，且 copy_clf.coef_ 也存在（已训练）
```

`clone` 的语义是"给我一个干净的、同参数的新对象"，`deepcopy` 做不到这点。

实现：

```python
def clone(estimator):
    params = estimator.get_params(deep=False)
    # 只取 __init__ 参数，不取 fit 后的属性
    return type(estimator)(**params)
```

### `clone` 的用途

`GridSearchCV` 在搜索时反复 `clone`：

```python
class GridSearchCV:
    def fit(self, X, y):
        for param_combo in self.param_grid:
            new_clf = clone(self.estimator)  # 干净的副本
            new_clf.set_params(**param_combo)
            new_clf.fit(X_train, y_train)
            score = new_clf.score(X_val, y_val)
            ...
```

如果用 `deepcopy`，每次复制的都是**已训练**的对象，状态污染会导致结果错误。

### 4.1 `clone` vs `deepcopy` 的详细对比

| 特性 | `clone` | `deepcopy` |
|------|---------|------------|
| 复制超参数 | ✅ | ✅ |
| 复制学习参数（`coef_` 等） | ❌ | ✅ |
| 结果是已训练 | ❌（未训练） | ✅（同原状态） |
| 用途 | 重新训练 | 备份当前状态 |
| 实现 | `get_params` + 重建 | 递归复制所有属性 |

```python
clf = LogisticRegression(C=1.0).fit(X, y)

# clone：未训练的同参数副本
new_clf = clone(clf)
print(new_clf.C)  # 1.0
print(hasattr(new_clf, 'coef_'))  # False

# deepcopy：完全相同的副本
copy_clf = deepcopy(clf)
print(copy_clf.C)  # 1.0
print(hasattr(copy_clf, 'coef_'))  # True
print((copy_clf.coef_ == clf.coef_).all())  # True
```

### 4.2 `clone` 的实现细节

```python
def clone(estimator, safe=True):
    # 1. 取 __init__ 参数
    params = estimator.get_params(deep=False)

    # 2. 递归 clone 嵌套估计器
    for name, value in params.items():
        if hasattr(value, 'get_params'):
            params[name] = clone(value)  # 递归

    # 3. 用参数重新构造
    new_object = type(estimator)(**params)

    return new_object
```

注意第 2 步：嵌套估计器也要递归 `clone`，否则内外会共享同一个嵌套对象。

### 4.3 `clone` 的常见用途

```python
# 1. GridSearchCV：每次搜索 clone
for params in param_grid:
    clf = clone(base_clf)
    clf.set_params(**params)
    clf.fit(X_train, y_train)

# 2. 交叉验证：每折 clone
for train_idx, val_idx in splits:
    clf = clone(base_clf)
    clf.fit(X[train_idx], y[train_idx])

# 3. 集成方法：每棵树 clone
for _ in range(n_estimators):
    tree = clone(base_tree)
    tree.fit(X_sample, y_sample)
```

共同点：每次重新训练前都 `clone`，保证从干净状态开始。

### 4.4 不用 `clone` 会出什么问题

```python
# ❌ 不用 clone，复用同一个对象
base_clf = LogisticRegression()

for params in param_grid:
    base_clf.set_params(**params)
    base_clf.fit(X_train, y_train)
    score = base_clf.score(X_val, y_val)
    # 问题：base_clf 已经 fit 了，下次循环 set_params 后状态混乱
    # 而且 base_clf.coef_ 是上次 fit 的，可能影响下次
```

不用 `clone`，每次循环复用同一个对象，状态会累积污染。`clone` 保证每次从干净状态开始。

---

## 5. `__repr__`：自动生成

`BaseEstimator.__repr__` 从 `get_params` 自动生成：

```python
def __repr__(self):
    params = self.get_params(deep=False)
    sorted_params = sorted(params.items())
    params_str = ", ".join(f"{k}={v!r}" for k, v in sorted_params)
    return f"{type(self).__name__}({params_str})"
```

输出：`LogisticRegression(C=1.0, max_iter=100, penalty='l2')`

为什么自动生成？保证 `repr` 与参数永远一致，不会漏写。

### 5.1 `__repr__` 的具体输出

```python
clf = LogisticRegression(C=1.0, max_iter=100, penalty='l2')
print(clf)
# LogisticRegression(C=1.0, max_iter=100, penalty='l2')

scaler = StandardScaler(with_mean=True, with_std=True)
print(scaler)
# StandardScaler()

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
print(pipe)
# Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
```

注意 `StandardScaler()` 没显示参数——因为都是默认值。sklearn 的 `__repr__` 会省略默认值，让输出更简洁。

### 5.2 省略默认值的实现

```python
def __repr__(self):
    params = self.get_params(deep=False)
    # 取默认值
    defaults = self._get_param_defaults()  # 假设有这个方法

    # 只显示非默认值
    non_default = {
        k: v for k, v in params.items()
        if k not in defaults or defaults[k] != v
    }

    sorted_params = sorted(non_default.items())
    params_str = ", ".join(f"{k}={v!r}" for k, v in sorted_params)
    return f"{type(self).__name__}({params_str})"
```

省略默认值让 `repr` 更简洁——`StandardScaler()` 比 `StandardScaler(copy=True, with_mean=True, with_std=True)` 易读。

### 5.3 `__repr__` 的用途

1. **调试**：`print(clf)` 看参数
2. **日志**：记录模型配置
3. **复现**：`repr` 包含所有非默认参数，可复制粘贴重建

```python
# 复现：从 repr 重建
clf = LogisticRegression(C=10.0, max_iter=200)
repr_str = repr(clf)  # "LogisticRegression(C=10.0, max_iter=200)"
# 可以 eval(repr_str) 重建（不推荐 eval，但理论上可行）
```

---

## 6. 参数管理的完整闭环

sklearn 的参数管理是一个**自洽的闭环**：

```
__init__ 只存参数
    ↓
get_params 用反射取参数
    ↓
clone 用 get_params 重建
    ↓
GridSearchCV 用 clone 反复重建
    ↓
set_params 设置嵌套参数
    ↓
__repr__ 从 get_params 生成
```

每个环节都依赖前一个。违反任一环节，整个链条断裂。

### 6.1 违反约定的后果

```python
# ❌ __init__ 做了转换
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = float(C)  # 转换

algo = BadAlgo(C="1.0")  # 传字符串

# 后果 1：get_params 返回的是转换后的值
algo.get_params()  # {'C': 1.0}，不是 '1.0'

# 后果 2：clone 重建时，__init__ 又转一次
new_algo = clone(algo)
# BadAlgo(C=1.0) → __init__ → float(1.0) = 1.0
# 这次没问题，但如果 __init__ 有副作用就出问题

# 后果 3：repr 不准确
print(algo)  # BadAlgo(C=1.0)，但用户传的是 '1.0'
```

### 6.2 闭环的优雅

这个闭环的优雅在于：**约定支撑机制，机制支撑上层功能**。

- `__init__` 约定 → 支撑 `get_params` 反射
- `get_params` 反射 → 支撑 `clone` 重建
- `clone` 重建 → 支撑 `GridSearchCV` 搜索
- `GridSearchCV` 搜索 → 支撑模型选择

违反 `__init__` 约定，`get_params` 失效，`clone` 失效，`GridSearchCV` 失效，模型选择失效。一环断，全链断。

---

## 7. 与其他框架的参数管理对比

### 7.1 sklearn vs PyTorch

```python
# sklearn：参数在 __init__
clf = LogisticRegression(C=1.0, max_iter=100)
clf.get_params()  # {'C': 1.0, 'max_iter': 100}

# PyTorch：参数在 __init__，但没 get_params
model = nn.Linear(10, 5)
model.state_dict()  # {'weight': ..., 'bias': ...}，但这是学习参数
# 超参数没有统一管理
```

PyTorch 的 `state_dict` 管理学习参数（weight, bias），但超参数（in_features, out_features）没有统一管理。sklearn 用 `get_params` 管理超参数，用属性（`coef_`）管理学习参数，两者分离。

### 7.2 sklearn vs Keras

```python
# Keras：参数在 __init__，有 get_config
model = keras.Sequential([keras.layers.Dense(10)])
config = model.get_config()  # 序列化配置
new_model = keras.Sequential.from_config(config)  # 重建
```

Keras 的 `get_config` / `from_config` 类似 sklearn 的 `get_params` / `clone`，但 Keras 用 dict，sklearn 用 kwargs。

### 7.3 sklearn vs HuggingFace

```python
# HuggingFace：参数在 config 对象
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
```

HuggingFace 把配置和模型分离——`config` 对象管超参数，`model` 管学习参数。sklearn 把两者放一个对象，用下划线区分。

---

## 8. 常见问题和陷阱

### 8.1 陷阱 1：`__init__` 做了校验

```python
# ❌ __init__ 校验
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        if C <= 0:
            raise ValueError("C 必须为正数")
        self.C = C

# 后果：clone 时，如果 C 被改成负数再 clone，报错
algo = BadAlgo(C=1.0)
algo.C = -1  # 手动改
clone(algo)  # BadAlgo(C=-1) → __init__ 校验 → 报错
```

### 8.2 陷阱 2：`__init__` 改了参数名

```python
# ❌ 参数名和属性名不一致
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        self._C = C  # 存成 _C

algo = BadAlgo(C=1.0)
algo.get_params()  # AttributeError: no 'C'
```

### 8.3 陷阱 3：`__init__` 初始化了非参数状态

```python
# ❌ 初始化了缓存
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
        self._cache = {}  # 非参数状态

# 后果：clone 后，_cache 丢失（因为 get_params 不取它）
algo = BadAlgo(C=1.0)
algo._cache['key'] = 'value'
new_algo = clone(algo)
# new_algo._cache 不存在！
```

### 8.4 陷阱 4：用了 `**kwargs`

```python
# ❌ 用 **kwargs
class BadAlgo(BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

algo = BadAlgo(C=1.0)
algo.get_params()  # {'kwargs': {'C': 1.0}}，不是 {'C': 1.0}
```

### 8.5 陷阱 5：嵌套估计器没递归 clone

```python
# ❌ 嵌套估计器共享引用
class BadAlgo(BaseEstimator):
    def __init__(self, base_clf):
        self.base_clf = base_clf

clf1 = BadAlgo(LogisticRegression())
clf2 = clone(clf1)
# clf2.base_clf 是 clf1.base_clf 的同一个对象！
# 改 clf2.base_clf 会影响 clf1.base_clf
```

正确的 `clone` 会递归 clone 嵌套估计器，避免共享引用。

---

## 9. 实际使用模式

### 9.1 模式 1：获取所有参数

```python
clf = LogisticRegression(C=1.0, max_iter=100)
params = clf.get_params()
print(params)
# {'C': 1.0, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs', ...}
```

### 9.2 模式 2：设置参数

```python
clf = LogisticRegression()
clf.set_params(C=10.0, max_iter=200)
```

### 9.3 模式 3：clone 后修改

```python
base_clf = LogisticRegression(C=1.0)
for C in [0.1, 1, 10]:
    clf = clone(base_clf)
    clf.set_params(C=C)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
```

### 9.4 模式 4：嵌套参数搜索

```python
pipe = Pipeline([('clf', LogisticRegression())])
grid = GridSearchCV(pipe, param_grid={'clf__C': [0.1, 1, 10]})
grid.fit(X, y)
```

### 9.5 模式 5：序列化

```python
import pickle

clf = LogisticRegression(C=1.0).fit(X, y)

# 序列化
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# 反序列化
with open('model.pkl', 'rb') as f:
    new_clf = pickle.load(f)

# 参数保留
print(new_clf.C)  # 1.0
# 学习参数也保留
print(new_clf.coef_)  # 和原 clf.coef_ 相同
```

---

## 10. 思考题和练习

### 10.1 思考题

1. 为什么 `__init__` 不能校验，但 `fit` 可以？校验放 `__init__` 有什么好处？
2. `clone` 用 `get_params` 重建，为什么不用 `copy.copy`（浅拷贝）？
3. `get_params(deep=True)` 展平嵌套参数，有什么场景下不希望展平？
4. 如果 `__init__` 用 `**kwargs`，反射会失效。有什么替代方案？
5. `set_params` 返回 `self` 支持链式，但 `__init__` 不返回 `self`。为什么？

### 10.2 练习

1. 实现一个违反 `__init__` 约定的估计器，观察 `clone` 和 `GridSearchCV` 的错误。
2. 手写一个不依赖反射的 `get_params`，比较和反射版本的代码量。
3. 实现一个支持 `**kwargs` 的估计器，思考如何让 `get_params` 工作。

---

## 11. 深入：反射机制的原理

### 11.1 `inspect` 模块详解

`get_params` 依赖 `inspect` 模块提取 `__init__` 签名。让我们深入看看：

```python
import inspect

class LogisticRegression:
    def __init__(self, C=1.0, max_iter=100, penalty='l2', *, solver='lbfgs'):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.solver = solver

# 获取 __init__ 的签名
sig = inspect.signature(LogisticRegression.__init__)
print(sig)
# (self, C=1.0, max_iter=100, penalty='l2', *, solver='lbfgs')

# 遍历参数
for name, param in sig.parameters.items():
    print(f"name={name}, kind={param.kind}, default={param.default}")

# name=self, kind=POSITIONAL_OR_KEYWORD, default=<empty>
# name=C, kind=POSITIONAL_OR_KEYWORD, default=1.0
# name=max_iter, kind=POSITIONAL_OR_KEYWORD, default=100
# name=penalty, kind=POSITIONAL_OR_KEYWORD, default='l2'
# name=solver, kind=KEYWORD_ONLY, default='lbfgs'
```

`inspect.signature` 返回 `Signature` 对象，包含所有参数的 `Parameter` 对象。每个 `Parameter` 有 `name`、`kind`、`default` 等属性。

### 11.2 参数的 kind 类型

`Parameter.kind` 有以下几种：

| kind | 例子 | sklearn 处理 |
|------|------|-------------|
| `POSITIONAL_ONLY` | `(x, /)` | 提取（Python 3.8+） |
| `POSITIONAL_OR_KEYWORD` | `(x)` | 提取 |
| `VAR_POSITIONAL` | `(*args)` | 排除 |
| `KEYWORD_ONLY` | `(*, x)` | 提取 |
| `VAR_KEYWORD` | `(**kwargs)` | 排除 |

sklearn 的 `_get_param_names` 排除 `VAR_POSITIONAL` 和 `VAR_KEYWORD`，只取命名参数：

```python
def _get_param_names(cls):
    signature = inspect.signature(cls.__init__)
    return [
        name for name, param in signature.parameters.items()
        if name != "self"
        and param.kind != param.VAR_POSITIONAL
        and param.kind != param.VAR_KEYWORD
    ]
```

### 11.3 反射的性能考量

反射有性能开销——每次 `get_params` 都要 `inspect.signature`。sklearn 做了缓存优化：

```python
def _get_param_names(cls):
    # 检查缓存
    if hasattr(cls, '_param_names_cache'):
        return cls._param_names_cache

    signature = inspect.signature(cls.__init__)
    param_names = [...]

    # 缓存到类属性
    cls._param_names_cache = param_names
    return param_names
```

缓存让 `inspect.signature` 只调一次，后续 `get_params` 直接用缓存的参数名列表。

### 11.4 反射的边界情况

```python
# 1. __init__ 用 functools.wraps 装饰
class MyAlgo(BaseEstimator):
    @functools.wraps(some_func)
    def __init__(self, C=1.0):
        self.C = C
# 反射可能取到 some_func 的签名，而非 __init__ 的

# 2. __init__ 用 *args 和 **kwargs
class MyAlgo(BaseEstimator):
    def __init__(self, *args, **kwargs):
        pass
# _get_param_names 返回 []，get_params 返回 {}

# 3. 动态创建的类
MyAlgo = type('MyAlgo', (BaseEstimator,), {'__init__': lambda self, C=1.0: setattr(self, 'C', C)})
# 反射能工作，但可读性差
```

这些边界情况是反射的局限——它依赖 `__init__` 有明确、静态的签名。

---

## 12. `get_params` / `set_params` 的对称性

`get_params` 和 `set_params` 应该是对称的：

```python
# 对称性：get 后 set 回去，状态不变
params = clf.get_params()
clf.set_params(**params)  # 应该等价于什么都没做
```

但有个微妙之处：`get_params(deep=True)` 返回嵌套展平的参数，`set_params` 也能接受嵌套展平的参数：

```python
pipe = Pipeline([('clf', LogisticRegression(C=1.0))])

# get_params(deep=True) 返回展平的
params = pipe.get_params(deep=True)
# {'steps': ..., 'clf': ..., 'clf__C': 1.0, 'clf__max_iter': 100, ...}

# set_params 接受展平的
pipe.set_params(**{k: v for k, v in params.items() if '__' in k or k in ['steps', 'memory', 'verbose']})
# 状态不变
```

这种对称性让 `clone` 能工作：`clone` 取 `get_params(deep=False)`，用 `set_params` 或构造函数重建。

### 12.1 不对称的情况

有些参数 `get_params` 能取到，但 `set_params` 不能直接设：

```python
# Pipeline 的 steps 参数
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
params = pipe.get_params()
# params['steps'] = [('scaler', StandardScaler()), ('clf', LogisticRegression())]

pipe.set_params(steps=[('new', StandardScaler())])  # 能设，但要整个列表
# 不能 pipe.set_params(steps__scaler=StandardScaler())  # 错
```

`steps` 是整体替换，不能部分修改。这是 `get_params` / `set_params` 的不对称之处。

---

## 13. `clone` 的深入分析

### 13.1 `clone` 的完整实现

```python
def clone(estimator, safe=True):
    """构造一个同参数的新估计器，未训练。"""
    estimator_type = type(estimator)

    # 处理非估计器（如 None、字符串）
    if estimator_type in (None, str, int, float, bool):
        return estimator

    # 取 __init__ 参数
    params = estimator.get_params(deep=False)

    # 递归 clone 嵌套估计器
    for name, value in params.items():
        if hasattr(value, 'get_params'):
            params[name] = clone(value)  # 递归
        elif isinstance(value, list):
            params[name] = [clone(v) if hasattr(v, 'get_params') else v for v in value]
        elif isinstance(value, dict):
            params[name] = {k: clone(v) if hasattr(v, 'get_params') else v for k, v in value.items()}

    # 用参数重建
    new_object = estimator_type(**params)

    if safe:
        # 校验：新对象的 get_params 应该和原对象一致
        new_params = new_object.get_params(deep=False)
        for name, value in params.items():
            if new_params[name] != value:
                raise RuntimeError(
                    f"Clone of {estimator_type.__name__} produced different params."
                )

    return new_object
```

注意几个细节：

1. **递归 clone 嵌套估计器**：避免共享引用
2. **处理 list 和 dict**：里面的估计器也要 clone
3. **safe 校验**：确保 clone 后参数一致

### 13.2 `clone` 递归的例子

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression()),
])

new_pipe = clone(pipe)

# clone 递归过程：
# 1. pipe.get_params(deep=False) → {'steps': [...], 'memory': None, 'verbose': False}
# 2. steps 是列表，递归 clone 每个元素
#    - ('scaler', StandardScaler()) → ('scaler', clone(StandardScaler()))
#    - ('clf', LogisticRegression()) → ('clf', clone(LogisticRegression()))
# 3. 用新 steps 重建 Pipeline
# 4. new_pipe 的 scaler 和 clf 是新对象，不和原 pipe 共享
```

递归 clone 保证整个嵌套结构都是新的，没有共享引用。

### 13.3 `clone` vs `deepcopy` 的边界

```python
# clone：只复制超参数，不复制学习参数
clf = LogisticRegression(C=1.0).fit(X, y)
new_clf = clone(clf)
# new_clf 没有 coef_

# deepcopy：复制所有
copy_clf = deepcopy(clf)
# copy_clf 有 coef_，和 clf.coef_ 相同

# 什么时候用 clone？什么时候用 deepcopy？
# clone：要重新训练（GridSearchCV、cross_val_score）
# deepcopy：要备份当前状态（保存已训练模型）
```

选择依据：是否需要保留学习参数。

---

## 14. 参数管理的实战技巧

### 14.1 批量修改参数

```python
clf = LogisticRegression()

# 批量设置
params = {'C': 10.0, 'max_iter': 200, 'penalty': 'l1'}
clf.set_params(**params)
```

### 14.2 参数对比

```python
clf1 = LogisticRegression(C=1.0)
clf2 = LogisticRegression(C=10.0)

# 比较参数差异
params1 = clf1.get_params()
params2 = clf2.get_params()
diff = {k: (params1[k], params2[k]) for k in params1 if params1[k] != params2[k]}
print(diff)  # {'C': (1.0, 10.0)}
```

### 14.3 从字典重建

```python
# 保存参数
params = clf.get_params()
class_name = type(clf).__name__

# 重建
import sklearn.linear_model
ClfClass = getattr(sklearn.linear_model, class_name)
new_clf = ClfClass(**params)
```

### 14.4 嵌套参数的批量搜索

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])

# 搜索多个嵌套参数
param_grid = {
    'scaler__with_mean': [True, False],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto'],
}

grid = GridSearchCV(pipe, param_grid)
grid.fit(X, y)
```

---

## 15. 参数管理的历史演进

### 15.1 早期：手写 `get_params`

早期 sklearn 的每个估计器都手写 `get_params`：

```python
class LogisticRegression:
    def get_params(self, deep=True):
        return {'C': self.C, 'max_iter': self.max_iter, 'penalty': self.penalty}
```

问题：容易漏参数，`__init__` 加了参数忘更新 `get_params`。

### 15.2 中期：反射统一

后来 sklearn 用反射统一 `get_params`，手写版本被废弃。但要求 `__init__` 只存同名属性——SLEP009 正式确立这一约定。

### 15.3 后期：`n_features_in_` 和 `_more_tags`

较新的版本引入了 `n_features_in_`（记录特征数）和 `_more_tags`（声明估计器标签），进一步完善参数管理。

### 15.4 SLEP010：`fit` 返回 `self`

SLEP010 正式规定 `fit` 必须返回 `self`。这之前，有些算法的 `fit` 返回 `None`，导致链式调用失效。SLEP010 统一了行为，但破坏了部分第三方代码。

### 15.5 SLEP011：下划线结尾约定

SLEP011 正式规定 `fit` 学出的属性必须以下划线结尾。这之前，有些算法用 `coef`（无下划线），导致 `clone` 无法区分。SLEP011 统一了约定。

### 15.6 从 `copy` 到不修改输入

早期转换器有 `copy=True` 参数。后来 sklearn 决定永远不修改输入，移除了 `copy` 参数。这是简化契约的演进。

### 15.7 `_more_tags` 机制

较新的 sklearn 引入 `_more_tags`，让估计器声明"能力"和"需求"：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def _more_tags(self):
        return {
            'requires_y': True,        # 需要 y
            'requires_positive_X': False,  # 不需要正 X
            'poor_score': True,        # 评分可能差
            'no_validation': False,    # 需要校验
            'multioutput': False,      # 不支持多输出
            'preserves_dtype': [np.float64],  # 保留 dtype
        }
```

`check_estimator` 用这些标签决定跑哪些测试。这是参数管理的延伸——不仅管超参数，还管"能力声明"。

---

## 16. 参数管理的边界情况

### 16.1 参数是函数

```python
# 参数是自定义函数
class MyAlgo(BaseEstimator):
    def __init__(self, kernel_fn=None):
        self.kernel_fn = kernel_fn

    def fit(self, X, y):
        if self.kernel_fn is not None:
            K = self.kernel_fn(X, X)  # 用自定义核
        ...

# 用 lambda
algo = MyAlgo(kernel_fn=lambda X1, X2: X1 @ X2.T)

# clone 时，lambda 会被保留
new_algo = clone(algo)
# new_algo.kernel_fn 是同一个 lambda
```

函数参数能被 `clone` 保留，因为 `get_params` 取到函数对象，重建时传入。

### 16.2 参数是 numpy 数组

```python
# 参数是数组
class MyAlgo(BaseEstimator):
    def __init__(self, init_weights=None):
        self.init_weights = init_weights

algo = MyAlgo(init_weights=np.array([1, 2, 3]))
new_algo = clone(algo)
# new_algo.init_weights 是同一个数组（共享引用）
```

注意：`clone` 不复制数组，新旧对象共享同一个数组引用。如果修改数组，两者都受影响。

### 16.3 参数是可变对象

```python
# 参数是列表
class MyAlgo(BaseEstimator):
    def __init__(self, layers=[10, 20]):
        self.layers = layers  # 危险！可变默认参数

algo1 = MyAlgo()
algo2 = MyAlgo()
# algo1.layers 和 algo2.layers 是同一个列表！
algo1.layers.append(30)
print(algo2.layers)  # [10, 20, 30]，被影响了！
```

这是 Python 的"可变默认参数"陷阱。sklearn 的约定：用 `None` 作默认值，在 `fit` 里初始化：

```python
class MyAlgo(BaseEstimator):
    def __init__(self, layers=None):
        self.layers = layers  # None 作默认

    def fit(self, X, y):
        if self.layers is None:
            self.layers = [10, 20]  # 在 fit 里初始化
        ...
```

---

## 17. 小结

| 机制 | 作用 | 关键约定 |
|------|------|---------|
| `__init__` 只存参数 | 保证 `clone` 可行 | 不做计算、不校验、不初始化状态 |
| `get_params` 反射 | 自动获取参数 | 依赖 `__init__` 签名 |
| `set_params` 嵌套 | 设置嵌套参数 | `step__param` 命名 |
| `clone` | 干净克隆 | 基于 `get_params` 重建，非 `deepcopy` |
| `__repr__` 自动 | 一致的字符串表示 | 从 `get_params` 生成 |

**核心洞察**：sklearn 的参数管理是一个**自洽的闭环**——`__init__` 约定支撑 `get_params`，`get_params` 支撑 `clone`，`clone` 支撑 `GridSearchCV`。违反任一环节，整个链条都会断裂。

### 16.1 本讲要点回顾

1. **`__init__` 不做事**：只原样存参数，校验放 `fit`。
2. **`get_params` 用反射**：从 `__init__` 签名自动提取参数名。
3. **`deep=True` 展平嵌套**：`clf__C` 命名支持嵌套搜索。
4. **`set_params` 支持嵌套**：`step__param` 设置嵌套参数。
5. **`clone` 非 `deepcopy`**：得到未训练的同参数副本。
6. **`__repr__` 自动生成**：从 `get_params` 生成，省略默认值。
7. **闭环自洽**：约定支撑机制，机制支撑上层功能。

---

## 上一讲 / 下一讲


[← 第二讲：Mixin 多继承架构](02-mixin-design.md) ｜  [第四讲：元估计器模式 →](04-meta-estimator.md）
