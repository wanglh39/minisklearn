# 第七讲：全局配置与演进

> **核心问题**：sklearn 怎么管理全局行为？它如何演进而不破坏向后兼容？一个有十几年历史的库怎么在保持兼容的同时持续进化？

---

## 1. 全局配置：`config_context`

sklearn 用 `config_context` 管理全局行为（如是否假设有序数组、是否转 DataFrame）：

```python
from sklearn import config_context

# 临时切换配置，退出 with 后恢复
with config_context(assume_finite=True):
    clf.fit(X, y)   # 在此期间跳过 NaN 检查，加速
# 退出后恢复默认
```

实现用 contextvars，线程安全。

### 1.1 为什么需要全局配置

有些行为不适合作为参数传入每个估计器。例如"是否跳过 NaN 检查"：

- 如果作为 `fit` 参数：`clf.fit(X, y, assume_finite=True)`，破坏统一 API 契约。
- 如果作为 `__init__` 参数：`LogisticRegression(assume_finite=True)`，每个估计器都要加一遍。
- 作为全局配置：一处设置，全局生效，不污染 API。

全局配置是"统一 API 契约"和"灵活行为控制"的折中。

### 1.2 基本用法

#### 1.2.1 临时切换

```python
from sklearn import config_context

with config_context(assume_finite=True):
    # 这里的 fit 跳过 NaN 检查
    clf.fit(X, y)
# 这里恢复默认（assume_finite=False）
clf.predict(X_test)   # 正常检查 NaN
```

`with` 退出后自动恢复，不会"忘关"。

#### 1.2.2 永久切换

```python
from sklearn import set_config

set_config(assume_finite=True)   # 全局永久生效
# 之后所有 fit 都跳过 NaN 检查
# 要手动恢复
set_config(assume_finite=False)
```

永久切换要谨慎，影响整个进程。

#### 1.2.3 查询当前配置

```python
from sklearn import get_config

print(get_config())   # {'assume_finite': False, 'display': 'diagram', ...}
```

### 1.3 实现原理：contextvars

sklearn 用 Python 3.7+ 的 `contextvars` 模块实现线程安全的全局配置：

```python
import contextvars

# 定义一个 context variable
_assume_finite = contextvars.ContextVar('assume_finite', default=False)

def get_config():
    return {'assume_finite': _assume_finite.get()}

def set_config(assume_finite=None):
    if assume_finite is not None:
        _assume_finite.set(assume_finite)

def config_context(**new_config):
    return _ConfigContext(**new_config)

class _ConfigContext:
    def __init__(self, **new_config):
        self.new_config = new_config
        self.tokens = []

    def __enter__(self):
        for key, value in self.new_config.items():
            ctx_var = _get_context_var(key)
            token = ctx_var.set(value)
            self.tokens.append((ctx_var, token))

    def __exit__(self, *args):
        for ctx_var, token in reversed(self.tokens):
            ctx_var.reset(token)
```

`contextvars` 的好处：

- **线程安全**：每个线程 / 协程有独立的配置副本。
- **自动恢复**：`with` 退出时自动 reset 到进入前的值。
- **可嵌套**：`with` 可以嵌套，内层退出后恢复到外层。

### 1.4 线程安全的意义

```python
import threading

def worker():
    with config_context(assume_finite=True):
        clf.fit(X, y)   # 这个线程跳过 NaN 检查

# 主线程不受影响
t = threading.Thread(target=worker)
t.start()
t.join()
# 主线程的 get_config() 还是 assume_finite=False
```

如果用普通全局变量，子线程的配置会泄漏到主线程。`contextvars` 隔离了每个线程的配置。

### 1.5 可用配置项

| 配置项                | 默认值     | 作用                               |
|-----------------------|------------|------------------------------------|
| `assume_finite`       | False      | 跳过 NaN/Inf 检查，加速            |
| `display`             | 'diagram'  | 估计器的显示方式（diagram / text / dict）|
| `print_changed_only`  | True       | repr 只显示非默认参数              |
| `skip_parameter_validation` | False | 跳过参数校验（极内部用）           |

#### 1.5.1 `assume_finite`

```python
with config_context(assume_finite=True):
    clf.fit(X_large, y)   # 跳过 NaN 检查，大数据加速 5-10%
```

风险：如果数据真有 NaN，会得到错误结果而不报错。只在确认数据干净时用。

#### 1.5.2 `display`

```python
from sklearn import set_config

set_config(display='diagram')   # Jupyter 里显示图示
set_config(display='text')      # 显示文本 repr
set_config(display='dict')      # 显示字典
```

```python
# display='diagram' 时，Jupyter 里 Pipeline 显示成流程图
Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
# → 图示：StandardScaler → LogisticRegression
```

#### 1.5.3 `print_changed_only`

```python
set_config(print_changed_only=True)
print(LogisticRegression(C=10))
# LogisticRegression(C=10)   # 只显示非默认参数

set_config(print_changed_only=False)
print(LogisticRegression(C=10))
# LogisticRegression(C=10, penalty='l2', tol=1e-4, ...)   # 显示全部
```

### 1.6 性能权衡

`assume_finite=True` 的加速效果：

```python
import timeit

X_large = np.random.randn(100000, 100)
y_large = np.random.randint(0, 2, 100000)

t1 = timeit.timeit(lambda: LogisticRegression().fit(X_large, y_large), number=5)

with config_context(assume_finite=True):
    t2 = timeit.timeit(lambda: LogisticRegression().fit(X_large, y_large), number=5)

print(f"不跳过: {t1:.2f}s, 跳过: {t2:.2f}s, 加速: {t1/t2:.2f}x")
# 典型：不跳过 5.2s, 跳过 4.8s, 加速 1.08x
```

加速不大（5-10%），因为 NaN 检查是 O(n)，相对 SVD 等算法开销小。但在"频繁 fit 小数据"的场景（例如 GridSearchCV）累积可观。

### 1.7 思考题

1. 为什么 `assume_finite` 是全局配置而不是 `fit` 参数？
2. `config_context` 嵌套时，内层退出后恢复到什么值？
3. 如果用普通全局变量而非 `contextvars`，多线程会出什么问题？
4. `display='diagram'` 在终端里会怎样？

---

## 2. 配置的层次：全局 / 线程 / 上下文

### 2.1 三层配置

sklearn 的配置有三个层次：

| 层次     | 设置方式              | 作用域           | 持续时间       |
|----------|-----------------------|------------------|----------------|
| 全局     | `set_config(...)`     | 整个进程         | 直到下次设置   |
| 线程     | `contextvars` 隔离    | 当前线程         | 线程生命周期   |
| 上下文   | `with config_context` | with 块内        | with 退出      |

### 2.2 优先级

上下文 > 线程 > 全局默认。

```python
set_config(assume_finite=False)   # 全局默认

def worker():
    # 线程内默认继承全局，但可以独立改
    with config_context(assume_finite=True):
        # 上下文覆盖
        assert get_config()['assume_finite'] is True
    # 退出上下文，回到线程默认（False）
    assert get_config()['assume_finite'] is False
```

### 2.3 配置的传递

配置不会通过函数调用传递，而是通过 `contextvars` 自动在**同一线程**内传递：

```python
def helper():
    # 自动看到外层的配置
    return get_config()['assume_finite']

with config_context(assume_finite=True):
    helper()   # 返回 True
helper()       # 返回 False
```

### 2.4 思考题

1. 为什么配置不通过函数参数传递，而用全局变量？
2. 在 multiprocessing 里，子进程能看到父进程的配置吗？
3. async/await 里配置会跨 await 传递吗？

---

## 3. SLEP：演进机制

sklearn 用 **SLEP**（Scikit-Learn Enhancement Proposals）管理演进，类似 Python 的 PEP：

- SLEP009：`__init__` 只存参数的约定
- SLEP010：关键词专用参数
- SLEP011：从 `fit_transform` 的默认实现中移除 `y`
- SLEP018：参数约束验证器
- SLEP007：模型选择的默认行为变更

每个 SLEP 是一份提案文档，经过社区讨论、投票、采纳。这保证了演进有据可查。

### 3.1 SLEP 是什么

SLEP = Scikit-Learn Enhancement Proposal。它是一份**结构化文档**，描述对 sklearn 的改进建议：

- **动机**：为什么要改。
- **提案**：具体怎么改。
- **向后兼容**：怎么处理旧代码。
- **替代方案**：考虑过但否决的其他方案。
- **讨论记录**：社区讨论的关键点。

### 3.2 SLEP 流程

1. **起草**：作者写 SLEP 草稿，发到邮件列表。
2. **讨论**：社区讨论，草稿迭代。
3. **评审**：核心开发者评审，决定 accept / reject / defer。
4. **实现**：accepted 后，作者或志愿者实现。
5. **合并**：实现通过 review 后合并到主分支。
6. **发布**：下个版本发布，可能带 deprecation warning。

### 3.3 重要 SLEP 详解

#### 3.3.1 SLEP009：`__init__` 只存参数

**动机**：早期估计器的 `__init__` 做各种工作（校验、转换），导致 `clone` 行为不可预测。

**提案**：`__init__` 只能把参数原样存到 `self`，不做任何工作。

```python
# 坏（SLEP009 之前）
class LogisticRegression:
    def __init__(self, C=1.0):
        self.C = float(C)   # 做了转换
        self._validate()    # 做了校验

# 好（SLEP009 之后）
class LogisticRegression:
    def __init__(self, C=1.0):
        self.C = C   # 原样存
    # 校验移到 fit
```

**影响**：`clone` 行为可预测，`get_params` / `set_params` 可靠。

#### 3.3.2 SLEP010：关键词专用参数

**动机**：某些参数应该只能用关键词传，避免位置参数歧义。

**提案**：新参数必须 keyword-only。

```python
# 旧：位置参数
clf.fit(X, y, sample_weight)   # sample_weight 是第 3 个位置参数

# 新：keyword-only
clf.fit(X, y, sample_weight=sample_weight)   # 必须用关键词
```

**影响**：API 更清晰，未来加参数不破坏位置参数顺序。

#### 3.3.3 SLEP011：`fit_transform` 移除 `y`

**动机**：`TransformerMixin.fit_transform` 默认实现接受 `y` 但忽略它，让人误以为 transformer 用了 `y`。

**提案**：`fit_transform` 不再接受 `y`（无监督 transformer）。

```python
# 旧
X_new = transformer.fit_transform(X, y)   # y 被忽略

# 新
X_new = transformer.fit_transform(X)   # 明确不用 y
```

**影响**：API 更明确，避免误导。

#### 3.3.4 SLEP018：参数约束验证器

**动机**：每个估计器自己写参数校验，重复且不一致。

**提案**：用装饰器声明参数约束，统一校验。

```python
from sklearn.utils.param_validation import validate_params, Interval

@validate_params(
    {"alpha": [Interval(float, 0, None, closed="left")]},
)
def __init__(self, alpha=1.0):
    self.alpha = alpha
```

**影响**：校验一致、错误信息统一、减少重复代码。

### 3.4 SLEP 与 PEP 的对比

| 方面       | PEP（Python）         | SLEP（sklearn）       |
|------------|------------------------|------------------------|
| 编号       | PEP 8, PEP 20, ...     | SLEP 009, SLEP 010, ...|
| 流程       | 起草 → 讨论 → 评审     | 起草 → 讨论 → 评审     |
| 影响范围   | Python 语言            | sklearn 库             |
| 例子       | PEP 8 风格指南         | SLEP 009 init 约定     |
| 强制性     | 部分（PEP 8 是建议）   | 强（核心约定）         |

SLEP 借鉴 PEP 的流程，但作用域是 sklearn 库本身。

### 3.5 思考题

1. 为什么 sklearn 需要 SLEP 而不是直接改代码？
2. SLEP009 如果不强制，会出什么问题？
3. SLEP010 的 keyword-only 在 Python 3 怎么实现？
4. 如果你想给 sklearn 提一个 SLEP，会提什么？

---

## 4. 向后兼容承诺

sklearn 对向后兼容非常谨慎：

- 公开 API 的移除要经过 **deprecation warning → 两个版本 → 移除** 的流程
- `fit` 后的属性名（`coef_` 等）一旦公开就不轻易改

这让用户代码不会因升级而突然崩溃。

### 4.1 deprecation 流程

#### 4.1.1 第一步：加 deprecation warning

```python
# 0.20 版本
def fit(self, X, y, n_jobs=None):
    if n_jobs is not None:
        warnings.warn(
            "'n_jobs' 在 0.22 弃化为弃用，1.0 移除。"
            "请用 sklearn.set_config(n_jobs=...) 代替。",
            FutureWarning,
        )
    # 仍然支持旧用法
    ...
```

#### 4.1.2 第二步：两个版本后移除

```python
# 1.0 版本（0.22 之后两个大版本）
def fit(self, X, y):
    # n_jobs 参数直接移除
    # 用户传 n_jobs 会报 TypeError
    ...
```

#### 4.1.3 时间线

```
0.20: 加 deprecation warning
0.22: 仍然有 warning，仍然能用
0.24: 仍然有 warning，仍然能用
1.0:  移除，传 n_jobs 报 TypeError
```

用户有 **3-4 个大版本**（约 1-2 年）的时间迁移。

### 4.2 deprecation warning 的实现

```python
from sklearn.utils.deprecation import deprecated

@deprecated("0.22", "1.0", "用 set_config(n_jobs=...) 代替")
def fit(self, X, y, n_jobs=None):
    ...
```

`deprecated` 装饰器自动生成 warning，包含版本号和替代方案。

### 4.3 不可变属性

`fit` 后的属性名一旦公开就不轻易改：

- `coef_`：线性模型系数，0.x 就有，永远不会改名。
- `classes_`：分类器类别，同上。
- `feature_importances_`：树模型特征重要性。

改属性名会破坏所有依赖它的代码（包括用户代码和下游库）。如果要改，要走 deprecation 流程，新旧并存一段时间。

```python
# 假设要把 coef_ 改成 weights_
def fit(self, X, y):
    self.weights_ = ...
    self.coef_ = self.weights_   # 旧名保留，加 deprecation
    warnings.warn("coef_ 弃化为 weights_", FutureWarning)
```

### 4.4 版本号约定

sklearn 用语义化版本（Semantic Versioning）的变体：

- **大版本**（0.20 → 0.22 → 0.24）：可能有 API 变更，带 deprecation warning。
- **小版本**（0.22.0 → 0.22.1）：只修 bug，不破 API。
- **1.0**：标记"API 稳定"，之后按严格 semver。

```python
import sklearn
print(sklearn.__version__)   # '1.3.0'
```

### 4.5 思考题

1. 为什么 deprecation 要等两个版本才移除？
2. `coef_` 改名为什么这么难？要走什么流程？
3. 语义化版本对 sklearn 的兼容承诺有什么影响？
4. 如果用户忽略了 deprecation warning，升级后代码崩了，是谁的责任？

---

## 5. deprecation warning 的实战

### 5.1 识别 deprecation warning

```python
import warnings
warnings.simplefilter('error', FutureWarning)   # 把 warning 变成 error

clf.fit(X, y)   # 如果用了弃用 API，会立刻报错
```

在测试里这么设置，能提前发现弃用用法。

### 5.2 抑制已知 deprecation

```python
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    clf.fit(X, y)   # 临时忽略
```

只在确认弃用无害时用，否则会掩盖真正的问题。

### 5.3 迁移示例

#### 5.3.1 `n_jobs` 从 `fit` 移到全局

```python
# 0.20：弃用
clf.fit(X, y, n_jobs=4)   # FutureWarning

# 迁移到
import sklearn
sklearn.set_config(n_jobs=4)   # 全局设置
clf.fit(X, y)                   # 不传 n_jobs
```

#### 5.3.2 `fit_transform` 的 `y` 移除

```python
# 0.20：弃用
X_new = transformer.fit_transform(X, y)   # FutureWarning

# 迁移到
X_new = transformer.fit_transform(X)   # 不传 y
```

#### 5.3.3 `OneHotEncoder` 的 `sparse` 改名

```python
# 旧
ohe = OneHotEncoder(sparse=False)

# 新
ohe = OneHotEncoder(sparse_output=False)
```

### 5.4 思考题

1. 怎么批量找出代码里所有弃用用法？
2. 抑制 deprecation warning 有什么风险？
3. 迁移期间新旧 API 并存，怎么测试两者都工作？

---

## 6. 版本迁移实战

### 6.1 0.20 → 0.22 迁移

主要变更：

- `fit_transform` 的 `y` 弃用。
- `OneHotEncoder` 默认 `sparse=True`（之前是 `sparse=False`）。
- 若干估计器的默认参数调整。

### 6.2 0.24 → 1.0 迁移

主要变更：

- 大量弃用 API 移除。
- `fit_transform` 的 `y` 完全移除。
- `OneHotEncoder` 的 `sparse` 改名 `sparse_output`。

### 6.3 迁移策略

1. **读 release notes**：每个版本的 CHANGELOG 列出所有变更。
2. **跑测试**：用 `warnings.simplefilter('error', FutureWarning)` 让弃用用法暴露。
3. **逐个迁移**：按 warning 信息逐个改。
4. **验证**：迁移后跑完整测试套件。

### 6.4 自动化迁移工具

sklearn 没有官方迁移工具，但可以用 `grep` 找弃用用法：

```bash
grep -r "n_jobs=" your_code/   # 找所有 n_jobs 用法
grep -r "fit_transform.*y" your_code/   # 找 fit_transform 传 y
```

### 6.5 思考题

1. 大版本迁移时，为什么要先跑测试再改代码？
2. release notes 应该重点读什么？
3. 如果迁移后精度下降，可能是什么原因？

---

## 7. 与其他框架对比

### 7.1 NumPy 的演进

NumPy 用 NEP（NumPy Enhancement Proposals），流程类似 SLEP：

- NEP 29：Python 版本支持策略。
- NEP 32：f2py 改进。

NumPy 的向后兼容承诺更强（数组是基础数据结构，改了影响整个生态）。

### 7.2 Pandas 的演进

Pandas 用 PDEP（Pandas Enhancement Proposals）：

- PDEP 8：API 标准化。
- PDEP 10：弃用 Panel。

Pandas 1.0 前后有大量 API 变更，迁移成本高。

### 7.3 PyTorch 的演进

PyTorch 没有正式的 PEP 机制，演进靠 RFC 和核心团队决策。向后兼容承诺较弱（PyTorch 还在快速演进）。

### 7.4 TensorFlow 的演进

TensorFlow 1.x → 2.x 是破坏性变更，没有 deprecation 过渡期，迁移成本极高。

### 7.5 对比表

| 框架     | 演进机制 | 兼容承诺 | 迁移成本 | 例子            |
|----------|----------|----------|----------|------------------|
| sklearn  | SLEP     | 强       | 低       | deprecation 流程 |
| NumPy    | NEP      | 极强     | 极低     | 几乎不破         |
| Pandas   | PDEP     | 中       | 中       | 1.0 大变更       |
| PyTorch  | RFC      | 弱       | 中       | 持续小变更       |
| TF       | RFC      | 弱       | 极高     | 1.x → 2.x        |

### 7.6 思考题

1. 为什么 sklearn 的兼容承诺比 PyTorch 强？
2. TF 1.x → 2.x 的破坏性变更有什么教训？
3. 演进机制和兼容承诺之间怎么平衡？

---

## 8. 演进的哲学

### 8.1 制度 vs 个人意志

sklearn 的演进靠**制度**（SLEP + deprecation 流程）而非个人意志。即使核心开发者也不能随意改 API。这种制度化是大型开源项目长期可维护的关键。

### 8.2 慢即是快

sklearn 的演进看起来慢（一个 SLEP 可能讨论一年），但：

- 慢讨论 → 想清楚 → 一次改对 → 不返工。
- 快改 → 没想清楚 → 反复改 → 兼容性包袱。

"慢即是快"在 API 设计里尤其成立。

### 8.3 兼容性是资产

向后兼容看起来是负担，其实是**资产**：

- 用户敢升级 → 生态健康。
- 下游库不用频繁适配 → 维护成本低。
- sklearn 的口碑 → 更多用户。

TF 1.x → 2.x 的破坏性变更让很多用户流失到 PyTorch，这就是兼容性资产丢失的代价。

### 8.4 何时可以破坏兼容

即使 sklearn 也有破坏兼容的时候（1.0 移除大量弃用 API）。判断标准：

- 弃用 API 已经警告了 **3-4 个版本**。
- 替代方案已经稳定。
- 用户有足够时间迁移。
- 继续维护旧 API 的成本太高。

破坏兼容是**最后的手段**，不是常规操作。

### 8.5 思考题

1. "制度 vs 个人意志"在开源项目里怎么体现？
2. "慢即是快"在什么情况下不成立？
3. 兼容性什么时候是负担？怎么判断？
4. 如果让你决定 sklearn 2.0 破坏什么兼容，你会破什么？

---

## 9. 配置与演进的协同

### 9.1 配置作为演进的缓冲

新行为可以通过配置引入，旧行为作为默认：

```python
# 0.22：新显示方式 diagram 作为可选
set_config(display='diagram')   # opt-in

# 0.24：diagram 成为默认
# 想用旧方式：set_config(display='text')
```

配置让用户**逐步适应**新行为，而不是一次性切换。

### 9.2 配置作为实验场

新功能可以先作为配置项引入，收集反馈，再决定是否成为默认：

```python
# 实验性配置
set_config(enable_experimental_feature=True)
```

如果反馈好，下个版本成为默认；如果反馈差，移除配置项。

### 9.3 配置的弃用

配置项本身也会弃用：

```python
# 旧配置项弃用
set_config(old_option=True)   # FutureWarning

# 迁移到
set_config(new_option=True)
```

走同样的 deprecation 流程。

### 9.4 思考题

1. 配置作为演进缓冲，有什么好处和坏处？
2. 实验性配置怎么管理？什么时候转正或移除？
3. 配置项太多会不会变成负担？怎么控制？

---

## 10. 实战：写一个带 deprecation 的估计器

```python
import warnings
from sklearn.base import BaseEstimator, RegressorMixin

class MyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, beta=None):
        self.alpha = alpha
        self.beta = beta   # 弃用参数

    def fit(self, X, y):
        if self.beta is not None:
            warnings.warn(
                "'beta' 在 1.2 弃用，1.4 移除。"
                "请用 'alpha' 代替，alpha=beta 的效果。",
                FutureWarning,
            )
            alpha = self.beta
        else:
            alpha = self.alpha

        self.mean_ = y.mean() * alpha
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)
```

### 10.1 测试 deprecation

```python
import pytest

def test_beta_deprecation():
    with pytest.warns(FutureWarning, match="beta.*弃用"):
        reg = MyRegressor(beta=2.0)
        reg.fit(X, y)

def test_alpha_still_works():
    reg = MyRegressor(alpha=2.0)
    reg.fit(X, y)   # 不应有 warning
```

### 10.2 思考题

1. deprecation warning 的信息应该包含什么？
2. 怎么测试"弃用参数仍然能用"？
3. 弃用参数和新参数同时传，应该怎么处理？

---

## 11. 常见问题与陷阱

### 11.1 配置不生效

```python
set_config(assume_finite=True)
# 在子进程里不生效（multiprocessing）
```

`contextvars` 不跨进程，子进程要单独设置。

### 11.2 deprecation warning 被吞

```python
warnings.simplefilter('ignore')   # 全局忽略，危险
clf.fit(X, y)   # 弃用 warning 被吞，不知道要迁移
```

### 11.3 版本判断错误

```python
import sklearn
if sklearn.__version__ < '1.0':   # 字符串比较，'0.24' < '1.0' 但 '0.10' < '0.9' 错
    ...
```

应该用 `packaging.version.parse`：

```python
from packaging.version import parse
if parse(sklearn.__version__) < parse('1.0'):
    ...
```

### 11.4 思考题

1. 配置在多进程里怎么正确传递？
2. 怎么确保 deprecation warning 不被意外吞掉？
3. 版本比较为什么不能用字符串比较？
4. 多进程 fork 后配置还在吗？为什么？

---

## 12. 进阶：配置系统的设计

### 12.1 配置项的注册

```python
_CONFIG_REGISTRY = {}

def register_config(name, default, validator=None):
    _CONFIG_REGISTRY[name] = {
        'default': default,
        'validator': validator,
        'var': contextvars.ContextVar(name, default=default),
    }

def set_config(**kwargs):
    for name, value in kwargs.items():
        if name not in _CONFIG_REGISTRY:
            raise ValueError(f"未知配置项 {name}")
        cfg = _CONFIG_REGISTRY[name]
        if cfg['validator']:
            cfg['validator'](value)
        cfg['var'].set(value)
```

### 12.2 配置的文档

每个配置项应该有：

- 名字、默认值、类型。
- 作用描述。
- 引入版本。
- 弃用状态（如果弃用）。

### 12.3 配置的测试

```python
def test_config_context_restores():
    old = get_config()
    with config_context(assume_finite=True):
        assert get_config()['assume_finite'] is True
    assert get_config() == old   # 恢复
```

### 12.4 思考题

1. 配置注册机制有什么好处？
2. 怎么给配置项加文档？
3. 配置系统的测试应该覆盖什么？
4. 配置注册能不能用 dataclass 或 pydantic 简化？

---

## 13. SLEP 案例研究

### 13.1 SLEP007：模型选择默认行为

**背景**：早期 `cross_val_score` 默认不 shuffle，导致分层 KFold 在有序数据上表现差。

**提案**：把默认 cv 从 KFold(3) 改成 StratifiedKFold(5, shuffle=True)。

**迁移**：

```python
# 0.20：旧行为
cross_val_score(clf, X, y, cv=3)   # KFold(3)

# 0.22：新行为
cross_val_score(clf, X, y)   # StratifiedKFold(5, shuffle=True)
```

**影响**：很多用户的 `cross_val_score` 结果变了，引发讨论。最终通过 SLEP 流程达成共识。

### 13.2 SLEP010：keyword-only 的实施

把位置参数改成 keyword-only 要改签名：

```python
# 旧
def fit(self, X, y, sample_weight=None):
    ...

# 新
def fit(self, X, y, *, sample_weight=None):   # * 之后是 keyword-only
    ...
```

`*` 之后的参数必须用关键词传。这是 Python 3 的语法。

### 13.3 SLEP018：参数验证装饰器

```python
from sklearn.utils.param_validation import validate_params, Interval, StrOptions

@validate_params(
    {
        "C": [Interval(float, 0, None, closed="left")],
        "penalty": [StrOptions({"l1", "l2", "elasticnet"})],
    },
    prefer_skip_nested_validation=True,
)
def __init__(self, C=1.0, penalty="l2"):
    self.C = C
    self.penalty = penalty
```

好处：

- 校验逻辑声明式，不重复。
- 错误信息统一。
- 可以从装饰器自动生成文档。

### 13.4 思考题

1. SLEP007 为什么引发争议？怎么解决的？
2. keyword-only 改动对调用者有什么影响？
3. SLEP018 的装饰器怎么和 `__init__` 只存参数的约定协调？
4. SLEP 的编号为什么不是连续的？跳过的编号代表什么？

---

## 14. 配置系统的边界

### 14.1 什么不该作为配置

- **算法参数**（如 `C`、`alpha`）：应该作为 `__init__` 参数，每个估计器独立。
- **数据相关**（如 `n_features`）：应该作为 `fit` 参数。
- **临时行为**（如 `verbose`）：可以作为参数，不污染全局。

### 14.2 什么应该作为配置

- **跨估计器的全局行为**（如 `assume_finite`）：所有估计器共享。
- **显示偏好**（如 `display`）：和算法无关。
- **性能调优**（如 `skip_parameter_validation`）：内部优化。

### 14.3 配置膨胀的控制

sklearn 的配置项很少（不到 10 个），因为：

- 能作为参数的尽量作为参数。
- 配置项要有**明确的全局语义**。
- 加配置要走 SLEP 流程。

### 14.4 思考题

1. `verbose` 为什么是参数而不是配置？
2. 配置项太多会有什么问题？
3. 怎么判断一个行为应该是配置还是参数？
4. 配置和参数的边界会不会随时间变化？

---

## 15. 演进中的权衡

### 15.1 速度 vs 稳定

- 快速演进 → 用户跟不上 → 流失。
- 过度稳定 → 技术债积累 → 维护成本高。

sklearn 倾向稳定，每半年一个大版本，变更可控。

### 15.2 新功能 vs 兼容

- 加新功能 → API 膨胀 → 学习成本高。
- 不加新功能 → 被竞品超越 → 用户流失。

sklearn 通过 SLEP 评审控制新功能的质量。

### 15.3 简洁 vs 灵活

- 简洁 API → 易学 → 但不够灵活。
- 灵活 API → 强大 → 但难学。

sklearn 用"统一 API + 配置"折中：核心 API 简洁，配置提供灵活性。

### 15.4 思考题

1. sklearn 半年一个大版本，PyTorch 几乎每月一版，哪个更好？
2. API 膨胀怎么控制？
3. 简洁和灵活的折中点在哪？
4. 演进速度和用户群体的关系是什么？

---

## 16. 历史教训

### 16.1 早期 `__init__` 做工作的代价

SLEP009 之前，`__init__` 做各种工作，导致：

- `clone` 行为不可预测。
- `get_params` 拿到的是转换后的值。
- 嵌套估计器（Pipeline）行为奇怪。

这个教训让 sklearn 团队后来对"约定"非常严格。

### 16.2 `fit_transform` 接受 `y` 的混淆

SLEP011 之前，`fit_transform(X, y)` 接受 `y` 但忽略，导致：

- 用户以为 transformer 用了 `y`（监督变换）。
- 实际是无监督，结果可复现性出问题。

移除 `y` 后，API 语义清晰。

### 16.3 TF 1.x → 2.x 的教训

TF 1.x → 2.x 没有 deprecation 过渡期，直接破坏兼容，导致：

- 大量用户代码崩溃。
- 用户流失到 PyTorch。
- TF 团队后来加了 `tf_upgrade_v2` 工具，但为时已晚。

sklearn 吸取教训，坚持 deprecation 流程。

### 16.4 思考题

1. SLEP009 之前的 `__init__` 做工作，具体怎么破坏 `clone`？
2. `fit_transform` 接受 `y` 为什么让人混淆？
3. TF 的教训对 sklearn 有什么影响？
4. 还有哪些框架经历过类似的"约定不严导致混乱"的教训？

---

## 17. 实战：模拟一次 SLEP 流程

### 17.1 提案：给所有估计器加 `n_jobs` 配置

**动机**：每个估计器自己有 `n_jobs` 参数，重复且不一致。统一到全局配置。

**提案**：

```python
# 弃用估计器的 n_jobs 参数
clf = LogisticRegression(n_jobs=4)   # FutureWarning
# 改用全局配置
set_config(n_jobs=4)
clf = LogisticRegression()
clf.fit(X, y)   # 用全局 n_jobs
```

**向后兼容**：

- 0.26：`n_jobs` 参数加 deprecation warning，仍然能用。
- 0.28：warning 继续。
- 1.2：移除 `n_jobs` 参数。

**讨论点**：

- 全局 `n_jobs` 会影响所有估计器，包括用户不想并行的。
- 嵌套估计器（Pipeline）的 `n_jobs` 怎么处理？
- 和 `joblib` 的并行机制怎么协调？

### 17.2 思考题

1. 这个提案的好处和坏处分别是什么？
2. 全局 `n_jobs` 会不会有意外影响？
3. 如果你是核心开发者，会投票 accept 还是 reject？
4. 这个提案如果要写 SLEP，应该包含哪些章节？
5. 怎么评估这个提案对下游库（imbalanced-learn 等）的影响？

---

## 18. 配置与并行的深度

### 18.1 joblib 并行的配置

sklearn 的并行靠 joblib，joblib 的行为可以通过 `joblib.parallel_backend` 配置：

```python
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=4):
    GridSearchCV(clf, param_grid).fit(X, y)   # 多线程

with parallel_backend('loky', n_jobs=4):
    GridSearchCV(clf, param_grid).fit(X, y)   # 多进程

# dask 后端
with parallel_backend('dask'):
    GridSearchCV(clf, param_grid).fit(X, y)   # 分布式
```

sklearn 的 `n_jobs` 参数最终传给 joblib，joblib 再根据 backend 决定怎么并行。

### 18.2 配置在并行中的传递

`contextvars` 在多线程里隔离，但 sklearn 通过 `joblib` 把配置显式传给子任务：

```python
from sklearn import get_config

def worker(X, y, config):
    # 子任务里恢复父任务的配置
    with config_context(**config):
        clf.fit(X, y)

config = get_config()
Parallel(n_jobs=4)(delayed(worker)(X_i, y_i, config) for X_i, y_i in chunks)
```

### 18.3 思考题

1. `contextvars` 在多线程里隔离，sklearn 怎么把配置传给并行任务？
2. `parallel_backend('dask')` 时，配置怎么传到 dask worker？
3. `n_jobs=-1` 是什么意思？怎么实现？
4. threading 后端和 loky 后端对配置传递有什么不同？
5. 并行任务里改配置，会影响主线程吗？

---

## 19. 演进的版本时间线

### 19.1 sklearn 大版本概览

| 版本  | 年份    | 重点变更                           |
|-------|---------|------------------------------------|
| 0.10  | 2013    | 引入 `check_estimator`             |
| 0.14  | 2014    | Pipeline 完善                      |
| 0.17  | 2015    | 大量新算法                         |
| 0.18  | 2016    | model_selection 模块重组           |
| 0.20  | 2018    | SLEP009 强制、大量 deprecation     |
| 0.22  | 2019    | 新画图 API、GradientBoosting 重写  |
| 0.24  | 2020    | SLEP018 参数验证                   |
| 1.0   | 2021    | API 稳定标记、移除大量弃用         |
| 1.1+  | 2022+   | 持续优化、新算法                   |

### 19.2 0.20 的意义

0.20 是分水岭：SLEP009 强制执行，所有估计器的 `__init__` 必须只存参数。这统一了内部约定，为后续的 `check_estimator` 加严打下基础。

### 19.3 1.0 的意义

1.0 标记"API 稳定"，意味着：

- 公开 API 不会大改。
- 弃用 API 大量移除。
- 之后按严格 semver。

### 19.4 思考题

1. 为什么 0.20 是分水岭？
2. 1.0 之前和之后的兼容承诺有什么不同？
3. semver 对 sklearn 意味着什么？
4. 0.x 阶段和 1.x 阶段的演进速度有什么变化？为什么？
5. 如果 sklearn 出 2.0，会破什么兼容？为什么？

---

## 20. 社区治理与演进

### 20.1 核心开发者

sklearn 由核心开发者团队治理，重大变更需要多数同意。

### 20.2 SLEP 投票

SLEP 通过邮件列表讨论，核心开发者投票。accept 需要足够多的赞成票。

### 20.3 用户反馈

用户反馈通过 GitHub issue 收集，影响 SLEP 的优先级。但 SLEP 不由用户投票决定——技术决策由核心开发者做。

### 20.4 思考题

1. 技术决策由核心开发者做，会不会忽视用户需求？
2. SLEP 投票的门槛应该多高？
3. 怎么平衡"技术正确"和"用户友好"？
4. 社区治理和公司治理（例如 TF 由 Google 主导）有什么区别？
5. 怎么避免"核心开发者倦怠"导致项目停滞？
6. 新人怎么参与 SLEP 讨论？门槛应该多高？

---

## 21. 配置的进阶模式

### 21.1 配置的继承与覆盖

```python
# 父函数设配置
def train_all():
    with config_context(assume_finite=True):
        for X, y in datasets:
            train_one(X, y)   # 子函数继承父配置

def train_one(X, y):
    # 自动看到 assume_finite=True
    clf.fit(X, y)
    with config_context(display='text'):   # 局部覆盖
        print(clf)
    # 退出后回到 assume_finite=True, display='diagram'
```

### 21.2 配置的序列化

配置可以保存到文件，下次恢复：

```python
import json
from sklearn import get_config, set_config

# 保存
with open('config.json', 'w') as f:
    json.dump(get_config(), f)

# 恢复
with open('config.json') as f:
    set_config(**json.load(f))
```

### 21.3 配置的调试

```python
from sklearn import get_config

def fit_with_logging(X, y):
    print(f"当前配置: {get_config()}")
    clf.fit(X, y)
```

### 21.4 思考题

1. 配置的继承在嵌套调用里怎么工作？
2. 序列化配置有什么用？什么场景需要？
3. 怎么调试"配置没生效"的问题？

---

## 22. deprecation 的高级模式

### 22.1 弃用整个类

```python
class OldEstimator:
    """已弃用，请用 NewEstimator。"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OldEstimator 在 1.2 弃用，请用 NewEstimator。",
            FutureWarning,
        )
        self._new = NewEstimator(*args, **kwargs)

    def fit(self, X, y):
        self._new.fit(X, y)
        return self

    def predict(self, X):
        return self._new.predict(X)
```

### 22.2 弃用参数的值

```python
# 弃用 penalty='l1'，改用 solver='liblinear'
def fit(self, X, y):
    if self.penalty == 'l1':
        warnings.warn(
            "penalty='l1' 在 1.2 弃用，请用 solver='liblinear' 自动选 l1。",
            FutureWarning,
        )
```

### 22.3 弃用默认值变更

```python
# 0.22: 默认 cv 从 3 改成 5
def cross_val_score(estimator, X, y, cv=None):
    if cv is None:
        warnings.warn(
            "cv 默认值从 3 改成 5。要保留旧行为，显式传 cv=3。",
            FutureWarning,
        )
        cv = 5
```

### 22.4 思考题

1. 弃用整个类和弃用方法有什么区别？
2. 弃用参数的某个值（而非整个参数）怎么处理？
3. 默认值变更的 deprecation 为什么特别麻烦？

---

## 23. 配置与测试的交互

### 23.1 测试里的配置隔离

```python
import pytest
from sklearn import set_config

@pytest.fixture(autouse=True)
def reset_config():
    """每个测试后恢复默认配置。"""
    old = get_config()
    yield
    set_config(**old)
```

确保一个测试改了配置不影响后续测试。

### 23.2 测试配置相关行为

```python
def test_assume_finite_skips_check():
    X = np.array([[1, 2], [np.nan, 3]])
    with config_context(assume_finite=True):
        # 不应报 NaN 错误（跳过检查）
        clf.fit(X, y)   # 可能产出错误结果，但不报错
```

### 23.3 思考题

1. 测试里不恢复配置会有什么后果？
2. 怎么测试"配置真的生效了"？
3. 配置相关的测试应该放在哪？
4. 怎么用 pytest fixture 自动恢复配置？
5. 测试 deprecation warning 时，怎么避免 warning 影响其他测试？
6. CI 里怎么强制所有 deprecation warning 都报错，防止漏迁移？

---

## 24. 配置与演进的常见误区

### 24.1 滥用全局配置

```python
# 坏：把算法参数塞到全局配置
set_config(learning_rate=0.01)   # 不应该
# 学习率是算法参数，应该 clf = LogisticRegression(learning_rate=0.01)
```

全局配置应该只放"跨估计器的全局行为"。

### 24.2 deprecation 不给替代方案

```python
# 坏：只说弃用，不说怎么改
warnings.warn("n_jobs 弃用", FutureWarning)

# 好：给替代方案
warnings.warn(
    "n_jobs 弃用，请用 set_config(n_jobs=...) 代替。"
    "详见 https://scikit-learn.org/...",
    FutureWarning,
)
```

### 24.3 破坏兼容不警告

```python
# 坏：直接改默认值，不警告
def cross_val_score(cv=5):   # 之前 cv=3，直接改了
    ...
```

应该先警告一个版本，再改默认值。

### 24.4 配置不恢复

```python
# 坏：改了配置不恢复
def my_func():
    set_config(assume_finite=True)
    clf.fit(X, y)
    # 忘了恢复，影响后续代码
```

应该用 `with config_context(...)` 自动恢复。

### 24.5 思考题

1. 滥用全局配置会有什么长期后果？
2. deprecation 不给替代方案，用户会怎样？
3. 破坏兼容不警告，对生态有什么影响？
4. 配置不恢复在长跑进程里会出什么问题？

---

## 25. 小结

| 要点             | 内容                                       |
|------------------|--------------------------------------------|
| 全局配置         | `config_context` / `set_config`            |
| 实现原理         | contextvars，线程安全                      |
| 演进机制         | SLEP，类似 PEP                             |
| 兼容承诺         | deprecation → 两版本 → 移除                |
| 不可变属性       | coef_ / classes_ 不轻易改                  |
| 配置与演进协同   | 配置作为新行为的缓冲                       |

**核心洞察**：sklearn 的演进靠**制度**（SLEP + deprecation 流程）而非个人意志。这种制度化的演进是大型开源项目长期可维护的关键。配置系统（`config_context`）让新行为可以平滑引入，deprecation 流程让旧 API 可以安全移除，两者协同让 sklearn 在十几年里持续进化而不失去用户。

---

## 14. 练习

### 14.1 基础练习

1. 用 `config_context` 临时切换 `assume_finite`，验证退出后恢复。
2. 写一个估计器，加一个弃用参数，发 FutureWarning。
3. 用 `warnings.simplefilter('error', FutureWarning)` 找出代码里的弃用用法。
4. 读 sklearn 的 CHANGELOG，列出 0.24 → 1.0 的主要变更。

### 25.2 进阶练习

5. 实现一个 mini `config_context`，用 `contextvars` 管理一个自定义配置项。
6. 写一个 `@deprecated` 装饰器，自动生成 deprecation warning。
7. 给 `MyRegressor` 写完整测试，覆盖弃用参数和新参数。
8. 用 `packaging.version.parse` 写一个版本兼容的工具函数。
9. 实现一个配置注册机制，支持动态添加配置项和校验。
10. 写一个脚本，扫描代码里所有 sklearn 弃用 API 的用法并报告。
11. 模拟一次 SLEP 流程：写一份 SLEP 草稿，包含动机、提案、兼容方案。

### 25.3 思考题

9. 如果让你给 sklearn 提一个 SLEP，会提什么？写出动机和提案。
10. sklearn 1.0 移除了哪些 0.x 的弃用 API？为什么是这些？
11. 配置系统和 `__init__` 参数的边界在哪？什么该作为配置，什么该作为参数？
12. 如果 sklearn 要支持 GPU，配置系统要加什么？怎么平滑引入？
13. deprecation 流程的"两个版本"是怎么定的？为什么不是一个或三个？
14. 配置在分布式计算（dask、ray）里怎么传递？有什么挑战？
15. 怎么用配置系统实现"实验性功能"的灰度发布？
16. SLEP 和 PEP 在治理上的根本区别是什么？各自适合什么规模的项目？
17. 如果一个第三方库依赖 sklearn 的弃用 API，它该怎么保护自己？
18. `config_context` 用 `contextvars`，如果改用 threading.local 会有什么区别？
19. deprecation warning 的"两个版本"在快速迭代项目和慢速迭代项目里分别合不合适？
20. 怎么设计一个"配置变更审计日志"，记录谁在什么时候改了什么配置？

---

## 上一讲 / 下一讲

[← 第六讲：一致性测试机制](06-consistency-testing.md) ｜  [第八讲：架构总览 →](08-architecture-overview.md）
