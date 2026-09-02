# 第二讲：Mixin 多继承架构

> **核心问题**：为什么 sklearn 不用一个大的 `BaseEstimator` 包揽所有功能，而是拆成 `BaseEstimator` + 4 个 Mixin？多继承在这里解决了什么问题？

---

## 1. 问题场景

考虑一个现实需求：

- 分类器需要 `score` 方法，返回**准确率**
- 回归器也需要 `score` 方法，返回 **R²**
- 转换器需要 `fit_transform` 方法
- 有些算法既是转换器又是分类器（如 LDA）
- 有些算法既是分类器又是聚类器

如果用一个大的基类：

```python
# 坏设计：上帝基类
class BaseEstimator:
    def score(self, X, y):
        if self._is_classifier:
            return accuracy(y, self.predict(X))
        elif self._is_regressor:
            return r2_score(y, self.predict(X))
        else:
            raise NotImplementedError
```

问题：

1. `score` 里要 `if-else` 判断"我是谁"——坏味道
2. 所有算法都继承同一个基类，基类越来越臃肿
3. 一个算法要同时是分类器和转换器时，`_is_classifier` 怎么设？

### 1.1 "上帝基类"的具体危害

让我们深入看看"上帝基类"为什么是坏设计：

```python
# 上帝基类：什么都往里塞
class BaseEstimator:
    def score(self, X, y):
        if self._is_classifier:
            return accuracy(y, self.predict(X))
        elif self._is_regressor:
            return r2_score(y, self.predict(X))
        elif self._is_clusterer:
            return silhouette_score(X, self.labels_)
        else:
            raise NotImplementedError

    def fit_transform(self, X, y=None):
        if self._is_transformer:
            return self.fit(X, y).transform(X)
        else:
            raise NotImplementedError

    def fit_predict(self, X, y=None):
        if self._is_clusterer:
            self.fit(X)
            return self.labels_
        else:
            raise NotImplementedError

    # ... 还有 get_params, set_params, clone, repr ...
```

危害清单：

1. **类型判断散落**：每个方法都要 `if-else` 判断"我是谁"，违反开闭原则
2. **基类膨胀**：每加一种新身份（如"密度估计器"），就要改基类
3. **身份冲突**：LDA 既是分类器又是转换器，`_is_classifier` 和 `_is_transformer` 都设 True？那 `score` 走哪个分支？
4. **测试困难**：测一个不相关的算法，也要被基类的所有方法影响
5. **理解困难**：看基类源码，要理解所有身份的逻辑

这是典型的"上帝对象"反模式——一个类承担太多职责。

### 1.2 如果用单继承层次

另一种方案是单继承层次：

```python
class BaseEstimator:
    def get_params(self): ...

class Predictor(BaseEstimator):
    def predict(self, X): ...

class Classifier(Predictor):
    def score(self, X, y):
        return accuracy(y, self.predict(X))

class Regressor(Predictor):
    def score(self, X, y):
        return r2_score(y, self.predict(X))

class Transformer(BaseEstimator):
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

问题来了：LDA 既是分类器又是转换器，单继承下无法表达：

```python
# 单继承下，LDA 只能继承一个
class LDA(Classifier): ...  # 那就不是 Transformer 了
class LDA(Transformer): ...  # 那就不是 Classifier 了
```

单继承无法表达"多重身份"——这正是 Mixin 多继承要解决的问题。

---

## 2. sklearn 的解法：Mixin

sklearn 把"身份"拆成独立的 Mixin，每个 Mixin 只提供该身份的协议方法：

```python
class BaseEstimator:
    """只管参数管理、克隆、repr —— 与算法类型无关"""
    def get_params(self, deep=True): ...
    def set_params(self, **params): ...

class ClassifierMixin:
    """只管分类器的 score（accuracy）"""
    def score(self, X, y):
        return accuracy(y, self.predict(X))

class RegressorMixin:
    """只管回归器的 score（R²）"""
    def score(self, X, y):
        return r2_score(y, self.predict(X))

class TransformerMixin:
    """只管转换器的 fit_transform"""
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

算 法类按需多继承：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin): ...
class LinearRegression(BaseEstimator, RegressorMixin): ...
class StandardScaler(BaseEstimator, TransformerMixin): ...
class LDA(BaseEstimator, ClassifierMixin, TransformerMixin): ...  # 双重身份！
```

### 2.1 这个设计为什么好

对比"上帝基类"方案，Mixin 方案的优势：

1. **职责分离**：每个 Mixin 只管一件事，源码清晰
2. **开闭原则**：加新身份只需加新 Mixin，不改现有代码
3. **多重身份**：LDA 自然地继承两个 Mixin，无需特殊处理
4. **按需组合**：算法只继承它需要的 Mixin，不背不必要的包袱
5. **测试独立**：每个 Mixin 可独立测试

### 2.2 Mixin 的字面含义

"Mixin"（混入）这个词源自冰淇淋店——你选一个基础口味（香草），然后"混入"各种配料（巧克力碎片、坚果、草莓酱）。

在 OOP 里：

- **基础口味** = `BaseEstimator`（提供参数管理等基础设施）
- **配料** = `ClassifierMixin` / `TransformerMixin` 等（提供特定能力）
- **最终产品** = `LogisticRegression(BaseEstimator, ClassifierMixin)`（香草 + 巧克力碎片）

这个比喻帮助理解 Mixin 的本质——**能力注入**，而非分类学继承。

---

## 3. Mixin 的三个关键特征

### 3.1 不存状态

Mixin **不定义 `__init__`**，不存储任何属性。它只提供方法，状态全靠宿主类管理。

```python
class ClassifierMixin:
    # 没有 __init__！
    # 没有属性！
    # 只有方法！

    def score(self, X, y):
        # 用 self.predict —— 这个方法由具体算法类提供
        return np.mean(self.predict(X) == y)
```

为什么重要？因为 Mixin 不参与初始化链，不会和宿主类的 `__init__` 冲突。多继承的 `__init__` 顺序是 Python 最容易出 bug 的地方，Mixin 避开了这个雷区。

#### 3.1.1 多继承 `__init__` 的雷区

让我们看看多继承 `__init__` 为什么容易出 bug：

```python
class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B:
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A, B):
    def __init__(self):
        print("C.__init__")
        super().__init__()

C()
# 输出顺序取决于 MRO，容易让人困惑
# C.__init__ → A.__init__ → B.__init__ → object.__init__
```

如果 A 和 B 的 `__init__` 期望不同的参数，就更混乱：

```python
class A:
    def __init__(self, a):
        self.a = a
        super().__init__()  # 但 B.__init__ 期望 b！

class B:
    def __init__(self, b):
        self.b = b
        super().__init__()

class C(A, B):
    def __init__(self, a, b):
        super().__init__(a, b)  # 怎么传？A 只要 a，B 只要 b
```

Mixin 不定义 `__init__`，完全避开了这些问题——宿主类的 `__init__` 独自管理所有状态，Mixin 只提供方法。

#### 3.1.2 Mixin 不存状态的具体体现

```python
# ClassifierMixin 不存任何状态
class ClassifierMixin:
    _estimator_type = "classifier"  # 类属性，不是实例属性

    def score(self, X, y):
        # 没有 self.xxx = ... 的赋值！
        # 只读取 self.predict(X) 的结果
        return np.mean(self.predict(X) == y)
```

对比会存状态的"坏 Mixin"：

```python
# ❌ 坏 Mixin：存了状态
class BadClassifierMixin:
    def score(self, X, y):
        self._last_score_ = np.mean(self.predict(X) == y)  # 存了状态！
        return self._last_score_
```

这个"坏 Mixin"会在宿主类上创建 `_last_score_` 属性，可能和宿主类的属性冲突，也违反了"predict/score 是纯查询"的约定。

### 3.2 依赖宿主类的方法

Mixin 的方法会调用 `self.predict`、`self.transform` 等——这些方法 Mixin 自己不实现，由宿主类提供。这是**协议**而非实现：

```python
class ClassifierMixin:
    def score(self, X, y):
        # 假设 self 有 predict 方法（由具体分类器实现）
        # 如果没有，运行时抛 AttributeError
        return np.mean(self.predict(X) == y)
```

这和抽象基类的关系：Mixin 不强制要求宿主类实现 `predict`，但宿主类不实现就会在调用 `score` 时报错。这种"软依赖"是 Mixin 的特点。

#### 3.2.1 协议 vs 实现的理解

Mixin 提供的是"协议"——"如果你有 `predict`，我就给你 `score`"：

```python
# Mixin 的逻辑：你有 predict，我给你 score
class ClassifierMixin:
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    #                ^^^^^^^^^^^^^^^^
    #                这个由你（宿主类）提供

# 宿主类提供 predict
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def predict(self, X):
        return ...  # 具体实现
    # 不用写 score，ClassifierMixin 给了
```

这是一种**能力组合**——宿主类提供"原子能力"（predict），Mixin 提供"派生能力"（score）。

#### 3.2.2 软依赖的错误处理

如果宿主类没实现 `predict`，调用 `score` 时会报错：

```python
class BrokenAlgo(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self
    # 忘了实现 predict！

algo = BrokenAlgo().fit(X, y)
algo.score(X, y)
# AttributeError: 'BrokenAlgo' object has no attribute 'predict'
```

错误信息不太友好（AttributeError 而非 NotImplementedError）。sklearn 用 `check_estimator` 在开发期发现这类问题。

### 3.3 可组合

一个类可以继承多个 Mixin，获得多种身份：

```python
# LDA 既能分类又能降维
class LinearDiscriminantAnalysis(
    BaseEstimator,
    ClassifierMixin,    # → score 返回 accuracy
    TransformerMixin,   # → fit_transform
):
    def fit(self, X, y): ...
    def predict(self, X): ...     # 分类
    def transform(self, X): ...   # 降维
```

用户可以 `lda.predict(X)` 做分类，也可以 `lda.transform(X)` 做降维。两种身份和平共处，因为 Mixin 的方法不冲突（`ClassifierMixin.score` 和 `TransformerMixin.fit_transform` 是不同方法）。

#### 3.3.1 多重身份的实际例子

sklearn 中有多重身份的算法：

| 算法 | 身份 | 用法 |
|------|------|------|
| LDA | Classifier + Transformer | 分类 or 降维 |
| GaussianMixture | Clusterer + Density | 聚类 or 密度估计 |
| KMeans | Clusterer + Transformer | 聚类 or 距离转换 |

```python
# LDA 做分类
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X_test)  # 分类

# LDA 做降维
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_2d = lda.transform(X)  # 降维到 2 维
```

两种用法，同一个类，靠 Mixin 多继承实现。

#### 3.3.2 组合不冲突的条件

多个 Mixin 组合不冲突的条件：它们提供的方法名不重复。

```python
# ✅ 不冲突：方法名不同
class ClassifierMixin:
    def score(self, X, y): ...      # score

class TransformerMixin:
    def fit_transform(self, X): ... # fit_transform

# 组合 OK
class LDA(BaseEstimator, ClassifierMixin, TransformerMixin): ...

# ❌ 冲突：方法名相同
class ClassifierMixin:
    def score(self, X, y):
        return accuracy(...)  # 返回 accuracy

class RegressorMixin:
    def score(self, X, y):
        return r2_score(...)  # 返回 R²

# 组合冲突：score 用哪个？
class BadAlgo(BaseEstimator, ClassifierMixin, RegressorMixin): ...
# MRO 决定用 ClassifierMixin.score，但这是设计错误
```

sklearn 的设计保证：**不会同时继承 `ClassifierMixin` 和 `RegressorMixin`**——一个算法不会既是分类器又是回归器。

---

## 4. MRO：多继承的方法解析顺序

多继承最让人头疼的是方法冲突。如果 `ClassifierMixin` 和 `RegressorMixin` 都定义了 `score`，继承两个时用哪个？

Python 用 **C3 线性化**算法计算 MRO（Method Resolution Order）：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    ...

print(LogisticRegression.__mro__)
# (LogisticRegression, BaseEstimator, ClassifierMixin, object)
```

sklearn 的 Mixin 设计**刻意避免了冲突**：

- `ClassifierMixin.score` 和 `RegressorMixin.score` 不会同时出现在一个类中（一个算法不会既是分类器又是回归器）
- `TransformerMixin` 提供 `fit_transform`，不与 `score` 冲突
- `BaseEstimator` 提供 `get_params` / `set_params`，不与 Mixin 冲突

所以 sklearn 的多继承虽然用了多继承，但 MRO 非常清晰，不会有"菱形继承"的歧义。

### 4.1 C3 线性化算法简介

Python 的 MRO 用 C3 线性化算法计算。简单说，C3 保证：

1. **子类在父类前**：`LogisticRegression` 在 `BaseEstimator` 前
2. **继承顺序保持**：`BaseEstimator` 在 `ClassifierMixin` 前（因为声明顺序）
3. **单调性**：如果 A 在 B 的 MRO 中，那 A 在所有子类的 MRO 中也在 B 前

```python
class A: pass
class B: pass
class C(A, B): pass

print(C.__mro__)
# (C, A, B, object)
# C 在最前，然后 A（先继承的），然后 B，最后 object
```

### 4.2 sklearn 的 MRO 实例

```python
from sklearn.linear_model import LogisticRegression
print(LogisticRegression.__mro__)
# (LogisticRegression, BaseEstimator, ClassifierMixin, object)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
print(LinearDiscriminantAnalysis.__mro__)
# (LinearDiscriminantAnalysis, BaseEstimator, ClassifierMixin, TransformerMixin, object)
```

注意 LDA 的 MRO：`BaseEstimator` → `ClassifierMixin` → `TransformerMixin`。如果两个 Mixin 有同名方法，`ClassifierMixin` 的会赢（因为它在前）。

### 4.3 菱形继承问题

菱形继承是 A → B, A → C, D → B, D → C：

```python
class A:
    def foo(self): return "A"

class B(A):
    def foo(self): return "B"

class C(A):
    def foo(self): return "C"

class D(B, C): pass

print(D().foo())  # "B"（MRO: D → B → C → A → object）
```

`D` 的 `foo` 是 `B` 的还是 `C` 的？C3 线性化给出确定答案（B 的，因为 B 先继承）。

sklearn 的 Mixin 设计**避免了菱形继承**：

```
BaseEstimator
    ↑
    ├── ClassifierMixin（不继承 BaseEstimator）
    └── TransformerMixin（不继承 BaseEstimator）
            ↑
        LDA(BaseEstimator, ClassifierMixin, TransformerMixin)
```

Mixin 不继承 `BaseEstimator`，所以没有菱形。这是刻意的设计——Mixin 是"平行的能力"，不是"层次的关系"。

### 4.4 查看 MRO 的技巧

调试多继承问题时，查看 MRO 很有用：

```python
# 打印 MRO
for cls in type(obj).__mro__:
    print(cls.__name__)

# 查看方法来自哪个类
import inspect
print(inspect.getmro(type(obj)))  # 同 __mro__
print(obj.score.__qualname__)  # 'ClassifierMixin.score'，看 score 来自哪
```

---

## 5. `_estimator_type`：身份标识

每个 Mixin 设置一个 `_estimator_type` 字符串，方便元估计器和测试框架判断身份：

```python
class ClassifierMixin:
    _estimator_type = "classifier"

class RegressorMixin:
    _estimator_type = "regressor"
```

这个属性用于：

```python
# 元估计器根据类型做不同处理
if estimator._estimator_type == "classifier":
    # 用 stratified split
else:
    # 用普通 split
```

为什么不用 `isinstance(estimator, ClassifierMixin)`？

1. 鸭子类型：用户自定义算法可能没继承 Mixin，但设了 `_estimator_type`
2. 灵活性：某些特殊算法可能想改变类型标识

### 5.1 `_estimator_type` 的实际用途

```python
# cross_val_score 根据类型选 split 策略
def cross_val_score(estimator, X, y, cv=5):
    if estimator._estimator_type == "classifier":
        # 分类：用 StratifiedKFold 保持类别比例
        splitter = StratifiedKFold(cv)
    else:
        # 回归：用普通 KFold
        splitter = KFold(cv)
    ...

# GridSearchCV 根据类型选评分指标
class GridSearchCV:
    def fit(self, X, y):
        if self.scoring is None:
            # 默认评分：分类用 accuracy，回归用 R²
            if self.estimator._estimator_type == "classifier":
                self.scoring = "accuracy"
            else:
                self.scoring = "r2"
        ...
```

`_estimator_type` 让元估计器能根据被包装算法的类型做不同处理，而不用 `isinstance` 检查。

### 5.2 鸭子类型下的 `_estimator_type`

```python
# 用户自定义算法，不继承 ClassifierMixin，但设了 _estimator_type
class MyAlgo:
    _estimator_type = "classifier"  # 手动设置

    def fit(self, X, y): ...
    def predict(self, X): ...

# cross_val_score 仍能正确识别
cross_val_score(MyAlgo(), X, y, cv=5)  # 用 StratifiedKFold
```

这是鸭子类型的体现——不看"你是不是 ClassifierMixin 的子类"，只看"你有没有 `_estimator_type = 'classifier'`"。

---

## 6. Mixin vs 其他设计模式

### 6.1 Mixin vs 接口（Java interface）

Java 的 interface 也定义协议但不存状态。区别：

- Java interface 是**编译期强制**：不实现接口方法会编译失败
- Python Mixin 是**运行期检查**：不实现会在调用时报错
- Java 一个类只能继承一个抽象类 + 多个接口
- Python 可以多继承多个 Mixin

#### 6.1.1 用 Java interface 模拟 sklearn

如果 sklearn 用 Java 写，大概是这样：

```java
interface Classifier {
    default double score(double[][] X, double[] y) {
        // Java 8+ 的 default method
        double[] yPred = predict(X);
        return accuracy(y, yPred);
    }
    double[] predict(double[][] X);  // 宿主类必须实现
}

class LogisticRegression extends BaseEstimator implements Classifier {
    public double[] predict(double[][] X) { ... }
    // score 由 Classifier 接口的 default method 提供
}
```

Java 8 的 default method 让 interface 能提供默认实现，类似 Mixin。但 Java 的 interface 不能有状态，而 Python Mixin 可以（虽然 sklearn 不用）。

### 6.2 Mixin vs Trait（Scala/Rust trait）

Trait 和 Mixin 概念上很接近，都是"可组合的协议"。区别：

- Rust trait 可以有默认实现，且编译期检查
- Python Mixin 有默认实现，但运行期检查
- Rust trait 的组合冲突需要显式解决
- Python Mixin 靠设计避免冲突

#### 6.2.1 Rust trait 的冲突处理

```rust
trait Classifier {
    fn score(&self, x: &[f64], y: &[f64]) -> f64 {
        accuracy(y, self.predict(x))
    }
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}

trait Regressor {
    fn score(&self, x: &[f64], y: &[f64]) -> f64 {
        r2_score(y, self.predict(x))
    }
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}

// 同时实现两个 trait，score 冲突
struct MyAlgo;
impl Classifier for MyAlgo { ... }
impl Regressor for MyAlgo { ... }
// Rust 要求显式解决冲突：调用时指定 Classifier::score(&obj, x, y) or Regressor::score(&obj, x, y)
```

Rust 强制你显式解决冲突，Python 靠 MRO 隐式决定。Rust 更安全，Python 更灵活。

### 6.3 Mixin vs 装饰器

装饰器也能给类增加方法：

```python
@classifier_mixin
class MyAlgo:
    def predict(self, X): ...
    # 自动获得 score 方法
```

但 Mixin 更自然：

1. 继承关系清晰，IDE 能看到
2. `isinstance` 检查可用
3. 多个 Mixin 组合比嵌套装饰器可读

#### 6.3.1 装饰器方案的实现

```python
def classifier_mixin(cls):
    """装饰器：给类加 score 方法"""
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    cls.score = score
    cls._estimator_type = "classifier"
    return cls

@classifier_mixin
class MyAlgo:
    def fit(self, X, y): return self
    def predict(self, X): return ...

# MyAlgo 现在有 score 方法
algo = MyAlgo().fit(X, y)
algo.score(X, y)  # 能用
```

但装饰器有缺点：

```python
# 1. isinstance 检查不可用
isinstance(MyAlgo(), ClassifierMixin)  # False（装饰器没改继承关系）

# 2. 多个装饰器嵌套，可读性差
@transformer_mixin
@classifier_mixin
class LDA: ...
# 顺序敏感，难调试

# 3. IDE 看不到 score 方法（动态添加）
```

所以 sklearn 选 Mixin 而非装饰器。

### 6.4 Mixin vs 多重继承的"分类学"

传统多继承用于"分类学"——"蝙蝠既是哺乳动物又会飞"：

```python
class Mammal: ...
class Flyable: ...

class Bat(Mammal, Flyable): ...  # 分类学多继承
```

sklearn 的 Mixin 不是分类学，而是"能力注入"——"LogisticRegression 有分类能力"：

```python
class BaseEstimator: ...
class ClassifierMixin: ...  # 能力，不是分类

class LogisticRegression(BaseEstimator, ClassifierMixin): ...
```

区别在于心智模型——分类学关注"是什么"，能力注入关注"能做什么"。

---

## 7. 自己实现：四个 Mixin

```python
class ClassifierMixin:
    """分类器 Mixin"""
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if sample_weight is not None:
            return np.average(y_pred == y, weights=sample_weight)
        return np.mean(y_pred == y)


class RegressorMixin:
    """回归器 Mixin"""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class TransformerMixin:
    """转换器 Mixin"""
    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            self.fit(X, **fit_params)
        else:
            self.fit(X, y, **fit_params)
        return self.transform(X)


class ClusterMixin:
    """聚类器 Mixin"""
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.labels_
```

注意每个 Mixin 的方法都调用 `self.predict` / `self.transform` / `self.fit`——这些由宿主类提供。Mixin 只定义"有了这些基础方法后，可以组合出什么"。

### 7.1 逐个解析

#### 7.1.1 ClassifierMixin

```python
class ClassifierMixin:
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if sample_weight is not None:
            return np.average(y_pred == y, weights=sample_weight)
        return np.mean(y_pred == y)
```

- `_estimator_type = "classifier"`：身份标识
- `score`：返回准确率（accuracy）
- `sample_weight`：支持样本权重
- 依赖 `self.predict`：由宿主类提供

#### 7.1.2 RegressorMixin

```python
class RegressorMixin:
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        return 1 - ss_res / ss_tot  # R²
```

- `score`：返回 R²（决定系数）
- R² = 1 - SS_res/SS_tot，越接近 1 越好
- 依赖 `self.predict`

#### 7.1.3 TransformerMixin

```python
class TransformerMixin:
    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            self.fit(X, **fit_params)
        else:
            self.fit(X, y, **fit_params)
        return self.transform(X)
```

- `fit_transform`：等价于 `fit(X).transform(X)`
- 依赖 `self.fit` 和 `self.transform`
- 允许子类覆盖以优化（如 PCA）

#### 7.1.4 ClusterMixin

```python
class ClusterMixin:
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.labels_  # fit 后学出的聚类标签
```

- `fit_predict`：fit 后返回聚类标签
- 依赖 `self.fit` 和 `self.labels_`（fit 后属性）

### 7.2 Mixin 之间的方法不冲突

检查四个 Mixin 提供的方法：

| Mixin | 提供的方法 |
|-------|-----------|
| ClassifierMixin | score |
| RegressorMixin | score |
| TransformerMixin | fit_transform |
| ClusterMixin | fit_predict |

冲突点：`ClassifierMixin.score` 和 `RegressorMixin.score` 同名。但 sklearn 设计保证一个类不会同时继承两者（一个算法不会既是分类器又是回归器）。

其余方法名都不同，可自由组合。

---

## 8. 为什么不用组合代替继承？

"组合优于继承"是面向对象的金科玉律。为什么 sklearn 在这里用了继承？

因为这里的继承表达的是 **is-a**（是一个）关系，而非 has-a（有一个）：

- `LogisticRegression` **是一个** `Classifier`（不是"有一个分类器"）
- `StandardScaler` **是一个** `Transformer`

Mixin 的继承语义是"获得某种能力"，而非"包含某个组件"。这和"组合优于继承"不矛盾——后者针对的是 `Car has-a Engine` 这种包含关系。

sklearn 在**元估计器**层面确实用了组合：

```python
# Pipeline 组合了多个估计器（has-a）
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # 组合，不是继承
```

所以 sklearn 的设计是：**能力用 Mixin 继承，组件用组合**。下一讲讲参数管理，再下一讲讲元估计器的组合。

### 8.1 is-a vs has-a 的区分

| 关系 | 例子 | 用什么 |
|------|------|--------|
| is-a（是一个） | LogisticRegression 是一个 Classifier | 继承（Mixin） |
| has-a（有一个） | Pipeline 有若干步骤 | 组合 |
| uses-a（使用） | fit 使用 check_X_y | 依赖（调用） |

```python
# is-a：LogisticRegression 是 Classifier
class LogisticRegression(BaseEstimator, ClassifierMixin): ...

# has-a：Pipeline 有步骤
class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps  # 持有步骤的引用

# uses-a：fit 使用 check_X_y
class LogisticRegression:
    def fit(self, X, y):
        X, y = check_X_y(X, y)  # 调用工具函数
```

三种关系，三种用法，各得其所。

### 8.2 "组合优于继承"的真正含义

"组合优于继承"针对的是**用继承表达 has-a** 的误用：

```python
# ❌ 误用继承表达 has-a
class Car(Engine):  # Car 不是 Engine
    def drive(self):
        self.start()  # 继承 Engine 的方法

# ✅ 用组合表达 has-a
class Car:
    def __init__(self):
        self.engine = Engine()  # Car 有 Engine
    def drive(self):
        self.engine.start()
```

sklearn 的 Mixin 继承不是这种误用——`LogisticRegression` 真的"是一个"分类器，继承 `ClassifierMixin` 获得 `score` 方法是合理的。

---

## 9. Mixin 的历史和业界使用

### 9.1 Mixin 的起源

Mixin 概念最早出现在 Lisp 的 Flavors 系统和 CLOS 中，后来被 Python、Ruby、Scala 等语言广泛采用。

- **Lisp Flavors**（1980s）：最早的多继承系统之一
- **CLOS**（Common Lisp Object System）：标准化多继承
- **Ruby**：`include` 和 `extend` 机制，Mixin 是核心设计
- **Python**：多继承 + MRO，Mixin 是常见模式
- **Scala**：Trait，类似 Mixin 但更强大
- **Rust**：Trait，编译期检查

### 9.2 不同语言的 Mixin 对比

| 语言 | 机制 | 冲突处理 | 状态 |
|------|------|---------|------|
| Python | 多继承 | MRO（C3） | Mixin 可存状态（但 sklearn 不用） |
| Ruby | include/extend | 后 include 覆盖 | Mixin 可存状态 |
| Scala | Trait | 显式解决 | Trait 可有状态 |
| Rust | Trait | 显式解决 | Trait 不能有状态 |
| Java | interface + default | 不允许冲突 | interface 不能有状态 |

sklearn 的 Mixin 用法是 Python 最经典的 Mixin 模式——不存状态、提供方法、可组合。

### 9.3 其他 Python 库的 Mixin 用法

```python
# Django 的 Class-Based View 用 Mixin
class LoginRequiredMixin:
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        return super().dispatch(request, *args, **kwargs)

class MyView(LoginRequiredMixin, ListView): ...

# Flask-View 的 Mixin
class CRUDMixin:
    def create(self, **kwargs): ...
    def update(self, **kwargs): ...
    def delete(self, **kwargs): ...

# REST framework 的 Mixin
class CreateModelMixin:
    def create(self, request, *args, **kwargs): ...
```

Django 和 Flask 都用 Mixin 提供"可插拔的能力"，和 sklearn 的思路一致——Mixin 是"能力注入"。

---

## 10. 常见问题和陷阱

### 10.1 陷阱 1：Mixin 存了状态

```python
# ❌ Mixin 存了状态
class BadMixin:
    def __init__(self):
        self.cache = {}  # Mixin 不应有 __init__

    def score(self, X, y):
        if X.tobytes() in self.cache:  # 用缓存
            return self.cache[X.tobytes()]
        ...
```

问题：Mixin 的 `__init__` 会和宿主类冲突，多继承时 `__init__` 顺序混乱。

### 10.2 陷阱 2：Mixin 之间方法冲突

```python
# ❌ 两个 Mixin 有同名方法
class MixinA:
    def foo(self): return "A"

class MixinB:
    def foo(self): return "B"

class C(MixinA, MixinB): pass

C().foo()  # "A"（MRO 决定），但可能不是你想要的
```

### 10.3 陷阱 3：Mixin 依赖的方法不存在

```python
class MyMixin:
    def score(self, X, y):
        return np.mean(self.predict(X) == y)  # 依赖 predict

class MyAlgo(BaseEstimator, MyMixin):
    def fit(self, X, y): return self
    # 忘了实现 predict

MyAlgo().fit(X, y).score(X, y)  # AttributeError: no 'predict'
```

### 10.4 陷阱 4：滥用 Mixin

```python
# ❌ 用 Mixin 表达 has-a
class EngineMixin:
    def start(self): ...

class Car(EngineMixin):  # Car 不是 Engine
    pass
```

Mixin 应该表达 is-a（能力），不是 has-a（组件）。

### 10.5 陷阱 5：Mixin 太多

```python
# ❌ 继承太多 Mixin，难以理解
class MyAlgo(
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    ClusterMixin,       # 既是分类器又是聚类器？
    RegressorMixin,     # 又是回归器？
    DensityMixin,
    OutlierMixin,
): ...
```

继承太多 Mixin 会让类的身份混乱，违反"一个类一个职责"。

---

## 11. 实际使用模式

### 11.1 模式 1：实现新算法

```python
# 实现一个新分类器
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, param=1.0):
        self.param = param

    def fit(self, X, y):
        # 学习逻辑
        return self

    def predict(self, X):
        # 预测逻辑
        return ...

    # score 由 ClassifierMixin 提供
```

### 11.2 模式 2：多重身份算法

```python
# 既是分类器又是转换器
class MyAlgo(BaseEstimator, ClassifierMixin, TransformerMixin):
    def fit(self, X, y): ...
    def predict(self, X): ...    # 分类
    def transform(self, X): ...  # 降维
    # score 由 ClassifierMixin 提供
    # fit_transform 由 TransformerMixin 提供
```

### 11.3 模式 3：自定义 Mixin

```python
# 自定义一个提供新能力的 Mixin
class FeatureImportanceMixin:
    """提供 feature_importances_ 属性的 Mixin"""
    def plot_importance(self):
        import matplotlib.pyplot as plt
        plt.barh(range(len(self.feature_importances_)),
                 self.feature_importances_)
        plt.show()

class MyAlgo(BaseEstimator, ClassifierMixin, FeatureImportanceMixin):
    def fit(self, X, y):
        ...
        self.feature_importances_ = ...  # fit 后设置
        return self
```

---

## 12. 思考题和练习

### 12.1 思考题

1. 如果 sklearn 用"上帝基类"而非 Mixin，加一个新身份（如"密度估计器"）要改多少代码？用 Mixin 呢？
2. Mixin 不存状态的好处是什么？如果允许存状态，会出什么问题？
3. 为什么 `ClassifierMixin` 和 `RegressorMixin` 不能同时继承？什么场景下会想同时继承？
4. Mixin 和装饰器都能"注入能力"，各自优缺点是什么？
5. Rust 的 Trait 和 Python 的 Mixin，哪个更适合大型项目？为什么？

### 12.2 练习

1. 实现一个 `DensityEstimatorMixin`，提供 `score_samples(X)` 返回对数密度。
2. 实现一个既是聚类器又是转换器的算法（如基于密度的聚类 + 距离转换）。
3. 用装饰器模拟 Mixin，比较两种方案的代码。

---

## 13. 深入：Mixin 的设计权衡

### 13.1 为什么 Mixin 不继承 BaseEstimator

一个常见疑问：为什么 `ClassifierMixin` 不继承 `BaseEstimator`？

```python
# ❌ 假设 Mixin 继承 BaseEstimator
class ClassifierMixin(BaseEstimator):
    def score(self, X, y): ...

class LogisticRegression(BaseEstimator, ClassifierMixin): ...
# MRO: LogisticRegression → BaseEstimator → ClassifierMixin → BaseEstimator → object
# 菱形继承！BaseEstimator 出现两次
```

如果 Mixin 继承 `BaseEstimator`，那 `LogisticRegression(BaseEstimator, ClassifierMixin)` 就有菱形继承——`BaseEstimator` 出现在两条路径上。

sklearn 的设计：Mixin **不继承任何类**（除了 `object`），保持 Mixin 之间的独立性：

```python
class ClassifierMixin:  # 不继承 BaseEstimator
    def score(self, X, y): ...

class LogisticRegression(BaseEstimator, ClassifierMixin): ...
# MRO: LogisticRegression → BaseEstimator → ClassifierMixin → object
# 无菱形
```

这是刻意的设计——Mixin 是"平行的能力"，不是"层次的关系"。

### 13.2 为什么不用 `__init_subclass__` 自动注入

Python 3.6+ 的 `__init_subclass__` 可以在子类创建时自动注入方法：

```python
class ClassifierMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 自动给子类注入 score 方法
        cls.score = lambda self, X, y: np.mean(self.predict(X) == y)
```

sklearn 没用这个，因为：

1. **显式优于隐式**：继承 `ClassifierMixin` 就知道获得了 `score`，不用看 `__init_subclass__` 的魔法
2. **可读性**：Mixin 的方法定义在类体里，IDE 能看到
3. **历史原因**：sklearn 早于 `__init_subclass__`（Python 3.6+）

### 13.3 Mixin 的 `super()` 调用

Mixin 的方法通常不调 `super()`，因为 Mixin 不参与继承链：

```python
class ClassifierMixin:
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
        # 没有 super().score(X, y)
```

但有些 Mixin 会调 `super()`，用于"装饰"父类方法：

```python
class VerboseMixin:
    def fit(self, X, y):
        print("Starting fit...")
        result = super().fit(X, y)  # 调父类的 fit
        print("Fit done.")
        return result
```

sklearn 的 Mixin 不用这种模式——它们提供新方法（`score`、`fit_transform`），而非装饰现有方法。

---

## 14. 实战案例：自定义 Mixin

让我们实现一个自定义 Mixin，把本讲知识点串起来：

```python
import numpy as np

class FeatureImportanceMixin:
    """提供特征重要性可视化能力的 Mixin。

    宿主类必须有 feature_importances_ 属性（fit 后设置）。
    """
    def plot_importance(self, feature_names=None, top_k=10):
        """绘制特征重要性条形图"""
        import matplotlib.pyplot as plt

        if not hasattr(self, 'feature_importances_'):
            raise AttributeError(
                f"{type(self).__name__} has no 'feature_importances_'. "
                f"Did you call fit()?"
            )

        importances = self.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        # 排序，取 top_k
        indices = np.argsort(importances)[::-1][:top_k]
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.show()

    def get_top_features(self, k=10):
        """返回 top-k 重要特征的索引"""
        if not hasattr(self, 'feature_importances_'):
            raise AttributeError("Call fit() first.")
        return np.argsort(self.feature_importances_)[::-1][:k]


class PersistMixin:
    """提供模型持久化能力的 Mixin。

    宿主类必须是 BaseEstimator（有 get_params）。
    """
    def save(self, path):
        """保存模型到文件"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """从文件加载模型"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# 组合使用
class MyRandomForest(
    BaseEstimator,
    ClassifierMixin,
    FeatureImportanceMixin,  # 可视化能力
    PersistMixin,            # 持久化能力
):
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        # ... 学习逻辑 ...
        self.feature_importances_ = np.random.rand(X.shape[1])  # 假装算的
        return self

    def predict(self, X):
        # ... 预测逻辑 ...
        return np.zeros(X.shape[0], dtype=int)


# 使用
rf = MyRandomForest(n_estimators=50)
rf.fit(X, y)
rf.plot_importance()           # 来自 FeatureImportanceMixin
rf.save('model.pkl')           # 来自 PersistMixin
loaded = MyRandomForest.load('model.pkl')  # 来自 PersistMixin
print(rf.score(X, y))          # 来自 ClassifierMixin
```

这个例子展示了 Mixin 的"能力注入"——`MyRandomForest` 通过继承多个 Mixin，获得了可视化、持久化、分类评分等多种能力，而不用自己实现这些方法。

---

## 15. Mixin 在 sklearn 源码中的实际样子

让我们看看 sklearn 源码中 Mixin 的真实样子（简化版）：

```python
# sklearn/base.py（简化）

class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn."""
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _more_tags(self):
        return {'requires_y': True}


class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='uniform_average')


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""
    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it."""
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)


class ClusterMixin:
    """Mixin class for all cluster estimators in scikit-learn."""
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **fit_params):
        """Compute cluster centers and predict cluster index for each sample."""
        return self.fit(X, **fit_params).labels_
```

注意几个细节：

1. **`_estimator_type` 是类属性**：不是实例属性，所有实例共享
2. **`score` 调用 `self.predict`**：依赖宿主类提供
3. **`_more_tags` 方法**：声明估计器的"标签"（如 `requires_y`），用于测试套件
4. **`fit_transform` 处理 `y=None`**：无监督转换器不接收 `y`

`_more_tags` 是较新的机制，让估计器声明自己的"能力"和"需求"，`check_estimator` 用它决定跑哪些测试。

---

## 16. Mixin 的替代方案探讨

### 16.1 方案 A：用 Protocol（PEP 544）

Python 3.8+ 的 `Protocol` 提供结构子类型：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Classifier(Protocol):
    def fit(self, X, y) -> 'Classifier': ...
    def predict(self, X): ...
    def score(self, X, y) -> float: ...
```

但 Protocol 不能提供默认实现（只能声明接口），所以不能替代 Mixin。

### 16.2 方案 B：用 `__init_subclass__` 注入

```python
class ClassifierCapability:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'score'):
            cls.score = lambda self, X, y: np.mean(self.predict(X) == y)
```

能工作，但隐式魔法降低可读性，sklearn 不用。

### 16.3 方案 C：用组合 + 委托

```python
class ClassifierCapability:
    def score(self, estimator, X, y):
        return np.mean(estimator.predict(X) == y)

class LogisticRegression(BaseEstimator):
    def __init__(self):
        self._classifier_cap = ClassifierCapability()

    def score(self, X, y):
        return self._classifier_cap.score(self, X, y)
```

能工作，但要写委托代码，不如 Mixin 简洁。

### 16.4 为什么 Mixin 胜出

Mixin 在 sklearn 场景下胜出，因为：

1. **简洁**：继承一个 Mixin 就获得方法，不用写委托
2. **显式**：方法定义在 Mixin 类体里，IDE 可见
3. **组合自然**：多继承多个 Mixin 即可
4. **历史成熟**：Python 多继承 + MRO 经过 decades 验证

---

## 17. 思考题深入

### 17.1 如果 sklearn 重新设计

如果今天重新设计 sklearn，会选 Mixin 还是其他方案？

考虑因素：

- **Type Hints**：现代 Python 重视类型，Protocol 可能更合适
- **性能**：Mixin 的方法查找走 MRO，有微小开销
- **静态检查**：mypy 对 Mixin 的支持不如 Protocol

但 Mixin 的简洁性和成熟度仍然占优。sklearn 即使重新设计，大概率仍选 Mixin。

### 17.2 Mixin 与函数式编程

Mixin 的"能力注入"类似函数式编程的"高阶函数"：

```python
# 函数式：高阶函数注入能力
def with_score(predict_fn):
    def score(X, y):
        return np.mean(predict_fn(X) == y)
    return score

# OOP：Mixin 注入能力
class ClassifierMixin:
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

两者本质相同——"能力的组合"。Mixin 是 OOP 里的高阶函数。

---

## 18. 小结

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| 基类职责 | `BaseEstimator` 只管参数 | 职责单一 |
| 身份表达 | Mixin 多继承 | 一个类可有多重身份 |
| Mixin 状态 | 不存状态（无 `__init__`） | 避免 `__init__` 冲突 |
| 方法冲突 | 设计上避免 | MRO 清晰 |
| 类型标识 | `_estimator_type` 字符串 | 鸭子类型友好 |
| 协议实现 | Mixin 调用 `self.predict` 等 | 软依赖、运行期检查 |

**核心洞察**：Mixin 是"能力注入"——`ClassifierMixin` 给宿主类注入"分类器的能力"（`score` 方法），而不关心宿主类具体怎么 `predict`。这种**能力与实现分离**的设计，让 sklearn 的类层次既灵活又清晰。

### 18.1 本讲要点回顾

1. **Mixin 解决"上帝基类"问题**：职责分离、开闭原则、多重身份。
2. **Mixin 不存状态**：避开多继承 `__init__` 雷区。
3. **Mixin 依赖宿主类方法**：协议而非实现，软依赖。
4. **Mixin 可组合**：LDA 同时是分类器和转换器。
5. **MRO 清晰**：设计上避免冲突，C3 线性化保证确定顺序。
6. **`_estimator_type` 身份标识**：鸭子类型友好，元估计器用。
7. **能力用继承，组件用组合**：is-a vs has-a 的正确表达。

---

## 上一讲 / 下一讲

[← 第一讲：统一 API 契约](01-unified-api.md) ｜  [第三讲：参数管理机制 →](03-parameter-management.md）
