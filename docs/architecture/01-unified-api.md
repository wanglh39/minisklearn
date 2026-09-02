# 第一讲：统一 API 契约

> **核心问题**：为什么 sklearn 所有算法都用 `fit` / `predict` / `transform`，而不是 `train` / `infer` / `convert`？这个约定从哪来？为什么它如此重要？

---

## 1. 一个观察：学一个就会用上百个

如果你用过 sklearn，一定有过这种体验：学会用 `LogisticRegression` 之后，换成 `RandomForestClassifier` 几乎零成本——只要改个类名，`fit` 和 `predict` 的用法完全一样。

这不是巧合，而是 sklearn 最核心的设计决策：**统一 API 契约**。

```python
# 不管用什么算法，套路完全一样
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

for Model in [LogisticRegression, RandomForestClassifier, SVC]:
    clf = Model()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
```

这种"一套接口走天下"的设计，降低了学习成本、促进了算法组合、使得自动化测试成为可能。它是 sklearn 能成为生态基石的根本原因。

### 1.1 这种体验背后意味着什么

让我们停下来认真想一想，上面那段循环代码为什么能成立。在大多数软件库里，不同算法往往有完全不同的调用方式：

- 有的算法要求你先 `build` 再 `compile` 再 `train`
- 有的算法要求你传入一个 `config` 对象
- 有的算法要求你实现一个 `Model` 子类并 override 若干方法
- 有的算法要求你调用 `model.run(X, y, mode="train")`

而 sklearn 把这一切都收敛到了三个动词：`fit`、`predict`、`transform`。这意味着：

1. **用户的心智负担被压到了最低**：你只需要记住三个动词，就能驱动上百个算法。
2. **算法之间可以无缝替换**：你可以把 `LogisticRegression` 换成 `RandomForestClassifier`，而下游代码一行都不用改。
3. **算法可以自由组合**：因为接口统一，`Pipeline` 才能串联任意算法；`GridSearchCV` 才能包装任意估计器。
4. **自动化测试成为可能**：sklearn 有一个统一的 `check_estimator` 测试套件，对所有算法跑同一组测试，因为接口统一。

### 1.2 一个对比：如果没有统一 API

假设 sklearn 没有统一 API，每个算法各自为政，那么用户代码可能长这样：

```python
# 假设的"反 sklearn"世界
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 每个算法调用方式都不同
lr = LogisticRegression()
lr.train(X_train, y_train, epochs=100)
y_pred_lr = lr.infer(X_test)

rf = RandomForestClassifier()
rf.fit_forest(X_train, y_train, n_trees=100)
y_pred_rf = rf.predict_proba(X_test).argmax(axis=1)

svm = SVC()
svm.optimize(X_train, y_train, kernel='rbf')
y_pred_svm = svm.classify(X_test)
```

这样的世界：

- 用户每换一个算法，就要重学一套 API
- 想把算法串成流水线？每个算法的调用方式不同，写不出通用的 `Pipeline`
- 想做网格搜索？每个算法的参数设置方式不同，写不出通用的 `GridSearchCV`
- 想写测试？每个算法的接口不同，写不出通用的 `check_estimator`

这就是为什么统一 API 是 sklearn 的"立身之本"——它不是锦上添花，而是整个生态能成立的前提。

### 1.3 历史背景：统一 API 从哪来

sklearn 的统一 API 并非凭空发明，它有深厚的历史渊源：

1. **统计学传统**：R 语言的 `lm()`、`glm()` 等函数都遵循 `fit` → `predict` → `summary` 的模式。sklearn 的 `fit` / `predict` 直接继承自这一传统。
2. **SciPy 生态**：sklearn 诞生于 SciPy 生态，与 NumPy、SciPy 风格一致——简洁、统一、面向科学计算。
3. **2007 年的初始设计**：sklearn 由 David Cournapeau 在 2007 年作为 Google Summer of Code 项目启动，当时就确立了"统一接口"的目标。
4. **SLEP（Scikit-Learn Enhancement Proposals）**：sklearn 的 API 约定通过 SLEP 正式记录，类似 Python 的 PEP，确保约定被严格遵守。

理解这段历史有助于理解为什么 sklearn 选了 `fit` 而不是 `train`——它从一开始就把自己定位在"统计学传统"而非"神经网络传统"。

### 1.4 统一 API 的"网络效应"

统一 API 的价值随算法数量**非线性增长**：

- 有 2 个算法统一：省 1 份学习成本
- 有 10 个算法统一：省 9 份学习成本，且能 10×10 = 100 种组合
- 有 100 个算法统一：省 99 份学习成本，且能 100×100 = 10000 种组合

sklearn 现在有上百个算法，统一 API 带来的价值是巨大的。这就是**网络效应**——参与者越多，每个参与者获得的价值越大。

这也解释了为什么新算法想进 sklearn 主仓库，必须遵守 API 契约——一个不遵守契约的算法，会破坏整个网络效应。

### 1.5 思考题

1. 你能想到其他"统一 API 带来网络效应"的软件库吗？（提示：Python 的文件对象协议、NumPy 的数组协议、Python 的迭代器协议）
2. 如果你要设计一个图像处理库，会怎么设计统一 API？
3. 统一 API 有没有缺点？什么场景下统一 API 反而是负担？

---

## 2. 契约的内容：三类核心方法

sklearn 的 API 契约围绕三类方法展开：

### 2.1 估计器（Estimator）：`fit`

**所有**算法的起点。`fit` 从数据中学习参数，返回 `self`（支持链式调用）。

```python
estimator.fit(X, y=None)
```

约定：

- `fit` 是**唯一**能从数据中学习的方法
- `fit` 返回 `self`（不是 `void`），方便链式写 `clf.fit(X, y).predict(X_test)`
- `fit` 后学到的参数以**下划线结尾**命名：`self.coef_`、`self.intercept_`、`self.classes_`
- 下划线结尾是**约定**：表示"这是 fit 学出来的，不是用户传入的超参数"

为什么用下划线区分？

```python
# 超参数：用户传入，fit 前就有
clf = LogisticRegression(C=1.0)   # clf.C = 1.0

# 学习参数：fit 后才有
clf.fit(X, y)                      # clf.coef_ 才被创建
```

这种命名约定让你一眼就能区分"配置"和"状态"，也方便 `clone` 判断哪些属性该丢弃。

#### 2.1.1 `fit` 的输入约定

`fit` 的输入有严格约定：

| 参数 | 类型 | shape | 说明 |
|------|------|-------|------|
| `X` | array-like | `(n_samples, n_features)` | 特征矩阵，二维 |
| `y` | array-like | `(n_samples,)` 或 `None` | 标签，一维；无监督算法为 `None` |

注意 `X` 必须是**二维**的，即使只有一个特征：

```python
# ✅ 正确：单特征也要二维
X = np.array([[1], [2], [3]])  # shape (3, 1)
clf.fit(X, y)

# ❌ 错误：传了一维
X = np.array([1, 2, 3])  # shape (3,)
clf.fit(X, y)  # 会报错或被警告
```

这是初学者最常犯的错误之一。sklearn 之所以要求二维，是因为要统一处理单特征和多特征的情况——如果允许一维，内部就要到处 `if X.ndim == 1: X = X.reshape(-1, 1)`，既丑陋又容易出错。

#### 2.1.2 `fit` 返回 `self` 的妙用

`fit` 返回 `self` 看似小细节，实则支持了链式调用：

```python
# 链式调用：一行搞定
y_pred = LogisticRegression(C=1.0).fit(X_train, y_train).predict(X_test)

# 等价于
clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

链式调用在交互式环境（如 Jupyter）中特别方便。但注意，**生产代码中不建议链式**，因为：

1. 可读性差：一行做了三件事
2. 调试困难：出错时不知道是 `fit` 还是 `predict` 出问题
3. 无法复用：`clf` 没保存下来，后续无法用

#### 2.1.3 下划线结尾的深层含义

`coef_`、`intercept_`、`classes_` 这些下划线结尾的属性有严格语义：

- **只在 `fit` 后才存在**：`fit` 前访问会报错（通过 `check_is_fitted` 检查）
- **是 `fit` 从数据中学出来的**：不是用户传入的
- **`clone` 时会被丢弃**：`clone` 只复制超参数，不复制学出的参数

```python
clf = LogisticRegression(C=1.0)
# clf.coef_  # ❌ AttributeError: 还没 fit

clf.fit(X, y)
print(clf.coef_)      # ✅ fit 后才能访问
print(clf.classes_)   # ✅

new_clf = clone(clf)
# new_clf.coef_  # ❌ AttributeError: clone 出的是干净的
print(new_clf.C)      # ✅ 1.0，超参数保留
```

这个约定让"配置"和"状态"在命名上就泾渭分明，是 sklearn 最受好评的设计之一。

#### 2.1.4 `fit` 的幂等性

`fit` 不是幂等的——多次 `fit` 会覆盖之前的学习结果：

```python
clf = LogisticRegression()
clf.fit(X1, y1)
coef1 = clf.coef_.copy()

clf.fit(X2, y2)  # 覆盖了第一次 fit 的结果
coef2 = clf.coef_

assert coef1 != coef2  # coef_ 已经变了
```

这意味着 `fit` 是有副作用的——它会改变对象状态。这是为什么 `predict` 必须是无副作用的（见下文）：如果 `fit` 和 `predict` 都有副作用，对象状态就不可预测了。

### 2.2 预测器（Predictor）：`predict`

分类和回归算法的核心输出方法。

```python
y_pred = estimator.predict(X)
```

约定：

- 输入 `X` 的 `shape` 必须是 `(n_samples, n_features)`
- 输出 `y_pred` 的 `shape` 是 `(n_samples,)` 或 `(n_samples, n_outputs)`
- `predict` **不能修改**估计器状态（纯查询方法）

#### 2.2.1 `predict` 的纯查询语义

`predict` 是**纯查询**方法——不修改对象任何状态。这意味着：

```python
clf = LogisticRegression().fit(X, y)

# 多次 predict，结果相同
y_pred1 = clf.predict(X_test)
y_pred2 = clf.predict(X_test)
assert (y_pred1 == y_pred2).all()

# predict 不会改变 clf 的状态
assert clf.coef_ is clf.coef_  # coef_ 没变
```

为什么这么重要？因为如果 `predict` 有副作用，下列代码就会出 bug：

```python
# 假设 predict 有副作用（错误示范）
for x in X_test:
    y_pred = clf.predict(x)  # 如果 predict 改了 clf 状态，每次结果都受影响
```

纯查询语义保证了 `predict` 可以安全地并行调用、重复调用、乱序调用。

#### 2.2.2 `predict` vs `predict_proba` vs `decision_function`

分类器通常有三个相关方法：

| 方法 | 返回 | shape | 用途 |
|------|------|-------|------|
| `predict` | 类标签 | `(n_samples,)` | 直接预测类别 |
| `predict_proba` | 概率 | `(n_samples, n_classes)` | 预测属于每类的概率 |
| `decision_function` | 决策值 | `(n_samples,)` 或 `(n_samples, n_classes-1)` | 预测的"置信度" |

```python
clf = LogisticRegression().fit(X, y)

y_pred = clf.predict(X_test)           # [0, 1, 1, 0, ...]
y_proba = clf.predict_proba(X_test)    # [[0.9, 0.1], [0.3, 0.7], ...]
y_score = clf.decision_function(X_test)  # [2.3, -0.5, ...]
```

注意 `predict` 通常是 `predict_proba` 的 argmax：

```python
# predict 内部通常这么实现
y_pred = clf.predict_proba(X_test).argmax(axis=1)
```

但不是所有分类器都有 `predict_proba`（如 SVM 默认没有），也不是所有都有 `decision_function`（如 RandomForest 没有）。这是为什么 sklearn 用鸭子类型而非抽象基类——不同算法提供不同子集的方法。

#### 2.2.3 `predict` 的输入校验

`predict` 内部通常会校验输入：

```python
def predict(self, X):
    check_is_fitted(self)  # 1. 检查是否 fit 过
    X = check_array(X)     # 2. 校验 X 的类型和 shape
    # 3. 校验 n_features 是否和 fit 时一致
    if X.shape[1] != self.n_features_in_:
        raise ValueError(...)
    return ...  # 4. 真正的预测
```

`check_is_fitted` 是一个常见错误来源：

```python
clf = LogisticRegression()
clf.predict(X_test)
# NotFittedError: This LogisticRegression instance is not fitted yet.
```

这个错误信息很清晰，是 sklearn 用心设计的——清晰错误信息是统一 API 的一部分。

### 2.3 转换器（Transformer）：`transform` / `fit_transform`

预处理和降维算法的核心方法。

```python
X_new = transformer.transform(X)
X_new = transformer.fit_transform(X)  # = fit(X).transform(X)
```

约定：

- `transform` 输出 `X_new`，`shape` 可以是 `(n_samples, n_features_new)`，特征数可变
- `fit_transform` 默认等价于 `fit(X).transform(X)`，但允许子类覆盖以优化性能

#### 2.3.1 `transform` vs `predict` 的区别

| 特性 | `transform` | `predict` |
|------|-------------|-----------|
| 输出 | `X_new`（特征） | `y`（标签） |
| 输出 shape | `(n_samples, n_features_new)` | `(n_samples,)` |
| 语义 | 把数据转换到新空间 | 预测标签 |
| 典型算法 | PCA、StandardScaler | LogisticRegression、SVM |

`transform` 输出的是"新的特征表示"，`predict` 输出的是"预测的标签"。这是预处理和预测的本质区别。

#### 2.3.2 `fit_transform` 的优化机会

`fit_transform` 默认是 `fit(X).transform(X)`，但有些算法可以优化：

```python
# 默认实现（在 TransformerMixin 中）
def fit_transform(self, X, y=None):
    return self.fit(X, y).transform(X)

# PCA 的优化实现
def fit_transform(self, X, y=None):
    # PCA 在 fit 时已经计算了转换结果，直接返回，不用再 transform 一次
    U, S, V = self._fit(X)  # SVD 分解
    return U * S  # 直接返回，省了一次矩阵乘法
```

这种"默认实现 + 允许覆盖"的模式是 sklearn 的常见手法——既保证一致性，又留出优化空间。

#### 2.3.3 `inverse_transform`

有些转换器提供 `inverse_transform`，把数据转回去：

```python
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_original = scaler.inverse_transform(X_scaled)
assert np.allclose(X, X_original)  # 还原
```

`inverse_transform` 在数据可视化、可解释性分析中很有用。

### 2.4 三类方法的组合关系

三类方法不是孤立的，它们组合出更复杂的行为：

```python
# 转换器 + 预测器 = Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),  # 转换器
    ('clf', LogisticRegression()),  # 预测器
])
pipe.fit(X, y)         # 依次 fit_transform + fit
y_pred = pipe.predict(X)  # 依次 transform + predict

# 估计器 + 估计器 = 集成
rf = RandomForestClassifier(n_estimators=100)  # 100 个决策树
rf.fit(X, y)
y_pred = rf.predict(X)
```

这种组合性是统一 API 的最大红利——下一讲讲 Mixin，第四讲讲元估计器，都会回到这一点。

---

## 3. 为什么是 `fit` / `predict` 而不是 `train` / `infer`？

这看似只是命名问题，但命名反映思维模型。

### 3.1 `fit` vs `train`

`fit`（拟合）来自统计学传统——"用模型去**拟合**数据分布"。

`train`（训练）来自神经网络传统——"通过迭代**训练**模型参数"。

sklearn 选 `fit` 的原因：

1. **统计视角更普适**：KMeans 不是"训练"出来的，是"拟合"数据分布；PCA 是"拟合"主方向。`train` 暗示迭代优化，但很多算法（如 KNN）根本不迭代。
2. **与统计学术语一致**：`fit` / `predict` / `transform` 是统计学的标准词汇，降低领域迁移成本。
3. **简短**：3 个字母比 5 个短，写起来快（这真的是一个考虑因素）。

#### 3.1.1 "拟合" vs "训练"的思维差异

"拟合"和"训练"看似同义，实则反映两种不同的思维模型：

| 维度 | 拟合（fit） | 训练（train） |
|------|-------------|----------------|
| 来源 | 统计学 | 神经网络/深度学习 |
| 暗示 | 一次性求解 | 迭代优化 |
| 数据 | 全量数据 | mini-batch |
| 关注 | 参数估计 | 损失下降 |
| 典型 | 线性回归、PCA | CNN、Transformer |

sklearn 选 `fit` 是因为它的算法大多来自统计学传统：

- 线性回归：最小二乘求解，一次性
- PCA：SVD 分解，一次性
- KMeans：虽然是迭代，但本质是"拟合聚类中心"
- KNN：根本不"训练"，只是存数据

而深度学习框架选 `train` 是因为它们的算法本质是迭代优化：

- CNN：反向传播，几万次迭代
- Transformer：梯度下降，几万次迭代

所以命名反映了算法的本质——sklearn 的算法大多"拟合"即可，不需要"训练"。

#### 3.1.2 命名的传染效应

好的命名会"传染"——被其他库借鉴。sklearn 的 `fit` / `predict` 就被广泛借鉴：

- **Keras**：`model.fit()` / `model.predict()` —— 直接照搬 sklearn
- **TensorFlow 2.x**：`model.fit()` —— 跟随 Keras
- **PyTorch Lightning**：`trainer.fit()` —— 借鉴 sklearn
- **H2O.ai**：`model.fit()` —— 借鉴 sklearn

这说明 sklearn 的命名选择是成功的——它成了事实标准。

### 3.2 `predict` vs `infer`

深度学习框架常用 `infer` 或 `forward`，sklearn 选 `predict`（预测）。

原因：

1. `infer` 在逻辑学里是"推理"，语义太宽泛（推理可以指很多事）
2. `predict` 明确就是"根据输入预测输出"，语义无歧义
3. 在分类和回归场景中，"预测"是最自然的词

#### 3.2.1 `forward` vs `predict` vs `infer`

| 框架 | 方法 | 语义 | 来源 |
|------|------|------|------|
| sklearn | `predict` | 预测标签 | 统计学 |
| PyTorch | `forward` | 前向计算 | 神经网络（前向传播） |
| TensorFlow | `predict` | 预测 | 借鉴 sklearn |
| ONNX | `infer` | 推理 | 逻辑学/编译器 |

`forward` 来自神经网络的"前向传播"——它强调的是"计算图的前向方向"，而非"预测"。在 PyTorch 里，`forward` 可以是预测，也可以是特征提取，也可以是任意计算——语义比 `predict` 宽泛。

sklearn 选 `predict` 是因为它的语义更明确——就是"预测标签"，不含糊。

### 3.3 `transform` 的来源

`transform`（转换）来自数据处理传统——"把数据从一种形式转换成另一种形式"。

这个词在多个领域都有用：

- 信号处理：傅里叶变换（Fourier Transform）
- 图像处理：仿射变换（Affine Transform）
- 数据库：ETL（Extract Transform Load）

sklearn 的 `transform` 沿用了这一传统——它表示"数据的形态变换"，而非"预测"。

---

## 4. 鸭子类型 vs 抽象基类

sklearn 契约的 enforcement（强制）方式是一个重要设计选择：**鸭子类型**，而非抽象基类。

### 4.1 什么是鸭子类型？

> 如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子。

sklearn 不检查"你是不是 `Classifier` 的子类"，只检查"你有没有 `fit` 和 `predict` 方法"。

```python
# sklearn 不要求你继承任何基类
class MyAlgo:  # 注意：没有继承 BaseEstimator
    def fit(self, X, y):
        return self
    def predict(self, X):
        return ...

# 但它可以被 Pipeline 接受！
Pipeline([('my_algo', MyAlgo())])  # 能用
```

#### 4.1.1 鸭子类型的实际例子

鸭子类型让 sklearn 极具包容性。你可以用任何符合契约的对象：

```python
# 用普通 dict 模拟一个"转换器"
class DictScaler:
    """一个不继承任何 sklearn 基类的转换器"""
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# 能被 Pipeline 接受
pipe = Pipeline([
    ('scaler', DictScaler()),  # 鸭子类型：有 fit/transform 就行
    ('clf', LogisticRegression()),
])
pipe.fit(X, y)
y_pred = pipe.predict(X)
```

这种包容性是 sklearn 生态繁荣的重要原因——第三方库实现的算法，只要符合契约，就能无缝接入 sklearn。

### 4.2 为什么不用抽象基类（ABC）？

```python
# 假设用抽象基类
from abc import ABC, abstractmethod

class Predictor(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...
```

问题在于：

1. **强制继承是负担**：用户想快速包装一个自定义算法，还得继承一堆基类
2. **多继承冲突**：一个类同时是 `Predictor` 和 `Transformer`，ABC 的 MRO 容易混乱
3. **动态语言的优势**：Python 的鸭子类型本就是优势，用 ABC 反而限制了灵活性
4. **测试弥补**：sklearn 用 `check_estimator` 测试套件来检查契约，不需要编译期强制

#### 4.2.1 ABC 的具体问题

让我们具体看看 ABC 会带来什么问题：

```python
from abc import ABC, abstractmethod

class Predictor(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...

class Transformer(ABC):
    @abstractmethod
    def fit(self, X, y=None): ...
    @abstractmethod
    def transform(self, X): ...

# LDA 既是 Predictor 又是 Transformer
class LDA(Predictor, Transformer):  # 多继承 ABC
    ...
```

问题：

1. **`fit` 签名冲突**：`Predictor.fit(self, X, y)` 和 `Transformer.fit(self, X, y=None)` 签名不同，ABC 会报错
2. **MRO 复杂**：两个 ABC 都继承 `ABC`，菱形继承
3. **强制实现**：用户想写个简单包装器，被迫实现所有抽象方法

鸭子类型避开了所有这些问题——不检查继承关系，只看方法是否存在。

#### 4.2.2 ABC 的另一个问题：抑制快速原型

```python
# 鸭子类型：快速原型，先写个 stub
class MyAlgo:
    def fit(self, X, y):
        return self
    def predict(self, X):
        raise NotImplementedError("还没写")

# 可以先放进 Pipeline 测试其他部分
pipe = Pipeline([('algo', MyAlgo()), ...])

# ABC：连实例化都不行
class MyAlgo(Predictor):
    def fit(self, X, y):
        return self
    # 没实现 predict，实例化就报错
MyAlgo()  # TypeError: Can't instantiate abstract class
```

鸭子类型支持渐进式开发——先写 stub，再慢慢实现。ABC 要求一次性实现所有方法。

### 4.3 鸭子类型的代价

鸭子类型的代价是**错误后置**：你传了一个没有 `fit` 方法的对象给 `Pipeline`，要等到运行 `pipeline.fit()` 时才报错，而不是传进去时就报错。

sklearn 的应对：

- 在关键入口做 `hasattr` 检查，给出清晰错误信息
- 用 `check_estimator` 测试套件在开发期发现问题

#### 4.3.1 错误后置的具体例子

```python
# 鸭子类型：错误后置
class NotAnEstimator:
    pass

pipe = Pipeline([('bad', NotAnEstimator())])  # 不报错
pipe.fit(X, y)  # 这里才报错：AttributeError: 'NotAnEstimator' object has no attribute 'fit'

# 如果用 ABC：错误前置
pipe = Pipeline([('bad', NotAnEstimator())])  # 这里就报错
```

错误后置的代价是：用户可能传了一个错误对象，但到运行时才发现，调试困难。

sklearn 的缓解措施：

```python
# sklearn 在关键入口做 hasattr 检查
def _validate_steps(steps):
    for name, step in steps:
        if not hasattr(step, 'fit'):
            raise TypeError(
                f"Step '{name}' ({type(step).__name__}) "
                f"does not have a 'fit' method. "
                f"All steps should be estimators."
            )
```

这样虽然还是运行时检查，但能在 `Pipeline` 构造时就发现错误，不用等到 `fit`。

#### 4.3.2 `check_estimator`：测试代替类型检查

sklearn 用 `check_estimator` 测试套件来弥补鸭子类型的不足：

```python
from sklearn.utils.estimator_checks import check_estimator

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    ...

# 跑 100+ 个测试，检查是否符合 sklearn 契约
check_estimator(MyLogisticRegression())
```

`check_estimator` 会检查：

- `fit` 返回 `self`
- `predict` 不修改状态
- `fit` 后属性以 `_` 结尾
- `get_params` / `set_params` 正确
- `clone` 行为正确
- 输入校验正确
- ...（100+ 项）

这是"测试代替类型检查"的典范——动态语言用测试弥补静态检查的不足。

---

## 5. 契约的层次：从 Estimator 到具体算法

sklearn 的 API 契约是分层的：

```
Estimator（估计器）
  ├── 有 fit 方法
  │
  ├── Predictor（预测器）
  │     ├── 有 predict 方法
  │     ├── ClassifierMixin → score 返回 accuracy
  │     └── RegressorMixin  → score 返回 R²
  │
  ├── Transformer（转换器）
  │     ├── 有 transform 方法
  │     └── TransformerMixin → fit_transform = fit + transform
  │
  └── ClusterMixin（聚类器）
        └── fit_predict = fit + labels_
```

注意：这个层次**不是继承关系**，而是**能力组合**。一个类可以同时是 Predictor 和 Transformer：

```python
# LDA（线性判别分析）既能降维（transform）又能分类（predict）
class LinearDiscriminantAnalysis(
    BaseEstimator,
    ClassifierMixin,    # 提供 score = accuracy
    TransformerMixin,   # 提供 fit_transform
):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def transform(self, X): ...
```

这就是 Mixin 多继承的威力——下一讲详谈。

### 5.1 分层的好处

分层让"身份"可以组合：

| 算法 | 身份 | 提供的方法 |
|------|------|-----------|
| LogisticRegression | Classifier | fit, predict, score(accuracy) |
| LinearRegression | Regressor | fit, predict, score(R²) |
| StandardScaler | Transformer | fit, transform, fit_transform |
| KMeans | Clusterer | fit, fit_predict, labels_ |
| PCA | Transformer | fit, transform, fit_transform |
| LDA | Classifier + Transformer | fit, predict, transform, score, fit_transform |

如果用单一基类，LDA 就要特殊处理——"我是分类器还是转换器？"。分层让 LDA 自然地拥有两种身份，无需特殊处理。

### 5.2 分层 vs 继承层次

注意 sklearn 的分层**不是严格的继承层次**：

- `ClassifierMixin` 不继承 `Predictor`
- `TransformerMixin` 不继承 `Estimator`
- 它们都是平行的 Mixin，按需组合

这和传统 OOP 的"继承层次"不同——sklearn 用的是"能力组合"而非"分类学"。下一讲会详细讲这个设计。

### 5.3 每一层的职责

| 层 | 职责 | 不做什么 |
|----|------|---------|
| `BaseEstimator` | 参数管理、clone、repr | 不定义 fit/predict |
| `ClassifierMixin` | score = accuracy | 不定义 predict |
| `RegressorMixin` | score = R² | 不定义 predict |
| `TransformerMixin` | fit_transform | 不定义 transform |
| `ClusterMixin` | fit_predict | 不定义 fit |

每一层只做一件事，不越权。这种"职责单一"让组合变得清晰——你想要什么能力，就加什么 Mixin。

---

## 6. 与其他框架的对比

| 框架 | 核心契约 | 设计哲学 |
|------|---------|---------|
| **sklearn** | `fit` / `predict` / `transform` | 统一接口、鸭子类型、无状态估计器 |
| **PyTorch** | `forward` / `backward` | 计算图、自动微分、面向研究 |
| **TensorFlow/Keras** | `fit` / `predict`（借鉴 sklearn） / `evaluate` | 静态图、面向生产 |
| **HuggingFace** | `train` / `predict` / `generate` | 面向预训练模型、流水线化 |

有趣的是，Keras 的 `fit` / `predict` 正是借鉴了 sklearn 的设计——这证明了这套契约的成功。

sklearn 契约的独特之处在于**最小化**：只有 `fit` 是必须的，`predict` / `transform` 按需提供。而 PyTorch 的 `forward` 是必须的，`backward` 框架自动生成。sklearn 的极简契约让它能覆盖从预处理到聚类到降维的广泛场景。

### 6.1 sklearn vs PyTorch：哲学对比

| 维度 | sklearn | PyTorch |
|------|---------|---------|
| 核心抽象 | 估计器 | 计算图 |
| 状态 | 参数 + 学出的属性 | 参数 + 梯度 + 计算图 |
| `fit`/`forward` | 一次性求解 | 前向计算（一次） |
| `predict`/`backward` | 纯查询 | 反向传播（建图） |
| 输入 | NumPy array | Tensor |
| 设备 | CPU（部分支持 GPU） | CPU/GPU/TPU |
| 求导 | 不需要 | 自动微分 |

sklearn 和 PyTorch 服务于不同场景：

- sklearn：传统机器学习，参数量小，CPU 即可
- PyTorch：深度学习，参数量大，需要 GPU

它们的 API 差异反映了这种场景差异——sklearn 的 `fit` 是"一次性求解"，PyTorch 的 `forward` + `backward` 是"迭代优化"。

### 6.2 sklearn vs Keras：谁借鉴了谁

Keras 的 `model.fit()` / `model.predict()` 明显借鉴了 sklearn。但 Keras 的 `fit` 和 sklearn 的 `fit` 有本质区别：

```python
# sklearn 的 fit：一次性求解
clf = LogisticRegression()
clf.fit(X, y)  # 内部用求解器一次性解出 coef_

# Keras 的 fit：迭代训练
model = keras.Sequential([...])
model.fit(X, y, epochs=100, batch_size=32)  # 迭代 100 轮
```

虽然都叫 `fit`，但 Keras 的 `fit` 有 `epochs`、`batch_size`、`callbacks` 等迭代相关参数——因为深度学习本质是迭代优化。sklearn 的 `fit` 没有这些参数——因为传统机器学习大多一次性求解。

这说明：**API 借鉴是表面的，底层哲学差异才是本质**。

### 6.3 sklearn vs HuggingFace：传统 ML vs 预训练

HuggingFace 的 Transformers 库用 `train` / `predict` / `generate`：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.train()  # 切换到训练模式
model.generate(text)  # 生成文本
```

HuggingFace 选 `train` 而非 `fit`，因为它面向的是预训练模型——这些模型本质是"训练"出来的（预训练 + 微调）。`generate` 是 HuggingFace 独有的，因为生成模型输出的是序列而非标签。

这再次说明：**命名反映哲学**。

### 6.4 为什么 sklearn 的契约最"瘦"

sklearn 的契约只有三个方法，却能覆盖上百个算法。为什么这么"瘦"的契约能这么强大？

因为 sklearn 把"算法差异"推到了**方法内部**，而非**接口表面**：

```python
# 接口表面：所有算法一样
clf.fit(X, y)
clf.predict(X)

# 方法内部：每个算法不同
# LogisticRegression.fit: 解凸优化
# RandomForest.fit: 建树
# KMeans.fit: 迭代更新聚类中心
# PCA.fit: SVD 分解
```

统一接口 + 内部差异 = 既统一又灵活。这是好的抽象的本质——**隐藏差异，暴露统一**。

---

## 7. 自己实现：BaseEstimator 的雏形

理解了契约，我们来看 `BaseEstimator` 如何支撑它。注意：`BaseEstimator` **不定义** `fit` / `predict`——这些由子类实现。`BaseEstimator` 只提供支撑契约的基础设施。

```python
class BaseEstimator:
    """所有估计器的基类。

    不实现 fit / predict / transform —— 这些由子类或 Mixin 提供。
    只提供：参数管理（get_params/set_params）、克隆（clone）、repr。
    """
    def get_params(self, deep=True): ...
    def set_params(self, **params): ...
    def __repr__(self): ...
```

一个具体的算法实现：

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100):
        self.C = C               # 原样存储，不做任何事
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)   # 校验输入
        # ... 学习逻辑 ...
        self.coef_ = ...         # 下划线结尾：fit 学出来的
        self.classes_ = ...
        return self              # 返回 self：链式调用

    def predict(self, X):
        check_is_fitted(self)    # 检查是否 fit 过
        X = check_array(X)       # 校验输入
        return ...               # 纯查询，不改状态
```

注意几个细节：

1. `__init__` 只存参数，不做计算（下一讲详谈为什么）
2. `fit` 返回 `self`
3. `fit` 创建的属性以 `_` 结尾
4. `predict` 先检查 `is_fitted`，再校验输入，最后纯查询

### 7.1 一个完整的简单实现

让我们从头实现一个符合契约的分类器，感受契约的实际运用：

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """多数投票分类器：预测永远是训练集中最多的类。

    这是一个"傻瓜"分类器，但完整遵守 sklearn 契约。
    """
    def __init__(self):
        # __init__ 不做事，不接收参数（这个算法没超参数）
        pass

    def fit(self, X, y):
        # 1. 校验输入
        X, y = check_X_y(X, y)

        # 2. 学习：找出最多的类
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class_ = unique[counts.argmax()]

        # 3. 记录类别（分类器约定）
        self.classes_ = unique

        # 4. 记录特征数（sklearn 约定）
        self.n_features_in_ = X.shape[1]

        # 5. 返回 self
        return self

    def predict(self, X):
        # 1. 检查是否 fit 过
        check_is_fitted(self)

        # 2. 校验输入
        X = check_array(X)

        # 3. 校验特征数
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this classifier "
                f"expects {self.n_features_in_} features."
            )

        # 4. 纯查询：返回多数类
        return np.full(X.shape[0], self.majority_class_)

    # score 方法由 ClassifierMixin 提供，不用自己写
```

用一下：

```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 1, 1, 1, 0])

clf = MajorityVoteClassifier()
clf.fit(X, y)
print(clf.predict([[10], [20]]))  # [1, 1]（多数类是 1）
print(clf.score(X, y))            # 0.6（accuracy）
print(clf)                        # MajorityVoteClassifier()
```

这个"傻瓜"分类器完整遵守了 sklearn 契约：

- `__init__` 不做事
- `fit` 返回 `self`，学出的属性以 `_` 结尾
- `predict` 先 `check_is_fitted`，纯查询
- `score` 由 `ClassifierMixin` 提供
- `get_params` / `set_params` / `clone` 由 `BaseEstimator` 提供

它可以无缝接入 sklearn 生态：

```python
# 放进 Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MajorityVoteClassifier()),
])
pipe.fit(X, y)
pipe.predict(X)

# 用 GridSearchCV（虽然这算法没参数可搜）
grid = GridSearchCV(MajorityVoteClassifier(), param_grid={})
grid.fit(X, y)
```

这就是统一 API 的威力——只要遵守契约，就能接入整个生态。

### 7.2 常见违反契约的错误

实现自己的估计器时，常见违反契约的错误：

```python
# ❌ 错误 1：__init__ 做了计算
class BadAlgo(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = float(C)  # 错！做了转换

# ❌ 错误 2：fit 不返回 self
class BadAlgo(BaseEstimator):
    def fit(self, X, y):
        self.coef_ = ...  # 忘了 return self

# ❌ 错误 3：学出的属性没用下划线
class BadAlgo(BaseEstimator):
    def fit(self, X, y):
        self.coef = ...  # 错！应该是 coef_

# ❌ 错误 4：predict 修改了状态
class BadAlgo(BaseEstimator):
    def predict(self, X):
        self.predictions_ = ...  # 错！predict 不应改状态
        return ...

# ❌ 错误 5：predict 没检查 is_fitted
class BadAlgo(BaseEstimator):
    def predict(self, X):
        return X @ self.coef_  # 没 check_is_fitted，未 fit 时会 AttributeError
```

这些错误都会导致 `clone`、`Pipeline`、`GridSearchCV` 等出问题。`check_estimator` 能帮你发现大部分。

---

## 8. 契约的演进：SLEP

sklearn 的 API 契约不是一成不变的，它通过 **SLEP（Scikit-Learn Enhancement Proposals）** 演进：

### 8.1 重要的 SLEP

| SLEP | 内容 | 影响 |
|------|------|------|
| SLEP009 | `__init__` 只存参数 | 强制约定，违反则 clone 失效 |
| SLEP010 | `fit` 返回 `self` | 支持链式调用 |
| SLEP011 | 下划线结尾属性 | 区分超参数和学习参数 |
| SLEP013 | `n_features_in_` | 记录 fit 时的特征数 |

### 8.2 为什么需要 SLEP

API 契约是"软约定"，没有编译器强制。如果没有正式文档记录，约定会逐渐被破坏。SLEP 的作用：

1. **正式记录约定**：让约定有据可查
2. **讨论变更**：想改约定时，通过 SLEP 讨论而非随意改
3. **向后兼容**：SLEP 变更时考虑向后兼容

这类似 Python 的 PEP——用正式流程管理 API 演进。

---

## 9. 实际使用模式

理解了契约，我们看看实际使用中的常见模式：

### 9.1 模式 1：算法替换

```python
# 想换算法？改一行就行
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 之前
clf = LogisticRegression(C=1.0)

# 换成 RandomForest
clf = RandomForestClassifier(n_estimators=100)

# 换成 SVM
clf = SVC(C=1.0)

# 下游代码完全不变
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))
```

### 9.2 模式 2：算法组合

```python
# Pipeline 组合任意算法
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = Pipeline([
    ('scaler', StandardScaler()),  # 标准化
    ('pca', PCA(n_components=10)),  # 降维
    ('clf', LogisticRegression()),  # 分类
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 9.3 模式 3：网格搜索

```python
# GridSearchCV 包装任意估计器
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

### 9.4 模式 4：交叉验证

```python
# cross_val_score 对任意估计器跑 CV
from sklearn.model_selection import cross_val_score

clf = LogisticRegression()
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean(), scores.std())
```

这些模式都建立在统一 API 之上——如果每个算法接口不同，这些通用模式都不可能实现。

---

## 10. 常见问题和陷阱

### 10.1 陷阱 1：一维 vs 二维

```python
# ❌ 一维特征
X = np.array([1, 2, 3, 4, 5])
clf.fit(X, y)  # 报错

# ✅ 二维特征
X = np.array([[1], [2], [3], [4], [5]])
clf.fit(X, y)  # 正确
```

### 10.2 陷阱 2：忘记 fit

```python
clf = LogisticRegression()
clf.predict(X)  # NotFittedError
```

### 10.3 陷阱 3：特征数不匹配

```python
clf.fit(X_train, y_train)  # X_train 有 10 个特征
clf.predict(X_test)  # X_test 有 9 个特征 → 报错
```

### 10.4 陷阱 4：修改了 fit 后的对象

```python
clf = LogisticRegression().fit(X, y)
clf.coef_ = np.zeros(10)  # 手动改了 coef_，predict 结果会错
```

### 10.5 陷阱 5：在 predict 里 fit

```python
# ❌ 错误：在 predict 里重新 fit
class BadAlgo(BaseEstimator):
    def predict(self, X):
        self.fit(X)  # 错！predict 不应 fit
        return ...
```

---

## 11. 小结

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| 方法命名 | `fit` / `predict` / `transform` | 统计传统、语义明确、简短 |
| 强制方式 | 鸭子类型 | 灵活、低负担、Python 风格 |
| 契约层次 | Mixin 组合 | 一个类可有多种身份 |
| 参数命名 | 下划线区分学出的参数 | 一眼区分配置与状态 |
| `fit` 返回值 | `self` | 支持链式调用 |
| `predict` 语义 | 纯查询 | 不改状态、可重复调用 |

**核心洞察**：sklearn 的统一 API 不是靠抽象基类强制的，而是靠**约定 + 测试套件**维护的。这种"软约束"换来了极大的灵活性，是动态语言优势的典范运用。

### 11.1 本讲要点回顾

1. **统一 API 是 sklearn 的立身之本**：三个动词（fit/predict/transform）驱动上百个算法。
2. **命名反映哲学**：`fit` 来自统计学，`predict` 语义明确，`transform` 表示数据变换。
3. **鸭子类型而非 ABC**：灵活、低负担，用测试套件弥补。
4. **下划线约定**：区分超参数和学习参数，支撑 clone。
5. **分层而非继承**：Mixin 组合让一个类可有多种身份。
6. **契约最小化**：只有 `fit` 必须，其余按需，覆盖最广场景。
7. **SLEP 管理演进**：正式流程记录和变更约定。

### 11.2 思考题

1. 如果你要为深度学习框架设计 API，会选 `fit`/`predict` 还是 `train`/`infer`？为什么？
2. sklearn 的鸭子类型有什么场景下会出问题？如何缓解？
3. 为什么 `predict` 必须是纯查询？如果允许修改状态，会出什么问题？
4. `fit_transform` 为什么不直接等于 `fit(X).transform(X)`？什么算法可以优化？
5. 如果 sklearn 用 ABC 而非鸭子类型，会失去什么？得到什么？

---

## 12. 延伸阅读：契约设计的理论基础

### 12.1 契约式设计（Design by Contract）

sklearn 的 API 契约让人想起 Bertrand Meyer 提出的"契约式设计"（DbC）。DbC 认为软件组件之间应有明确的"契约"，包含三个部分：

- **前置条件**（Precondition）：调用方法前必须满足的条件
- **后置条件**（Postcondition）：方法执行后保证的条件
- **不变式**（Invariant）：方法执行前后都保持的条件

sklearn 的契约可以套用这个框架：

| 方法 | 前置条件 | 后置条件 | 不变式 |
|------|---------|---------|--------|
| `fit(X, y)` | `X` 是二维数组 | 返回 `self`；创建 `_` 结尾属性 | 超参数不变 |
| `predict(X)` | 已 `fit`；`X` 特征数匹配 | 返回 `(n_samples,)` 数组 | 对象状态不变 |
| `transform(X)` | 已 `fit`；`X` 特征数匹配 | 返回 `(n_samples, n_features_new)` | 对象状态不变 |

理解这个框架有助于设计自己的估计器——明确每个方法的前置、后置、不变式。

### 12.2 里氏替换原则（LSP）

里氏替换原则说：子类应该能替换父类而不破坏程序正确性。

在 sklearn 里，LSP 体现为：**任何 `ClassifierMixin` 的子类都能替换 `ClassifierMixin`**，因为它们都提供 `score` 方法（基于 `predict`）。

```python
# 任何分类器都能用这个函数
def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# LogisticRegression、RandomForest、SVC 都能传进来
# 因为它们都遵守 ClassifierMixin 契约
train_and_evaluate(LogisticRegression(), ...)
train_and_evaluate(RandomForestClassifier(), ...)
train_and_evaluate(SVC(), ...)
```

统一 API 让 LSP 自然成立——所有分类器互换不影响下游代码。

### 12.3 接口隔离原则（ISP）

接口隔离原则说：客户端不应被迫依赖它不使用的方法。

sklearn 的 Mixin 设计符合 ISP：

- `ClassifierMixin` 只提供 `score`，不强迫分类器实现 `transform`
- `TransformerMixin` 只提供 `fit_transform`，不强迫转换器实现 `predict`
- 算法只继承它需要的 Mixin，不背不必要的包袱

对比"上帝基类"违反 ISP——所有算法都被迫继承 `score`、`fit_transform`、`fit_predict` 等所有方法，即使用不到。

### 12.4 开闭原则（OCP）

开闭原则说：对扩展开放，对修改关闭。

sklearn 的契约支持 OCP：

- **对扩展开放**：加新算法只需实现 `fit`/`predict`，不用改现有代码
- **对修改关闭**：加新算法不用改 `Pipeline`、`GridSearchCV` 等元估计器

```python
# 加新算法：不用改 Pipeline
class MyNewAlgo(BaseEstimator, ClassifierMixin):
    def fit(self, X, y): ...
    def predict(self, X): ...

# Pipeline 自动支持新算法
pipe = Pipeline([('clf', MyNewAlgo())])  # 能用，不用改 Pipeline
```

这是统一 API 的最大价值——新算法自动接入整个生态，不用改任何现有代码。

---

## 13. 历史演进：API 的变迁

### 13.1 sklearn 早期（2009-2013）

最早的 sklearn API 还不统一：

- 有些算法用 `learn` 而非 `fit`
- 有些算法的 `predict` 返回概率而非标签
- `fit` 不一定返回 `self`

经过几年演进，才统一到现在的 `fit` / `predict` / `transform`。

### 13.2 SLEP009：`__init__` 约定

2013 年左右，sklearn 正式通过 SLEP009，要求 `__init__` 只存参数。这之前，很多算法在 `__init__` 里做校验，导致 `clone` 出问题。SLEP009 统一了约定，但破坏了很多第三方代码——大量算法被迫重写 `__init__`。

### 13.3 下划线结尾约定的确立

`coef_`、`intercept_` 等下划线结尾的约定也是在演进中确立的。早期有些算法用 `coef`（无下划线），导致 `clone` 无法区分超参数和学习参数。下划线约定解决了这个问题，但同样破坏了向后兼容。

### 13.4 `n_features_in_` 的引入

较新的 sklearn 版本引入了 `n_features_in_` 属性——`fit` 时记录特征数，`predict` 时校验。这之前，`predict` 不校验特征数，传错特征数会得到静默错误。`n_features_in_` 让错误提前暴露。

### 13.5 从 `copy` 参数到不修改输入

早期 sklearn 的转换器有 `copy=True` 参数，控制是否复制输入。后来 sklearn 决定**永远不修改输入**，移除了 `copy` 参数。这是 API 演进的另一个例子——简化契约，减少参数。

---

## 14. 实战案例：从零实现一个完整的估计器

让我们从零实现一个完整的、符合 sklearn 契约的估计器，把本讲所有知识点串起来：

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    """最近质心分类器。

    每个类用其样本均值（质心）表示，预测时找最近的质心。

    参数
    ----
    shrink_threshold : float or None, default=None
        质心收缩阈值。None 表示不收缩。

    属性
    ----
    centroids_ : ndarray, shape (n_classes, n_features)
        每个类的质心。

    classes_ : ndarray, shape (n_classes,)
        类标签。

    n_features_in_ : int
        fit 时的特征数。
    """
    def __init__(self, shrink_threshold=None):
        self.shrink_threshold = shrink_threshold

    def fit(self, X, y):
        # 1. 校验输入
        X, y = check_X_y(X, y)

        # 2. 记录特征数
        self.n_features_in_ = X.shape[1]

        # 3. 记录类别
        self.classes_ = np.unique(y)

        # 4. 计算每个类的质心
        self.centroids_ = np.array([
            X[y == c].mean(axis=0) for c in self.classes_
        ])

        # 5. 可选：质心收缩（正则化）
        if self.shrink_threshold is not None:
            self.centroids_ = self._shrink(self.centroids_, self.shrink_threshold)

        return self

    def predict(self, X):
        # 1. 检查是否 fit
        check_is_fitted(self)

        # 2. 校验输入
        X = check_array(X)

        # 3. 校验特征数
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this classifier "
                f"expects {self.n_features_in_} features."
            )

        # 4. 找最近质心
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        nearest = distances.argmin(axis=1)
        return self.classes_[nearest]

    def _shrink(self, centroids, threshold):
        """软阈值收缩"""
        return np.sign(centroids) * np.maximum(np.abs(centroids) - threshold, 0)
```

测试：

```python
X = np.array([[1, 2], [1, 3], [5, 6], [6, 7]])
y = np.array([0, 0, 1, 1])

clf = NearestCentroidClassifier()
clf.fit(X, y)
print(clf.centroids_)  # [[1, 2.5], [5.5, 6.5]]
print(clf.predict([[1, 2], [6, 7]]))  # [0, 1]
print(clf.score(X, y))  # 1.0
print(clf)  # NearestCentroidClassifier()
```

这个实现完整遵守了 sklearn 契约：

- `__init__` 只存参数，不做计算
- `fit` 返回 `self`，学出的属性以 `_` 结尾
- `predict` 先 `check_is_fitted`，校验输入，纯查询
- `score` 由 `ClassifierMixin` 提供
- `get_params` / `set_params` / `clone` 由 `BaseEstimator` 提供
- `__repr__` 自动生成

它能无缝接入 sklearn 生态：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', NearestCentroidClassifier()),
])
pipe.fit(X, y)
pipe.predict(X)

# GridSearchCV
grid = GridSearchCV(
    NearestCentroidClassifier(),
    param_grid={'shrink_threshold': [None, 0.1, 0.5, 1.0]},
)
grid.fit(X, y)

# cross_val_score
scores = cross_val_score(NearestCentroidClassifier(), X, y, cv=2)
```

这就是统一 API 的威力——遵守契约，接入生态。

---

## 15. 总结：统一 API 的设计哲学

### 15.1 哲学一：约定优于配置

sklearn 用约定（`fit`/`predict`/`transform`、下划线结尾）而非配置（XML、注解）定义 API。只要遵守约定，就能接入生态。这是"约定优于配置"（CoC）原则的体现。

### 15.2 哲学二：最小化契约

sklearn 的契约只有三个方法，却能覆盖上百个算法。最小化契约降低了学习成本和实现负担，是"做减法"的设计智慧。

### 15.3 哲学三：软约束优于硬约束

sklearn 用鸭子类型（软约束）而非 ABC（硬约束）。软约束换来灵活性，用测试套件弥补检查不足。这是动态语言优势的典范运用。

### 15.4 哲学四：组合优于继承

sklearn 用 Mixin 组合身份，用组合构建元估计器。避免了"上帝基类"和菱形继承，让类层次清晰。

### 15.5 哲学五：命名反映哲学

`fit`/`predict`/`transform` 来自统计学传统，反映 sklearn 的定位。命名不是小事——它影响心智模型和生态传播。

---

## 下一讲

[第二讲：Mixin 多继承架构 →](02-mixin-design.md）
