# 第八讲：架构总览

> 把前七讲串起来，看 sklearn 架构的全貌。这一讲不是新内容，而是**回望**——把散落的七块拼图拼成完整的图，看清它们怎么互相支撑，怎么形成一个自洽的闭环。

---

## 1. 架构全景图

```
                        ┌─────────────────────────────────┐
                        │         统一 API 契约            │
                        │   fit / predict / transform      │
                        │   (第一讲)                       │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────┴──────────────────┐
                        │          基类系统                 │
                        │                                   │
                        │  BaseEstimator                    │
                        │    ├── get_params / set_params    │
                        │    ├── clone                      │
                        │    └── __repr__                  │
                        │    (第三讲)                       │
                        │                                   │
                        │  + Mixin (第二讲)                 │
                        │    ├── ClassifierMixin → score    │
                        │    ├── RegressorMixin  → score    │
                        │    ├── TransformerMixin → fit_transform │
                        │    └── ClusterMixin    → fit_predict   │
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────┴─────────┐   ┌─────────┴─────────┐   ┌─────────┴─────────┐
    │     基础估计器      │   │     元估计器       │   │     数据校验       │
    │                     │   │                     │   │                     │
    │  LinearRegression   │   │  Pipeline           │   │  check_array        │
    │  LogisticRegression │   │  GridSearchCV       │   │  check_X_y          │
    │  KNN / DecisionTree │   │  RandomForest       │   │  check_is_fitted    │
    │  KMeans / PCA       │   │  (第四讲)            │   │  (第五讲)           │
    └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                        ┌──────────────┴──────────────────┐
                        │       一致性测试套件              │
                        │    check_estimator(AnyEstimator)  │
                        │    (第六讲)                       │
                        └───────────────────────────────────┘
```

### 1.1 各层的职责

| 层             | 职责                           | 讲次   |
|----------------|--------------------------------|--------|
| 统一 API 契约  | 定义 fit/predict/transform 接口 | 第一讲 |
| 基类系统       | 提供通用方法（clone、repr 等）  | 第二、三讲 |
| 基础估计器     | 实现具体算法                   | 各算法 |
| 元估计器       | 组合其他估计器                 | 第四讲 |
| 数据校验       | 入口防御性编程                 | 第五讲 |
| 一致性测试     | 验证契约被遵守                 | 第六讲 |
| 配置与演进     | 全局行为与版本管理             | 第七讲 |

### 1.2 数据流

一个典型的"训练 + 预测"数据流：

```
用户输入 X, y
    ↓
check_X_y 校验（第五讲）
    ↓
BaseEstimator.fit（第一讲契约）
    ↓
具体算法（基础估计器）
    ↓
学出属性 coef_ 等（下划线约定）
    ↓
用户调 predict
    ↓
check_is_fitted + check_array（第五讲）
    ↓
具体算法 predict
    ↓
返回预测结果
```

### 1.3 控制流

元估计器的控制流：

```
Pipeline.fit(X, y)
    ↓
for step in steps:
    step.fit_transform(X, y)   # TransformerMixin 提供默认实现（第二讲）
    ↓
最后一步 .fit(X, y)
```

元估计器不关心具体是哪种估计器，只依赖统一 API 契约。这是"约定优于配置"的回报。

### 1.4 思考题

1. 全景图里哪一层最重要？去掉哪一层整个架构会塌？
2. 数据校验和一致性测试都"防御"，它们防御的是什么？
3. 元估计器为什么放在基础估计器旁边，而不是上面？

---

## 2. 七个设计决策的内在逻辑

| 讲次 | 设计决策 | 支撑了什么 |
|------|---------|-----------|
| 1 | 统一 API 契约 | 通用测试、元估计器组合、学习成本降低 |
| 2 | Mixin 多继承 | 多重身份、职责单一 |
| 3 | 参数管理 | clone → GridSearchCV → 嵌套搜索 |
| 4 | 元估计器组合 | Pipeline、GridSearchCV、集成 |
| 5 | 数据约定与校验 | 错误前置、防御式编程 |
| 6 | 一致性测试 | 契约的 enforcement |
| 7 | 全局配置与演进 | 长期可维护性 |

它们不是孤立的，而是一个**自洽的闭环**：

```
统一 API → 通用测试可行 → 契约有保障 → 用户放心组合
    ↑                                          ↓
参数管理 ← clone 可行 ← 约定 __init__ 不做事 ← ┘
```

### 2.1 决策的依赖关系

```
第一讲（统一 API）
    ├── 第二讲（Mixin）：统一 API 让 Mixin 能提供默认 score
    ├── 第三讲（参数管理）：统一 API 让 get_params 有统一格式
    ├── 第四讲（元估计器）：统一 API 让 Pipeline 能组合任意估计器
    └── 第六讲（一致性测试）：统一 API 让 check_estimator 能测所有

第三讲（参数管理）
    └── 第四讲（元估计器）：clone 让 GridSearchCV 能复制估计器

第五讲（数据校验）
    └── 所有估计器：入口校验是通用模式

第七讲（配置与演进）
    └── 所有决策：SLEP 管理所有约定的演化
```

### 2.2 闭环的精妙之处

这个闭环的精妙在于：

1. **统一 API** 让通用工具（测试、元估计器）成为可能。
2. **通用测试** 保证契约被遵守，让用户敢用通用工具。
3. **用户敢组合** 让元估计器生态繁荣。
4. **元估计器繁荣** 让 `clone` / `get_params` 的价值放大。
5. **clone 可靠** 让 GridSearchCV 等能工作，进一步放大价值。
6. **价值放大** 让社区愿意遵守约定，回到 1。

这是一个**正反馈循环**：约定越强 → 通用工具越强 → 价值越大 → 越多人遵守约定。

### 2.3 如果打破一个约定

假设去掉"统一 API 契约"：

- 元估计器没法组合任意估计器（每个要特判）。
- 通用测试没法写（每个估计器要单独测）。
- 学习成本升高（每个估计器接口不同）。
- 生态萎缩（没有通用工具加持）。

一个约定破了，整个闭环塌了。这就是为什么 sklearn 对约定这么严格。

### 2.4 思考题

1. 闭环里哪个环节最关键？哪个最脆弱？
2. 如果让你加一个新约定，怎么确保它融入闭环？
3. 闭环是正反馈，会不会"过强"导致僵化？怎么避免？

---

## 3. 设计决策的深度分析

### 3.1 第一讲：统一 API 契约

**决策**：所有估计器有 `fit` / `predict`（或 `transform`）。

**收益**：

- 学习成本：用户学一次，会用所有算法。
- 通用工具：Pipeline、GridSearchCV、cross_val_score 自动支持所有算法。
- 测试：`check_estimator` 一套测所有。

**代价**：

- 算法必须削足适履：不符合 fit/predict 的算法（如强化学习）进不来。
- 表达力受限：复杂训练逻辑（如 GAN 的交替训练）难表达。

**权衡**：sklearn 选择"覆盖 80% 常见场景，牺牲 20% 复杂场景"。深度学习框架（PyTorch）反过来，覆盖复杂场景但牺牲统一性。

### 3.2 第二讲：Mixin 多继承

**决策**：用 Mixin 给估计器加"身份"（分类 / 回归 / 变换 / 聚类）和默认方法。

**收益**：

- 多重身份：一个估计器可以同时是分类器和变换器。
- 职责单一：每个 Mixin 只管一件事。
- 默认方法：`ClassifierMixin` 提供 `score`，不用每个分类器自己写。

**代价**：

- 多继承的复杂性：MRO（方法解析顺序）可能让人困惑。
- Mixin 之间冲突：多个 Mixin 定义同名方法时优先级不清。

**权衡**：sklearn 的 Mixin 设计克制，每个 Mixin 很小，冲突少。

### 3.3 第三讲：参数管理

**决策**：`__init__` 只存参数，`get_params` / `set_params` / `clone` 基于此工作。

**收益**：

- `clone` 可靠：能创建干净副本。
- GridSearchCV 可工作：能复制估计器试不同参数。
- 嵌套搜索：GridSearchCV 里套 GridSearchCV。

**代价**：

- `__init__` 不能做工作：校验、转换要移到 `fit`。
- 参数必须是简单类型：不能存复杂对象（虽然实际可以，但不推荐）。

**权衡**：把"工作"推迟到 `fit`，换取 `clone` 的可靠性。

### 3.4 第四讲：元估计器组合

**决策**：元估计器把其他估计器当参数，通过统一 API 调用它们。

**收益**：

- Pipeline：串联任意估计器。
- GridSearchCV：搜索任意超参数。
- 集成：组合任意基估计器。
- 嵌套：元估计器套元估计器。

**代价**：

- 调试复杂：错误信息可能深埋多层。
- 性能：每层有开销。

**权衡**：组合力远超调试成本。

### 3.5 第五讲：数据约定与校验

**决策**：统一数据形状 `(n_samples, n_features)`，入口校验。

**收益**：

- 错误前置：清晰报错而非深层崩溃。
- 通用校验：所有估计器复用 `check_array`。
- 与 NumPy/pandas 无缝：数据流转顺畅。

**代价**：

- 校验开销：大数据有 5-10% 开销。
- 灵活性：不支持的数据形状要 reshape。

**权衡**：正确性优先，性能可配置（`assume_finite`）。

### 3.6 第六讲：一致性测试

**决策**：`check_estimator` 一套测试测所有估计器。

**收益**：

- 新增估计器自动获得测试。
- 契约有 enforcement，不是纸上文字。
- 测试代码量骤减。

**代价**：

- 契约测试不保证算法正确。
- 某些估计器天然不满足某些契约，要 skip。

**权衡**：契约测试是第一道防线，算法测试补充。

### 3.7 第七讲：配置与演进

**决策**：全局配置 `config_context`，SLEP 管理演进，deprecation 流程保兼容。

**收益**：

- 长期可维护：十几年还在进化。
- 用户敢升级：兼容承诺强。
- 演进有据：SLEP 文档化。

**代价**：

- 演进慢：一个 SLEP 可能讨论一年。
- 兼容包袱：旧 API 要维护很久。

**权衡**：稳定优先，速度其次。

### 3.8 思考题

1. 每个决策的"代价"值得吗？有没有更好的折中？
2. 如果让你重新设计 sklearn，会改哪个决策？
3. 这些决策之间有没有冲突？怎么调和？

---

## 4. 架构的分层

### 4.1 三层架构

sklearn 可以分三层：

```
应用层：用户代码
    ↓
框架层：Pipeline、GridSearchCV、cross_val_score（元估计器 + 通用工具）
    ↓
算法层：LogisticRegression、KMeans、PCA（基础估计器）
    ↓
基础层：BaseEstimator、Mixin、check_array（基类 + 校验）
```

### 4.2 各层的依赖方向

依赖**向下**：上层依赖下层，下层不依赖上层。

```
应用层 → 框架层 → 算法层 → 基础层
```

这保证：

- 基础层改了，上层可能要改（但下层不知道）。
- 上层改了，下层不受影响。
- 测试可以分层：基础层先测，算法层后测。

### 4.3 依赖反转

元估计器（框架层）依赖基础估计器（算法层）的**抽象**（统一 API），而非具体类。这是依赖反转：

```
Pipeline 不 import LogisticRegression
Pipeline 只假设输入是"有 fit/predict 的对象"
```

这让 Pipeline 能组合任意估计器，包括用户自定义的、第三方库的。

### 4.4 思考题

1. 为什么依赖要向下？反向依赖会出什么问题？
2. 依赖反转在 sklearn 里还体现在哪？
3. 三层架构和 MVC 有什么相似之处？

---

## 5. 数据流与控制流

### 5.1 训练时的数据流

```
X, y → check_X_y → fit 内部 → 学出属性 → 返回 self
```

数据从用户进来，先过校验，再到算法，最后产出学出属性。

### 5.2 预测时的数据流

```
X → check_array → check_is_fitted → predict 内部 → 返回预测
```

预测时先校验输入和模型状态，再算。

### 5.3 元估计器的控制流

```
Pipeline.fit(X, y):
    for step in steps[:-1]:
        X = step.fit_transform(X, y)   # 变换 + 传到下一步
    steps[-1].fit(X, y)                # 最后一步只 fit
```

元估计器把数据在子估计器间传递，自己不做算法。

### 5.4 GridSearchCV 的控制流

```
GridSearchCV.fit(X, y):
    for params in param_grid:
        est = clone(base_est).set_params(**params)
        score = cross_val_score(est, X, y)
    best = base_est.set_params(**best_params).fit(X, y)
```

GridSearchCV 用 `clone` 复制估计器，试不同参数，选最优。

### 5.5 思考题

1. 训练和预测的数据流有什么不同？为什么？
2. Pipeline 的控制流为什么是"前 n-1 步 fit_transform，最后一步 fit"？
3. GridSearchCV 为什么用 `clone` 而不是直接 `set_params`？

---

## 6. 与其他框架的架构对比

### 6.1 PyTorch

PyTorch 的架构：

```
应用层：用户训练循环
    ↓
框架层：nn.Module、optim、autograd
    ↓
算法层：nn.Linear、nn.Conv2d
    ↓
基础层：Tensor、C++ 后端
```

对比 sklearn：

- PyTorch 没有统一 `fit` / `predict`，训练循环用户写。
- PyTorch 没有 `clone` / `get_params`（有 `state_dict` 但语义不同）。
- PyTorch 没有通用测试套件。
- PyTorch 的"组合"靠 `nn.Sequential`，类似 Pipeline 但更灵活。

PyTorch 选择"灵活性优先"，sklearn 选择"约定优先"。

### 6.2 TensorFlow / Keras

Keras 的架构：

```
应用层：用户代码
    ↓
框架层：Model.fit（内置训练循环）
    ↓
算法层：Layer
    ↓
基础层：Tensor、tf.function
```

Keras 有统一 `fit` / `predict`，但：

- 没有 `clone` / `get_params`。
- 没有通用测试套件。
- 组合靠 `Sequential` 或 Functional API。

Keras 介于 sklearn 和 PyTorch 之间：有统一 API 但不如 sklearn 严格。

### 6.3 R caret / tidymodels

R 的 tidymodels：

```
应用层：用户代码
    ↓
框架层：workflow、tune、recipes
    ↓
算法层：rand_forest、logistic_reg
    ↓
基础层：parsnip、rsample
```

tidymodels 借鉴了 sklearn 的理念（统一 API、Pipeline、GridSearch），但用 R 的语法和生态。

### 6.4 对比表

| 维度         | sklearn       | PyTorch       | Keras         | tidymodels    |
|--------------|---------------|---------------|---------------|---------------|
| 统一 API     | 强            | 弱            | 中            | 强            |
| clone        | 有            | 无            | 无            | 有            |
| 通用测试     | 有            | 无            | 部分          | 有            |
| 元估计器     | 丰富          | 少            | 中            | 丰富          |
| 灵活性       | 中            | 高            | 中高          | 中            |
| 学习成本     | 低            | 高            | 中            | 低            |

### 6.5 思考题

1. PyTorch 没有 `fit` / `predict`，是缺陷还是选择？
2. Keras 介于 sklearn 和 PyTorch 之间，它的折中点在哪？
3. tidymodels 借鉴 sklearn，有什么 R 特色的改进？

---

## 7. 架构的演化

### 7.1 早期（0.1-0.10）：探索期

- API 不统一，每个估计器自己定接口。
- 没有 `check_estimator`。
- `clone` 行为不可预测。

### 7.2 中期（0.11-0.19）：统一期

- 统一 API 契约确立。
- `check_estimator` 引入。
- Pipeline、GridSearchCV 完善。

### 7.3 后期（0.20-0.24）：严格期

- SLEP009 强制 `__init__` 只存参数。
- 大量 deprecation。
- 契约测试加严。

### 7.4 稳定期（1.0+）：成熟期

- API 稳定标记。
- 移除弃用 API。
- 持续优化、新算法。

### 7.5 演化的教训

- **早期不严**：约定不严导致后续大量重构。
- **中期加严**：SLEP009 等强制约定，短期痛苦长期收益。
- **后期稳定**：1.0 标记成熟，进入维护期。

### 7.6 思考题

1. 早期不严的代价是什么？能不能一开始就严？
2. 中期加严为什么短期痛苦？具体痛苦在哪？
3. 1.0 之后 sklearn 还能怎么演进？

---

## 8. 一句话总结

> sklearn 的架构本质是：**用最小的约定（统一 API + `__init__` 只存参数），换最大的自由（通用测试 + 元估计器组合 + 嵌套搜索）**。

这种"以约定换能力"的设计哲学，值得每一个库作者学习。

### 8.1 这句话的拆解

- **最小的约定**：只有两条核心约定（统一 API + `__init__` 只存参数），其他都派生。
- **最大的自由**：通用测试、元估计器、嵌套搜索都自动获得。
- **以约定换能力**：约定是投入，能力是回报。

### 8.2 为什么是"最小"约定

sklearn 的约定很少：

- `fit` / `predict` / `transform` 的签名。
- `__init__` 只存参数。
- 下划线结尾是学出属性。
- 数据形状 `(n_samples, n_features)`。

就这几条，撑起了整个生态。这是"少即是多"在架构里的体现。

### 8.3 为什么是"最大"自由

从这几条约定派生出：

- 几十种算法都通过 `check_estimator`。
- Pipeline 能组合任意估计器。
- GridSearchCV 能搜任意超参数。
- 嵌套搜索、嵌套 Pipeline 任意深度。
- clone、pickle、repr 都自动工作。

约定越统一，派生的自由越多。

### 8.4 思考题

1. "最小约定"能不能再减？减了会怎样？
2. "最大自由"有没有上限？会不会过度自由导致混乱？
3. "以约定换能力"在其他领域（如 API 设计、协议设计）有体现吗？

---

## 9. 架构的代价

### 9.1 表达力受限

统一 API 的代价：

- 不支持不符合 fit/predict 的算法（强化学习、GAN）。
- 复杂训练逻辑（交替训练、课程学习）难表达。
- 非标准数据流（多输入、多输出）要包装。

### 9.2 性能开销

约定的代价：

- 入口校验：5-10% 开销。
- clone 复制：GridSearchCV 里大量 clone。
- 元估计器层层转发：每层有调用开销。

### 9.3 演进缓慢

兼容承诺的代价：

- 旧 API 要维护很久。
- 新功能要走 SLEP 流程，慢。
- 技术债积累（不能痛快地重构）。

### 9.4 学习曲线

约定的代价：

- 初学者要理解"为什么 `__init__` 不做事"。
- 下划线约定、clone 语义要学。
- 错误信息虽然清晰，但约定本身要学。

### 9.5 思考题

1. 这些代价值得吗？哪个最值得？哪个最不值得？
2. 如果不要某个代价，要放弃什么收益？
3. 怎么在"约定"和"灵活"之间找新平衡点？
4. 性能开销 5-10% 对大规模生产环境意味着什么？

---

## 10. 架构的启示

### 10.1 给库作者的启示

1. **统一 API 是最大的杠杆**：一次定义，所有实现受益。
2. **约定要早立**：早期不严，后期重构成本高。
3. **测试即契约 enforcement**：没有测试的约定是空话。
4. **兼容是资产**：用户敢升级是长期生存的关键。
5. **配置 vs 参数要分清**：全局行为用配置，算法行为用参数。

### 10.2 给用户的启示

1. **遵守约定**：自定义估计器遵守 sklearn 约定，自动获得通用工具支持。
2. **用元估计器**：Pipeline、GridSearchCV 比手写循环好。
3. **看懂下划线**：`coef_` 是学出的，`C` 是超参数。
4. **校验报错是好事**：清晰报错比深层崩溃好。

### 10.3 给设计者的启示

1. **以约定换能力**：少几条约定，多很多能力。
2. **闭环设计**：约定 → 通用工具 → 价值 → 更多约定，正反馈。
3. **分层清晰**：基础层 / 算法层 / 框架层，依赖向下。
4. **依赖反转**：框架层依赖算法层的抽象，而非具体。

### 10.4 思考题

1. 这些启示里，哪个对你最有触动？为什么？
2. 你写的代码里，有没有"约定不严导致重复"的地方？
3. 怎么把"以约定换能力"用到你自己的项目里？
4. 给库作者的 5 条启示，按重要性排序你会怎么排？

---

## 11. 把七讲串起来：一个完整的例子

用一个完整例子串起七讲：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

# 第一讲：统一 API——StandardScaler 和 LogisticRegression 都有 fit/transform/predict
# 第二讲：Mixin——StandardScaler 是 TransformerMixin，LogisticRegression 是 ClassifierMixin
# 第三讲：参数管理——LogisticRegression(C=1.0) 的 C 是超参数
# 第四讲：元估计器——Pipeline 组合 StandardScaler 和 LogisticRegression
pipe = Pipeline([
    ("scaler", StandardScaler()),       # 第五讲：check_array 校验输入
    ("clf", LogisticRegression()),      # 下划线约定：fit 后有 coef_
])

# 第四讲：GridSearchCV 是元估计器，组合 Pipeline
param_grid = {
    "clf__C": [0.1, 1.0, 10.0],         # 第三讲：get_params 看到嵌套参数
}
search = GridSearchCV(pipe, param_grid)

# 第六讲：check_estimator 能测 Pipeline 和 GridSearchCV
check_estimator(Pipeline)
check_estimator(GridSearchCV)

# 第七讲：config_context 控制全局行为
from sklearn import config_context
with config_context(assume_finite=True):
    search.fit(X, y)                    # 跳过 NaN 检查

# 第四讲：元估计器的 predict 转发到子估计器
search.predict(X_test)
```

这个例子用到了全部七讲的内容，体现了架构的**整体性**。

### 11.1 思考题

1. 这个例子里，每一讲具体体现在哪行代码？
2. 如果没有某一讲的约定，这个例子会怎么崩？
3. 怎么给这个例子加一个自定义估计器，让它融入架构？
4. 这个例子如果换成 PyTorch 写，会多多少代码？

---

## 12. 架构的未来说

### 12.1 挑战

- **深度学习集成**：怎么和 PyTorch / TF 协作？
- **大数据**：怎么支持超出内存的数据？
- **GPU**：怎么利用 GPU 加速？
- **流式学习**：怎么支持在线学习？

### 12.2 方向

- **更严格的类型提示**：mypy 静态检查补充运行时校验。
- **更丰富的元估计器**：AutoML、超参数优化。
- **更好的可解释性**：统一的 feature importance 接口。
- **更高效的实现**：Cython、numba、GPU 后端。

### 12.3 思考题

1. 这些挑战里，哪个最难？为什么？
2. sklearn 的架构能容纳这些新方向吗？要改什么？
3. 如果 sklearn 要支持 GPU，约定要怎么变？
4. 流式学习如果要成为一等公民，fit 的语义要怎么改？
5. 深度学习集成最自然的接口是什么？

---

## 13. 常见问题

### 13.1 为什么 sklearn 不用继承而用 Mixin？

继承（is-a）表达"是什么"，Mixin（has-ability）表达"能做什么"。一个估计器"能分类"也能"能变换"，用多重 Mixin 比多重继承清晰。

### 13.2 为什么不用 ABC 强制契约？

ABC 在 Python 里只能强制方法存在，不能强制行为。`check_estimator` 测行为（幂等、不改状态），比 ABC 强。

### 13.3 为什么不用 dataclass？

dataclass 适合纯数据容器，估计器有行为（fit/predict）和状态（学出属性），不适合。

### 13.4 思考题

1. Mixin 和继承的边界在哪？什么时候用哪个？
2. ABC + check_estimator 能互补吗？各自抓什么？
3. dataclass 能用在哪？参数管理能用吗？
4. 为什么 sklearn 不用 Protocol（结构化子类型）？
5. dataclass 的 frozen=True 对估计器有意义吗？

---

## 14. 架构的量化评估

### 14.1 代码量

- 基础层（BaseEstimator、Mixin、check_array）：约 5000 行。
- 算法层（所有估计器）：约 50000 行。
- 框架层（Pipeline、GridSearchCV 等）：约 10000 行。
- 测试层：约 30000 行。

基础层只占 5%，但撑起其他 95%。这是"杠杆"的体现。

### 14.2 约定的数量

核心约定只有 4-5 条（统一 API、`__init__` 只存参数、下划线、数据形状、校验），派生出上百种算法和几十种通用工具。

### 14.3 测试覆盖率

sklearn 测试覆盖率 > 90%，其中 `check_estimator` 贡献了相当一部分（通用测试）。

### 14.4 思考题

1. 基础层只占 5% 代码量但撑起 95%，这说明什么？
2. 4-5 条约定派生上百种算法，"杠杆率"是多少？
3. 测试覆盖率 90% 够吗？剩下的 10% 是什么？
4. 基础层的 5% 代码改一行，可能影响多少上层代码？
5. 怎么量化"架构的杠杆率"？

---

## 15. 下一步

架构设计讲完了，接下来进入[算法实现](../algorithms/index.md)环节——亲手实现每个算法，体会"有了这套架构，加新算法只需要写 `fit` 和 `predict`"。

### 15.1 学完架构后该做什么

1. **实现一个估计器**：从零写一个，体会约定。
2. **跑 `check_estimator`**：看你的估计器能不能通过。
3. **读 sklearn 源码**：看真实估计器怎么遵守约定。
4. **写元估计器**：组合现有估计器，体会元估计器模式。

### 15.2 推荐阅读

- 《Design Patterns》：经典设计模式，很多在 sklearn 里体现。
- 《API Design for C++》：API 设计原则，语言无关。
- sklearn 的 SLEP 文档：真实的架构决策记录。
- sklearn 的源码：最好的"架构文档"。

### 15.3 思考题

1. 学完架构后，你最想实现哪个算法？为什么？
2. 读 sklearn 源码时，重点看什么？
3. 怎么把架构理念用到你自己的项目里？
4. 算法实现和架构理解，哪个应该先做？
5. 实现 3 个算法后，你对架构的理解会怎么变化？

---

## 16. 架构的反模式

### 16.1 违反统一 API

```python
# 坏：自定义方法而非 fit/predict
class MyModel:
    def train(self, X, y): ...      # 不是 fit
    def infer(self, X): ...         # 不是 predict
```

这种模型进不了 Pipeline、GridSearchCV，失去所有通用工具支持。

### 16.2 `__init__` 做工作

```python
# 坏：__init__ 里校验
class MyModel:
    def __init__(self, C=1.0):
        if C <= 0:
            raise ValueError       # __init__ 不该报错
        self.C = C
```

`clone` 后 `__init__` 重跑，可能副作用。

### 16.3 学出属性没下划线

```python
# 坏：coef 没下划线
class MyModel:
    def fit(self, X, y):
        self.coef = ...            # 应该是 coef_
```

`get_params` 会把 `coef` 当超参数，`clone` 保留它，破坏语义。

### 16.4 predict 改状态

```python
# 坏：predict 改 coef_
class MyModel:
    def predict(self, X):
        self.coef_ += 0.001        # 不该改
        return X @ self.coef_
```

多次 predict 结果不一致，违反"predict 是纯查询"。

### 16.5 不做入口校验

```python
# 坏：直接用 X，不校验
class MyModel:
    def fit(self, X, y):
        self.coef_ = np.linalg.lstsq(X, y)   # X 有 NaN 时崩溃信息晦涩
```

应该先 `check_array(X)` 把错误前置。

### 16.6 思考题

1. 这些反模式里，哪个最常见？哪个最难发现？
2. 反模式不修，短期和长期分别有什么后果？
3. `check_estimator` 能抓到哪些反模式？哪些抓不到？
4. 怎么用 lint 规则自动检测这些反模式？
5. 反模式在代码审查时怎么识别？

---

## 17. 架构与设计模式

sklearn 用了很多经典设计模式：

### 17.1 模板方法（Template Method）

`BaseEstimator` 提供 `__repr__`、`get_params` 等通用方法，子类只实现 `fit` / `predict`。

### 17.2 策略（Strategy）

元估计器（Pipeline、GridSearchCV）把"具体算法"作为策略传入，运行时调用。

```python
Pipeline([("clf", LogisticRegression())])   # 策略是 LogisticRegression
Pipeline([("clf", RandomForest())])          # 换策略
```

### 17.3 装饰器（Decorator）

Mixin 装饰估计器，加新方法（`score`、`fit_transform`）。

### 17.4 组合（Composite）

元估计器组合其他估计器，形成树状结构。Pipeline 套 Pipeline 是组合的递归。

### 17.5 原型（Prototype）

`clone` 用原型模式创建估计器副本。

### 17.6 上下文（Context）

`config_context` 用上下文模式管理全局状态。

### 17.7 思考题

1. 这些设计模式在 sklearn 里是显式还是隐式？
2. 还有哪些设计模式在 sklearn 里体现？
3. 设计模式过度使用会不会让代码难读？
4. 模板方法和策略模式在 sklearn 里怎么协作？
5. 原型模式（clone）为什么不用 copy.deepcopy？

---

## 18. 架构的测试策略

### 18.1 测试金字塔

```
        /\
       /  \     集成测试（端到端 Pipeline）
      /----\
     /      \   契约测试（check_estimator）
    /--------\
   /          \ 算法测试（每个算法的正确性）
  /------------\
 /              \单元测试（工具函数）
/----------------\
```

底层多、上层少。

### 18.2 各层测试的职责

| 层       | 测什么               | 谁来跑         |
|----------|----------------------|----------------|
| 单元     | 工具函数             | 每个函数       |
| 算法     | 算法正确性           | 每个算法       |
| 契约     | API 契约             | 所有估计器     |
| 集成     | 组合行为             | 估计器 + 元估计器 |

### 18.3 测试的运行时间

- 单元测试：秒级。
- 算法测试：分钟级。
- 契约测试：分钟级（参数化跑所有估计器）。
- 集成测试：分钟级。

sklearn 全套测试约 10-30 分钟。

### 18.4 思考题

1. 测试金字塔的形状为什么是底宽顶窄？
2. 契约测试放哪层？它和算法测试的边界在哪？
3. 怎么把测试时间从 30 分钟降到 5 分钟？
4. 集成测试太少会有什么风险？太多呢？
5. 怎么用 pytest marker 分层跑测试？

---

## 19. 架构与文档

### 19.1 文档的层次

```
教程（user guide）：怎么用
    ↓
API reference：每个类的参数和方法
    ↓
架构文档（这七讲）：为什么这么设计
    ↓
源码：最终真相
```

### 19.2 架构文档的价值

- 帮助贡献者理解约定，写出符合规范的代码。
- 帮助用户理解"为什么"，不只是"怎么用"。
- 记录设计决策的权衡，避免后人重复讨论。

### 19.3 文档与代码的同步

架构改了，文档要更新。sklearn 的 SLEP 流程要求提案包含文档更新。

### 19.4 思考题

1. 架构文档和 API 文档的区别是什么？
2. 怎么保证文档和代码同步？
3. 这七讲文档还能怎么改进？
4. 架构文档应该写给谁看？初学者还是贡献者？
5. 怎么衡量架构文档的质量？

---

## 20. 架构的跨语言启示

### 20.1 Julia MLJ

Julia 的 MLJ 框架借鉴了 sklearn 的架构：

- 统一 `fit` / `predict` API。
- `machine` 类似 sklearn 的估计器。
- Pipeline、GridSearch 等元估计器。

但用 Julia 的多重分派，比 Python 的 Mixin 更优雅。

### 20.2 Rust linfa

Rust 的 linfa 框架也借鉴 sklearn：

- 统一 `fit` / `predict`。
- 用 trait（类似 Mixin）定义身份。
- 强类型保证契约。

Rust 的类型系统让很多 sklearn 在运行时检查的契约变成编译时检查。

### 20.3 Go golearn

Go 的 golearn 尝试移植 sklearn，但 Go 没有继承，用接口实现统一 API。不如 sklearn 成功，部分原因是 Go 生态不适合 ML。

### 20.4 启示

sklearn 的架构理念（统一 API + 约定 + 元估计器）可以跨语言，但具体实现要适应语言特性。

### 20.5 思考题

1. Julia 的多重分派比 Python 的 Mixin 好在哪？差在哪？
2. Rust 的类型系统能把哪些 sklearn 运行时检查变成编译时？
3. Go 不适合 ML，是语言的限制还是生态的限制？
4. 如果用 C++ 实现 sklearn 架构，会怎么设计？
5. 跨语言移植架构时，哪些能移植，哪些不能？

---

## 21. 架构的实战检验

### 21.1 用架构写一个新估计器

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

class RidgeRegression(BaseEstimator, RegressorMixin):
    """岭回归：用架构约定写一个新估计器。"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha   # 第三讲：__init__ 只存参数

    def fit(self, X, y):
        # 第五讲：入口校验
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 第五讲：记录特征数
        self.n_features_in_ = X.shape[1]

        # 算法
        n = X.shape[0]
        XtX = X.T @ X + n * self.alpha * np.eye(self.n_features_in_)
        Xty = X.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)   # 第六讲：下划线结尾

        return self   # 第一讲：fit 返回 self

    def predict(self, X):
        # 第五讲：检查已拟合
        check_is_fitted(self, 'coef_')
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("特征数不匹配")
        return X @ self.coef_
```

### 21.2 检验它融入架构

```python
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# 第六讲：通过契约测试
check_estimator(RidgeRegression)

# 第四讲：能进 Pipeline
pipe = Pipeline([("ridge", RidgeRegression())])
pipe.fit(X, y)

# 第四讲：能进 GridSearchCV
search = GridSearchCV(RidgeRegression(), {"alpha": [0.1, 1.0, 10.0]})
search.fit(X, y)

# 第三讲：clone 可工作
from sklearn.base import clone
r = RidgeRegression(alpha=5.0)
r_clone = clone(r)
assert r_clone.alpha == 5.0
assert not hasattr(r_clone, 'coef_')   # 干净副本
```

这就是架构的回报：写一个估计器，自动获得所有通用工具支持。

### 21.3 思考题

1. 这个 RidgeRegression 通过 `check_estimator` 吗？哪些可能失败？
2. 如果 `__init__` 里加了校验，会破坏什么？
3. 怎么给这个估计器加 `score` 方法？需要自己写吗？
4. 这个估计器能进 imbalanced-learn 的 Pipeline 吗？为什么？
5. 怎么给它加 `feature_importances_` 属性？合理吗？

---

## 22. 架构的边界：什么 sklearn 做不好

### 22.1 深度学习

sklearn 的统一 API 假设"fit 一次，predict 多次"。深度学习的训练循环复杂（epoch、batch、early stopping），难塞进 `fit`。

### 22.2 流式学习

sklearn 的 `fit` 假设"一次性给全部数据"。流式学习（partial_fit 存在但支持有限）不是一等公民。

### 22.3 自定义损失

sklearn 的算法用固定损失。自定义损失要重写算法，不像 PyTorch 那样灵活。

### 22.4 GPU 加速

sklearn 主要在 CPU。GPU 支持要靠 cuML 等外部库，不是原生。

### 22.5 大规模数据

超出内存的数据 sklearn 不直接支持。Dask-ml 等外部库补充，但生态不如 Spark MLlib。

### 22.6 思考题

1. 这些边界里，哪个最难突破？为什么？
2. sklearn 为什么不自己支持深度学习？
3. 这些边界是架构的缺陷还是有意的选择？
4. 流式学习为什么在 sklearn 里是二等公民？
5. GPU 支持为什么靠外部库而不是核心？

---

## 23. 架构与生态

### 23.1 sklearn 生态

```
sklearn 核心
    ├── scikit-learn-contrib（官方认可的扩展）
    │   ├── imbalanced-learn
    │   ├── scikit-learn-extra
    │   └── ...
    ├── 第三方扩展
    │   ├── xgboost（sklearn API）
    │   ├── lightgbm（sklearn API）
    │   └── ...
    └── 互操作
        ├── cuML（GPU 版 sklearn API）
        └── dask-ml（分布式 sklearn API）
```

### 23.2 生态的黏合剂

统一 API 契约是生态的黏合剂：

- xgboost 提供 `XGBClassifier`，有 `fit` / `predict`，能进 sklearn Pipeline。
- imbalanced-learn 的 `SMOTE` 有 `fit_resample`，能进 imblearn Pipeline。
- cuML 的 `LogisticRegression` 有相同 API，能直接替换 sklearn 的。

### 23.3 生态的健康指标

- 估计器数量：sklearn 核心 100+，contrib 50+，第三方 100+。
- 通过 `check_estimator` 的比例：核心 100%，contrib 大部分。
- 互操作性：xgboost / lightgbm 都能进 sklearn Pipeline。

### 23.4 思考题

1. 统一 API 怎么成为生态黏合剂？
2. 第三方库为什么要兼容 sklearn API？
3. 生态健康和架构设计什么关系？
4. 如果 xgboost 不兼容 sklearn API，会损失什么？
5. 怎么衡量一个生态的健康度？

---

## 24. 架构的反思

### 24.1 如果重新设计

如果今天重新设计 sklearn，可能会：

- 用类型提示 + pydantic 做参数校验（比 SLEP018 装饰器更现代）。
- 用 dataclass 简化 `__init__`（但要小心不破坏 `__init__` 只存参数的约定）。
- 用 async/await 支持流式学习。
- 用 trait（类似 Rust）替代 Mixin。

### 24.2 不会改的

- 统一 `fit` / `predict` API。
- `__init__` 只存参数。
- 下划线约定。
- 元估计器组合。

这些是架构的核心，改了就不是 sklearn 了。

### 24.3 架构的年龄

sklearn 从 2007 年到现在（2026 年）近 20 年。架构核心没大变，说明设计经得起时间检验。

### 24.4 思考题

1. 重新设计时，哪些现代特性值得引入？哪些不值得？
2. 架构核心 20 年没大变，是好事还是僵化？
3. 如果 sklearn 2.0 破坏兼容，最该破什么？
4. 类型提示能替代哪些运行时检查？哪些不能？
5. async/await 对 sklearn 的 fit/predict 意味着什么？

---

## 25. 架构的微观：一个方法的旅程

### 25.1 `clone` 的实现旅程

```python
def clone(estimator, safe=True):
    # 1. 拿到 estimator 的所有超参数
    params = estimator.get_params(deep=False)

    # 2. 递归 clone 嵌套的元估计器参数
    for name, param in params.items():
        if isinstance(param, BaseEstimator):
            params[name] = clone(param, safe=False)

    # 3. 用超参数重新 __init__
    new = type(estimator)(**params)

    # 4. 不复制任何学出属性（__init__ 只存参数保证这点）
    return new
```

这个简单的函数依赖：

- 第三讲：`get_params` 能拿到所有超参数。
- 第三讲：`__init__` 只存参数，所以新实例是干净的。
- 第四讲：递归 clone 处理嵌套元估计器。

### 25.2 `Pipeline.fit` 的旅程

```python
def fit(self, X, y=None):
    # 1. 校验
    X, y = check_X_y(X, y)   # 第五讲

    # 2. 遍历前 n-1 步
    for step in self.steps[:-1]:
        transformer = step[1]
        X = transformer.fit_transform(X, y)   # 第二讲：Mixin 提供 fit_transform

    # 3. 最后一步
    final = self.steps[-1][1]
    final.fit(X, y)   # 第一讲：统一 fit

    return self
```

依赖：

- 第一讲：所有子估计器有 `fit` / `fit_transform`。
- 第二讲：TransformerMixin 提供 `fit_transform` 默认实现。
- 第五讲：入口校验。

### 25.3 `GridSearchCV.fit` 的旅程

```python
def fit(self, X, y):
    # 1. 校验
    X, y = check_X_y(X, y)

    # 2. 遍历参数网格
    for params in self._param_grid:
        # 3. clone 基估计器
        est = clone(self.estimator)   # 第三讲：clone

        # 4. 设新参数
        est.set_params(**params)   # 第三讲：set_params

        # 5. 交叉验证
        score = cross_val_score(est, X, y, cv=self.cv)

    # 6. 选最优，用全数据重训
    best = clone(self.estimator).set_params(**best_params)
    best.fit(X, y)
    self.best_estimator_ = best

    return self
```

依赖：

- 第三讲：`clone` + `set_params`。
- 第四讲：`cross_val_score` 是元估计器。
- 第五讲：校验。

### 25.4 思考题

1. `clone` 如果没有 `__init__` 只存参数的约定，会怎样？
2. `Pipeline.fit` 如果子估计器没有 `fit_transform`，会怎样？
3. `GridSearchCV.fit` 如果 `clone` 不可靠，会怎样？
4. 这三个方法的旅程里，哪步依赖最多讲的约定？
5. 如果要给 `clone` 加缓存，要注意什么？

---

## 26. 架构的宏观：从用户到源码

### 26.1 用户的视角

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([("clf", LogisticRegression(C=1.0))])
pipe.fit(X, y)
pipe.predict(X_test)
```

用户看到的是简洁的 API，背后是七讲的架构在支撑。

### 26.2 调用的展开

```
pipe.fit(X, y)
    → Pipeline.fit
        → check_X_y(X, y)              # 第五讲
        → LogisticRegression.fit(X, y)  # 第一讲契约
            → check_array(X)            # 第五讲
            → 算法求解
            → self.coef_ = ...          # 下划线约定
        → return self
```

### 26.3 源码的视角

在源码里，每一步都体现约定：

- `Pipeline.fit` 假设子估计器有 `fit`（第一讲）。
- `LogisticRegression.fit` 先 `check_array`（第五讲）。
- 学出属性 `coef_` 有下划线（第五讲）。

### 26.4 思考题

1. 用户的简洁 API 背后有多少架构在支撑？
2. 如果某层约定破了，用户会看到什么现象？
3. 怎么向用户解释"为什么 API 这么简洁"？
4. 用户不需要懂架构就能用 sklearn，这是好事还是坏事？
5. 怎么在"隐藏复杂度"和"让用户理解"之间平衡？

---

## 27. 架构的哲学总结

### 27.1 约定优于配置

sklearn 是"约定优于配置"的极致：

- 约定 `fit` / `predict`，不用配置每个算法的接口。
- 约定 `__init__` 只存参数，不用配置 clone 行为。
- 约定下划线，不用配置哪些是学出属性。

### 27.2 少即是多

4-5 条核心约定，派生上百种算法和几十种工具。约束越多，自由越多（在正确的约束下）。

### 27.3 慢即是快

SLEP 流程慢，但想得清楚，避免反复改。deprecation 流程慢，但用户敢升级，生态健康。

### 27.4 制度优于个人

SLEP + deprecation 是制度，不依赖个人意志。这让 sklearn 20 年稳定演进。

### 27.5 思考题

1. 这四条哲学里，哪条最深刻？
2. "约束越多，自由越多"在什么前提下成立？
3. 怎么把这套哲学用到你自己的项目里？
4. "约定优于配置"有没有反面——约定太严导致僵化？怎么避免？
5. "少即是多"和"简单不等于容易"有什么关系？
6. "制度优于个人"在开源和公司项目里分别怎么体现？

---

## 28. 架构的练习与自测

### 28.1 自测题

1. 画出 sklearn 的架构全景图，标出七讲对应的位置。
2. 解释"统一 API → 通用测试 → 契约保障 → 用户组合"这个闭环。
3. 列出 4-5 条核心约定，说明每条派生什么。
4. 比较 sklearn 和 PyTorch 的架构取舍。
5. 写一个自定义估计器，通过 `check_estimator`。

### 28.2 实践题

6. 实现一个 `MedianRegressor`，预测训练集中位数。
7. 把 `MedianRegressor` 放进 Pipeline 和 GridSearchCV，验证能工作。
8. 给 `MedianRegressor` 写算法正确性测试。
9. 读 sklearn 一个真实估计器的源码，标注每行体现哪讲约定。
10. 写一份"如果重新设计 sklearn"的提案，列出会改和不会改的。

### 28.3 思考题

11. 学完八讲，你对"架构"的理解有什么变化？
12. sklearn 的架构理念能用到非 ML 项目吗？举一个例子。
13. 如果你要给团队分享这八讲，会怎么组织？
14. 这八讲里哪讲最难理解？为什么？
15. 怎么把"以最小约定换最大自由"落实到下周的代码里？

---

## 29. 导航

[← 第七讲：全局配置与演进](07-config-and-evolution.md) ｜  [回到首页](../index.md）

### 16.1 七讲回顾

| 讲次 | 主题             | 核心洞察                                 |
|------|------------------|------------------------------------------|
| 1    | 统一 API 契约    | fit/predict 统一让通用工具成为可能       |
| 2    | Mixin 多继承     | 多重身份、职责单一                       |
| 3    | 参数管理         | `__init__` 只存参数让 clone 可靠         |
| 4    | 元估计器组合     | 约定让组合任意估计器成为可能             |
| 5    | 数据约定与校验   | 错误前置、防御式编程                     |
| 6    | 一致性测试       | 通用测试是统一契约的回报                 |
| 7    | 全局配置与演进   | 制度化演进是长期可维护的关键             |
| 8    | 架构总览         | 以最小约定换最大自由                     |

### 16.2 思考题

1. 七讲里哪讲对你启发最大？为什么？
2. 如果只能记住一条洞察，你记哪条？
3. 怎么把这条洞察用到你下周的工作里？
