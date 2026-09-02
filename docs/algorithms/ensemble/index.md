# 集成学习：随机森林

> 随机森林 = Bagging + 随机子空间 + 决策树。通过多棵树的集体智慧，克服单棵决策树容易过拟合的弱点。本章将从动机、随机性来源、算法流程、偏差-方差分析、超参数、对比 sklearn、常见陷阱等多维度，把随机森林讲透。

---

## 一、为什么需要随机森林？

单棵决策树的问题：**过拟合**。决策树会不断分裂直到每个叶子纯度最高，相当于记住了训练数据，泛化能力差。

随机森林的解法：**多棵树的集体智慧**。如果每棵树都有不同的"视角"（不同的样本、不同的特征），它们的错误方向不同，平均后错误相互抵消。

数学基础：若 $n$ 棵独立同分布的树，每棵方差 $\sigma^2$，则平均后方差 $\sigma^2/n$。实际树之间有相关性，方差降低不如理想，但仍显著。

### 1.1 偏差-方差分解

对回归问题，期望泛化误差可分解：

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

- 单棵深树：偏差低（拟合好），方差高（对数据敏感）
- 随机森林：偏差略增（每棵树见部分数据），方差大降（平均去噪）
- 净效果：泛化误差降低

### 1.2 集成学习的三大流派

| 流派 | 代表 | 思想 |
|------|------|------|
| Bagging | 随机森林 | 并行训练独立基学习器，投票/平均降方差 |
| Boosting | GBDT/AdaBoost | 串行训练，每轮纠正前轮错误，降偏差 |
| Stacking | Stacking | 用元学习器结合基学习器输出 |

随机森林属 Bagging 流派，本章重点。

### 1.3 强可学习与弱可学习

PAC 学习理论中，强可学习（任意高精度）与弱可学习（略好于随机）等价。Boosting 基于此把弱学习器提升为强学习器。Bagging 则通过降方差把不稳定学习器（如深树）变稳定。

---

## 二、三个随机性来源

### 2.1 样本随机（Bagging）

**Bootstrap Aggregating**：从训练集有放回采样生成子数据集。

```python
def _bootstrap_sample(X, y, rng):
    n = X.shape[0]
    indices = rng.randint(0, n, size=n)  # 有放回采样
    return X[indices], y[indices]
```

有放回采样的性质：约 **63.2%** 的唯一样本被选中，其余 36.8% 是**袋外样本**（OOB），可用于内部验证。

#### 2.1.1 63.2% 的推导

某样本在一次采样中未被选中的概率 $= 1 - 1/n$。$n$ 次采样都未选中：

$$
P(\text{未选中}) = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368
$$

故被选中概率 $\approx 1 - 0.368 = 0.632$。

#### 2.1.2 OOB 评估

每棵树只用了 63.2% 样本训练，剩余 36.8% 可作该树的"内部验证集"。对所有样本，用没见过它的树的预测做平均，得 OOB 分数。无需额外划分验证集，是随机森林的免费交叉验证。

```python
# OOB 评估伪代码
for i in range(n_samples):
    predictors = [tree for tree in forest if i not in tree.training_indices]
    oob_pred[i] = average(tree.predict(x_i) for tree in predictors)
oob_score = accuracy(y, oob_pred)
```

minisklearn 暂未实现 OOB，sklearn 用 `oob_score=True` 开启。

### 2.2 特征随机（随机子空间）

每次分裂只考虑部分特征，而非全部：

```python
# 分类默认：sqrt(n_features)
max_features = int(np.sqrt(n_features))

# 回归默认：n_features / 3
max_features = n_features // 3
```

作用：让不同的树关注不同的特征，增加树之间的差异性。

#### 2.2.1 为什么默认 sqrt(d)？

经验上 $\sqrt{d}$ 在偏差-方差间取得好平衡：
- 太大（=d）：每棵树都看全部特征，树之间相似，方差降低弱
- 太小（=1）：每棵树只看一个特征，树差异大但每棵太弱，偏差高
- $\sqrt{d}$ 是甜点

#### 2.2.2 max_features 的影响

| max_features | 树差异 | 方差降低 | 偏差 |
|-------------|--------|---------|------|
| n_features（全部） | 小 | 弱 | 低 |
| sqrt(n_features)（默认） | 中 | 中 | 中 |
| 1（单个） | 大 | 强 | 高 |

### 2.3 组合效果

两个随机性叠加 → 树之间相关性低 → 平均后方差降低显著。

#### 2.3.1 方差降低公式

设每棵树方差 $\sigma^2$，树之间相关性 $\rho$。$B$ 棵树平均的方差：

$$
\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
$$

- $B \to \infty$：第二项消失，方差 $\to \rho \sigma^2$
- $\rho$ 小：方差降低显著
- Bagging + 随机子空间的目的就是降 $\rho$

---

## 三、算法流程

```
训练：
  for i in range(n_estimators):
      1. Bootstrap 采样 → (X_i, y_i)
      2. 在 (X_i, y_i) 上训练决策树，分裂时随机选 max_features 个特征
      3. 存储树

预测（分类）：
  对每个样本，所有树投票，多数类为结果

预测（回归）：
  对每个样本，所有树预测值取平均
```

### 3.1 训练伪代码

```python
def fit(self, X, y):
    self.estimators_ = []
    for i in range(self.n_estimators):
        X_boot, y_boot = bootstrap_sample(X, y, rng)
        tree = DecisionTreeClassifier(max_features=self.max_features, ...)
        tree.fit(X_boot, y_boot)
        self.estimators_.append(tree)
    return self
```

### 3.2 预测伪代码

```python
def predict(self, X):
    # 收集所有树的预测
    all_preds = np.array([tree.predict(X) for tree in self.estimators_])  # (n_trees, n_samples)
    # 投票（分类）
    from scipy.stats import mode
    return mode(all_preds, axis=0).mode[0]
    # 或回归：return all_preds.mean(axis=0)
```

### 3.3 并行性

各树训练独立，可并行：

```python
from joblib import Parallel, delayed
trees = Parallel(n_jobs=-1)(
    delayed(train_one_tree)(X, y, rng) for _ in range(n_estimators)
)
```

sklearn 用 `n_jobs` 参数。minisklearn 串行，教学优先。

---

## 四、max_features 的影响

| max_features | 树差异 | 方差降低 | 偏差 |
|-------------|--------|---------|------|
| n_features（全部） | 小 | 弱 | 低 |
| sqrt(n_features)（默认） | 中 | 中 | 中 |
| 1（单个） | 大 | 强 | 高 |

- **越大**：每棵树越准（偏差低），但树之间越像（方差降低弱）
- **越小**：树差异大（方差降低强），但每棵树越不准（偏差高）
- 默认值是偏差-方差权衡的甜点

### 4.1 调优示例

```python
for mf in ['sqrt', 'log2', 0.5, 1.0, None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf)
    s = cross_val_score(rf, X, y, cv=5).mean()
    print(f"max_features={mf}: {s:.3f}")
```

---

## 五、n_estimators 的影响

树越多，方差越低，但收益递减：

```
分数
  ^
  |        ___________
  |       /
  |      /
  |     /
  |    /
  +---+--------+----> n_estimators
     50   100  200
```

- 50 棵：已获大部分收益
- 100 棵：默认，通常够
- 500+ 棵：边际收益小，但不会变差（不像过拟合），只是浪费算力

```python
for n in [10, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n).fit(X_tr, y_tr)
    print(f"n={n}: 训练={rf.score(X_tr, y_tr)}, 测试={rf.score(X_te, y_te)}")
```

### 5.1 何时停止加树

- 测试分数稳定不再提升
- 计算预算耗尽
- OOB 分数收敛

随机森林不会因树多而过拟合（最多收敛到某值），但浪费算力。

---

## 六、偏差-方差分析

### 6.1 单树 vs 森林

| | 单棵深树 | 随机森林 |
|---|---|---|
| 偏差 | 低 | 略高（每棵树见部分数据） |
| 方差 | 高（数据敏感） | 低（平均去噪） |
| 训练分数 | 1.0 | < 1.0（每棵树未见过全部数据） |
| 测试分数 | 较低 | 较高 |

### 6.2 随机森林不降低偏差

随机森林降方差但不降偏差。若单树偏差高（如树太浅），森林也偏差高。解决：保证每棵树足够深（`max_depth=None`），让偏差低，再靠集成降方差。

### 6.3 与 Boosting 对比

| | Bagging（RF） | Boosting（GBDT） |
|---|---|---|
| 目标 | 降方差 | 降偏差 |
| 基学习器 | 深树（低偏差高方差） | 浅树（高偏差低方差） |
| 训练 | 并行 | 串行 |
| 过拟合 | 不易 | 易（需学习率/早停） |

---

## 七、使用示例

### 7.1 分类

```python
from minisklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0).fit(X_tr, y_tr)
print("训练:", rf.score(X_tr, y_tr))
print("测试:", rf.score(X_te, y_te))
```

### 7.2 回归

```python
from minisklearn.ensemble import RandomForestRegressor
import numpy as np

X = np.random.randn(200, 3)
y = X @ [1, -2, 3] + np.random.randn(200) * 0.1

rf = RandomForestRegressor(n_estimators=50).fit(X, y)
print("R²:", rf.score(X, y))
```

### 7.3 与单树对比

```python
from minisklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
print(f"单树: {dt.score(X_te, y_te)}")
print(f"森林: {rf.score(X_te, y_te)}")  # 通常高 2-5%
```

---

## 八、与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 基学习器 | DecisionTree（Cython） | DecisionTree（纯 Python） |
| 并行 | n_jobs | 串行 |
| OOB | oob_score=True | 暂不支持 |
| class_weight | 支持 | 暂不支持 |
| max_features | sqrt/log2/int/float | sqrt/log2/int/float |
| 数值精度 | 一致 | ✅ |
| 速度 | 快 10-50x | 慢 |

### 8.1 数值一致性

```python
from sklearn.ensemble import RandomForestClassifier as SkRF
from minisklearn.ensemble import RandomForestClassifier as MnRF
sk = SkRF(n_estimators=50, max_depth=5, random_state=0).fit(X_tr, y_tr)
mn = MnRF(n_estimators=50, max_depth=5, random_state=0).fit(X_tr, y_tr)
print(sk.score(X_te, y_te), mn.score(X_te, y_te))  # 接近
```

### 8.2 性能对比

| n | d | trees | sklearn | minisklearn | 比值 |
|---|---|-------|---------|-------------|------|
| 1000 | 5 | 100 | 50ms | 500ms | 10x |
| 10000 | 10 | 100 | 300ms | 5000ms | 15x |
| 50000 | 20 | 200 | 2s | 30s | 15x |

sklearn 用 Cython 树 + 并行，快一个量级。

---

## 九、特征重要性

随机森林的特征重要性 = 各树特征重要性的平均：

$$
\text{importance}(j) = \frac{1}{B} \sum_{b=1}^B \text{importance}_b(j)
$$

每棵树的重要性 = 该特征在所有分裂中带来的增益总和（归一化）。

### 9.1 使用

```python
rf = RandomForestClassifier(n_estimators=100).fit(X, y)
importances = rf.feature_importances_
for j, imp in enumerate(importances):
    print(f"特征 {j}: {imp:.4f}")
```

### 9.2 置换重要性

更可靠的特征重要性：打乱某特征的值，看分数下降多少。下降多说明该特征重要。sklearn 的 `inspection.permutation_importance` 实现。

### 9.3 注意事项

- 树模型偏向高基数特征（取值多的特征看似更重要）
- 相关特征间重要性会分散（两个相关特征各占一半重要性）
- 用置换重要性更可靠

---

## 十、超参数调优

### 10.1 主要参数

| 参数 | 默认 | 作用 | 调优方向 |
|------|------|------|---------|
| n_estimators | 100 | 树数 | 增多到分数稳定 |
| max_depth | None | 树深 | 过拟合时减小 |
| max_features | sqrt | 分裂特征数 | 试 sqrt/log2 |
| min_samples_leaf | 1 | 叶子最小样本 | 过拟合时增大 |
| bootstrap | True | 是否 bootstrap | False 则用全数据 |

### 10.2 调优策略

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5).fit(X, y)
print(gs.best_params_)
```

### 10.3 经验法则

- 先调 n_estimators 到分数稳定（通常 100-200）
- 再调 max_depth（过拟合减，欠拟合增）
- max_features 通常用默认 sqrt
- min_samples_leaf 增大可防过拟合
- 不必精调，随机森林对超参数鲁棒

---

## 十一、复杂度分析

### 11.1 训练

- 单树：$O(h \cdot d' n \log n)$，$d'$ = max_features
- $B$ 棵树：$O(B \cdot h \cdot d' n \log n)$
- 并行 $P$ 核：$O(B \cdot h \cdot d' n \log n / P)$

### 11.2 预测

- 单样本：$O(B \cdot h)$（每棵树 $O(h)$）
- 批量 $m$：$O(m B h)$

### 11.3 空间

- $B$ 棵树，每棵 $O(\text{叶子数})$
- 总 $O(B \cdot \text{叶子数})$

树多则模型大，预测慢。生产中常限制 $B \leq 500$。

---

## 十二、几何直觉

### 12.1 决策边界

随机森林的决策边界是各树边界的"投票平均"。单树边界是轴对齐阶梯，森林边界是多个阶梯的叠加，更平滑。

```
单树边界（锯齿）:     森林边界（平滑）:
  |  □  □              |  □□□
  | □□ □□              | □□□□□
  |  □  □              |  □□□
```

### 12.2 多树投票的几何意义

每个样本点被 $B$ 棵树分类，每棵树把它划入某叶子的多数类。森林取 $B$ 个投票的多数。几何上，这是在"多个轴对齐划分"上做软投票，等价于一个更复杂的边界。

---

## 十三、数值稳定性

### 13.1 随机种子

随机森林有随机性（bootstrap + max_features），需 `random_state` 保证可复现：

```python
rf1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
assert np.array_equal(rf1.predict(X), rf2.predict(X))  # 一致
```

### 13.2 平票处理

投票平票时 `mode` 取第一个（小类）。可加随机扰动或用软投票（predict_proba 平均）。

### 13.3 树数与稳定性

树少时分数方差大（随机性影响大）。树多时稳定。100 棵通常足够稳定。

---

## 十四、常见问题与陷阱

| 问题 | 现象 | 解决 |
|------|------|------|
| 树太少 | 分数不稳但低 | 增 n_estimators |
| 过拟合 | 训练 1.0 测试低 | 减 max_depth 或增 min_samples_leaf |
| 训练慢 | 树多/深 | 减 n_estimators 或 max_depth |
| 预测慢 | 树多 | 减 n_estimators 或用 sklearn |
| 不可复现 | 分数变 | 设 random_state |
| 类别不平衡 | 多数类主导 | class_weight='balanced' |
| 高维稀疏 | 性能差 | 降维或换线性模型 |
| 特征重要性偏 | 高基数特征虚高 | 用置换重要性 |

### 14.1 调试技巧

```python
# 检查树数是否够
for n in [10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n).fit(X_tr, y_tr)
    print(f"n={n}: {rf.score(X_te, y_te)}")
# 分数稳定后即够

# 检查过拟合
rf = RandomForestClassifier().fit(X_tr, y_tr)  # 不限深
print(f"训练={rf.score(X_tr, y_tr)}, 测试={rf.score(X_te, y_te)}")
# 训练 1.0 但测试不低 → RF 抗过拟合，OK
```

---

## 十五、实战教程

### 15.1 端到端分类

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine
from minisklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
cv = cross_val_score(rf, X_tr, y_tr, cv=5)
print(f"CV: {cv.mean():.3f} ± {cv.std():.3f}")
rf.fit(X_tr, y_tr)
print(f"测试: {rf.score(X_te, y_te):.3f}")
```

### 15.2 调 n_estimators

```python
for n in [10, 30, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=0)
    s = cross_val_score(rf, X_tr, y_tr, cv=5).mean()
    print(f"n={n}: {s:.3f}")
```

### 15.3 调 max_depth

```python
for d in [3, 5, 10, 20, None]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=0)
    s = cross_val_score(rf, X_tr, y_tr, cv=5).mean()
    print(f"depth={d}: {s:.3f}")
```

### 15.4 特征重要性

```python
rf = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
for j, imp in enumerate(rf.feature_importances_):
    print(f"特征 {j}: {imp:.4f}")
```

### 15.5 与单树对比

```python
from minisklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_tr, y_tr)
print(f"单树: 训练={dt.score(X_tr, y_tr):.3f} 测试={dt.score(X_te, y_te):.3f}")
print(f"森林: 训练={rf.score(X_tr, y_tr):.3f} 测试={rf.score(X_te, y_te):.3f}")
```

---

## 十六、进阶话题

### 16.1 Extra Trees

Extremely Randomized Trees：分裂阈值也随机选（不从候选中找最优），进一步增加随机性。sklearn 的 `ExtraTreesClassifier`。通常与 RF 性能接近，训练更快（不找最优阈值）。

### 16.2 Isolation Forest

用随机森林的随机划分做异常检测。异常点路径短（易被孤立）。sklearn 的 `IsolationForest`。

### 16.3 GBDT

梯度提升树：串行训练，每轮拟合前轮的残差（负梯度）。比随机森林通常更准，但易过拟合，需学习率/早停。代表：XGBoost、LightGBM、CatBoost。

### 16.4 Stacking

用元学习器结合多个基学习器输出：

```python
from sklearn.ensemble import StackingClassifier
estimators = [('rf', RandomForestClassifier()), ('lr', LogisticRegression())]
stack = StackingClassifier(estimators, final_estimator=LogisticRegression())
```

### 16.5 AdaBoost

给错分样本加权，串行训练弱学习器。历史上第一个成功的 Boosting 算法。

---

## 十七、数学补充

### 17.1 Bagging 的方差降低

设 $B$ 个独立同分布预测器 $\hat{f}_1, \ldots, \hat{f}_B$，各自方差 $\sigma^2$。平均 $\bar{f} = \frac{1}{B}\sum \hat{f}_b$：

$$
\text{Var}(\bar{f}) = \frac{\sigma^2}{B}
$$

$B \to \infty$ 时方差 $\to 0$。但树之间有相关性 $\rho$：

$$
\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
$$

$\rho$ 由随机子空间降低。

### 17.2 投票的理论

分类投票是多数原则。由 Condorcet 陪审团定理：若每棵树正确率 $p > 0.5$ 且独立，则 $B \to \infty$ 时投票正确率 $\to 1$。实际树相关，但仍显著提升。

### 17.3 随机森林的一致性

随机森林在 $B \to \infty$ 且 $n \to \infty$ 时一致收敛到某极限（取决于 max_features、max_depth）。理论复杂，实践中 $B=100$ 通常够。

---

## 十八、与 sklearn 详细对比

### 18.1 功能对比

| 功能 | sklearn | minisklearn |
|------|---------|-------------|
| n_jobs 并行 | 支持 | 串行 |
| oob_score | 支持 | 暂不支持 |
| class_weight | 支持 | 暂不支持 |
| warm_start | 支持（增量加树） | 暂不支持 |
| max_samples | bootstrap 样本数 | 暂不支持 |
| ccp_alpha 后剪枝 | 支持 | 暂不支持 |
| feature_importances_ | 支持 | 暂不支持 |

### 18.2 数值一致性测试

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRF
from minisklearn.ensemble import RandomForestClassifier as MnRF
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
sk = SkRF(n_estimators=50, max_depth=5, random_state=0).fit(X, y)
mn = MnRF(n_estimators=50, max_depth=5, random_state=0).fit(X, y)
print(f"sklearn: {sk.score(X, y):.4f}")
print(f"minisklearn: {mn.score(X, y):.4f}")
# 应非常接近（随机种子相同，但实现细节可能略异）
```

### 18.3 性能对比

| 场景 | sklearn | minisklearn |
|------|---------|-------------|
| 训练 100 棵树 (1000 样本) | 50ms | 500ms |
| 训练 100 棵树 (10000 样本) | 300ms | 5000ms |
| 预测 1000 样本 | 5ms | 50ms |

minisklearn 慢 10-15x，因纯 Python 树 + 串行。教学用足够。

---

## 十九、超参数调优指南

### 19.1 调优顺序

1. 先固定 n_estimators=100，调 max_depth
2. 再调 max_features
3. 最后增 n_estimators 到分数稳定

### 19.2 网格搜索

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 5],
}
gs = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5).fit(X, y)
print(gs.best_params_, gs.best_score_)
```

### 19.3 经验

- 随机森林对超参数鲁棒，默认参数通常不错
- 不必精调，调一两个参数即可
- 树多不会过拟合，只是慢，可放心增大
- max_depth=None + min_samples_leaf=1 让每棵树充分拟合，靠集成防过拟合

---

## 二十、生产环境注意事项

### 20.1 模型大小

随机森林模型 = $B$ 棵树。树多则模型大，序列化/部署成本高。生产中常限制 $B \leq 200$。

### 20.2 预测延迟

预测需遍历 $B$ 棵树，延迟与 $B$ 成正比。延迟敏感场景减小 $B$ 或用 sklearn 并行。

### 20.3 在线学习

随机森林不支持在线（新数据需重训）。`warm_start=True` 可增量加树，但不能更新已有树。需在线用 Hoeffding 森林。

### 20.4 可解释性

随机森林可解释性低于单树（多树投票难直视）。用特征重要性 + 单树可视化近似解释。SHAP 值可更精确解释。

---

## 二十一、完整实战教程

### 21.1 端到端分类流水线

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_wine
from minisklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
}
gs = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5).fit(X_tr, y_tr)
print(f"最优参数: {gs.best_params_}")
print(f"CV 分数: {gs.best_score_:.3f}")
print(f"测试分数: {gs.score(X_te, y_te):.3f}")
```

### 21.2 回归示例

```python
import numpy as np
from minisklearn.ensemble import RandomForestRegressor

# 拟合非线性函数
np.random.seed(0)
X = np.random.randn(500, 2)
y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(500) * 0.1

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0).fit(X_tr, y_tr)
print(f"训练 R²: {rf.score(X_tr, y_tr):.3f}")
print(f"测试 R²: {rf.score(X_te, y_te):.3f}")
```

### 21.3 与单树、线性模型对比

```python
from minisklearn.tree import DecisionTreeClassifier
from minisklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 非线性数据
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

dt = DecisionTreeClassifier(max_depth=5).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
lr = LogisticRegression().fit(X_tr_s, y_tr)

print(f"单树: {dt.score(X_te, y_te):.3f}")
print(f"森林: {rf.score(X_te, y_te):.3f}")
print(f"线性: {lr.score(X_te_s, y_te):.3f}")
# 森林通常最高，线性因数据非线性而低
```

### 21.4 特征重要性分析

```python
rf = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for j in indices:
    print(f"特征 {j}: {importances[j]:.4f}")
```

### 21.5 树数收敛曲线

```python
import numpy as np
ns = [10, 20, 50, 100, 200, 500]
scores = []
for n in ns:
    rf = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_tr, y_tr)
    scores.append(rf.score(X_te, y_te))
    print(f"n={n}: {scores[-1]:.4f}")
# 分数随 n 增加收敛
```

---

## 二十二、随机森林的变体

### 22.1 Extra Trees（极随机树）

分裂阈值随机选（不从候选中找最优），进一步增加随机性，训练更快：

```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100).fit(X_tr, y_tr)
```

通常与 RF 性能接近，某些数据集上更优。

### 22.2 Random Patches

同时 bootstrap 样本和特征（不只是分裂时选特征，整棵树只见部分特征）。sklearn 的 `BaggingClassifier` 可配置。

### 22.3 Isolation Forest

用随机划分做异常检测，异常点路径短：

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=100).fit(X)
scores = iso.score_samples(X)  # 越低越异常
```

### 22.4 Random Forest Embedding

用叶节点索引作为特征嵌入，每棵树把样本映射到叶节点 one-hot。可用于无监督特征学习。

---

## 二十三、与 Boosting 对比深入

### 23.1 训练方式

| | Bagging (RF) | Boosting (GBDT) |
|---|---|---|
| 训练 | 并行 | 串行 |
| 基学习器 | 深树（低偏差） | 浅树（高偏差） |
| 目标 | 降方差 | 降偏差 |
| 加权 | 等权 | 后轮纠错加权 |

### 23.2 性能

- RF：准确、鲁棒、易调参
- GBDT：通常更准（尤其结构化数据），但易过拟合，需精调

### 23.3 何时用哪个

- 数据少、初学、要稳定：RF
- 数据多、要极致准确、肯调参：GBDT
- 高维稀疏：RF 通常更好
- 结构化表格：GBDT（XGBoost/LightGBM）通常更好

---

## 二十四、常见问题汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| 树太少 | 分数低/不稳 | 增 n_estimators |
| 过拟合 | 训练远高于测试 | 减 max_depth |
| 训练慢 | 树多/深 | 减 n_estimators 或 max_depth |
| 预测慢 | 树多 | 减 n_estimators |
| 不可复现 | 分数变 | 设 random_state |
| 不平衡 | 多数类主导 | class_weight='balanced' |
| 高维稀疏 | 性能差 | 降维 |
| 特征重要性偏 | 高基数虚高 | 用置换重要性 |
| 模型大 | 序列化大 | 减 n_estimators |
| 在线需求 | 不支持 | 用 Hoeffding 森林 |

### 24.1 学习曲线诊断

```python
import numpy as np
sizes = np.linspace(0.1, 1.0, 10)
for s in sizes:
    n = int(s * len(X_tr))
    rf = RandomForestClassifier(n_estimators=100).fit(X_tr[:n], y_tr[:n])
    tr = rf.score(X_tr[:n], y_tr[:n])
    te = rf.score(X_te, y_te)
    print(f"n={n}: 训练={tr:.3f} 测试={te:.3f}")
```

---

## 架构回扣

随机森林是**元估计器**（第四讲）——它包装了 `DecisionTreeClassifier` 作为基学习器：

```python
class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_features=..., ...)
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
```

这体现了"组合优于继承"：随机森林**组合**了多棵决策树，而非继承决策树。

`estimators_` 以列表形式存储多棵树，`get_params` / `clone` 自动处理（因为 `n_estimators` 是 `__init__` 参数，而 `estimators_` 是 fit 后属性）。

### 类层级

```
BaseEstimator
   ├── RandomForestClassifier + ClassifierMixin
   └── RandomForestRegressor  + RegressorMixin
        (组合 DecisionTreeClassifier/Regressor 作为基学习器)
```

### fit 后属性

| 算法 | 属性 | 含义 |
|------|------|------|
| RandomForestClassifier | `estimators_`, `classes_`, `n_classes_` | 树列表、类别、类别数 |
| RandomForestRegressor | `estimators_` | 树列表 |

`estimators_` 是 fit 后属性（单下划线结尾），存储 $B$ 棵决策树实例。

### 设计哲学

- **元估计器模式**：随机森林不直接建模，而是组合多个基学习器
- **组合优于继承**：不继承 DecisionTree，而是持有一组 DecisionTree 实例
- **随机性注入**：通过 `random_state` 控制随机性，保证可复现
- **并行友好**：各树独立，天然可并行（虽然 minisklearn 串行）

### 与基学习器的契约

随机森林复用 `DecisionTreeClassifier.fit` / `predict` 接口。基学习器只需遵循 sklearn 估计器 API，即可被随机森林组合。这是元估计器模式的威力——任意估计器都可作为基学习器。

### 与预处理器的契约

随机森林（基于决策树）不需要特征缩放。但类别特征仍需 OneHot 编码。

### 与单树的对比

| | 单树 | 随机森林 |
|---|---|---|
| 过拟合 | 易 | 难 |
| 准确率 | 中 | 高 |
| 训练 | 快 | 慢 |
| 预测 | 快 | 慢 |
| 可解释性 | 高 | 中 |
| 超参数 | 少 | 多 |

### 进一步学习

- sklearn 的 `ExtraTreesClassifier` 极随机树
- XGBoost / LightGBM / CatBoost 梯度提升树
- Isolation Forest 异常检测
- SHAP 值解释随机森林
- Hoeffding 森林在线学习

### 总结

随机森林通过 Bagging + 随机子空间 + 决策树的组合，用多棵树的集体智慧克服单树过拟合。它对超参数鲁棒、不需缩放、准确率高，是机器学习的"瑞士军刀"。理解随机森林有助掌握集成学习的核心思想：通过组合降低方差。

### 关键要点回顾

1. **三个随机性**：bootstrap 样本、随机特征、（可选）随机阈值
2. **降方差**：树之间相关性低，平均后方差大降
3. **不降偏差**：每棵树需足够深保证低偏差
4. **树多无害**：不会过拟合，只是浪费算力
5. **OOB 评估**：免费内部验证，无需划分验证集
6. **特征重要性**：分裂增益累计，注意高基数偏置
7. **元估计器**：组合多棵树，体现"组合优于继承"
8. **不需缩放**：基于决策树，对特征尺度不敏感
9. **可并行**：各树独立，sklearn 用 n_jobs 加速
10. **可复现**：设 random_state 保证结果一致
11. **复杂度**：训练 $O(B \cdot h \cdot d' n \log n)$，预测 $O(B h)$
12. **抗过拟合**：训练 1.0 不必担心，看测试分数即可
13. **vs Boosting**：Bagging 降方差，Boosting 降偏差
14. **vs 单树**：通常准确率高 2-5%，但可解释性降低
15. **生产部署**：模型大、预测慢，需权衡树数
16. **OOB 计算**：每棵树约 36.8% 样本未见过，可做内部验证
17. **bootstrap 推导**：$(1-1/n)^n \to e^{-1} \approx 0.368$
18. **方差公式**：$\text{Var}(\bar{f}) = \rho \sigma^2 + (1-\rho)\sigma^2/B$
19. **Condorcet 定理**：独立树正确率 $p>0.5$ 时投票随 $B$ 增高趋 1
20. **Extra Trees**：阈值也随机，训练更快，性能相近
21. **Isolation Forest**：用随机划分做异常检测
22. **Stacking**：元学习器结合基学习器，比 RF 更复杂
23. **warm_start**：增量加树，sklearn 支持
24. **max_samples**：控制 bootstrap 样本数，sklearn 支持
25. **SHAP 值**：精确解释随机森林预测，比特征重要性更细

---

## 二十五、深入数学推导与证明

### 25.1 Bagging 方差降低的完整推导

**定理**：设 $B$ 个预测器 $\hat{f}_1, \ldots, \hat{f}_B$ 同分布，各自方差 $\sigma^2$，两两相关性 $\rho$（即 $\text{Cov}(\hat{f}_i, \hat{f}_j) = \rho \sigma^2$ for $i \neq j$）。则平均 $\bar{f} = \frac{1}{B}\sum_{b=1}^B \hat{f}_b$ 的方差为：

$$
\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
$$

**证明**：

$$
\text{Var}(\bar{f}) = \text{Var}\left(\frac{1}{B}\sum_b \hat{f}_b\right) = \frac{1}{B^2} \text{Var}\left(\sum_b \hat{f}_b\right)
$$

$$
= \frac{1}{B^2} \left[ \sum_b \text{Var}(\hat{f}_b) + \sum_{i \neq j} \text{Cov}(\hat{f}_i, \hat{f}_j) \right]
$$

$$
= \frac{1}{B^2} \left[ B \sigma^2 + B(B-1) \rho \sigma^2 \right] = \frac{\sigma^2}{B} + \frac{(B-1)\rho \sigma^2}{B}
$$

$$
= \rho \sigma^2 + \frac{(1-\rho)\sigma^2}{B} \quad \square
$$

**推论**：
- $B \to \infty$：$\text{Var}(\bar{f}) \to \rho \sigma^2$。方差降低受限于相关性 $\rho$。
- $\rho = 0$（独立）：$\text{Var}(\bar{f}) = \sigma^2/B \to 0$。
- $\rho = 1$（完全相关）：$\text{Var}(\bar{f}) = \sigma^2$，无降低。

**随机森林的设计目的**：通过 bootstrap + 随机子空间降低 $\rho$。

### 25.2 Bootstrap 采样 63.2% 的推导

**定理**：从 $n$ 个样本有放回采样 $n$ 次，某固定样本未被选中的概率趋近 $e^{-1} \approx 0.368$。

**证明**：

某样本单次未被选中概率 $= 1 - 1/n$。$n$ 次独立采样都未选中：

$$
P(\text{未选中}) = \left(1 - \frac{1}{n}\right)^n
$$

取极限：

$$
\lim_{n \to \infty} \left(1 - \frac{1}{n}\right)^n = e^{-1} \approx 0.36788
$$

故被选中概率 $\approx 1 - 0.368 = 0.632$。$\square$

### 25.3 Condorcet 陪审团定理

**定理**：若 $B$ 个分类器独立投票，每个正确率 $p > 0.5$，则多数投票的正确率随 $B \to \infty$ 趋近 1。

**证明**：

多数投票正确 = 至少 $\lceil B/2 \rceil$ 个分类器正确。设 $X$ = 正确分类器数，$X \sim \text{Binomial}(B, p)$。

由大数定律，$X/B \xrightarrow{p} p > 0.5$，故 $P(X > B/2) \to 1$。$\square$

**实际限制**：随机森林的树不独立（共享训练集），但相关性低，仍显著提升。

### 25.4 随机森林的泛化误差界

**定理**（Breiman, 2001）：随机森林的泛化误差 $PE^*$ 满足：

$$
PE^* \leq \bar{\rho} \frac{(1-s^2)}{s^2}
$$

其中 $s$ 是单棵树的"强度"（margin 相关），$\bar{\rho}$ 是树间平均相关性。

**含义**：树越强（$s$ 大）、相关性越低（$\bar{\rho}$ 小），泛化越好。这指导了随机森林的设计——强树 + 降相关。

### 25.5 OOB 估计的无偏性

**定理**：OOB 估计是泛化误差的无偏估计。

**证明**：

对样本 $i$，只用未见过 $i$ 的树预测。这些树的训练集不含 $i$，故预测与 $i$ 独立，OOB 估计模拟了测试集场景。每棵树约 36.8% 样本未见过，OOB 用这些"内部测试集"评估，无偏。$\square$

---

## 二十六、更多代码示例与对比实验

### 26.1 树数收敛曲线

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from minisklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print("n_estimators | 训练分数 | 测试分数 | CV 分数")
print("-" * 55)
for n in [1, 5, 10, 20, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_tr, y_tr)
    tr = rf.score(X_tr, y_tr)
    te = rf.score(X_te, y_te)
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=n, random_state=0), X_tr, y_tr, cv=5
    ).mean()
    print(f"{n:12d} | {tr:.4f}   | {te:.4f}   | {cv:.4f}")
# 测试分数随 n 增加收敛
```

### 26.2 max_features 对比

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for mf in [1, 5, 10, 'sqrt', 'log2', 0.5, None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=0).fit(X_tr, y_tr)
    te = rf.score(X_te, y_te)
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_features=mf, random_state=0), X_tr, y_tr, cv=5
    ).mean()
    print(f"max_features={str(mf):8s}: 测试={te:.4f}, CV={cv:.4f}")
```

### 26.3 随机森林 vs 单树 vs GBDT

```python
from minisklearn.tree import DecisionTreeClassifier

models = {
    '单树 (depth=5)': DecisionTreeClassifier(max_depth=5),
    '单树 (不限深)': DecisionTreeClassifier(),
    'RF (50树)': RandomForestClassifier(n_estimators=50, random_state=0),
    'RF (200树)': RandomForestClassifier(n_estimators=200, random_state=0),
    'RF (depth=5)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 26.4 特征重要性分析

```python
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征排名:")
for rank, idx in enumerate(indices[:10]):
    print(f"  {rank+1}. 特征{idx}: {importances[idx]:.4f} {'█' * int(importances[idx]*200)}")

# 对比置换重要性（更可靠）
# from sklearn.inspection import permutation_importance
# result = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)
```

### 26.5 OOB 评估（sklearn）

```python
from sklearn.ensemble import RandomForestClassifier as SkRF

rf = SkRF(n_estimators=100, oob_score=True, random_state=0).fit(X_tr, y_tr)
print(f"OOB 分数: {rf.oob_score_:.4f}")
print(f"测试分数: {rf.score(X_te, y_te):.4f}")
# OOB 分数应接近测试分数
```

### 26.6 不同随机种子的稳定性

```python
scores = []
for rs in range(20):
    rf = RandomForestClassifier(n_estimators=100, random_state=rs).fit(X_tr, y_tr)
    scores.append(rf.score(X_te, y_te))

print(f"20 次运行: 均值={np.mean(scores):.4f}, 标准差={np.std(scores):.4f}")
print(f"范围: [{min(scores):.4f}, {max(scores):.4f}]")
# 树多时标准差小（稳定）
```

---

## 二十七、参数调优进阶指南

### 27.1 系统调优流程

```python
from sklearn.model_selection import GridSearchCV

# 第一步：固定 n_estimators=100，调 max_depth
param_grid_1 = {'max_depth': [3, 5, 10, 20, None]}
gs1 = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=0),
    param_grid_1, cv=5
).fit(X_tr, y_tr)
print(f"第一步最优 max_depth: {gs1.best_params_}")

# 第二步：调 max_features
param_grid_2 = {'max_features': ['sqrt', 'log2', 0.3, 0.5, None]}
gs2 = GridSearchCV(
    RandomForestClassifier(n_estimators=100, max_depth=gs1.best_params_['max_depth'], random_state=0),
    param_grid_2, cv=5
).fit(X_tr, y_tr)
print(f"第二步最优 max_features: {gs2.best_params_}")

# 第三步：增大 n_estimators 到收敛
for n in [100, 200, 500]:
    rf = RandomForestClassifier(
        n_estimators=n, max_depth=gs1.best_params_['max_depth'],
        max_features=gs2.best_params_['max_features'], random_state=0
    )
    s = cross_val_score(rf, X_tr, y_tr, cv=5).mean()
    print(f"n={n}: CV={s:.4f}")
```

### 27.2 联合网格搜索

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 5],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=0), param_grid, cv=5, n_jobs=-1
).fit(X_tr, y_tr)
print(f"最优参数: {gs.best_params_}")
print(f"最优分数: {gs.best_score_:.4f}")
```

### 27.3 调优经验法则

| 场景 | n_estimators | max_depth | max_features | min_samples_leaf |
|------|-------------|-----------|-------------|-----------------|
| 默认 | 100 | None | sqrt | 1 |
| 过拟合 | 100 | 减小 | sqrt | 增大 |
| 欠拟合 | 增大 | None | 增大 | 1 |
| 高维 | 200 | None | sqrt | 1 |
| 小数据 | 100 | 减小 | sqrt | 增大 |
| 快速原型 | 50 | 10 | sqrt | 1 |

### 27.4 随机搜索 vs 网格搜索

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, 20, None],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_leaf': [1, 2, 5, 10],
}
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=0), param_dist,
    n_iter=20, cv=5, random_state=0
).fit(X_tr, y_tr)
print(f"随机搜索最优: {rs.best_score_:.4f}")
# 参数空间大时随机搜索比网格搜索高效
```

---

## 二十八、常见错误与调试技巧

### 28.1 典型错误清单

```python
# 错误 1：树太少导致不稳定
rf = RandomForestClassifier(n_estimators=5).fit(X_tr, y_tr)
# 分数随机性大，不可靠

# 错误 2：忘记 random_state
rf1 = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
rf2 = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
# rf1.predict(X) 可能 != rf2.predict(X)

# 错误 3：用回归树做分类（或反之）
# from minisklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor().fit(X_tr, y_tr)  # y_tr 是类别标签

# 错误 4：max_features 过小导致欠拟合
rf = RandomForestClassifier(n_estimators=100, max_features=1).fit(X_tr, y_tr)
# 每棵树只看 1 个特征，太弱

# 错误 5：类别特征未编码
# RF 不处理类别特征，需先 OneHot
```

### 28.2 调试检查清单

```python
def debug_random_forest(rf, X_tr, y_tr, X_te, y_te):
    """随机森林调试。"""
    print("=== 随机森林调试 ===")
    print(f"n_estimators={rf.n_estimators}, max_depth={rf.max_depth}")
    print(f"max_features={rf.max_features}")
    
    tr = rf.score(X_tr, y_tr)
    te = rf.score(X_te, y_te)
    print(f"训练: {tr:.4f}, 测试: {te:.4f}")
    
    if tr == 1.0 and te < tr - 0.1:
        print("⚠ 训练 1.0 但测试低")
        print("  → RF 抗过拟合，但若 gap 大可减 max_depth")
    
    # 检查树数稳定性
    scores = []
    for rs in range(5):
        rf_temp = RandomForestClassifier(
            n_estimators=rf.n_estimators, random_state=rs
        ).fit(X_tr, y_tr)
        scores.append(rf_temp.score(X_te, y_te))
    print(f"5 次运行: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    if np.std(scores) > 0.02:
        print("⚠ 不稳定，增大 n_estimators")
```

---

## 二十九、与其他算法的深入对比

### 29.1 Bagging vs Boosting 理论对比

| 维度 | Bagging (RF) | Boosting (GBDT) |
|------|-------------|-----------------|
| 训练 | 并行 | 串行 |
| 基学习器 | 深树（低偏差高方差） | 浅树（高偏差低方差） |
| 目标 | 降方差 | 降偏差 |
| 过拟合 | 不易 | 易（需学习率/早停） |
| 异常值 | 鲁棒（投票） | 敏感（指数加权） |
| 超参数 | 少（n_estimators, max_depth） | 多（learning_rate, n_estimators, ...） |
| 典型实现 | RandomForest | XGBoost, LightGBM |

### 29.2 随机森林 vs GBDT 实验对比

```python
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=0).fit(X_tr, y_tr)

print(f"随机森林: {rf.score(X_te, y_te):.4f}")
print(f"GBDT:     {gb.score(X_te, y_te):.4f}")
# GBDT 通常略优，但需精调 learning_rate
```

### 29.3 随机森林 vs 线性模型

```python
from minisklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 线性数据：线性模型更优
X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                           n_redundant=0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
lr = LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(X_tr), y_tr)

print(f"线性数据 - RF: {rf.score(X_te, y_te):.4f}")
print(f"线性数据 - LR: {lr.score(StandardScaler().fit_transform(X_te), y_te):.4f}")
```

---

## 三十、实际应用场景详解

### 30.1 特征重要性做特征选择

```python
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
importances = rf.feature_importances_

# 选 top-k 特征
k = 10
top_features = np.argsort(importances)[::-1][:k]
X_tr_selected = X_tr[:, top_features]
X_te_selected = X_te[:, top_features]

rf_selected = RandomForestClassifier(n_estimators=100).fit(X_tr_selected, y_tr)
print(f"全特征: {rf.score(X_te, y_te):.4f}")
print(f"top-{k} 特征: {rf_selected.score(X_te_selected, y_te):.4f}")
```

### 30.2 缺失值处理

```python
# 随机森林可处理部分缺失（sklearn 的 RandomForestClassifier 有限支持）
# 通常用中位数/众数填补后再 RF
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(n_estimators=100)),
])
```

### 30.3 异常检测（Isolation Forest）

```python
from sklearn.ensemble import IsolationForest

np.random.seed(0)
X_normal = np.random.randn(1000, 5)
X_outlier = np.random.uniform(10, 20, size=(50, 5))
X = np.vstack([X_normal, X_outlier])

iso = IsolationForest(n_estimators=100, random_state=0).fit(X)
predictions = iso.predict(X)  # 1=正常, -1=异常
print(f"检出异常数: {(predictions == -1).sum()}")
```

### 30.4 推荐系统中的 RF

```python
# 用 RF 预测用户是否会点击
# 特征：[用户年龄, 性别, 历史点击率, 物品类别, 物品热度, ...]
# 标签：是否点击
# RF 能捕捉特征间的非线性交互
```

---

## 三十一、思考题与练习

### 基础题

1. **为什么随机森林不易过拟合？**
   <details><summary>答案</summary>
   多树平均降方差，即使单树过拟合，平均后错误抵消。
   </details>

2. **bootstrap 采样为什么用有放回？**
   <details><summary>答案</summary>
   有放回使每棵树见不同样本子集（约 63.2%），增加树间差异；同时 OOB 样本可做内部验证。
   </details>

3. **为什么 max_features 默认 sqrt(d)？**
   <details><summary>答案</summary>
   平衡树间差异（降相关）和单树强度（保偏差），sqrt(d) 是经验甜点。
   </details>

### 中级题

4. **推导 Bagging 方差降低公式。**
5. **解释 OOB 评估为何无偏。**
6. **分析 max_features 对偏差-方差的影响。**

### 高级题

7. **证明 Condorcet 陪审团定理。**
8. **分析 Breiman 泛化误差界。**
9. **比较 RF 与 GBDT 的偏差-方差分解。**

### 编程练习

10. **实现 OOB 评估。**
11. **实现并行训练随机森林（用 joblib）。**
12. **实现 Extra Trees（分裂阈值随机选）。**
13. **用 RF 做特征选择 pipeline。**
14. **比较 RF、GBDT、XGBoost 在多个数据集上的表现。**

---

## 三十二、扩展阅读

### 32.1 经典论文

- **Breiman (2001)**：*Random Forests*——随机森林奠基论文
- **Breiman (1996)**：*Bagging Predictors*——Bagging 理论
- **Ho (1998)**：*The Random Subspace Method*——随机子空间
- **Amit & Geman (1997)**：随机化决策树

### 32.2 教材章节

- *The Elements of Statistical Learning* 第 15 章——随机森林
- *Pattern Recognition and Machine Learning*——决策树与集成

### 32.3 进阶主题

- **XGBoost** / **LightGBM** / **CatBoost**：梯度提升树工业实现
- **Stacking**：元学习器结合基学习器
- **SHAP 值**：精确解释随机森林预测
- **Hoeffding 森林**：在线随机森林
- **分布式随机森林**：Spark MLlib

### 32.4 相关算法

- **AdaBoost**：历史上第一个成功 Boosting
- **GBDT**：梯度提升决策树
- **Extra Trees**：极随机树
- **Isolation Forest**：异常检测
- **Random Forest Embedding**：无监督特征学习

---

[← 返回算法列表](../index.md)
