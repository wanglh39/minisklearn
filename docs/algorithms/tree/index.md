# 决策树：CART 分类与回归

> 决策树是最直观的机器学习算法——用一系列 if-else 规则对特征空间做递归划分。CART（Classification and Regression Trees）是其中最经典的实现。本章将从算法原理、分裂准则、几何直觉、剪枝、复杂度、对比 sklearn、常见陷阱等多维度，把决策树讲透。

---

## 一、CART 算法原理

### 1.1 核心思想

递归二分特征空间：

1. 在当前节点的样本中寻找最佳分裂（特征 + 阈值）
2. 按分裂将样本分为左右子集
3. 对左右子集递归建树
4. 达到停止条件时创建叶子节点

```
                特征1 <= 2.5?
               /            \
           是 /                \ 否
             /                  \
      特征2 <= 1.5?          [类别 B]
     /          \
  是 /            \ 否
   /              \
[类别 A]      [类别 B]
```

#### 1.1.1 几何直觉

每次分裂沿一个特征轴切一刀，把空间分成两个超矩形。递归分裂后，整个空间被划分成若干轴对齐的矩形区域，每个区域对应一个叶子，叶子内预测恒定（分类为多数类，回归为均值）。

```
特征2
  ^
  |    |  A  |  B  |
  |    |     |     |
  |----+-----+-----|
  |    |     |  B  |
  |  A |  B  |     |
  +----+-----+-----+---> 特征1
```

决策边界是轴对齐的阶梯状，与线性模型的光滑边界形成对比。

#### 1.1.2 与线性模型的对比

| | 决策树 | 线性模型 |
|---|---|---|
| 决策边界 | 轴对齐阶梯 | 光滑超平面 |
| 可解释性 | 极高（if-else 规则） | 中（看权重） |
| 特征缩放 | 不需要 | 需要 |
| 非线性 | 是 | 否（除非特征工程） |
| 外推 | 无（叶子值固定） | 有（线性外推） |

### 1.2 分裂选择

对每个特征的每个可能阈值，计算分裂后的加权纯度：

$$
\text{impurity\_split} = \frac{n_L}{n} \text{impurity}(L) + \frac{n_R}{n} \text{impurity}(R)
$$

选择使 `impurity_split` 最小的分裂。

#### 1.2.1 信息增益

等价地，最大化信息增益：

$$
\text{Gain} = \text{impurity}(D) - \text{impurity\_split}
$$

由于 $\text{impurity}(D)$ 与分裂无关，最小化 `impurity_split` 等价于最大化 `Gain`。

#### 1.2.2 为什么二分？

CART 恒二分（每节点恰好两个子节点）。多分（如 ID3/C4.5 对类别特征多路分裂）会让树变宽、数据碎片化快。二分可通过连续阈值分裂模拟多分，且更通用（连续特征天然二分）。

### 1.3 阈值选择优化

对特征排序后，只在**相邻不同值的中点**处尝试分裂：

```python
for i in range(n - 1):
    if sorted_values[i] == sorted_values[i + 1]:
        continue  # 相同值之间不分裂
    threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
```

相同值之间分裂无意义（分不开任何样本）。

#### 1.3.1 候选阈值数量

对 $n$ 个样本，某特征最多有 $n-1$ 个候选阈值（去重后更少）。每个阈值评估 $O(n)$（统计左右子集纯度），总 $O(n^2)$ 每特征，$O(d n^2)$ 每节点。这是朴素实现的复杂度。

#### 1.3.2 排序优化

先按特征排序 $O(n \log n)$，然后线性扫描 $O(n)$ 累积左右子集统计量，总 $O(n \log n)$ 每特征。sklearn 用此优化。minisklearn 朴素实现 $O(n^2)$，教学优先。

#### 1.3.3 为什么用中点而非值本身？

用中点而非训练值本身作阈值，保证阈值不在训练样本上（避免边界歧义）。例如值 $[1, 2, 3]$，阈值 $1.5$ 把 $1$ 分左、$2, 3$ 分右，无歧义。

---

## 二、分类树：基尼系数

### 2.1 基尼系数定义

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

其中 $p_k$ 是类别 $k$ 在 $D$ 中的比例。

- 纯节点（所有样本同类）：Gini = 0
- 二分类均匀分布：Gini = 0.5
- Gini 越小越纯

#### 2.1.1 推导：基尼的期望含义

基尼系数 = 从 $D$ 中**随机抽两个样本，它们类别不同的概率**：

$$
P(\text{不同类}) = 1 - \sum_k P(\text{都类 } k) = 1 - \sum_k p_k^2
$$

故 Gini 越小，越可能抽到同类，节点越纯。

#### 2.1.2 数值示例

- 节点 $[A, A, A, A]$：$p_A=1, \text{Gini}=1-1=0$（纯）
- 节点 $[A, A, B, B]$：$p_A=p_B=0.5, \text{Gini}=1-0.25-0.25=0.5$
- 节点 $[A, B, C, D]$：$p=0.25$ each, $\text{Gini}=1-4 \cdot 0.0625=0.75$

### 2.2 为什么用基尼而非信息熵？

| | 基尼系数 | 信息熵 |
|---|---|---|
| 计算 | $O(K)$，无需 log | $O(K)$，需 log |
| 范围 | $[0, 1-1/K]$ | $[0, \log K]$ |
| 增长速度 | 二次 | 对数 |
| 分裂偏好 | 倾向均衡分裂 | 倾向均衡分裂 |

两者效果相近，基尼计算更快（无需 log），sklearn 默认用基尼。

#### 2.2.1 信息熵定义

$$
H(D) = -\sum_k p_k \log_2 p_k
$$

- 纯节点：$H = 0$
- 二分类均匀：$H = 1$
- $H$ 越小越纯

#### 2.2.2 基尼 vs 熵的图像

```
纯度
  ^
  |     熵 ___
  |        /
  | 基尼 /
  |     /
  |    /
  |   /
  +--+--+--+--> p (二分类一类比例)
     0 .5  1
```

熵在 $p=0.5$ 处更陡（对不纯更敏感），但分裂选择上两者高度一致。

### 2.3 叶子预测

叶子节点的预测值 = 多数类（`np.argmax(counts)`）。

#### 2.3.1 概率输出

`predict_proba` 返回叶子内各类比例：

$$
P(y=k | x \in \text{leaf}) = \frac{\text{leaf 内类 } k \text{ 样本数}}{\text{leaf 内总样本数}}
$$

这是叶子内的经验分布。注意：树的概率输出是分段的（每个叶子一个分布），不如逻辑回归光滑。

---

## 三、回归树：MSE

### 3.1 MSE 定义

$$
\text{MSE}(D) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2 = \text{Var}(y)
$$

- 纯节点（所有目标值相同）：MSE = 0
- MSE 越小越纯

#### 3.1.1 为什么用方差？

回归树的分裂目标是让叶子内目标值尽量一致。方差度量一致性，方差 0 即所有值相同。最小化加权 MSE = 让每个叶子内方差小。

#### 3.1.2 推导：MSE = 方差

$$
\text{MSE} = \frac{1}{n}\sum_i (y_i - \bar{y})^2 = \frac{1}{n}\sum_i y_i^2 - \bar{y}^2 = \mathbb{E}[y^2] - (\mathbb{E}[y])^2 = \text{Var}(y)
$$

### 3.2 叶子预测

叶子节点的预测值 = 该节点样本目标值的均值（`np.mean(y)`）。

#### 3.2.1 为什么用均值？

均值是使 MSE 最小的常数预测。证明：对 $\hat{y} = c$，最小化 $\sum (y_i - c)^2$ 对 $c$ 求导令零得 $c = \bar{y}$。

### 3.3 与线性回归的对比

| | 决策树回归 | 线性回归 |
|---|---|---|
| 预测形式 | 分段常数 | 全局线性 |
| 决策边界 | 轴对齐超平面 | 任意超平面 |
| 外推能力 | 无（叶子值固定） | 有（线性外推） |
| 适合 | 非线性、分段 | 全局线性关系 |
| 特征缩放 | 不需要 | 需要 |
| 可解释性 | if-else 规则 | 权重 |

#### 3.3.1 外推问题

决策树在训练数据范围外的预测恒为最近叶子的均值，无法外推。例如训练数据 $x \in [0, 10]$，预测 $x=100$ 返回最右叶子的均值，而非线性增长。这是树模型的固有局限。

---

## 四、停止条件

| 条件 | 含义 | 参数 |
|------|------|------|
| 样本太少 | $n < $ `min_samples_split` | `min_samples_split` |
| 已纯 | impurity = 0 | 自动检测 |
| 深度到顶 | depth $\geq$ `max_depth` | `max_depth` |
| 叶子太小 | 分裂后某侧 $< $ `min_samples_leaf` | `min_samples_leaf` |

#### 4.0.1 各参数的作用

- `max_depth`：限制树深，防止过拟合。深树记细节但泛化差
- `min_samples_split`：节点至少多少样本才考虑分裂。大值防过拟合
- `min_samples_leaf`：叶子至少多少样本。保证叶子统计可靠
- `max_features`：分裂时考虑多少特征。随机森林用，单树默认全部

#### 4.0.2 参数调优

```python
for depth in [3, 5, 7, 10, None]:
    clf = DecisionTreeClassifier(max_depth=depth).fit(X_tr, y_tr)
    print(f"depth={depth}: 训练={clf.score(X_tr, y_tr)}, 测试={clf.score(X_te, y_te)}")
# None（不限深）通常训练 1.0，测试低 → 过拟合
```

---

## 五、实现结构

```python
class _TreeNode:
    feature: 分裂特征索引
    threshold: 分裂阈值
    left / right: 左右子树
    value: 叶子预测值
    is_leaf: 是否叶子

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if 停止条件:
            return _TreeNode(is_leaf=True, value=多数类)
        best = self._find_best_split(X, y)
        left = self._build_tree(X[left], y[left], depth+1)
        right = self._build_tree(X[right], y[right], depth+1)
        return _TreeNode(feature, threshold, left, right)

    def predict(self, X):
        return [self.tree_.predict_one(x) for x in X]
```

### 5.1 _find_best_split 实现

```python
def _find_best_split(self, X, y):
    best_gain, best_feat, best_thr = -inf, None, None
    for j in range(X.shape[1]):
        thresholds = self._candidate_thresholds(X[:, j])
        for thr in thresholds:
            left = X[:, j] <= thr
            gain = self._impurity(y) - self._weighted_impurity(y[left], y[~left])
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, j, thr
    return best_feat, best_thr
```

### 5.2 predict_one 实现

```python
def predict_one(self, x, node):
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value
```

从根到叶的路径长度 = 树深，预测 $O(\text{depth})$。

---

## 六、复杂度分析

### 6.1 训练

- 朴素：每节点 $O(d n^2)$（$d$ 特征各 $n$ 阈值各 $O(n)$ 评估）
- 排序优化：每节点 $O(d n \log n)$
- 总（树深 $h$）：$O(h \cdot d n \log n)$

最坏 $h = n$（退化树），总 $O(d n^2 \log n)$。平衡时 $h = \log n$，总 $O(d n \log^2 n)$。

### 6.2 预测

- 单样本：$O(h)$（根到叶路径）
- 批量 $m$ 样本：$O(m h)$

非常快，这是树的优势。

### 6.3 空间

- 节点数 $O(\text{叶子数})$，最坏 $O(n)$
- 每节点存 feature、threshold、子节点指针，$O(1)$
- 总 $O(n)$

---

## 七、剪枝

### 7.1 预剪枝

通过停止条件限制树生长：
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`
- `max_leaf_nodes`

简单有效，sklearn 默认用预剪枝。

### 7.2 后剪枝

先长成满树，再自底向上合并叶子。CART 用代价复杂度剪枝：

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

$R(T)$ 是训练误差，$|T|$ 是叶子数，$\alpha$ 控制复杂度惩罚。sklearn 的 `ccp_alpha` 参数。

### 7.3 何时剪枝

- 训练远高于测试 → 过拟合 → 剪枝
- 树很深（>20）→ 可能过拟合
- 数据少 → 易过拟合 → 强剪枝

---

## 八、使用示例

### 8.1 分类

```python
from minisklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
print(clf.predict([[0.5, 0.5], [5.5, 5.5]]))  # [0, 1]
print(clf.score(X, y))
```

### 8.2 回归

```python
from minisklearn.tree import DecisionTreeRegressor

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([1, 2, 3, 4, 5], dtype=float)

reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
print(reg.predict([[2.5]]))
```

### 8.3 完整流水线

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from minisklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
print("训练:", clf.score(X_tr, y_tr))
print("测试:", clf.score(X_te, y_te))
```

---

## 九、与 sklearn 对比

| 方面 | sklearn | minisklearn |
|------|---------|-------------|
| 算法 | CART 优化版 | CART 朴素版 |
| 分裂准则 | gini/entropy/log_loss | gini/entropy |
| `max_features` | 支持 | 暂不支持（恒全部） |
| 剪枝 | 预+后（ccp_alpha） | 仅预剪枝 |
| 稀疏矩阵 | 支持 | 暂不支持 |
| 数值精度 | 一致 | ✅ |
| 速度 | 快（Cython） | 慢 10-100x |

### 9.1 数值一致性

```python
from sklearn.tree import DecisionTreeClassifier as SkD
from minisklearn.tree import DecisionTreeClassifier as MnD
X, y = load_iris(return_X_y=True)
sk = SkD(max_depth=3, random_state=0).fit(X, y)
mn = MnD(max_depth=3).fit(X, y)
# 分裂可能略不同（实现细节），但准确率接近
print(sk.score(X, y), mn.score(X, y))
```

### 9.2 性能对比

| n | d | sklearn | minisklearn | 比值 |
|---|---|---|---|---|
| 1000 | 5 | 5ms | 50ms | 10x |
| 10000 | 10 | 50ms | 1000ms | 20x |
| 50000 | 20 | 300ms | 10000ms | 30x |

sklearn 用 Cython + 排序优化，快一个量级。教学用 minisklearn 足够。

---

## 十、几何直觉深入

### 10.1 决策边界

树决策边界是轴对齐的矩形划分。每个叶子是一个轴对齐超矩形（特征空间的笛卡尔积区间）。

```
二维特征空间划分:
  x2
  ^
  |  C1 |  C2
  |-----+------
  |  C2 |  C1
  |     |
  +-----+-------> x1
```

### 10.2 与线性边界的对比

线性模型边界是斜的超平面，树边界是阶梯。若真实边界是斜的，树需很多阶梯近似（深树），线性模型一步到位。若真实边界是轴对齐的，树天然适合。

### 10.3 特征重要性

树的特征重要性 = 该特征在所有分裂中带来的信息增益总和（归一化）：

$$
\text{importance}(j) = \frac{\sum_{\text{node } n \text{ split on } j} \Delta I(n)}{\sum_{\text{all nodes}} \Delta I(n)}
$$

sklearn 的 `feature_importances_` 实现此。minisklearn 暂未实现。

---

## 十一、数值稳定性

### 11.1 阈值精度

阈值是训练值中点，浮点精度足够。极端值（如 $10^{15}$）的中点可能有精度损失，但实际罕见。

### 11.2 类别比例

$p_k = n_k / n$，$n=0$ 时除零。停止条件保证 $n \geq$ `min_samples_leaf` > 0。

### 11.3 排序稳定性

`np.argsort` 对相同值顺序不定，但相同值之间不分裂（1.3 跳过），故无影响。

---

## 十二、常见问题与陷阱

| 问题 | 现象 | 解决 |
|------|------|------|
| 不限深 | 过拟合 | 设 max_depth |
| 训练 1.0 | 过拟合 | 剪枝 |
| 测试低 | 泛化差 | 减深/增 min_samples_leaf |
| 高维稀疏 | 树深 | 降维或用线性模型 |
| 类别特征 | 需编码 | OneHot 或 LabelEncoder |
| 外推 | 叶子值固定 | 用线性模型或加趋势特征 |
| 不平衡 | 多数类主导 | class_weight='balanced' |
| 偏斜分裂 | 树退化 | min_samples_leaf |

### 12.1 调试技巧

```python
clf = DecisionTreeClassifier().fit(X_tr, y_tr)  # 不限深
print("训练:", clf.score(X_tr, y_tr))  # 通常 1.0
print("测试:", clf.score(X_te, y_te))  # 较低
# 训练 1.0 + 测试低 → 过拟合，加深限制

# 检查树深
def depth(node):
    if node.is_leaf: return 0
    return 1 + max(depth(node.left), depth(node.right))
print("树深:", depth(clf.tree_))
```

---

## 十三、实战教程

### 13.1 调 max_depth

```python
from sklearn.model_selection import cross_val_score
for d in [2, 3, 5, 7, 10, 15, None]:
    clf = DecisionTreeClassifier(max_depth=d)
    s = cross_val_score(clf, X, y, cv=5).mean()
    print(f"depth={d}: {s:.3f}")
```

### 13.2 调 min_samples_leaf

```python
for m in [1, 5, 10, 20, 50]:
    clf = DecisionTreeClassifier(min_samples_leaf=m)
    s = cross_val_score(clf, X, y, cv=5).mean()
    print(f"min_samples_leaf={m}: {s:.3f}")
```

### 13.3 与随机森林对比

```python
from minisklearn.ensemble import RandomForestClassifier
dt = DecisionTreeClassifier(max_depth=5).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_tr, y_tr)
print(f"单树: {dt.score(X_te, y_te)}")
print(f"森林: {rf.score(X_te, y_te)}")  # 通常更高
```

---

## 十四、进阶话题

### 14.1 ID3 / C4.5

ID3 只处理类别特征，多路分裂，用信息增益。C4.5 用增益率（修正信息增益偏向多值特征的问题）。CART 是二分版，更通用。

### 14.2 增益率

$$
\text{GainRatio} = \frac{\text{Gain}}{\text{SplitInfo}}, \quad \text{SplitInfo} = -\sum \frac{n_i}{n} \log \frac{n_i}{n}
$$

修正信息增益偏向高基数特征的问题。

### 14.3 直方图优化

对大量候选阈值，LightGBM/XGBoost 用直方图把连续值分桶，候选阈值从 $n$ 降到桶数（如 256），大幅加速。sklearn 0.24+ 也加了直方图版 `HistGradientBoosting`。

### 14.4 软分裂

模糊决策树用软阈值（sigmoid）替代硬 if-else，让分裂可微，可用梯度优化。研究性质，主流不用。

---

## 十五、数学补充

### 15.1 基尼的凸性

基尼是 $p$ 的凸函数（$\sum p_k^2$ 是凸的，负号+1 仍凸）。凸性保证：分裂后加权基尼 $\leq$ 分裂前基尼，即分裂总不增加不纯度。这是树训练单调下降的保证。

### 15.2 信息增益的偏置

信息增益偏向选择多值特征（如 ID）。极端例：用样本 ID 作特征，每值唯一，分裂后每叶子纯，Gain 最大，但泛化为 0。增益率修正此偏置。

### 15.3 树的容量

深度 $h$ 的二叉树最多 $2^h$ 叶子，每个叶子一个预测值。故深度 $h$ 的树能表示 $2^h$ 个不同的区域。容量随深度指数增长，易过拟合。

### 15.4 叶子预测的最优性

分类：多数类是使 0-1 损失最小的预测。
回归：均值是使 MSE 最小的预测。
都是叶子内经验风险最小化。

---

## 十六、与 sklearn 详细对比测试

### 16.1 分类对比

```python
from sklearn.tree import DecisionTreeClassifier as SkD
from minisklearn.tree import DecisionTreeClassifier as MnD
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
sk = SkD(max_depth=3, random_state=42).fit(X, y)
mn = MnD(max_depth=3).fit(X, y)
print("sklearn:", sk.score(X, y))
print("minisklearn:", mn.score(X, y))
```

### 16.2 回归对比

```python
from sklearn.tree import DecisionTreeRegressor as SkR
from minisklearn.tree import DecisionTreeRegressor as MnR
import numpy as np
X = np.random.randn(200, 3)
y = X @ [1, -2, 3] + np.random.randn(200) * 0.1
sk = SkR(max_depth=4).fit(X, y)
mn = MnR(max_depth=4).fit(X, y)
print("sklearn R²:", sk.score(X, y))
print("minisklearn R²:", mn.score(X, y))
```

### 16.3 功能差异

| 功能 | sklearn | minisklearn |
|------|---------|-------------|
| criterion | gini/entropy/log_loss | gini/entropy |
| splitter | best/random | best |
| max_features | 支持 | 恒全部 |
| class_weight | 支持 | 暂不支持 |
| ccp_alpha 后剪枝 | 支持 | 暂不支持 |
| min_weight_fraction_leaf | 支持 | 暂不支持 |
| feature_importances_ | 支持 | 暂不支持 |
| apply（返回叶子索引） | 支持 | 暂不支持 |
| 决策路径 | 支持 | 暂不支持 |

---

## 十七、超参数调优指南

### 17.1 主要参数

| 参数 | 默认 | 作用 | 调优方向 |
|------|------|------|---------|
| max_depth | None | 树深 | 过拟合时减小 |
| min_samples_split | 2 | 分裂最小样本 | 过拟合时增大 |
| min_samples_leaf | 1 | 叶子最小样本 | 过拟合时增大 |
| max_features | all | 分裂考虑特征 | 随机森林用 |
| criterion | gini | 分裂准则 | 试 gini/entropy |

### 17.2 网格搜索

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy'],
}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5).fit(X, y)
print(gs.best_params_, gs.best_score_)
```

### 17.3 调优经验

- 先用默认参数看是否过拟合（训练 1.0 测试低）
- 过拟合则减 max_depth 或增 min_samples_leaf
- 欠拟合则增 max_depth
- criterion 通常 gini/entropy 差异小
- 单树调优上限有限，过拟合严重时换随机森林

---

## 十八、生产环境注意事项

### 18.1 模型序列化

```python
import pickle
clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
with open("tree.pkl", "wb") as f:
    pickle.dump(clf, f)
# 树模型通常比 KNN 小（只存节点，不存全数据）
```

### 18.2 在线学习

决策树不支持在线学习（新增数据需重训）。若需在线，用 Hoeffding Tree（VFDT）。

### 18.3 预测延迟

树预测 $O(\text{depth})$，非常快。深度 10 的树每次预测 10 次比较，纳秒级。

### 18.4 可解释性利用

树可直接翻译成 if-else 规则，给业务人员看：

```python
def tree_to_rules(node, conditions=[]):
    if node.is_leaf:
        return [f"{' & '.join(conditions)} → 类别 {node.value}"]
    rules = []
    rules += tree_to_rules(node.left, conditions + [f"x[{node.feature}] <= {node.threshold}"])
    rules += tree_to_rules(node.right, conditions + [f"x[{node.feature}] > {node.threshold}"])
    return rules
```

---

## 十九、完整实战教程

### 19.1 端到端分类

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine
from minisklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 调 max_depth
best_d, best_s = 1, 0
for d in [2, 3, 5, 7, 10, None]:
    clf = DecisionTreeClassifier(max_depth=d)
    s = cross_val_score(clf, X_tr, y_tr, cv=5).mean()
    if s > best_s:
        best_d, best_s = d, s
    print(f"depth={d}: CV={s:.3f}")

clf = DecisionTreeClassifier(max_depth=best_d).fit(X_tr, y_tr)
print(f"最优 depth={best_d}, 测试={clf.score(X_te, y_te):.3f}")
```

### 19.2 调 min_samples_leaf

```python
for m in [1, 2, 5, 10, 20]:
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=m)
    s = cross_val_score(clf, X_tr, y_tr, cv=5).mean()
    print(f"min_samples_leaf={m}: {s:.3f}")
```

### 19.3 回归示例

```python
import numpy as np
from minisklearn.tree import DecisionTreeRegressor

# 拟合分段函数
np.random.seed(0)
X = np.sort(np.random.rand(200) * 10).reshape(-1, 1)
y = np.where(X.ravel() < 5, X.ravel(), 10 - X.ravel()) + np.random.randn(200) * 0.1

reg = DecisionTreeRegressor(max_depth=4).fit(X, y)
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = reg.predict(X_test)
# 树能很好拟合分段常数
```

### 19.4 与线性模型对比

```python
from minisklearn.linear_model import LinearRegression

# 非线性数据
X = np.random.randn(300, 1) * 3
y = (X.ravel() ** 2) + np.random.randn(300) * 0.5

tree = DecisionTreeRegressor(max_depth=5).fit(X, y)
lr = LinearRegression().fit(X, y)
print(f"树 R²: {tree.score(X, y)}")   # 高（能拟合非线性）
print(f"线性 R²: {lr.score(X, y)}")   # 低（线性模型拟合 x² 差）
```

### 19.5 过拟合演示

```python
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for d in [1, 2, 3, 5, 10, None]:
    clf = DecisionTreeClassifier(max_depth=d).fit(X_tr, y_tr)
    tr = clf.score(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    print(f"depth={d}: 训练={tr:.3f} 测试={te:.3f} gap={tr-te:.3f}")
# depth=None: 训练=1.0 测试低，过拟合
# depth=3: 训练≈测试，平衡
```

---

## 二十、决策树的可解释性

### 20.1 树可视化

```python
# 文本打印树
def print_tree(node, depth=0, prefix="根"):
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}{prefix} → 叶子: {node.value}")
    else:
        print(f"{indent}{prefix}: x[{node.feature}] <= {node.threshold:.3f}")
        print_tree(node.left, depth+1, "左")
        print_tree(node.right, depth+1, "右")

print_tree(clf.tree_)
```

### 20.2 提取规则

```python
def extract_rules(node, path=[]):
    if node.is_leaf:
        return [(" & ".join(path), node.value)]
    rules = []
    rules += extract_rules(node.left, path + [f"x[{node.feature}] <= {node.threshold:.2f}"])
    rules += extract_rules(node.right, path + [f"x[{node.feature}] > {node.threshold:.2f}"])
    return rules

for cond, val in extract_rules(clf.tree_):
    print(f"IF {cond} THEN 类别 {val}")
```

### 20.3 单样本决策路径

```python
def decision_path(node, x, path=[]):
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            path.append(f"x[{node.feature}]={x[node.feature]:.2f} <= {node.threshold:.2f}")
            node = node.left
        else:
            path.append(f"x[{node.feature}]={x[node.feature]:.2f} > {node.threshold:.2f}")
            node = node.right
    return path, node.value

x = X_te[0]
path, pred = decision_path(clf.tree_, x)
print(" → ".join(path) + f" → 预测 {pred}")
```

---

## 二十一、常见问题汇总

| 问题 | 现象 | 解决 |
|------|------|------|
| 不限深过拟合 | 训练 1.0 测试低 | 设 max_depth |
| 树太深 | 叶子样本少 | 增 min_samples_leaf |
| 测试低 | 泛化差 | 剪枝或换随机森林 |
| 高维稀疏 | 树深慢 | 降维 |
| 类别特征 | 需编码 | OneHot |
| 外推差 | 叶子值固定 | 加趋势特征或用线性 |
| 不平衡 | 多数类主导 | class_weight |
| 偏斜分裂 | 退化树 | min_samples_leaf |
| 训练慢 | 朴素实现 | 用 sklearn |
| 不可复现 | 分裂不唯一 | 设 random_state |

### 21.1 学习曲线诊断

```python
import numpy as np
sizes = np.linspace(0.1, 1.0, 10)
for s in sizes:
    n = int(s * len(X_tr))
    clf = DecisionTreeClassifier(max_depth=5).fit(X_tr[:n], y_tr[:n])
    tr = clf.score(X_tr[:n], y_tr[:n])
    te = clf.score(X_te, y_te)
    print(f"n={n}: 训练={tr:.3f} 测试={te:.3f}")
```

---

## 架构回扣

决策树继承 `ClassifierMixin` / `RegressorMixin`，自动获得 `score`。`fit` 后的 `tree_` 属性以单下划线结尾（不是 `coef_`），因为树结构不是"学出来的参数"而是"构建的结构"。

### 类层级

```
BaseEstimator
   ├── DecisionTreeClassifier + ClassifierMixin
   └── DecisionTreeRegressor  + RegressorMixin
```

### fit 后属性

| 算法 | 属性 | 含义 |
|------|------|------|
| DecisionTreeClassifier | `tree_`, `classes_` | 树结构、类别 |
| DecisionTreeRegressor | `tree_` | 树结构 |

`tree_` 是 `_TreeNode` 根节点，递归包含整棵树。

### 设计哲学

- **递归结构**：树天然递归，`_build_tree` 递归调用自身
- **预剪枝优先**：用停止条件防过拟合，简单有效
- **分裂与预测分离**：`_find_best_split` 找最优分裂，`predict_one` 沿树下行

### 与随机森林的关系

随机森林是元估计器，组合多棵决策树。决策树作为基学习器，其 `fit` / `predict` 接口被随机森林复用。这体现"组合优于继承"。

### 与预处理器的契约

决策树**不需要特征缩放**（分裂阈值与尺度无关，单调变换不改变分裂顺序）。这是树相对线性模型的一大优势——少一道预处理。但类别特征需编码（OneHot 或 LabelEncoder）。

### 与 KNN 的对比

| | 决策树 | KNN |
|---|---|---|
| 训练 | $O(h \cdot d n \log n)$ | $O(1)$ |
| 预测 | $O(h)$ | $O(nd)$ |
| 模型大小 | $O(\text{叶子数})$ | $O(nd)$ |
| 特征缩放 | 不需要 | 需要 |
| 可解释性 | 高（规则） | 低 |
| 外推 | 无 | 无 |

两者都是非参数模型，都不外推，但树训练贵预测便宜，KNN 反之。

### 与随机森林的对比

| | 单树 | 随机森林 |
|---|---|---|
| 过拟合 | 易 | 难（集成） |
| 准确率 | 中 | 高 |
| 可解释性 | 高 | 中（特征重要性） |
| 训练 | 快 | 慢（多棵树） |
| 预测 | 快 | 慢（多棵树投票） |

### 进一步学习

- sklearn 的 `tree.export_text` / `plot_tree` 可视化
- LightGBM / XGBoost / CatBoost 梯度提升树
- Hoeffding Tree 在线学习
- 软决策树与可微树
- 多变量决策树（每次分裂用多个特征线性组合）
- Oblivious Tree（同层共享分裂，LightGBM 用）
- 极随机树（ExtraTrees，分裂阈值随机选）

### 设计模式回顾

决策树体现了**递归 + 贪心**的经典模式：
- 递归：`_build_tree` 调用自身处理子集
- 贪心：每节点选当前最优分裂，不回溯全局
- 贪心的代价：可能陷入局部最优（不一定是全局最优树），但实践中效果好且快

### 总结

决策树以直观的 if-else 规则递归划分特征空间，可解释性极强，无需缩放，能捕捉非线性。但单树易过拟合，实践中常用随机森林或梯度提升树集成。理解决策树是理解这些集成方法的基础。

### 关键要点回顾

1. **CART 二分**：每节点恰好两个子节点，沿特征轴分裂
2. **基尼/熵**：分类纯度度量，基尼更快，效果相近
3. **MSE/方差**：回归纯度度量，叶子预测为均值
4. **预剪枝**：用 max_depth / min_samples_leaf 控制复杂度
5. **不需缩放**：分裂阈值与尺度无关
6. **不外推**：叶子值固定，训练范围外预测恒定
7. **贪心递归**：每节点局部最优，不保证全局最优
8. **可解释**：直接翻译成 if-else 规则，业务友好
9. **特征重要性**：分裂增益累计，可用于特征选择
10. **集成基础**：随机森林、GBDT 都以决策树为基学习器
11. **复杂度**：训练 $O(h \cdot d n \log n)$，预测 $O(h)$
12. **数值稳定**：阈值取中点，无除零风险
13. **类别特征**：需 OneHot 或 LabelEncoder 预处理
14. **不平衡数据**：需 class_weight 或重采样处理
15. **后剪枝**：ccp_alpha 代价复杂度剪枝，minisklearn 暂未实现
16. **直方图优化**：LightGBM/HistGBDT 把连续值分桶加速
17. **多变量分裂**：用多特征线性组合分裂，边界可斜

---

## 二十二、深入数学推导与证明

### 22.1 基尼系数的凸性证明

**定理**：基尼系数 $\text{Gini}(D) = 1 - \sum_k p_k^2$ 是概率向量 $p$ 的凸函数。

**证明**：

$\sum_k p_k^2$ 是 $p$ 的凸函数（二次型，Hessian $= 2I \succeq 0$）。

$\text{Gini} = 1 - \sum_k p_k^2$ 是凸函数的负值加常数，是**凹函数**。

**推论**：分裂后加权基尼 $\leq$ 分裂前基尼（凹函数的 Jensen 不等式）：
$$
\frac{n_L}{n}\text{Gini}(L) + \frac{n_R}{n}\text{Gini}(R) \leq \text{Gini}\left(\frac{n_L}{n}L + \frac{n_R}{n}R\right) = \text{Gini}(D)
$$

故分裂总不增加不纯度，树训练单调下降。$\square$

### 22.2 信息熵的凸性

**定理**：信息熵 $H(D) = -\sum_k p_k \log p_k$ 是 $p$ 的凹函数。

**证明**：$-p \log p$ 的二阶导数 $= -1/p < 0$，故 $-p\log p$ 凹，和保持凹性。$\square$

**推论**：信息增益 $\geq 0$，分裂总不增加熵。

### 22.3 基尼系数的期望含义

**定理**：$\text{Gini}(D) = P(\text{随机抽两样本，类别不同})$。

**证明**：
$$
P(\text{同类}) = \sum_k P(\text{都类 } k) = \sum_k p_k^2
$$
$$
P(\text{不同类}) = 1 - \sum_k p_k^2 = \text{Gini}(D) \quad \square
$$

### 22.4 基尼 vs 熵的泰勒展开

在 $p = 0.5$ 附近（二分类），令 $p = 0.5 + \epsilon$：

- $\text{Gini} = 1 - p^2 - (1-p)^2 = 2p(1-p) = 0.5 - 2\epsilon^2$
- $H = -p\log_2 p - (1-p)\log_2(1-p) \approx 1 - \frac{2}{\ln 2}\epsilon^2 \approx 1 - 2.885\epsilon^2$

熵在 $p=0.5$ 处更陡（系数 2.885 vs 2），对不纯更敏感。但分裂选择上两者高度一致。

### 22.5 均值最小化 MSE 的证明

**定理**：对常数预测 $\hat{y} = c$，$\sum_i (y_i - c)^2$ 在 $c = \bar{y}$ 时最小。

**证明**：
$$
\frac{d}{dc}\sum_i (y_i - c)^2 = -2\sum_i (y_i - c) = 0 \Rightarrow c = \frac{1}{n}\sum_i y_i = \bar{y}
$$

二阶导数 $= 2n > 0$，故 $\bar{y}$ 是最小值。$\square$

### 22.6 多数类最小化 0-1 损失的证明

**定理**：对常数预测 $\hat{y} = c$，$\sum_i \mathbb{1}[y_i \neq c]$ 在 $c = \arg\max_k n_k$（多数类）时最小。

**证明**：
$$
\sum_i \mathbb{1}[y_i \neq c] = n - n_c
$$

最小化等价于最大化 $n_c$，即 $c$ 为多数类。$\square$

### 22.7 信息增益偏向多值特征的证明

**定理**：信息增益偏向选择取值多的特征。

**证明思路**：若特征 $A$ 有 $m$ 个值，分裂后每叶子可能纯（每值一类），$H(D|A) = 0$，$\text{Gain} = H(D)$。$m$ 越大越容易纯，故增益偏向 $m$ 大的特征。

极端例：用样本 ID 作特征，$m = n$，每叶子一个样本，$\text{Gain}$ 最大，但泛化为 0。

**修正**：增益率 $\text{GainRatio} = \text{Gain}/\text{SplitInfo}$，$\text{SplitInfo} = -\sum \frac{n_i}{n}\log\frac{n_i}{n}$ 惩罚多值分裂。$\square$

### 22.8 树的容量与过拟合

**定理**：深度 $h$ 的二叉决策树最多 $2^h$ 叶子，能表示 $2^h$ 个不同区域。

**含义**：容量随深度指数增长。$h = \log_2 n$ 时叶子数 $= n$，可记住每个训练样本（过拟合）。故需限制 $h$ 或用集成。

### 22.9 CART 的贪心不保证全局最优

**反例**：考虑二维数据，真实边界是斜线 $x_1 + x_2 = 0$。CART 用轴对齐分裂，需多步阶梯近似。贪心选第一步最优分裂可能非全局最优树。

**实践**：贪心虽非全局最优，但效果好且快。全局最优树搜索是 NP 难。

---

## 二十三、更多代码示例与对比实验

### 23.1 不同 max_depth 的过拟合演示

```python
import numpy as np
from minisklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print("max_depth | 训练分数 | 测试分数 | gap | 叶子数")
print("-" * 55)
for d in [1, 2, 3, 5, 7, 10, 20, None]:
    clf = DecisionTreeClassifier(max_depth=d).fit(X_tr, y_tr)
    tr = clf.score(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    # 估算叶子数
    n_leaves = count_leaves(clf.tree_)
    print(f"{str(d):9s} | {tr:.4f}   | {te:.4f}   | {tr-te:.4f} | {n_leaves}")

def count_leaves(node):
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
```

### 23.2 基尼 vs 熵对比

```python
from sklearn.tree import DecisionTreeClassifier as SkD

X, y = load_iris(return_X_y=True)

for criterion in ['gini', 'entropy']:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=3, criterion=criterion), X, y, cv=10
    )
    print(f"{criterion:8s}: {scores.mean():.4f} ± {scores.std():.4f}")
# 两者通常差异 < 1%
```

### 23.3 min_samples_leaf 的影响

```python
for m in [1, 2, 5, 10, 20, 50]:
    clf = DecisionTreeClassifier(min_samples_leaf=m)
    scores = cross_val_score(clf, X, y, cv=5)
    tr = DecisionTreeClassifier(min_samples_leaf=m).fit(X_tr, y_tr).score(X_tr, y_tr)
    te = DecisionTreeClassifier(min_samples_leaf=m).fit(X_tr, y_tr).score(X_te, y_te)
    print(f"min_samples_leaf={m:3d}: CV={scores.mean():.4f}, 训练={tr:.4f}, 测试={te:.4f}")
```

### 23.4 决策树 vs 线性模型 vs 随机森林

```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles

datasets = {
    'Iris (线性)': load_iris(return_X_y=True),
    'Moons (非线性)': make_moons(n_samples=300, noise=0.2, random_state=0),
    'Circles (非线性)': make_circles(n_samples=300, noise=0.1, random_state=0),
}

for ds_name, (X, y) in datasets.items():
    print(f"\n{ds_name}:")
    for name, clf in [('单树', DecisionTreeClassifier(max_depth=5)),
                       ('随机森林', RandomForestClassifier(n_estimators=100, random_state=0)),
                       ('逻辑回归', LogisticRegression(max_iter=2000))]:
        if name == '逻辑回归':
            X_s = StandardScaler().fit_transform(X)
            s = cross_val_score(clf, X_s, y, cv=5).mean()
        else:
            s = cross_val_score(clf, X, y, cv=5).mean()
        print(f"  {name:10s}: {s:.4f}")
```

### 23.5 回归树拟合分段函数

```python
from minisklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(0)
X = np.sort(np.random.uniform(0, 10, 200)).reshape(-1, 1)
y = np.where(X.ravel() < 3, X.ravel(),
    np.where(X.ravel() < 7, 3, 10 - X.ravel())) + np.random.randn(200) * 0.1

for d in [2, 3, 5, 10]:
    reg = DecisionTreeRegressor(max_depth=d).fit(X, y)
    y_pred = reg.predict(X)
    mse = np.mean((y_pred - y) ** 2)
    print(f"depth={d:2d}: MSE={mse:.4f}")
```

### 23.6 外推问题演示

```python
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)  # y = 2x

tree = DecisionTreeRegressor().fit(X, y)
lr = LinearRegression().fit(X, y)

X_test = np.array([[100]], dtype=float)
print(f"树预测 x=100: {tree.predict(X_test)[0]}")  # 10（最近叶子值）
print(f"线性预测 x=100: {lr.predict(X_test)[0]}")  # 200（线性外推）
```

---

## 二十四、参数调优进阶指南

### 24.1 系统调优流程

```python
from sklearn.model_selection import GridSearchCV

# 第一步：调 max_depth
param_grid = {
    'max_depth': [2, 3, 5, 7, 10, 15, None],
    'criterion': ['gini', 'entropy'],
}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5).fit(X_tr, y_tr)
print(f"最优: {gs.best_params_}, 分数={gs.best_score_:.4f}")

# 第二步：调 min_samples_leaf
best_depth = gs.best_params_['max_depth']
for m in [1, 2, 5, 10]:
    clf = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=m)
    s = cross_val_score(clf, X_tr, y_tr, cv=5).mean()
    print(f"min_samples_leaf={m}: {s:.4f}")
```

### 24.2 调优经验法则

| 场景 | max_depth | min_samples_leaf | criterion | 备注 |
|------|-----------|-----------------|-----------|------|
| 默认 | None | 1 | gini | 通常过拟合 |
| 防过拟合 | 3-10 | 5-20 | gini | 限制复杂度 |
| 小数据 | 3-5 | 5-10 | gini | 强限制 |
| 大数据 | 10-20 | 1-5 | gini | 可放宽 |
| 可解释 | 3-5 | 10+ | gini | 简单规则 |
| 高精度 | None | 1 | gini | 配合集成 |

### 24.3 诊断过拟合

```python
def diagnose_overfitting(X_tr, y_tr, X_te, y_te):
    """诊断决策树过拟合。"""
    clf = DecisionTreeClassifier().fit(X_tr, y_tr)  # 不限深
    tr = clf.score(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    
    print(f"不限深: 训练={tr:.4f}, 测试={te:.4f}, gap={tr-te:.4f}")
    
    if tr == 1.0 and tr - te > 0.1:
        print("→ 严重过拟合")
        print("  建议：减 max_depth 或增 min_samples_leaf 或用随机森林")
    
    # 找最优深度
    best_d, best_s = 1, 0
    for d in range(1, 20):
        s = cross_val_score(DecisionTreeClassifier(max_depth=d), X_tr, y_tr, cv=5).mean()
        if s > best_s:
            best_d, best_s = d, s
    print(f"最优 max_depth={best_d}, CV={best_s:.4f}")
```

---

## 二十五、常见错误与调试技巧

### 25.1 典型错误清单

```python
# 错误 1：不限深导致过拟合
clf = DecisionTreeClassifier().fit(X_tr, y_tr)
# 训练 1.0 测试低

# 错误 2：类别特征未编码
# X = [['男', 25], ['女', 30]]  # 文本特征
# 需先 OneHot 或 LabelEncoder

# 错误 3：用分类树做回归
# DecisionTreeClassifier().fit(X, y_continuous)  # y 应离散

# 错误 4：max_depth=1 太浅
clf = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
# 只分裂一次，欠拟合

# 错误 5：min_samples_leaf 太大
clf = DecisionTreeClassifier(min_samples_leaf=100).fit(X_tr, y_tr)
# 叶子太大，无法细分
```

### 25.2 调试检查清单

```python
def debug_decision_tree(clf, X_tr, y_tr, X_te, y_te):
    """决策树调试。"""
    print("=== 决策树调试 ===")
    print(f"max_depth={clf.max_depth}, min_samples_leaf={clf.min_samples_leaf}")
    
    tr = clf.score(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    print(f"训练: {tr:.4f}, 测试: {te:.4f}, gap: {tr-te:.4f}")
    
    if tr == 1.0:
        print("⚠ 训练 100%，可能过拟合")
    
    # 树深度
    def depth(node):
        if node.is_leaf: return 0
        return 1 + max(depth(node.left), depth(node.right))
    print(f"实际树深: {depth(clf.tree_)}")
    
    # 叶子数
    def n_leaves(node):
        if node.is_leaf: return 1
        return n_leaves(node.left) + n_leaves(node.right)
    print(f"叶子数: {n_leaves(clf.tree_)}")
```

---

## 二十六、与其他算法的深入对比

### 26.1 决策树 vs 线性模型

| 维度 | 决策树 | 线性模型 |
|------|--------|---------|
| 决策边界 | 轴对齐阶梯 | 光滑超平面 |
| 特征缩放 | 不需要 | 需要 |
| 非线性 | 是 | 否（需特征工程） |
| 外推 | 无 | 有 |
| 可解释性 | 极高（规则） | 高（权重） |
| 过拟合 | 易 | 难 |
| 训练 | $O(dn\log n)$ | $O(nd^2)$ |
| 预测 | $O(\text{depth})$ | $O(d)$ |

### 26.2 决策树 vs KNN

| 维度 | 决策树 | KNN |
|------|--------|-----|
| 训练 | $O(dn\log n)$ | $O(1)$ |
| 预测 | $O(\text{depth})$ | $O(nd)$ |
| 模型大小 | $O(\text{叶子数})$ | $O(nd)$ |
| 特征缩放 | 不需要 | 需要 |
| 可解释性 | 高 | 低 |

### 26.3 单树 vs 随机森林 vs GBDT

```python
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    '单树': DecisionTreeClassifier(max_depth=5),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=0),
    'GBDT': GradientBoostingClassifier(n_estimators=100, random_state=0),
}

for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    print(f"{name:10s}: 训练={clf.score(X_tr, y_tr):.4f}, 测试={clf.score(X_te, y_te):.4f}")
```

---

## 二十七、实际应用场景详解

### 27.1 信用评分规则

```python
# 决策树生成的规则可直接用于业务
# 例：IF 收入 > 50000 AND 负债率 < 0.3 THEN 批准
# 可解释性强，符合监管要求
```

### 27.2 医疗诊断决策

```python
# 特征：[年龄, 血压, 胆固醇, 血糖]
# 决策树生成诊断规则，医生可理解
np.random.seed(0)
X = np.column_stack([
    np.random.uniform(20, 80, 500),
    np.random.normal(120, 20, 500),
    np.random.normal(200, 40, 500),
    np.random.normal(90, 15, 500),
])
y = ((X[:,0] > 50) & (X[:,1] > 140) | (X[:,2] > 240)).astype(int)

clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
# 提取规则给医生看
```

### 27.3 客户流失预测

```python
# 特征：[使用频次, 最后登录距今天数, 投诉次数, 套餐类型]
# 决策树识别流失模式，生成干预规则
```

### 27.4 规则提取

```python
def tree_to_rules(node, conditions=[]):
    """将决策树翻译成 if-else 规则。"""
    if node.is_leaf:
        return [f"IF {' AND '.join(conditions)} THEN 类别 {node.value}"]
    rules = []
    left_cond = conditions + [f"x[{node.feature}] <= {node.threshold:.2f}"]
    right_cond = conditions + [f"x[{node.feature}] > {node.threshold:.2f}"]
    rules += tree_to_rules(node.left, left_cond)
    rules += tree_to_rules(node.right, right_cond)
    return rules

clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
for rule in tree_to_rules(clf.tree_):
    print(rule)
```

---

## 二十八、思考题与练习

### 基础题

1. **为什么决策树不需要特征缩放？**
   <details><summary>答案</summary>
   分裂基于阈值，单调变换不改变分裂顺序。
   </details>

2. **决策树为什么不能外推？**
   <details><summary>答案</summary>
   叶子值固定，训练范围外预测恒为最近叶子值。
   </details>

3. **基尼系数和熵哪个更快？**
   <details><summary>答案</summary>
   基尼（无需 log 计算）。
   </details>

### 中级题

4. **证明基尼系数的凸性。**
5. **解释信息增益偏向多值特征的原因。**
6. **推导均值最小化 MSE。**

### 高级题

7. **证明 CART 贪心不保证全局最优。**
8. **分析树深度与过拟合的关系。**
9. **比较预剪枝与后剪枝的理论差异。**

### 编程练习

10. **实现后剪枝（代价复杂度剪枝）。**
11. **实现特征重要性计算。**
12. **实现决策树可视化。**
13. **用决策树做特征选择。**
14. **实现多变量决策树（斜分裂）。**

---

## 二十九、扩展阅读

### 29.1 经典论文

- **Breiman et al. (1984)**：*Classification and Regression Trees*——CART 奠基
- **Quinlan (1986)**：*Induction of Decision Trees*——ID3
- **Quinlan (1993)**：*C4.5: Programs for Machine Learning*
- **Breiman (2001)**：*Random Forests*

### 29.2 教材章节

- *The Elements of Statistical Learning* 第 9 章——决策树
- *Pattern Classification*（Duda 等）第 8 章
- *统计学习方法*（李航）第 5 章

### 29.3 进阶主题

- **C4.5**：增益率、后剪枝
- **CART 后剪枝**：代价复杂度剪枝
- **多变量决策树**：斜分裂
- **Oblivious Tree**：同层共享分裂（LightGBM）
- **软决策树**：可微分裂
- **Hoeffding Tree**：在线学习

### 29.4 相关算法

- **随机森林**：Bagging 多棵树
- **GBDT / XGBoost / LightGBM**：Boosting 树
- **ID3 / C4.5**：多路分裂
- **M5**：回归树 + 线性模型叶子

---

[← 返回算法列表](../index.md)
