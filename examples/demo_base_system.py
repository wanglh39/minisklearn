"""端到端示例：用基类系统实现一个简单的均值分类器。

运行：python examples/demo_base_system.py
"""

import numpy as np
from minisklearn.base import BaseEstimator, ClassifierMixin, clone
from minisklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MeanClassifier(BaseEstimator, ClassifierMixin):
    """按特征均值分类的简单分类器。

    如果样本的特征均值 > 阈值，预测 1，否则预测 0。
    这是一个教学用分类器，用于演示基类系统的运作。
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["classes_"])
        X = check_array(X)
        return (X.mean(axis=1) > self.threshold).astype(int)


def main():
    print("=" * 60)
    print("minisklearn 基类系统演示")
    print("=" * 60)

    X = np.array([[1, 2], [3, 4], [5, 6], [0, 1]])
    y = np.array([0, 1, 1, 0])

    clf = MeanClassifier(threshold=2.0)

    print("\n1. repr 自动生成:")
    print(f"   {clf!r}")

    print("\n2. get_params 反射:")
    print(f"   {clf.get_params()}")

    print("\n3. fit + predict:")
    clf.fit(X, y)
    print(f"   预测: {clf.predict(X)}")
    print(f"   score: {clf.score(X, y):.2f}")

    print("\n4. clone 产生干净副本:")
    cloned = clone(clf)
    print(f"   原对象已训练: {hasattr(clf, 'classes_')}")
    print(f"   克隆未训练:   {not hasattr(cloned, 'classes_')}")
    print(f"   参数一致:     {cloned.threshold == clf.threshold}")

    print("\n5. set_params 链式调用:")
    clf.set_params(threshold=3.0)
    print(f"   新阈值: {clf.threshold}")
    print(f"   新预测: {clf.predict(X)}")

    print("\n" + "=" * 60)
    print("基类系统运作正常！")
    print("=" * 60)


if __name__ == "__main__":
    main()