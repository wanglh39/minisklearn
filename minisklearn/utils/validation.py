"""数据校验工具 —— sklearn 防御式编程的核心。

==============================================================================
为什么要做数据校验？
==============================================================================

机器学习代码最常见的崩溃不是算法错误，而是数据错误：

    - 传了 list 进来，但算法需要 ndarray 的 .shape 属性
    - 传了 NaN / Inf，导致后续矩阵运算静默出错
    - 传了 (n_features, n_samples) 的转置矩阵（维度搞反了）
    - 传了 object 类型的数组，运算时类型不兼容

sklearn 的解法是"入口校验"：每个估计器的 fit / predict / transform 开头
都调用 check_array / check_X_y 把输入规范化为合法的 numpy 数组。
这把错误前置到入口处，报错信息清晰，而不是在算法深处抛出晦涩的异常。

设计权衡：
    - 校验太严：用户觉得啰嗦（比如不允许 list 输入）
    - 校验太松：错误后置，调试困难
    sklearn 的选择：默认中等严格，可通过参数调节（force_all_finite 等）
"""

import numpy as np
import numbers


def _is_array_like(x):
    """判断对象是否"像数组"（有 __array__ 或是 list / tuple）。

    鸭子类型：不要求必须是 numpy 数组，只要有数组的行为就接受。
    """
    return hasattr(x, "__array__") or isinstance(x, (list, tuple))


def check_array(array, dtype=None, ensure_2d=True, force_all_finite=True,
                ensure_min_samples=1, ensure_min_features=1, copy=False):
    """校验并规范化输入数组为 numpy ndarray。

    这是 sklearn 最基础的校验函数。所有其他校验（check_X_y 等）都基于它。

    参数：
        array: 输入数据（list / ndarray / sparse matrix 等）
        dtype: 期望的数据类型，None 表示自动推断
        ensure_2d: 是否强制二维（大多数 sklearn 算法要求二维）
        force_all_finite: 是否禁止 NaN / Inf
        ensure_min_samples: 最少样本数
        ensure_min_features: 最少特征数
        copy: 是否强制复制（防止修改原数组）

    返回：
        np.ndarray: 校验后的数组

    设计细节：
        1. 接受 list / tuple → 转成 ndarray（方便用户）
        2. 接受一维 → 如果 ensure_2d，reshape 成 (n, 1)（兼容单特征）
        3. 检查 NaN / Inf → 提前报错而非静默出错
        4. 检查维度下限 → 防止空数据导致算法除零
    """
    if array is None:
        raise ValueError("输入数组不能为 None")

    if not _is_array_like(array):
        raise TypeError(
            f"期望数组类输入（list / ndarray 等），得到 {type(array).__name__}"
        )

    # 转为 ndarray
    array = np.asarray(array, dtype=dtype)

    if copy:
        array = array.copy()

    # 处理 object dtype 或字符串 dtype（通常是混合类型或字符串输入导致的）
    if array.dtype == object or np.issubdtype(array.dtype, np.character):
        try:
            array = array.astype(np.float64)
        except (ValueError, TypeError):
            raise TypeError(
                "输入数组包含无法转为数值的元素，请检查数据类型"
            )

    # NaN / Inf 检查
    if force_all_finite:
        if not np.all(np.isfinite(array)):
            if np.any(np.isnan(array)):
                raise ValueError("输入包含 NaN，请先处理缺失值")
            raise ValueError("输入包含无穷大（inf），请先处理异常值")

    # 维度检查
    if ensure_2d:
        if array.ndim == 1:
            # 一维自动转为列向量 (n_samples, 1)
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError(
                f"期望二维数组 (n_samples, n_features)，得到 {array.ndim} 维"
            )

        if ensure_min_samples > 0:
            n_samples = array.shape[0]
            if n_samples < ensure_min_samples:
                raise ValueError(
                    f"样本数 {n_samples} 少于最少要求 {ensure_min_samples}"
                )

        if ensure_min_features > 0 and array.shape[1] < ensure_min_features:
            raise ValueError(
                f"特征数 {array.shape[1]} 少于最少要求 {ensure_min_features}"
            )

    return array


def check_X_y(X, y, accept_sparse=False, dtype=None, force_all_finite=True,
              ensure_2d=True, copy=False):
    """校验特征矩阵 X 和标签向量 y，保证两者样本数一致。

    这是监督学习算法的标准入口校验。

    参数：
        X: 特征矩阵，shape (n_samples, n_features)
        y: 标签向量，shape (n_samples,) 或 (n_samples, n_outputs)
        其余参数同 check_array

    返回：
        (X, y): 校验后的 (X, y) 元组

    设计要点：
        1. X 和 y 的第一维必须相同（样本数一致）
        2. y 允许一维（单输出）或二维（多输出）
        3. y 的 dtype 不强制（分类标签可以是字符串）
    """
    X = check_array(X, dtype=dtype, ensure_2d=ensure_2d,
                    force_all_finite=force_all_finite, copy=copy)

    if y is None:
        raise ValueError("y 不能为 None（监督学习需要标签）")

    y = np.asarray(y)

    # y 允许一维或二维
    if y.ndim > 2:
        raise ValueError(f"y 应为一维或二维，得到 {y.ndim} 维")

    # 样本数一致性检查 —— 这是最常见的用户错误
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X 的样本数 {X.shape[0]} 与 y 的样本数 {y.shape[0]} 不一致"
        )

    return X, y


def check_is_fitted(estimator, attributes=None, msg=None):
    """检查估计器是否已训练（是否调用过 fit）。

    设计动机：
        如果用户忘了 fit 就调用 predict，算法会去访问 self.coef_ 等属性，
        但这些属性是 fit 时才创建的，没 fit 就不存在，会抛出 AttributeError。
        但 AttributeError 的信息很晦涩（"'XXX' object has no attribute 'coef_'"），
        用户不知道是忘了 fit 还是参数名写错了。

        check_is_fitted 用 NotFittedError 给出明确提示："请先调用 fit()"。

    实现原理：
        fit 后的属性约定以下划线结尾（如 coef_、intercept_、classes_）。
        检查这些属性是否存在即可判断是否 fit 过。

    参数：
        estimator: 要检查的估计器
        attributes: 要检查的属性名列表，None 则自动找所有以 _ 结尾的属性
        msg: 自定义错误信息
    """
    if attributes is None:
        # 自动查找所有以 _ 结尾且不以 __ 开头的属性
        attributes = [
            attr for attr in vars(estimator)
            if attr.endswith("_") and not attr.startswith("__")
        ]

    if not attributes:
        # 没有任何以 _ 结尾的属性，说明没 fit 过
        raise TypeError(
            f"check_is_fitted 无法判断 {type(estimator).__name__} 是否已训练："
            f"该估计器没有任何以 _ 结尾的属性。请显式传入 attributes 参数。"
        )

    fitted = all(hasattr(estimator, attr) for attr in attributes)

    if not fitted:
        if msg is None:
            msg = (
                f"此 {type(estimator).__name__} 实例尚未训练，"
                f"请先调用 fit() 方法后再使用 predict() / transform()。"
            )
        from ..exceptions import NotFittedError
        raise NotFittedError(msg)


def check_random_state(seed):
    """将 seed 统一转为 np.random.RandomState 实例。

    设计动机：
        用户可能传入 int、None、或 RandomState 实例，算法内部需要统一接口。
        这是 sklearn 处理随机性的标准方式。

    参数：
        seed: None / int / np.random.RandomState

    返回：
        np.random.RandomState
    """
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise TypeError(
        f"seed 应为 None / int / RandomState，得到 {type(seed).__name__}"
    )