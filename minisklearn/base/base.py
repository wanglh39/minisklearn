"""BaseEstimator —— 所有估计器的根基类。

本模块是理解 sklearn 架构设计的核心入口。BaseEstimator 本身不实现任何机器学习
算法，它只做三件事：

    1. 参数管理：get_params / set_params
    2. 克隆机制：clone
    3. 自动 repr：__repr__

这三件事看似简单，却是整个 sklearn 生态能运转的基石。

==============================================================================
设计决策 1：为什么 __init__ 只存参数，不做事？
==============================================================================

sklearn 有一条硬性约定（在 SLEP009 中明确）：

    所有超参数必须通过 __init__ 传入，且 __init__ 只能把参数原样存到 self 上，
    不允许做任何计算、校验、转换。

为什么？

    (1) clone 的需要 —— 元估计器（如 GridSearchCV）需要反复克隆基础估计器，
        克隆的依据就是 __init__ 的参数签名。如果 __init__ 做了计算或修改了
        参数，clone 就无法还原原始状态。

    (2) 可序列化 —— 只存参数的 __init__ 使得对象状态可由参数完全描述，
        方便 pickle 和 inspect。

    (3) 一致性 —— 所有估计器遵循同一约定，用户不需要猜参数在哪设置。

反例（错误写法）：
    def __init__(self, C=1.0):
        self.C = float(C)          # 错！做了转换
        self._cache = {}           # 错！初始化了非参数状态

正确写法：
    def __init__(self, C=1.0):
        self.C = C                 # 原样存储，不做任何事

==============================================================================
设计决策 2：get_params / set_params 用反射实现
==============================================================================

为什么不手动写 get_params 返回字典？

    因为 sklearn 有上百个估计器，每个都手写 get_params 会产生大量重复代码，
    且容易漏参数。反射（inspect）能自动从 __init__ 签名提取参数名，
    保证 get_params 与 __init__ 永远一致。

    这也是"约定优于配置"的体现：只要你遵守"__init__ 只存同名属性"的约定，
    get_params 就能自动工作，无需你写任何额外代码。

==============================================================================
设计决策 3：clone 为什么不用 copy.deepcopy？
==============================================================================

    deepcopy 会复制对象的全部状态，包括 fit 之后学到的参数（coef_ 等）。
    但 clone 的语义是"得到一个未训练的同参数新对象"，所以必须基于
    __init__ 参数重建，而不是复制当前状态。

    clone(estimator) 的过程：
        1. 用 get_params 取出当前参数
        2. 用 type(estimator)(**params) 重新构造一个新实例
        3. 新实例是未训练的，但参数与原实例一致
"""

import copy
import inspect
import warnings

from ..exceptions import NotFittedError


def _get_param_names(cls):
    """从类的 __init__ 签名中提取参数名（不含 self 和 *args / **kwargs）。

    这是反射的核心：通过 inspect 拿到 __init__ 的形参列表，
    过滤掉 self 和变长参数，剩下的就是超参数名。

    设计细节：
        - 排除 *args / **kwargs：因为它们无法被 get_params 表达为固定字典。
          如果一个估计器的 __init__ 有 *args，说明它违反了 sklearn 约定。
        - 按 __init__ 中定义的顺序返回：保证 __repr__ 输出稳定可读。

    参数：
        cls: 要检查的类（不是实例）

    返回：
        list[str]: 参数名列表，按 __init__ 定义顺序
    """
    init_signature = inspect.signature(cls.__init__)
    param_names = [
        param_name
        for param_name, param in init_signature.parameters.items()
        if param_name != "self"
        and param.kind != param.VAR_POSITIONAL
        and param.kind != param.VAR_KEYWORD
    ]
    return param_names


def clone(estimator, *, safe=True):
    """构造一个与 estimator 参数相同但未训练的新实例。

    这是元估计器的基石。GridSearchCV 在搜索时，会对基础估计器做大量 clone，
    每次克隆出一个干净的副本去 fit 一个参数组合。

    为什么不直接 deepcopy？
        deepcopy 会把 fit 后学到的 coef_、classes_ 等全部复制过去，
        但我们要的是一个"全新的、未训练的、同参数的"估计器。
        所以必须走 get_params → 重新 __init__ 的路径。

    参数：
        estimator: 要克隆的估计器
        safe: 如果 True，当克隆结果与原对象类型不同时报错（防止意外）

    返回：
        与 estimator 同类型、同参数的未训练新实例
    """
    # 非 BaseEstimator 对象直接深拷贝，不递归 clone
    # （clone 只对估计器有意义，对 float / str / list 等会出错）
    if not isinstance(estimator, BaseEstimator):
        return copy.deepcopy(estimator)

    # 如果子类覆盖了 __sklearn_clone__，使用自定义克隆逻辑
    # （Pipeline 等元估计器需要特殊处理嵌套估计器的克隆）
    if type(estimator).__sklearn_clone__ is not BaseEstimator.__sklearn_clone__:
        new_object = estimator.__sklearn_clone__()
        if safe and not isinstance(new_object, type(estimator)):
            raise RuntimeError(
                f"clone({estimator}) 返回了类型 {type(new_object)}，"
                f"与原类型 {type(estimator)} 不一致。"
            )
        return new_object

    estimator_type = type(estimator)
    param_names = _get_param_names(estimator_type)

    # 用 get_params 取出当前参数（注意：取的是 __init__ 参数，不是 fit 后的属性）
    params = {name: clone(getattr(estimator, name), safe=False)
              for name in param_names}

    new_object = estimator_type(**params)

    if safe and not isinstance(new_object, estimator_type):
        raise RuntimeError(
            f"clone({estimator}) 返回了类型 {type(new_object)}，"
            f"与原类型 {estimator_type} 不一致，这通常意味着 __init__ 实现有误。"
        )
    return new_object


class BaseEstimator:
    """所有估计器的基类。

    本类不实现 fit / predict / transform —— 这些由子类或 Mixin 提供。
    本类只提供与具体算法无关的通用能力：参数管理、克隆、repr。

    ==================================================================
    为什么 BaseEstimator 不定义 fit / predict 的抽象方法？
    ==================================================================
        因为 sklearn 采用鸭子类型而非抽象基类。一个对象只要有 fit 方法
        就是估计器，有 predict 就是预测器，有 transform 就是转换器。
        这种松散约定带来了极大的灵活性（比如 Pipeline 可以组合任意估计器），
        代价是缺少编译期检查，但通过 check_estimator 测试套件来弥补。

    子类约定：
        __init__ 必须把所有超参数原样存为同名属性，不做任何计算。
        例如 def __init__(self, C=1.0): self.C = C
    """

    @classmethod
    def _get_param_names(cls):
        """获取本类的超参数名列表（类方法，便于子类调用）。"""
        return _get_param_names(cls)

    def get_params(self, deep=True):
        """获取估计器的所有超参数。

        参数：
            deep: 是否递归获取嵌套估计器的参数。
                  例如 Pipeline(steps=[('clf', LogisticRegression(C=2))])
                  当 deep=True 时会返回 {'clf__C': 2, ...}。
                  这个嵌套命名规则是 GridSearchCV 能用 'clf__C' 搜索参数的基础。

        返回：
            dict[str, Any]: 参数名到参数值的映射

        实现原理：
            1. 用反射拿到 __init__ 的参数名
            2. 对每个参数名，从 self 上 getattr 取值
            3. 如果 deep=True 且值本身是 BaseEstimator，递归获取其参数，
               并用 '子参数名__孙参数名' 的格式展平
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                # 嵌套估计器：展平为 key__subkey 形式
                nested_params = value.get_params(deep=True)
                for sub_key, sub_value in nested_params.items():
                    out[f"{key}__{sub_key}"] = sub_value
                out[key] = value
            else:
                out[key] = value
        return out

    def set_params(self, **params):
        """设置估计器的超参数。

        参数：
            **params: 要设置的参数，支持嵌套命名 'step__param'

        返回：
            self（为了支持链式调用：est.set_params(a=1).set_params(b=2)）

        实现原理：
            1. 把 'step__param' 形式的键拆分，找到嵌套的估计器
            2. 对顶层参数直接 setattr
            3. 对嵌套参数，递归调用子估计器的 set_params

        为什么返回 self？
            方便链式调用，也方便在 Pipeline 等元估计器中统一处理。
        """
        if not params:
            return self

        # 分离顶层参数和嵌套参数
        nested_params = {}
        for key, value in params.items():
            if "__" in key:
                # 嵌套参数：step__param = value
                step_name, sub_param = key.split("__", 1)
                if step_name not in self._get_param_names():
                    raise ValueError(
                        f"无效参数 {step_name}：不是 {type(self).__name__} 的超参数"
                    )
                nested_params.setdefault(step_name, {})[sub_param] = value
            else:
                # 顶层参数
                if key not in self._get_param_names():
                    raise ValueError(
                        f"无效参数 {key}：不是 {type(self).__name__} 的超参数"
                    )
                setattr(self, key, value)

        # 递归设置嵌套参数
        for step_name, sub_params in nested_params.items():
            sub_estimator = getattr(self, step_name)
            sub_estimator.set_params(**sub_params)

        return self

    def __repr__(self, n_char_max=700):
        """自动生成可读的字符串表示。

        为什么自动生成？
            因为手写 __repr__ 太繁琐，且容易漏参数。自动从 get_params 生成
            能保证 repr 与参数永远一致。

        输出格式示例：
            LogisticRegression(C=1.0, max_iter=100, penalty='l2')
        """
        class_name = type(self).__name__
        params = self.get_params(deep=False)

        # 按参数名排序，保证 repr 稳定
        sorted_params = sorted(params.items())
        params_str = ", ".join(f"{k}={v!r}" for k, v in sorted_params)

        repr_str = f"{class_name}({params_str})"
        if len(repr_str) > n_char_max:
            # 过长时截断，避免打印巨型估计器时刷屏
            repr_str = repr_str[:n_char_max - 3] + "..."
        return repr_str

    def __sklearn_clone__(self):
        """clone 钩子方法。

        子类可以覆盖此方法来自定义克隆行为。默认实现走 clone() 函数。
        这个钩子的存在是为了给某些特殊估计器（如 FunctionTransformer）
        留出逃生通道。
        """
        return clone(self, safe=False)