"""minisklearn 异常定义。

本模块定义了 minisklearn 中使用的自定义异常。这些异常的存在本身就是一个
设计决策：为什么不直接用 Python 内置异常？

设计动机：
    1. NotFittedError —— 区分"模型未训练"和"其他属性错误"。
       如果用 AttributeError，调用者很难判断是参数写错了还是忘了 fit。
       语义明确的异常能让错误链路更清晰。

    2. 保留与 sklearn 一致的异常名，方便用户迁移和查阅文档。
"""


class NotFittedError(Exception):
    """当在未训练的估计器上调用 predict / transform 等方法时抛出。

    设计要点：
        继承自 Exception 而非 ValueError 或 RuntimeError，是为了
        让用户可以精确捕获这一类错误，而不误伤其他 ValueError。

    典型用法：
        if not hasattr(self, "coef_"):
            raise NotFittedError("请先调用 fit() 训练模型")
    """
    pass