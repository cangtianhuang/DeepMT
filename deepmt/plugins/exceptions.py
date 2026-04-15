"""
插件层统一异常。

所有 FrameworkPlugin 实现在可预期失败点应抛出以下异常之一，便于
上层（CrossFrameworkTester / BatchTestRunner / health checker）分支处理。
"""


class FrameworkPluginError(Exception):
    """插件层错误基类"""


class OperatorNotMapped(FrameworkPluginError):
    """算子名称在当前插件的映射表中不存在。"""

    def __init__(self, framework: str, operator: str) -> None:
        super().__init__(
            f"{framework}: operator {operator!r} is not mapped to an implementation."
        )
        self.framework = framework
        self.operator = operator


class PrimitiveUnsupported(FrameworkPluginError):
    """插件声称自己是 FrameworkPlugin 但缺失某个必需原语实现。"""

    def __init__(self, framework: str, primitive: str) -> None:
        super().__init__(
            f"{framework}: required primitive {primitive!r} is not implemented."
        )
        self.framework = framework
        self.primitive = primitive


class FrameworkRuntimeError(FrameworkPluginError):
    """框架原生调用期间的运行时错误（算子执行崩溃、形状不兼容等）。"""

    def __init__(self, framework: str, operator: str, detail: str) -> None:
        super().__init__(f"{framework}: runtime error in {operator!r}: {detail}")
        self.framework = framework
        self.operator = operator
        self.detail = detail
