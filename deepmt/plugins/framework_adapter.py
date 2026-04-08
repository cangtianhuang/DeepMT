"""
框架适配器：将 transform_code 字符串绑定为可执行函数

职责范围（单一）：
  - bind_transform_code：将 lambda 字符串编译为可调用对象，
    注入 apply_operator 占位符解析

oracle 表达式的评估已移至 deepmt/analysis/mr_verifier.py（MRVerifier）。
"""

from typing import Any, Callable, Optional

from deepmt.core.logger import logger


class FrameworkAdapter:
    """框架适配器：负责 transform_code 的字符串→函数绑定"""

    def __init__(self, plugin: Any):
        """
        Args:
            plugin: FrameworkPlugin 实例（持有引用供将来 transform 扩展使用）
        """
        self.plugin = plugin

    def bind_transform_code(
        self,
        transform_code: str,
        operator_func: Callable,
    ) -> Optional[Callable]:
        """
        将 transform_code lambda 字符串编译为可执行函数。

        处理 apply_operator() 占位符：若表达式中包含 apply_operator，
        则将其绑定为 operator_func。

        Args:
            transform_code: lambda 表达式字符串
                例："lambda k: {**k, 'input': 2.0 * k['input']}"
                例："lambda k: {**k, 'input': apply_operator(k['input'])}"
            operator_func:  具体的算子函数（用于 apply_operator 绑定）

        Returns:
            绑定后的可调用对象，失败返回 None
        """
        try:
            safe_dict: dict = {}
            if "apply_operator" in transform_code:
                safe_dict["apply_operator"] = operator_func

            func = eval(transform_code, {"__builtins__": {}}, safe_dict)

            if not callable(func):
                logger.warning(
                    f"transform_code compiled to non-callable: {type(func)}"
                )
                return None

            return func

        except Exception as e:
            logger.error(f"Failed to bind transform_code: {e}")
            logger.debug(f"transform_code: {transform_code}")
            return None
