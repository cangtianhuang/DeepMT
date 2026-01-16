"""
框架适配器插件：处理框架无关的 MR 到具体框架的绑定

将 FrameworkAdapter 重构为插件系统的一部分，便于统一管理。
"""

from typing import Any, Callable, Optional

from core.logger import get_logger
from core.oracle_evaluator import OracleEvaluator


class FrameworkAdapter:
    """框架适配器：将框架无关的 MR 绑定到具体框架"""

    def __init__(self, framework: str = "pytorch"):
        """
        初始化框架适配器

        Args:
            framework: 框架名称（pytorch, tensorflow, paddle）
        """
        self.logger = get_logger(self.__class__.__name__)
        self.framework = framework
        self.oracle_evaluator = OracleEvaluator()
        self.logger.debug(f"FrameworkAdapter initialized for {framework}")

    def bind_transform_code(
        self,
        transform_code: str,
        operator_func: Callable,
    ) -> Optional[Callable]:
        """
        将框架无关的 transform_code 绑定到具体的算子函数

        处理 transform_code 中的 apply_operator() 占位符。

        Args:
            transform_code: 框架无关的 lambda 代码
            operator_func: 具体的算子函数

        Returns:
            绑定后的可执行函数，如果失败则返回 None

        Examples:
            输入: "lambda k: {**k, 'input': apply_operator(k['input'])}"
            输出: lambda k: {**k, 'input': operator_func(k['input'])}

            输入: "lambda k: {**k, 'input': 2 * k['input']}"
            输出: lambda k: {**k, 'input': 2 * k['input']} (无需修改)
        """
        try:
            # 创建安全的执行环境
            safe_dict = {}

            if "apply_operator" in transform_code:
                # 添加 operator_func 到环境
                safe_dict["apply_operator"] = operator_func

            # 编译 transform_code
            func = eval(transform_code, {"__builtins__": {}}, safe_dict)

            if not callable(func):
                self.logger.warning(
                    f"transform_code is not callable after binding: {type(func)}"
                )
                return None

            return func

        except Exception as e:
            self.logger.error(f"Failed to bind transform_code: {e}")
            self.logger.debug(f"Transform code: {transform_code}")
            return None

    def bind_oracle_expr(
        self,
        oracle_expr: str,
        operator_func: Optional[Callable] = None,
        tolerance: float = 1e-6,
    ) -> Callable[[Any, Any, Any], bool]:
        """
        将框架无关的 oracle_expr 绑定到具体的验证函数

        处理 oracle_expr 中的数学表达式，转换为框架特定的验证代码。

        Args:
            oracle_expr: 框架无关的表达式（如 "orig == trans", "all(trans == 0)"）
            operator_func: 算子函数（用于嵌套调用，如 "apply_operator(trans)"）
            tolerance: 数值容差

        Returns:
            验证函数：def oracle(orig, trans, x=None) -> bool

        Examples:
            输入: "orig == trans"
            输出: 框架特定的相等检查函数

            输入: "all(trans == 0)"
            输出: 框架特定的元素级相等检查函数

            输入: "trans == 2 * orig"
            输出: 框架特定的比例检查函数
        """
        try:
            # 预处理表达式：处理 all() 语法
            processed_expr = self._preprocess_oracle_expr(oracle_expr)

            # 使用 OracleEvaluator 编译表达式
            return self.oracle_evaluator.compile_expression(
                expr=processed_expr,
                framework=self.framework,
                tolerance=tolerance,
            )

        except Exception as e:
            self.logger.error(f"Failed to bind oracle_expr: {e}")
            self.logger.debug(f"Oracle expression: {oracle_expr}")
            raise

    def _preprocess_oracle_expr(self, oracle_expr: str) -> str:
        """
        预处理 oracle_expr 表达式

        处理特殊语法，如：
        - all(expr): 元素级别的检查
        - apply_operator(x): 应用算子（需要在运行时处理）

        Args:
            oracle_expr: 原始表达式

        Returns:
            处理后的表达式
        """
        expr = oracle_expr.strip()

        # 处理 all(expr) 语法
        # 这在 OracleEvaluator 中已经通过 all_equal 函数处理
        # 我们只需要确保 all() 的参数被正确提取
        if expr.startswith("all(") and expr.endswith(")"):
            # 提取 inner_expr
            inner_expr = expr[4:-1].strip()
            # OracleEvaluator 的 _compile_code 方法会处理张量比较
            # 所以我们只需要返回带 all 标记的表达式
            # 实际的比较逻辑由 OracleEvaluator 处理
            return expr

        return expr

    def get_framework(self) -> str:
        """
        获取当前框架名称

        Returns:
            框架名称
        """
        return self.framework
