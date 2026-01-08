"""
Oracle 表达式评估器：框架无关的 MR 验证表达式解析

将简单的数学表达式（如 "orig == trans"）转换为框架特定的代码并执行
"""

import re
from typing import Any, Callable, Dict, Tuple

from core.logger import get_logger


class OracleEvaluator:
    """Oracle 表达式评估器：将数学表达式转换为可执行代码"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def compile_expression(
        self,
        expr: str,
        framework: str = "pytorch",
        tolerance: float = 1e-6,
    ) -> Callable[[Any, Any, Any], bool]:
        """
        将数学表达式编译为可执行函数

        Args:
            expr: 数学表达式字符串（如 "orig == trans"）
            framework: 框架名称（"pytorch", "tensorflow", "paddle"）
            tolerance: 数值容差

        Returns:
            验证函数：def oracle(orig, trans, x=None) -> bool
        """
        try:
            # 规范化表达式
            expr = self._normalize_expression(expr)

            # 根据框架生成代码
            code = self._generate_code(expr, framework, tolerance)

            # 编译为函数
            return self._compile_code(code, framework)

        except Exception as e:
            self.logger.error(f"Failed to compile oracle expression '{expr}': {e}")
            raise

    def _normalize_expression(self, expr: str) -> str:
        """
        规范化表达式

        - 将 `==` 转换为框架特定的相等比较
        - 处理容差
        """
        return expr.strip()

    def _generate_code(self, expr: str, framework: str, tolerance: float) -> str:
        """
        生成框架特定的比较代码

        Args:
            expr: 数学表达式（如 "orig == trans", "all(trans == 0)", "orig + trans == abs(x)"）
            framework: 框架名称
            tolerance: 容差

        Returns:
            框架特定的 Python 代码字符串
        """
        # 检查是否包含 all() 包装
        if expr.startswith("all(") and expr.endswith(")"):
            # 提取内部表达式
            inner_expr = expr[4:-1].strip()
            # 生成元素级检查代码
            return self._generate_elementwise_check(inner_expr, framework, tolerance)

        # 检查是否包含比较操作符
        has_comparison = any(op in expr for op in ["==", "!=", "<", ">", "<=", ">="])

        if not has_comparison:
            # 如果没有显式比较，添加容差检查
            return self._generate_tolerance_check(expr, framework, tolerance)
        else:
            # 如果有显式比较，简单比较使用 all_equal
            # 复杂表达式（包含运算）使用 numpy 处理
            if any(op in expr for op in ["+", "-", "*", "/"]):
                # 复杂表达式，需要转换为 numpy
                return self._generate_complex_expression(expr, framework, tolerance)
            else:
                # 简单比较，使用 all_equal
                return f"return all_equal(orig, trans, {tolerance})"

    def _generate_complex_expression(
        self, expr: str, framework: str, tolerance: float
    ) -> str:
        """
        生成复杂表达式的代码（包含数学运算）

        处理如 "orig + trans == abs(x)" 这样的复杂表达式。

        Args:
            expr: 复杂表达式
            framework: 框架名称
            tolerance: 容差

        Returns:
            生成的代码字符串
        """
        # 分割比较符
        for op in ["==", "!=", "<=", ">=", "<", ">"]:
            if op in expr:
                parts = expr.split(op)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    # 使用 numpy 处理表达式
                    return f"return np.allclose(to_numpy({left}), to_numpy({right}), atol={tolerance})"

        # 默认：返回 bool
        return f"result = {expr}\n    return bool(result)"

    def _generate_elementwise_check(
        self, expr: str, framework: str, tolerance: float
    ) -> str:
        """
        生成元素级检查代码（用于 all() 语法）

        将 all(expr) 转换为对每个元素的检查。

        Args:
            expr: 内部表达式（如 "trans == 0"）
            framework: 框架名称
            tolerance: 容差

        Returns:
            元素级检查代码
        """
        # 提取比较操作符和值
        if "==" in expr:
            parts = expr.split("==")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                # 生成元素级相等检查
                if (
                    left == "orig"
                    and right == "trans"
                    or left == "trans"
                    and right == "orig"
                ):
                    return f"return all_equal(orig, trans, {tolerance})"
                elif right in ["0", "0.0"]:
                    return f"return all_equal({left}, 0, {tolerance})"
                elif left in ["0", "0.0"]:
                    return f"return all_equal({right}, 0, {tolerance})"
                else:
                    return f"return all_equal({left}, {right}, {tolerance})"
        elif ">=" in expr:
            parts = expr.split(">=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return f"return np.all(to_numpy({left}) >= to_numpy({right}) - {tolerance})"
        elif "<=" in expr:
            parts = expr.split("<=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return f"return np.all(to_numpy({left}) <= to_numpy({right}) + {tolerance})"
        elif ">" in expr:
            parts = expr.split(">")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return (
                    f"return np.all(to_numpy({left}) > to_numpy({right}) + {tolerance})"
                )
        elif "<" in expr:
            parts = expr.split("<")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return (
                    f"return np.all(to_numpy({left}) < to_numpy({right}) - {tolerance})"
                )

        # 默认：直接执行表达式并返回结果
        return f"result = {expr}\n    return bool(result)"

    def _generate_tolerance_check(
        self, expr: str, framework: str, tolerance: float
    ) -> str:
        """
        生成带容差的相等检查代码

        将 "expr" 转换为 "abs(expr) <= tolerance"
        """
        # 提取原始表达式（如果 expr 已经是比较）
        if "==" in expr or "!=" in expr:
            # 已经是比较，直接返回
            return f"return {expr}"

        # 添加容差检查
        return f"return abs({expr}) <= {tolerance}"

    def _compile_code(
        self, code: str, framework: str
    ) -> Callable[[Any, Any, Any], bool]:
        """
        编译代码为可执行函数

        Args:
            code: Python 代码字符串
            framework: 框架名称（用于导入正确的库）

        Returns:
            可执行函数：def oracle(orig, trans, x=None) -> bool
        """
        # 根据框架导入相应的库
        if framework.lower() == "pytorch":
            import_stmt = "import torch as framework_lib"
            allclose_func = "torch.allclose"
        elif framework.lower() == "tensorflow":
            import_stmt = "import tensorflow as tf"
            allclose_func = "tf.reduce_all(tf.abs(orig - trans) < tolerance)"
        elif framework.lower() in ["paddle", "paddlepaddle"]:
            import_stmt = "import paddle as framework_lib"
            allclose_func = "paddle.allclose"
        else:
            # 默认使用 NumPy
            import_stmt = "import numpy as framework_lib"
            allclose_func = "np.allclose"

        # 构建完整的函数代码
        func_code = f"""
def oracle(orig, trans, x=None, tolerance=1e-6):
    {import_stmt}
    import numpy as np

    # 辅助函数：转换为 numpy
    def to_numpy(arr):
        if hasattr(arr, 'detach'):
            return arr.detach().cpu().numpy()
        elif hasattr(arr, 'numpy'):
            return arr.numpy()
        else:
            return np.array(arr)

    # 框架特定的相等比较
    def all_equal(a, b, tol=tolerance):
        if hasattr(a, 'shape') and hasattr(b, 'shape'):
            # 张量/数组比较
            if a.shape != b.shape:
                return False

            # 转换为 numpy 进行比较
            try:
                a_np = to_numpy(a)
                b_np = to_numpy(b)
                return bool(np.allclose(a_np, b_np, atol=tol, rtol=tol))
            except:
                return False
        else:
            # 标量比较
            return abs(a - b) <= tol

    # 元素级大于等于比较
    def all_geq(a, b, tol=tolerance):
        a_np = to_numpy(a)
        b_np = to_numpy(b)
        return bool(np.all(a_np >= b_np - tol))

    # 元素级小于等于比较
    def all_leq(a, b, tol=tolerance):
        a_np = to_numpy(a)
        b_np = to_numpy(b)
        return bool(np.all(a_np <= b_np + tol))

    # 确保 abs 函数可用（用于标量和数组）
    abs_func = abs

    # 执行用户表达式（支持框架特定操作）
    {code}
"""

        # 编译代码
        namespace = {}
        exec(func_code, namespace)

        return namespace["oracle"]

    def evaluate(
        self,
        expr: str,
        orig: Any,
        trans: Any,
        x: Any = None,
        framework: str = "pytorch",
        tolerance: float = 1e-6,
    ) -> bool:
        """
        直接评估表达式

        Args:
            expr: 数学表达式字符串
            orig: 原始输出
            trans: 变换后输出
            x: 原始输入（可选）
            framework: 框架名称
            tolerance: 容差

        Returns:
            表达式结果（True/False）
        """
        try:
            oracle_func = self.compile_expression(expr, framework, tolerance)
            return oracle_func(orig, trans, x, tolerance)

        except Exception as e:
            self.logger.error(f"Failed to evaluate expression '{expr}': {e}")
            raise
