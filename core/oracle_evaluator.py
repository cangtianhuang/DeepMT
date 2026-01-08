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
            expr: 数学表达式（如 "orig == trans"）
            framework: 框架名称
            tolerance: 容差

        Returns:
            框架特定的 Python 代码字符串
        """
        # 检查是否包含比较操作符
        has_comparison = any(op in expr for op in ["==", "!=", "<", ">", "<=", ">="])

        if not has_comparison:
            # 如果没有显式比较，添加容差检查
            return self._generate_tolerance_check(expr, framework, tolerance)
        else:
            # 如果有显式比较，直接使用（框架会自动处理）
            return f"return {expr}"

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
    
    # 框架特定的相等比较
    def all_equal(a, b, tol=tolerance):
        if hasattr(a, 'shape') and hasattr(b, 'shape'):
            # 张量/数组比较
            if a.shape != b.shape:
                return False
            
            # 转换为 numpy 进行比较
            try:
                if hasattr(a, 'detach'):
                    a = a.detach().cpu().numpy()
                if hasattr(b, 'detach'):
                    b = b.detach().cpu().numpy()
                elif hasattr(a, 'numpy'):
                    a = a.numpy()
                elif hasattr(b, 'numpy'):
                    b = b.numpy()
                    
                return bool(np.allclose(a, b, atol=tol, rtol=tol))
            except:
                return False
        else:
            # 标量比较
            return abs(a - b) <= tol
    
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
