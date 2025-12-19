"""
SymPy证明引擎：使用SymPy进行形式化证明
将算子代码转换为SymPy表达式，然后使用simplify(LHS - RHS) == 0进行证明
"""

import sympy as sp
from typing import Callable, Any, List, Optional, Dict
import numpy as np

from ir.schema import MetamorphicRelation
from core.logger import get_logger


class SymPyProver:
    """SymPy形式化证明引擎"""

    def __init__(self):
        self.logger = get_logger()
        # 算子到SymPy操作的映射表
        self.operator_mapping = self._init_operator_mapping()

    def _init_operator_mapping(self) -> Dict[str, Callable]:
        """
        初始化算子到SymPy操作的映射表

        返回:
            映射字典 {算子名称: SymPy操作}
        """
        return {
            # 基本算术运算
            "Add": lambda x, y: sp.Add(x, y),
            "add": lambda x, y: sp.Add(x, y),
            "Multiply": lambda x, y: sp.Mul(x, y),
            "multiply": lambda x, y: sp.Mul(x, y),
            "Subtract": lambda x, y: sp.Add(x, -y),
            "subtract": lambda x, y: sp.Add(x, -y),
            "Divide": lambda x, y: sp.Mul(x, sp.Pow(y, -1)),
            "divide": lambda x, y: sp.Mul(x, sp.Pow(y, -1)),
            # 数学函数
            "Power": lambda x, y: sp.Pow(x, y),
            "power": lambda x, y: sp.Pow(x, y),
            "Exp": lambda x: sp.exp(x),
            "exp": lambda x: sp.exp(x),
            "Log": lambda x: sp.log(x),
            "log": lambda x: sp.log(x),
            "Sqrt": lambda x: sp.sqrt(x),
            "sqrt": lambda x: sp.sqrt(x),
            # 三角函数
            "Sin": lambda x: sp.sin(x),
            "sin": lambda x: sp.sin(x),
            "Cos": lambda x: sp.cos(x),
            "cos": lambda x: sp.cos(x),
            "Tan": lambda x: sp.tan(x),
            "tan": lambda x: sp.tan(x),
            # 最大值最小值
            "Max": lambda x, y: sp.Max(x, y),
            "max": lambda x, y: sp.Max(x, y),
            "Min": lambda x, y: sp.Min(x, y),
            "min": lambda x, y: sp.Min(x, y),
            # 绝对值
            "Abs": lambda x: sp.Abs(x),
            "abs": lambda x: sp.Abs(x),
        }

    def code_to_sympy(self, operator_name: str, num_inputs: int) -> Optional[Callable]:
        """
        将算子名称转换为SymPy表达式生成函数

        Args:
            operator_name: 算子名称
            num_inputs: 输入数量

        Returns:
            SymPy表达式生成函数，如果无法转换则返回None
        """
        # 查找映射
        if operator_name in self.operator_mapping:
            op_func = self.operator_mapping[operator_name]
            return op_func

        # 尝试规范化名称（去除框架前缀）
        normalized_name = operator_name.split(".")[-1]  # 如 "torch.add" -> "add"
        if normalized_name in self.operator_mapping:
            op_func = self.operator_mapping[normalized_name]
            return op_func

        self.logger.warning(
            f"Cannot map operator '{operator_name}' to SymPy expression. "
            f"Available operators: {list(self.operator_mapping.keys())}"
        )
        return None

    def create_sympy_expression(
        self, operator_func: Callable, num_inputs: int, operator_name: str = None
    ) -> Optional[sp.Expr]:
        """
        创建SymPy表达式

        Args:
            operator_func: 算子函数（用于推断类型）
            num_inputs: 输入数量
            operator_name: 算子名称（可选，用于查找映射）

        Returns:
            SymPy表达式，如果无法创建则返回None
        """
        # 如果提供了算子名称，尝试直接映射
        if operator_name:
            sympy_op = self.code_to_sympy(operator_name, num_inputs)
            if sympy_op:
                # 创建符号变量
                symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]
                if num_inputs == 1:
                    return sympy_op(symbols[0])
                elif num_inputs == 2:
                    return sympy_op(symbols[0], symbols[1])
                else:
                    # 对于多输入，尝试递归应用
                    result = symbols[0]
                    for i in range(1, num_inputs):
                        result = sympy_op(result, symbols[i])
                    return result

        # 如果无法映射，返回None
        return None

    def prove_mr(
        self, operator_name: str, mr: MetamorphicRelation, num_inputs: int
    ) -> tuple[bool, Optional[str]]:
        """
        使用SymPy证明MR

        Args:
            operator_name: 算子名称
            mr: 蜕变关系
            num_inputs: 输入数量

        Returns:
            (是否证明成功, 错误信息)
        """
        try:
            # 创建SymPy表达式
            sympy_expr = self.create_sympy_expression(None, num_inputs, operator_name)
            if sympy_expr is None:
                return (
                    False,
                    f"Cannot convert operator '{operator_name}' to SymPy expression",
                )

            # 创建符号变量
            symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

            # 构建LHS（原始表达式）
            if num_inputs == 1:
                lhs = sympy_expr.subs({symbols[0]: symbols[0]})
            elif num_inputs == 2:
                lhs = sympy_expr.subs({symbols[0]: symbols[0], symbols[1]: symbols[1]})
            else:
                lhs = sympy_expr

            # 构建RHS（变换后的表达式）
            # 应用MR变换到符号变量
            transformed_symbols = mr.transform(*symbols)
            if not isinstance(transformed_symbols, (tuple, list)):
                transformed_symbols = (transformed_symbols,)

            # 重新创建表达式，使用变换后的符号
            if num_inputs == 1:
                rhs = sympy_expr.subs({symbols[0]: transformed_symbols[0]})
            elif num_inputs == 2:
                rhs = sympy_expr.subs(
                    {
                        symbols[0]: transformed_symbols[0],
                        symbols[1]: transformed_symbols[1],
                    }
                )
            else:
                # 对于多输入，需要更复杂的处理
                subs_dict = {
                    symbols[i]: transformed_symbols[i]
                    for i in range(min(len(symbols), len(transformed_symbols)))
                }
                rhs = sympy_expr.subs(subs_dict)

            # 根据期望关系类型进行证明
            if mr.expected == "equal":
                # 证明 LHS == RHS，即 simplify(LHS - RHS) == 0
                diff = sp.simplify(lhs - rhs)
                is_proven = diff == 0

            elif mr.expected == "proportional":
                # 证明 LHS == k * RHS 或 LHS == -RHS
                # 检查 LHS + RHS == 0 (负比例)
                diff1 = sp.simplify(lhs + rhs)
                # 检查 LHS / RHS 是常数（正比例）
                try:
                    ratio = sp.simplify(lhs / rhs)
                    # 如果ratio是常数且不等于0，则成比例
                    is_proven = (diff1 == 0) or (ratio.is_constant())
                except:
                    is_proven = diff1 == 0

            else:
                # 其他关系类型，默认使用相等检查
                diff = sp.simplify(lhs - rhs)
                is_proven = diff == 0

            if is_proven:
                self.logger.debug(f"MR proven: {mr.description}")
                return True, None
            else:
                error_msg = (
                    f"SymPy proof failed: {diff if 'diff' in locals() else 'unknown'}"
                )
                return False, error_msg

        except Exception as e:
            error_msg = f"SymPy proof error: {str(e)}"
            self.logger.debug(f"SymPy proof exception: {error_msg}")
            return False, error_msg

    def prove_mrs(
        self, operator_name: str, mrs: List[MetamorphicRelation], num_inputs: int
    ) -> List[MetamorphicRelation]:
        """
        证明MR列表，返回经过证明的MR

        Args:
            operator_name: 算子名称
            mrs: MR列表
            num_inputs: 输入数量

        Returns:
            经过证明的MR列表
        """
        proven_mrs = []

        self.logger.info(f"Proving {len(mrs)} MRs using SymPy...")

        for i, mr in enumerate(mrs):
            is_proven, error_msg = self.prove_mr(operator_name, mr, num_inputs)

            if is_proven:
                proven_mrs.append(mr)
                self.logger.debug(f"MR {i+1} proven: {mr.description}")
            else:
                self.logger.debug(
                    f"MR {i+1} proof failed: {mr.description}. " f"Reason: {error_msg}"
                )

        self.logger.info(
            f"SymPy proof completed: {len(proven_mrs)}/{len(mrs)} MRs proven"
        )

        return proven_mrs
