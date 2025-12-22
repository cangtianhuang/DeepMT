"""
AST到SymPy解析器：解析Python AST并转换为SymPy表达式
算子相关功能，用于将Python代码转换为SymPy表达式
"""

import ast
import re
from typing import List, Optional

import sympy as sp

from core.logger import get_logger


class ASTToSymPyParser:
    """
    AST到SymPy解析器
    解析Python AST并转换为SymPy表达式
    """

    def __init__(self):
        self.logger = get_logger()
        # AST节点到SymPy操作的映射
        self.ast_to_sympy = {
            ast.Add: lambda a, b: sp.Add(a, b),
            ast.Sub: lambda a, b: sp.Add(a, sp.Mul(-1, b)),
            ast.Mult: lambda a, b: sp.Mul(a, b),
            ast.Div: lambda a, b: sp.Mul(a, sp.Pow(b, -1)),
            ast.Pow: lambda a, b: sp.Pow(a, b),
            ast.USub: lambda a: sp.Mul(-1, a),
            ast.UAdd: lambda a: a,
        }

        # 函数名到SymPy函数的映射
        self.func_mapping = {
            "abs": sp.Abs,
            "max": sp.Max,
            "min": sp.Min,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
        }

    def parse_to_sympy(self, code: str, num_inputs: int = None) -> Optional[sp.Expr]:
        """
        解析Python代码为SymPy表达式

        Args:
            code: Python代码字符串
            num_inputs: 输入数量（用于创建符号变量）

        Returns:
            SymPy表达式
        """
        try:
            tree = ast.parse(code)

            # 查找函数定义
            func_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_node = node
                    break

            if func_node is None:
                self.logger.warning("No function definition found in code")
                return None

            # 推断输入数量
            if num_inputs is None:
                num_inputs = len(func_node.args.args)

            # 创建符号变量
            symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

            # 解析函数体
            if len(func_node.body) == 1 and isinstance(func_node.body[0], ast.Return):
                return self._parse_expr(func_node.body[0].value, symbols)
            else:
                # 处理多个语句（简化：只处理最后一个return）
                for stmt in reversed(func_node.body):
                    if isinstance(stmt, ast.Return):
                        return self._parse_expr(stmt.value, symbols)

            return None

        except Exception as e:
            self.logger.warning(f"AST parsing error: {e}")
            return None

    def _parse_expr(self, node: ast.AST, symbols: List[sp.Symbol]) -> sp.Expr:
        """递归解析AST表达式节点"""
        if isinstance(node, ast.Name):
            name = node.id
            match = re.match(r"x(\d+)", name)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(symbols):
                    return symbols[idx]
            return sp.Symbol(name)

        elif isinstance(node, ast.Constant):
            return (
                sp.Integer(node.value)
                if isinstance(node.value, int)
                else sp.Float(node.value)
            )

        elif isinstance(node, ast.Num):  # Python < 3.8
            return sp.Integer(node.n) if isinstance(node.n, int) else sp.Float(node.n)

        elif isinstance(node, ast.BinOp):
            op_func = self.ast_to_sympy.get(type(node.op))
            if op_func:
                left = self._parse_expr(node.left, symbols)
                right = self._parse_expr(node.right, symbols)
                return op_func(left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        elif isinstance(node, ast.UnaryOp):
            op_func = self.ast_to_sympy.get(type(node.op))
            if op_func:
                operand = self._parse_expr(node.operand, symbols)
                return op_func(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                sympy_func = self.func_mapping.get(func_name)
                if sympy_func:
                    args = [self._parse_expr(arg, symbols) for arg in node.args]
                    return sympy_func(*args)
                else:
                    args = [self._parse_expr(arg, symbols) for arg in node.args]
                    func_symbol = sp.Symbol(func_name)
                    return sp.Function(func_symbol)(*args)

        elif isinstance(node, ast.IfExp):
            condition = self._parse_expr(node.test, symbols)
            true_expr = self._parse_expr(node.body, symbols)
            false_expr = self._parse_expr(node.orelse, symbols)
            return sp.Piecewise((true_expr, condition), (false_expr, True))

        elif isinstance(node, ast.Compare):
            left = self._parse_expr(node.left, symbols)
            if len(node.comparators) == 1:
                right = self._parse_expr(node.comparators[0], symbols)
                op = type(node.ops[0])
                if op == ast.Eq:
                    return sp.Eq(left, right)
                elif op == ast.Lt:
                    return left < right
                elif op == ast.Gt:
                    return left > right
                elif op == ast.LtE:
                    return left <= right
                elif op == ast.GtE:
                    return left >= right

        return sp.Symbol("unknown")
