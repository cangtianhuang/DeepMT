"""
AST到SymPy解析器：解析Python AST并转换为SymPy表达式
"""

import ast
from typing import Dict, List, Optional

import sympy as sp

from core.logger import get_logger


class ASTParser:
    """
    AST到SymPy解析器
    解析Python AST并转换为SymPy表达式
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        # AST二元操作符到SymPy操作的映射
        self.binop_mapping = {
            ast.Add: lambda a, b: sp.Add(a, b),
            ast.Sub: lambda a, b: sp.Add(a, sp.Mul(-1, b)),
            ast.Mult: lambda a, b: sp.Mul(a, b),
            ast.Div: lambda a, b: sp.Mul(a, sp.Pow(b, -1)),
            ast.FloorDiv: lambda a, b: sp.floor(a / b),
            ast.Mod: lambda a, b: sp.Mod(a, b),
            ast.Pow: lambda a, b: sp.Pow(a, b),
        }

        # AST一元操作符到SymPy操作的映射
        self.unaryop_mapping = {
            ast.USub: lambda a: sp.Mul(-1, a),
            ast.UAdd: lambda a: a,
            ast.Not: lambda a: sp.Not(a),
        }

        # 函数名到SymPy函数的映射
        self.func_mapping = {
            # 基础数学函数
            "abs": sp.Abs,
            "max": sp.Max,
            "min": sp.Min,
            "pow": sp.Pow,
            "round": lambda x: sp.floor(x + sp.Rational(1, 2)),
            # 指数和对数
            "exp": sp.exp,
            "log": sp.log,
            "log10": lambda x: sp.log(x, 10),
            "log2": lambda x: sp.log(x, 2),
            "sqrt": sp.sqrt,
            # 三角函数
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            # 取整函数
            "floor": sp.floor,
            "ceil": sp.ceiling,
            # 符号函数
            "sign": sp.sign,
        }

    def parse_to_sympy(self, code: str, num_inputs: int = None) -> Optional[sp.Expr]:
        """
        解析Python代码为SymPy表达式

        Args:
            code: Python代码字符串
            num_inputs: 输入数量（用于创建符号变量，如果为None则自动推断）

        Returns:
            SymPy表达式，如果解析失败则返回None
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

            # 获取参数名列表
            param_names = [arg.arg for arg in func_node.args.args]

            # 推断输入数量
            if num_inputs is None:
                num_inputs = len(param_names)

            # 创建符号变量（使用标准命名 x0, x1, ...）
            symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

            # 建立参数名到符号的映射
            param_to_symbol: Dict[str, sp.Symbol] = {}
            for i, param_name in enumerate(param_names):
                if i < len(symbols):
                    param_to_symbol[param_name] = symbols[i]

            # 解析函数体
            return self._parse_function_body(func_node.body, symbols, param_to_symbol)

        except SyntaxError as e:
            self.logger.warning(f"Syntax error in code: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"AST parsing error: {e}")
            return None

    def _parse_function_body(
        self,
        body: List[ast.stmt],
        symbols: List[sp.Symbol],
        param_to_symbol: Dict[str, sp.Symbol],
    ) -> Optional[sp.Expr]:
        """解析函数体，提取return表达式"""
        # 简单情况：只有一个return语句
        if len(body) == 1 and isinstance(body[0], ast.Return):
            return self._parse_expr(body[0].value, symbols, param_to_symbol)

        # 复杂情况：多个语句，查找最后一个return
        for stmt in reversed(body):
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                return self._parse_expr(stmt.value, symbols, param_to_symbol)

        self.logger.warning("No return statement found in function body")
        return None

    def _parse_expr(
        self,
        node: ast.AST,
        symbols: List[sp.Symbol],
        param_to_symbol: Dict[str, sp.Symbol],
    ) -> sp.Expr:
        """递归解析AST表达式节点"""

        # 变量名
        if isinstance(node, ast.Name):
            name = node.id
            # 优先从参数映射中查找
            if name in param_to_symbol:
                return param_to_symbol[name]
            # 尝试匹配 x0, x1, ... 格式
            if name.startswith("x") and name[1:].isdigit():
                idx = int(name[1:])
                if 0 <= idx < len(symbols):
                    return symbols[idx]
            # 返回通用符号
            return sp.Symbol(name)

        # 常量（Python 3.8+）
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return sp.true if value else sp.false
            elif isinstance(value, int):
                return sp.Integer(value)
            elif isinstance(value, float):
                return sp.Float(value)
            else:
                return sp.Symbol(str(value))

        # 常量（Python < 3.8）
        elif isinstance(node, ast.Num):
            return sp.Integer(node.n) if isinstance(node.n, int) else sp.Float(node.n)

        # 二元操作
        elif isinstance(node, ast.BinOp):
            op_func = self.binop_mapping.get(type(node.op))
            if op_func:
                left = self._parse_expr(node.left, symbols, param_to_symbol)
                right = self._parse_expr(node.right, symbols, param_to_symbol)
                return op_func(left, right)
            else:
                raise ValueError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )

        # 一元操作
        elif isinstance(node, ast.UnaryOp):
            op_func = self.unaryop_mapping.get(type(node.op))
            if op_func:
                operand = self._parse_expr(node.operand, symbols, param_to_symbol)
                return op_func(operand)
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )

        # 函数调用
        elif isinstance(node, ast.Call):
            return self._parse_call(node, symbols, param_to_symbol)

        # 条件表达式 (a if condition else b)
        elif isinstance(node, ast.IfExp):
            condition = self._parse_expr(node.test, symbols, param_to_symbol)
            true_expr = self._parse_expr(node.body, symbols, param_to_symbol)
            false_expr = self._parse_expr(node.orelse, symbols, param_to_symbol)
            return sp.Piecewise((true_expr, condition), (false_expr, True))

        # 比较操作
        elif isinstance(node, ast.Compare):
            return self._parse_compare(node, symbols, param_to_symbol)

        # 布尔操作 (and, or)
        elif isinstance(node, ast.BoolOp):
            return self._parse_boolop(node, symbols, param_to_symbol)

        # 元组（返回第一个元素，适用于单输出函数）
        elif isinstance(node, ast.Tuple):
            if node.elts:
                return self._parse_expr(node.elts[0], symbols, param_to_symbol)

        # 下标访问（简单处理）
        elif isinstance(node, ast.Subscript):
            # 对于简单的下标访问，尝试解析值
            return self._parse_expr(node.value, symbols, param_to_symbol)

        # 未支持的节点类型
        self.logger.warning(f"Unsupported AST node type: {type(node).__name__}")
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    def _parse_call(
        self,
        node: ast.Call,
        symbols: List[sp.Symbol],
        param_to_symbol: Dict[str, sp.Symbol],
    ) -> sp.Expr:
        """解析函数调用"""
        # 获取函数名
        func_name = self._get_func_name(node.func)

        # 解析参数
        args = [self._parse_expr(arg, symbols, param_to_symbol) for arg in node.args]

        # 查找映射的SymPy函数
        sympy_func = self.func_mapping.get(func_name)
        if sympy_func:
            return sympy_func(*args)

        # 未知函数，创建符号函数
        self.logger.debug(f"Unknown function '{func_name}', creating symbolic function")
        return sp.Function(func_name)(*args)

    def _get_func_name(self, node: ast.AST) -> str:
        """获取函数名（支持简单名称和属性访问）"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # 处理 math.sqrt, np.abs 等
            # 只取最后的属性名
            return node.attr
        else:
            return "unknown_func"

    def _parse_compare(
        self,
        node: ast.Compare,
        symbols: List[sp.Symbol],
        param_to_symbol: Dict[str, sp.Symbol],
    ) -> sp.Expr:
        """解析比较操作"""
        left = self._parse_expr(node.left, symbols, param_to_symbol)

        if len(node.comparators) == 1:
            right = self._parse_expr(node.comparators[0], symbols, param_to_symbol)
            op = type(node.ops[0])

            if op == ast.Eq:
                return sp.Eq(left, right)
            elif op == ast.NotEq:
                return sp.Ne(left, right)
            elif op == ast.Lt:
                return sp.Lt(left, right)
            elif op == ast.Gt:
                return sp.Gt(left, right)
            elif op == ast.LtE:
                return sp.Le(left, right)
            elif op == ast.GtE:
                return sp.Ge(left, right)

        # 多重比较 (a < b < c)
        result = None
        current_left = left
        for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            right = self._parse_expr(comparator, symbols, param_to_symbol)
            comparison = self._make_comparison(current_left, type(op), right)
            result = comparison if result is None else sp.And(result, comparison)
            current_left = right

        return result if result is not None else sp.true

    def _make_comparison(self, left: sp.Expr, op_type: type, right: sp.Expr) -> sp.Expr:
        """创建单个比较表达式"""
        if op_type == ast.Eq:
            return sp.Eq(left, right)
        elif op_type == ast.NotEq:
            return sp.Ne(left, right)
        elif op_type == ast.Lt:
            return sp.Lt(left, right)
        elif op_type == ast.Gt:
            return sp.Gt(left, right)
        elif op_type == ast.LtE:
            return sp.Le(left, right)
        elif op_type == ast.GtE:
            return sp.Ge(left, right)
        else:
            return sp.true

    def _parse_boolop(
        self,
        node: ast.BoolOp,
        symbols: List[sp.Symbol],
        param_to_symbol: Dict[str, sp.Symbol],
    ) -> sp.Expr:
        """解析布尔操作 (and, or)"""
        values = [self._parse_expr(v, symbols, param_to_symbol) for v in node.values]

        if isinstance(node.op, ast.And):
            return sp.And(*values)
        elif isinstance(node.op, ast.Or):
            return sp.Or(*values)
        else:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
