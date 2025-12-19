"""
代码到SymPy转换器：将任意Python代码转换为SymPy表达式
使用LLM + AST解析实现自动转换
"""

import ast
import inspect
import sympy as sp
from typing import Callable, Any, Optional, Dict, List
import re

from core.logger import get_logger


class CodeToSymPyTranslator:
    """
    代码到SymPy转换器
    使用LLM将任意Python代码翻译为SymPy可解析的参考实现
    """

    def __init__(self, llm_client=None):
        """
        初始化转换器

        Args:
            llm_client: LLM客户端（可选，如果为None则使用基础AST解析）
        """
        self.logger = get_logger()
        self.llm_client = llm_client
        self.ast_parser = ASTToSymPyParser()

    def translate(
        self,
        code: Optional[str] = None,
        func: Optional[Callable] = None,
        doc: Optional[str] = None,
        signature: Optional[str] = None,
    ) -> Optional[sp.Expr]:
        """
        将代码转换为SymPy表达式

        Args:
            code: Python代码字符串
            func: Python函数对象
            doc: 函数文档字符串
            signature: 函数签名字符串

        Returns:
            SymPy表达式，如果转换失败则返回None
        """
        # 1. 获取代码和文档
        if func is not None:
            code = inspect.getsource(func) if code is None else code
            doc = inspect.getdoc(func) if doc is None else doc
            sig = str(inspect.signature(func)) if signature is None else signature
        else:
            sig = signature or ""

        if code is None:
            self.logger.warning("No code provided for translation")
            return None

        # 2. 尝试LLM翻译（如果有LLM客户端）
        if self.llm_client:
            sympy_code = self._llm_translate(code, doc, sig)
            if sympy_code:
                try:
                    # 执行LLM生成的SymPy代码
                    return self._execute_sympy_code(sympy_code)
                except Exception as e:
                    self.logger.warning(
                        f"LLM translation failed: {e}, falling back to AST"
                    )

        # 3. 回退到AST解析
        return self.ast_parser.parse_to_sympy(code)

    def _llm_translate(self, code: str, doc: str, signature: str) -> Optional[str]:
        """
        使用LLM翻译代码为SymPy表达式

        Args:
            code: Python代码
            doc: 文档字符串
            signature: 函数签名

        Returns:
            SymPy代码字符串
        """
        prompt = f"""你是一个代码分析专家。请将以下Python代码转换为SymPy表达式。

函数签名：{signature}

代码：
```python
{code}
```

文档：
{doc if doc else "无"}

要求：
1. 将代码转换为SymPy符号表达式
2. 使用SymPy的符号变量（如 sp.Symbol('x')）
3. 将Python操作转换为SymPy操作（如 + -> sp.Add, * -> sp.Mul）
4. 处理条件表达式使用 sp.Piecewise
5. 只返回SymPy表达式代码，不要包含其他说明

输出格式（Python代码）：
```python
import sympy as sp
# 创建符号变量
x, y = sp.symbols('x y')
# SymPy表达式
result = <SymPy表达式>
```

只返回代码块，不要包含markdown标记外的其他文字。
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a code analysis expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()

            # 提取代码块
            if "```python" in content:
                code_block = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                code_block = content.split("```")[1].split("```")[0].strip()
            else:
                code_block = content

            return code_block

        except Exception as e:
            self.logger.warning(f"LLM translation error: {e}")
            return None

    def _execute_sympy_code(self, code: str) -> Optional[sp.Expr]:
        """
        执行SymPy代码并返回表达式

        Args:
            code: SymPy代码字符串

        Returns:
            SymPy表达式
        """
        try:
            # 创建执行环境
            exec_globals = {"sp": sp, "sympy": sp}
            exec(code, exec_globals)

            # 查找result变量
            if "result" in exec_globals:
                return exec_globals["result"]
            else:
                # 尝试查找最后一个表达式
                # 简化处理：返回None，让AST解析器处理
                return None

        except Exception as e:
            self.logger.warning(f"Error executing SymPy code: {e}")
            return None


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
            # 解析AST
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
        """
        递归解析AST表达式节点

        Args:
            node: AST节点
            symbols: 符号变量列表

        Returns:
            SymPy表达式
        """
        if isinstance(node, ast.Name):
            # 变量名 -> 符号
            name = node.id
            # 尝试匹配符号变量（x0, x1, ...）
            match = re.match(r"x(\d+)", name)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(symbols):
                    return symbols[idx]
            # 否则创建新符号
            return sp.Symbol(name)

        elif isinstance(node, ast.Constant):
            # 常量
            return (
                sp.Integer(node.value)
                if isinstance(node.value, int)
                else sp.Float(node.value)
            )

        elif isinstance(node, ast.Num):  # Python < 3.8
            return sp.Integer(node.n) if isinstance(node.n, int) else sp.Float(node.n)

        elif isinstance(node, ast.BinOp):
            # 二元操作
            op_func = self.ast_to_sympy.get(type(node.op))
            if op_func:
                left = self._parse_expr(node.left, symbols)
                right = self._parse_expr(node.right, symbols)
                return op_func(left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        elif isinstance(node, ast.UnaryOp):
            # 一元操作
            op_func = self.ast_to_sympy.get(type(node.op))
            if op_func:
                operand = self._parse_expr(node.operand, symbols)
                return op_func(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.Call):
            # 函数调用
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                sympy_func = self.func_mapping.get(func_name)
                if sympy_func:
                    args = [self._parse_expr(arg, symbols) for arg in node.args]
                    return sympy_func(*args)
                else:
                    # 未知函数，尝试作为符号
                    args = [self._parse_expr(arg, symbols) for arg in node.args]
                    func_symbol = sp.Symbol(func_name)
                    return sp.Function(func_symbol)(*args)

        elif isinstance(node, ast.IfExp):
            # 条件表达式 -> Piecewise
            condition = self._parse_expr(node.test, symbols)
            true_expr = self._parse_expr(node.body, symbols)
            false_expr = self._parse_expr(node.orelse, symbols)
            return sp.Piecewise((true_expr, condition), (false_expr, True))

        elif isinstance(node, ast.Compare):
            # 比较表达式（用于条件）
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

        # 默认：返回符号
        return sp.Symbol("unknown")
