"""
代码翻译器：使用LLM将任意代码转换为SymPy表达式
算子相关功能，用于将Python代码转换为SymPy表达式
"""

import inspect
from typing import Callable, Optional

import sympy as sp

from core.logger import get_logger
from tools.llm.client import LLMClient


class SympyTranslator:
    """代码到SymPy转换器"""

    def __init__(self):

        self.logger = get_logger(self.__class__.__name__)

        # 导入AST解析器
        from mr_generator.operator.ast_parser import ASTParser

        self.ast_parser = ASTParser()
        self.llm_client = LLMClient()

    def translate(
        self,
        code: Optional[str] = None,
        func: Optional[Callable] = None,
        doc: Optional[str] = None,
        signature: Optional[str] = None,
        use_proxy_path: bool = True,
    ) -> Optional[sp.Expr]:
        """
        将代码转换为SymPy表达式

        Args:
            code: Python代码字符串
            func: Python函数对象
            doc: 函数文档字符串
            signature: 函数签名字符串
            use_proxy_path: 是否使用代理路径（推荐），False则使用直接路径

        Returns:
            SymPy表达式，如果转换失败则返回None
        """
        # 1. 获取代码和文档
        if func is not None:
            if code is None:
                try:
                    code = inspect.getsource(func)
                except Exception:
                    pass
            if doc is None:
                doc = inspect.getdoc(func)
            if signature is None:
                try:
                    signature = str(inspect.signature(func))
                except Exception:
                    signature = ""
        else:
            signature = signature or ""

        if code is None:
            self.logger.warning("No code provided for translation")
            return None

        # 2. 尝试代理路径：LLM → Python参考实现 → AST → SymPy
        if use_proxy_path:
            self.logger.debug("Trying proxy path: LLM → Python reference → AST → SymPy")
            result = self._try_proxy_path(code, doc or "", signature)
            if result is not None:
                return result
            self.logger.debug("Proxy path failed, trying direct path")

        # 3. 尝试直接路径：LLM → SymPy表达式代码
        self.logger.debug("Trying direct path: LLM → SymPy code")
        result = self._try_direct_path(code, doc or "", signature)
        if result is not None:
            return result
        self.logger.debug("Direct path failed, trying AST fallback")

        # 4. 回退到纯AST解析
        self.logger.debug("Falling back to AST parsing of original code")
        return self.ast_parser.parse_to_sympy(code)

    def _try_proxy_path(self, code: str, doc: str, signature: str) -> Optional[sp.Expr]:
        """
        尝试代理路径：LLM → Python参考实现 → AST → SymPy
        """
        try:
            python_ref = self._llm_to_python_reference(code, doc, signature)
            if python_ref:
                result = self.ast_parser.parse_to_sympy(python_ref)
                if result is not None:
                    self.logger.debug("Successfully converted via proxy path")
                    return result
        except Exception as e:
            self.logger.debug(f"Proxy path error: {e}")
        return None

    def _try_direct_path(
        self, code: str, doc: str, signature: str
    ) -> Optional[sp.Expr]:
        """
        尝试直接路径：LLM → SymPy表达式代码
        """
        try:
            sympy_code = self._llm_to_sympy_code(code, doc, signature)
            if sympy_code:
                result = self._execute_sympy_code(sympy_code)
                if result is not None:
                    self.logger.debug("Successfully converted via direct path")
                    return result
        except Exception as e:
            self.logger.debug(f"Direct path error: {e}")
        return None

    def _llm_to_python_reference(
        self, code: str, doc: str, signature: str
    ) -> Optional[str]:
        """
        使用LLM将代码转换为清晰的Python参考实现

        这是代理路径的第一步：将复杂代码简化为清晰的数学表达，
        便于后续AST解析。
        """
        prompt = f"""请将以下Python代码转换为一个清晰的Python参考实现。

函数签名：{signature}

原始代码：
```python
{code}
```

文档：{doc if doc else "无"}

要求：
1. 输出一个清晰的Python函数，参数命名为 x0, x1, x2, ... 
2. 使用标准Python数学操作（+, -, *, /, **, max, min, abs等）
3. 条件表达式使用三元表达式 (a if condition else b)
4. 只返回代码，不要包含说明文字

输出格式：
```python
def reference_impl(x0, x1, ...):
    return <表达式>
```"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a code expert. Output clean Python code only.",
                },
                {"role": "user", "content": prompt},
            ]
            content = self.llm_client.chat_completion(messages)
            return self._extract_code_block(content)
        except Exception as e:
            self.logger.debug(f"LLM to Python reference error: {e}")
            return None

    def _llm_to_sympy_code(self, code: str, doc: str, signature: str) -> Optional[str]:
        """
        使用LLM将代码直接转换为SymPy表达式代码
        """
        prompt = f"""请将以下Python代码转换为SymPy表达式。

函数签名：{signature}

代码：
```python
{code}
```

文档：{doc if doc else "无"}

要求：
1. 使用符号变量 x0, x1, x2, ... (用 sp.Symbol('x0') 等)
2. 将操作转换为SymPy操作
3. 条件表达式使用 sp.Piecewise
4. 最终表达式赋值给变量 result
5. 只返回代码，不要包含说明文字

输出格式：
```python
import sympy as sp
x0 = sp.Symbol('x0')
x1 = sp.Symbol('x1')
result = <SymPy表达式>
```"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a code expert. Output SymPy code only.",
                },
                {"role": "user", "content": prompt},
            ]
            content = self.llm_client.chat_completion(messages)
            return self._extract_code_block(content)
        except Exception as e:
            self.logger.debug(f"LLM to SymPy code error: {e}")
            return None

    def _extract_code_block(self, content: str) -> Optional[str]:
        """从LLM响应中提取代码块"""
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        else:
            return content.strip()

    def _execute_sympy_code(self, code: str) -> Optional[sp.Expr]:
        """执行SymPy代码并返回表达式"""
        try:
            exec_globals = {"sp": sp, "sympy": sp}
            exec(code, exec_globals)

            if "result" in exec_globals:
                return exec_globals["result"]
            return None
        except Exception as e:
            self.logger.debug(f"Error executing SymPy code: {e}")
            return None


