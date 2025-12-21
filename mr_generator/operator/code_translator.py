"""
代码翻译器：使用LLM将任意代码转换为SymPy表达式
算子相关功能，用于将Python代码转换为SymPy表达式
"""

import ast
import inspect
import sympy as sp
from typing import Callable, Any, Optional

from tools.llm.client import LLMClient
from core.logger import get_logger


class CodeToSymPyTranslator:
    """
    代码到SymPy转换器（算子相关功能）

    使用LLM + AST解析将任意Python代码转换为SymPy表达式
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化转换器

        Args:
            llm_client: LLM客户端（如果为None则创建默认客户端）
        """
        self.logger = get_logger()

        # 导入AST解析器（延迟导入避免循环依赖）
        from mr_generator.operator.ast_parser import ASTToSymPyParser

        self.ast_parser = ASTToSymPyParser()

        if llm_client:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = LLMClient()
            except Exception as e:
                self.logger.warning(f"Failed to create LLM client: {e}")
                self.llm_client = None

        if self.llm_client is None:
            try:
                self.llm_client = LLMClient()
            except Exception as e:
                self.logger.warning(f"Failed to create LLM client: {e}")

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
            use_proxy_path: 是否使用代理路径（LLM→Python参考实现→AST→SymPy），
                           如果为False则使用直接路径（LLM→SymPy）

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
            if use_proxy_path:
                # 标准路径：LLM → Python参考实现（代理）→ AST → SymPy
                self.logger.debug("Using proxy path: LLM → Python reference → AST → SymPy")
                python_ref = self._llm_translate_to_python_ref(code, doc, sig)
                if python_ref:
                    try:
                        # 使用AST解析Python参考实现
                        sympy_expr = self.ast_parser.parse_to_sympy(python_ref)
                        if sympy_expr is not None:
                            self.logger.debug("Successfully converted via proxy path")
                            return sympy_expr
                        else:
                            self.logger.warning(
                                "AST parsing failed for Python reference, trying direct path"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Proxy path failed: {e}, trying direct path"
                        )
                
                # 如果代理路径失败，回退到直接路径
                self.logger.debug("Falling back to direct path: LLM → SymPy")
            
            # 直接路径：LLM → SymPy表达式（保留原有路径）
            sympy_code = self._llm_translate_direct(code, doc, sig)
            if sympy_code:
                try:
                    # 执行LLM生成的SymPy代码
                    result = self._execute_sympy_code(sympy_code)
                    if result is not None:
                        self.logger.debug("Successfully converted via direct path")
                        return result
                except Exception as e:
                    self.logger.warning(
                        f"Direct LLM translation failed: {e}, falling back to AST"
                    )

        # 3. 回退到AST解析原始代码
        self.logger.debug("Falling back to AST parsing of original code")
        return self.ast_parser.parse_to_sympy(code)

    def _llm_translate_to_python_ref(
        self, code: str, doc: str, signature: str
    ) -> Optional[str]:
        """
        使用LLM将多源数据翻译为Python参考实现（代理路径的第一步）
        
        这是标准路径的第一步：将代码、文档、签名等转换为清晰的Python参考实现，
        然后由AST解析器解析为SymPy表达式。这样可以提高转换的准确性和可解释性。
        """
        prompt = f"""你是一个代码分析专家。请将以下Python代码转换为一个清晰的Python参考实现。

函数签名：{signature}

原始代码：
```python
{code}
```

文档：
{doc if doc else "无"}

要求：
1. 将代码转换为一个清晰的、数学化的Python函数实现
2. 使用标准的Python数学操作（+、-、*、/、**、max、min等）
3. 处理条件表达式使用 if-else 或三元表达式
4. 确保代码逻辑清晰，便于后续符号化处理
5. 只返回Python函数代码，不要包含其他说明

输出格式（Python代码）：
```python
def reference_implementation(x0, x1, ...):
    # 清晰的数学实现
    return <表达式>
```

只返回代码块，不要包含markdown标记外的其他文字。
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a code analysis expert. Convert code to a clear Python reference implementation.",
                },
                {"role": "user", "content": prompt},
            ]

            content = self.llm_client.chat_completion(messages, temperature=0.3)

            # 提取代码块
            if "```python" in content:
                code_block = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                code_block = content.split("```")[1].split("```")[0].strip()
            else:
                code_block = content

            return code_block

        except Exception as e:
            self.logger.warning(f"LLM translation to Python reference error: {e}")
            return None

    def _llm_translate_direct(self, code: str, doc: str, signature: str) -> Optional[str]:
        """
        使用LLM直接将代码翻译为SymPy表达式（直接路径）
        
        这是原有的直接路径：直接从代码、文档、签名生成SymPy表达式代码。
        保留此路径作为备选方案。
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
            messages = [
                {"role": "system", "content": "You are a code analysis expert."},
                {"role": "user", "content": prompt},
            ]

            content = self.llm_client.chat_completion(messages, temperature=0.3)

            # 提取代码块
            if "```python" in content:
                code_block = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                code_block = content.split("```")[1].split("```")[0].strip()
            else:
                code_block = content

            return code_block

        except Exception as e:
            self.logger.warning(f"LLM direct translation error: {e}")
            return None

    def _execute_sympy_code(self, code: str) -> Optional[sp.Expr]:
        """执行SymPy代码并返回表达式"""
        try:
            exec_globals = {"sp": sp, "sympy": sp}
            exec(code, exec_globals)

            if "result" in exec_globals:
                return exec_globals["result"]
            else:
                return None

        except Exception as e:
            self.logger.warning(f"Error executing SymPy code: {e}")
            return None


