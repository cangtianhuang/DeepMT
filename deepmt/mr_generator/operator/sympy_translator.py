"""
代码翻译器：将算子代码或名称转换为 SymPy 表达式
翻译优先级：已知映射表 → 持久化缓存 → LLM 翻译 → AST 回退
"""

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import sympy as sp

from deepmt.core.logger import logger
from deepmt.tools.llm.client import LLMClient

# ── 已知算子的直接 SymPy 映射（确定性最高，无需 LLM）─────────────────────────────
# key: 算子全限定名（module.name），value: 以 x0 为输入变量的 SymPy 表达式
# 覆盖常见 C 扩展算子，省去 inspect.getsource 失败后的 LLM 调用

_x0 = sp.Symbol("x0")

_KNOWN_SYMPY_EXPRS: Dict[str, Any] = {
    "torch.exp":                     sp.exp(_x0),
    "torch.abs":                     sp.Abs(_x0),
    "torch.sin":                     sp.sin(_x0),
    "torch.cos":                     sp.cos(_x0),
    "torch.log":                     sp.log(_x0),
    "torch.sqrt":                    sp.sqrt(_x0),
    "torch.tanh":                    sp.tanh(_x0),
    "torch.nn.functional.relu":      sp.Max(sp.Integer(0), _x0),
    "torch.nn.functional.sigmoid":   sp.Integer(1) / (1 + sp.exp(-_x0)),
}


class SympyTranslator:
    """代码到 SymPy 转换器"""

    _CACHE_DIR = Path(__file__).parents[3] / "data" / "sympy_cache"

    def __init__(self, use_llm: bool = True):
        self._use_llm = use_llm

        from deepmt.mr_generator.operator.ast_parser import ASTParser
        self.ast_parser = ASTParser()
        self.llm_client = LLMClient()
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def translate(
        self,
        code: Optional[str] = None,
        func: Optional[Callable] = None,
        doc: Optional[str] = None,
        signature: Optional[str] = None,
        use_proxy_path: bool = True,
        use_llm: Optional[bool] = None,
    ) -> Optional[sp.Expr]:
        """
        将算子转换为 SymPy 表达式。

        优先级：
          1. 已知映射表（deterministic，无 LLM 成本）
          2. 持久化缓存（避免重复调用 LLM）
          3. LLM 翻译（代理路径 → 直接路径；无源码时按算子名推断）
          4. AST 回退（仅当有真实源码时）
        """
        # 1. 提取函数元数据（算子名、签名、文档、源码）
        op_name = ""
        if func is not None:
            _mod = getattr(func, "__module__", "") or ""
            _nm  = getattr(func, "__name__",   "") or ""
            op_name = f"{_mod}.{_nm}".strip(".")

            if code is None:
                try:
                    code = inspect.getsource(func)
                except Exception:
                    pass  # C 扩展等无源码，后续步骤处理
            if doc is None:
                doc = inspect.getdoc(func)
            if signature is None:
                try:
                    signature = str(inspect.signature(func))
                except Exception:
                    signature = ""
        else:
            signature = signature or ""

        # 2. 已知映射表（优先级最高）
        if op_name in _KNOWN_SYMPY_EXPRS:
            logger.info(f"⚡ [GEN] Using built-in SymPy mapping for '{op_name}'")
            return _KNOWN_SYMPY_EXPRS[op_name]

        # 3. 持久化缓存
        cache_key = self._cache_key(op_name, code)
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug(f"SymPy translation cache hit: {cache_key}")
                return cached

        # 4. 无源码处理：构造哨兵，允许 LLM 按算子名推断
        _use_llm = self._use_llm if use_llm is None else use_llm
        no_source = False
        if code is None:
            if not _use_llm:
                logger.warning(
                    f"No source code for '{op_name}' and LLM disabled — translation skipped"
                )
                return None
            no_source = True
            code = f"# NO_SOURCE: {op_name}"
            logger.info(
                f"⚡ [GEN] No Python source for '{op_name}', asking LLM to infer SymPy expression"
            )

        # 5. LLM 翻译
        result = None
        if _use_llm and use_proxy_path:
            logger.debug("Trying proxy path: LLM → Python reference → AST → SymPy")
            result = self._try_proxy_path(code, doc or "", signature or "", no_source=no_source)
            if result is None:
                logger.debug("Proxy path failed, trying direct path")

        if result is None and _use_llm:
            logger.debug("Trying direct path: LLM → SymPy code")
            result = self._try_direct_path(code, doc or "", signature or "", no_source=no_source)
            if result is None:
                logger.debug("Direct path failed")

        # 6. AST 回退（仅当有真实源码时）
        if result is None and not no_source:
            logger.debug("Falling back to AST parsing of original code")
            result = self.ast_parser.parse_to_sympy(code)

        # 7. 写入缓存
        if result is not None and cache_key:
            self._save_cache(cache_key, result)

        return result

    # ── 持久化缓存 ────────────────────────────────────────────────────────────

    def _cache_key(self, op_name: str, code: Optional[str]) -> Optional[str]:
        """生成缓存 key：优先用算子全名，fallback 用代码 hash。"""
        if op_name:
            return op_name.replace(".", "_")
        if code and not code.startswith("# NO_SOURCE:"):
            import hashlib
            return "code_" + hashlib.md5(code.encode()).hexdigest()[:12]
        return None

    def _load_cache(self, key: str) -> Optional[sp.Expr]:
        path = self._CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return sp.sympify(data["expr_repr"])
        except Exception as e:
            logger.debug(f"Failed to load SymPy cache '{key}': {e}")
            return None

    def _save_cache(self, key: str, expr: sp.Expr) -> None:
        try:
            path = self._CACHE_DIR / f"{key}.json"
            path.write_text(json.dumps({"expr_repr": str(expr)}), encoding="utf-8")
            logger.debug(f"Saved SymPy cache: {key} → {expr}")
        except Exception as e:
            logger.debug(f"Failed to save SymPy cache '{key}': {e}")

    # ── LLM 翻译路径 ──────────────────────────────────────────────────────────

    def _try_proxy_path(
        self, code: str, doc: str, signature: str, no_source: bool = False
    ) -> Optional[sp.Expr]:
        """代理路径：LLM → Python 参考实现 → AST → SymPy"""
        try:
            python_ref = self._llm_to_python_reference(code, doc, signature, no_source=no_source)
            if python_ref:
                result = self.ast_parser.parse_to_sympy(python_ref)
                if result is not None:
                    logger.debug("Successfully converted via proxy path")
                    return result
        except Exception as e:
            logger.debug(f"Proxy path error: {e}")
        return None

    def _try_direct_path(
        self, code: str, doc: str, signature: str, no_source: bool = False
    ) -> Optional[sp.Expr]:
        """直接路径：LLM → SymPy 表达式代码"""
        try:
            sympy_code = self._llm_to_sympy_code(code, doc, signature, no_source=no_source)
            if sympy_code:
                result = self._execute_sympy_code(sympy_code)
                if result is not None:
                    logger.debug("Successfully converted via direct path")
                    return result
        except Exception as e:
            logger.debug(f"Direct path error: {e}")
        return None

    def _llm_to_python_reference(
        self, code: str, doc: str, signature: str, no_source: bool = False
    ) -> Optional[str]:
        """使用 LLM 将算子转换为纯 Python 参考实现。"""
        if no_source:
            op_name = code[len("# NO_SOURCE:"):].strip()
            prompt = f"""算子 `{op_name}` 的 Python 源码不可用（C 扩展或内置函数）。
请根据算子名称直接给出其数学定义的纯 Python 参考实现。

函数签名：{signature or '未知'}
文档：{doc or '无'}

要求：
1. 输出一个清晰的 Python 函数，参数命名为 x0, x1, x2, ...
2. 使用标准 Python 数学操作（+, -, *, /, **, max, min, abs, math.exp 等）
3. 条件表达式使用三元表达式 (a if condition else b)
4. 只返回代码，不要包含说明文字

输出格式：
```python
def reference_impl(x0):
    return <表达式>
```"""
        else:
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
            logger.debug(f"LLM to Python reference error: {e}")
            return None

    def _llm_to_sympy_code(
        self, code: str, doc: str, signature: str, no_source: bool = False
    ) -> Optional[str]:
        """使用 LLM 将算子直接转换为 SymPy 表达式代码。"""
        if no_source:
            op_name = code[len("# NO_SOURCE:"):].strip()
            prompt = f"""算子 `{op_name}` 的 Python 源码不可用（C 扩展或内置函数）。
请根据算子名称直接给出其数学定义对应的 SymPy 表达式。

函数签名：{signature or '未知'}
文档：{doc or '无'}

要求：
1. 使用符号变量 x0, x1, x2, ... (用 sp.Symbol('x0') 等)
2. 将数学操作转换为 SymPy 操作（如 sp.exp, sp.Abs, sp.sin 等）
3. 最终表达式赋值给变量 result
4. 只返回代码，不要包含说明文字

输出格式：
```python
import sympy as sp
x0 = sp.Symbol('x0')
result = <SymPy表达式>
```"""
        else:
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
            logger.debug(f"LLM to SymPy code error: {e}")
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
            logger.debug(f"Error executing SymPy code: {e}")
            return None
