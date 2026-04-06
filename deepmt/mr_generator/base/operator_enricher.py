"""
算子条目自动丰富器（OperatorEnricher）

为算子目录条目自动填充 input_specs、signature 等字段。

三层策略（按优先级递增，后层补充前层未能覆盖的部分）：
1. inspect 模块（离线）：提取参数名、类型注解、默认值
2. HTML 结构化解析（需网络）：从文档页面解析参数类型描述，补充 dtype 信息
3. LLM 提取（需网络 + LLM）：从文档文本中提取约束条件（value_range、shape 等）

所有自动生成的 input_specs 均标记 input_specs_auto: true，
提示维护者进行人工核查和修正。
"""

import importlib
import inspect
import json
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from deepmt.core.logger import logger

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 默认浮点 dtype 列表（覆盖大多数激活函数 / 数学运算）
_FLOAT_DTYPES = ["float16", "bfloat16", "float32", "float64"]
_INT_DTYPES = ["int8", "int16", "int32", "int64", "uint8"]
_ALL_NUMERIC_DTYPES = _FLOAT_DTYPES + _INT_DTYPES

# 启发式：参数名属于 Tensor 输入的可能性较高
_TENSOR_PARAM_NAMES = {
    "input", "weight", "bias", "target", "other", "x", "y",
    "input1", "input2", "tensor", "data", "src", "indices",
    "values", "grad_output",
}


def _annotation_is_tensor(annotation) -> bool:
    """判断类型注解是否表示 Tensor 类型"""
    if annotation is inspect.Parameter.empty:
        return False
    ann_str = str(annotation)
    return "Tensor" in ann_str


_DTYPE_ANY = "any"  # sentinel: 无 dtype 限制（有别于 [] 的"未知"）


def _annotation_dtype(annotation):
    """从类型注解字符串推断支持的 dtype。

    返回值类型：
      []         — 无法推断（未知）
      "any"      — 注解表明可接受任意 dtype（如纯 Tensor 无约束）
      [str, ...] — 明确的 dtype 列表
    """
    if annotation is inspect.Parameter.empty:
        return []
    ann_str = str(annotation)
    if "Bool" in ann_str:
        return ["bool"]
    if "Int" in ann_str and "Float" not in ann_str:
        return _INT_DTYPES[:]
    if "Tensor" in ann_str or "Float" in ann_str:
        # 纯 Tensor 注解：标记为 any，由 HTML/LLM 进一步精化
        return _DTYPE_ANY
    return []


class OperatorEnricher:
    """
    为算子目录条目自动填充 input_specs 等字段。

    用法::

        enricher = OperatorEnricher()
        updates = enricher.enrich(
            name="torch.nn.functional.relu",
            api_type="function",
            doc_url="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html",
            use_llm=True,
            llm_client=llm_client,
        )
        # updates: {"input_specs": [...], "input_specs_auto": True, "signature": "..."}
    """

    def enrich(
        self,
        name: str,
        api_type: str = "",
        doc_url: str = "",
        framework: str = "pytorch",
        use_llm: bool = True,
        llm_client=None,
    ) -> Dict[str, Any]:
        """
        对一个算子条目执行三层丰富，返回需要更新的字段字典。

        Args:
            name:       算子名称（完整路径，如 torch.nn.functional.relu）
            api_type:   API 类型（class / function），用于 inspect 策略
            doc_url:    文档页面 URL，用于 HTML 解析和 LLM 提取
            framework:  目标框架（pytorch / tensorflow / paddlepaddle）
            use_llm:    是否使用 LLM（需提供 llm_client）
            llm_client: LLMClient 实例；为 None 时跳过 LLM 步骤

        Returns:
            需要更新的字段字典。若无法提取任何信息则返回 {}。
            包含 input_specs 时会自动附加 input_specs_auto: True。
        """
        updates: Dict[str, Any] = {}

        # ── Step 1: inspect（离线，最快）────────────────────────────────────────
        inspect_result = self._from_inspect(name, api_type)
        if inspect_result.get("input_specs"):
            updates["input_specs"] = inspect_result["input_specs"]
        if inspect_result.get("signature"):
            updates["signature"] = inspect_result["signature"]

        # ── Step 2: HTML 解析（网络，结构化）────────────────────────────────────
        # 仅当 inspect 存在 dtype 为空的 Tensor 参数时才发起网络请求
        if doc_url and self._has_empty_dtype(updates):
            html = self._fetch_html(doc_url)
            if html:
                self._enrich_from_html(html, updates, framework=framework)

        # ── Step 3: LLM 提取（网络 + LLM，最准确的约束信息）──────────────────
        if use_llm and llm_client and doc_url:
            doc_text = self._fetch_doc_text(doc_url, framework=framework)
            if doc_text:
                llm_specs = self._extract_with_llm(name, doc_text, llm_client, framework=framework)
                if llm_specs:
                    self._merge_llm_specs(updates, llm_specs)

        # 标记为自动生成
        if "input_specs" in updates:
            updates["input_specs_auto"] = True

        return updates

    # ── Step 1: inspect ────────────────────────────────────────────────────────

    def _from_inspect(self, name: str, api_type: str) -> Dict[str, Any]:
        """使用 inspect 模块提取参数签名和 input_specs。

        支持两种形式：
        - 模块级对象：torch.nn.functional.relu → import torch.nn.functional, get relu
        - 类方法：torch.Tensor.argmin → import torch, get Tensor.argmin（逐级属性访问）
        """
        try:
            obj = self._resolve_obj(name)
            if obj is None:
                return {}

            if api_type == "class" or inspect.isclass(obj):
                return self._inspect_class(obj)
            else:
                result = self._inspect_callable(obj)
                # 对于 builtin 方法（如 torch.Tensor.argmin），inspect.signature 失败；
                # 从 __doc__ 第一行解析签名，并为 self（输入张量）生成 dtype=any 的 spec。
                if not result:
                    result = self._inspect_from_doc(obj, name)
                return result
        except Exception as e:
            logger.debug(f"[Enricher] inspect failed for {name}: {e}")
            return {}

    def _inspect_from_doc(self, obj, name: str) -> Dict[str, Any]:
        """当 inspect.signature 失败（C builtin）时，从 __doc__ 解析签名。

        对 torch.Tensor.xxx 方法：
        - __doc__ 第一行格式为 "method(params) -> ReturnType"（不含 self）
        - self 对应输入张量，dtype 标记为 "any"（无文档约束）
        """
        doc = inspect.getdoc(obj) or ""
        if not doc:
            return {}

        # 解析第一行：method(params) -> ReturnType
        first_line = doc.splitlines()[0].strip()
        m = re.match(r"[\w.]+\(([^)]*)\)", first_line)
        if not m:
            return {}

        params_str = m.group(1).strip()
        # 重构为完整签名字符串（在 self/input 前加上）
        if params_str:
            signature = f"(input, {params_str})"
        else:
            signature = "(input)"

        # self 即输入张量；dtype 标记为 any（有实际张量，但无文档 dtype 限制）
        spec = _InputSpec(name="input", dtype=_DTYPE_ANY, shape="any", required=True)
        return {
            "signature": signature,
            "input_specs": [spec.to_dict()],
        }

    def _resolve_obj(self, name: str):
        """将完整名称解析为 Python 对象（支持模块 + 属性链）。"""
        parts = name.split(".")
        # 从最长模块路径开始，逐步缩短，剩余部分作为属性链
        for i in range(len(parts), 0, -1):
            mod_path = ".".join(parts[:i])
            attr_chain = parts[i:]
            try:
                obj = importlib.import_module(mod_path)
                for attr in attr_chain:
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None:
                    return obj
            except (ImportError, ModuleNotFoundError):
                continue
        return None

    def _inspect_callable(self, func) -> Dict[str, Any]:
        """提取可调用对象（函数/方法）的参数信息"""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return {}
        params = sig.parameters
        sig_str = self._build_signature_str(params)
        specs = self._params_to_specs(params)
        result: Dict[str, Any] = {}
        if sig_str:
            result["signature"] = sig_str
        if specs:
            result["input_specs"] = [s.to_dict() for s in specs]
        return result

    def _inspect_class(self, cls) -> Dict[str, Any]:
        """提取类的 forward 方法参数（跳过 __init__ 的 weight/bias 等构造参数）"""
        # 优先取 forward（nn.Module 的实际计算接口）
        forward = getattr(cls, "forward", None)
        if forward is None:
            return {}
        try:
            sig = inspect.signature(forward)
        except (ValueError, TypeError):
            return {}
        # 过滤掉 self
        params = {k: v for k, v in sig.parameters.items() if k != "self"}
        sig_str = self._build_signature_str(params)
        specs = self._params_to_specs(params)
        result: Dict[str, Any] = {}
        if sig_str:
            result["signature"] = f"({sig_str})"
        if specs:
            result["input_specs"] = [s.to_dict() for s in specs]
        return result

    def _build_signature_str(self, params: dict) -> str:
        """将参数字典转换为签名字符串"""
        parts = []
        for name, param in params.items():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                parts.append(f"*{name}")
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                parts.append(f"**{name}")
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if param.default is inspect.Parameter.empty:
                    parts.append(name)
                else:
                    parts.append(f"{name}={param.default!r}")
            else:
                if param.default is inspect.Parameter.empty:
                    parts.append(name)
                else:
                    parts.append(f"{name}={param.default!r}")
        return "(" + ", ".join(parts) + ")" if parts else ""

    def _params_to_specs(self, params: dict) -> List["_InputSpec"]:
        """将参数字典转换为 InputSpec 列表，仅保留 Tensor 参数"""
        specs = []
        for pname, param in params.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if not self._is_tensor_param(pname, param):
                continue
            required = (
                param.default is inspect.Parameter.empty
                and param.kind != inspect.Parameter.VAR_POSITIONAL
            )
            # Optional[Tensor] → not required
            ann_str = str(param.annotation) if param.annotation is not inspect.Parameter.empty else ""
            if "Optional" in ann_str:
                required = False
            dtypes = _annotation_dtype(param.annotation)
            specs.append(_InputSpec(
                name=pname,
                dtype=dtypes,
                shape="any",
                value_range=None,
                required=required,
            ))
        return specs

    def _is_tensor_param(self, name: str, param: inspect.Parameter) -> bool:
        """判断参数是否为 Tensor 类型（注解优先，名称启发式兜底）"""
        if _annotation_is_tensor(param.annotation):
            return True
        if name.lower() in _TENSOR_PARAM_NAMES:
            return True
        return False

    def _has_empty_dtype(self, updates: Dict[str, Any]) -> bool:
        """检查 updates 中是否有 dtype 为空列表（未知）的 input_specs 条目。
        dtype == "any" 视为已填充，不触发 HTML 请求。"""
        for spec in updates.get("input_specs", []):
            dtype = spec.get("dtype")
            if dtype == [] or dtype is None:
                return True
        return False

    # ── Step 2: HTML 解析 ──────────────────────────────────────────────────────

    def _fetch_html(self, url: str) -> Optional[str]:
        """获取 HTML 内容"""
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.debug(f"[Enricher] HTML fetch failed for {url}: {e}")
            return None

    def _enrich_from_html(self, html: str, updates: Dict[str, Any], framework: str = "pytorch") -> None:
        """
        从 HTML 文档补充 dtype 信息。

        PyTorch 文档（Sphinx 生成）参数描述的典型结构：
          <dl class="field-list simple">
            <dd>
              <ul>
                <li><p><strong>input</strong> (<em>Tensor</em>) – ...</p></li>
              </ul>
            </dd>
          </dl>

        其他框架的文档结构不同，暂未实现（跳过而非报错）。
        """
        if framework != "pytorch":
            logger.debug(f"[Enricher] HTML enrichment not implemented for '{framework}', skipping")
            return
        existing_specs: List[dict] = updates.get("input_specs", [])
        if not existing_specs:
            return

        soup = BeautifulSoup(html, "html.parser")

        # 建立 param_name → spec 的映射，加速查找
        spec_by_name = {s["name"]: s for s in existing_specs}

        for li in soup.find_all("li"):
            strong = li.find("strong")
            if not strong:
                continue
            param_name = strong.get_text(strip=True)
            spec = spec_by_name.get(param_name)
            if spec is None:
                continue

            # 提取类型标注（<em> 标签）
            em = li.find("em")
            if em:
                type_text = em.get_text(strip=True)
                inferred = self._dtype_from_html_type(type_text)
                if inferred and not spec.get("dtype"):
                    spec["dtype"] = inferred

    def _dtype_from_html_type(self, type_text: str):
        """从 HTML 中的类型描述文本推断 dtype。

        返回值同 _annotation_dtype：[] / "any" / [str, ...]
        """
        t = type_text.lower()
        if "tensor" not in t:
            return []
        if "bool" in t:
            return ["bool"]
        if "int" in t and "float" not in t:
            return _INT_DTYPES[:]
        if "float" in t:
            return _FLOAT_DTYPES[:]
        # 纯 "Tensor"：无更多 dtype 信息，标记为 any
        return _DTYPE_ANY

    # ── Step 3: LLM 提取 ──────────────────────────────────────────────────────

    def _fetch_doc_text(self, url: str, framework: str = "pytorch") -> Optional[str]:
        """获取文档文本（使用 SearchAgent 的解析逻辑）"""
        try:
            from deepmt.tools.web_search.search_agent import SearchAgent
            agent = SearchAgent()
            return agent.fetch_operator_doc_by_url(url, framework=framework)
        except Exception as e:
            logger.debug(f"[Enricher] doc text fetch failed: {e}")
            return None

    def _extract_with_llm(
        self,
        name: str,
        doc_text: str,
        llm_client,
        framework: str = "pytorch",
    ) -> Optional[List[dict]]:
        """使用 LLM 从文档文本中提取结构化 input_specs"""
        prompt = f"""给定 {framework} 算子 `{name}` 的官方文档：

{doc_text[:3000]}

请提取该算子的输入张量（Tensor）参数规范，以 JSON 数组格式输出：
[
  {{
    "name": "input",
    "dtype": ["float32", "float64"],
    "shape": "any",
    "value_range": null,
    "required": true
  }}
]

规则：
- 只包含 Tensor 类型参数，跳过标量（int/float/bool/str）参数
- dtype 只能从以下选择：float16, bfloat16, float32, float64, int8, int16, int32, int64, uint8, bool, complex64, complex128
- shape 可选值："any" | "nd>=N"（如 "nd>=2"）| "(n,)" | "(n,m)" | "(n,c,h,w)" 等
- value_range 格式：null（无约束）| [最小值或null, 最大值或null]
- 不确定的值填 [] 或 null，不要猜测
- 只输出 JSON 数组，不要解释"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.chat_completion(messages)
            text = response.strip()
            # 提取 JSON 数组部分
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.debug(f"[Enricher] LLM extraction failed: {e}")
        return None

    def _merge_llm_specs(
        self, updates: Dict[str, Any], llm_specs: List[dict]
    ) -> None:
        """将 LLM 提取的 specs 合并到 updates 中（补充空字段，不覆盖已有值）"""
        if not llm_specs:
            return
        existing = updates.get("input_specs", [])
        existing_by_name = {s["name"]: s for s in existing}

        for llm_spec in llm_specs:
            pname = llm_spec.get("name", "")
            if not pname:
                continue
            if pname in existing_by_name:
                spec = existing_by_name[pname]
                # 只补充空字段
                if not spec.get("dtype") and llm_spec.get("dtype"):
                    spec["dtype"] = llm_spec["dtype"]
                if spec.get("value_range") is None and llm_spec.get("value_range") is not None:
                    spec["value_range"] = llm_spec["value_range"]
                if spec.get("shape", "any") == "any" and llm_spec.get("shape", "any") != "any":
                    spec["shape"] = llm_spec["shape"]
            else:
                # LLM 发现了 inspect 未检测到的 Tensor 参数
                existing.append({
                    "name": pname,
                    "dtype": llm_spec.get("dtype", []),
                    "shape": llm_spec.get("shape", "any"),
                    "value_range": llm_spec.get("value_range"),
                    "required": llm_spec.get("required", True),
                })
        if existing:
            updates["input_specs"] = existing


class _InputSpec:
    """内部用 InputSpec 数据容器"""

    def __init__(
        self,
        name: str,
        dtype: List[str],
        shape: str = "any",
        value_range: Optional[List] = None,
        required: bool = True,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.value_range = value_range
        self.required = required

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "value_range": self.value_range,
            "required": self.required,
        }
