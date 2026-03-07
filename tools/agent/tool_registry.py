"""工具注册表：管理智能体可用的原子工具，支持 OpenAI Function Calling 格式"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional


class ToolDef:
    """工具定义：包含函数本身和 OpenAI 格式的 schema"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters  # JSON Schema object

    def to_openai_schema(self) -> Dict[str, Any]:
        """返回 OpenAI Function Calling 格式的工具描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：集中管理所有可用工具"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool_def: ToolDef) -> None:
        """注册一个工具"""
        self._tools[tool_def.name] = tool_def

    def register_func(self, func: Callable) -> None:
        """从带有 @tool 装饰器的函数中注册工具"""
        if not hasattr(func, "_tool_name"):
            raise ValueError(f"函数 {func.__name__} 没有 @tool 装饰器")
        self.register(
            ToolDef(
                name=func._tool_name,
                description=func._tool_description,
                func=func,
                parameters=func._tool_parameters,
            )
        )

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """返回所有工具的 OpenAI Function Calling 格式描述列表"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        执行指定工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数（与 JSON Schema 对应的 kwargs）

        Returns:
            工具执行结果字符串
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"工具 '{tool_name}' 不存在"}, ensure_ascii=False)
        try:
            result = self._tools[tool_name].func(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps(
                {"error": f"执行工具 '{tool_name}' 时出错: {e}"},
                ensure_ascii=False,
            )

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    将函数注册为智能体工具的装饰器。

    Args:
        name: 工具名称（LLM 调用时使用）
        description: 工具描述（告知 LLM 何时使用此工具）
        parameters: JSON Schema object，描述参数。若为 None，则自动从函数签名推断。

    Usage::

        @tool(
            name="fetch_page",
            description="获取指定 URL 的网页文本内容",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "目标 URL"},
                },
                "required": ["url"],
            },
        )
        def fetch_page(url: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        func._tool_name = name
        func._tool_description = description
        func._tool_parameters = parameters if parameters is not None else _infer_parameters(func)
        return func

    return decorator


def _infer_parameters(func: Callable) -> Dict[str, Any]:
    """从函数签名自动推断 JSON Schema parameters（简单版）"""
    sig = inspect.signature(func)
    hints = func.__annotations__

    _type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)
        # 处理 Optional[X] -> X
        origin = getattr(py_type, "__origin__", None)
        if origin is type(None):
            py_type = str
        elif origin is not None:
            args = getattr(py_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            py_type = non_none[0] if non_none else str

        json_type = _type_map.get(py_type, "string")
        prop: Dict[str, Any] = {"type": json_type}

        # 从 docstring 中提取参数说明（简单约定：参数名: 说明）
        doc = func.__doc__ or ""
        for line in doc.splitlines():
            line = line.strip()
            if line.startswith(f"{param_name}:") or line.startswith(f"{param_name} :"):
                prop["description"] = line.split(":", 1)[1].strip()
                break

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def build_default_registry() -> ToolRegistry:
    """构建包含所有默认工具的注册表"""
    from tools.agent.tools.browse import fetch_page, fetch_page_html
    from tools.agent.tools.extract import (
        extract_links,
        extract_text,
        find_version_tags,
        search_in_text,
    )

    registry = ToolRegistry()
    for func in [fetch_page, fetch_page_html, extract_links, extract_text, find_version_tags, search_in_text]:
        registry.register_func(func)
    return registry
