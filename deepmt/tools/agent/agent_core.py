"""
CrawlAgent：基于 ReAct（Reason + Act）模式的网页爬取智能体。

工作流程：
  LLM 决策（Thought + Action）→ 执行工具 → 观察结果（Observation）→ 循环直到完成
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deepmt.core.config_loader import get_config_value
from deepmt.core.logger import get_logger, log_structured
from deepmt.tools.llm.client import LLMClient

# 任务规格 YAML 默认搜索目录
_TASKS_DIR = Path(__file__).parent / "tasks"

# 特殊工具名：智能体完成任务时调用
_FINISH_TOOL = "finish"


@dataclass
class TaskSpec:
    """任务规格：描述智能体需要完成的目标与约束"""

    task_id: str
    description: str
    inputs: List[Dict[str, Any]]           # 输入参数定义列表
    output_schema: Dict[str, Any]          # 期望的输出结构描述
    entry_points: Dict[str, str]           # framework/key -> URL 模板
    hints: List[str] = field(default_factory=list)
    max_steps: int = 15
    cache_ttl_days: int = 7

    @classmethod
    def from_yaml(cls, path: str) -> "TaskSpec":
        """从 YAML 文件加载任务规格"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            inputs=data.get("inputs", []),
            output_schema=data.get("output_schema", {}),
            entry_points=data.get("entry_points", {}),
            hints=data.get("hints", []),
            max_steps=data.get("max_steps", 15),
            cache_ttl_days=data.get("cache_ttl_days", 7),
        )

    @classmethod
    def from_task_id(cls, task_id: str) -> "TaskSpec":
        """
        按 task_id 从默认 tasks/ 目录加载规格。

        Args:
            task_id: 任务 ID，对应 tasks/<task_id>.yaml 文件
        """
        yaml_path = _TASKS_DIR / f"{task_id}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"任务规格文件不存在: {yaml_path}\n"
                f"可用任务: {[p.stem for p in _TASKS_DIR.glob('*.yaml')]}"
            )
        return cls.from_yaml(str(yaml_path))


class CrawlAgent:
    """
    基于 ReAct 模式的爬取智能体。

    使用方式::

        from deepmt.tools.agent import CrawlAgent, TaskSpec
        from deepmt.tools.agent.tool_registry import build_default_registry

        registry = build_default_registry()
        agent = CrawlAgent(registry=registry, verbose=True)

        spec = TaskSpec.from_task_id("get_framework_version")
        result = agent.run(spec, inputs={"framework": "pytorch"})
        print(result)  # {"version": "2.5.1", "release_date": "...", ...}
    """

    def __init__(
        self,
        registry=None,
        llm_client: Optional[LLMClient] = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            registry: ToolRegistry 实例；若为 None，自动构建包含默认工具的注册表
            llm_client: LLMClient 实例；若为 None，使用默认配置创建
            verbose: 是否在控制台打印每一步的执行过程
        """
        from deepmt.tools.agent.tool_registry import ToolRegistry, build_default_registry

        self.logger = get_logger(self.__class__.__name__)
        self.registry = registry if registry is not None else build_default_registry()
        self.llm = llm_client if llm_client is not None else LLMClient()
        self.verbose = verbose

    def run(
        self,
        task_spec: TaskSpec,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行任务，返回结构化输出。

        Args:
            task_spec: 任务规格
            inputs: 任务输入参数，如 {"framework": "pytorch"}

        Returns:
            与 task_spec.output_schema 匹配的字典

        Raises:
            RuntimeError: 超过 max_steps 仍未完成，或 LLM 无法生成有效结果
        """
        # agent.max_steps 全局配置可覆盖 TaskSpec 的 max_steps
        global_max_steps = get_config_value("agent.max_steps")
        effective_max_steps = (
            int(global_max_steps) if global_max_steps is not None else task_spec.max_steps
        )

        log_structured(
            self.logger,
            "AGENT",
            f"Starting task '{task_spec.task_id}'",
            inputs=inputs,
            max_steps=effective_max_steps,
        )
        if self.verbose:
            print(f"\n[Agent] 任务: {task_spec.task_id}  输入: {inputs}  最大步数: {effective_max_steps}")

        # 渲染 entry_points 中的模板变量（如 {version}、{framework}）
        resolved_entry_points = _resolve_entry_points(task_spec.entry_points, inputs)

        messages = self._build_initial_messages(task_spec, inputs, resolved_entry_points, effective_max_steps)
        tools = self._build_tools_schema(task_spec)

        result = self._react_loop(messages, tools, task_spec, effective_max_steps)

        log_structured(
            self.logger,
            "AGENT",
            f"Task '{task_spec.task_id}' completed",
            result_keys=list(result.keys()),
        )
        if self.verbose:
            print(f"[Agent] 任务完成，结果字段: {list(result.keys())}")
        return result

    # ---------------------------------------------------------------------- #
    #  内部实现                                                                #
    # ---------------------------------------------------------------------- #

    def _build_initial_messages(
        self,
        task_spec: TaskSpec,
        inputs: Dict[str, Any],
        resolved_entry_points: Dict[str, str],
        max_steps: int = 0,
    ) -> List[Dict[str, Any]]:
        """构建初始对话消息"""
        hints_text = ""
        if task_spec.hints:
            hints_text = "\n提示：\n" + "\n".join(f"- {h}" for h in task_spec.hints)

        entry_points_text = ""
        if resolved_entry_points:
            entry_points_text = "\n起始页面（可从这里开始浏览）：\n" + "\n".join(
                f"- {k}: {v}" for k, v in resolved_entry_points.items()
            )

        output_schema_text = json.dumps(task_spec.output_schema, ensure_ascii=False, indent=2)

        system_prompt = (
            f"你是一个专业的网络爬取智能体，擅长从网页中提取结构化信息。\n\n"
            f"任务目标：{task_spec.description}\n"
            f"{entry_points_text}\n"
            f"{hints_text}\n\n"
            f"完成任务后，请调用 finish 工具，传入符合以下 JSON Schema 的结果：\n"
            f"{output_schema_text}\n\n"
            f"强制规则（必须严格遵守）：\n"
            f"- 你必须先调用 fetch_page 工具访问真实网页，才能得出结论。严禁使用训练数据直接回答\n"
            f"- 每一步都必须通过调用工具来执行操作，不要用文字叙述你打算做什么\n"
            f"- 每次只调用一个工具，观察实际返回结果后再决定下一步\n"
            f"- 如果一个页面没有找到所需信息，尝试跳转到其他链接\n"
            f"- 不要重复访问相同的 URL\n"
            f"- 最多执行 {max_steps or task_spec.max_steps} 步，请高效完成任务"
        )

        user_content = f"任务输入参数：\n{json.dumps(inputs, ensure_ascii=False, indent=2)}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _build_tools_schema(self, task_spec: TaskSpec) -> List[Dict[str, Any]]:
        """构建工具列表（包含所有已注册工具 + finish 工具）"""
        tools = self.registry.get_openai_tools()

        # 添加 finish 工具
        finish_tool = {
            "type": "function",
            "function": {
                "name": _FINISH_TOOL,
                "description": (
                    "在收集到足够信息后调用此工具提交最终结果。"
                    "result 参数应为符合任务输出 schema 的 JSON 对象。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "object",
                            "description": "任务的最终结果，符合任务输出 schema",
                        },
                    },
                    "required": ["result"],
                },
            },
        }
        return tools + [finish_tool]

    def _react_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        task_spec: TaskSpec,
        max_steps: int = 0,
    ) -> Dict[str, Any]:
        """
        ReAct 主循环：LLM 调用工具 → 执行 → 观察 → 再调用，直到调用 finish。
        """
        effective = max_steps or task_spec.max_steps
        visited_urls: set = set()

        for step in range(effective):
            log_structured(
                self.logger,
                "AGENT",
                f"Step {step + 1}/{effective}",
                level="DEBUG",
            )
            if self.verbose:
                print(f"\n[Step {step + 1}/{effective}] 正在思考...", flush=True)

            # 调用 LLM（Function Calling 模式，tool_choice="required" 强制必须调用工具）
            try:
                response = self.llm.chat_completion_with_tools(
                    messages, tools, tool_choice="required"
                )
            except Exception as e:
                raise RuntimeError(f"LLM 调用失败: {e}") from e

            tool_calls = response.get("tool_calls")
            text_content = response.get("content")

            # LLM 返回纯文本而非工具调用（通常是结束信号或错误）
            if not tool_calls:
                if text_content:
                    # 尝试从文本中解析 JSON 结果
                    parsed = _try_parse_json(text_content)
                    if parsed and isinstance(parsed, dict):
                        return parsed
                raise RuntimeError(
                    f"第 {step + 1} 步 LLM 返回了文本而非工具调用，无法提取结果。\n"
                    f"LLM 输出：{text_content}"
                )

            # 将 LLM 的 tool_calls 加入消息历史（OpenAI 格式要求）
            messages.append(
                {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # 执行所有工具调用
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_call_id = tc["id"]

                # 检测重复访问同一 URL
                if "url" in tool_args:
                    url = tool_args["url"]
                    if url in visited_urls:
                        observation = f"[跳过] URL '{url}' 已经访问过，请选择其他链接。"
                    else:
                        visited_urls.add(url)
                        observation = None
                else:
                    observation = None

                # finish 工具：提取结果并退出循环
                if tool_name == _FINISH_TOOL:
                    result = tool_args.get("result", {})
                    log_structured(
                        self.logger,
                        "AGENT",
                        f"Task finished at step {step + 1}",
                        result_keys=list(result.keys()) if isinstance(result, dict) else [],
                    )
                    if self.verbose:
                        print(f"[Step {step + 1}] ✓ finish — 任务完成", flush=True)
                    return result if isinstance(result, dict) else {"result": result}

                # 执行普通工具
                if observation is None:
                    log_structured(
                        self.logger,
                        "AGENT",
                        f"Calling tool: {tool_name}",
                        args={k: str(v)[:100] for k, v in tool_args.items()},
                        level="DEBUG",
                    )
                    # verbose：打印工具调用摘要
                    if self.verbose:
                        args_summary = "  ".join(
                            f"{k}={str(v)[:80]!r}" for k, v in tool_args.items()
                        )
                        print(f"[Step {step + 1}] → {tool_name}({args_summary})", flush=True)
                    observation = self.registry.execute(tool_name, tool_args)
                elif self.verbose:
                    print(f"[Step {step + 1}] ⚠ 跳过重复 URL: {tool_args.get('url', '')}", flush=True)

                log_structured(
                    self.logger,
                    "AGENT",
                    f"Tool '{tool_name}' result ({len(observation)} chars)",
                    level="DEBUG",
                )
                if self.verbose:
                    print(f"         ← {len(observation)} 字符", flush=True)

                # 将工具结果加入消息历史
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": observation[:6000],  # 截断防止超出上下文
                    }
                )

        raise RuntimeError(
            f"任务 '{task_spec.task_id}' 在 {effective} 步内未完成。"
            "请检查任务规格或增加 agent.max_steps 配置项。"
        )


# --------------------------------------------------------------------------- #
#  辅助函数                                                                    #
# --------------------------------------------------------------------------- #

def _resolve_entry_points(
    entry_points: Dict[str, str],
    inputs: Dict[str, Any],
) -> Dict[str, str]:
    """将 entry_points 中的 {变量} 替换为实际输入值"""
    resolved = {}
    for key, url_template in entry_points.items():
        try:
            resolved[key] = url_template.format(**inputs)
        except KeyError:
            resolved[key] = url_template  # 模板变量缺失时保留原样
    return resolved


def _try_parse_json(text: str) -> Optional[Any]:
    """尝试从文本中提取 JSON 对象"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { ... } 块
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
