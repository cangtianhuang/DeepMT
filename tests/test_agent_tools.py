"""
Phase 1 单元测试：工具函数、ToolRegistry、TaskSpec、ReAct 循环

所有测试均不依赖真实网络或 LLM 调用（使用 mock）。
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
#  工具函数测试（tools/agent/tools/）                                           #
# --------------------------------------------------------------------------- #

class TestExtractLinks:
    """extract_links 工具函数测试"""

    def test_basic_links(self):
        from tools.agent.tools.extract import extract_links

        html = """
        <html><body>
          <a href="https://example.com/a">Link A</a>
          <a href="https://example.com/b">Link B</a>
        </body></html>
        """
        result = json.loads(extract_links(html))
        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/a"
        assert result[0]["text"] == "Link A"

    def test_relative_url_resolution(self):
        from tools.agent.tools.extract import extract_links

        html = '<a href="/docs/relu">ReLU</a>'
        result = json.loads(extract_links(html, base_url="https://pytorch.org"))
        assert result[0]["url"] == "https://pytorch.org/docs/relu"

    def test_filter_pattern(self):
        from tools.agent.tools.extract import extract_links

        html = """
        <a href="/releases/tag/v2.5.1">v2.5.1</a>
        <a href="/issues/123">Issue</a>
        <a href="/releases/tag/v2.4.0">v2.4.0</a>
        """
        result = json.loads(extract_links(html, filter_pattern=r"/releases/tag/"))
        assert len(result) == 2
        assert all("/releases/tag/" in r["url"] for r in result)

    def test_deduplicate_urls(self):
        from tools.agent.tools.extract import extract_links

        html = """
        <a href="https://example.com">Link 1</a>
        <a href="https://example.com">Link 2</a>
        """
        result = json.loads(extract_links(html))
        urls = [r["url"] for r in result]
        assert len(urls) == len(set(urls))

    def test_skip_anchor_and_javascript(self):
        from tools.agent.tools.extract import extract_links

        html = """
        <a href="#section">Anchor</a>
        <a href="javascript:void(0)">JS</a>
        <a href="https://valid.com">Valid</a>
        """
        result = json.loads(extract_links(html))
        assert len(result) == 1
        assert result[0]["url"] == "https://valid.com"

    def test_max_results(self):
        from tools.agent.tools.extract import extract_links

        links = "".join(f'<a href="https://example.com/{i}">Link {i}</a>' for i in range(100))
        html = f"<html><body>{links}</body></html>"
        result = json.loads(extract_links(html, max_results=10))
        assert len(result) == 10

    def test_empty_html(self):
        from tools.agent.tools.extract import extract_links

        result = json.loads(extract_links(""))
        assert result == []


class TestExtractText:
    """extract_text 工具函数测试"""

    def test_basic_text_extraction(self):
        from tools.agent.tools.extract import extract_text

        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        result = extract_text(html)
        assert "Title" in result
        assert "Content" in result

    def test_css_selector(self):
        from tools.agent.tools.extract import extract_text

        html = """
        <html><body>
          <div class="main">Main Content</div>
          <div class="sidebar">Sidebar</div>
        </body></html>
        """
        result = extract_text(html, selector=".main")
        assert "Main Content" in result
        assert "Sidebar" not in result

    def test_selector_not_found(self):
        from tools.agent.tools.extract import extract_text

        html = "<html><body><p>Text</p></body></html>"
        result = extract_text(html, selector=".nonexistent")
        assert "[未找到]" in result

    def test_strips_scripts_and_styles(self):
        from tools.agent.tools.extract import extract_text

        html = """
        <html><body>
          <script>alert('hidden');</script>
          <style>.x { color: red; }</style>
          <p>Visible</p>
        </body></html>
        """
        result = extract_text(html)
        assert "alert" not in result
        assert "color: red" not in result
        assert "Visible" in result

    def test_max_chars_truncation(self):
        from tools.agent.tools.extract import extract_text

        long_text = "A" * 5000
        html = f"<html><body><p>{long_text}</p></body></html>"
        result = extract_text(html, max_chars=100)
        assert len(result) <= 200  # 允许截断提示信息
        assert "[已截断" in result


class TestFindVersionTags:
    """find_version_tags 工具函数测试"""

    def test_basic_version_extraction(self):
        from tools.agent.tools.extract import find_version_tags

        text = "PyTorch 2.5.1 was released on Oct 2024\nPyTorch 2.4.0 was released on..."
        result = json.loads(find_version_tags(text))
        versions = [r["version"] for r in result]
        assert "2.5.1" in versions
        assert "2.4.0" in versions

    def test_detects_latest_marker(self):
        from tools.agent.tools.extract import find_version_tags

        text = "Latest stable release: 2.5.1\nOlder release: 2.4.0"
        result = json.loads(find_version_tags(text))
        latest_items = [r for r in result if r["is_latest"]]
        assert len(latest_items) >= 1
        assert latest_items[0]["version"] == "2.5.1"

    def test_handles_v_prefix(self):
        from tools.agent.tools.extract import find_version_tags

        text = "Release v2.5.1 is now available"
        result = json.loads(find_version_tags(text))
        versions = [r["version"] for r in result]
        assert "2.5.1" in versions

    def test_html_input(self):
        from tools.agent.tools.extract import find_version_tags

        html = """
        <html><body>
          <h1>Latest Release: v2.5.1</h1>
          <p>Previous: 2.4.0</p>
        </body></html>
        """
        result = json.loads(find_version_tags(html))
        versions = [r["version"] for r in result]
        assert "2.5.1" in versions

    def test_no_version_found(self):
        from tools.agent.tools.extract import find_version_tags

        result = json.loads(find_version_tags("No version information here."))
        assert result == []

    def test_deduplicates_versions(self):
        from tools.agent.tools.extract import find_version_tags

        text = "version 2.5.1 and again 2.5.1 mentioned twice"
        result = json.loads(find_version_tags(text))
        versions = [r["version"] for r in result]
        assert versions.count("2.5.1") == 1


class TestSearchInText:
    """search_in_text 工具函数测试"""

    def test_basic_search(self):
        from tools.agent.tools.extract import search_in_text

        text = "line one\ntorch.relu function\nline three"
        result = search_in_text(text, r"torch\.relu")
        assert "torch.relu" in result
        assert "line one" not in result

    def test_case_insensitive(self):
        from tools.agent.tools.extract import search_in_text

        text = "PyTorch\npytorch\nPYTORCH"
        result = search_in_text(text, "pytorch")
        assert result.count(":") >= 3  # 三行都匹配

    def test_line_numbers(self):
        from tools.agent.tools.extract import search_in_text

        text = "alpha\nbeta\ngamma"
        result = search_in_text(text, "beta")
        assert "2:" in result  # beta 在第 2 行

    def test_invalid_regex(self):
        from tools.agent.tools.extract import search_in_text

        result = search_in_text("some text", "[invalid")
        assert "[正则错误]" in result

    def test_no_match(self):
        from tools.agent.tools.extract import search_in_text

        result = search_in_text("hello world", "notfound")
        assert "[未找到]" in result

    def test_max_results_limit(self):
        from tools.agent.tools.extract import search_in_text

        text = "\n".join(["match"] * 50)
        result = search_in_text(text, "match", max_results=5)
        lines = [l for l in result.splitlines() if ":" in l]
        assert len(lines) <= 5


class TestFetchPageOffline:
    """fetch_page / fetch_page_html 离线测试（mock requests）"""

    def test_fetch_page_returns_text(self):
        from tools.agent.tools.browse import fetch_page

        mock_html = "<html><body><main><p>Hello World</p></main></body></html>"
        with patch("tools.agent.tools.browse._get_html", return_value=mock_html):
            result = fetch_page("https://example.com")
        assert "Hello World" in result
        assert "<html>" not in result  # 应返回纯文本

    def test_fetch_page_with_selector(self):
        from tools.agent.tools.browse import fetch_page

        mock_html = """
        <html><body>
          <div class="target">Target Content</div>
          <div class="other">Other</div>
        </body></html>
        """
        with patch("tools.agent.tools.browse._get_html", return_value=mock_html):
            result = fetch_page("https://example.com", selector=".target")
        assert "Target Content" in result
        assert "Other" not in result

    def test_fetch_page_error(self):
        from tools.agent.tools.browse import fetch_page

        with patch("tools.agent.tools.browse._get_html", return_value=None):
            result = fetch_page("https://unreachable.example.com")
        assert "[错误]" in result

    def test_fetch_page_html_returns_html(self):
        from tools.agent.tools.browse import fetch_page_html

        mock_html = "<html><body><p>Content</p></body></html>"
        with patch("tools.agent.tools.browse._get_html", return_value=mock_html):
            result = fetch_page_html("https://example.com")
        assert "<html>" in result
        assert "<p>" in result


# --------------------------------------------------------------------------- #
#  ToolRegistry 测试                                                            #
# --------------------------------------------------------------------------- #

class TestToolRegistry:
    """ToolRegistry 注册表测试"""

    def test_register_and_execute(self):
        from tools.agent.tool_registry import ToolDef, ToolRegistry

        def add(a: int, b: int) -> str:
            return str(a + b)

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="add",
                description="Add two numbers",
                func=add,
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            )
        )

        result = registry.execute("add", {"a": 3, "b": 4})
        assert result == "7"

    def test_execute_unknown_tool(self):
        from tools.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_get_openai_tools_format(self):
        from tools.agent.tool_registry import ToolDef, ToolRegistry

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="my_tool",
                description="A test tool",
                func=lambda: "ok",
                parameters={"type": "object", "properties": {}},
            )
        )

        tools = registry.get_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "my_tool"
        assert "description" in tools[0]["function"]
        assert "parameters" in tools[0]["function"]

    def test_tool_decorator_registration(self):
        from tools.agent.tool_registry import ToolRegistry, tool

        @tool(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        registry = ToolRegistry()
        registry.register_func(greet)

        assert registry.has("greet")
        assert registry.execute("greet", {"name": "World"}) == "Hello, World!"

    def test_execute_with_exception_returns_error_json(self):
        from tools.agent.tool_registry import ToolDef, ToolRegistry

        def broken():
            raise ValueError("Something went wrong")

        registry = ToolRegistry()
        registry.register(
            ToolDef("broken", "A broken tool", broken, {"type": "object", "properties": {}})
        )
        result = json.loads(registry.execute("broken", {}))
        assert "error" in result

    def test_build_default_registry(self):
        from tools.agent.tool_registry import build_default_registry

        registry = build_default_registry()
        expected_tools = [
            "fetch_page",
            "fetch_page_html",
            "extract_links",
            "extract_text",
            "find_version_tags",
            "search_in_text",
        ]
        for name in expected_tools:
            assert registry.has(name), f"Expected tool '{name}' not found in registry"


# --------------------------------------------------------------------------- #
#  TaskSpec 测试                                                                #
# --------------------------------------------------------------------------- #

class TestTaskSpec:
    """TaskSpec 加载测试"""

    def test_load_from_yaml_file(self, tmp_path):
        from tools.agent.agent_core import TaskSpec

        yaml_content = """
task_id: test_task
description: A test task
inputs:
  - name: framework
    type: string
output_schema:
  version: string
entry_points:
  pytorch: "https://example.com/{framework}"
hints:
  - "Do something"
max_steps: 5
cache_ttl_days: 1
"""
        yaml_file = tmp_path / "test_task.yaml"
        yaml_file.write_text(yaml_content)

        spec = TaskSpec.from_yaml(str(yaml_file))
        assert spec.task_id == "test_task"
        assert spec.max_steps == 5
        assert spec.cache_ttl_days == 1
        assert len(spec.hints) == 1
        assert "pytorch" in spec.entry_points

    def test_load_builtin_get_framework_version(self):
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_framework_version")
        assert spec.task_id == "get_framework_version"
        assert "pytorch" in spec.entry_points
        assert "tensorflow" in spec.entry_points
        assert "paddlepaddle" in spec.entry_points
        assert spec.max_steps > 0

    def test_load_builtin_get_operator_list(self):
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_list")
        assert spec.task_id == "get_operator_list"
        assert len(spec.hints) > 0

    def test_load_builtin_get_operator_doc(self):
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_doc")
        assert spec.task_id == "get_operator_doc"

    def test_load_nonexistent_task(self):
        from tools.agent.agent_core import TaskSpec

        with pytest.raises(FileNotFoundError, match="不存在"):
            TaskSpec.from_task_id("nonexistent_task_xyz")


# --------------------------------------------------------------------------- #
#  CrawlAgent ReAct 循环测试（全 mock，不依赖 LLM 和网络）                      #
# --------------------------------------------------------------------------- #

class TestCrawlAgentReActLoop:
    """CrawlAgent 核心逻辑测试"""

    def _make_agent_with_mock_llm(self, tool_call_sequence):
        """
        创建一个 CrawlAgent，其 LLM 按照 tool_call_sequence 依次返回工具调用。

        tool_call_sequence: list of
          - dict {"tool_calls": [...]}  表示一次工具调用响应
          - dict {"content": "..."}     表示纯文本响应
        """
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = tool_call_sequence

        registry = build_default_registry()
        agent = CrawlAgent(registry=registry, llm_client=mock_llm)
        return agent

    def _make_simple_spec(self) -> "TaskSpec":
        from tools.agent.agent_core import TaskSpec

        return TaskSpec(
            task_id="test",
            description="Test task",
            inputs=[],
            output_schema={"version": "string"},
            entry_points={"pytorch": "https://example.com"},
            hints=["Test hint"],
            max_steps=5,
            cache_ttl_days=1,
        )

    def test_finish_on_first_call(self):
        """智能体第一步就调用 finish，直接返回结果"""
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "name": "finish",
                    "arguments": {"result": {"version": "2.5.1"}},
                }
            ],
        }

        agent = CrawlAgent(
            registry=build_default_registry(),
            llm_client=mock_llm,
        )
        result = agent.run(self._make_simple_spec(), inputs={"framework": "pytorch"})
        assert result == {"version": "2.5.1"}

    def test_tool_call_then_finish(self):
        """智能体先调用一个工具，观察结果后调用 finish"""
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = [
            # Step 1: 调用 fetch_page
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "name": "fetch_page",
                        "arguments": {"url": "https://example.com"},
                    }
                ],
            },
            # Step 2: 调用 finish
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_002",
                        "name": "finish",
                        "arguments": {"result": {"version": "2.5.1", "release_url": "https://example.com"}},
                    }
                ],
            },
        ]

        with patch("tools.agent.tools.browse._get_html", return_value="<html><body>PyTorch 2.5.1</body></html>"):
            agent = CrawlAgent(
                registry=build_default_registry(),
                llm_client=mock_llm,
            )
            result = agent.run(self._make_simple_spec(), inputs={"framework": "pytorch"})

        assert result["version"] == "2.5.1"
        assert mock_llm.chat_completion_with_tools.call_count == 2

    def test_max_steps_exceeded_raises(self):
        """超过 max_steps 应抛出 RuntimeError"""
        from tools.agent.agent_core import CrawlAgent, TaskSpec
        from tools.agent.tool_registry import build_default_registry

        # LLM 永远返回 fetch_page（每次不同 URL 避免重复跳过），永不 finish
        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = [
            {
                "content": None,
                "tool_calls": [{"id": f"c{i}", "name": "fetch_page", "arguments": {"url": f"https://example.com/{i}"}}],
            }
            for i in range(10)
        ]

        spec = TaskSpec(
            task_id="test",
            description="Test",
            inputs=[],
            output_schema={},
            entry_points={},
            hints=[],
            max_steps=3,  # 只允许 3 步
            cache_ttl_days=1,
        )

        with patch("tools.agent.tools.browse._get_html", return_value="<html><body></body></html>"):
            agent = CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)
            with pytest.raises(RuntimeError, match="未完成"):
                agent.run(spec, inputs={})

    def test_duplicate_url_skipped(self):
        """重复访问同一 URL 应被跳过，不会再次 fetch"""
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        call_count = {"n": 0}

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = [
            # 第 1 步：访问 URL
            {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "fetch_page", "arguments": {"url": "https://dup.com"}}],
            },
            # 第 2 步：再次访问同一 URL（应被跳过）
            {
                "content": None,
                "tool_calls": [{"id": "c2", "name": "fetch_page", "arguments": {"url": "https://dup.com"}}],
            },
            # 第 3 步：finish
            {
                "content": None,
                "tool_calls": [{"id": "c3", "name": "finish", "arguments": {"result": {"done": True}}}],
            },
        ]

        def mock_get_html(url):
            call_count["n"] += 1
            return "<html><body>content</body></html>"

        with patch("tools.agent.tools.browse._get_html", side_effect=mock_get_html):
            agent = CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)
            result = agent.run(self._make_simple_spec(), inputs={})

        # 实际网络调用只应发生 1 次
        assert call_count["n"] == 1
        assert result == {"done": True}


# --------------------------------------------------------------------------- #
#  TaskRunner 缓存测试                                                          #
# --------------------------------------------------------------------------- #

class TestTaskRunnerCache:
    """TaskRunner 缓存机制测试"""

    def test_cache_write_and_read(self, tmp_path):
        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path))

        # 直接测试 _save_cache / _load_cache，不涉及 agent
        runner._save_cache("test_key", {"version": "2.5.1"})

        cached = runner._load_cache("test_key", ttl_days=7)
        assert cached is not None
        assert cached["version"] == "2.5.1"
        # 内部元数据字段不应暴露
        assert "_saved_at" not in cached

    def test_cache_expired(self, tmp_path):
        import json
        from datetime import datetime, timedelta

        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path))

        # 写入一个过期的缓存文件
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        cache_file = tmp_path / "old_key.json"
        cache_file.write_text(
            json.dumps({"_saved_at": old_time, "version": "1.0.0"})
        )

        result = runner._load_cache("old_key", ttl_days=7)
        assert result is None  # 已过期

    def test_cache_key_deterministic(self):
        from tools.agent.task_runner import _make_cache_key

        key1 = _make_cache_key("get_framework_version", {"framework": "pytorch"})
        key2 = _make_cache_key("get_framework_version", {"framework": "pytorch"})
        key3 = _make_cache_key("get_framework_version", {"framework": "tensorflow"})

        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("get_framework_version_")

    def test_clear_cache(self, tmp_path):
        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path))
        runner._save_cache("task_a_001", {"data": 1})
        runner._save_cache("task_a_002", {"data": 2})
        runner._save_cache("task_b_001", {"data": 3})

        count = runner.clear_cache(task_id="task_a")
        assert count == 2

        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == 1
        assert "task_b" in remaining[0].name

    def test_list_cache(self, tmp_path):
        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path))
        runner._save_cache("key1", {"a": 1})
        runner._save_cache("key2", {"b": 2})

        entries = runner.list_cache()
        assert len(entries) == 2
        assert all("saved_at" in e for e in entries)
        assert all("size_bytes" in e for e in entries)
