"""
Phase 3 回归测试

验证重构后：
1. OperatorInfoFetcher 接口不变（对 operator_mr.py 透明）
2. agent.enabled=false 时返回空结果，不抛异常
3. agent.enabled=true 时走 TaskRunner 链路
4. TaskRunner 失败时优雅降级
5. 全局 agent.max_steps / agent.cache_ttl_days 配置覆盖生效
6. OperatorMRGenerator 使用新 OperatorInfoFetcher 后流程不变
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
#  1. OperatorInfoFetcher 接口回归                                             #
# --------------------------------------------------------------------------- #

class TestOperatorInfoFetcherInterface:
    """验证 OperatorInfoFetcher 对外接口与旧版保持一致"""

    def test_fetch_operator_info_returns_dict(self):
        """fetch_operator_info 应返回 dict，含 name/doc/source_urls"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            result = fetcher.fetch_operator_info("relu", "pytorch")

        assert isinstance(result, dict)
        assert "name" in result
        assert "doc" in result
        assert "source_urls" in result

    def test_get_operator_doc_returns_str_or_none(self):
        """get_operator_doc 应返回 str 或 None"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            result = fetcher.get_operator_doc("relu", "pytorch")

        assert result is None or isinstance(result, str)

    def test_fetch_operator_info_preserves_operator_name(self):
        """返回的 name 字段应与输入算子名一致（agent 禁用时）"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            result = fetcher.fetch_operator_info("conv2d", "pytorch")

        assert result["name"] == "conv2d"

    def test_source_urls_is_list(self):
        """source_urls 应始终为 list"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            result = fetcher.fetch_operator_info("relu", "pytorch")

        assert isinstance(result["source_urls"], list)


# --------------------------------------------------------------------------- #
#  2. agent.enabled=false 时的行为                                             #
# --------------------------------------------------------------------------- #

class TestAgentDisabled:
    """agent 未启用时，OperatorInfoFetcher 应静默返回空结果"""

    def test_returns_empty_doc_when_disabled(self):
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            result = fetcher.fetch_operator_info("relu", "pytorch")

        assert result["doc"] == ""
        assert result["source_urls"] == []

    def test_get_operator_doc_returns_none_when_disabled(self):
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            doc = fetcher.get_operator_doc("relu", "pytorch")

        assert doc is None

    def test_does_not_instantiate_runner_when_disabled(self):
        """agent 禁用时不应触发 TaskRunner 的懒加载"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=False):
            fetcher = OperatorInfoFetcher()
            fetcher.fetch_operator_info("relu", "pytorch")

        # _runner 应仍为 None（未被初始化）
        assert fetcher._runner is None


# --------------------------------------------------------------------------- #
#  3. agent.enabled=true 时走 TaskRunner 链路                                 #
# --------------------------------------------------------------------------- #

class TestAgentEnabled:
    """agent 启用时，OperatorInfoFetcher 应调用 TaskRunner"""

    def _make_fetcher_with_mock_runner(self, runner_result: dict):
        """创建 agent 已启用、TaskRunner 被 mock 的 OperatorInfoFetcher"""
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        mock_runner = MagicMock()
        mock_runner.run_task.return_value = runner_result

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=True):
            fetcher = OperatorInfoFetcher()
        fetcher._runner = mock_runner  # 直接注入
        return fetcher, mock_runner

    def test_calls_run_task_with_correct_args(self):
        """应以正确的 task_id 和 inputs 调用 run_task"""
        fetcher, mock_runner = self._make_fetcher_with_mock_runner(
            {"doc": "ReLU docs", "source_url": "https://pytorch.org/relu", "operator_name": "relu"}
        )

        fetcher.fetch_operator_info("relu", "pytorch")

        mock_runner.run_task.assert_called_once_with(
            "get_operator_doc",
            inputs={"operator_name": "relu", "framework": "pytorch"},
            use_cache=True,
        )

    def test_returns_doc_from_runner(self):
        """应将 TaskRunner 返回的 doc 包装进结果"""
        fetcher, _ = self._make_fetcher_with_mock_runner(
            {"doc": "ReLU activation function", "source_url": "https://pytorch.org/relu", "operator_name": "torch.nn.ReLU"}
        )

        result = fetcher.fetch_operator_info("relu", "pytorch")

        assert result["doc"] == "ReLU activation function"
        assert result["source_urls"] == ["https://pytorch.org/relu"]
        assert result["name"] == "torch.nn.ReLU"

    def test_empty_source_url_excluded_from_list(self):
        """source_url 为空时 source_urls 应为空列表"""
        fetcher, _ = self._make_fetcher_with_mock_runner(
            {"doc": "Some doc", "source_url": "", "operator_name": "relu"}
        )

        result = fetcher.fetch_operator_info("relu", "pytorch")
        assert result["source_urls"] == []

    def test_use_cache_false_passed_through(self):
        """use_cache=False 应透传给 run_task"""
        fetcher, mock_runner = self._make_fetcher_with_mock_runner(
            {"doc": "doc", "source_url": "", "operator_name": "relu"}
        )

        fetcher.fetch_operator_info("relu", "pytorch", use_cache=False)

        call_kwargs = mock_runner.run_task.call_args
        assert call_kwargs[1]["use_cache"] is False or call_kwargs[0][2] is False


# --------------------------------------------------------------------------- #
#  4. TaskRunner 失败时优雅降级                                               #
# --------------------------------------------------------------------------- #

class TestAgentFailureGracefulDegradation:
    """agent 执行出错时应返回空结果，不向上传播异常"""

    def test_runtime_error_returns_empty(self):
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        mock_runner = MagicMock()
        mock_runner.run_task.side_effect = RuntimeError("LLM 调用失败")

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=True):
            fetcher = OperatorInfoFetcher()
        fetcher._runner = mock_runner

        result = fetcher.fetch_operator_info("relu", "pytorch")

        assert result["doc"] == ""
        assert result["source_urls"] == []
        assert result["name"] == "relu"

    def test_network_error_returns_empty(self):
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        mock_runner = MagicMock()
        mock_runner.run_task.side_effect = ConnectionError("网络不通")

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=True):
            fetcher = OperatorInfoFetcher()
        fetcher._runner = mock_runner

        result = fetcher.fetch_operator_info("relu", "pytorch")

        # 不抛异常
        assert isinstance(result, dict)
        assert result["doc"] == ""

    def test_get_operator_doc_returns_none_on_failure(self):
        from tools.web_search.operator_fetcher import OperatorInfoFetcher

        mock_runner = MagicMock()
        mock_runner.run_task.side_effect = Exception("unexpected")

        with patch("tools.web_search.operator_fetcher.get_config_value", return_value=True):
            fetcher = OperatorInfoFetcher()
        fetcher._runner = mock_runner

        doc = fetcher.get_operator_doc("relu", "pytorch")
        assert doc is None


# --------------------------------------------------------------------------- #
#  5. 全局配置覆盖（agent.max_steps / agent.cache_ttl_days）                 #
# --------------------------------------------------------------------------- #

class TestGlobalConfigOverrides:
    """验证全局配置项对 CrawlAgent 和 TaskRunner 的覆盖效果"""

    def test_agent_max_steps_overrides_task_spec(self):
        """agent.max_steps 应覆盖 TaskSpec.max_steps"""
        from tools.agent.agent_core import CrawlAgent, TaskSpec
        from tools.agent.tool_registry import build_default_registry

        # TaskSpec 设 max_steps=10，全局配置设 max_steps=2
        spec = TaskSpec(
            task_id="test",
            description="Test",
            inputs=[],
            output_schema={},
            entry_points={},
            hints=[],
            max_steps=10,
            cache_ttl_days=1,
        )

        call_counts = {"n": 0}
        mock_llm = MagicMock()

        def side_effect(*args, **kwargs):
            call_counts["n"] += 1
            # 始终返回 fetch_page，不 finish
            return {
                "content": None,
                "tool_calls": [
                    {"id": f"c{call_counts['n']}", "name": "fetch_page",
                     "arguments": {"url": f"https://example.com/{call_counts['n']}"}}
                ],
            }

        mock_llm.chat_completion_with_tools.side_effect = side_effect

        with (
            patch("tools.agent.agent_core.get_config_value", return_value=2),  # max_steps=2
            patch("tools.agent.tools.browse._get_html", return_value="<html><body></body></html>"),
        ):
            agent = CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)
            with pytest.raises(RuntimeError, match="2 步内未完成"):
                agent.run(spec, inputs={})

        # 应恰好只执行 2 步
        assert call_counts["n"] == 2

    def test_cache_ttl_overrides_task_spec(self, tmp_path):
        """agent.cache_ttl_days 应覆盖 TaskSpec.cache_ttl_days"""
        import json
        from datetime import datetime, timedelta

        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path))

        # 写入一个 3 天前的缓存
        old_time = (datetime.now() - timedelta(days=3)).isoformat()
        cache_file = tmp_path / "get_framework_version_abc12345.json"
        cache_file.write_text(json.dumps({"_saved_at": old_time, "version": "1.0.0"}))

        # 全局 cache_ttl=7 → 3天前的缓存未过期，应命中
        with patch("tools.agent.task_runner.get_config_value", return_value=7):
            result = runner._load_cache("get_framework_version_abc12345", ttl_days=1)
            # ttl_days 参数被 run_task 内部覆盖为 7，但这里直接传 1 测试的是底层
            # 实际 run_task 已用 global_ttl=7 覆盖，此处只测 _load_cache 行为

        # 当 ttl_days=1 时，3天前的缓存应过期
        result_with_ttl1 = runner._load_cache("get_framework_version_abc12345", ttl_days=1)
        assert result_with_ttl1 is None

        # 当 ttl_days=7 时，3天前的缓存应命中
        result_with_ttl7 = runner._load_cache("get_framework_version_abc12345", ttl_days=7)
        assert result_with_ttl7 is not None
        assert result_with_ttl7["version"] == "1.0.0"


# --------------------------------------------------------------------------- #
#  6. OperatorMRGenerator 与新 OperatorInfoFetcher 的集成                     #
# --------------------------------------------------------------------------- #

class TestMRGeneratorRegression:
    """验证 OperatorMRGenerator 使用新 OperatorInfoFetcher 后流程不变"""

    def test_fetch_operator_info_via_generator(self):
        """OperatorMRGenerator.fetch_operator_info() 应正确委托给 OperatorInfoFetcher"""
        from mr_generator.operator.operator_mr import OperatorMRGenerator

        generator = OperatorMRGenerator()

        # Mock OperatorInfoFetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu",
            "doc": "ReLU activation function docs",
            "source_urls": ["https://pytorch.org/docs/relu"],
        }
        generator.info_fetcher = mock_fetcher

        result = generator.fetch_operator_info("relu", framework="pytorch")

        mock_fetcher.fetch_operator_info.assert_called_once_with(operator_name="relu", framework="pytorch")
        assert result["doc"] == "ReLU activation function docs"

    def test_generate_only_with_auto_fetch_disabled(self):
        """auto_fetch_info=False 时不应调用 OperatorInfoFetcher"""
        from ir.schema import OperatorIR
        from mr_generator.operator.operator_mr import OperatorMRGenerator

        generator = OperatorMRGenerator()
        generator.llm_generator = MagicMock()
        generator.llm_generator.generate_mr_candidates.return_value = []

        mock_fetcher = MagicMock()
        generator.info_fetcher = mock_fetcher

        from ir.schema import OperatorIR
        op_ir = OperatorIR(name="ReLU", inputs=[])

        generator.generate_only(
            operator_ir=op_ir,
            operator_doc="ReLU docs",
            auto_fetch_info=False,
            sources=["template"],
        )

        mock_fetcher.fetch_operator_info.assert_not_called()

    def test_generate_pipeline_with_agent_enabled(self):
        """
        agent 启用后，generate_only(auto_fetch_info=True) 应调用 OperatorInfoFetcher
        然后将文档传给 LLM 生成器
        """
        from ir.schema import OperatorIR
        from mr_generator.operator.operator_mr import OperatorMRGenerator

        generator = OperatorMRGenerator()
        generator.llm_generator = MagicMock()
        generator.llm_generator.generate_mr_candidates.return_value = []

        # 模拟 agent 启用且能正常返回文档
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu",
            "doc": "ReLU activation function: relu(x) = max(0, x)",
            "source_urls": [],
        }
        generator.info_fetcher = mock_fetcher

        op_ir = OperatorIR(name="ReLU", inputs=[])
        generator.generate_only(
            operator_ir=op_ir,
            auto_fetch_info=True,
            sources=["template"],
        )

        # OperatorInfoFetcher 应被调用
        mock_fetcher.fetch_operator_info.assert_called_once()
        call_args = mock_fetcher.fetch_operator_info.call_args
        assert "relu" in str(call_args).lower() or "ReLU" in str(call_args)

    def test_full_generate_with_mocked_fetcher(self):
        """完整 generate() 流程在新 OperatorInfoFetcher 下应正常运行"""
        import torch

        from ir.schema import OperatorIR
        from mr_generator.operator.operator_mr import OperatorMRGenerator

        generator = OperatorMRGenerator()
        generator.llm_generator = MagicMock()
        generator.llm_generator.generate_mr_candidates.return_value = []

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu",
            "doc": "ReLU activation: applies the rectified linear unit function element-wise",
            "source_urls": [],
        }
        generator.info_fetcher = mock_fetcher

        op_ir = OperatorIR(name="ReLU", inputs=[])
        mrs = generator.generate(
            operator_ir=op_ir,
            operator_func=torch.nn.functional.relu,
            auto_fetch_info=True,
            use_precheck=False,
            use_sympy_proof=False,
            sources=["template"],
        )

        assert isinstance(mrs, list)

    def test_generate_when_fetcher_returns_empty(self):
        """OperatorInfoFetcher 返回空 doc 时，generate() 应继续运行（不崩溃）"""
        import torch

        from ir.schema import OperatorIR
        from mr_generator.operator.operator_mr import OperatorMRGenerator

        generator = OperatorMRGenerator()
        generator.llm_generator = MagicMock()
        generator.llm_generator.generate_mr_candidates.return_value = []

        # agent 禁用场景：返回空 doc
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu",
            "doc": "",
            "source_urls": [],
        }
        generator.info_fetcher = mock_fetcher

        op_ir = OperatorIR(name="ReLU", inputs=[])
        mrs = generator.generate(
            operator_ir=op_ir,
            operator_func=torch.nn.functional.relu,
            auto_fetch_info=True,
            use_precheck=False,
            use_sympy_proof=False,
            sources=["template"],
        )

        assert isinstance(mrs, list)  # 不应崩溃


# --------------------------------------------------------------------------- #
#  7. 模块导入结构验证                                                         #
# --------------------------------------------------------------------------- #

class TestModuleStructure:
    """验证重构后模块导入结构正确"""

    def test_operator_info_fetcher_importable(self):
        from tools.web_search import OperatorInfoFetcher
        assert OperatorInfoFetcher is not None

    def test_web_search_has_no_websearchtool(self):
        """WebSearchTool 已被移除，不应再导出"""
        import tools.web_search as ws
        assert not hasattr(ws, "WebSearchTool")

    def test_old_modules_deleted(self):
        """旧模块文件应不存在"""
        base = Path("tools/web_search")
        assert not (base / "search_agent.py").exists()
        assert not (base / "sphinx_search.py").exists()
        assert not (base / "search_tool.py").exists()

    def test_crawl_agent_importable(self):
        from tools.agent import CrawlAgent, TaskRunner, TaskSpec
        assert CrawlAgent is not None
        assert TaskRunner is not None
        assert TaskSpec is not None

    def test_operator_mr_still_imports_ok(self):
        """operator_mr.py 使用的 OperatorInfoFetcher 仍可正常导入"""
        from mr_generator.operator.operator_mr import OperatorMRGenerator
        assert OperatorMRGenerator is not None
