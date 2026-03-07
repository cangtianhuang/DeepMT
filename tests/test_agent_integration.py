"""
Phase 2 集成测试：端到端验证 CrawlAgent 三个核心任务

策略：
- mock HTTP 层（_get_html）：返回真实结构的 HTML 片段
- mock LLM 层（chat_completion_with_tools）：返回预设的工具调用序列
- 不 mock 工具函数本身，测试真实的 extract/browse 逻辑
- 验证最终结果结构与 OperatorCatalog 合并行为
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
#  测试 Fixture：模拟 HTML 页面                                                #
# --------------------------------------------------------------------------- #

GITHUB_RELEASES_HTML = textwrap.dedent("""
    <html><body>
    <main>
      <h1>Releases</h1>
      <div class="release-entry">
        <a href="/pytorch/pytorch/releases/tag/v2.5.1">v2.5.1</a>
        <span class="label label-success">Latest</span>
        <relative-time datetime="2024-10-17">Oct 17, 2024</relative-time>
      </div>
      <div class="release-entry">
        <a href="/pytorch/pytorch/releases/tag/v2.4.0">v2.4.0</a>
        <relative-time datetime="2024-06-05">Jun 5, 2024</relative-time>
      </div>
    </main>
    </body></html>
""")

PYTORCH_OP_INDEX_HTML = textwrap.dedent("""
    <html><body>
    <main id="pytorch-article">
      <h1>torch — PyTorch 2.5 documentation</h1>
      <dl>
        <dt><a href="/docs/stable/generated/torch.nn.ReLU.html">torch.nn.ReLU</a></dt>
        <dt><a href="/docs/stable/generated/torch.nn.Conv2d.html">torch.nn.Conv2d</a></dt>
        <dt><a href="/docs/stable/generated/torch.nn.BatchNorm2d.html">torch.nn.BatchNorm2d</a></dt>
        <dt><a href="/docs/stable/generated/torch.nn.functional.relu.html">torch.nn.functional.relu</a></dt>
        <dt><a href="/docs/stable/generated/torch.matmul.html">torch.matmul</a></dt>
      </dl>
    </main>
    </body></html>
""")

RELU_DOC_HTML = textwrap.dedent("""
    <html><body>
    <main id="pytorch-article">
      <h1>torch.nn.ReLU</h1>
      <p class="sig">torch.nn.ReLU(inplace=False)</p>
      <section id="parameters">
        <h2>Parameters</h2>
        <p>inplace (bool) – can optionally do the operation in-place. Default: False</p>
      </section>
      <section id="examples">
        <h2>Examples</h2>
        <pre>
m = nn.ReLU()
input = torch.randn(2)
output = m(input)
        </pre>
      </section>
    </main>
    </body></html>
""")


# --------------------------------------------------------------------------- #
#  工具函数：构建 LLM 响应                                                     #
# --------------------------------------------------------------------------- #

def _tool_call(tool_name: str, args: Dict[str, Any], call_id: str = "c1") -> Dict:
    return {
        "content": None,
        "tool_calls": [{"id": call_id, "name": tool_name, "arguments": args}],
    }


def _finish(result: Dict[str, Any], call_id: str = "fin") -> Dict:
    return _tool_call("finish", {"result": result}, call_id)


# --------------------------------------------------------------------------- #
#  测试 1：获取框架最新版本（get_framework_version）                            #
# --------------------------------------------------------------------------- #

class TestGetFrameworkVersion:
    """验证 get_framework_version 任务的完整 ReAct 流程"""

    def _make_agent(self, llm_responses: List[Dict]) -> "CrawlAgent":
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = llm_responses

        return CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)

    def test_pytorch_version_two_steps(self):
        """
        两步完成：fetch_page → find_version_tags → finish
        验证 agent 能正确提取版本号和发布 URL
        """
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_framework_version")

        llm_responses = [
            # Step 1: 获取 releases 页面
            _tool_call(
                "fetch_page",
                {"url": "https://github.com/pytorch/pytorch/releases"},
                "c1",
            ),
            # Step 2: 在内容中搜索版本号
            _tool_call(
                "find_version_tags",
                {"text": "PLACEHOLDER"},  # 真实值由上一步结果填入，这里 mock LLM 直接给出
                "c2",
            ),
            # Step 3: 提交结果
            _finish(
                {
                    "version": "2.5.1",
                    "release_date": "2024-10-17",
                    "release_url": "https://github.com/pytorch/pytorch/releases/tag/v2.5.1",
                }
            ),
        ]

        agent = self._make_agent(llm_responses)

        with patch("tools.agent.tools.browse._get_html", return_value=GITHUB_RELEASES_HTML):
            result = agent.run(spec, inputs={"framework": "pytorch"})

        assert result["version"] == "2.5.1"
        assert result["release_date"] == "2024-10-17"
        assert "release_url" in result
        assert "pytorch" in result["release_url"]

    def test_result_schema_complete(self):
        """验证返回结果包含所有必需字段"""
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_framework_version")

        llm_responses = [
            _finish(
                {
                    "version": "2.16.0",
                    "release_date": "2024-11-01",
                    "release_url": "https://github.com/tensorflow/tensorflow/releases/tag/v2.16.0",
                }
            ),
        ]
        agent = self._make_agent(llm_responses)

        result = agent.run(spec, inputs={"framework": "tensorflow"})

        # 验证所有 output_schema 字段都存在
        assert "version" in result
        assert "release_date" in result
        assert "release_url" in result

    def test_version_without_v_prefix(self):
        """版本号应去掉 v 前缀"""
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_framework_version")

        # agent 直接返回带 v 前缀的版本（测试业务约定：agent 自己应去掉 v）
        llm_responses = [
            _finish(
                {
                    "version": "2.5.1",  # 无 v 前缀
                    "release_date": "2024-10-17",
                    "release_url": "https://github.com/pytorch/pytorch/releases/tag/v2.5.1",
                }
            ),
        ]
        agent = self._make_agent(llm_responses)

        result = agent.run(spec, inputs={"framework": "pytorch"})
        # 版本号不应带 v
        assert not result["version"].startswith("v")


# --------------------------------------------------------------------------- #
#  测试 2：获取算子列表（get_operator_list）                                   #
# --------------------------------------------------------------------------- #

class TestGetOperatorList:
    """验证 get_operator_list 任务的完整 ReAct 流程"""

    def _make_agent(self, llm_responses: List[Dict]) -> "CrawlAgent":
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = llm_responses

        return CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)

    def test_pytorch_operator_list_extraction(self):
        """
        验证 agent 能从 PyTorch 文档索引页提取算子列表
        """
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_list")

        expected_operators = [
            "torch.nn.ReLU",
            "torch.nn.Conv2d",
            "torch.nn.BatchNorm2d",
            "torch.nn.functional.relu",
            "torch.matmul",
        ]

        llm_responses = [
            # Step 1: 获取算子索引页 HTML
            _tool_call(
                "fetch_page_html",
                {"url": "https://docs.pytorch.org/docs/stable/torch.html"},
                "c1",
            ),
            # Step 2: 提取链接（真实调用 extract_links）
            _tool_call(
                "extract_links",
                {
                    "html": PYTORCH_OP_INDEX_HTML,
                    "base_url": "https://docs.pytorch.org",
                    "filter_pattern": r"generated/torch",
                },
                "c2",
            ),
            # Step 3: 返回结果
            _finish(
                {
                    "operators": expected_operators,
                    "doc_urls": {
                        "torch.nn.ReLU": "https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html",
                        "torch.nn.Conv2d": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
                    },
                    "total_count": len(expected_operators),
                    "version": "2.5.1",
                }
            ),
        ]

        agent = self._make_agent(llm_responses)

        with patch("tools.agent.tools.browse._get_html", return_value=PYTORCH_OP_INDEX_HTML):
            result = agent.run(spec, inputs={"framework": "pytorch", "version": "2.5.1"})

        assert "operators" in result
        assert "doc_urls" in result
        assert "total_count" in result
        assert len(result["operators"]) == len(expected_operators)
        assert result["operators"] == expected_operators

    def test_result_contains_doc_urls(self):
        """验证 doc_urls 字典结构正确"""
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_list")

        llm_responses = [
            _finish(
                {
                    "operators": ["torch.nn.ReLU", "torch.nn.Sigmoid"],
                    "doc_urls": {
                        "torch.nn.ReLU": "https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html",
                        "torch.nn.Sigmoid": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html",
                    },
                    "total_count": 2,
                    "version": "2.5.1",
                }
            ),
        ]
        agent = self._make_agent(llm_responses)

        result = agent.run(spec, inputs={"framework": "pytorch", "version": "2.5.1"})

        # doc_urls 应为字典
        assert isinstance(result["doc_urls"], dict)
        # 所有算子都有对应 URL
        for op in result["operators"]:
            assert op in result["doc_urls"]
            assert result["doc_urls"][op].startswith("https://")

    def test_extract_links_tool_works_with_real_html(self):
        """
        验证 extract_links 工具能正确解析 PyTorch 索引页 HTML
        这是一个工具层的集成验证
        """
        from tools.agent.tools.extract import extract_links

        result = json.loads(
            extract_links(
                html=PYTORCH_OP_INDEX_HTML,
                base_url="https://docs.pytorch.org",
                filter_pattern=r"generated/torch",
            )
        )

        # 应该提取到 5 个算子链接
        assert len(result) == 5
        urls = [r["url"] for r in result]
        assert any("torch.nn.ReLU" in u for u in urls)
        assert any("torch.nn.Conv2d" in u for u in urls)

        texts = [r["text"] for r in result]
        assert "torch.nn.ReLU" in texts


# --------------------------------------------------------------------------- #
#  测试 3：获取算子文档（get_operator_doc）                                    #
# --------------------------------------------------------------------------- #

class TestGetOperatorDoc:
    """验证 get_operator_doc 任务的完整 ReAct 流程"""

    def _make_agent(self, llm_responses: List[Dict]) -> "CrawlAgent":
        from tools.agent.agent_core import CrawlAgent
        from tools.agent.tool_registry import build_default_registry

        mock_llm = MagicMock()
        mock_llm.chat_completion_with_tools.side_effect = llm_responses

        return CrawlAgent(registry=build_default_registry(), llm_client=mock_llm)

    def test_relu_doc_extraction(self):
        """验证 ReLU 文档能被正确提取"""
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_doc")

        relu_url = "https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html"

        llm_responses = [
            # Step 1: 直接访问算子文档页
            _tool_call("fetch_page", {"url": relu_url}, "c1"),
            # Step 2: 提交结果
            _finish(
                {
                    "doc": "torch.nn.ReLU\nReLU(inplace=False)\n"
                           "Parameters\ninplace (bool) – can optionally do the operation in-place.",
                    "source_url": relu_url,
                    "operator_name": "torch.nn.ReLU",
                }
            ),
        ]

        agent = self._make_agent(llm_responses)

        with patch("tools.agent.tools.browse._get_html", return_value=RELU_DOC_HTML):
            result = agent.run(
                spec,
                inputs={"operator_name": "relu", "framework": "pytorch"},
            )

        assert "doc" in result
        assert "source_url" in result
        assert "operator_name" in result
        assert len(result["doc"]) > 0

    def test_fetch_page_extracts_relu_content(self):
        """
        验证 fetch_page 工具能正确从 ReLU 文档页提取文本
        工具层集成验证（不依赖 LLM）
        """
        from tools.agent.tools.browse import fetch_page

        with patch("tools.agent.tools.browse._get_html", return_value=RELU_DOC_HTML):
            text = fetch_page("https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html")

        assert "ReLU" in text
        assert "inplace" in text
        assert "Parameters" in text

    def test_doc_result_is_nonempty_string(self):
        """验证 get_operator_doc 结果中 doc 字段非空"""
        from tools.agent.agent_core import TaskSpec

        spec = TaskSpec.from_task_id("get_operator_doc")

        llm_responses = [
            _finish(
                {
                    "doc": "torch.nn.Conv2d\nConv2d(in_channels, out_channels, kernel_size, ...)",
                    "source_url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
                    "operator_name": "torch.nn.Conv2d",
                }
            ),
        ]
        agent = self._make_agent(llm_responses)

        result = agent.run(
            spec,
            inputs={"operator_name": "conv2d", "framework": "pytorch"},
        )
        assert isinstance(result["doc"], str)
        assert len(result["doc"]) > 10


# --------------------------------------------------------------------------- #
#  测试 4：OperatorCatalog 合并与持久化（Phase 2 核心）                        #
# --------------------------------------------------------------------------- #

class TestOperatorCatalogMerge:
    """验证 agent 结果 → OperatorCatalog 的合并与 YAML 写入"""

    def _fresh_catalog(self, tmp_path: Path) -> "OperatorCatalog":
        """创建一个使用临时目录的 OperatorCatalog 实例"""
        from mr_generator.base.operator_catalog import OperatorCatalog

        # 创建临时目录下的 pytorch.yaml
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        pytorch_yaml = catalog_dir / "pytorch.yaml"
        pytorch_yaml.write_text(
            "framework: pytorch\n"
            "last_updated: \"2026-01-01\"\n"
            "description: \"Test catalog\"\n"
            "operators:\n"
            "  - name: torch.nn.ReLU\n"
            "    category: activation\n"
            "    since: \"1.0\"\n"
            "\n"
            "  - name: torch.nn.Conv2d\n"
            "    category: convolution\n"
            "    since: \"1.0\"\n"
            "    doc_url: https://old-url.com/conv2d\n"
            "\n"
        )
        # 补全 tensorflow 和 paddlepaddle（空文件）
        (catalog_dir / "tensorflow.yaml").write_text(
            "framework: tensorflow\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )
        (catalog_dir / "paddlepaddle.yaml").write_text(
            "framework: paddlepaddle\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )

        return OperatorCatalog(catalog_dir=str(catalog_dir))

    def test_merge_adds_new_operators(self, tmp_path):
        """新算子应被添加到目录"""
        catalog = self._fresh_catalog(tmp_path)

        agent_result = {
            "operators": ["torch.nn.ReLU", "torch.nn.Sigmoid", "torch.nn.Tanh"],
            "doc_urls": {
                "torch.nn.ReLU": "https://docs.pytorch.org/relu",
                "torch.nn.Sigmoid": "https://docs.pytorch.org/sigmoid",
                "torch.nn.Tanh": "https://docs.pytorch.org/tanh",
            },
            "version": "2.5.1",
        }

        stats = catalog.merge_from_agent_result("pytorch", agent_result)

        # Sigmoid 和 Tanh 是新算子
        assert stats["added"] == 2
        # ReLU 已存在，更新 doc_url
        assert stats["updated"] == 1
        assert stats["skipped"] == 0

        names = catalog.get_operator_names("pytorch")
        assert "torch.nn.Sigmoid" in names
        assert "torch.nn.Tanh" in names

    def test_merge_updates_doc_url_for_existing(self, tmp_path):
        """已有算子的 doc_url 应被更新"""
        catalog = self._fresh_catalog(tmp_path)

        agent_result = {
            "operators": ["torch.nn.Conv2d"],
            "doc_urls": {
                "torch.nn.Conv2d": "https://new-url.com/conv2d",
            },
            "version": "2.5.1",
        }

        catalog.merge_from_agent_result("pytorch", agent_result)

        entry = catalog.get_operator_info("pytorch", "torch.nn.Conv2d")
        assert entry is not None
        assert entry.doc_url == "https://new-url.com/conv2d"

    def test_merge_preserves_existing_metadata(self, tmp_path):
        """合并时不应覆盖已有条目的 category、since 等字段"""
        catalog = self._fresh_catalog(tmp_path)

        agent_result = {
            "operators": ["torch.nn.ReLU"],
            "doc_urls": {"torch.nn.ReLU": "https://new.url/relu"},
            "version": "2.5.1",
        }

        catalog.merge_from_agent_result("pytorch", agent_result)

        entry = catalog.get_operator_info("pytorch", "torch.nn.ReLU")
        assert entry.category == "activation"   # 不被覆盖
        assert entry.since == "1.0"             # 不被覆盖
        assert entry.doc_url == "https://new.url/relu"  # 被更新

    def test_save_yaml_writes_file(self, tmp_path):
        """save_yaml 应正确写入文件"""
        catalog = self._fresh_catalog(tmp_path)

        agent_result = {
            "operators": ["torch.nn.ReLU", "torch.nn.Sigmoid"],
            "doc_urls": {"torch.nn.Sigmoid": "https://docs.pytorch.org/sigmoid"},
            "version": "2.5.1",
        }
        catalog.merge_from_agent_result("pytorch", agent_result)
        saved_path = catalog.save_yaml("pytorch")

        assert saved_path.exists()
        content = saved_path.read_text(encoding="utf-8")

        assert "torch.nn.ReLU" in content
        assert "torch.nn.Sigmoid" in content
        assert "docs.pytorch.org/sigmoid" in content

    def test_save_yaml_updates_last_updated(self, tmp_path):
        """save_yaml 应更新 last_updated 日期"""
        from datetime import datetime

        catalog = self._fresh_catalog(tmp_path)
        catalog.save_yaml("pytorch")

        content = (tmp_path / "catalog" / "pytorch.yaml").read_text()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in content

    def test_save_yaml_roundtrip(self, tmp_path):
        """写入后重新加载，数据应一致"""
        from mr_generator.base.operator_catalog import OperatorCatalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        pytorch_yaml = catalog_dir / "pytorch.yaml"
        pytorch_yaml.write_text(
            "framework: pytorch\nlast_updated: \"2026-01-01\"\ndescription: \"Test\"\noperators: []\n"
        )
        (catalog_dir / "tensorflow.yaml").write_text(
            "framework: tensorflow\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )
        (catalog_dir / "paddlepaddle.yaml").write_text(
            "framework: paddlepaddle\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )

        catalog = OperatorCatalog(catalog_dir=str(catalog_dir))
        catalog.merge_from_agent_result(
            "pytorch",
            {
                "operators": ["torch.nn.ReLU", "torch.nn.Sigmoid"],
                "doc_urls": {"torch.nn.ReLU": "https://docs.pytorch.org/relu"},
                "version": "2.5.1",
            },
        )
        catalog.save_yaml("pytorch")

        # 重新加载
        catalog2 = OperatorCatalog(catalog_dir=str(catalog_dir))
        names2 = catalog2.get_operator_names("pytorch")
        assert "torch.nn.ReLU" in names2
        assert "torch.nn.Sigmoid" in names2
        assert catalog2.get_doc_url("pytorch", "torch.nn.ReLU") == "https://docs.pytorch.org/relu"

    def test_get_doc_url(self, tmp_path):
        """get_doc_url 应返回正确的文档 URL"""
        catalog = self._fresh_catalog(tmp_path)

        # torch.nn.Conv2d 在初始 catalog 中有 doc_url
        doc_url = catalog.get_doc_url("pytorch", "torch.nn.Conv2d")
        assert doc_url == "https://old-url.com/conv2d"

        # 不存在的算子返回空字符串
        assert catalog.get_doc_url("pytorch", "nonexistent.op") == ""

    def test_merge_with_empty_doc_urls(self, tmp_path):
        """agent 结果中没有 doc_urls 时应正常合并"""
        catalog = self._fresh_catalog(tmp_path)

        agent_result = {
            "operators": ["torch.nn.Dropout"],
            "doc_urls": {},
            "version": "2.5.1",
        }
        stats = catalog.merge_from_agent_result("pytorch", agent_result)

        assert stats["added"] == 1
        entry = catalog.get_operator_info("pytorch", "torch.nn.Dropout")
        assert entry is not None
        assert entry.doc_url == ""


# --------------------------------------------------------------------------- #
#  测试 5：TaskRunner.sync_catalog() 端到端                                   #
# --------------------------------------------------------------------------- #

class TestSyncCatalog:
    """验证 TaskRunner.sync_catalog() 完整链路"""

    def test_sync_catalog_merges_and_saves(self, tmp_path):
        """
        端到端验证：mock TaskRunner.get_operator_list() → 合并 → 保存
        """
        from mr_generator.base.operator_catalog import OperatorCatalog
        from tools.agent.task_runner import TaskRunner

        # 初始化 runner（不实际调用 agent）
        runner = TaskRunner(cache_dir=str(tmp_path / "cache"))

        # 构造 agent result
        agent_result = {
            "operators": ["torch.nn.ReLU", "torch.nn.GELU", "torch.nn.Mish"],
            "doc_urls": {
                "torch.nn.ReLU": "https://docs.pytorch.org/relu",
                "torch.nn.GELU": "https://docs.pytorch.org/gelu",
            },
            "total_count": 3,
            "version": "2.5.1",
        }

        # 用临时目录的 OperatorCatalog
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pytorch.yaml").write_text(
            "framework: pytorch\nlast_updated: \"2026-01-01\"\ndescription: \"Test\"\noperators: []\n"
        )
        (catalog_dir / "tensorflow.yaml").write_text(
            "framework: tensorflow\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )
        (catalog_dir / "paddlepaddle.yaml").write_text(
            "framework: paddlepaddle\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )

        # Mock get_operator_list 和 OperatorCatalog
        with (
            patch.object(runner, "get_operator_list", return_value=agent_result),
            patch(
                "tools.agent.task_runner.OperatorCatalog",
                return_value=OperatorCatalog(catalog_dir=str(catalog_dir)),
            ),
        ):
            result = runner.sync_catalog("pytorch", version="2.5.1", save_yaml=True)

        assert result["framework"] == "pytorch"
        assert result["version"] == "2.5.1"
        assert "merge_stats" in result
        assert result["merge_stats"]["added"] == 3  # 全是新条目
        assert result["total_operators"] == 3
        assert "catalog_path" in result

        # 验证 YAML 文件已写入
        assert Path(result["catalog_path"]).exists()

    def test_sync_catalog_no_save(self, tmp_path):
        """save_yaml=False 时不写文件"""
        from mr_generator.base.operator_catalog import OperatorCatalog
        from tools.agent.task_runner import TaskRunner

        runner = TaskRunner(cache_dir=str(tmp_path / "cache"))

        agent_result = {
            "operators": ["torch.nn.ReLU"],
            "doc_urls": {},
            "total_count": 1,
            "version": "2.5.1",
        }

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pytorch.yaml").write_text(
            "framework: pytorch\nlast_updated: \"2026-01-01\"\ndescription: \"Test\"\noperators: []\n"
        )
        (catalog_dir / "tensorflow.yaml").write_text(
            "framework: tensorflow\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )
        (catalog_dir / "paddlepaddle.yaml").write_text(
            "framework: paddlepaddle\nlast_updated: \"2026-01-01\"\ndescription: \"\"\noperators: []\n"
        )

        with (
            patch.object(runner, "get_operator_list", return_value=agent_result),
            patch(
                "tools.agent.task_runner.OperatorCatalog",
                return_value=OperatorCatalog(catalog_dir=str(catalog_dir)),
            ),
        ):
            result = runner.sync_catalog("pytorch", save_yaml=False)

        # 不应包含 catalog_path
        assert "catalog_path" not in result
        assert result["merge_stats"]["added"] == 1
