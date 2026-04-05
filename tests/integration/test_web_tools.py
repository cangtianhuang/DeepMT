"""Web工具集成测试：OperatorInfoFetcher、文档获取、版本查询"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from unittest.mock import MagicMock, patch

import pytest

from deepmt.tools.web_search.search_agent import SearchAgent
from deepmt.tools.web_search.operator_fetcher import OperatorInfoFetcher


@pytest.fixture(scope="module")
def agent():
    return SearchAgent()


MATMUL_STUB = """
<html><body><article id='pytorch-article'>
  <h1>torch.matmul</h1>
  <p>Matrix product of two tensors.</p>
  <dl><dt>Parameters</dt><dd>input (Tensor)</dd><dd>other (Tensor)</dd></dl>
</article></body></html>
"""

API_LIST_STUB = """
<html><body><article id='pytorch-article'>
  <a class='reference internal' href='https://docs.pytorch.org/docs/stable/torch.html'>torch</a>
  <a class='reference internal' href='https://docs.pytorch.org/docs/stable/nn.html'>torch.nn</a>
  <a class='reference internal' href='https://docs.pytorch.org/docs/stable/functional.html'>torch.nn.functional</a>
</article></body></html>
"""


def _mock_http(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status.return_value = None
    return resp


def _pypi_response(version: str, releases: dict | None = None) -> dict:
    return {
        "info": {"version": version},
        "releases": releases or {version: [{"upload_time": "2025-01-01T00:00:00"}]},
    }


def _mock_json(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# OperatorInfoFetcher 接口
# ---------------------------------------------------------------------------

class TestOperatorInfoFetcher:

    def _disabled_fetcher(self):
        with patch("deepmt.tools.web_search.operator_fetcher.get_config_value", return_value=False):
            return OperatorInfoFetcher()

    def test_returns_dict_with_required_keys(self):
        fetcher = self._disabled_fetcher()
        result = fetcher.fetch_operator_info("relu", "pytorch")
        assert isinstance(result, dict)
        assert "name" in result
        assert "doc" in result
        assert "source_urls" in result

    def test_preserves_operator_name(self):
        fetcher = self._disabled_fetcher()
        result = fetcher.fetch_operator_info("conv2d", "pytorch")
        assert result["name"] == "conv2d"

    def test_source_urls_is_list(self):
        fetcher = self._disabled_fetcher()
        result = fetcher.fetch_operator_info("relu", "pytorch")
        assert isinstance(result["source_urls"], list)

    def test_get_operator_doc_returns_str_or_none(self):
        fetcher = self._disabled_fetcher()
        result = fetcher.get_operator_doc("relu", "pytorch")
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# 文档获取（fetch_operator_doc_by_url）
# ---------------------------------------------------------------------------

class TestFetchOperatorDocByUrl:

    def test_returns_none_on_network_error(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get") as mock:
            mock.side_effect = Exception("network error")
            assert agent.fetch_operator_doc_by_url("https://example.com/doc") is None

    def test_returns_string_on_success(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(MATMUL_STUB)):
            result = agent.fetch_operator_doc_by_url("https://docs.pytorch.org/stable/torch.matmul.html")
        assert isinstance(result, str) and len(result) > 0

    def test_no_html_tags_in_output(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(MATMUL_STUB)):
            result = agent.fetch_operator_doc_by_url("https://example.com/doc")
        assert "<html" not in result and "<body" not in result

    def test_contains_operator_content(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(MATMUL_STUB)):
            result = agent.fetch_operator_doc_by_url("https://docs.pytorch.org/stable/torch.matmul.html")
        assert "matmul" in result.lower()


# ---------------------------------------------------------------------------
# API 列表获取（fetch_api_list）
# ---------------------------------------------------------------------------

class TestFetchApiList:

    def test_returns_empty_on_network_error(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get") as mock:
            mock.side_effect = Exception("network error")
            assert agent.fetch_api_list("https://example.com/api", use_cache=False) == []

    def test_returns_list_on_success(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(API_LIST_STUB)):
            result = agent.fetch_api_list("https://docs.pytorch.org/stable/api.html", use_cache=False)
        assert isinstance(result, list) and len(result) == 3

    def test_entries_have_name_and_url(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(API_LIST_STUB)):
            result = agent.fetch_api_list("https://docs.pytorch.org/stable/api.html", use_cache=False)
        for entry in result:
            assert "name" in entry and "url" in entry

    def test_cache_hit_skips_network(self, agent):
        url = "https://docs.pytorch.org/stable/api-cache-test.html"
        with patch("deepmt.tools.web_search.search_agent.requests.get", return_value=_mock_http(API_LIST_STUB)):
            agent.fetch_api_list(url, use_cache=False)
        with patch("deepmt.tools.web_search.search_agent.requests.get") as mock:
            result = agent.fetch_api_list(url, use_cache=True)
        mock.assert_not_called()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 版本查询（get_latest_stable_version + fetch_framework_versions）
# ---------------------------------------------------------------------------

class TestVersionFetching:

    def test_latest_version_returns_string(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get",
                   return_value=_mock_json(_pypi_response("2.6.0"))):
            result = agent.get_latest_stable_version("pytorch")
        assert isinstance(result, str) and result == "2.6.0"

    def test_unknown_framework_returns_none(self, agent):
        assert agent.get_latest_stable_version("unknown_xyz") is None

    def test_network_error_returns_none(self, agent):
        with patch("deepmt.tools.web_search.search_agent.requests.get") as mock:
            mock.side_effect = Exception("timeout")
            assert agent.get_latest_stable_version("pytorch") is None

    def test_fetch_versions_returns_list(self, agent):
        releases = {
            "2.6.0": [{"upload_time": "2025-01-01T00:00:00"}],
            "2.5.0": [{"upload_time": "2024-09-01T00:00:00"}],
        }
        with patch("deepmt.tools.web_search.search_agent.requests.get",
                   return_value=_mock_json(_pypi_response("2.6.0", releases))):
            result = agent.fetch_framework_versions("pytorch")
        assert isinstance(result, list) and len(result) == 2
        assert result[0]["version"] == "2.6.0"

    def test_fetch_versions_skips_empty_releases(self, agent):
        releases = {
            "2.6.0": [{"upload_time": "2025-01-01T00:00:00"}],
            "2.6.0rc1": [],
        }
        with patch("deepmt.tools.web_search.search_agent.requests.get",
                   return_value=_mock_json(_pypi_response("2.6.0", releases))):
            result = agent.fetch_framework_versions("pytorch")
        assert len(result) == 1

    def test_fetch_versions_unknown_framework(self, agent):
        assert agent.fetch_framework_versions("unknown_xyz") == []
