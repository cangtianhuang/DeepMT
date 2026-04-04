"""搜索工具单元测试：HTML解析 + Sphinx本地搜索"""

import os
import time
import pytest

from deepmt.tools.web_search.search_agent import SearchAgent
from deepmt.tools.web_search.sphinx_search import SphinxSearchIndex, load_json_cache, save_json_cache


# ---------------------------------------------------------------------------
# HTML 解析
# ---------------------------------------------------------------------------

PYTORCH_DOC_HTML = """
<html><body>
<article id="pytorch-article">
  <h1>torch.relu</h1>
  <p>Applies the rectified linear unit function.</p>
  <script>var x = 1;</script>
</article>
</body></html>
"""

API_LIST_HTML = """
<html><body>
<div id="pytorch-article">
  <a class="reference internal" href="https://docs.pytorch.org/stable/torch.html">torch</a>
  <a class="reference internal" href="/docs/stable/nn.html">torch.nn</a>
</div>
</body></html>
"""


@pytest.fixture(scope="module")
def agent():
    return SearchAgent()


class TestParseOperatorDoc:

    def test_empty_returns_empty(self, agent):
        assert agent.parse_operator_doc("") == ""

    def test_pytorch_article_id(self, agent):
        result = agent.parse_operator_doc(PYTORCH_DOC_HTML)
        assert "relu" in result.lower()

    def test_no_html_tags_in_output(self, agent):
        result = agent.parse_operator_doc(PYTORCH_DOC_HTML)
        assert "<" not in result and ">" not in result

    def test_strips_script_content(self, agent):
        result = agent.parse_operator_doc(PYTORCH_DOC_HTML)
        assert "var x" not in result

    def test_fallback_to_main(self, agent):
        html = "<html><body><main><p>Main content here</p></main></body></html>"
        result = agent.parse_operator_doc(html)
        assert "Main content here" in result

    def test_fallback_to_article(self, agent):
        html = "<html><body><article><p>Article content</p></article></body></html>"
        result = agent.parse_operator_doc(html)
        assert "Article content" in result


class TestParseApiList:

    def test_empty_returns_empty(self, agent):
        assert agent.parse_api_list("", "https://base.com") == []

    def test_basic_extraction(self, agent):
        result = agent.parse_api_list(API_LIST_HTML, "https://docs.pytorch.org")
        assert len(result) == 2

    def test_entries_have_name_and_url(self, agent):
        result = agent.parse_api_list(API_LIST_HTML, "https://docs.pytorch.org")
        for entry in result:
            assert "name" in entry and "url" in entry

    def test_relative_url_resolved(self, agent):
        result = agent.parse_api_list(API_LIST_HTML, "https://docs.pytorch.org")
        urls = [e["url"] for e in result]
        assert all(u.startswith("http") for u in urls)

    def test_skips_empty_href(self, agent):
        html = '<html><body><div id="pytorch-article"><a href="">empty</a></div></body></html>'
        result = agent.parse_api_list(html, "https://base.com")
        assert result == []


# ---------------------------------------------------------------------------
# Sphinx 本地搜索（使用 mock 索引数据，不联网）
# ---------------------------------------------------------------------------

MOCK_INDEX = {
    "docnames": ["torch.relu", "torch.leaky_relu", "torch.relu6", "torch.sigmoid"],
    "titles": ["torch.relu", "torch.leaky_relu", "torch.relu6", "torch.sigmoid"],
    "filenames": [
        "generated/torch.relu.rst",
        "generated/torch.leaky_relu.rst",
        "generated/torch.relu6.rst",
        "generated/torch.sigmoid.rst",
    ],
    "terms": {
        "relu": [0, 1, 2],
        "leaky": 1,
        "leaky_relu": 1,
        "relu6": 2,
        "sigmoid": 3,
        "activation": [0, 1, 2, 3],
    },
    "titleterms": {
        "relu": 0,
        "leaky_relu": 1,
        "relu6": 2,
        "sigmoid": 3,
    },
}


@pytest.fixture
def index():
    idx = SphinxSearchIndex("https://docs.pytorch.org/docs/stable/")
    idx._index = MOCK_INDEX
    idx._docnames = MOCK_INDEX["docnames"]
    idx._titles = dict(enumerate(MOCK_INDEX["titles"]))
    idx._filenames = dict(enumerate(MOCK_INDEX["filenames"]))
    idx._terms = MOCK_INDEX["terms"]
    idx._titleterms = MOCK_INDEX["titleterms"]
    idx._sorted_terms = sorted(idx._terms.keys())
    idx._sorted_titleterms = sorted(idx._titleterms.keys())
    idx._reversed_terms = sorted((t[::-1], t) for t in idx._terms.keys())
    idx._reversed_titleterms = sorted((t[::-1], t) for t in idx._titleterms.keys())
    return idx


class TestExactMatch:

    def test_exact_title_match(self, index):
        results = index.search("relu")
        assert len(results) >= 1
        assert results[0]["title"] == "torch.relu"

    def test_exact_title_match_score(self, index):
        results = index.search("relu")
        assert results[0]["relevance_score"] == 1.0

    def test_sigmoid_found(self, index):
        results = index.search("sigmoid")
        assert any(r["title"] == "torch.sigmoid" for r in results)

    def test_url_is_html(self, index):
        results = index.search("relu")
        assert results[0]["url"].endswith(".html")

    def test_url_contains_base(self, index):
        results = index.search("relu")
        assert results[0]["url"].startswith("https://docs.pytorch.org/docs/stable/")


class TestFuzzyMatch:

    def test_prefix_relu_finds_relu6(self, index):
        results = index.search("relu")
        titles = [r["title"] for r in results]
        assert "torch.relu6" in titles

    def test_suffix_relu_finds_leaky_relu(self, index):
        results = index.search("relu")
        titles = [r["title"] for r in results]
        assert "torch.leaky_relu" in titles

    def test_threshold_filters_low_scores(self, index):
        index.threshold = 0.9
        results = index.search("relu")
        for r in results:
            assert r["relevance_score"] >= 0.9
        index.threshold = 0.1  # restore

    def test_empty_query_returns_empty(self, index):
        assert index.search("") == []

    def test_no_match_returns_empty(self, index):
        assert index.search("zzznomatch") == []

    def test_max_results_respected(self, index):
        assert len(index.search("relu", max_results=1)) <= 1


# ---------------------------------------------------------------------------
# JSON 缓存工具
# ---------------------------------------------------------------------------

class TestJsonCache:

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "test.json"
        data = [{"name": "torch", "url": "https://example.com"}]
        save_json_cache(path, data)
        assert load_json_cache(path) == data

    def test_expired_cache_returns_none(self, tmp_path):
        path = tmp_path / "expired.json"
        save_json_cache(path, {"key": "value"})
        old_mtime = path.stat().st_mtime - 2 * 24 * 3600
        os.utime(path, (old_mtime, old_mtime))
        assert load_json_cache(path, ttl=24 * 3600) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert load_json_cache(tmp_path / "nonexistent.json") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json")
        assert load_json_cache(path) is None
