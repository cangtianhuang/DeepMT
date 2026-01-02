"""网络搜索工具：从PyTorch文档、GitHub、网络搜索等获取算子信息"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import requests

from core.config_loader import get_config_value, get_web_search_config
from core.logger import get_logger
from tools.web_search.search_agent import SearchAgent


@dataclass
class SearchResult:
    """搜索结果数据结构"""

    title: str
    url: str
    snippet: str
    source: str  # docs, github, web_search
    relevance_score: float = 0.0


class WebSearchTool:
    """
    网络搜索工具：从多个源搜索算子信息（单例模式）

    支持：
    - 框架官方文档（使用智能搜索，支持 PyTorch/TensorFlow/PaddlePaddle 等）
    - GitHub 仓库
    - 网络搜索（百度搜索 API）
    """

    _instance: Optional["WebSearchTool"] = None

    def __new__(cls) -> "WebSearchTool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self) -> None:
        """初始化实例属性（仅在首次创建时调用）"""
        self.logger = get_logger()
        self.timeout = get_config_value("web_search.timeout", 10)
        self.max_results = get_config_value("web_search.max_results", 5)

        self.framework_docs = {
            "pytorch": {
                "search_url": "https://docs.pytorch.org/docs/stable/search.html",
                "docs_base": "https://docs.pytorch.org/docs/stable/",
                "github_repo": "pytorch/pytorch",
            },
            "tensorflow": {
                "search_url": "https://www.tensorflow.org/api_docs/python/tf",
                "docs_base": "https://www.tensorflow.org/api_docs/python/",
                "github_repo": "tensorflow/tensorflow",
            },
            "paddlepaddle": {
                "search_url": "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html",
                "docs_base": "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/",
                "github_repo": "PaddlePaddle/Paddle",
            },
        }

        self.github_base = "https://api.github.com/search/code"
        self.baidu_search_url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        self.baidu_api_key = get_config_value("web_search.baidu_api_key") or ""
        self.github_token = get_config_value("web_search.github_token") or ""
        self.custom_sites = get_config_value("web_search.custom_sites", [])
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        self.search_agent = SearchAgent()

    def search_operator(
        self,
        operator_name: str,
        framework: str = "pytorch",
        sources: Optional[Dict[str, bool]] = None,
    ) -> List[SearchResult]:
        """
        搜索算子信息

        Args:
            operator_name: 算子名称（如 "relu", "ReLU"）
            framework: 框架名称（默认pytorch，支持pytorch/tensorflow/paddlepaddle）
            sources: 搜索源开关字典（如果为None则使用配置中的源）
                    格式: {"docs": True, "github": True, "web_search": True}

        Returns:
            搜索结果列表
        """
        if sources is None:
            sources = get_config_value(
                "web_search.sources",
                {
                    "docs": True,
                    "github": True,
                    "web_search": True,
                },
            )

        if not isinstance(sources, dict):
            sources = {"docs": True, "github": True, "web_search": True}

        all_results = []
        normalized_name = self._normalize_operator_name(operator_name)

        self.logger.info(
            f"Searching for operator '{operator_name}' ({normalized_name}) "
            f"in framework '{framework}' from sources: {sources}"
        )

        if sources.get("docs", False):
            all_results.extend(self._search_docs(normalized_name, framework))

        if sources.get("github", False):
            all_results.extend(self._search_github(normalized_name, framework))

        if sources.get("web_search", False):
            all_results.extend(self._search_web(normalized_name, framework))

        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.info(f"Found {len(all_results)} search results")
        return all_results

    def _normalize_operator_name(self, name: str) -> str:
        """规范化算子名称，例如：ReLU -> relu, torch.nn.ReLU -> relu"""
        name = re.sub(r"^(torch\.(nn\.)?|tf\.|paddle\.)", "", name.lower())
        name = re.sub(r"(layer|module|function)$", "", name)
        return name.strip()

    def _search_docs(
        self, operator_name: str, framework: str = "pytorch"
    ) -> List[SearchResult]:
        """搜索框架官方文档（委托给 SearchAgent）"""
        framework_lower = framework.lower()
        if framework_lower not in self.framework_docs:
            self.logger.warning(
                f"Framework '{framework}' not supported. "
                f"Supported: {list(self.framework_docs.keys())}"
            )
            return []

        search_url = self.framework_docs[framework_lower]["search_url"]

        try:
            return self.search_agent.search_docs(
                query=operator_name,
                search_url=search_url,
                framework=framework,
                max_results=self.max_results,
            )
        except ValueError:
            raise
        except Exception as e:
            self.logger.warning(f"{framework.capitalize()} docs search failed: {e}")
            return []

    def _search_github(self, operator_name: str, framework: str) -> List[SearchResult]:
        """搜索GitHub仓库"""
        framework_lower = framework.lower()
        if framework_lower not in self.framework_docs:
            return []

        github_repo = self.framework_docs[framework_lower]["github_repo"]
        search_paths = {
            "pytorch": "torch/nn",
            "tensorflow": "tensorflow/python",
            "paddlepaddle": "paddle/fluid/operators",
        }
        search_path = search_paths.get(framework_lower, "")

        try:
            query = (
                f"{operator_name} repo:{github_repo} path:{search_path}"
                if search_path
                else f"{operator_name} repo:{github_repo}"
            )
            headers = self.headers.copy()
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            response = requests.get(
                self.github_base,
                params={"q": query, "per_page": self.max_results},
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                results = [
                    SearchResult(
                        title=f"{item.get('name', '')} - {item.get('path', '')}",
                        url=item.get("html_url", ""),
                        snippet=item.get("text_matches", [{}])[0].get("fragment", ""),
                        source="github",
                        relevance_score=0.8,
                    )
                    for item in response.json().get("items", [])[: self.max_results]
                ]
                return results
            elif response.status_code == 403:
                self.logger.warning(
                    "GitHub API rate limit exceeded. Consider setting web_search.github_token in config.yaml"
                )
        except Exception as e:
            self.logger.warning(f"GitHub search failed: {e}")

        return []

    def _search_web(self, operator_name: str, framework: str) -> List[SearchResult]:
        """使用百度搜索API进行网络搜索"""
        if not self.baidu_api_key:
            self.logger.warning(
                "Baidu search API key not configured. Set baidu_search.api_key in config.yaml"
            )
            return []

        try:
            query = f"{framework} {operator_name}"
            request_body = {
                "messages": [{"content": query, "role": "user"}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": self.max_results}],
            }

            if self.custom_sites:
                request_body["search_filter"] = {"match": {"site": self.custom_sites}}

            response = requests.post(
                self.baidu_search_url,
                headers={
                    "Authorization": f"Bearer {self.baidu_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                if "code" in data and data["code"]:
                    self.logger.warning(
                        f"Baidu search API error: {data.get('message', 'Unknown error')}"
                    )
                    return []

                return [
                    SearchResult(
                        title=ref.get("title", ""),
                        url=ref.get("url", ""),
                        snippet=ref.get("content", ""),
                        source="web_search",
                        relevance_score=ref.get("rerank_score", 0.7),
                    )
                    for ref in data.get("references", [])[: self.max_results]
                    if ref.get("type") == "web"
                ]

            self.logger.warning(
                f"Baidu search API request failed with status {response.status_code}"
            )
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")

        return []

    def _extract_title(self, html: str) -> Optional[str]:
        """从HTML提取标题"""
        if match := re.search(r"<title>(.*?)</title>", html, re.IGNORECASE):
            return match.group(1)
        return None

    def _extract_snippet(
        self, html: str, keyword: str, max_length: int = 500
    ) -> Optional[str]:
        """从HTML提取包含关键词的片段"""
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        keyword_lower = keyword.lower()
        text_lower = text.lower()

        if (idx := text_lower.find(keyword_lower)) != -1:
            start = max(0, idx - 100)
            end = min(len(text), idx + max_length)
            return text[start:end].strip()

        return text[:max_length].strip() if text else None

    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        return re.sub(r"<[^>]+>", "", text)
