"""
网络搜索工具：从PyTorch文档、GitHub、网络搜索等获取算子信息
这是一个可复用的工具，用于后续开发
"""

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
        # 从配置加载器获取配置值
        self.timeout = get_config_value("web_search.timeout", 10)
        self.max_results = get_config_value("web_search.max_results", 5)

        # 框架文档 URL 映射
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

        # GitHub API 配置
        self.github_base = "https://api.github.com/search/code"

        # 百度搜索 API 配置
        self.baidu_search_url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        self.baidu_api_key = get_config_value("web_search.baidu_api_key") or ""

        # GitHub API token
        self.github_token = get_config_value("web_search.github_token") or ""

        # 用户指定的网站来源
        self.custom_sites = get_config_value("web_search.custom_sites", [])

        # 用户代理（避免被网站屏蔽）
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # 初始化搜索智能体
        from tools.llm.client import LLMClient

        llm_client = LLMClient()
        self.search_agent = SearchAgent(llm_client=llm_client)

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

        # 确保sources是字典
        if not isinstance(sources, dict):
            sources = {
                "docs": True,
                "github": True,
                "web_search": True,
            }

        all_results = []

        # 规范化算子名称
        normalized_name = self._normalize_operator_name(operator_name)

        self.logger.info(
            f"Searching for operator '{operator_name}' ({normalized_name}) "
            f"in framework '{framework}' from sources: {sources}"
        )

        # 从各个源搜索
        if sources.get("docs", False):
            results = self._search_docs(normalized_name, framework)
            all_results.extend(results)

        if sources.get("github", False):
            results = self._search_github(normalized_name, framework)
            all_results.extend(results)

        if sources.get("web_search", False):
            results = self._search_web(normalized_name, framework)
            all_results.extend(results)

        # 按相关性排序
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.info(f"Found {len(all_results)} search results")
        return all_results

    def _normalize_operator_name(self, name: str) -> str:
        """
        规范化算子名称

        例如：ReLU -> relu, torch.nn.ReLU -> relu
        """
        # 移除框架前缀
        name = re.sub(r"^(torch\.(nn\.)?|tf\.|paddle\.)", "", name.lower())
        # 移除常见后缀
        name = re.sub(r"(layer|module|function)$", "", name)
        return name.strip()

    def _search_docs(
        self, operator_name: str, framework: str = "pytorch"
    ) -> List[SearchResult]:
        """
        搜索框架官方文档（委托给 SearchAgent）

        Args:
            operator_name: 算子名称
            framework: 框架名称（pytorch/tensorflow/paddlepaddle）

        Returns:
            搜索结果列表
        """
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
        except ValueError as e:
            # 输入错误（如拼写错误），重新抛出以便上层处理
            self.logger.warning(f"Search input error: {e}")
            raise
        except Exception as e:
            self.logger.warning(f"{framework.capitalize()} docs search failed: {e}")
            return []

    def _search_github(self, operator_name: str, framework: str) -> List[SearchResult]:
        """
        搜索GitHub仓库

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            搜索结果列表
        """
        results = []

        # 获取框架的GitHub仓库信息
        framework_lower = framework.lower()
        if framework_lower not in self.framework_docs:
            return results

        github_repo = self.framework_docs[framework_lower]["github_repo"]

        # 根据框架构建搜索路径
        search_paths = {
            "pytorch": "torch/nn",
            "tensorflow": "tensorflow/python",
            "paddlepaddle": "paddle/fluid/operators",
        }
        search_path = search_paths.get(framework_lower, "")

        try:
            # 构建GitHub API搜索查询
            if search_path:
                query = f"{operator_name} repo:{github_repo} path:{search_path}"
            else:
                query = f"{operator_name} repo:{github_repo}"
            params = {"q": query, "per_page": self.max_results}

            # 添加认证头（如果有token）
            headers = self.headers.copy()
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            response = requests.get(
                self.github_base,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                for item in data.get("items", [])[: self.max_results]:
                    results.append(
                        SearchResult(
                            title=f"{item.get('name', '')} - {item.get('path', '')}",
                            url=item.get("html_url", ""),
                            snippet=item.get("text_matches", [{}])[0].get(
                                "fragment", ""
                            ),
                            source="github",
                            relevance_score=0.8,
                        )
                    )
            elif response.status_code == 403:
                self.logger.warning(
                    "GitHub API rate limit exceeded. Consider setting web_search.github_token in config.yaml"
                )
        except Exception as e:
            self.logger.warning(f"GitHub search failed: {e}")

        return results

    def _search_web(self, operator_name: str, framework: str) -> List[SearchResult]:
        """
        使用百度搜索API进行网络搜索

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            搜索结果列表
        """
        results = []

        if not self.baidu_api_key:
            self.logger.warning(
                "Baidu search API key not configured. Set baidu_search.api_key in config.yaml"
            )
            return results

        try:
            # 构建搜索查询
            query = f"{framework} {operator_name}"

            # 构建搜索过滤器（如果指定了自定义网站）
            search_filter = None
            if self.custom_sites:
                search_filter = {"match": {"site": self.custom_sites}}

            # 构建请求体
            request_body = {
                "messages": [{"content": query, "role": "user"}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": self.max_results}],
            }

            if search_filter:
                request_body["search_filter"] = search_filter

            # 发送请求
            headers = {
                "Authorization": f"Bearer {self.baidu_api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.baidu_search_url,
                headers=headers,
                json=request_body,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()

                # 检查错误
                if "code" in data and data["code"]:
                    self.logger.warning(
                        f"Baidu search API error: {data.get('message', 'Unknown error')}"
                    )
                    return results

                # 解析结果（只处理网页结果，图片OCR在search_agent中处理）
                references = data.get("references", [])
                for ref in references[: self.max_results]:
                    if ref.get("type") == "web":
                        results.append(
                            SearchResult(
                                title=ref.get("title", ""),
                                url=ref.get("url", ""),
                                snippet=ref.get("content", ""),
                                source="web_search",
                                relevance_score=ref.get("rerank_score", 0.7),
                            )
                        )
            else:
                self.logger.warning(
                    f"Baidu search API request failed with status {response.status_code}"
                )

        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")

        return results

    def _extract_title(self, html: str) -> Optional[str]:
        """从HTML提取标题"""
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_snippet(
        self, html: str, keyword: str, max_length: int = 500
    ) -> Optional[str]:
        """从HTML提取包含关键词的片段"""
        # 移除HTML标签
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)

        # 查找包含关键词的部分
        keyword_lower = keyword.lower()
        text_lower = text.lower()

        idx = text_lower.find(keyword_lower)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(text), idx + max_length)
            snippet = text[start:end].strip()
            return snippet

        # 如果没有找到关键词，返回开头部分
        return text[:max_length].strip() if text else None

    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        return re.sub(r"<[^>]+>", "", text)
