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
    - 框架官方文档（使用智能搜索，支持PyTorch/TensorFlow/PaddlePaddle等）
    - GitHub仓库
    - 网络搜索（百度搜索API）
    """

    _instance: Optional["WebSearchTool"] = None
    _initialized = False

    def __new__(
        cls,
        timeout: int = 10,
        max_results_per_source: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        创建或获取WebSearchTool实例（单例模式）

        Args:
            timeout: 请求超时时间（秒）
            max_results_per_source: 每个源的最大结果数
            config: 配置字典（包含API密钥等）

        Returns:
            WebSearchTool实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        timeout: int = 10,
        max_results_per_source: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化搜索工具

        Args:
            timeout: 请求超时时间（秒）
            max_results_per_source: 每个源的最大结果数
            config: 配置字典（包含API密钥等）
        """
        # 如果已经初始化过，跳过
        if WebSearchTool._initialized:
            return

        self.logger = get_logger()
        self.timeout = timeout
        self.max_results = max_results_per_source
        self.config = config or {}

        web_search_config = self.config.get("web_search", {})

        # 框架文档URL映射
        self.framework_docs = {
            "pytorch": {
                "search_url": "https://pytorch.org/docs/stable/search.html",
                "docs_base": "https://pytorch.org/docs/stable/",
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

        # GitHub API配置
        self.github_base = "https://api.github.com/search/code"

        # 百度搜索API配置
        self.baidu_search_url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        self.baidu_api_key = web_search_config.get("baidu_api_key") or ""

        # GitHub API token
        self.github_token = web_search_config.get("github_token") or ""

        # 用户指定的网站来源
        self.custom_sites = web_search_config.get("custom_sites", [])

        # 用户代理（避免被网站屏蔽）
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # 初始化搜索智能体
        from tools.llm.client import LLMClient

        llm_client = LLMClient()
        self.search_agent = SearchAgent(llm_client=llm_client)

        WebSearchTool._initialized = True

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
        web_search_config = self.config.get("web_search", {})
        if sources is None:
            sources = web_search_config.get(
                "sources",
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
        搜索框架官方文档（使用智能搜索代理）

        Args:
            operator_name: 算子名称
            framework: 框架名称（pytorch/tensorflow/paddlepaddle）

        Returns:
            搜索结果列表
        """
        results = []

        # 获取框架文档配置
        framework_lower = framework.lower()
        if framework_lower not in self.framework_docs:
            self.logger.warning(
                f"Framework '{framework}' not supported. Supported frameworks: {list(self.framework_docs.keys())}"
            )
            return results

        docs_config = self.framework_docs[framework_lower]
        search_url = docs_config["search_url"]
        docs_base = docs_config["docs_base"]

        try:
            # 使用智能搜索代理进行搜索
            search_results = self.search_agent.search_and_understand(
                query=operator_name,
                search_url=search_url,
                max_results=self.max_results,
            )

            # 对每个搜索结果，获取并理解内容
            for item in search_results:
                url = item.get("url", "")
                if not url:
                    continue

                # 获取并理解内容
                content = self.search_agent.fetch_and_understand_content(url)
                if content:
                    results.append(
                        SearchResult(
                            title=item.get(
                                "title",
                                f"{framework.capitalize()} {operator_name} Documentation",
                            ),
                            url=url,
                            snippet=content,
                            source="docs",
                            relevance_score=item.get("relevance_score", 0.9),
                        )
                    )

            # 如果没有找到结果，尝试直接访问常见路径（fallback）
            if not results:
                doc_urls = self._get_fallback_doc_urls(
                    operator_name, framework_lower, docs_base
                )

                for url in doc_urls:
                    try:
                        response = requests.get(
                            url, headers=self.headers, timeout=self.timeout
                        )
                        if response.status_code == 200:
                            content = response.text
                            title = self._extract_title(content)
                            snippet = self._extract_snippet(content, operator_name)

                            if snippet:
                                results.append(
                                    SearchResult(
                                        title=title
                                        or f"{framework.capitalize()} {operator_name} Documentation",
                                        url=url,
                                        snippet=snippet,
                                        source="docs",
                                        relevance_score=0.8,
                                    )
                                )
                                break
                    except Exception as e:
                        self.logger.debug(f"Failed to fetch {url}: {e}")
                        continue

        except Exception as e:
            self.logger.warning(f"{framework.capitalize()} docs search failed: {e}")

        return results[: self.max_results]

    def _get_fallback_doc_urls(
        self, operator_name: str, framework: str, docs_base: str
    ) -> List[str]:
        """
        获取框架文档的fallback URL列表

        Args:
            operator_name: 算子名称
            framework: 框架名称
            docs_base: 文档基础URL

        Returns:
            URL列表
        """
        if framework == "pytorch":
            return [
                f"{docs_base}generated/torch.nn.{operator_name.capitalize()}.html",
                f"{docs_base}nn.html#{operator_name}",
                f"{docs_base}nn.functional.html#{operator_name}",
            ]
        elif framework == "tensorflow":
            return [
                f"{docs_base}tf/{operator_name}",
                f"{docs_base}tf/nn/{operator_name}",
                f"{docs_base}tf/keras/layers/{operator_name}",
            ]
        elif framework == "paddlepaddle":
            return [
                f"{docs_base}paddle/{operator_name}_cn.html",
                f"{docs_base}paddle/nn/{operator_name}_cn.html",
            ]
        else:
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

                # 解析结果
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

    def extract_operator_info(
        self, search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        从搜索结果中提取算子信息

        Args:
            search_results: 搜索结果列表

        Returns:
            算子信息字典（包含代码、文档、签名等）
        """
        info = {
            "name": "",
            "code": "",
            "doc": "",
            "signature": "",
            "examples": [],
            "source_urls": [],
        }

        # 优先使用框架官方文档的结果
        docs_results = [r for r in search_results if r.source == "docs"]
        if docs_results:
            result = docs_results[0]
            info["source_urls"].append(result.url)
            info["doc"] = result.snippet

            # 尝试从文档中提取代码示例
            code_examples = self._extract_code_from_text(result.snippet)
            if code_examples:
                info["examples"] = code_examples

        # 从GitHub结果中提取代码
        github_results = [r for r in search_results if r.source == "github"]
        for result in github_results[:2]:  # 最多使用2个GitHub结果
            info["source_urls"].append(result.url)
            code = self._extract_code_from_text(result.snippet)
            if code and not info["code"]:
                info["code"] = code[0]  # 使用第一个代码片段

        # 从网络搜索结果中提取信息
        web_results = [r for r in search_results if r.source == "web_search"]
        for result in web_results[:2]:  # 最多使用2个网络搜索结果
            info["source_urls"].append(result.url)
            if result.snippet and not info["doc"]:
                info["doc"] = result.snippet

        # 合并所有文档片段
        all_docs = [r.snippet for r in search_results if r.snippet]
        if all_docs:
            info["doc"] = "\n\n".join(all_docs[:3])  # 最多3个片段

        return info

    def _extract_code_from_text(self, text: str) -> List[str]:
        """从文本中提取代码片段"""
        code_blocks = []

        # 查找代码块（```python ... ```）
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        code_blocks.extend(matches)

        # 查找内联代码（`code`）
        pattern = r"`([^`]+)`"
        matches = re.findall(pattern, text)
        # 过滤出看起来像函数调用的
        code_blocks.extend([m for m in matches if "(" in m and ")" in m])

        return code_blocks
