"""
网络搜索工具：从PyTorch文档、GitHub、博客等获取算子信息
这是一个可复用的工具，用于后续开发
"""

import requests
import re
from typing import List, Dict, Optional, Any
from urllib.parse import quote, urljoin
import time
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class SearchResult:
    """搜索结果数据结构"""

    title: str
    url: str
    snippet: str
    source: str  # pytorch_docs, github, stackoverflow, blogs
    relevance_score: float = 0.0


class WebSearchTool:
    """
    网络搜索工具：从多个源搜索算子信息

    支持：
    - PyTorch官方文档
    - GitHub仓库
    - Stack Overflow
    - 技术博客
    """

    def __init__(self, timeout: int = 10, max_results_per_source: int = 5):
        """
        初始化搜索工具

        Args:
            timeout: 请求超时时间（秒）
            max_results_per_source: 每个源的最大结果数
        """
        self.logger = get_logger()
        self.timeout = timeout
        self.max_results = max_results_per_source

        # PyTorch文档基础URL
        self.pytorch_docs_base = "https://pytorch.org/docs/stable/"
        self.pytorch_github_base = "https://api.github.com/search/code"
        self.pytorch_github_repo = "pytorch/pytorch"

        # 用户代理（避免被网站屏蔽）
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def search_operator(
        self,
        operator_name: str,
        framework: str = "pytorch",
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        搜索算子信息

        Args:
            operator_name: 算子名称（如 "relu", "ReLU"）
            framework: 框架名称（默认pytorch）
            sources: 搜索源列表（如果为None则使用所有源）

        Returns:
            搜索结果列表
        """
        if sources is None:
            sources = ["pytorch_docs", "github", "stackoverflow", "blogs"]

        all_results = []

        # 规范化算子名称
        normalized_name = self._normalize_operator_name(operator_name)

        self.logger.info(
            f"Searching for operator '{operator_name}' ({normalized_name})"
        )

        # 从各个源搜索
        if "pytorch_docs" in sources:
            results = self._search_pytorch_docs(normalized_name)
            all_results.extend(results)

        if "github" in sources:
            results = self._search_github(normalized_name, framework)
            all_results.extend(results)

        if "stackoverflow" in sources:
            results = self._search_stackoverflow(normalized_name, framework)
            all_results.extend(results)

        if "blogs" in sources:
            results = self._search_blogs(normalized_name, framework)
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

    def _search_pytorch_docs(self, operator_name: str) -> List[SearchResult]:
        """
        搜索PyTorch官方文档

        Args:
            operator_name: 算子名称

        Returns:
            搜索结果列表
        """
        results = []

        try:
            # 尝试直接访问文档页面
            doc_urls = [
                f"{self.pytorch_docs_base}generated/torch.nn.{operator_name.capitalize()}.html",
                f"{self.pytorch_docs_base}nn.html#{operator_name}",
                f"{self.pytorch_docs_base}nn.functional.html#{operator_name}",
            ]

            for url in doc_urls:
                try:
                    response = requests.get(
                        url, headers=self.headers, timeout=self.timeout
                    )
                    if response.status_code == 200:
                        # 解析HTML内容
                        content = response.text
                        title = self._extract_title(content)
                        snippet = self._extract_snippet(content, operator_name)

                        if snippet:
                            results.append(
                                SearchResult(
                                    title=title
                                    or f"PyTorch {operator_name} Documentation",
                                    url=url,
                                    snippet=snippet,
                                    source="pytorch_docs",
                                    relevance_score=1.0,
                                )
                            )
                            break  # 找到第一个有效结果即可
                except Exception as e:
                    self.logger.debug(f"Failed to fetch {url}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"PyTorch docs search failed: {e}")

        return results[: self.max_results]

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

        try:
            # 构建GitHub API搜索查询
            query = f"{operator_name} repo:{self.pytorch_github_repo} path:torch/nn"
            params = {"q": query, "per_page": self.max_results}

            response = requests.get(
                self.pytorch_github_base,
                params=params,
                headers=self.headers,
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
        except Exception as e:
            self.logger.warning(f"GitHub search failed: {e}")

        return results

    def _search_stackoverflow(
        self, operator_name: str, framework: str
    ) -> List[SearchResult]:
        """
        搜索Stack Overflow

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            搜索结果列表
        """
        results = []

        try:
            # Stack Overflow API
            api_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "q": f"{framework} {operator_name}",
                "site": "stackoverflow",
                "pagesize": self.max_results,
                "order": "relevance",
                "sort": "relevance",
            }

            response = requests.get(api_url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                for item in data.get("items", [])[: self.max_results]:
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=self._clean_html(item.get("excerpt", "")),
                            source="stackoverflow",
                            relevance_score=0.6,
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Stack Overflow search failed: {e}")

        return results

    def _search_blogs(self, operator_name: str, framework: str) -> List[SearchResult]:
        """
        搜索技术博客（使用通用搜索）

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            搜索结果列表
        """
        # 简化实现：返回空列表
        # 实际可以使用Google Custom Search API或其他搜索API
        return []

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

        # 优先使用PyTorch官方文档的结果
        pytorch_results = [r for r in search_results if r.source == "pytorch_docs"]
        if pytorch_results:
            result = pytorch_results[0]
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

