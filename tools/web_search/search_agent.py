"""
网络搜索智能体：使用LLM的react能力进行智能搜索和内容理解
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from core.logger import get_logger
from tools.llm.client import LLMClient
from tools.web_search.sphinx_search import SphinxSearchIndex

if TYPE_CHECKING:
    from tools.web_search.search_tool import SearchResult


class SearchAgent:
    """
    网络搜索智能体：使用LLM进行智能搜索、重排和理解
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化搜索智能体

        Args:
            llm_client: LLM客户端（如果为None则创建默认客户端）
        """
        self.logger = get_logger()
        self.llm_client = llm_client or LLMClient()
        # 缓存 Sphinx 搜索索引实例
        self._sphinx_indexes: Dict[str, SphinxSearchIndex] = {}

    def search_docs(
        self,
        query: str,
        search_url: str,
        framework: str = "pytorch",
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        完整的文档搜索流程：搜索 -> 重新排序 -> 获取内容。

        该方法集成了完整的搜索工作流：
        1. 在文档站点执行搜索
        2. 解析并去重结果
        3. 使用LLM对结果重新排序
        4. 获取每个结果的完整内容

        Args:
            query: 搜索查询（算子名称）
            search_url: 搜索页面的网址（例如PyTorch文档搜索）
            framework: 结果元数据中的框架名称
            max_results: 返回的最大结果数量

        Returns:
            List of SearchResult with full content（包含完整内容的搜索结果列表）

        Raises:
            ValueError: 如果未找到任何结果（如果有拼写错误建议则提供拼写建议）
        """
        # 运行时导入以避免循环依赖
        from tools.web_search.search_tool import SearchResult

        # 第一步：搜索并重新排序
        ranked_results = self._search_and_rerank(search_url, query, max_results)

        # 第二步：获取每个结果的完整内容
        results: List[SearchResult] = []
        for item in ranked_results:
            url = item.get("url", "")
            if not url:
                continue

            content = self.fetch_content(url)
            if content:
                results.append(
                    SearchResult(
                        title=item.get(
                            "title",
                            f"{framework.capitalize()} {query} Documentation",
                        ),
                        url=url,
                        snippet=content,
                        source="docs",
                        relevance_score=item.get("relevance_score", 0.9),
                    )
                )

        return results[:max_results]

    def _search_and_rerank(
        self,
        search_url: str,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        利用大语言模型（LLM）执行搜索并对结果进行重新排序。

        Args:
            search_url: 搜索页面的网址
            query: 搜索查询内容
            max_results: 结果的最大数量

        Returns:
            重新排序后的结果列表

        Raises:
            ValueError: 如果未找到任何结果
        """
        raw_results = self._execute_search(search_url, query, max_results)

        # 去重：按URL
        seen_urls: set = set()
        unique_results: List[Dict[str, Any]] = []
        for result in raw_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # 如果未找到结果，尝试检测并纠正拼写错误
        if not unique_results:
            corrected_query = self._detect_and_correct_input_error(query)
            if corrected_query and corrected_query.lower() != query.lower():
                self.logger.info(
                    f"No results for '{query}', trying corrected query: '{corrected_query}'"
                )
                raw_results = self._execute_search(
                    search_url, corrected_query, max_results
                )

                seen_urls = set()
                unique_results = []
                for result in raw_results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)

                if not unique_results:
                    raise ValueError(
                        f"搜索 '{query}' 未找到结果。"
                        f"系统建议您可能想搜索 '{corrected_query}'，但该查询也未找到结果。"
                        f"请检查算子名称是否正确。"
                    )
                else:
                    raise ValueError(
                        f"搜索 '{query}' 未找到结果。"
                        f"您是否想搜索 '{corrected_query}'？如果是，请使用 '{corrected_query}' 重新搜索。"
                    )
            else:
                raise ValueError(f"搜索 '{query}' 未找到结果。请检查算子名称是否正确。")

        # 使用LLM对结果进行重新排序
        ranked_results = self._rerank_with_llm(query, unique_results[: max_results * 2])
        return ranked_results[:max_results]

    def _detect_and_correct_input_error(self, query: str) -> Optional[str]:
        """
        检测并纠正可能的输入错误（如拼写错误）

        Args:
            query: 原始搜索查询

        Returns:
            纠正后的查询字符串，如果没有找到纠正建议则返回None
        """
        prompt = f"""The user searched for: "{query}"

No results were found in PyTorch documentation. Analyze if this might be a spelling error.

If you detect a spelling error, return the most likely correct operator name.
If the query appears valid or you're unsure, return the original query.

Return JSON format only:
{{
    "corrected_query": "corrected operator name",
    "confidence": "high/medium/low"
}}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a PyTorch operator name correction expert. Detect and correct spelling errors.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.3)

            # 解析JSON响应
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            correction_data = json.loads(response)
            corrected_query = correction_data.get("corrected_query", query)
            confidence = correction_data.get("confidence", "low")

            # 只在置信度较高时返回纠正结果
            if confidence in ["high", "medium"] and corrected_query != query:
                return corrected_query
            return None

        except Exception as e:
            self.logger.warning(f"Failed to detect input error: {e}")
            return None

    def _execute_search(
        self, search_url: str, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        执行搜索请求

        Args:
            search_url: 搜索URL
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            原始搜索结果列表
        """

        try:
            # pytorch 使用 Sphinx 搜索索引
            if "pytorch" in search_url:
                return self._search_with_sphinx_index(search_url, query, max_results)

            # 其他框架使用传统的 HTML 解析方式
            full_url = f"{search_url}?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()
            return self._parse_search_results(response.text, max_results)
        except Exception as e:
            self.logger.warning(f"Search execution failed: {e}")
            return []

    def _search_with_sphinx_index(
        self, search_url: str, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        使用 Sphinx 搜索索引执行搜索

        Args:
            search_url: 搜索页面 URL
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        try:
            # 从 search_url 提取 base_url
            parsed = urlparse(search_url)
            path = parsed.path
            # 移除 search.html 或类似的文件名
            if path.endswith("/search.html"):
                base_path = path[: -len("search.html")]
            elif path.endswith("/search"):
                base_path = path[: -len("search")] + "/"
            else:
                # 尝试获取目录路径
                base_path = path.rsplit("/", 1)[0] + "/"

            base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"

            # 获取或创建搜索索引实例
            if base_url not in self._sphinx_indexes:
                self._sphinx_indexes[base_url] = SphinxSearchIndex(base_url)

            sphinx_index = self._sphinx_indexes[base_url]
            return sphinx_index.search(query, max_results)

        except Exception as e:
            self.logger.warning(f"Sphinx search index failed: {e}")
            return []

    def _parse_search_results(
        self, html: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        解析搜索结果HTML

        Args:
            html: HTML内容
            max_results: 最大结果数

        Returns:
            解析后的结果列表
        """
        results = []
        try:
            soup = BeautifulSoup(html, "html.parser")

            # 以 pytorch 特有的结构为例：
            # <div id="search-results"> -> <ul class="search"> -> <li>
            search_results_div = soup.find("div", id="search-results")
            if search_results_div:
                search_ul = search_results_div.find("ul", class_="search")
                if search_ul:
                    list_items = search_ul.find_all("li", recursive=False)
                    for li in list_items[:max_results]:
                        link = li.find("a", href=True)
                        if not link:
                            continue

                        title = link.get_text(strip=True)
                        url = str(link.get("href", ""))

                        # Get relevance score from data-score attribute
                        score_attr = link.get("data-score")
                        try:
                            score = (
                                float(str(score_attr)) / 100.0 if score_attr else 0.5
                            )
                        except (ValueError, TypeError):
                            score = 0.5

                        # Get snippet from <p class="context">
                        snippet = ""
                        context_p = li.find("p", class_="context")
                        if context_p:
                            snippet = context_p.get_text(strip=True)[:300]

                        if title and url:
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "relevance_score": score,
                                }
                            )

        except Exception as e:
            self.logger.warning(f"Failed to parse search results: {e}")

        return results[:max_results]

    def _rerank_with_llm(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用LLM对搜索结果进行相关性重排序

        Args:
            query: 原始搜索查询
            results: 待排序的搜索结果列表

        Returns:
            重排序后的结果列表。如果处理失败则返回原始顺序的结果。
            每个结果字典中会新增'relevance_score'字段表示相关性分数(0-1)。
        """
        # 空结果直接返回
        if not results:
            return []

        # 构建重排序提示词
        results_text = "\n".join(
            [
                f"{i+1}. {r.get('title', '')} - {r.get('url', '')}"
                for i, r in enumerate(results)
            ]
        )

        prompt = f"""Query: "{query}"

Search results:
{results_text}

Rate each result's relevance (0-1 scale) to the query.

Return JSON only:
{{
    "ranked_results": [
        {{"index": 1, "relevance_score": 0.95}}
    ]
}}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a search result ranking expert. Rate relevance scores.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.3)

            # 解析LLM响应
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            ranking_data = json.loads(response)
            ranked_indices = {
                item["index"] - 1: item["relevance_score"]
                for item in ranking_data.get("ranked_results", [])
            }

            # 根据重排结果重新排序
            ranked_results = []
            for i, result in enumerate(results):
                if i in ranked_indices:
                    result["relevance_score"] = ranked_indices[i]
                    ranked_results.append(result)

            # 按相关性分数排序
            ranked_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return ranked_results

        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}, using original order")
            return results

    def fetch_content(self, url: str) -> Optional[str]:
        """
        获取URL内容并提取（包括OCR识别图片，不进行理解）

        Args:
            url: 目标URL

        Returns:
            原始内容文本（包含OCR识别的图片内容）
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                # 解析HTML内容
                soup = BeautifulSoup(response.text, "html.parser")

                # 提取文本内容
                text_content = soup.get_text(separator="\n", strip=True)

                # 检测并处理所有图片（使用OCR识别）
                ocr_content = self._extract_content_with_ocr(soup, url)

                # 合并OCR识别的内容和文本内容
                if ocr_content:
                    # 将OCR结果插入到原始文本中（替换图片位置）
                    full_content = f"{text_content}\n\n[图片内容]\n{ocr_content}"
                else:
                    full_content = text_content

                return full_content

        except Exception as e:
            self.logger.warning(f"Failed to fetch content from {url}: {e}")

        return None

    def _extract_content_with_ocr(
        self, soup: BeautifulSoup, base_url: str
    ) -> Optional[str]:
        """
        从HTML中提取所有图片并使用OCR识别

        Args:
            soup: BeautifulSoup解析的HTML对象
            base_url: 基础URL（用于解析相对路径）

        Returns:
            OCR识别的文本内容，如果没有图片或OCR未启用则返回None
        """
        from core.config_loader import get_config_value
        from tools.llm.ocr_client import OCRClient

        # 检查OCR是否启用（不保存完整配置）
        if not get_config_value("web_search.ocr", False):
            return None

        try:
            # 查找所有图片标签
            images = soup.find_all("img")
            if not images:
                return None

            ocr_client = OCRClient()
            ocr_results = []

            for img in images:
                img_src = img.get("src", "")
                if not img_src:
                    continue

                # 确保 img_src 是字符串
                img_src = str(img_src)

                # 构建完整URL
                if img_src.startswith("http://") or img_src.startswith("https://"):
                    img_url = img_src
                elif img_src.startswith("//"):
                    img_url = f"https:{img_src}"
                elif img_src.startswith("/"):
                    from urllib.parse import urlparse

                    parsed = urlparse(base_url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_src}"
                else:
                    from urllib.parse import urljoin

                    img_url = urljoin(base_url, img_src)

                # 判断是否为公式图片（用于选择合适的OCR方法）
                alt_attr = img.get("alt")
                alt_text = str(alt_attr).lower() if alt_attr else ""
                class_attr = img.get("class")
                if isinstance(class_attr, list):
                    class_str = " ".join(str(c) for c in class_attr).lower()
                else:
                    class_str = str(class_attr).lower() if class_attr else ""

                is_formula = (
                    "formula" in alt_text
                    or "formula" in class_str
                    or "math" in alt_text
                    or "math" in class_str
                    or "equation" in alt_text
                    or "equation" in class_str
                )

                # 使用OCR识别所有图片
                if is_formula:
                    # 识别公式（使用专门的公式识别）
                    recognized_text = ocr_client.recognize_formula(
                        str(img_url), use_layout_detection=False
                    )
                    if recognized_text:
                        ocr_results.append(f"[公式] {recognized_text}")
                else:
                    # 识别普通图片（包括图表、文本等）
                    # 使用版面分析以获得更好的识别效果
                    recognized_text = ocr_client.recognize_text(
                        str(img_url), use_layout_detection=True
                    )
                    if recognized_text:
                        # 添加图片位置标记（如果有alt或class信息）
                        img_label = alt_text or class_str or "图片"
                        ocr_results.append(f"[{img_label}] {recognized_text}")

            return "\n".join(ocr_results) if ocr_results else None

        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return None
