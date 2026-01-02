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

        Args:
            query: 搜索查询（算子名称）
            search_url: 搜索页面的网址（例如PyTorch文档搜索）
            framework: 结果元数据中的框架名称
            max_results: 返回的最大结果数量

        Returns:
            包含完整内容的搜索结果列表

        Raises:
            ValueError: 如果未找到任何结果（如果有拼写错误建议则提供拼写建议）
        """
        from tools.web_search.search_tool import SearchResult

        ranked_results = self._search_and_rerank(search_url, query, max_results)

        results: List[SearchResult] = []
        for item in ranked_results:
            if not (url := item.get("url")):
                continue

            if content := self.fetch_content(url):
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
        利用LLM执行搜索并对结果进行重新排序。

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
        unique_results = self._deduplicate_results(raw_results)

        if not unique_results:
            if corrected_query := self._detect_and_correct_input_error(query):
                if corrected_query.lower() != query.lower():
                    self.logger.info(
                        f"No results for '{query}', trying corrected query: '{corrected_query}'"
                    )
                    raw_results = self._execute_search(
                        search_url, corrected_query, max_results
                    )
                    unique_results = self._deduplicate_results(raw_results)

                    if not unique_results:
                        raise ValueError(
                            f"搜索 '{query}' 未找到结果。"
                            f"系统建议您可能想搜索 '{corrected_query}'，但该查询也未找到结果。"
                            f"请检查算子名称是否正确。"
                        )
                    raise ValueError(
                        f"搜索 '{query}' 未找到结果。"
                        f"您是否想搜索 '{corrected_query}'？如果是，请使用 '{corrected_query}' 重新搜索。"
                    )

            raise ValueError(f"搜索 '{query}' 未找到结果。请检查算子名称是否正确。")

        ranked_results = self._rerank_with_llm(query, unique_results[: max_results * 2])
        return ranked_results[:max_results]

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """按URL去重搜索结果"""
        seen_urls: set[str] = set()
        unique_results: List[Dict[str, Any]] = []
        for result in results:
            if url := result.get("url"):
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
        return unique_results

    def _detect_and_correct_input_error(self, query: str) -> Optional[str]:
        """检测并纠正可能的输入错误（如拼写错误）"""
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

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            correction_data = json.loads(response)
            corrected_query = correction_data.get("corrected_query", query)
            confidence = correction_data.get("confidence", "low")

            if confidence in ["high", "medium"] and corrected_query != query:
                return corrected_query
            return None

        except Exception as e:
            self.logger.warning(f"Failed to detect input error: {e}")
            return None

    def _execute_search(
        self, search_url: str, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """执行搜索请求"""

        try:
            if "pytorch" in search_url:
                return self._search_with_sphinx_index(search_url, query, max_results)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(
                f"{search_url}?q={query}", headers=headers, timeout=10
            )
            response.raise_for_status()
            return self._parse_search_results(response.text, max_results)
        except Exception as e:
            self.logger.warning(f"Search execution failed: {e}")
            return []

    def _search_with_sphinx_index(
        self, search_url: str, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """使用 Sphinx 搜索索引执行搜索"""
        try:
            parsed = urlparse(search_url)
            path = parsed.path

            if path.endswith("/search.html"):
                base_path = path[: -len("search.html")]
            elif path.endswith("/search"):
                base_path = path[: -len("search")] + "/"
            else:
                base_path = path.rsplit("/", 1)[0] + "/"

            base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"

            if base_url not in self._sphinx_indexes:
                self._sphinx_indexes[base_url] = SphinxSearchIndex(base_url)

            return self._sphinx_indexes[base_url].search(query, max_results)

        except Exception as e:
            self.logger.warning(f"Sphinx search index failed: {e}")
            return []

    def _parse_search_results(
        self, html: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """解析搜索结果HTML"""
        results = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            search_results_div = soup.find("div", id="search-results")
            search_ul = (
                search_results_div.find("ul", class_="search")
                if search_results_div
                else None
            )

            if search_ul:
                for li in search_ul.find_all("li", recursive=False)[:max_results]:
                    if not (link := li.find("a", href=True)):
                        continue

                    title = link.get_text(strip=True)
                    url = str(link.get("href", ""))

                    score_attr = link.get("data-score")
                    try:
                        score = float(str(score_attr)) / 100.0 if score_attr else 0.5
                    except (ValueError, TypeError):
                        score = 0.5

                    context_p = li.find("p", class_="context")
                    snippet = context_p.get_text(strip=True)[:300] if context_p else ""

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
        """
        使用LLM对搜索结果进行相关性重排序

        Returns:
            重排序后的结果列表，每个结果包含relevance_score字段(0-1)
        """
        if not results:
            return []

        results_text = "\n".join(
            f"{i+1}. {r.get('title', '')} - {r.get('url', '')}"
            for i, r in enumerate(results)
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

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            ranking_data = json.loads(response)
            ranked_indices = {
                item["index"] - 1: item["relevance_score"]
                for item in ranking_data.get("ranked_results", [])
            }

            ranked_results = [
                {**result, "relevance_score": ranked_indices[i]}
                for i, result in enumerate(results)
                if i in ranked_indices
            ]
            ranked_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return ranked_results

        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}, using original order")
            return results

    def fetch_content(self, url: str) -> Optional[str]:
        """获取URL内容并提取（包括OCR识别图片）"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text_content = soup.get_text(separator="\n", strip=True)

                if ocr_content := self._extract_content_with_ocr(soup, url):
                    return f"{text_content}\n\n[图片内容]\n{ocr_content}"
                return text_content

        except Exception as e:
            self.logger.warning(f"Failed to fetch content from {url}: {e}")

        return None

    def _extract_content_with_ocr(
        self, soup: BeautifulSoup, base_url: str
    ) -> Optional[str]:
        """从HTML中提取所有图片并使用OCR识别"""
        from urllib.parse import urljoin, urlparse

        from core.config_loader import get_config_value
        from tools.llm.ocr_client import OCRClient

        if not get_config_value("web_search.ocr", False):
            return None

        try:
            if not (images := soup.find_all("img")):
                return None

            ocr_client = OCRClient()
            ocr_results = []

            for img in images:
                if not (img_src := img.get("src")):
                    continue

                img_src = str(img_src)

                if img_src.startswith(("http://", "https://")):
                    img_url = img_src
                elif img_src.startswith("//"):
                    img_url = f"https:{img_src}"
                elif img_src.startswith("/"):
                    parsed = urlparse(base_url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_src}"
                else:
                    img_url = urljoin(base_url, img_src)

                alt_text = str(img.get("alt", "")).lower()
                class_attr = img.get("class")
                class_str = (
                    " ".join(str(c) for c in class_attr).lower()
                    if isinstance(class_attr, list)
                    else str(class_attr).lower() if class_attr else ""
                )

                is_formula = any(
                    keyword in text
                    for keyword in ["formula", "math", "equation"]
                    for text in [alt_text, class_str]
                )

                if is_formula:
                    if recognized_text := ocr_client.recognize_formula(
                        str(img_url), use_layout_detection=False
                    ):
                        ocr_results.append(f"[公式] {recognized_text}")
                else:
                    if recognized_text := ocr_client.recognize_text(
                        str(img_url), use_layout_detection=True
                    ):
                        img_label = alt_text or class_str or "图片"
                        ocr_results.append(f"[{img_label}] {recognized_text}")

            return "\n".join(ocr_results) if ocr_results else None

        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return None
