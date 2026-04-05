"""
网络搜索智能体：使用LLM的react能力进行智能搜索和内容理解
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup

from deepmt.core.config_loader import get_config_value
from deepmt.core.logger import logger
from deepmt.tools.llm.client import LLMClient
from deepmt.tools.llm.ocr_client import OCRClient
from deepmt.tools.web_search.sphinx_search import CACHE_DIR, SphinxSearchIndex, load_json_cache, save_json_cache

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_PYPI_URLS: Dict[str, str] = {
    "pytorch": "https://pypi.org/pypi/torch/json",
    "tensorflow": "https://pypi.org/pypi/tensorflow/json",
    "paddlepaddle": "https://pypi.org/pypi/paddlepaddle/json",
}

_FRAMEWORK_API_PAGES: Dict[str, str] = {
    "pytorch": "https://docs.pytorch.org/docs/stable/pytorch-api.html",
}


def _find_article_container(soup: BeautifulSoup) -> Any:
    """从 BeautifulSoup 对象中找到文章容器（#pytorch-article → main → article）"""
    return soup.find(id="pytorch-article") or soup.find("main") or soup.find("article")

if TYPE_CHECKING:
    from deepmt.tools.web_search.search_tool import SearchResult


class SearchAgent:
    """
    网络搜索智能体：使用LLM进行智能搜索、重排和理解
    """

    def __init__(self):
        self._llm_client: Optional[LLMClient] = None
        self._ocr_client: Optional[OCRClient] = None
        self._sphinx_indexes: Dict[str, SphinxSearchIndex] = {}

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    @property
    def ocr_client(self) -> OCRClient:
        if self._ocr_client is None:
            self._ocr_client = OCRClient()
        return self._ocr_client

    def search_docs(
        self,
        query: str,
        search_url: str,
        framework: str = "pytorch",
        max_results: int = 5,
    ) -> List["SearchResult"]:
        """
        完整的文档搜索和处理流程：搜索 -> 获取内容 -> 重排 -> 过滤 -> 清洗

        Args:
            query: 搜索查询（算子名称）
            search_url: 搜索页面的网址（例如PyTorch文档搜索）
            framework: 结果元数据中的框架名称
            max_results: 返回的最大结果数量

        Returns:
            包含处理后内容的搜索结果列表

        Raises:
            ValueError: 如果未找到任何结果（如果有拼写错误建议则提供拼写建议）
        """
        from deepmt.tools.web_search.search_tool import SearchResult

        # 获取原始文档
        logger.debug(f"Executing search for '{query}'...")
        raw_results = self._execute_search(search_url, query, max_results * 2)
        unique_results = self._deduplicate_results(raw_results)
        logger.debug(f"Found {len(unique_results)} raw results")

        if not unique_results:
            if corrected_query := self._detect_and_correct_input_error(query):
                logger.debug(f"No results, try '{corrected_query}' instead?")
                raise ValueError(
                    f"搜索 '{query}' 未找到结果。"
                    f"您是否想搜索 '{corrected_query}'？如果是，请使用 '{corrected_query}' 重新搜索。"
                )
            raise ValueError(f"搜索 '{query}' 未找到结果。请检查算子名称是否正确。")

        # 获取文档内容
        logger.debug(f"Fetching {len(unique_results)} result contents...")
        urls_to_fetch: List[tuple[str, Dict[str, Any]]] = [
            (item["url"], item) for item in unique_results if item.get("url")
        ]

        results = asyncio.run(self._fetch_contents_async(urls_to_fetch))
        logger.debug(f"Fetched {len(results)} result contents")

        for result in results:
            result["content"] = self._clean_document(result["content"])

        # 根据内容进行重排和过滤
        logger.debug(f"Reranking {len(results)} results...")
        ranked_results = self._rerank_and_filter(query, results)

        # 构建结果
        return [
            SearchResult(
                title=item.get(
                    "title", f"{framework.capitalize()} {query} Documentation"
                ),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="docs",
                relevance_score=item.get("relevance_score", 0.9),
            )
            for item in ranked_results[:max_results]
        ]

    def parse_operator_doc(self, html: str) -> str:
        """从 HTML 字符串中提取算子文档正文

        Args:
            html: 算子文档页面的 HTML 字符串

        Returns:
            清洗后的文档文本
        """
        soup = BeautifulSoup(html, "html.parser")
        container = _find_article_container(soup)
        text = (
            container.get_text(separator="\n", strip=True)
            if container
            else soup.get_text(separator="\n", strip=True)
        )
        return self._clean_document(text)

    def fetch_operator_doc_by_url(self, url: str) -> Optional[str]:
        """从指定 URL 获取并解析算子文档

        Args:
            url: 算子文档页面的 URL

        Returns:
            清洗后的文档文本，获取失败时返回 None
        """
        try:
            response = requests.get(url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
            return self.parse_operator_doc(response.text)
        except Exception as e:
            logger.warning(f"Failed to fetch operator doc from {url}: {e}")
            return None

    def parse_api_list(
        self, html: str, base_url: str = ""
    ) -> List[Dict[str, str]]:
        """从 PyTorch 参考 API 页面的 HTML 中提取 API 模块/函数列表

        Args:
            html: 参考 API 页面的 HTML 字符串
            base_url: 用于解析相对链接的基础 URL

        Returns:
            API 条目列表，每项包含 'name' 和 'url'
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find(id="pytorch-article") or soup.find("main") or soup.body

        results: List[Dict[str, str]] = []
        if not article:
            return results

        for a_tag in article.find_all("a", class_="reference"):
            href = str(a_tag.get("href") or "")
            if not href:
                continue
            name = a_tag.get_text(strip=True)
            if not name:
                continue
            resolved = href if href.startswith("http") else urljoin(base_url, href)
            results.append({"name": name, "url": resolved})

        return results

    def fetch_api_list(
        self,
        url: Optional[str] = None,
        use_cache: bool = True,
        framework: str = "pytorch",
        version: str = "stable",
    ) -> List[Dict[str, str]]:
        """从指定 URL 获取并解析 API 模块列表，支持文件缓存

        Args:
            url:       参考 API 列表页面的 URL；若不提供则根据 framework/version 自动推导
            use_cache: 是否使用本地文件缓存（TTL 与 SphinxSearchIndex 共享）
            framework: 框架名称（目前仅 "pytorch" 可用），url 未提供时使用
            version:   文档版本，"stable"（默认）或具体版本如 "2.1"
                       非 stable 尚未实现，会抛 NotImplementedError

        Returns:
            API 条目列表，每项包含 'name' 和 'url'（模块级，非个体 API）

        Note:
            此方法返回的是模块级链接（如 torch.nn、torch.nn.functional）。
            如需个体 API 列表，请使用 tools.web_search.api_list_fetcher.APIListFetcher。
        """
        if version != "stable":
            raise NotImplementedError(
                f"Versioned API list (version='{version}') is not yet implemented. "
                f"Currently only version='stable' is supported."
            )
        if url is None:
            url = _FRAMEWORK_API_PAGES.get(framework)
            if not url:
                raise NotImplementedError(
                    f"No API index page configured for framework '{framework}'. "
                    f"Currently supported: {list(_FRAMEWORK_API_PAGES.keys())}"
                )
        cache_path = CACHE_DIR / f"api_list_{hashlib.md5(url.encode()).hexdigest()[:16]}.json"
        if use_cache:
            cached = load_json_cache(cache_path)
            if cached is not None:
                return cached
        try:
            response = requests.get(url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
            result = self.parse_api_list(response.text, base_url=url)
            save_json_cache(cache_path, result, indent=2)
            logger.debug(f"API list cached: {cache_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch API list from {url}: {e}")
            return []

    def _api_list_cache_path(self, url: str) -> Path:
        return CACHE_DIR / f"api_list_{hashlib.md5(url.encode()).hexdigest()[:16]}.json"

    def fetch_framework_versions(self, framework: str) -> List[Dict[str, str]]:
        """从 PyPI 获取框架版本列表（不需要 LLM，纯 HTTP）

        Args:
            framework: 框架名称（pytorch / tensorflow / paddlepaddle）

        Returns:
            版本列表，每项包含 'version' 和 'upload_time'，按版本降序排列
        """
        pypi_url = _PYPI_URLS.get(framework)
        if not pypi_url:
            logger.warning(f"No PyPI URL configured for framework: {framework}")
            return []
        try:
            response = requests.get(pypi_url, headers=_HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
            releases = data.get("releases", {})
            versions = []
            for ver, files in releases.items():
                if not files:
                    continue
                upload_time = files[0].get("upload_time", "")
                versions.append({"version": ver, "upload_time": upload_time})
            # 按版本号降序（字典序足够过滤明显旧版本，精确排序通过 upload_time）
            versions.sort(key=lambda x: x["upload_time"], reverse=True)
            return versions
        except Exception as e:
            logger.warning(f"Failed to fetch versions for {framework}: {e}")
            return []

    def get_latest_stable_version(self, framework: str) -> Optional[str]:
        """从 PyPI 获取框架最新稳定版本号（不需要 LLM）

        Args:
            framework: 框架名称

        Returns:
            版本字符串如 "2.6.0"，失败时返回 None
        """
        pypi_url = _PYPI_URLS.get(framework)
        if not pypi_url:
            return None
        try:
            response = requests.get(pypi_url, headers=_HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
            return str(data["info"]["version"])
        except Exception as e:
            logger.warning(f"Failed to get latest version for {framework}: {e}")
            return None

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

            response = self.llm_client.chat_completion(messages)

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
            logger.warning(f"Failed to detect input error: {e}")
            return None

    def _execute_search(
        self, search_url: str, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """执行搜索请求 (PyTorch 使用 Sphinx 搜索索引，其他框架使用搜索引擎)"""
        try:
            if "pytorch" in search_url:
                return self._search_with_sphinx_index(search_url, query, max_results)

            response = requests.get(
                f"{search_url}?q={query}", headers=_HEADERS, timeout=10
            )
            response.raise_for_status()
            return self._parse_search_results(response.text, max_results)
        except Exception as e:
            logger.warning(f"Search execution failed: {e}")
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
            logger.warning(f"Sphinx search index failed: {e}")
            return []

    def _parse_search_results(
        self, html: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """解析搜索结果HTML"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            search_results_div = soup.find("div", id="search-results")
            search_ul = (
                search_results_div.find("ul", class_="search")
                if search_results_div
                else None
            )

            if not search_ul:
                return []

            results = []
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

                if title and url:
                    context_p = li.find("p", class_="context")
                    snippet = context_p.get_text(strip=True)[:300] if context_p else ""
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "relevance_score": score,
                        }
                    )

            return results

        except Exception as e:
            logger.warning(f"Failed to parse search results: {e}")
            return []

    def _rerank_and_filter(
        self,
        query: str,
        results: List[Dict[str, Any]],
        filter_zero_score: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        使用LLM根据文档内容对搜索结果进行相关性重排序和过滤

        Args:
            query: 搜索查询
            results: 包含content字段的搜索结果列表
            filter_zero_score: 是否过滤掉0分（不相关）的结果

        Returns:
            重排序并过滤后的结果列表，每个结果包含relevance_score字段(0-1)
        """
        if not results:
            return []

        results_text = "\n".join(
            f"{i+1}. Title: {r.get('title', '')}\n"
            f"   URL: {r.get('url', '')}\n"
            f"   Snippet: {(r.get('content', '') or r.get('snippet', ''))[:500]}"
            for i, r in enumerate(results)
        )

        prompt = f"""Query: "{query}"

Search results:
{results_text}

Rate each result's relevance to the query on a 0-1 scale:

**Scoring Guidelines:**
- **1.0**: Perfect match for the EXACT operator documentation
- **0.7-0.9**: Highly relevant - standard API for the same operator (e.g., class vs functional)
- **0.4-0.6**: Moderately relevant - mentions the operator but not the main documentation
- **0.1-0.3**: Slightly relevant - tangentially related content
- **0.0**: NOT RELEVANT - should be completely excluded, including:
  * Quantized/quantization versions (quantized.Conv1d, int8 variants)
  * Specialized variants that are not the standard operator
  * Unrelated content that happens to contain the keyword

**IMPORTANT**: Be strict! Give score 0.0 for:
- Any result with "quantized", "quantize", "int8" in title or URL (unless the query itself is about quantization)
- Variants that are clearly different from the standard operator
- Duplicate or redundant content when the standard version is already present

Return JSON only:
{{
    "ranked_results": [
        {{"index": 1, "relevance_score": 0.95, "reason": "brief explanation"}}
    ]
}}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a search result ranking expert specializing in deep learning framework documentation. "
                    "Rate relevance scores accurately. Set score to 0 for completely irrelevant results.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client.chat_completion(messages)

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
                and not (filter_zero_score and ranked_indices[i] == 0)
            ]

            ranked_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return ranked_results

        except Exception as e:
            logger.warning(
                f"⚠️  WARN | LLM reranking failed: {e}, using original order"
            )
            return results

    async def _fetch_contents_async(
        self, urls_to_fetch: List[tuple[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """异步并发获取多个URL的内容"""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers=_HEADERS,
        ) as session:
            tasks = [
                self._fetch_content_async(session, url) for url, _ in urls_to_fetch
            ]
            contents = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for (url, item), content in zip(urls_to_fetch, contents):
                if isinstance(content, Exception):
                    logger.warning(
                        f"Failed to fetch content from {url}: {str(content)}"
                    )
                elif content:
                    results.append({**item, "content": content})

        return results

    async def _fetch_content_async(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        """异步获取单个URL的内容"""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                article_container = _find_article_container(soup)
                text_content = (
                    article_container.get_text(separator="\n", strip=True)
                    if article_container
                    else soup.get_text(separator="\n", strip=True)
                )

                if ocr_content := self._extract_content_with_ocr(
                    soup, url, article_container
                ):
                    return f"{text_content}\n\n[图片内容]\n{ocr_content}"

                return text_content

        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return None

    def _extract_content_with_ocr(
        self,
        soup: BeautifulSoup,
        base_url: str,
        article_container: Optional[Any] = None,
    ) -> Optional[str]:
        """从HTML中提取内容相关的图片并使用OCR识别"""
        if not get_config_value("web_search.ocr", False):
            return None

        try:
            search_scope = article_container if article_container else soup
            if not (images := search_scope.find_all("img")):
                return None

            # 用于过滤小图标和无关图片的路径关键词
            skip_patterns = {
                "logo",
                "icon",
                "favicon",
                "avatar",
                "badge",
                "button",
                "arrow",
                "spinner",
                "loading",
                "social",
                "twitter",
                "facebook",
                "github",
                "linkedin",
            }

            ocr_results = []
            for img in images:
                if not (img_src := img.get("src")):
                    continue

                img_src = str(img_src)
                img_src_lower = img_src.lower()

                # 跳过小图标和无关图片
                if (
                    any(pattern in img_src_lower for pattern in skip_patterns)
                    or img_src.startswith("data:")
                    or "_images/" in img_src
                ):
                    continue

                # 跳过 PyTorch 示意图
                if "_images/" in img_src:
                    continue

                # 解析图片 URL
                if img_src.startswith(("http://", "https://")):
                    img_url = img_src
                elif img_src.startswith("//"):
                    img_url = f"https:{img_src}"
                elif img_src.startswith("/"):
                    parsed = urlparse(base_url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_src}"
                else:
                    img_url = urljoin(base_url, img_src)

                alt_text = str(img.get("alt", ""))
                alt_text_lower = alt_text.lower()
                class_attr = img.get("class")
                class_str = (
                    " ".join(str(c) for c in class_attr).lower()
                    if isinstance(class_attr, list)
                    else str(class_attr).lower() if class_attr else ""
                )

                # 判断是否为公式图片
                is_formula = any(
                    keyword in text
                    for keyword in ["formula", "math", "equation", "latex"]
                    for text in [alt_text_lower, class_str, img_src_lower]
                )

                if is_formula:
                    if recognized_text := self.ocr_client.recognize_formula(
                        img_url, use_layout_detection=False
                    ):
                        ocr_results.append(f"[公式] {recognized_text}")
                else:
                    if recognized_text := self.ocr_client.recognize_text(
                        img_url, use_layout_detection=True
                    ):
                        img_label = alt_text or img_src.split("/")[-1].split(".")[0]
                        ocr_results.append(f"[{img_label}] {recognized_text}")

            return "\n".join(ocr_results) if ocr_results else None

        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None

    def _clean_document(self, doc: str, clean_latex: bool = False) -> str:
        """清洗文档内容：删除HTML标签、多余空白字符、LaTeX公式"""
        if not doc or not doc.strip():
            return ""

        doc = re.sub(r"<[^>]+>", "", doc)
        doc = "\n".join(
            re.sub(r"\s+", " ", line).strip()
            for line in doc.split("\n")
            if line.strip()
        )

        return self._clean_latex(doc) if clean_latex else doc

    def _clean_latex(self, doc: str) -> str:
        """使用 pylatexenc 清洗 LaTeX 公式，转换为可读的纯文本"""
        from pylatexenc.latex2text import LatexNodes2Text

        logger.debug("Using pylatexenc for LaTeX cleaning")
        text = LatexNodes2Text().latex_to_text(doc)
        return re.sub(r"\n\s*\n", "\n\n", text)
