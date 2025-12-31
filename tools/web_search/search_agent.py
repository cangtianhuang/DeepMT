"""
网络搜索智能体：使用LLM的react能力进行智能搜索和内容理解
"""

import json
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from core.logger import get_logger
from tools.llm.client import LLMClient


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

    def search_and_rerank(
        self,
        query: str,
        search_url: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        使用LLM进行智能搜索和重排（不进行内容理解）

        Args:
            query: 搜索查询
            search_url: 搜索URL（如PyTorch文档搜索）
            max_results: 最大结果数

        Returns:
            搜索结果列表，每个结果包含title, url, snippet等

        Raises:
            ValueError: 如果搜索没有结果且检测到可能的输入错误（如拼写错误）
        """
        # PyTorch 官网不支持多关键词搜索，直接使用原始查询
        raw_results = self._execute_search(search_url, query, max_results)

        # 去重（基于URL）
        seen_urls = set()
        unique_results = []
        for result in raw_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # 如果没有找到结果，使用LLM检测可能的输入错误
        if not unique_results:
            corrected_query = self._detect_and_correct_input_error(query)
            if corrected_query and corrected_query.lower() != query.lower():
                # 使用纠正后的查询重新搜索
                self.logger.info(
                    f"No results for '{query}', trying corrected query: '{corrected_query}'"
                )
                raw_results = self._execute_search(
                    search_url, corrected_query, max_results
                )

                # 再次去重
                seen_urls = set()
                unique_results = []
                for result in raw_results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)

                # 如果纠正后的查询也没有结果，抛出错误
                if not unique_results:
                    raise ValueError(
                        f"搜索 '{query}' 未找到结果。"
                        f"系统建议您可能想搜索 '{corrected_query}'，但该查询也未找到结果。"
                        f"请检查算子名称是否正确。"
                    )
                else:
                    # 抛出错误告知用户可能的输入错误
                    raise ValueError(
                        f"搜索 '{query}' 未找到结果。"
                        f"您是否想搜索 '{corrected_query}'？如果是，请使用 '{corrected_query}' 重新搜索。"
                    )
            else:
                # 没有找到纠正建议，直接抛出错误
                raise ValueError(f"搜索 '{query}' 未找到结果。请检查算子名称是否正确。")

        # 使用LLM重排结果（不进行理解）
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
        prompt = f"""你是一个PyTorch算子名称纠正专家。用户搜索了：{query}

PyTorch文档中没有找到结果。请分析这个查询，判断是否是常见的拼写错误。

常见的PyTorch算子包括：
- ReLU (不是 rlu, relu, ReLu)
- Conv2d (不是 conv2D, Conv2D)
- BatchNorm (不是 BatchNorm1d, BatchNorm2d 的简写)
- MaxPool2d (不是 MaxPool)

如果发现可能是拼写错误，请返回最可能的正确算子名称。
如果无法确定或查询看起来合理，返回原始查询。

返回JSON格式：
{{
    "corrected_query": "正确的算子名称（如果发现错误）",
    "confidence": "high/medium/low"
}}

只返回JSON，不要包含其他内容。
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a PyTorch operator name correction expert. Detect and correct spelling errors in operator names.",
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
            # 构建搜索URL
            if "?" in search_url:
                full_url = f"{search_url}&q={query}"
            else:
                full_url = f"{search_url}?q={query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(full_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # 解析搜索结果页面
                return self._parse_search_results(response.text, max_results)
        except Exception as e:
            self.logger.warning(f"Search execution failed: {e}")

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

            # 多种策略解析搜索结果
            # 策略1: 查找包含搜索结果的容器
            search_containers = soup.find_all(
                ["div", "ul", "ol"],
                class_=re.compile(r"search|result|item|hit", re.I),
            )

            for container in search_containers:
                # 在容器中查找链接
                links = container.find_all("a", href=True)
                for link in links[:max_results]:
                    title = link.get_text(strip=True)
                    url_attr = link.get("href")
                    url = str(url_attr) if url_attr else ""
                    if url:
                        if not url.startswith("http"):
                            # 相对URL转换为绝对URL
                            if url.startswith("/"):
                                url = f"https://pytorch.org{url}"
                            else:
                                url = (
                                    f"https://pytorch.org/docs/stable/{url.lstrip('/')}"
                                )

                        if title and url and url not in [r.get("url") for r in results]:
                            # 尝试获取摘要
                            snippet = ""
                            parent = link.find_parent()
                            if parent:
                                snippet_elem = parent.find(
                                    ["p", "span", "div"],
                                    class_=re.compile(r"snippet|summary|desc", re.I),
                                )
                                if snippet_elem:
                                    snippet = snippet_elem.get_text(strip=True)[:200]

                            results.append(
                                {"title": title, "url": url, "snippet": snippet}
                            )

            # 策略2: 如果策略1没有找到结果，直接查找所有链接
            if not results:
                all_links = soup.find_all("a", href=re.compile(r"\.html|#"))
                for link in all_links[: max_results * 2]:
                    title = link.get_text(strip=True)
                    url_attr = link.get("href")
                    url = str(url_attr) if url_attr else ""
                    if url and title and len(title) > 3:
                        if not url.startswith("http"):
                            if url.startswith("/"):
                                url = f"https://pytorch.org{url}"
                            else:
                                url = (
                                    f"https://pytorch.org/docs/stable/{url.lstrip('/')}"
                                )

                        if url not in [r.get("url") for r in results]:
                            results.append({"title": title, "url": url, "snippet": ""})

        except Exception as e:
            self.logger.warning(f"Failed to parse search results: {e}")

        return results[:max_results]

    def _rerank_with_llm(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用LLM对搜索结果进行重排和理解

        Args:
            query: 原始查询
            results: 搜索结果列表

        Returns:
            重排后的结果列表
        """
        if not results:
            return []

        # 构建重排提示
        results_text = "\n".join(
            [
                f"{i+1}. {r.get('title', '')} - {r.get('url', '')}"
                for i, r in enumerate(results)
            ]
        )

        prompt = f"""你是一个搜索结果分析专家。用户搜索了：{query}

以下是搜索结果：
{results_text}

请：
1. 评估每个结果与查询的相关性（0-1分）
2. 选择最相关的Top-{len(results)}个结果
3. 对每个选中的结果，生成一个简短的摘要说明为什么它相关

返回JSON格式：
{{
    "ranked_results": [
        {{
            "index": 1,
            "relevance_score": 0.95,
            "reason": "这个结果直接包含用户查询的内容"
        }}
    ]
}}

只返回JSON，不要包含其他内容。
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a search result ranking expert. Analyze and rank search results.",
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
