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

    def search_and_understand(
        self,
        query: str,
        search_url: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        使用LLM进行智能搜索和理解

        Args:
            query: 搜索查询
            search_url: 搜索URL（如PyTorch文档搜索）
            max_results: 最大结果数

        Returns:
            搜索结果列表，每个结果包含title, url, content等
        """
        # 第一步：使用LLM生成优化的搜索关键词列表
        search_keywords = self._generate_search_strategy(query)
        self.logger.debug(f"Generated search keywords: {search_keywords}")

        # 第二步：使用多个关键词尝试搜索，直到找到足够的结果
        all_results = []
        for keyword in search_keywords:
            raw_results = self._execute_search(search_url, keyword, max_results)
            all_results.extend(raw_results)

            # 如果已经找到足够的结果，停止搜索
            if len(all_results) >= max_results:
                break

        # 去重（基于URL）
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # 第三步：使用LLM重排和理解结果
        if unique_results:
            ranked_results = self._rerank_with_llm(
                query, unique_results[: max_results * 2]
            )
            return ranked_results[:max_results]

        return []

    def _generate_search_strategy(self, query: str) -> List[str]:
        """
        使用LLM生成优化的搜索关键词列表

        Args:
            query: 原始搜索查询

        Returns:
            优化的搜索关键词列表，用于多轮搜索尝试
        """
        prompt = f"""你是一个搜索优化专家。用户想要搜索：{query}

请分析这个查询，生成3-5个优化的搜索关键词，用于在不同搜索引擎中尝试。

要求：
1. 关键词应该更精确、更具体
2. 包含可能的同义词、变体或相关术语
3. 考虑中英文混合的情况
4. 优先使用最可能找到准确结果的关键词

返回JSON格式：
{{
    "keywords": [
        "关键词1",
        "关键词2",
        "关键词3"
    ]
}}

只返回JSON，不要包含其他内容。
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a search optimization expert. Generate optimized search keywords.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.3)

            # 解析JSON响应
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            strategy_data = json.loads(response)
            keywords = strategy_data.get("keywords", [query])

            # 确保至少包含原始查询
            if query not in keywords:
                keywords.insert(0, query)

            return keywords[:5]  # 最多返回5个关键词

        except Exception as e:
            self.logger.warning(f"Failed to generate search strategy: {e}")
            return [query]  # 失败时返回原始查询

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

    def fetch_and_understand_content(self, url: str) -> Optional[str]:
        """
        获取URL内容并使用LLM理解

        Args:
            url: 目标URL

        Returns:
            理解后的内容摘要
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                # 解析HTML内容
                soup = BeautifulSoup(response.text, "html.parser")

                # 检测并处理图片（特别是公式图片）
                content = self._extract_content_with_ocr(soup, url)

                # 提取文本内容
                text_content = soup.get_text(separator="\n", strip=True)
                if content:
                    # 合并OCR识别的公式和文本内容
                    full_content = f"{text_content}\n\n[公式识别结果]\n{content}"
                else:
                    full_content = text_content

                # 使用LLM理解内容
                return self._understand_content_with_llm(
                    full_content[:5000]
                )  # 限制长度

        except Exception as e:
            self.logger.warning(f"Failed to fetch content from {url}: {e}")

        return None

    def _extract_content_with_ocr(
        self, soup: BeautifulSoup, base_url: str
    ) -> Optional[str]:
        """
        从HTML中提取图片并使用OCR识别（特别是公式图片）

        Args:
            soup: BeautifulSoup解析的HTML对象
            base_url: 基础URL（用于解析相对路径）

        Returns:
            OCR识别的文本内容，如果没有图片或OCR未启用则返回None
        """
        from core.config_loader import get_config
        from tools.llm.ocr_client import OCRClient

        # 检查OCR是否启用
        config = get_config()
        web_search_config = config.get("web_search", {})
        if not web_search_config.get("ocr", False):
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

                # 判断是否为公式图片（通过alt、class等属性）
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

                # 使用OCR识别
                if is_formula:
                    # 识别公式
                    formula_text = ocr_client.recognize_formula(
                        str(img_url), use_layout_detection=False
                    )
                    if formula_text:
                        ocr_results.append(f"公式: {formula_text}")
                else:
                    # 识别普通文本（可选，如果图片看起来包含文本）
                    # 这里可以根据需要决定是否识别所有图片
                    pass

            return "\n".join(ocr_results) if ocr_results else None

        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return None

    def _understand_content_with_llm(self, content: str) -> str:
        """
        使用LLM理解内容并提取关键信息

        Args:
            content: 原始内容

        Returns:
            理解后的摘要
        """
        prompt = f"""你是一个文档分析专家。请分析以下内容，提取关键信息：

{content[:3000]}

请提取：
1. 函数/类的定义和签名
2. 主要功能和用途
3. 参数说明
4. 返回值说明
5. 使用示例（如果有）

返回结构化的摘要。
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a documentation analysis expert. Extract key information from documents.",
                },
                {"role": "user", "content": prompt},
            ]

            summary = self.llm_client.chat_completion(messages, temperature=0.3)
            return summary

        except Exception as e:
            self.logger.warning(f"LLM content understanding failed: {e}")
            return content[:500]  # 返回前500字符作为fallback
