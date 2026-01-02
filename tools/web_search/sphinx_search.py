"""Sphinx 文档搜索索引解析器"""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from core.logger import get_logger


class SphinxSearchIndex:
    """Sphinx 文档搜索索引解析器"""

    def __init__(self, base_url: str):
        """初始化搜索索引"""
        self.base_url = base_url.rstrip("/") + "/"
        self.index_url = urljoin(self.base_url, "searchindex.js")
        self.logger = get_logger()
        self._index: Optional[Dict[str, Any]] = None
        self._docnames: List[str] = []
        self._titles: Dict[int, str] = {}
        self._filenames: Dict[int, str] = {}
        self._terms: Dict[str, List[int]] = {}
        self._titleterms: Dict[str, List[int]] = {}

    def _load_index(self) -> bool:
        """下载并解析搜索索引"""
        if self._index is not None:
            return True

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(self.index_url, headers=headers, timeout=30)

            if response.status_code != 200:
                self.logger.warning(
                    f"Failed to fetch search index: HTTP {response.status_code}"
                )
                return False

            text = response.text.strip()
            if text.startswith("Search.setIndex("):
                text = text[len("Search.setIndex(") : -1]

            index_data = json.loads(text)
            self._index = index_data
            self._docnames = index_data.get("docnames", [])
            self._titles = dict(enumerate(index_data.get("titles", [])))
            self._filenames = dict(enumerate(index_data.get("filenames", [])))
            self._terms = index_data.get("terms", {})
            self._titleterms = index_data.get("titleterms", {})

            self.logger.info(
                f"Loaded search index: {len(self._docnames)} docs, "
                f"{len(self._terms)} terms"
            )
            return True

        except Exception as e:
            self.logger.warning(f"Failed to load search index: {e}")
            return False

    def _normalize_doc_ids(self, value: Any) -> List[int]:
        """标准化文档 ID 列表（Sphinx 索引中的值可能是单个整数或整数列表）"""
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            return value
        return []

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """搜索文档"""
        if not self._load_index():
            return []

        query_terms = query.lower().split()
        if not query_terms:
            return []

        doc_scores: Dict[int, float] = {}

        for term in query_terms:
            if term in self._titleterms:
                for doc_id in self._normalize_doc_ids(self._titleterms[term]):
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 10
            if term in self._terms:
                for doc_id in self._normalize_doc_ids(self._terms[term]):
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

            for index_term, doc_ids in self._titleterms.items():
                if index_term.startswith(term) and index_term != term:
                    for doc_id in self._normalize_doc_ids(doc_ids):
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 5
            for index_term, doc_ids in self._terms.items():
                if index_term.startswith(term) and index_term != term:
                    for doc_id in self._normalize_doc_ids(doc_ids):
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:max_results]:
            title = self._titles.get(doc_id, "")
            filename = self._filenames.get(doc_id, "")

            if filename:
                if filename.endswith(".rst"):
                    filename = filename[:-4] + ".html"
                elif not filename.endswith(".html"):
                    filename = filename + ".html"
                url = urljoin(self.base_url, filename)
            else:
                url = self.base_url

            results.append(
                {
                    "title": title,
                    "url": url,
                    "relevance_score": min(score / 20.0, 1.0), # 归一化
                    "snippet": "",
                }
            )

        return results
