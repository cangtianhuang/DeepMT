"""Sphinx 文档搜索索引解析器"""

import hashlib
import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

from deepmt.core.logger import get_logger

# 缓存目录和过期时间（秒）
CACHE_DIR = Path("data/web_search_cache")
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 1 day


# ---------------------------------------------------------------------------
# 共用 JSON 文件缓存工具（供本模块及 search_agent.py 使用）
# ---------------------------------------------------------------------------

def load_json_cache(path: Path, ttl: float = CACHE_EXPIRY_SECONDS) -> Optional[Any]:
    """从文件加载 JSON 缓存，若不存在或已过期返回 None"""
    if not path.exists() or time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json_cache(path: Path, data: Any, indent: Optional[int] = None) -> None:
    """将数据写入 JSON 文件缓存（失败时静默忽略）"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception:
        pass


class SphinxSearchIndex:
    """Sphinx 文档搜索索引解析器"""

    def __init__(self, base_url: str, threshold: float = 0.1):
        """初始化搜索索引

        Args:
            base_url: 文档基础URL
            threshold: 评分阈值，低于此值的文档不会被返回（默认0.1）
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.index_url = urljoin(self.base_url, "searchindex.js")
        self.logger = get_logger(self.__class__.__name__)
        self.threshold = threshold
        self._index: Optional[Dict[str, Any]] = None
        self._docnames: List[str] = []
        self._titles: Dict[int, str] = {}
        self._filenames: Dict[int, str] = {}
        self._terms: Dict[str, List[int]] = {}
        self._titleterms: Dict[str, List[int]] = {}
        self._sorted_terms: List[str] = []
        self._sorted_titleterms: List[str] = []
        self._reversed_terms: List[Tuple[str, str]] = []
        self._reversed_titleterms: List[Tuple[str, str]] = []

    def _get_cache_path(self) -> Path:
        url_hash = hashlib.md5(self.index_url.encode()).hexdigest()[:16]
        return CACHE_DIR / f"{url_hash}.json"

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        data = load_json_cache(self._get_cache_path())
        if data is not None:
            self.logger.debug(f"Sphinx search index loaded from cache: {self._get_cache_path()}")
        return data

    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        save_json_cache(self._get_cache_path(), data)
        self.logger.debug(f"Sphinx search index saved to cache: {self._get_cache_path()}")

    def _load_index(self) -> bool:
        """下载并解析搜索索引（支持缓存）"""
        if self._index is not None:
            return True

        index_data = self._load_from_cache()

        if index_data is None:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(self.index_url, headers=headers, timeout=30)
                response.raise_for_status()

                text = response.text.strip()
                if text.startswith("Search.setIndex("):
                    text = text[len("Search.setIndex(") : -1]

                index_data = json.loads(text)
                self._save_to_cache(index_data)

            except Exception as e:
                self.logger.warning(f"Sphinx search index failed to load: {e}")
                return False

        self._index = index_data
        self._docnames = index_data.get("docnames", [])
        self._titles = dict(enumerate(index_data.get("titles", [])))
        self._filenames = dict(enumerate(index_data.get("filenames", [])))
        self._terms = index_data.get("terms", {})
        self._titleterms = index_data.get("titleterms", {})

        self._sorted_terms = sorted(self._terms.keys())
        self._sorted_titleterms = sorted(self._titleterms.keys())
        self._reversed_terms = sorted((t[::-1], t) for t in self._terms.keys())
        self._reversed_titleterms = sorted(
            (t[::-1], t) for t in self._titleterms.keys()
        )

        self.logger.debug(
            f"Sphinx search index loaded {len(self._docnames)} docs, "
            f"{len(self._terms)} terms"
        )
        return True

    def _normalize_doc_ids(self, value: Any) -> List[int]:
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            return value
        return []

    def _find_prefix_matches(
        self, prefix: str, sorted_keys: List[str], term_dict: Dict[str, Any]
    ) -> List[Tuple[str, Any]]:
        """使用二分查找找到所有前缀匹配的词项"""
        if not prefix:
            return []
        left = bisect_left(sorted_keys, prefix)
        right_bound = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        right = bisect_left(sorted_keys, right_bound)
        return [
            (sorted_keys[i], term_dict[sorted_keys[i]])
            for i in range(left, right)
            if sorted_keys[i] != prefix
        ]

    def _find_suffix_matches(
        self,
        suffix: str,
        reversed_keys: List[Tuple[str, str]],
        term_dict: Dict[str, Any],
    ) -> List[Tuple[str, Any]]:
        """使用二分查找找到所有后缀匹配的词项"""
        if not suffix:
            return []
        reversed_suffix = suffix[::-1]
        left = bisect_left(reversed_keys, (reversed_suffix, ""))
        right_bound = reversed_suffix[:-1] + chr(ord(reversed_suffix[-1]) + 1)
        right = bisect_left(reversed_keys, (right_bound, ""))
        return [
            (reversed_keys[i][1], term_dict[reversed_keys[i][1]])
            for i in range(left, right)
            if reversed_keys[i][1] != suffix
        ]

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """搜索文档

        Args:
            query: 单个算子名称（已由调用方预处理）
            max_results: 最大返回结果数
        """
        if not self._load_index():
            return []

        term = query.lower().strip()
        if not term:
            return []

        doc_scores: Dict[int, float] = {}

        # 精确匹配（权重最高）
        if term in self._titleterms:
            for doc_id in self._normalize_doc_ids(self._titleterms[term]):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 10
        if term in self._terms:
            for doc_id in self._normalize_doc_ids(self._terms[term]):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

        # 前缀匹配（如 relu -> relu6）
        for _, doc_ids in self._find_prefix_matches(term, self._sorted_titleterms, self._titleterms):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 5
        for _, doc_ids in self._find_prefix_matches(term, self._sorted_terms, self._terms):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5

        # 后缀匹配（如 relu -> leaky_relu）
        for _, doc_ids in self._find_suffix_matches(term, self._reversed_titleterms, self._titleterms):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 3
        for _, doc_ids in self._find_suffix_matches(term, self._reversed_terms, self._terms):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.3

        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            normalized_score = min(score / 10.0, 1.0)
            if normalized_score < self.threshold:
                continue

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

            results.append({"title": title, "url": url, "relevance_score": normalized_score, "snippet": ""})
            if len(results) >= max_results:
                break

        return results
