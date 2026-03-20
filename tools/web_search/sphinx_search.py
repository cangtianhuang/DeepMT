"""Sphinx 文档搜索索引解析器"""

import hashlib
import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

from core.logger import get_logger

# 缓存目录和过期时间（秒）
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 1 day


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
        self._reversed_terms: List[Tuple[str, str]] = (
            []
        )  # (reversed_term, original_term)
        self._reversed_titleterms: List[Tuple[str, str]] = []

    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        url_hash = hashlib.md5(self.index_url.encode()).hexdigest()[:16]
        return CACHE_DIR / f"{url_hash}.json"

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """从缓存加载索引数据"""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None

        # 检查缓存是否过期
        mtime = cache_path.stat().st_mtime
        if time.time() - mtime > CACHE_EXPIRY_SECONDS:
            self.logger.debug(f"Sphinx search index cache expired: {cache_path}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.debug(f"Sphinx search index loaded from cache: {cache_path}")
            return data
        except Exception as e:
            self.logger.debug(f"Sphinx search index failed to load from cache: {e}")
            return None

    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        """保存索引数据到缓存"""
        cache_path = self._get_cache_path()
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.logger.debug(f"Sphinx search index saved to cache: {cache_path}")
        except Exception as e:
            self.logger.debug(f"Sphinx search index failed to save to cache: {e}")

    def _load_index(self) -> bool:
        """下载并解析搜索索引（支持缓存）"""
        if self._index is not None:
            return True

        # 1. 尝试从缓存加载
        index_data = self._load_from_cache()

        # 2. 缓存不存在或过期，从网络下载
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
                # 保存到缓存
                self._save_to_cache(index_data)

            except Exception as e:
                self.logger.warning(f"Sphinx search index failed to load: {e}")
                return False

        # 3. 解析索引数据
        self._index = index_data
        self._docnames = index_data.get("docnames", [])
        self._titles = dict(enumerate(index_data.get("titles", [])))
        self._filenames = dict(enumerate(index_data.get("filenames", [])))
        self._terms = index_data.get("terms", {})
        self._titleterms = index_data.get("titleterms", {})

        # 预排序词项列表，用于 O(log n) 前缀匹配
        self._sorted_terms = sorted(self._terms.keys())
        self._sorted_titleterms = sorted(self._titleterms.keys())

        # 反向排序词项列表，用于后缀匹配
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
        """标准化文档 ID 列表"""
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
        # 找到第一个 >= prefix 的位置
        left = bisect_left(sorted_keys, prefix)
        # 找到第一个 >= prefix+1 的位置（即不再以 prefix 开头）
        right_bound = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        right = bisect_left(sorted_keys, right_bound)

        results = []
        for i in range(left, right):
            key = sorted_keys[i]
            if key != prefix:  # 排除精确匹配
                results.append((key, term_dict[key]))
        return results

    def _find_suffix_matches(
        self,
        suffix: str,
        reversed_keys: List[Tuple[str, str]],
        term_dict: Dict[str, Any],
    ) -> List[Tuple[str, Any]]:
        """使用二分查找找到所有后缀匹配的词项"""
        if not suffix:
            return []
        # 将后缀反转，在反向排序列表中进行前缀匹配
        reversed_suffix = suffix[::-1]
        left = bisect_left(reversed_keys, (reversed_suffix, ""))
        right_bound = reversed_suffix[:-1] + chr(ord(reversed_suffix[-1]) + 1)
        right = bisect_left(reversed_keys, (right_bound, ""))

        results = []
        for i in range(left, right):
            reversed_term, original_term = reversed_keys[i]
            if original_term != suffix:  # 排除精确匹配
                results.append((original_term, term_dict[original_term]))
        return results

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """搜索文档

        Args:
            query: 单个算子名称（已由调用方预处理）
            max_results: 最大返回结果数
        """
        if not self._load_index():
            return []

        # 查询已经是处理过的单个算子名，直接使用
        term = query.lower().strip()
        if not term:
            return []

        doc_scores: Dict[int, float] = {}

        # 1. 精确匹配（权重最高）
        if term in self._titleterms:
            for doc_id in self._normalize_doc_ids(self._titleterms[term]):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 10
        if term in self._terms:
            for doc_id in self._normalize_doc_ids(self._terms[term]):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

        # 2. 前缀匹配（如 relu -> relu6, relus）
        for _, doc_ids in self._find_prefix_matches(
            term, self._sorted_titleterms, self._titleterms
        ):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 5
        for _, doc_ids in self._find_prefix_matches(
            term, self._sorted_terms, self._terms
        ):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5

        # 3. 后缀匹配（如 relu -> leaky_relu, prelu）
        for _, doc_ids in self._find_suffix_matches(
            term, self._reversed_titleterms, self._titleterms
        ):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 3
        for _, doc_ids in self._find_suffix_matches(
            term, self._reversed_terms, self._terms
        ):
            for doc_id in self._normalize_doc_ids(doc_ids):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.3

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs:
            # 如果评分低于阈值，跳过
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

            results.append(
                {
                    "title": title,
                    "url": url,
                    "relevance_score": normalized_score,
                    "snippet": "",
                }
            )

            if len(results) >= max_results:
                break

        return results
