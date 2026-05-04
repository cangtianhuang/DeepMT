"""算子信息获取器：自动从网络搜索获取算子的定义、代码、文档"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from deepmt.core.config_manager import get_config_value
from deepmt.core.plugins_manager import FrameworkType
from deepmt.core.logger import logger
from deepmt.tools.web_search.search_tool import WebSearchTool

# 文档缓存目录（data/cache/operator_docs/）
_CACHE_DIR = Path(__file__).parents[4] / "data" / "cache" / "operator_docs"


class OperatorInfoFetcher:
    """算子信息获取器"""

    def __init__(self) -> None:
        """初始化实例属性"""
        self.search_tool = WebSearchTool()
        self.enabled = get_config_value("web_search.enabled", True)

    # ── 缓存辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(operator_name: str, framework: str) -> str:
        raw = f"{framework}:{operator_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _cache_path(key: str) -> Path:
        return _CACHE_DIR / f"{key}.json"

    @staticmethod
    def _load_cache(key: str) -> Optional[Dict[str, Any]]:
        path = OperatorInfoFetcher._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _save_cache(key: str, info: Dict[str, Any]) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = OperatorInfoFetcher._cache_path(key)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"🔍 [CACHE] Failed to write cache: {e}")

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def fetch_operator_info(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        获取算子信息

        Args:
            operator_name: 算子名称（如 "relu", "ReLU", "torch.nn.ReLU"）
            framework: 框架名称（默认pytorch，支持pytorch/tensorflow/paddlepaddle）
            use_cache: 是否读写本地磁盘文档缓存（默认True；命中时跳过网络请求）

        Returns:
            算子信息字典，包含 name, doc, source_urls
        """
        fw_str = str(framework)

        # 尝试读缓存
        if use_cache:
            cache_key = self._cache_key(operator_name, fw_str)
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug(f"🔍 [CACHE] Hit cache for '{operator_name}' ({fw_str})")
                return cached

        if not self.enabled:
            logger.warning("⚠️ [WARN] Web search is disabled in config")
            return {"name": operator_name, "doc": "", "source_urls": []}

        try:
            search_results = self.search_tool.search_operator(
                operator_name=operator_name,
                framework=framework,
                sources=get_config_value("web_search.sources"),
            )
        except ValueError as e:
            logger.opt(exception=e).error("❌ " + f"Search failed for '{operator_name}'")
            return {"name": operator_name, "doc": "", "source_urls": []}

        if not search_results:
            logger.warning("⚠️ [WARN] " + f"No results found for '{operator_name}'")
            return {"name": operator_name, "doc": "", "source_urls": []}

        docs_results = [r for r in search_results if r.source == "docs"]
        source_results = docs_results if docs_results else search_results

        doc_parts = [r.snippet for r in source_results if r.snippet]
        source_urls = [r.url for r in source_results if r.url]
        doc = "\n\n".join(doc_parts)

        operator_info = {
            "name": operator_name,
            "doc": doc,
            "source_urls": source_urls,
        }

        logger.debug(f"🔍 [SEARCH] Found {len(source_urls)} sources | {len(doc)} chars")

        # 写缓存（仅在实际获取到内容时）
        if use_cache and doc:
            self._save_cache(cache_key, operator_info)
            logger.debug(f"🔍 [CACHE] Cached docs for '{operator_name}' ({fw_str})")

        return operator_info

    def get_operator_doc(
        self, operator_name: str, framework: FrameworkType = "pytorch"
    ) -> Optional[str]:
        """
        获取算子文档

        Args:
            operator_name: 算子名称
            framework: 框架名称（默认pytorch，支持pytorch/tensorflow/paddlepaddle）

        Returns:
            算子文档字符串，如果未找到则返回 None
        """
        return self.fetch_operator_info(operator_name, framework).get("doc")
