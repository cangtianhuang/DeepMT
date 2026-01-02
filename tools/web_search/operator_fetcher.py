"""算子信息获取器：自动从网络搜索获取算子的定义、代码、文档"""

import re
from typing import Any, Dict, Optional

from core.config_loader import get_config_value
from core.logger import get_logger
from tools.web_search.search_tool import WebSearchTool


class OperatorInfoFetcher:
    """算子信息获取器（单例模式）"""

    _instance: Optional["OperatorInfoFetcher"] = None

    def __new__(cls) -> "OperatorInfoFetcher":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self) -> None:
        """初始化实例属性（仅在首次创建时调用）"""
        self.logger = get_logger()
        self.search_tool = WebSearchTool()
        self.enabled = get_config_value("web_search.enabled", True)

    def fetch_operator_info(
        self, operator_name: str, framework: str = "pytorch", use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        获取算子信息（简化为直接返回文档内容）

        Args:
            operator_name: 算子名称（如 "relu", "ReLU", "torch.nn.ReLU"）
            framework: 框架名称（默认pytorch）
            use_cache: 是否使用缓存（保留接口兼容性，暂未实现）

        Returns:
            算子信息字典，包含 name, doc, source_urls
        """
        if not self.enabled:
            self.logger.warning("Web search is disabled in config")
            return {}

        self.logger.info(
            f"Fetching operator info for '{operator_name}' from {framework}"
        )

        try:
            search_results = self.search_tool.search_operator(
                operator_name=operator_name,
                framework=framework,
                sources=get_config_value("web_search.sources"),
            )
        except ValueError as e:
            self.logger.warning(f"Search failed: {e}")
            return {"name": operator_name, "doc": "", "source_urls": []}

        if not search_results:
            self.logger.warning(f"No search results found for '{operator_name}'")
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

        self.logger.info(
            f"Fetched operator info: "
            f"doc={'found' if doc else 'not found'} ({len(doc)} chars), "
            f"source_urls={len(source_urls)}"
        )

        return operator_info

    def get_operator_doc(
        self, operator_name: str, framework: str = "pytorch"
    ) -> Optional[str]:
        """获取算子文档"""
        return self.fetch_operator_info(operator_name, framework).get("doc")
