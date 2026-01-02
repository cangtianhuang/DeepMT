"""
算子信息获取器：自动从网络搜索获取算子的定义、代码、文档
"""

import re
from typing import Any, Dict, Optional

from core.config_loader import get_config_value
from core.logger import get_logger
from tools.web_search.search_tool import WebSearchTool


class OperatorInfoFetcher:
    """
    算子信息获取器（单例模式）

    功能：
    1. 从配置文件读取设置
    2. 使用 WebSearchTool 搜索算子信息
    3. 提取和整理算子代码、文档、签名
    """

    _instance: Optional["OperatorInfoFetcher"] = None

    def __new__(cls) -> "OperatorInfoFetcher":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self) -> None:
        """初始化实例属性（仅在首次创建时调用）"""
        self.logger = get_logger()
        # 初始化搜索工具（单例）
        self.search_tool = WebSearchTool()
        # 从配置加载器获取配置值
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
            算子信息字典，包含：
            - name: 算子名称
            - doc: 文档字符串（来自搜索结果，保留原始信息）
            - source_urls: 来源URL列表
        """
        if not self.enabled:
            self.logger.warning("Web search is disabled in config")
            return {}

        self.logger.info(
            f"Fetching operator info for '{operator_name}' from {framework}"
        )

        # 搜索算子信息
        try:
            search_results = self.search_tool.search_operator(
                operator_name=operator_name,
                framework=framework,
                sources=get_config_value("web_search.sources"),
            )
        except ValueError as e:
            # 输入错误（如拼写错误），记录并返回空结果
            self.logger.warning(f"Search failed: {e}")
            return {"name": operator_name, "doc": "", "source_urls": []}

        if not search_results:
            self.logger.warning(f"No search results found for '{operator_name}'")
            return {"name": operator_name, "doc": "", "source_urls": []}

        # 直接合并所有搜索结果的文档内容（保留原始信息）
        doc_parts = []
        source_urls = []

        # 优先使用 docs 源的结果
        docs_results = [r for r in search_results if r.source == "docs"]
        if docs_results:
            for result in docs_results:
                if result.snippet:
                    doc_parts.append(result.snippet)
                if result.url:
                    source_urls.append(result.url)

        # 如果没有 docs 结果，使用其他源的结果
        if not doc_parts:
            for result in search_results:
                if result.snippet:
                    doc_parts.append(result.snippet)
                if result.url:
                    source_urls.append(result.url)

        # 合并文档内容（保留所有信息，不进行理解）
        doc = "\n\n".join(doc_parts) if doc_parts else ""

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
        """
        获取算子文档

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            文档字符串，如果未找到则返回None
        """
        info = self.fetch_operator_info(operator_name, framework)
        return info.get("doc")
