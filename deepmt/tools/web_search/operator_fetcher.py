"""算子信息获取器：自动从网络搜索获取算子的定义、代码、文档"""

from typing import Any, Dict, Optional

from deepmt.core.config_manager import get_config_value
from deepmt.core.plugins_manager import FrameworkType
from deepmt.core.logger import logger
from deepmt.tools.web_search.search_tool import WebSearchTool


class OperatorInfoFetcher:
    """算子信息获取器"""

    def __init__(self) -> None:
        """初始化实例属性"""
        self.search_tool = WebSearchTool()
        self.enabled = get_config_value("web_search.enabled", True)

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
            use_cache: 是否使用缓存（保留接口兼容性，暂未实现）

        Returns:
            算子信息字典，包含 name, doc, source_urls
        """
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
