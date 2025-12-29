"""
算子信息获取器：自动从网络搜索获取算子的定义、代码、文档
"""

import re
from typing import Any, Dict, Optional

from core.config_loader import get_config
from core.logger import get_logger
from tools.web_search.search_tool import WebSearchTool


class OperatorInfoFetcher:
    """
    算子信息获取器（单例模式）

    功能：
    1. 从配置文件读取设置
    2. 使用WebSearchTool搜索算子信息
    3. 提取和整理算子代码、文档、签名
    """

    _instance: Optional["OperatorInfoFetcher"] = None
    _initialized = False

    def __new__(cls):
        """
        创建或获取OperatorInfoFetcher实例（单例模式）

        Returns:
            OperatorInfoFetcher实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化算子信息获取器

        配置由 core/config_loader.py 统一管理
        """
        # 如果已经初始化过，跳过
        if OperatorInfoFetcher._initialized:
            return

        self.logger = get_logger()
        # 使用统一的配置加载器
        self.config = get_config()

        # 初始化搜索工具（单例，配置由内部从 config_loader 获取）
        self.search_tool = WebSearchTool()

        web_search_config = self.config.get("web_search", {})
        self.enabled = web_search_config.get("enabled", True)
        OperatorInfoFetcher._initialized = True

    def fetch_operator_info(
        self, operator_name: str, framework: str = "pytorch", use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        获取算子信息

        Args:
            operator_name: 算子名称（如 "relu", "ReLU", "torch.nn.ReLU"）
            framework: 框架名称（默认pytorch）
            use_cache: 是否使用缓存

        Returns:
            算子信息字典，包含：
            - name: 算子名称
            - code: 代码字符串
            - doc: 文档字符串
            - signature: 函数签名
            - examples: 代码示例列表
            - source_urls: 来源URL列表
        """
        if not self.enabled:
            self.logger.warning("Web search is disabled in config")
            return {}

        self.logger.info(
            f"Fetching operator info for '{operator_name}' from {framework}"
        )

        # 搜索算子信息
        search_results = self.search_tool.search_operator(
            operator_name=operator_name,
            framework=framework,
            sources=self.config.get("web_search", {}).get("sources"),
        )

        if not search_results:
            self.logger.warning(f"No search results found for '{operator_name}'")
            return {}

        # 提取算子信息
        operator_info = self.search_tool.extract_operator_info(search_results)
        operator_info["name"] = operator_name

        # 尝试从文档中提取函数签名
        if not operator_info.get("signature"):
            operator_info["signature"] = self._extract_signature_from_doc(
                operator_info.get("doc", "")
            )

        self.logger.info(
            f"Fetched operator info: "
            f"code={bool(operator_info.get('code'))}, "
            f"doc={bool(operator_info.get('doc'))}, "
            f"signature={bool(operator_info.get('signature'))}"
        )

        return operator_info

    def _extract_signature_from_doc(self, doc: str) -> str:
        """
        从文档中提取函数签名

        Args:
            doc: 文档字符串

        Returns:
            函数签名字符串
        """
        # 查找函数签名模式
        patterns = [
            r"def\s+(\w+)\s*\([^)]*\)",  # def function_name(...)
            r"(\w+)\s*\([^)]*\)",  # function_name(...)
        ]

        for pattern in patterns:
            match = re.search(pattern, doc)
            if match:
                return match.group(0)

        return ""

    def get_operator_code(
        self, operator_name: str, framework: str = "pytorch"
    ) -> Optional[str]:
        """
        获取算子代码

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            代码字符串，如果未找到则返回None
        """
        info = self.fetch_operator_info(operator_name, framework)
        return info.get("code")

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
