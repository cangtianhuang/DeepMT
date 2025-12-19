"""
算子信息获取器：自动从网络搜索获取算子的定义、代码、文档
"""

import yaml
import os
import re
from typing import Dict, Any, Optional
from pathlib import Path

from mr_generator.web_search_tool import WebSearchTool
from core.logger import get_logger


class OperatorInfoFetcher:
    """
    算子信息获取器

    功能：
    1. 从配置文件读取设置
    2. 使用WebSearchTool搜索算子信息
    3. 提取和整理算子代码、文档、签名
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化算子信息获取器

        Args:
            config_path: 配置文件路径（如果为None则使用默认路径）
        """
        self.logger = get_logger()
        self.config = self._load_config(config_path)

        # 初始化搜索工具
        web_search_config = self.config.get("web_search", {})
        self.search_tool = WebSearchTool(
            timeout=web_search_config.get("timeout", 10),
            max_results_per_source=web_search_config.get("max_results", 5),
        )

        self.enabled = web_search_config.get("enabled", True)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

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
