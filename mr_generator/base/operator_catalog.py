"""
算子目录（Operator Catalog）

为 PyTorch、TensorFlow、PaddlePaddle 三个框架维护一份结构化的常用算子名称列表。
每个算子条目含框架版本信息（since / deprecated / removed），
支持按框架、版本、分类进行过滤查询。

目录数据来源：mr_generator/config/operator_catalog/<framework>.yaml
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.framework import FRAMEWORK_ALIASES, SUPPORTED_FRAMEWORKS
from core.logger import get_logger


def _parse_version(version_str: str) -> tuple:
    """将版本字符串解析为可比较的整数元组，例如 "2.1" -> (2, 1)"""
    try:
        return tuple(int(x) for x in str(version_str).split("."))
    except (ValueError, AttributeError):
        return (0,)


def _version_ge(v: str, baseline: str) -> bool:
    """v >= baseline"""
    return _parse_version(v) >= _parse_version(baseline)


def _version_lt(v: str, limit: str) -> bool:
    """v < limit"""
    return _parse_version(v) < _parse_version(limit)


class OperatorEntry:
    """单个算子条目（对应 YAML 中的一个 operators 列表项）"""

    def __init__(self, data: dict):
        self.name: str = data["name"]
        self.category: str = data.get("category", "")
        self.since: str = str(data.get("since", "0.0"))
        self.deprecated: Optional[str] = (
            str(data["deprecated"]) if "deprecated" in data else None
        )
        self.removed: Optional[str] = (
            str(data["removed"]) if "removed" in data else None
        )
        self.aliases: List[str] = data.get("aliases") or []
        self.note: str = data.get("note", "")

    def is_available_in(self, version: str) -> bool:
        """
        判断该算子在指定版本是否可用。

        规则：
          - version >= since（已引入）
          - removed 未设置，或 version < removed（未被移除）
        """
        if not _version_ge(version, self.since):
            return False
        if self.removed and not _version_lt(version, self.removed):
            return False
        return True

    def is_deprecated_in(self, version: str) -> bool:
        """判断该算子在指定版本是否被标记为废弃"""
        if self.deprecated is None:
            return False
        return _version_ge(version, self.deprecated)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "category": self.category,
            "since": self.since,
        }
        if self.deprecated:
            d["deprecated"] = self.deprecated
        if self.removed:
            d["removed"] = self.removed
        if self.aliases:
            d["aliases"] = self.aliases
        if self.note:
            d["note"] = self.note
        return d

    def __repr__(self) -> str:
        return f"<OperatorEntry name={self.name!r} since={self.since!r}>"


class OperatorCatalog:
    """
    算子目录管理器

    从 YAML 配置文件加载三个框架的算子列表，提供查询接口。

    基本用法：
        catalog = OperatorCatalog()

        # 获取 PyTorch 2.1 中所有可用算子名称
        names = catalog.get_operator_names("pytorch", version="2.1")

        # 获取 TensorFlow 2.0 的激活函数算子
        act_ops = catalog.get_by_category("tensorflow", "activation", version="2.0")

        # 查询某算子的详细信息
        info = catalog.get_operator_info("pytorch", "torch.nn.ReLU")

        # 判断某算子在指定版本是否可用
        ok = catalog.is_available("paddlepaddle", "paddle.nn.ReLU", version="2.0")
    """

    # 目录 YAML 文件所在目录（相对于本文件的两级父目录）
    _CATALOG_DIR = Path(__file__).parent.parent / "config" / "operator_catalog"

    # 框架名称 -> YAML 文件名（不含后缀）
    _FRAMEWORK_FILES: Dict[str, str] = {
        "pytorch": "pytorch",
        "tensorflow": "tensorflow",
        "paddlepaddle": "paddlepaddle",
    }

    def __init__(self, catalog_dir: Optional[str] = None):
        """
        初始化算子目录。

        Args:
            catalog_dir: 自定义 YAML 目录路径；为 None 时使用默认路径。
        """
        self.logger = get_logger(self.__class__.__name__)
        self._catalog_dir = Path(catalog_dir) if catalog_dir else self._CATALOG_DIR

        # framework -> List[OperatorEntry]
        self._entries: Dict[str, List[OperatorEntry]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # 内部加载方法
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for framework, filename in self._FRAMEWORK_FILES.items():
            path = self._catalog_dir / f"{filename}.yaml"
            entries = self._load_yaml(framework, path)
            self._entries[framework] = entries
            self.logger.debug(
                f"Loaded {len(entries)} operators for '{framework}' from {path}"
            )

    def _load_yaml(self, framework: str, path: Path) -> List[OperatorEntry]:
        if not path.exists():
            self.logger.warning(
                f"Operator catalog file not found for '{framework}': {path}"
            )
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            raw_list = data.get("operators", []) or []
            return [OperatorEntry(item) for item in raw_list if "name" in item]
        except Exception as e:
            self.logger.error(
                f"Failed to load operator catalog for '{framework}': {e}"
            )
            return []

    # ------------------------------------------------------------------
    # 框架名称标准化
    # ------------------------------------------------------------------

    def _normalize_framework(self, framework: str) -> str:
        """将别名统一转换为标准框架名称（e.g. "torch" -> "pytorch"）"""
        normalized = FRAMEWORK_ALIASES.get(framework.lower(), framework.lower())
        if normalized not in self._FRAMEWORK_FILES:
            raise ValueError(
                f"Unsupported framework: {framework!r}. "
                f"Supported: {SUPPORTED_FRAMEWORKS}"
            )
        return normalized

    # ------------------------------------------------------------------
    # 公开查询接口
    # ------------------------------------------------------------------

    def get_all_frameworks(self) -> List[str]:
        """返回目录中已加载的所有框架名称列表"""
        return list(self._FRAMEWORK_FILES.keys())

    def get_all_entries(
        self,
        framework: str,
        version: Optional[str] = None,
        include_deprecated: bool = True,
    ) -> List[OperatorEntry]:
        """
        返回指定框架的 OperatorEntry 列表。

        Args:
            framework:          框架名称（支持别名，如 "torch"）
            version:            若提供，则只返回在该版本可用的算子；None 表示返回全部
            include_deprecated: 是否包含已废弃的算子（默认 True）

        Returns:
            OperatorEntry 列表
        """
        fw = self._normalize_framework(framework)
        entries = self._entries.get(fw, [])

        if version is not None:
            entries = [e for e in entries if e.is_available_in(version)]
            if not include_deprecated:
                entries = [e for e in entries if not e.is_deprecated_in(version)]

        return entries

    def get_operator_names(
        self,
        framework: str,
        version: Optional[str] = None,
        include_deprecated: bool = True,
    ) -> List[str]:
        """
        返回算子名称（name 字段）列表。

        Args:
            framework:          框架名称
            version:            版本过滤（None 表示不过滤）
            include_deprecated: 是否包含已废弃的算子

        Returns:
            算子名称字符串列表
        """
        return [
            e.name
            for e in self.get_all_entries(framework, version, include_deprecated)
        ]

    def get_by_category(
        self,
        framework: str,
        category: str,
        version: Optional[str] = None,
    ) -> List[OperatorEntry]:
        """
        按分类过滤算子。

        Args:
            framework: 框架名称
            category:  算子分类（如 "activation"、"normalization"）
            version:   版本过滤（None 表示不过滤）

        Returns:
            符合分类条件的 OperatorEntry 列表
        """
        return [
            e
            for e in self.get_all_entries(framework, version)
            if e.category == category
        ]

    def get_operator_info(
        self, framework: str, operator_name: str
    ) -> Optional[OperatorEntry]:
        """
        按算子名称或别名查找 OperatorEntry。

        Args:
            framework:     框架名称
            operator_name: 算子名称或别名

        Returns:
            匹配的 OperatorEntry，未找到时返回 None
        """
        fw = self._normalize_framework(framework)
        for entry in self._entries.get(fw, []):
            if entry.name == operator_name or operator_name in entry.aliases:
                return entry
        return None

    def is_available(
        self, framework: str, operator_name: str, version: str
    ) -> bool:
        """
        判断指定算子在给定版本是否可用（已引入且未被移除）。

        Args:
            framework:     框架名称
            operator_name: 算子名称或别名
            version:       框架版本字符串（如 "2.1"）

        Returns:
            True 表示可用，False 表示不可用或未在目录中
        """
        entry = self.get_operator_info(framework, operator_name)
        if entry is None:
            return False
        return entry.is_available_in(version)

    def get_categories(self, framework: str) -> List[str]:
        """返回指定框架目录中所有出现过的算子分类（去重、有序）"""
        fw = self._normalize_framework(framework)
        seen = []
        for e in self._entries.get(fw, []):
            if e.category and e.category not in seen:
                seen.append(e.category)
        return seen

    def summary(self, framework: Optional[str] = None) -> Dict[str, int]:
        """
        返回各框架的算子数量摘要。

        Args:
            framework: 若指定则只返回该框架；None 表示所有框架

        Returns:
            { framework_name: operator_count } 字典
        """
        if framework is not None:
            fw = self._normalize_framework(framework)
            return {fw: len(self._entries.get(fw, []))}
        return {fw: len(entries) for fw, entries in self._entries.items()}

    def reload(self) -> None:
        """从磁盘重新加载所有 YAML 文件（用于开发期热更新）"""
        self._entries.clear()
        self._load_all()
        self.logger.info("Operator catalog reloaded.")
