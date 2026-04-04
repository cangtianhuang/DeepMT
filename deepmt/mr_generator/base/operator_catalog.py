"""
算子目录（Operator Catalog）

为 PyTorch、TensorFlow、PaddlePaddle 三个框架维护一份结构化的常用算子名称列表。
每个算子条目含框架版本信息（since / deprecated / removed），
支持按框架、版本、分类进行过滤查询。

目录数据来源：mr_generator/config/operator_catalog/<framework>.yaml

Phase 2 新增：
- OperatorEntry 新增 doc_url 字段
- OperatorCatalog 新增 merge_from_agent_result() / save_yaml() 写入接口
  用于将 CrawlAgent 抓取的算子列表自动合并到持久化 YAML
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from deepmt.core.framework import FRAMEWORK_ALIASES, SUPPORTED_FRAMEWORKS
from deepmt.core.logger import get_logger


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
        # Phase 2 新增：文档 URL（由 CrawlAgent 填充，可选）
        self.doc_url: str = data.get("doc_url", "")
        # 参数签名快照（可选，用于 check-updates 签名对比）
        # 示例："(in_features, out_features, bias=True)"
        self.signature: str = data.get("signature", "")

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
        d: Dict[str, Any] = {
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
        if self.doc_url:
            d["doc_url"] = self.doc_url
        if self.signature:
            d["signature"] = self.signature
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

    # ------------------------------------------------------------------
    # Phase 2：写入接口（供 CrawlAgent 结果合并使用）
    # ------------------------------------------------------------------

    def merge_from_agent_result(
        self,
        framework: str,
        agent_result: Dict[str, Any],
        overwrite_doc_url: bool = True,
    ) -> Dict[str, int]:
        """
        将 CrawlAgent 获取的算子列表合并到当前目录。

        agent_result 期望包含：
          - operators: list[str]          算子名称列表
          - doc_urls:  dict[str, str]     算子名 -> 文档 URL（可选）
          - version:   str                版本号（可选，用于设置 since 字段）

        合并规则：
          - 名称已存在的条目：不改变 category/since/aliases，可选更新 doc_url
          - 名称不存在的新条目：以 since="0.0"、category="" 创建占位条目

        Args:
            framework:        框架名称
            agent_result:     TaskRunner.get_operator_list() 的返回结果
            overwrite_doc_url: 是否用 agent 结果覆盖已有条目的 doc_url

        Returns:
            {"added": int, "updated": int, "skipped": int}
        """
        fw = self._normalize_framework(framework)
        existing_names = {e.name for e in self._entries.get(fw, [])}

        operators: List[str] = agent_result.get("operators", [])
        doc_urls: Dict[str, str] = agent_result.get("doc_urls", {})
        version: str = str(agent_result.get("version", "0.0"))

        stats = {"added": 0, "updated": 0, "skipped": 0}

        for op_name in operators:
            if not op_name or not isinstance(op_name, str):
                continue

            if op_name in existing_names:
                # 已存在：按需更新 doc_url
                if overwrite_doc_url and op_name in doc_urls:
                    for entry in self._entries[fw]:
                        if entry.name == op_name:
                            entry.doc_url = doc_urls[op_name]
                            break
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # 新条目：创建占位条目
                new_entry = OperatorEntry(
                    {
                        "name": op_name,
                        "category": "",
                        "since": version,
                        "doc_url": doc_urls.get(op_name, ""),
                    }
                )
                self._entries.setdefault(fw, []).append(new_entry)
                existing_names.add(op_name)
                stats["added"] += 1

        self.logger.info(
            f"Merged agent result into '{fw}' catalog: "
            f"added={stats['added']}, updated={stats['updated']}, skipped={stats['skipped']}"
        )
        return stats

    def save_yaml(self, framework: str) -> Path:
        """
        将指定框架的目录写回 YAML 文件（更新 last_updated）。

        Args:
            framework: 框架名称

        Returns:
            保存的文件路径
        """
        fw = self._normalize_framework(framework)
        filename = self._FRAMEWORK_FILES[fw]
        path = self._catalog_dir / f"{filename}.yaml"

        # 读取现有文件头部（description、注释等），保留 framework/description/注释行
        header_lines: List[str] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("operators:"):
                        break
                    # 跳过 last_updated 行，稍后重写
                    if line.startswith("last_updated:"):
                        continue
                    header_lines.append(line.rstrip())

        # 构造新的文件内容
        today = datetime.now().strftime("%Y-%m-%d")
        entries = self._entries.get(fw, [])

        with path.open("w", encoding="utf-8") as f:
            # 写入头部（framework/description/注释）
            for line in header_lines:
                f.write(line + "\n")
            # 插入更新日期
            f.write(f'last_updated: "{today}"\n')
            f.write("\n")
            f.write("operators:\n")
            # 写入所有条目
            for entry in entries:
                d = entry.to_dict()
                f.write(f"  - name: {d['name']}\n")
                if d.get("category"):
                    f.write(f"    category: {d['category']}\n")
                f.write(f"    since: \"{d['since']}\"\n")
                if d.get("deprecated"):
                    f.write(f"    deprecated: \"{d['deprecated']}\"\n")
                if d.get("removed"):
                    f.write(f"    removed: \"{d['removed']}\"\n")
                if d.get("aliases"):
                    aliases_str = "[" + ", ".join(d["aliases"]) + "]"
                    f.write(f"    aliases: {aliases_str}\n")
                if d.get("doc_url"):
                    f.write(f"    doc_url: {d['doc_url']}\n")
                if d.get("signature"):
                    f.write(f"    signature: \"{d['signature']}\"\n")
                if d.get("note"):
                    f.write(f"    note: \"{d['note']}\"\n")
                f.write("\n")

        self.logger.info(f"Saved {len(entries)} operators for '{fw}' to {path}")
        return path

    def get_doc_url(self, framework: str, operator_name: str) -> str:
        """
        获取指定算子的文档 URL。

        Args:
            framework:     框架名称
            operator_name: 算子名称或别名

        Returns:
            文档 URL 字符串，未找到时返回空字符串
        """
        entry = self.get_operator_info(framework, operator_name)
        if entry is None:
            return ""
        return entry.doc_url

    # ------------------------------------------------------------------
    # 排除列表与 diff（用于 check-updates）
    # ------------------------------------------------------------------

    def load_exclude_config(self, framework: str) -> Dict[str, Any]:
        """
        加载框架的 API 排除配置。

        Returns:
            {
                "excluded_namespaces": List[str],  # 命名空间前缀排除（须以 "." 分隔）
                "excluded_prefixes":   List[str],  # 任意前缀排除（startswith，无需 "." 分隔）
                "excluded_exact":      Set[str],   # 精确名称排除
            }
        """
        fw = self._normalize_framework(framework)
        path = self._catalog_dir / f"{fw}_exclude.yaml"
        if not path.exists():
            return {"excluded_namespaces": [], "excluded_prefixes": [], "excluded_exact": set()}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "excluded_namespaces": data.get("excluded_namespaces", []),
                "excluded_prefixes": data.get("excluded_prefixes", []),
                "excluded_exact": set(data.get("excluded_exact", [])),
            }
        except Exception as e:
            self.logger.warning(f"Failed to load exclude config for '{fw}': {e}")
            return {"excluded_namespaces": [], "excluded_prefixes": [], "excluded_exact": set()}

    def is_excluded(self, name: str, exclude_config: Dict[str, Any]) -> bool:
        """判断一个 API 名称是否被排除"""
        if name in exclude_config.get("excluded_exact", set()):
            return True
        for ns in exclude_config.get("excluded_namespaces", []):
            # 命名空间匹配：torch.cuda 排除 torch.cuda.amp.* 等，严格以 "." 分隔
            if name == ns or name.startswith(ns + "."):
                return True
        for prefix in exclude_config.get("excluded_prefixes", []):
            # 任意前缀匹配：torch._foreach_ 排除 torch._foreach_abs / torch._foreach_abs_ 等
            if name.startswith(prefix):
                return True
        return False

    def diff_with_fetched(
        self,
        framework: str,
        fetched_apis: List[Dict],
    ) -> Dict[str, List]:
        """
        将从官网提取的 API 列表与本地维护的目录/排除列表对比，生成差异报告。

        Args:
            framework:    框架名称
            fetched_apis: APIListFetcher.fetch_all_apis() 的返回结果

        Returns:
            {
                "changed_signature": [  # 在目录中，且存储了签名，但签名与当前不同
                    {"name": str, "stored_sig": str, "current_sig": str, "url": str}
                ],
                "no_stored_signature": [  # 在目录中，但未记录签名（首次检测）
                    {"name": str, "current_sig": str, "url": str}
                ],
                "not_found_in_docs": [  # 在目录中，但未出现在文档中（可能已改名/删除）
                    {"name": str}
                ],
                "new_uncategorized": [  # 既不在目录、也不在排除列表中的新 API
                    {"name": str, "type": str, "signature": str, "url": str}
                ],
            }
        """
        fw = self._normalize_framework(framework)
        catalog_entries = {e.name: e for e in self._entries.get(fw, [])}
        exclude_config = self.load_exclude_config(framework)

        # 以 name 为键建立 fetched 索引
        fetched_by_name = {api["name"]: api for api in fetched_apis}

        result: Dict[str, List] = {
            "changed_signature": [],
            "no_stored_signature": [],
            "not_found_in_docs": [],
            "new_uncategorized": [],
        }

        # 1. 扫描目录中的每个算子
        for name, entry in catalog_entries.items():
            if name in fetched_by_name:
                fetched = fetched_by_name[name]
                current_sig = fetched.get("signature", "")
                if entry.signature:
                    # 有存储签名：对比
                    if entry.signature != current_sig:
                        result["changed_signature"].append({
                            "name": name,
                            "stored_sig": entry.signature,
                            "current_sig": current_sig,
                            "url": fetched.get("url", ""),
                        })
                else:
                    # 无存储签名：记录当前签名供参考
                    result["no_stored_signature"].append({
                        "name": name,
                        "current_sig": current_sig,
                        "url": fetched.get("url", ""),
                    })
            else:
                # 目录中有，文档中未找到
                result["not_found_in_docs"].append({"name": name})

        # 2. 扫描 fetched 中不在目录也不在排除列表的 API
        for name, api in fetched_by_name.items():
            if name in catalog_entries:
                continue
            if self.is_excluded(name, exclude_config):
                continue
            # 也排除别名：如果 name 在某个 catalog 条目的 aliases 中
            is_alias = any(
                name in e.aliases
                for e in self._entries.get(fw, [])
            )
            if is_alias:
                continue
            result["new_uncategorized"].append({
                "name": name,
                "type": api.get("type", ""),
                "signature": api.get("signature", ""),
                "url": api.get("url", ""),
            })

        # 按名称排序，输出稳定
        for key in result:
            result[key].sort(key=lambda x: x["name"])

        return result
