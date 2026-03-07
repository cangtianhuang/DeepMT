"""
TaskRunner：任务执行入口，提供高层语义 API，管理文件缓存。

缓存策略：
- 结果以 JSON 文件存储在 data/agent_cache/ 目录
- 缓存键 = task_id + inputs 的 MD5
- 超过 cache_ttl_days 天后自动刷新

Phase 2 新增：
- sync_catalog(framework)：调用 agent 获取算子列表并合并写入 OperatorCatalog YAML
- get_operator_list() 返回结构与 OperatorCatalog 的期望格式对齐
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import get_config_value
from core.framework import FrameworkType
from core.logger import get_logger, log_structured
from mr_generator.base.operator_catalog import OperatorCatalog
from tools.agent.agent_core import CrawlAgent, TaskSpec

_DEFAULT_CACHE_DIR = Path("data/agent_cache")


class TaskRunner:
    """
    任务执行入口，对外暴露语义化的高层方法。

    所有方法均支持缓存，可通过 use_cache=False 强制刷新。
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        agent: Optional[CrawlAgent] = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        cache_dir_cfg = cache_dir or get_config_value("agent.cache_dir", str(_DEFAULT_CACHE_DIR))
        self.cache_dir = Path(cache_dir_cfg)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._agent: Optional[CrawlAgent] = agent

    @property
    def agent(self) -> CrawlAgent:
        """懒加载 CrawlAgent（首次使用时初始化，避免启动时加载 LLM）"""
        if self._agent is None:
            self._agent = CrawlAgent()
        return self._agent

    # ---------------------------------------------------------------------- #
    #  公开高层 API                                                            #
    # ---------------------------------------------------------------------- #

    def get_framework_version(
        self,
        framework: FrameworkType,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        获取框架最新稳定版本信息。

        Args:
            framework: 框架名称（pytorch / tensorflow / paddlepaddle）
            use_cache: 是否使用缓存（缓存有效期由任务规格的 cache_ttl_days 决定）

        Returns:
            {
                "version": "2.5.1",
                "release_date": "2024-10-01",
                "release_url": "https://..."
            }
        """
        return self.run_task(
            "get_framework_version",
            inputs={"framework": framework},
            use_cache=use_cache,
        )

    def get_operator_list(
        self,
        framework: FrameworkType,
        version: str = "latest",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        获取框架指定版本的算子目录。

        Args:
            framework: 框架名称
            version: 版本号，如 "2.5.1"；"latest" 表示自动获取最新版
            use_cache: 是否使用缓存

        Returns:
            {
                "operators": ["relu", "conv2d", ...],
                "doc_urls": {"relu": "https://...", ...},
                "total_count": 150
            }
        """
        resolved_version = version
        if version == "latest":
            ver_info = self.get_framework_version(framework, use_cache=use_cache)
            resolved_version = ver_info.get("version", "stable")

        return self.run_task(
            "get_operator_list",
            inputs={"framework": framework, "version": resolved_version},
            use_cache=use_cache,
        )

    def get_operator_doc(
        self,
        operator_name: str,
        framework: FrameworkType,
        use_cache: bool = True,
    ) -> str:
        """
        获取单个算子的文档内容。

        Args:
            operator_name: 算子名称（如 "relu"、"conv2d"）
            framework: 框架名称
            use_cache: 是否使用缓存

        Returns:
            算子文档的纯文本字符串
        """
        result = self.run_task(
            "get_operator_doc",
            inputs={"operator_name": operator_name, "framework": framework},
            use_cache=use_cache,
        )
        return result.get("doc", "")

    def sync_catalog(
        self,
        framework: FrameworkType,
        version: str = "latest",
        use_cache: bool = True,
        save_yaml: bool = True,
    ) -> Dict[str, Any]:
        """
        完整的目录同步流程：获取算子列表 → 合并到 OperatorCatalog → 持久化 YAML。

        Args:
            framework:  框架名称
            version:    目标版本，"latest" 表示自动获取最新版本
            use_cache:  是否复用缓存的 agent 结果
            save_yaml:  是否将合并结果写回 YAML 文件

        Returns:
            {
                "framework": "pytorch",
                "version": "2.5.1",
                "merge_stats": {"added": 5, "updated": 10, "skipped": 100},
                "total_operators": 115,
                "catalog_path": "/path/to/pytorch.yaml"  # 仅当 save_yaml=True
            }
        """
        # Step 1: 获取算子列表（自动解析版本）
        operator_result = self.get_operator_list(framework, version=version, use_cache=use_cache)
        resolved_version = operator_result.get("version", version)

        log_structured(
            self.logger,
            "SYNC",
            f"Syncing catalog for '{framework}' v{resolved_version}",
            operators_count=len(operator_result.get("operators", [])),
        )

        # Step 2: 合并到 OperatorCatalog
        catalog = OperatorCatalog()
        merge_stats = catalog.merge_from_agent_result(framework, operator_result)

        result: Dict[str, Any] = {
            "framework": framework,
            "version": resolved_version,
            "merge_stats": merge_stats,
            "total_operators": len(catalog.get_operator_names(framework)),
        }

        # Step 3: 持久化 YAML
        if save_yaml:
            saved_path = catalog.save_yaml(framework)
            result["catalog_path"] = str(saved_path)
            log_structured(
                self.logger,
                "SYNC",
                f"Catalog saved",
                path=str(saved_path),
                total=result["total_operators"],
            )

        return result

    def run_task(
        self,
        task_id: str,
        inputs: Dict[str, Any],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        通用任务执行入口。

        Args:
            task_id: 任务 ID（对应 tools/agent/tasks/<task_id>.yaml）
            inputs: 任务输入参数
            use_cache: 是否使用缓存

        Returns:
            任务执行结果字典
        """
        task_spec = TaskSpec.from_task_id(task_id)
        cache_key = _make_cache_key(task_id, inputs)

        # agent.cache_ttl_days 全局配置可覆盖 TaskSpec 的 cache_ttl_days
        global_ttl = get_config_value("agent.cache_ttl_days")
        ttl_days = int(global_ttl) if global_ttl is not None else task_spec.cache_ttl_days

        # 尝试读缓存
        if use_cache:
            cached = self._load_cache(cache_key, ttl_days)
            if cached is not None:
                log_structured(
                    self.logger,
                    "CACHE",
                    f"Cache hit for task '{task_id}'",
                    inputs=inputs,
                )
                return cached

        # 执行任务
        log_structured(
            self.logger,
            "TASK",
            f"Running task '{task_id}'",
            inputs=inputs,
        )
        result = self.agent.run(task_spec, inputs)

        # 写缓存
        self._save_cache(cache_key, result)
        return result

    # ---------------------------------------------------------------------- #
    #  缓存管理                                                                #
    # ---------------------------------------------------------------------- #

    def _load_cache(
        self,
        cache_key: str,
        ttl_days: int,
    ) -> Optional[Dict[str, Any]]:
        """读取缓存文件，若过期或不存在返回 None"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查 TTL
            saved_at_str = data.get("_saved_at")
            if saved_at_str:
                saved_at = datetime.fromisoformat(saved_at_str)
                if datetime.now() - saved_at > timedelta(days=ttl_days):
                    log_structured(
                        self.logger,
                        "CACHE",
                        f"Cache expired (ttl={ttl_days}d): {cache_file.name}",
                        level="DEBUG",
                    )
                    return None

            # 去除内部元数据字段后返回
            return {k: v for k, v in data.items() if not k.startswith("_")}

        except Exception as e:
            self.logger.warning(f"读取缓存失败 {cache_file}: {e}")
            return None

    def _save_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """将任务结果写入缓存文件"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            payload = {
                "_saved_at": datetime.now().isoformat(),
                **data,
            }
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log_structured(
                self.logger,
                "CACHE",
                f"Saved cache: {cache_file.name}",
                level="DEBUG",
            )
        except Exception as e:
            self.logger.warning(f"写入缓存失败 {cache_file}: {e}")

    def clear_cache(self, task_id: Optional[str] = None) -> int:
        """
        清除缓存文件。

        Args:
            task_id: 若指定，只清除该任务的缓存；若为 None，清除全部缓存

        Returns:
            删除的缓存文件数量
        """
        count = 0
        prefix = f"{task_id}_" if task_id else ""
        for cache_file in self.cache_dir.glob("*.json"):
            if prefix and not cache_file.name.startswith(prefix):
                continue
            cache_file.unlink()
            count += 1
        log_structured(
            self.logger,
            "CACHE",
            f"Cleared {count} cache file(s)",
            task_id=task_id or "all",
        )
        return count

    def list_cache(self) -> List[Dict[str, Any]]:
        """列出所有缓存条目的元信息"""
        entries = []
        for cache_file in sorted(self.cache_dir.glob("*.json")):
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(
                    {
                        "file": cache_file.name,
                        "saved_at": data.get("_saved_at", "unknown"),
                        "size_bytes": cache_file.stat().st_size,
                    }
                )
            except Exception:
                pass
        return entries


# --------------------------------------------------------------------------- #
#  辅助函数                                                                    #
# --------------------------------------------------------------------------- #

def _make_cache_key(task_id: str, inputs: Dict[str, Any]) -> str:
    """生成缓存键：task_id + inputs 的 MD5 前缀"""
    inputs_str = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(inputs_str.encode()).hexdigest()[:8]
    # 保留 task_id 前缀便于识别
    return f"{task_id}_{digest}"
