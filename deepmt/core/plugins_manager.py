"""
插件管理器：从注册表（YAML）加载框架适配插件。
同时维护框架类型定义（FrameworkType、别名、支持列表）。
"""

import importlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from deepmt.core.logger import logger

# ── 框架类型定义 ──────────────────────────────────────────────────────────────

FrameworkType = Literal["pytorch", "tensorflow", "paddlepaddle"]

# 全部已知框架（含尚未实现插件的）
SUPPORTED_FRAMEWORKS: List[str] = ["pytorch", "tensorflow", "paddlepaddle"]

# 框架别名映射（用于 CLI 输入或用户传入的非规范名称）
FRAMEWORK_ALIASES: Dict[str, str] = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "paddlepaddle": "paddlepaddle",
    "paddle": "paddlepaddle",
}


class PluginsManager:
    """插件管理器：根据 plugins.yaml 注册表加载框架适配插件"""

    def __init__(self, plugins_dir: str = None):
        if plugins_dir is None:
            self.plugins_dir = Path(__file__).resolve().parent.parent / "plugins"
        else:
            self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Any] = {}

    def load_plugins(self):
        """从 plugins/plugins.yaml 注册表加载插件"""
        registry_file = self.plugins_dir / "plugins.yaml"
        if not registry_file.exists():
            logger.warning(f"Plugin registry not found: {registry_file}")
            return

        with open(registry_file) as f:
            config = yaml.safe_load(f)

        for entry in config.get("plugins", []):
            name = entry["name"]
            module_path = entry["module"]
            class_name = entry["class"]
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                plugin = cls()
                self.plugins[name] = plugin
                logger.debug(f"Loaded plugin: {name} ({class_name})")
            except Exception as e:
                logger.error(f"Failed to load plugin '{name}' from {module_path}: {e}")

        if self.plugins:
            logger.info(f"🚀 [INIT] {len(self.plugins)} plugin(s) loaded: {list(self.plugins.keys())}")
        else:
            logger.debug("Total plugins loaded: 0")

    def get_plugin(self, name: str):
        name = name.lower()
        if name not in self.plugins:
            available = ", ".join(self.plugins.keys())
            raise KeyError(f"Plugin '{name}' not found. Available plugins: {available}")
        return self.plugins[name]

    def list_plugins(self) -> list:
        return list(self.plugins.keys())

    @staticmethod
    def normalize_framework(framework: str) -> str:
        """将框架别名统一转为标准名称（e.g. "torch" -> "pytorch"）。

        Raises:
            ValueError: 若 framework 不在已知别名表中
        """
        normalized = FRAMEWORK_ALIASES.get(framework.lower(), framework.lower())
        if normalized not in SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework: {framework!r}. "
                f"Supported: {SUPPORTED_FRAMEWORKS}"
            )
        return normalized


# ── 进程级单例 ────────────────────────────────────────────────────────────────

_shared_plugins_manager: Optional[PluginsManager] = None


def get_plugins_manager() -> PluginsManager:
    global _shared_plugins_manager
    if _shared_plugins_manager is None:
        _shared_plugins_manager = PluginsManager()
        _shared_plugins_manager.load_plugins()
    return _shared_plugins_manager
