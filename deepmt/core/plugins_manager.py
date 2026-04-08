"""
插件管理器：从注册表（YAML）加载框架适配插件
"""

import importlib
from pathlib import Path
from typing import Any, Dict

import yaml

from deepmt.core.logger import logger
from deepmt.plugins.framework_adapter import FrameworkAdapter


class PluginsManager:
    """插件管理器：根据 plugins.yaml 注册表加载框架适配插件"""

    def __init__(self, plugins_dir: str = None):
        if plugins_dir is None:
            self.plugins_dir = Path(__file__).resolve().parent.parent / "plugins"
        else:
            self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Any] = {}
        self.framework_adapters: Dict[str, FrameworkAdapter] = {}

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
                self.framework_adapters[name] = FrameworkAdapter(plugin=plugin)
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

    def get_framework_adapter(self, framework: str) -> FrameworkAdapter:
        framework = framework.lower()
        if framework not in self.framework_adapters:
            available = ", ".join(self.framework_adapters.keys())
            raise KeyError(
                f"Framework adapter for '{framework}' not found. Available adapters: {available}"
            )
        return self.framework_adapters[framework]

    def list_plugins(self) -> list:
        return list(self.plugins.keys())
