"""
插件管理器：动态加载和管理框架适配插件
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict

from deepmt.core.logger import logger
from deepmt.plugins.framework_adapter import FrameworkAdapter


class PluginsManager:
    """插件管理器：负责加载和管理框架适配插件"""

    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Any] = {}
        self.framework_adapters: Dict[str, FrameworkAdapter] = {}

    def load_plugins(self):
        """动态加载plugins目录中的所有插件"""
        if not self.plugins_dir.exists():
            logger.debug(f"Plugins directory not found: {self.plugins_dir}")
            return

        logger.debug(f"Loading plugins from {self.plugins_dir}")

        plugin_files = list(self.plugins_dir.glob("*_plugin.py"))

        for plugin_file in plugin_files:
            try:
                module_name = f"plugins.{plugin_file.stem}"
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name.endswith("Plugin") and obj.__module__ == module_name:
                        plugin_name = self._extract_plugin_name(name)
                        plugin_instance = obj()
                        self.plugins[plugin_name] = plugin_instance
                        self.framework_adapters[plugin_name] = FrameworkAdapter(
                            framework=plugin_name
                        )
                        logger.debug(f"Loaded plugin: {plugin_name} ({name})")

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

        if self.plugins:
            logger.info(f"🚀 [INIT] {len(self.plugins)} plugin(s) loaded: {list(self.plugins.keys())}")
        else:
            logger.debug("Total plugins loaded: 0")

    def _extract_plugin_name(self, class_name: str) -> str:
        name = class_name.replace("Plugin", "").lower()
        name_mapping = {
            "pytorch": "pytorch",
            "tensorflow": "tensorflow",
            "tf": "tensorflow",
            "paddle": "paddle",
            "paddlepaddle": "paddle",
        }
        return name_mapping.get(name, name)

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

    def register_plugin(self, name: str, plugin_instance: Any):
        self.plugins[name.lower()] = plugin_instance
        logger.info(f"🚀 [INIT] Manually registered plugin: {name}")

    def list_plugins(self) -> list:
        return list(self.plugins.keys())
