"""
插件管理器：动态加载和管理框架适配插件
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict

from core.logger import get_logger, log_structured
from plugins.framework_adapter import FrameworkAdapter


class PluginsManager:
    """插件管理器：负责加载和管理框架适配插件"""

    def __init__(self, plugins_dir: str = "plugins"):
        """
        初始化插件管理器

        Args:
            plugins_dir: 插件目录路径
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Any] = {}
        self.framework_adapters: Dict[str, FrameworkAdapter] = {}
        self.logger = get_logger(self.__class__.__name__)

    def load_plugins(self):
        """动态加载plugins目录中的所有插件"""
        if not self.plugins_dir.exists():
            self.logger.debug(f"Plugins directory not found: {self.plugins_dir}")
            return

        self.logger.debug(f"Loading plugins from {self.plugins_dir}")

        # 查找所有插件文件
        plugin_files = list(self.plugins_dir.glob("*_plugin.py"))

        for plugin_file in plugin_files:
            try:
                # 构建模块名
                module_name = f"plugins.{plugin_file.stem}"

                # 动态导入模块
                module = importlib.import_module(module_name)

                # 查找插件类（以Plugin结尾的类）
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name.endswith("Plugin") and obj.__module__ == module_name:
                        # 获取插件名称（从类名提取，如 PyTorchPlugin -> pytorch）
                        plugin_name = self._extract_plugin_name(name)

                        # 实例化插件
                        plugin_instance = obj()
                        self.plugins[plugin_name] = plugin_instance

                        # 同时创建对应的框架适配器
                        self.framework_adapters[plugin_name] = FrameworkAdapter(
                            framework=plugin_name
                        )

                        self.logger.debug(f"Loaded plugin: {plugin_name} ({name})")

            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file}: {e}")

        if self.plugins:
            log_structured(
                self.logger,
                "INIT",
                f"{len(self.plugins)} plugin(s) loaded: {list(self.plugins.keys())}",
            )
        else:
            self.logger.debug("Total plugins loaded: 0")

    def _extract_plugin_name(self, class_name: str) -> str:
        """
        从类名提取插件名称
        例如: PyTorchPlugin -> pytorch, TensorFlowPlugin -> tensorflow
        """
        # 移除Plugin后缀
        name = class_name.replace("Plugin", "")

        # 转换为小写
        name = name.lower()

        # 处理特殊命名
        name_mapping = {
            "pytorch": "pytorch",
            "tensorflow": "tensorflow",
            "tf": "tensorflow",
            "paddle": "paddle",
            "paddlepaddle": "paddle",
        }

        return name_mapping.get(name, name)

    def get_plugin(self, name: str):
        """
        获取指定名称的插件

        Args:
            name: 插件名称（如 "pytorch", "tensorflow", "paddle"）

        Returns:
            插件实例

        Raises:
            KeyError: 如果插件不存在
        """
        name = name.lower()

        if name not in self.plugins:
            available = ", ".join(self.plugins.keys())
            raise KeyError(f"Plugin '{name}' not found. Available plugins: {available}")

        return self.plugins[name]

    def get_framework_adapter(self, framework: str) -> FrameworkAdapter:
        """
        获取指定框架的适配器

        Args:
            framework: 框架名称（如 "pytorch", "tensorflow", "paddle"）

        Returns:
            FrameworkAdapter 实例

        Raises:
            KeyError: 如果框架适配器不存在
        """
        framework = framework.lower()

        if framework not in self.framework_adapters:
            available = ", ".join(self.framework_adapters.keys())
            raise KeyError(
                f"Framework adapter for '{framework}' not found. Available adapters: {available}"
            )

        return self.framework_adapters[framework]

    def register_plugin(self, name: str, plugin_instance: Any):
        """
        手动注册插件

        Args:
            name: 插件名称
            plugin_instance: 插件实例
        """
        self.plugins[name.lower()] = plugin_instance
        log_structured(
            self.logger,
            "INIT",
            f"Manually registered plugin: {name}",
        )

    def list_plugins(self) -> list:
        """列出所有已加载的插件"""
        return list(self.plugins.keys())
