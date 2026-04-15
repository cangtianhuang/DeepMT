"""deepmt.plugins — 框架适配器插件

注册表驱动加载（详见 plugins.yaml）：
  - PluginsManager 读取 plugins.yaml，按 name / module / class 三元组按需 import
  - 未安装的框架在实例化时抛 ImportError，被 PluginsManager 捕获并降级为 WARNING

本 __init__ 暴露声明式 PLUGIN_REGISTRY 供健康检查与测试反射使用。
每个条目描述一个"可登记到 plugins.yaml"的候选插件，含 `optional` 标志与可用性探测。
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from deepmt.plugins.numpy_plugin import NumpyPlugin
from deepmt.plugins.pytorch_plugin import PyTorchPlugin


@dataclass(frozen=True)
class PluginEntry:
    """插件登记表条目。"""

    name: str                 # 与 FrameworkType 对齐的框架短名
    module: str               # Python 模块路径
    class_name: str           # 插件类名
    optional: bool            # 框架依赖缺失时是否允许降级为 WARNING
    is_available: Callable[[], bool]  # 运行时探测函数

    def load_class(self) -> type:
        import importlib

        mod = importlib.import_module(self.module)
        return getattr(mod, self.class_name)


def _paddle_available() -> bool:
    try:
        from deepmt.plugins.paddle_plugin import PaddlePlugin

        return PaddlePlugin.is_available()
    except Exception:
        return False


def _tf_available() -> bool:
    try:
        from deepmt.plugins.tensorflow_plugin import TensorFlowPlugin

        return TensorFlowPlugin.is_available()
    except Exception:
        return False


PLUGIN_REGISTRY: List[PluginEntry] = [
    PluginEntry(
        name="pytorch",
        module="deepmt.plugins.pytorch_plugin",
        class_name="PyTorchPlugin",
        optional=False,
        is_available=lambda: True,
    ),
    PluginEntry(
        name="numpy",
        module="deepmt.plugins.numpy_plugin",
        class_name="NumpyPlugin",
        optional=False,
        is_available=lambda: True,
    ),
    PluginEntry(
        name="paddlepaddle",
        module="deepmt.plugins.paddle_plugin",
        class_name="PaddlePlugin",
        optional=True,
        is_available=_paddle_available,
    ),
    PluginEntry(
        name="tensorflow",
        module="deepmt.plugins.tensorflow_plugin",
        class_name="TensorFlowPlugin",
        optional=True,
        is_available=_tf_available,
    ),
]


__all__ = [
    "PyTorchPlugin",
    "NumpyPlugin",
    "PluginEntry",
    "PLUGIN_REGISTRY",
]
