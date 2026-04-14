"""deepmt.plugins — 框架适配器插件"""

from deepmt.plugins.pytorch_plugin import PyTorchPlugin
from deepmt.plugins.numpy_plugin import NumpyPlugin

__all__ = [
    "PyTorchPlugin",
    "NumpyPlugin",
]
