"""
PyTorch 框架适配插件
"""

from typing import Any, Callable

import torch

from deepmt.plugins.framework_plugin import FrameworkPlugin


class PyTorchPlugin(FrameworkPlugin):
    """PyTorch 框架适配插件"""

    _root_modules = [torch]
    _overrides: dict = {}

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
        if isinstance(value, (int, float)):
            return torch.tensor(value, dtype=torch.float32)
        return torch.tensor(value)

    def _execute_operator(self, func: Callable, inputs: list) -> torch.Tensor:
        return func(*inputs)
