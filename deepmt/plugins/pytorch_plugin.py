"""
PyTorch 框架适配插件
"""

from typing import Any, Callable, Tuple

import numpy as np
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

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def allclose(self, a: Any, b: Any, atol: float) -> Tuple[bool, float]:
        a_np = self.to_numpy(a).astype(float)
        b_np = self.to_numpy(b).astype(float)
        if a_np.shape != b_np.shape:
            return False, float("inf")
        diff = np.abs(a_np - b_np)
        return bool(np.all(diff <= atol)), float(np.max(diff))
