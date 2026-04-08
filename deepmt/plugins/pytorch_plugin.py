"""
PyTorch 框架适配插件
"""

from typing import Any, Callable, Tuple

import numpy as np
import torch

from deepmt.plugins.framework_plugin import CompareResult, FrameworkPlugin


# ── 消息解析器 ────────────────────────────────────────────────────────────────


def _parse_assert_close_msg(msg: str) -> Tuple[float, float, int, int]:
    """
    解析 torch.testing.assert_close 的 AssertionError 消息，提取差值统计。
    仅扫描前 8 行（格式固定），使用 startswith + split，无正则。

    Tensor 格式（多元素张量）：
      Mismatched elements: 2 / 3 (66.7%)
      Greatest absolute difference: 2.0 at index (1,) (up to X allowed)
      Greatest relative difference: 1.0 at index (1,) (up to X allowed)

    Scalar 格式（0-d 张量）：
      Absolute difference: 1.0 (up to X allowed)
      Relative difference: 0.5 (up to X allowed)

    Returns:
        (max_abs_diff, max_rel_diff, mismatched_elements, total_elements)
        解析失败时对应字段返回 inf / 1。
    """
    max_abs_diff = float("inf")
    max_rel_diff = float("inf")
    mismatched_elements = 1
    total_elements = 1

    for line in msg.splitlines()[:8]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Mismatched elements:"):
            # "Mismatched elements: 2 / 3 (66.7%)"
            _, rest = line.split(":", 1)
            left, right = rest.split("/", 1)
            mismatched_elements = int(left.strip())
            total_elements = int(right.split()[0])
        elif line.startswith("Greatest absolute difference:"):
            _, rest = line.split(":", 1)
            max_abs_diff = float(rest.split()[0])
        elif line.startswith("Absolute difference:"):
            _, rest = line.split(":", 1)
            max_abs_diff = float(rest.split()[0])
        elif line.startswith("Greatest relative difference:"):
            _, rest = line.split(":", 1)
            max_rel_diff = float(rest.split()[0])
        elif line.startswith("Relative difference:"):
            _, rest = line.split(":", 1)
            max_rel_diff = float(rest.split()[0])

    return max_abs_diff, max_rel_diff, mismatched_elements, total_elements


# ── 插件 ──────────────────────────────────────────────────────────────────────


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

    def get_shape(self, tensor: Any) -> tuple:
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape)
        return np.asarray(tensor).shape

    def allclose(self, a: Any, b: Any, atol: float, rtol: float = 0.0) -> CompareResult:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                return CompareResult(
                    passed=False,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    mismatched_elements=0,
                    total_elements=0,
                )

            total_elements = a.numel()

            # 单次调用（atol=0, rtol=0）取真实 diff 统计；
            # 完全相等则直接通过，否则解析消息并用调用方指定的容差判定。
            try:
                torch.testing.assert_close(
                    a,
                    b,
                    atol=0,
                    rtol=0,
                    check_dtype=False,
                    equal_nan=False,
                )
                return CompareResult(
                    passed=True,
                    max_abs_diff=0.0,
                    max_rel_diff=0.0,
                    mismatched_elements=0,
                    total_elements=total_elements,
                )
            except AssertionError as e:
                max_abs_diff, max_rel_diff, mismatched_elements, total_elements = (
                    _parse_assert_close_msg(str(e))
                )
                # 判定条件：绝对差值在 atol 内，或相对差值在 rtol 内
                passed = max_abs_diff <= atol or max_rel_diff <= rtol
                return CompareResult(
                    passed=passed,
                    max_abs_diff=max_abs_diff,
                    max_rel_diff=max_rel_diff,
                    mismatched_elements=mismatched_elements,
                    total_elements=total_elements,
                )

        # ── 非张量输入回退至 numpy ─────────────────────────────────────────
        a_np = self.to_numpy(a).astype(float)
        b_np = self.to_numpy(b).astype(float)
        if a_np.shape != b_np.shape:
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )
        abs_b = np.abs(b_np)
        abs_diff = np.abs(a_np - b_np)
        max_abs_diff = float(abs_diff.max())
        max_rel_diff = float((abs_diff / np.maximum(abs_b, np.finfo(float).tiny)).max())
        threshold = atol + rtol * abs_b
        mismatched_mask = abs_diff > threshold
        return CompareResult(
            passed=bool(mismatched_mask.sum() == 0),
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            mismatched_elements=int(mismatched_mask.sum()),
            total_elements=a_np.size,
        )
