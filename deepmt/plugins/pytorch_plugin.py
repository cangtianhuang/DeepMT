"""
PyTorch 框架适配插件
"""

from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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

    # 泛化短名覆盖表：处理根模块上不存在或需要指定到 nn.functional 的算子。
    # 键为泛化名（与 MR 知识库 subject_name 一致），值为可调用对象。
    # 对于直接存在于 torch.* 的算子（abs/exp/log 等），无需在此注册。
    _overrides: ClassVar[Dict[str, Callable]] = {
        "relu":         F.relu,
        "sigmoid":      F.sigmoid,
        "tanh":         F.tanh,
        "softmax":      F.softmax,
        "log_softmax":  F.log_softmax,
        "leaky_relu":   F.leaky_relu,
        "gelu":         F.gelu,
        "elu":          F.elu,
        "silu":         F.silu,
        "hardswish":    F.hardswish,
        "batch_norm":   F.batch_norm,
        "layer_norm":   F.layer_norm,
        "dropout":      F.dropout,
        "max_pool1d":   F.max_pool1d,
        "max_pool2d":   F.max_pool2d,
        "avg_pool1d":   F.avg_pool1d,
        "avg_pool2d":   F.avg_pool2d,
        "cross_entropy":F.cross_entropy,
        "mse_loss":     F.mse_loss,
    }

    _DTYPE_MAP: ClassVar[Dict[str, torch.dtype]] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32":   torch.int32,
        "int64":   torch.int64,
        "bool":    torch.bool,
    }

    _TORCH_OPS: ClassVar[Dict[str, Any]] = {
        "abs":  torch.abs,
        # oracle_expr 中常出现 exp(1) 这类标量调用，torch.exp 要求 Tensor，故做标量兼容
        "exp":  lambda x: torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))),
        "sqrt": torch.sqrt,
        "log":  torch.log,
        "sin":  torch.sin,
        "cos":  torch.cos,
        "all":  torch.all,
        "any":  torch.any,
        "sum":  torch.sum,
        "mean": torch.mean,
    }

    _CMP_FN: ClassVar[Dict[str, Any]] = {
        "!=": torch.ne,
        "<":  torch.lt,
        "<=": torch.le,
        ">":  torch.gt,
        ">=": torch.ge,
    }

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

    def make_tensor(
        self,
        shape: tuple,
        dtype: str,
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> torch.Tensor:
        """根据完全确定的 shape/dtype/value_range 创建随机 PyTorch 张量。"""
        th_dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        if th_dtype in (torch.int32, torch.int64):
            lo = int(value_range[0]) if value_range and value_range[0] is not None else -10
            hi = int(value_range[1]) if value_range and value_range[1] is not None else 10
            return torch.randint(lo, hi + 1, shape, dtype=th_dtype)
        lo_f = float(value_range[0]) if value_range and value_range[0] is not None else None
        hi_f = float(value_range[1]) if value_range and value_range[1] is not None else None
        if lo_f is not None and hi_f is not None:
            return torch.rand(shape, dtype=th_dtype) * (hi_f - lo_f) + lo_f
        return torch.randn(shape, dtype=th_dtype) * 10.0

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def get_shape(self, tensor: Any) -> tuple:
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape)
        return np.asarray(tensor).shape

    def allclose(self, a: Any, b: Any, atol: float, rtol: float = 1e-5) -> CompareResult:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            try:
                a, b = torch.broadcast_tensors(a, b)
            except RuntimeError:
                return CompareResult(
                    passed=False,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    mismatched_elements=0,
                    total_elements=0,
                )

            total_elements = a.numel()
            try:
                torch.testing.assert_close(
                    a, b, atol=0, rtol=0, check_dtype=False, equal_nan=False,
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
                passed = max_abs_diff <= atol or max_rel_diff <= rtol
                return CompareResult(
                    passed=passed,
                    max_abs_diff=max_abs_diff,
                    max_rel_diff=max_rel_diff,
                    mismatched_elements=mismatched_elements,
                    total_elements=total_elements,
                )

        # ── 非张量输入回退至 numpy ─────────────────────────────────────────
        a_np = self.to_numpy(a)
        b_np = self.to_numpy(b)
        try:
            a_np, b_np = np.broadcast_arrays(a_np.astype(float), b_np.astype(float))
        except ValueError:
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )
        abs_b = np.abs(b_np)
        abs_diff = np.abs(a_np - b_np)
        threshold = atol + rtol * abs_b
        mismatched_mask = abs_diff > threshold
        return CompareResult(
            passed=bool(mismatched_mask.sum() == 0),
            max_abs_diff=float(abs_diff.max()),
            max_rel_diff=float((abs_diff / np.maximum(abs_b, np.finfo(float).tiny)).max()),
            mismatched_elements=int(mismatched_mask.sum()),
            total_elements=a_np.size,
        )

    def eval_expr(self, expr: str, orig: Any, trans: Any, x: Any) -> Any:
        ns = {
            "__builtins__": {},
            "orig": orig,
            "trans": trans,
            "x": x,
            **self._TORCH_OPS,
        }
        result = eval(expr, ns)  # noqa: S307
        if not isinstance(result, torch.Tensor):
            dtype = orig.dtype if isinstance(orig, torch.Tensor) else torch.float32
            result = torch.tensor(result, dtype=dtype)
        return result

    def element_compare(self, a: Any, b: Any, op: str) -> CompareResult:
        fn = self._CMP_FN.get(op)
        if fn is None:
            raise ValueError(f"Unsupported comparison operator: {op!r}")
        a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
        b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
        try:
            mask = fn(a_t, b_t)
        except RuntimeError:
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )
        mismatched = int((~mask).sum())
        a_f = a_t.float()
        b_f = b_t.float()
        abs_diff = (a_f - b_f).abs()
        rel_diff = abs_diff / b_f.abs().clamp(min=torch.finfo(torch.float32).tiny)
        return CompareResult(
            passed=mismatched == 0,
            max_abs_diff=float(abs_diff.max()),
            max_rel_diff=float(rel_diff.max()),
            mismatched_elements=mismatched,
            total_elements=int(mask.numel()),
        )
