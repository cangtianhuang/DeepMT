"""
NumPy 参考后端插件（用于 D6 跨框架一致性实验）

设计目的：
  为 CrossFrameworkTester 提供第二个框架后端，让系统能在"单机无第二框架安装"的情况下
  完成跨框架一致性测试。NumPy 可作为"数学参考实现"：
    - PyTorch float32 vs NumPy float32 的数值差异反映实现细节差异
    - NumPy float64 vs PyTorch float32 的差异反映精度差异
    - 与 FaultyPlugin 结合时，可检验故意引入的缺陷是否在 numpy 参考实现中不存在

注意：
  此插件不加入 plugins.yaml 主注册表（避免影响正常批量测试流程），
  由 CrossFrameworkTester._get_backend("numpy") 直接实例化。

算子映射表（OPERATOR_EQUIVALENCE_MAP）：
  记录 PyTorch 算子名称 → NumPy 等价实现的映射关系，
  可作为论文中"跨框架算子等价性声明"的依据。
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

from deepmt.plugins.framework_plugin import CompareResult, FrameworkPlugin

# ── NumPy 算子实现 ─────────────────────────────────────────────────────────────
# 所有函数接受 **kwargs，其中 'input' 键为主输入张量（numpy ndarray）。
# 对 BatchTestRunner 生成的 kwargs = {"input": arr} 格式完全兼容。


def _np_softmax(input: np.ndarray, dim: int = -1, **kw) -> np.ndarray:
    """数值稳定的 softmax（沿 dim 轴）。"""
    shifted = input - input.max(axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (exp_x.sum(axis=dim, keepdims=True) + 1e-12)


_NUMPY_OPERATORS: Dict[str, Any] = {
    # 激活函数
    "torch.nn.functional.relu":        lambda **kw: np.maximum(0.0, kw["input"]),
    "torch.nn.functional.elu":         lambda **kw: np.where(kw["input"] >= 0, kw["input"], np.exp(kw["input"]) - 1.0),
    "torch.nn.functional.leaky_relu":  lambda **kw: np.where(kw["input"] >= 0, kw["input"], 0.01 * kw["input"]),
    "torch.nn.functional.sigmoid":     lambda **kw: 1.0 / (1.0 + np.exp(-kw["input"])),
    "torch.nn.functional.softmax":     lambda **kw: _np_softmax(**kw),
    "torch.nn.functional.silu":        lambda **kw: kw["input"] * (1.0 / (1.0 + np.exp(-kw["input"]))),
    "torch.nn.functional.hardswish":   lambda **kw: kw["input"] * np.clip(kw["input"] + 3.0, 0.0, 6.0) / 6.0,
    # 元素级数学
    "torch.exp":        lambda **kw: np.exp(kw["input"]),
    "torch.log":        lambda **kw: np.log(kw["input"]),
    "torch.sqrt":       lambda **kw: np.sqrt(kw["input"]),
    "torch.abs":        lambda **kw: np.abs(kw["input"]),
    "torch.tanh":       lambda **kw: np.tanh(kw["input"]),
    "torch.sigmoid":    lambda **kw: 1.0 / (1.0 + np.exp(-kw["input"])),
    "torch.sin":        lambda **kw: np.sin(kw["input"]),
    "torch.cos":        lambda **kw: np.cos(kw["input"]),
    "torch.neg":        lambda **kw: -kw["input"],
    "torch.reciprocal": lambda **kw: 1.0 / kw["input"],
    "torch.sign":       lambda **kw: np.sign(kw["input"]),
    "torch.floor":      lambda **kw: np.floor(kw["input"]),
    "torch.ceil":       lambda **kw: np.ceil(kw["input"]),
    "torch.round":      lambda **kw: np.round(kw["input"]),
    # 二元算子（第二操作数从 kwargs["other"] 或 kwargs["arg1"] 读取）
    "torch.add":  lambda **kw: kw["input"] + kw.get("other", kw.get("arg1", 0)),
    "torch.mul":  lambda **kw: kw["input"] * kw.get("other", kw.get("arg1", 1)),
    "torch.div":  lambda **kw: kw["input"] / kw.get("other", kw.get("arg1", 1)),
    "torch.pow":  lambda **kw: kw["input"] ** kw.get("exponent", kw.get("arg1", 2)),
    "torch.clamp": lambda **kw: np.clip(kw["input"], kw.get("min", None), kw.get("max", None)),
}

# ── 跨框架算子等价性声明表（论文依据） ─────────────────────────────────────────
OPERATOR_EQUIVALENCE_MAP: Dict[str, str] = {
    "torch.nn.functional.relu":    "np.maximum(0, x)",
    "torch.exp":                   "np.exp(x)",
    "torch.log":                   "np.log(x)",
    "torch.abs":                   "np.abs(x)",
    "torch.sqrt":                  "np.sqrt(x)",
    "torch.tanh":                  "np.tanh(x)",
    "torch.nn.functional.sigmoid": "1 / (1 + np.exp(-x))",
    "torch.nn.functional.softmax": "stable softmax along dim=-1",
    "torch.sin":                   "np.sin(x)",
    "torch.cos":                   "np.cos(x)",
    "torch.neg":                   "-x",
    "torch.reciprocal":            "1 / x",
}


# ── NumpyPlugin ────────────────────────────────────────────────────────────────


class NumpyPlugin(FrameworkPlugin):
    """
    NumPy 参考后端插件。

    将 PyTorch 风格的算子名称映射到 NumPy 等价实现，
    提供完整的 FrameworkPlugin 接口，可直接用于 BatchTestRunner 和 CrossFrameworkTester。

    框架标识符（CLI / FrameworkType 中使用）："numpy"
    """

    _root_modules = []            # 不走属性链解析，全用 _overrides
    _overrides = _NUMPY_OPERATORS

    _DTYPE_MAP: Dict[str, Any] = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32":   np.int32,
        "int64":   np.int64,
        "bool":    np.bool_,
    }

    # oracle_expr 中允许使用的数学函数（映射到 numpy）
    _OPS: Dict[str, Any] = {
        "abs":  np.abs,
        "exp":  np.exp,
        "sqrt": np.sqrt,
        "log":  np.log,
        "sin":  np.sin,
        "cos":  np.cos,
        "all":  np.all,
        "any":  np.any,
        "sum":  np.sum,
        "mean": np.mean,
    }

    _CMP_FN: Dict[str, Any] = {
        "!=": np.not_equal,
        "<":  np.less,
        "<=": np.less_equal,
        ">":  np.greater,
        ">=": np.greater_equal,
    }

    # ── 抽象方法实现 ──────────────────────────────────────────────────────────

    def _to_tensor(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    def _execute_operator(self, func, inputs: list) -> np.ndarray:
        return func(*inputs)

    def make_tensor(
        self,
        shape: tuple,
        dtype: str = "float32",
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> np.ndarray:
        """生成指定形状与范围的随机 NumPy 数组。"""
        np_dtype = self._DTYPE_MAP.get(dtype, np.float32)
        if np_dtype in (np.int32, np.int64):
            lo = int(value_range[0]) if value_range and value_range[0] is not None else -10
            hi = int(value_range[1]) if value_range and value_range[1] is not None else 10
            return np.random.randint(lo, hi + 1, size=shape).astype(np_dtype)
        lo_f = float(value_range[0]) if value_range and value_range[0] is not None else None
        hi_f = float(value_range[1]) if value_range and value_range[1] is not None else None
        if lo_f is not None and hi_f is not None:
            return (np.random.rand(*shape) * (hi_f - lo_f) + lo_f).astype(np_dtype)
        return (np.random.randn(*shape) * 10.0).astype(np_dtype)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return np.asarray(tensor, dtype=float)

    def get_shape(self, tensor: Any) -> tuple:
        return tuple(np.asarray(tensor).shape)

    def allclose(self, a: Any, b: Any, atol: float, rtol: float = 1e-5) -> CompareResult:
        try:
            a_np = np.asarray(a, dtype=float)
            b_np = np.asarray(b, dtype=float)
            a_np, b_np = np.broadcast_arrays(a_np, b_np)
        except (ValueError, TypeError):
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )

        total = a_np.size
        if total == 0:
            return CompareResult(passed=True, max_abs_diff=0.0, max_rel_diff=0.0,
                                 mismatched_elements=0, total_elements=0)

        abs_diff = np.abs(a_np - b_np)
        rel_diff = abs_diff / (np.abs(b_np) + 1e-12)
        mismatch_mask = abs_diff > (atol + rtol * np.abs(b_np))
        mismatched = int(np.sum(mismatch_mask))

        return CompareResult(
            passed=(mismatched == 0),
            max_abs_diff=float(abs_diff.max()),
            max_rel_diff=float(rel_diff.max()),
            mismatched_elements=mismatched,
            total_elements=total,
        )

    def eval_expr(self, expr: str, orig: Any, trans: Any, x: Any) -> np.ndarray:
        """在 numpy 命名空间中对 oracle 子表达式求值。"""
        ns = {
            "__builtins__": {},
            "orig": np.asarray(orig, dtype=float),
            "trans": np.asarray(trans, dtype=float),
            "x": np.asarray(x, dtype=float),
            **self._OPS,
        }
        result = eval(expr, ns)  # noqa: S307
        return np.asarray(result, dtype=np.float32)

    def element_compare(self, a: Any, b: Any, op: str) -> CompareResult:
        """逐元素不等式比较，返回完整统计。"""
        try:
            a_np = np.asarray(a, dtype=float)
            b_np = np.asarray(b, dtype=float)
            a_np, b_np = np.broadcast_arrays(a_np, b_np)
        except (ValueError, TypeError):
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )

        fn = self._CMP_FN.get(op, np.equal)
        sat_mask = fn(a_np, b_np)
        total = sat_mask.size
        mismatched = int(np.sum(~sat_mask))
        abs_diff = np.abs(a_np - b_np)

        return CompareResult(
            passed=(mismatched == 0),
            max_abs_diff=float(abs_diff.max()) if total > 0 else 0.0,
            max_rel_diff=0.0,
            mismatched_elements=mismatched,
            total_elements=total,
        )

    # ── 算子解析（覆盖基类） ──────────────────────────────────────────────────

    def _resolve_operator(self, name: str):
        """
        解析算子名称 → NumPy 等价函数。

        先查 _NUMPY_OPERATORS 映射表，未命中时抛出 ValueError（不支持该算子的 NumPy 等价）。
        """
        fn = _NUMPY_OPERATORS.get(name)
        if fn is None:
            raise ValueError(
                f"NumpyPlugin: 算子 '{name}' 无对应 NumPy 实现。"
                f"请在 numpy_plugin._NUMPY_OPERATORS 中添加映射。"
            )
        return fn

    @staticmethod
    def supported_operators() -> list:
        """返回当前支持跨框架对比的算子列表。"""
        return list(_NUMPY_OPERATORS.keys())
