"""
PaddlePaddle 框架适配插件

将 PyTorch 风格的算子名称映射到 PaddlePaddle 等价实现，
提供完整的 FrameworkPlugin 接口，可直接用于 BatchTestRunner 和 CrossFrameworkTester。

框架标识符（CLI / FrameworkType 中使用）："paddlepaddle" 或别名 "paddle"

算子映射策略：
  1. _overrides（_PADDLE_OPERATORS）：将 torch.* 名称映射到 paddle 等价实现（lambda with **kw）
  2. _root_modules = [paddle]：原生 paddle.* 名称通过属性链自动解析
  优先级：_overrides > _root_modules

安装检查：
  若 paddlepaddle 未安装，import 本模块仍可成功；
  但实例化 PaddlePlugin 时会抛出 ImportError 并给出安装提示。
"""

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from deepmt.plugins.framework_plugin import CompareResult, FrameworkPlugin

# ── 可用性检查 ────────────────────────────────────────────────────────────────

try:
    import paddle
    import paddle.nn.functional as _F

    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False
    paddle = None  # type: ignore[assignment]
    _F = None      # type: ignore[assignment]


def _require_paddle() -> None:
    """若 paddlepaddle 未安装则抛出 ImportError，给出安装提示。"""
    if not _PADDLE_AVAILABLE:
        raise ImportError(
            "paddlepaddle is not installed. "
            "Install it with:  uv pip install paddlepaddle\n"
            "Or for GPU:       uv pip install paddlepaddle-gpu"
        )


# ── PaddlePaddle 算子映射表（PyTorch 名称 → PaddlePaddle 实现） ───────────────
# 所有函数接受 **kwargs，其中 'input' 键为主输入张量（paddle.Tensor）。
# 此格式与 CrossFrameworkTester 和 MRPreChecker._build_kwargs 保持一致。

def _build_paddle_operators() -> Dict[str, Any]:
    """延迟构建算子映射表（paddle 可用时才调用）。"""
    return {
        # ── 激活函数 ─────────────────────────────────────────────────────────
        "torch.nn.functional.relu":
            lambda **kw: paddle.nn.functional.relu(kw["input"]),
        "torch.nn.functional.elu":
            lambda **kw: paddle.nn.functional.elu(kw["input"]),
        "torch.nn.functional.leaky_relu":
            lambda **kw: paddle.nn.functional.leaky_relu(kw["input"]),
        "torch.nn.functional.sigmoid":
            lambda **kw: paddle.nn.functional.sigmoid(kw["input"]),
        "torch.nn.functional.softmax":
            lambda **kw: paddle.nn.functional.softmax(kw["input"], axis=kw.get("dim", -1)),
        "torch.nn.functional.silu":
            lambda **kw: paddle.nn.functional.silu(kw["input"]),
        "torch.nn.functional.hardswish":
            lambda **kw: paddle.nn.functional.hardswish(kw["input"]),
        "torch.nn.functional.tanh":
            lambda **kw: paddle.nn.functional.tanh(kw["input"]),
        "torch.nn.functional.gelu":
            lambda **kw: paddle.nn.functional.gelu(kw["input"]),
        "torch.nn.functional.mish":
            lambda **kw: paddle.nn.functional.mish(kw["input"]),

        # ── 元素级数学算子 ────────────────────────────────────────────────────
        "torch.exp":        lambda **kw: paddle.exp(kw["input"]),
        "torch.log":        lambda **kw: paddle.log(kw["input"]),
        "torch.sqrt":       lambda **kw: paddle.sqrt(kw["input"]),
        "torch.abs":        lambda **kw: paddle.abs(kw["input"]),
        "torch.tanh":       lambda **kw: paddle.tanh(kw["input"]),
        "torch.sigmoid":    lambda **kw: paddle.nn.functional.sigmoid(kw["input"]),
        "torch.sin":        lambda **kw: paddle.sin(kw["input"]),
        "torch.cos":        lambda **kw: paddle.cos(kw["input"]),
        "torch.neg":        lambda **kw: paddle.neg(kw["input"]),
        "torch.reciprocal": lambda **kw: paddle.reciprocal(kw["input"]),
        "torch.sign":       lambda **kw: paddle.sign(kw["input"]),
        "torch.floor":      lambda **kw: paddle.floor(kw["input"]),
        "torch.ceil":       lambda **kw: paddle.ceil(kw["input"]),
        "torch.round":      lambda **kw: paddle.round(kw["input"]),
        "torch.log2":       lambda **kw: paddle.log2(kw["input"]),
        "torch.log10":      lambda **kw: paddle.log10(kw["input"]),

        # ── 二元算子 ──────────────────────────────────────────────────────────
        "torch.add":
            lambda **kw: paddle.add(kw["input"], kw.get("other", kw.get("arg1", paddle.zeros_like(kw["input"])))),
        "torch.mul":
            lambda **kw: paddle.multiply(kw["input"], kw.get("other", kw.get("arg1", paddle.ones_like(kw["input"])))),
        "torch.div":
            lambda **kw: paddle.divide(kw["input"], kw.get("other", kw.get("arg1", paddle.ones_like(kw["input"])))),
        "torch.pow":
            lambda **kw: paddle.pow(kw["input"], kw.get("exponent", kw.get("arg1", 2))),
        "torch.clamp":
            lambda **kw: paddle.clip(kw["input"], min=kw.get("min", None), max=kw.get("max", None)),
        "torch.maximum":
            lambda **kw: paddle.maximum(kw["input"], kw.get("other", kw.get("arg1", kw["input"]))),
        "torch.minimum":
            lambda **kw: paddle.minimum(kw["input"], kw.get("other", kw.get("arg1", kw["input"]))),
    }


# ── 跨框架算子等价性声明表（论文依据） ─────────────────────────────────────────
OPERATOR_EQUIVALENCE_MAP: Dict[str, str] = {
    "torch.nn.functional.relu":        "paddle.nn.functional.relu",
    "torch.nn.functional.elu":         "paddle.nn.functional.elu",
    "torch.nn.functional.leaky_relu":  "paddle.nn.functional.leaky_relu",
    "torch.nn.functional.sigmoid":     "paddle.nn.functional.sigmoid",
    "torch.nn.functional.softmax":     "paddle.nn.functional.softmax(axis=-1)",
    "torch.nn.functional.silu":        "paddle.nn.functional.silu",
    "torch.nn.functional.hardswish":   "paddle.nn.functional.hardswish",
    "torch.nn.functional.gelu":        "paddle.nn.functional.gelu",
    "torch.exp":                        "paddle.exp",
    "torch.log":                        "paddle.log",
    "torch.sqrt":                       "paddle.sqrt",
    "torch.abs":                        "paddle.abs",
    "torch.tanh":                       "paddle.tanh",
    "torch.sigmoid":                    "paddle.nn.functional.sigmoid",
    "torch.sin":                        "paddle.sin",
    "torch.cos":                        "paddle.cos",
    "torch.neg":                        "paddle.neg",
    "torch.reciprocal":                 "paddle.reciprocal",
    "torch.sign":                       "paddle.sign",
    "torch.floor":                      "paddle.floor",
    "torch.ceil":                       "paddle.ceil",
    "torch.round":                      "paddle.round",
    "torch.add":                        "paddle.add",
    "torch.mul":                        "paddle.multiply",
    "torch.div":                        "paddle.divide",
    "torch.pow":                        "paddle.pow",
    "torch.clamp":                      "paddle.clip",
}


# ── PaddlePlugin ──────────────────────────────────────────────────────────────


class PaddlePlugin(FrameworkPlugin):
    """
    PaddlePaddle 框架适配插件。

    支持两种算子名称格式：
      - PyTorch 名称（如 "torch.nn.functional.relu"）：通过 _PADDLE_OPERATORS 映射表解析
      - PaddlePaddle 原生名称（如 "paddle.exp"）：通过 _root_modules 属性链解析

    框架标识符（CLI 使用）："paddlepaddle" 或别名 "paddle"
    """

    _root_modules: ClassVar[list] = []   # 延迟初始化（paddle 可用时才赋值）
    _overrides: ClassVar[dict] = {}      # 延迟初始化

    _DTYPE_MAP: ClassVar[Dict[str, str]] = {
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
        "int32":   "int32",
        "int64":   "int64",
        "bool":    "bool",
    }

    _PADDLE_OPS: ClassVar[Dict[str, Any]] = {}   # 延迟初始化

    _CMP_FN: ClassVar[Dict[str, Any]] = {}       # 延迟初始化

    def __init__(self) -> None:
        _require_paddle()
        # 延迟初始化类属性（保证 import 本模块不触发 paddle 导入）
        if not PaddlePlugin._root_modules:
            PaddlePlugin._root_modules = [paddle]
        if not PaddlePlugin._overrides:
            PaddlePlugin._overrides = _build_paddle_operators()
        if not PaddlePlugin._PADDLE_OPS:
            PaddlePlugin._PADDLE_OPS = {
                "abs":  paddle.abs,
                "exp":  lambda x: paddle.exp(
                    x if isinstance(x, paddle.Tensor) else paddle.to_tensor(float(x))
                ),
                "sqrt": paddle.sqrt,
                "log":  paddle.log,
                "sin":  paddle.sin,
                "cos":  paddle.cos,
                "all":  paddle.all,
                "any":  paddle.any,
                "sum":  paddle.sum,
                "mean": paddle.mean,
            }
        if not PaddlePlugin._CMP_FN:
            PaddlePlugin._CMP_FN = {
                "!=": paddle.not_equal,
                "<":  paddle.less_than,
                "<=": paddle.less_equal,
                ">":  paddle.greater_than,
                ">=": paddle.greater_equal,
            }

    # ── 抽象方法实现 ───────────────────────────────────────────────────────────

    def _to_tensor(self, value: Any) -> "paddle.Tensor":
        if isinstance(value, paddle.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return paddle.to_tensor(value)
        if isinstance(value, (list, tuple)):
            return paddle.to_tensor(value, dtype="float32")
        if isinstance(value, (int, float)):
            return paddle.to_tensor([value], dtype="float32")
        return paddle.to_tensor(value)

    def _execute_operator(self, func: Callable, inputs: list) -> "paddle.Tensor":
        return func(*inputs)

    def make_tensor(
        self,
        shape: tuple,
        dtype: str,
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> "paddle.Tensor":
        """根据完全确定的 shape/dtype/value_range 创建随机 PaddlePaddle 张量。"""
        pd_dtype = self._DTYPE_MAP.get(dtype, "float32")
        if pd_dtype in ("int32", "int64"):
            lo = int(value_range[0]) if value_range and value_range[0] is not None else -10
            hi = int(value_range[1]) if value_range and value_range[1] is not None else 10
            return paddle.randint(lo, hi + 1, shape=list(shape), dtype=pd_dtype)
        lo_f = float(value_range[0]) if value_range and value_range[0] is not None else None
        hi_f = float(value_range[1]) if value_range and value_range[1] is not None else None
        if lo_f is not None and hi_f is not None:
            t = paddle.rand(shape=list(shape), dtype="float32") * (hi_f - lo_f) + lo_f
            return t.astype(pd_dtype)
        t = paddle.normal(mean=0.0, std=10.0, shape=list(shape))
        return t.astype(pd_dtype)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if isinstance(tensor, paddle.Tensor):
            return tensor.numpy()
        return np.asarray(tensor)

    def get_shape(self, tensor: Any) -> tuple:
        if isinstance(tensor, paddle.Tensor):
            return tuple(tensor.shape)
        return np.asarray(tensor).shape

    def allclose(self, a: Any, b: Any, atol: float, rtol: float = 1e-5) -> CompareResult:
        try:
            a_np = self.to_numpy(a).astype(float)
            b_np = self.to_numpy(b).astype(float)
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
            return CompareResult(
                passed=True, max_abs_diff=0.0, max_rel_diff=0.0,
                mismatched_elements=0, total_elements=0,
            )

        abs_diff = np.abs(a_np - b_np)
        abs_b = np.abs(b_np)
        rel_diff = abs_diff / (abs_b + np.finfo(float).tiny)
        mismatch_mask = abs_diff > (atol + rtol * abs_b)
        mismatched = int(np.sum(mismatch_mask))

        return CompareResult(
            passed=(mismatched == 0),
            max_abs_diff=float(abs_diff.max()),
            max_rel_diff=float(rel_diff.max()),
            mismatched_elements=mismatched,
            total_elements=total,
        )

    def eval_expr(self, expr: str, orig: Any, trans: Any, x: Any) -> "paddle.Tensor":
        """在 paddle 张量空间内对 oracle 子表达式求值。"""
        ns = {
            "__builtins__": {},
            "orig": orig if isinstance(orig, paddle.Tensor) else paddle.to_tensor(orig, dtype="float32"),
            "trans": trans if isinstance(trans, paddle.Tensor) else paddle.to_tensor(trans, dtype="float32"),
            "x": x if isinstance(x, paddle.Tensor) else paddle.to_tensor(x, dtype="float32"),
            **self._PADDLE_OPS,
        }
        result = eval(expr, ns)  # noqa: S307
        if not isinstance(result, paddle.Tensor):
            dtype = orig.dtype if isinstance(orig, paddle.Tensor) else "float32"
            result = paddle.to_tensor(result, dtype=str(dtype).split(".")[-1])
        return result

    def element_compare(self, a: Any, b: Any, op: str) -> CompareResult:
        fn = self._CMP_FN.get(op)
        if fn is None:
            raise ValueError(f"Unsupported comparison operator: {op!r}")

        a_t = a if isinstance(a, paddle.Tensor) else paddle.to_tensor(a, dtype="float32")
        b_t = b if isinstance(b, paddle.Tensor) else paddle.to_tensor(b, dtype="float32")

        try:
            mask = fn(a_t, b_t)
        except Exception:
            return CompareResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mismatched_elements=0,
                total_elements=0,
            )

        mismatched = int(paddle.sum(~mask).item())
        a_f = a_t.astype("float32")
        b_f = b_t.astype("float32")
        abs_diff = paddle.abs(a_f - b_f)
        tiny = float(np.finfo(np.float32).tiny)
        rel_diff = abs_diff / (paddle.abs(b_f) + tiny)

        return CompareResult(
            passed=(mismatched == 0),
            max_abs_diff=float(abs_diff.max().item()),
            max_rel_diff=float(rel_diff.max().item()),
            mismatched_elements=mismatched,
            total_elements=int(mask.numel()),
        )

    # ── 算子解析（覆盖基类，同时支持 torch.* 和 paddle.* 命名） ─────────────────

    def _resolve_operator(self, name: str) -> Callable:
        """
        算子名称解析，支持两种命名风格：
          1. torch.* 名称 → 查 _overrides（_PADDLE_OPERATORS）
          2. paddle.* 名称 → 通过 _root_modules 属性链解析
        """
        # 优先查 _overrides
        if name in self._overrides:
            return self._overrides[name]

        # 原生 paddle.* 名称通过属性链解析
        parts = name.split(".")
        if parts[0] == "paddle":
            try:
                obj = paddle
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                pass

        raise ValueError(
            f"PaddlePlugin: 算子 '{name}' 无对应实现。\n"
            f"  支持格式：torch.* 名称（通过映射表）或 paddle.* 原生名称。\n"
            f"  请在 paddle_plugin._PADDLE_OPERATORS 中添加 torch→paddle 映射，"
            f"或直接使用 paddle.* 命名。"
        )

    @staticmethod
    def supported_operators() -> List[str]:
        """返回支持 PyTorch→PaddlePaddle 跨框架对比的算子列表。"""
        if not _PADDLE_AVAILABLE:
            return []
        return list(_build_paddle_operators().keys())

    @staticmethod
    def is_available() -> bool:
        """返回 paddlepaddle 是否已安装。"""
        return _PADDLE_AVAILABLE
