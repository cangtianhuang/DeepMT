"""
PaddlePaddle 框架适配插件

将泛化算子名称映射到 PaddlePaddle 等价实现，
提供完整的 FrameworkPlugin 接口，可直接用于 BatchTestRunner 和 CrossFrameworkTester。

框架标识符（CLI / FrameworkType 中使用）："paddlepaddle" 或别名 "paddle"

算子映射策略：
  1. _overrides：将泛化短名（relu/exp 等）映射到 paddle 等价实现（lambda with **kw）
  2. _root_modules = [paddle]：原生 paddle.* 全路径名通过属性链自动解析
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


# ── PaddlePaddle 算子映射表（泛化名 → PaddlePaddle 实现） ─────────────────────
# 所有函数接受 **kwargs，其中 'input' 键为主输入张量（paddle.Tensor）。
# 此格式与 CrossFrameworkTester 和 MRPreChecker._build_kwargs 保持一致。
# 键为泛化短名（与 MR 知识库 subject_name 一致）。

def _build_paddle_operators() -> Dict[str, Any]:
    """延迟构建算子映射表（paddle 可用时才调用）。"""
    return {
        # ── 激活函数（泛化名） ────────────────────────────────────────────────
        "relu":
            lambda **kw: paddle.nn.functional.relu(kw["input"]),
        "elu":
            lambda **kw: paddle.nn.functional.elu(kw["input"]),
        "leaky_relu":
            lambda **kw: paddle.nn.functional.leaky_relu(kw["input"]),
        "sigmoid":
            lambda **kw: paddle.nn.functional.sigmoid(kw["input"]),
        "softmax":
            lambda **kw: paddle.nn.functional.softmax(kw["input"], axis=kw.get("dim", -1)),
        "silu":
            lambda **kw: paddle.nn.functional.silu(kw["input"]),
        "hardswish":
            lambda **kw: paddle.nn.functional.hardswish(kw["input"]),
        "tanh":
            lambda **kw: paddle.tanh(kw["input"]),
        "gelu":
            lambda **kw: paddle.nn.functional.gelu(kw["input"]),
        "mish":
            lambda **kw: paddle.nn.functional.mish(kw["input"]),

        # ── 元素级数学算子（泛化名） ──────────────────────────────────────────
        "exp":        lambda **kw: paddle.exp(kw["input"]),
        "log":        lambda **kw: paddle.log(kw["input"]),
        "sqrt":       lambda **kw: paddle.sqrt(kw["input"]),
        "abs":        lambda **kw: paddle.abs(kw["input"]),
        "sin":        lambda **kw: paddle.sin(kw["input"]),
        "cos":        lambda **kw: paddle.cos(kw["input"]),
        "neg":        lambda **kw: paddle.neg(kw["input"]),
        "reciprocal": lambda **kw: paddle.reciprocal(kw["input"]),
        "sign":       lambda **kw: paddle.sign(kw["input"]),
        "floor":      lambda **kw: paddle.floor(kw["input"]),
        "ceil":       lambda **kw: paddle.ceil(kw["input"]),
        "round":      lambda **kw: paddle.round(kw["input"]),
        "log2":       lambda **kw: paddle.log2(kw["input"]),
        "log10":      lambda **kw: paddle.log10(kw["input"]),
        "log1p":      lambda **kw: paddle.log1p(kw["input"]),
        "expm1":      lambda **kw: paddle.expm1(kw["input"]),
        "tan":        lambda **kw: paddle.tan(kw["input"]),
        "sinh":       lambda **kw: paddle.sinh(kw["input"]),
        "cosh":       lambda **kw: paddle.cosh(kw["input"]),
        "asin":       lambda **kw: paddle.asin(kw["input"]),
        "acos":       lambda **kw: paddle.acos(kw["input"]),
        "atan":       lambda **kw: paddle.atan(kw["input"]),
        "erf":        lambda **kw: paddle.erf(kw["input"]),
        "square":     lambda **kw: paddle.square(kw["input"]),

        # ── 归约 / 统计 ───────────────────────────────────────────────────────
        "sum":        lambda **kw: paddle.sum(kw["input"], axis=kw.get("dim", None)),
        "mean":       lambda **kw: paddle.mean(kw["input"], axis=kw.get("dim", None)),
        "var":        lambda **kw: paddle.var(kw["input"], axis=kw.get("dim", None), unbiased=kw.get("unbiased", True)),
        "std":        lambda **kw: paddle.std(kw["input"], axis=kw.get("dim", None), unbiased=kw.get("unbiased", True)),
        "cumsum":     lambda **kw: paddle.cumsum(kw["input"], axis=kw.get("dim", -1)),
        "logsumexp":  lambda **kw: paddle.logsumexp(kw["input"], axis=kw.get("dim", -1), keepdim=kw.get("keepdim", False)),

        # ── Softmax 族 ────────────────────────────────────────────────────────
        "log_softmax":
            lambda **kw: paddle.nn.functional.log_softmax(kw["input"], axis=kw.get("dim", -1)),

        # ── 归一化 ───────────────────────────────────────────────────────────
        "layer_norm":
            lambda **kw: paddle.nn.functional.layer_norm(
                kw["input"],
                normalized_shape=kw.get("normalized_shape", (kw["input"].shape[-1],)),
                weight=kw.get("weight", None),
                bias=kw.get("bias", None),
                epsilon=kw.get("eps", 1e-5),
            ),
        "batch_norm":
            lambda **kw: paddle.nn.functional.batch_norm(
                kw["input"],
                running_mean=kw.get("running_mean", paddle.zeros([kw["input"].shape[1]])),
                running_var=kw.get("running_var", paddle.ones([kw["input"].shape[1]])),
                weight=kw.get("weight", None),
                bias=kw.get("bias", None),
                training=kw.get("training", False),
                momentum=kw.get("momentum", 0.9),
                epsilon=kw.get("eps", 1e-5),
            ),

        # ── 二元算子（泛化名） ─────────────────────────────────────────────────
        "add":
            lambda **kw: paddle.add(kw["input"], kw.get("other", kw.get("arg1", paddle.zeros_like(kw["input"])))),
        "mul":
            lambda **kw: paddle.multiply(kw["input"], kw.get("other", kw.get("arg1", paddle.ones_like(kw["input"])))),
        "multiply":
            lambda **kw: paddle.multiply(kw["input"], kw.get("other", kw.get("arg1", paddle.ones_like(kw["input"])))),
        "div":
            lambda **kw: paddle.divide(kw["input"], kw.get("other", kw.get("arg1", paddle.ones_like(kw["input"])))),
        "pow":
            lambda **kw: paddle.pow(kw["input"], kw.get("exponent", kw.get("arg1", 2))),
        "clamp":
            lambda **kw: paddle.clip(kw["input"], min=kw.get("min", None), max=kw.get("max", None)),
        "maximum":
            lambda **kw: paddle.maximum(kw["input"], kw.get("other", kw.get("arg1", kw["input"]))),
        "minimum":
            lambda **kw: paddle.minimum(kw["input"], kw.get("other", kw.get("arg1", kw["input"]))),
    }


# ── 跨框架算子等价性声明表（论文依据） ─────────────────────────────────────────
# 键为泛化名（与 MR 知识库 subject_name 一致）
OPERATOR_EQUIVALENCE_MAP: Dict[str, str] = {
    "relu":        "paddle.nn.functional.relu",
    "elu":         "paddle.nn.functional.elu",
    "leaky_relu":  "paddle.nn.functional.leaky_relu",
    "sigmoid":     "paddle.nn.functional.sigmoid",
    "softmax":     "paddle.nn.functional.softmax(axis=-1)",
    "silu":        "paddle.nn.functional.silu",
    "hardswish":   "paddle.nn.functional.hardswish",
    "gelu":        "paddle.nn.functional.gelu",
    "exp":         "paddle.exp",
    "log":         "paddle.log",
    "sqrt":        "paddle.sqrt",
    "abs":         "paddle.abs",
    "tanh":        "paddle.tanh",
    "sin":         "paddle.sin",
    "cos":         "paddle.cos",
    "neg":         "paddle.neg",
    "reciprocal":  "paddle.reciprocal",
    "sign":        "paddle.sign",
    "floor":       "paddle.floor",
    "ceil":        "paddle.ceil",
    "round":       "paddle.round",
    "add":         "paddle.add",
    "mul":         "paddle.multiply",
    "div":         "paddle.divide",
    "pow":         "paddle.pow",
    "clamp":       "paddle.clip",
    "log1p":       "paddle.log1p",
    "expm1":       "paddle.expm1",
    "tan":         "paddle.tan",
    "sinh":        "paddle.sinh",
    "cosh":        "paddle.cosh",
    "asin":        "paddle.asin",
    "acos":        "paddle.acos",
    "atan":        "paddle.atan",
    "erf":         "paddle.erf",
    "square":      "paddle.square",
    "log_softmax": "paddle.nn.functional.log_softmax(axis=-1)",
    "logsumexp":   "paddle.logsumexp",
    "layer_norm":  "paddle.nn.functional.layer_norm",
    "batch_norm":  "paddle.nn.functional.batch_norm(training=False)",
    "var":         "paddle.var",
    "std":         "paddle.std",
    "cumsum":      "paddle.cumsum",
    "sum":         "paddle.sum",
    "mean":        "paddle.mean",
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

    @classmethod
    def framework_name(cls) -> str:
        return "paddlepaddle"

    @classmethod
    def framework_version(cls) -> str:
        if not _PADDLE_AVAILABLE:
            return "uninstalled"
        return paddle.__version__

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

    # ── 算子解析（覆盖基类，支持泛化短名和 paddle.* 全路径名） ────────────────

    def _resolve_operator(self, name: str) -> Callable:
        """
        算子名称解析，支持两种命名风格：
          1. 泛化短名（relu/exp 等） → 查 _overrides（映射到 paddle 实现）
          2. paddle.* 全路径名 → 通过 _root_modules 属性链解析
        """
        # 优先查 _overrides（泛化短名）
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
            f"  支持格式：泛化短名（通过映射表）或 paddle.* 原生全路径名。\n"
            f"  请在 paddle_plugin._build_paddle_operators 中添加泛化名→paddle 映射。"
        )

    @classmethod
    def supported_operators(cls) -> List[str]:
        """返回支持 PyTorch→PaddlePaddle 跨框架对比的算子列表。"""
        if not _PADDLE_AVAILABLE:
            return []
        return sorted(_build_paddle_operators().keys())

    @classmethod
    def is_available(cls) -> bool:
        """返回 paddlepaddle 是否已安装。"""
        return _PADDLE_AVAILABLE
