"""
TensorFlow 框架适配插件（Phase O MVP）

范围：
  - 仅覆盖 Phase O §4 任务 O4 定义的"一级 9 算子"
    (relu, tanh, exp, abs, sigmoid, gelu, softmax, log_softmax, leaky_relu)
  - 仅 CPU；不考虑 XLA / 分布式 / eager-vs-graph 差异
  - FrameworkPlugin 全部原语均按 PaddlePlugin 的模板实现

安装检查：
  与 paddle_plugin 采用相同懒加载模式：import 本模块时不要求 tensorflow 可用；
  实例化 TensorFlowPlugin 时若 TF 未安装，抛出带安装提示的 ImportError。

框架标识符（CLI / FrameworkType 中使用）："tensorflow" 或别名 "tf"
"""

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from deepmt.plugins.framework_plugin import CompareResult, FrameworkPlugin

# ── 可用性检查 ────────────────────────────────────────────────────────────────

try:
    import tensorflow as tf  # type: ignore[import-not-found]

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    tf = None  # type: ignore[assignment]


def _require_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "tensorflow is not installed. "
            "Install with:  uv pip install tensorflow-cpu\n"
            "Phase O MVP 仅验证过 tensorflow-cpu 2.15+；GPU/XLA 不在范围内。"
        )


# ── 算子映射表 ────────────────────────────────────────────────────────────────


def _build_tf_operators() -> Dict[str, Any]:
    """延迟构建算子映射表（tf 可用时才调用）。"""
    return {
        # Phase O MVP 一级算子
        "relu":        lambda **kw: tf.nn.relu(kw["input"]),
        "tanh":        lambda **kw: tf.nn.tanh(kw["input"]),
        "exp":         lambda **kw: tf.math.exp(kw["input"]),
        "abs":         lambda **kw: tf.math.abs(kw["input"]),
        "sigmoid":     lambda **kw: tf.nn.sigmoid(kw["input"]),
        "gelu":        lambda **kw: tf.nn.gelu(kw["input"], approximate=False),
        "softmax":     lambda **kw: tf.nn.softmax(kw["input"], axis=kw.get("dim", -1)),
        "log_softmax": lambda **kw: tf.nn.log_softmax(kw["input"], axis=kw.get("dim", -1)),
        "leaky_relu":  lambda **kw: tf.nn.leaky_relu(
            kw["input"], alpha=kw.get("negative_slope", 0.01)
        ),
        # 常用元素级数学（可选，为 oracle_expr eval 提供原料）
        "log":         lambda **kw: tf.math.log(kw["input"]),
        "sqrt":        lambda **kw: tf.math.sqrt(kw["input"]),
        "sin":         lambda **kw: tf.math.sin(kw["input"]),
        "cos":         lambda **kw: tf.math.cos(kw["input"]),
        "neg":         lambda **kw: tf.math.negative(kw["input"]),
    }


# 跨框架算子等价性声明表（论文依据）
OPERATOR_EQUIVALENCE_MAP: Dict[str, str] = {
    "relu":        "tf.nn.relu",
    "tanh":        "tf.nn.tanh",
    "exp":         "tf.math.exp",
    "abs":         "tf.math.abs",
    "sigmoid":     "tf.nn.sigmoid",
    "gelu":        "tf.nn.gelu(approximate=False)",
    "softmax":     "tf.nn.softmax(axis=-1)",
    "log_softmax": "tf.nn.log_softmax(axis=-1)",
    "leaky_relu":  "tf.nn.leaky_relu",
}


# ── TensorFlowPlugin ──────────────────────────────────────────────────────────


class TensorFlowPlugin(FrameworkPlugin):
    """
    TensorFlow 框架适配插件（Phase O MVP）。

    仅覆盖一级 9 算子；其他算子抛 OperatorNotMapped（通过基类 ValueError 传播）。
    """

    _root_modules: ClassVar[list] = []   # 延迟
    _overrides: ClassVar[dict] = {}      # 延迟

    _DTYPE_MAP: ClassVar[Dict[str, str]] = {
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
        "int32":   "int32",
        "int64":   "int64",
        "bool":    "bool",
    }

    _TF_OPS: ClassVar[Dict[str, Any]] = {}   # 延迟
    _CMP_FN: ClassVar[Dict[str, Any]] = {}   # 延迟

    @classmethod
    def framework_name(cls) -> str:
        return "tensorflow"

    @classmethod
    def framework_version(cls) -> str:
        if not _TF_AVAILABLE:
            return "uninstalled"
        return tf.__version__

    @classmethod
    def is_available(cls) -> bool:
        return _TF_AVAILABLE

    @classmethod
    def supported_operators(cls) -> List[str]:
        if not _TF_AVAILABLE:
            return []
        return sorted(_build_tf_operators().keys())

    def __init__(self) -> None:
        _require_tf()
        if not TensorFlowPlugin._root_modules:
            TensorFlowPlugin._root_modules = [tf]
        if not TensorFlowPlugin._overrides:
            TensorFlowPlugin._overrides = _build_tf_operators()
        if not TensorFlowPlugin._TF_OPS:
            TensorFlowPlugin._TF_OPS = {
                "abs":  tf.math.abs,
                "exp":  lambda x: tf.math.exp(
                    x if isinstance(x, tf.Tensor) else tf.constant(float(x), dtype=tf.float32)
                ),
                "sqrt": tf.math.sqrt,
                "log":  tf.math.log,
                "sin":  tf.math.sin,
                "cos":  tf.math.cos,
                "all":  tf.reduce_all,
                "any":  tf.reduce_any,
                "sum":  tf.reduce_sum,
                "mean": tf.reduce_mean,
            }
        if not TensorFlowPlugin._CMP_FN:
            TensorFlowPlugin._CMP_FN = {
                "!=": tf.not_equal,
                "<":  tf.less,
                "<=": tf.less_equal,
                ">":  tf.greater,
                ">=": tf.greater_equal,
            }

    # ── 抽象方法实现 ───────────────────────────────────────────────────────────

    def _to_tensor(self, value: Any) -> "tf.Tensor":
        if isinstance(value, tf.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return tf.convert_to_tensor(value)
        if isinstance(value, (list, tuple)):
            return tf.constant(value, dtype=tf.float32)
        if isinstance(value, (int, float)):
            return tf.constant(float(value), dtype=tf.float32)
        return tf.convert_to_tensor(value)

    def _execute_operator(self, func: Callable, inputs: list) -> "tf.Tensor":
        return func(*inputs)

    def make_tensor(
        self,
        shape: tuple,
        dtype: str,
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> "tf.Tensor":
        tf_dtype_str = self._DTYPE_MAP.get(dtype, "float32")
        if tf_dtype_str in ("int32", "int64"):
            lo = int(value_range[0]) if value_range and value_range[0] is not None else -10
            hi = int(value_range[1]) if value_range and value_range[1] is not None else 10
            return tf.random.uniform(
                shape=list(shape), minval=lo, maxval=hi + 1, dtype=getattr(tf, tf_dtype_str)
            )
        lo_f = float(value_range[0]) if value_range and value_range[0] is not None else None
        hi_f = float(value_range[1]) if value_range and value_range[1] is not None else None
        if lo_f is not None and hi_f is not None:
            return tf.cast(
                tf.random.uniform(shape=list(shape), minval=lo_f, maxval=hi_f),
                getattr(tf, tf_dtype_str),
            )
        return tf.cast(
            tf.random.normal(shape=list(shape), mean=0.0, stddev=10.0),
            getattr(tf, tf_dtype_str),
        )

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        return np.asarray(tensor)

    def get_shape(self, tensor: Any) -> tuple:
        if isinstance(tensor, tf.Tensor):
            return tuple(tensor.shape.as_list())
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

    def eval_expr(self, expr: str, orig: Any, trans: Any, x: Any) -> "tf.Tensor":
        ns = {
            "__builtins__": {},
            "orig": orig if isinstance(orig, tf.Tensor) else tf.constant(orig, dtype=tf.float32),
            "trans": trans if isinstance(trans, tf.Tensor) else tf.constant(trans, dtype=tf.float32),
            "x": x if isinstance(x, tf.Tensor) else tf.constant(x, dtype=tf.float32),
            **self._TF_OPS,
        }
        result = eval(expr, ns)  # noqa: S307
        if not isinstance(result, tf.Tensor):
            dtype = orig.dtype if isinstance(orig, tf.Tensor) else tf.float32
            result = tf.constant(result, dtype=dtype)
        return result

    def element_compare(self, a: Any, b: Any, op: str) -> CompareResult:
        fn = self._CMP_FN.get(op)
        if fn is None:
            raise ValueError(f"Unsupported comparison operator: {op!r}")
        a_t = a if isinstance(a, tf.Tensor) else tf.constant(a, dtype=tf.float32)
        b_t = b if isinstance(b, tf.Tensor) else tf.constant(b, dtype=tf.float32)
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
        mask_np = mask.numpy()
        mismatched = int(np.sum(~mask_np))
        a_np = self.to_numpy(a_t).astype(float)
        b_np = self.to_numpy(b_t).astype(float)
        abs_diff = np.abs(a_np - b_np)
        rel_diff = abs_diff / (np.abs(b_np) + np.finfo(float).tiny)
        return CompareResult(
            passed=(mismatched == 0),
            max_abs_diff=float(abs_diff.max()),
            max_rel_diff=float(rel_diff.max()),
            mismatched_elements=mismatched,
            total_elements=int(mask_np.size),
        )

    # ── 算子解析（覆盖基类） ──────────────────────────────────────────────────

    def _resolve_operator(self, name: str) -> Callable:
        if name in self._overrides:
            return self._overrides[name]

        parts = name.split(".")
        if parts[0] in ("tensorflow", "tf"):
            try:
                obj = tf
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                pass

        raise ValueError(
            f"TensorFlowPlugin: operator '{name}' is not mapped.\n"
            f"  Add a mapping in tensorflow_plugin._build_tf_operators."
        )
