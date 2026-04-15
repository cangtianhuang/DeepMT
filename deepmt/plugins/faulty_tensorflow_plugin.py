"""
含预设缺陷的 TensorFlow 插件（Phase O MVP 骨架）。

与 FaultyPyTorchPlugin 对称；当前仅作为 Phase O 架构对等占位：
  - 继承 TensorFlowPlugin，覆盖 _resolve_operator
  - 只支持 identity / negate 两类最轻量变异
  - 完整缺陷目录等 TensorFlow 正式接入真实扫描后再按 Phase M 流程扩充

使用方式与 FaultyPyTorchPlugin 保持一致：
    plugin = FaultyTensorFlowPlugin(fault_specs={"tf.nn.relu": "negate"})

若 tensorflow 未安装，实例化时抛出 ImportError（来自基类 __init__ 的 _require_tf）。
"""

from typing import Any, Callable, Dict, Optional

from deepmt.plugins.tensorflow_plugin import TensorFlowPlugin, _require_tf


def _apply_mutation(func: Callable, mutant: str) -> Callable:
    """包装算子函数以应用变异。"""
    if mutant == "identity":
        def wrapper(**kw):
            return kw.get("input")
        return wrapper
    if mutant == "negate":
        def wrapper(**kw):
            out = func(**kw)
            return -out
        return wrapper
    raise ValueError(
        f"FaultyTensorFlowPlugin: unsupported mutant {mutant!r}. "
        f"Supported: 'identity', 'negate'."
    )


class FaultyTensorFlowPlugin(TensorFlowPlugin):
    """TensorFlow 缺陷注入插件（骨架）。"""

    # 最小内置目录，仅用于契约回归
    BUILTIN_FAULT_CATALOG: Dict[str, str] = {}

    def __init__(self, fault_specs: Optional[Dict[str, str]] = None) -> None:
        _require_tf()
        super().__init__()
        self.fault_specs: Dict[str, str] = dict(fault_specs or {})

    def _resolve_operator(self, name: str) -> Callable:
        func = super()._resolve_operator(name)
        mutant = self.fault_specs.get(name)
        if mutant is None:
            return func
        return _apply_mutation(func, mutant)

    @classmethod
    def framework_name(cls) -> str:
        return "faulty_tensorflow"
