"""
Phase O · O7 — 跨插件对等性集成测试

对 Phase O MVP 的一级 9 算子，在 pytorch / numpy / paddlepaddle 三框架插件之间
执行同一输入 → 比对输出形状与数值（atol=1e-4）。

本测试不依赖 MR 知识库，直接调用插件原语。
"""

import numpy as np
import pytest

from deepmt.plugins import PLUGIN_REGISTRY

LEVEL_1_OPERATORS = [
    "relu", "tanh", "exp", "abs", "sigmoid", "gelu",
    "softmax", "log_softmax", "leaky_relu",
]


def _load_plugins():
    loaded = {}
    for entry in PLUGIN_REGISTRY:
        if entry.name not in ("pytorch", "numpy", "paddlepaddle"):
            continue
        if not entry.is_available():
            continue
        try:
            loaded[entry.name] = entry.load_class()()
        except Exception:
            continue
    return loaded


PLUGINS = _load_plugins()


@pytest.mark.parametrize("op", LEVEL_1_OPERATORS)
def test_cross_plugin_numeric_parity(op):
    if len(PLUGINS) < 2:
        pytest.skip("少于两个可用插件，无法进行对等性比对")

    rng = np.random.RandomState(42)
    x = rng.randn(4, 8).astype(np.float32)

    outputs = {}
    for name, plugin in PLUGINS.items():
        func = plugin._resolve_operator(op)
        tensor = plugin._to_tensor(x)
        out = func(input=tensor)
        outputs[name] = plugin.to_numpy(out).astype(np.float64)

    shapes = {n: o.shape for n, o in outputs.items()}
    assert len(set(shapes.values())) == 1, f"{op}: shapes diverge: {shapes}"

    names = list(outputs.keys())
    ref = outputs[names[0]]
    for other in names[1:]:
        diff = np.abs(ref - outputs[other]).max()
        assert diff < 1e-4, (
            f"{op}: {names[0]} vs {other} max_abs_diff={diff:.2e}"
        )
