"""模型基准注册表：将预定义模型与 ModelIR 对象绑定。

用法::

    registry = ModelBenchmarkRegistry()
    models = registry.list_models()          # 所有基准 ModelIR
    ir = registry.get("SimpleMLP")          # 按名称获取，含 model_instance
    ir = registry.get("SimpleMLP", with_instance=False)  # 仅元数据

"""

from typing import List, Optional

from deepmt.core.logger import logger
from deepmt.ir import ModelIR

# 懒导入 PyTorch 模型，避免在无 torch 环境中报错
_pytorch_models = None


def _get_pytorch_models():
    global _pytorch_models
    if _pytorch_models is None:
        try:
            import deepmt.benchmarks.models.pytorch_models as m
            _pytorch_models = m
        except ImportError:
            _pytorch_models = None
    return _pytorch_models


# ── 基准模型元数据配置 ──────────────────────────────────────────────────────────

_BENCHMARK_SPECS = [
    {
        "name": "SimpleMLP",
        "framework": "pytorch",
        "model_type": "mlp",
        "task_type": "classification",
        "input_shape": (64,),
        "output_shape": (10,),
        "num_classes": 10,
        "constructor": lambda m: m.SimpleMLP(input_dim=64, hidden_dim=128, num_classes=10),
        "metadata": {"hidden_dim": 128, "input_dim": 64},
    },
    {
        "name": "SimpleCNN",
        "framework": "pytorch",
        "model_type": "cnn",
        "task_type": "classification",
        "input_shape": (1, 28, 28),
        "output_shape": (10,),
        "num_classes": 10,
        "constructor": lambda m: m.SimpleCNN(in_channels=1, num_classes=10),
        "metadata": {"in_channels": 1},
    },
    {
        "name": "SimpleRNN",
        "framework": "pytorch",
        "model_type": "rnn",
        "task_type": "classification",
        "input_shape": (16,),     # seq_len
        "output_shape": (10,),
        "num_classes": 10,
        "constructor": lambda m: m.SimpleRNN(
            vocab_size=100, embed_dim=32, hidden_dim=64, num_classes=10, seq_len=16
        ),
        "metadata": {"vocab_size": 100, "seq_len": 16, "input_dtype": "int64"},
    },
    {
        "name": "TinyTransformer",
        "framework": "pytorch",
        "model_type": "transformer",
        "task_type": "classification",
        "input_shape": (16,),     # seq_len
        "output_shape": (10,),
        "num_classes": 10,
        "constructor": lambda m: m.TinyTransformer(
            vocab_size=100, embed_dim=32, num_heads=4, num_classes=10, seq_len=16
        ),
        "metadata": {"vocab_size": 100, "seq_len": 16, "input_dtype": "int64"},
    },
]


class ModelBenchmarkRegistry:
    """模型基准注册表。

    提供所有预定义基准模型的 ModelIR 及（可选）实例化的模型对象。
    """

    def list_models(self, framework: Optional[str] = None) -> List[ModelIR]:
        """返回所有基准模型的 ModelIR 列表（不含 model_instance）。

        Args:
            framework: 框架名过滤；None 表示返回全部
        """
        results = []
        for spec in _BENCHMARK_SPECS:
            if framework is not None and spec["framework"] != framework:
                continue
            ir = self._build_ir(spec, with_instance=False)
            results.append(ir)
        return results

    def get(self, name: str, with_instance: bool = True) -> Optional[ModelIR]:
        """按名称获取基准模型 ModelIR。

        Args:
            name:          模型名称（如 "SimpleMLP"）
            with_instance: True 时实例化模型并附到 model_instance 字段

        Returns:
            ModelIR 对象；未找到则返回 None
        """
        for spec in _BENCHMARK_SPECS:
            if spec["name"] == name:
                return self._build_ir(spec, with_instance=with_instance)
        logger.warning(f"[ModelBenchmarkRegistry] 未找到模型: {name!r}")
        return None

    def names(self, framework: Optional[str] = None) -> List[str]:
        """返回所有基准模型名称列表。"""
        return [
            s["name"]
            for s in _BENCHMARK_SPECS
            if framework is None or s["framework"] == framework
        ]

    # ── 内部 ───────────────────────────────────────────────────────────────────

    def _build_ir(self, spec: dict, with_instance: bool) -> ModelIR:
        ir = ModelIR(
            name=spec["name"],
            framework=spec["framework"],
            model_type=spec["model_type"],
            task_type=spec["task_type"],
            input_shape=spec["input_shape"],
            output_shape=spec["output_shape"],
            num_classes=spec.get("num_classes"),
            metadata=dict(spec.get("metadata", {})),
        )
        if with_instance:
            m = _get_pytorch_models()
            if m is not None:
                try:
                    instance = spec["constructor"](m)
                    instance.eval()
                    ir.model_instance = instance
                except Exception as e:
                    logger.warning(
                        f"[ModelBenchmarkRegistry] 实例化 {spec['name']} 失败: {e}"
                    )
        return ir
