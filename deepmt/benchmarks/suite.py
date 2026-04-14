"""
论文实验 Benchmark Suite（固化版本）。

设计目标：
  提供单一入口获取论文实验中使用的全部 benchmark 对象：
  - 算子层：代表性算子子集（30+ 个覆盖激活、数学、归一化、池化等分类）
  - 模型层：SimpleMLP / SimpleCNN / SimpleRNN / TinyTransformer
  - 应用层：ImageClassification / TextSentiment

原则：
  - 本文件只负责"固定清单"，不负责实际执行逻辑
  - 清单一旦固化，需同步更新 docs/dev/status.md
  - 可被 ExperimentOrganizer、export_thesis_tables.py 等直接引用

用法::

    from deepmt.benchmarks.suite import BenchmarkSuite

    suite = BenchmarkSuite()
    print(suite.operator_names())
    print(suite.model_names())
    print(suite.application_names())
    ops = suite.operator_benchmark()
    models = suite.model_benchmark()
    apps = suite.application_benchmark()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── 算子层 benchmark 固定清单 ─────────────────────────────────────────────────
#
# 论文实验选取的代表性算子，覆盖常用功能分类。
# 标注 framework="pytorch" 表示当前实现的主框架。

@dataclass(frozen=True)
class OperatorBenchmarkEntry:
    """单个算子 benchmark 条目。"""

    name: str
    """算子全限定名（与 OperatorCatalog 一致）。"""

    framework: str
    """目标框架。"""

    category: str
    """功能分类（activation / math / normalization / pooling / loss / reduction）。"""

    layer: str = "operator"
    """benchmark 所在层次（固定为 operator）。"""

    notes: str = ""


_OPERATOR_BENCHMARK_ENTRIES: List[OperatorBenchmarkEntry] = [
    # ── 激活函数 ──────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.nn.functional.relu",    "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.sigmoid", "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.tanh",    "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.leaky_relu", "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.gelu",    "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.softmax", "pytorch", "activation"),
    OperatorBenchmarkEntry("torch.nn.functional.log_softmax", "pytorch", "activation"),
    # ── 基础数学算子 ──────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.abs",  "pytorch", "math"),
    OperatorBenchmarkEntry("torch.exp",  "pytorch", "math"),
    OperatorBenchmarkEntry("torch.log",  "pytorch", "math"),
    OperatorBenchmarkEntry("torch.sqrt", "pytorch", "math"),
    OperatorBenchmarkEntry("torch.pow",  "pytorch", "math"),
    OperatorBenchmarkEntry("Add",        "pytorch", "math",
                           notes="逐元素加法（torch.add / tensor.__add__）"),
    OperatorBenchmarkEntry("Multiply",   "pytorch", "math",
                           notes="逐元素乘法（torch.mul / tensor.__mul__）"),
    # ── 归一化 ────────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.nn.functional.batch_norm",  "pytorch", "normalization"),
    OperatorBenchmarkEntry("torch.nn.functional.layer_norm",  "pytorch", "normalization"),
    OperatorBenchmarkEntry("torch.nn.functional.group_norm",  "pytorch", "normalization"),
    # ── 池化 ──────────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.nn.functional.max_pool2d",    "pytorch", "pooling"),
    OperatorBenchmarkEntry("torch.nn.functional.avg_pool2d",    "pytorch", "pooling"),
    OperatorBenchmarkEntry("torch.nn.functional.adaptive_avg_pool2d", "pytorch", "pooling"),
    # ── 损失函数 ──────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.nn.functional.cross_entropy", "pytorch", "loss"),
    OperatorBenchmarkEntry("torch.nn.functional.mse_loss",      "pytorch", "loss"),
    OperatorBenchmarkEntry("torch.nn.functional.binary_cross_entropy_with_logits",
                           "pytorch", "loss"),
    # ── 归约 ──────────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.sum",  "pytorch", "reduction"),
    OperatorBenchmarkEntry("torch.mean", "pytorch", "reduction"),
    OperatorBenchmarkEntry("torch.max",  "pytorch", "reduction"),
    OperatorBenchmarkEntry("torch.min",  "pytorch", "reduction"),
    OperatorBenchmarkEntry("torch.std",  "pytorch", "reduction"),
    # ── 线性代数 ──────────────────────────────────────────────────────────────
    OperatorBenchmarkEntry("torch.nn.functional.linear", "pytorch", "linear_algebra"),
    OperatorBenchmarkEntry("torch.matmul", "pytorch", "linear_algebra"),
    OperatorBenchmarkEntry("torch.transpose", "pytorch", "linear_algebra"),
]


# ── 模型层 benchmark（引用 ModelBenchmarkRegistry 中已定义的模型）─────────────

_MODEL_BENCHMARK_NAMES = [
    "SimpleMLP",
    "SimpleCNN",
    "SimpleRNN",
    "TinyTransformer",
]

# ── 应用层 benchmark（引用 ApplicationBenchmarkRegistry 中已定义的场景）────────

_APPLICATION_BENCHMARK_NAMES = [
    "ImageClassification",
    "TextSentiment",
]


# ── BenchmarkSuite 入口 ────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    论文实验 Benchmark Suite。

    提供算子/模型/应用三层的固化 benchmark 清单及访问接口。
    """

    # ── 算子层 ────────────────────────────────────────────────────────────────

    def operator_benchmark(
        self,
        framework: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[OperatorBenchmarkEntry]:
        """
        返回算子层 benchmark 条目列表。

        Args:
            framework: 框架过滤（None=全部）
            category:  功能分类过滤（None=全部）
        """
        entries = list(_OPERATOR_BENCHMARK_ENTRIES)
        if framework is not None:
            entries = [e for e in entries if e.framework == framework]
        if category is not None:
            entries = [e for e in entries if e.category == category]
        return entries

    def operator_names(self, framework: Optional[str] = None) -> List[str]:
        """返回算子名称列表。"""
        return [e.name for e in self.operator_benchmark(framework=framework)]

    def operator_categories(self) -> List[str]:
        """返回所有功能分类（去重有序）。"""
        seen = []
        for e in _OPERATOR_BENCHMARK_ENTRIES:
            if e.category not in seen:
                seen.append(e.category)
        return seen

    def operator_count_by_category(self) -> Dict[str, int]:
        """按分类统计算子数量。"""
        result: Dict[str, int] = {}
        for e in _OPERATOR_BENCHMARK_ENTRIES:
            result[e.category] = result.get(e.category, 0) + 1
        return result

    # ── 模型层 ────────────────────────────────────────────────────────────────

    def model_names(self) -> List[str]:
        """返回模型层 benchmark 名称列表。"""
        return list(_MODEL_BENCHMARK_NAMES)

    def model_benchmark(self, with_instance: bool = False) -> List[Any]:
        """
        返回模型层 benchmark 的 ModelIR 列表。

        Args:
            with_instance: 是否实例化 PyTorch 模型（默认 False，仅返回元数据）
        """
        from deepmt.benchmarks.models.model_registry import ModelBenchmarkRegistry
        registry = ModelBenchmarkRegistry()
        result = []
        for name in _MODEL_BENCHMARK_NAMES:
            ir = registry.get(name, with_instance=with_instance)
            if ir is not None:
                result.append(ir)
        return result

    # ── 应用层 ────────────────────────────────────────────────────────────────

    def application_names(self) -> List[str]:
        """返回应用层 benchmark 名称列表。"""
        return list(_APPLICATION_BENCHMARK_NAMES)

    def application_benchmark(self) -> List[Any]:
        """返回应用层 benchmark 的 ApplicationScenario 列表。"""
        from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
        registry = ApplicationBenchmarkRegistry()
        result = []
        for name in _APPLICATION_BENCHMARK_NAMES:
            sc = registry.get(name)
            if sc is not None:
                result.append(sc)
        return result

    # ── 汇总 ──────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        返回 benchmark 规模汇总。

        Returns:
            {
              "operator_count": int,
              "model_count": int,
              "application_count": int,
              "operator_categories": {category: count},
              "model_names": [...],
              "application_names": [...],
            }
        """
        return {
            "operator_count": len(_OPERATOR_BENCHMARK_ENTRIES),
            "model_count": len(_MODEL_BENCHMARK_NAMES),
            "application_count": len(_APPLICATION_BENCHMARK_NAMES),
            "operator_categories": self.operator_count_by_category(),
            "model_names": self.model_names(),
            "application_names": self.application_names(),
        }
