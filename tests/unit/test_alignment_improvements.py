"""
对齐改进功能的最小单元测试（T1-T8）。
所有测试均不依赖 LLM、网络或 deepmt.__init__ 的全量导入链。
"""

import importlib
import math
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))


# ─── T1: 边界值注入 ───────────────────────────────────────────────────────────


def _import_random_generator():
    import importlib
    m = importlib.import_module("deepmt.analysis.verification.random_generator")
    return m.RandomGenerator


def test_t1_random_generator_constructor(monkeypatch):
    # deepmt.__init__ 因 requests 缺失而失败；直接从子模块路径加载
    import importlib
    m = importlib.import_module("deepmt.analysis.verification.random_generator")
    gen = m.RandomGenerator(boundary_injection_prob=0.5)
    assert gen.boundary_injection_prob == 0.5


def test_t1_random_generator_default_prob():
    RG = _import_random_generator()
    gen = RG()
    assert gen.boundary_injection_prob == 0.15


def test_t1_random_generator_disable():
    RG = _import_random_generator()
    gen = RG(boundary_injection_prob=0.0)
    assert gen.boundary_injection_prob == 0.0


# ─── T2: BenchmarkSuite 算子数量 ──────────────────────────────────────────────


def _import_suite():
    import deepmt.benchmarks.suite as m
    return m.BenchmarkSuite, m._OPERATOR_BENCHMARK_ENTRIES


def test_t2_operator_count_53():
    BS, entries = _import_suite()
    assert len(entries) == 53, f"期望 53 个算子，实际 {len(entries)}"


def test_t2_tensor_op_category_exists():
    BS, entries = _import_suite()
    cats = {e.category for e in entries}
    assert "tensor_op" in cats, "缺少 tensor_op 类别"


def test_t2_five_major_categories():
    BS, entries = _import_suite()
    cats = {e.category for e in entries}
    # 五大类别在代码中分布为以下内部分类
    expected = {"activation", "math", "reduction", "tensor_op", "linear_algebra"}
    assert expected.issubset(cats), f"缺少类别: {expected - cats}"


# ─── T3: 工业级模型清单 ────────────────────────────────────────────────────────


def test_t3_benchmark_model_names():
    BS, _ = _import_suite()
    suite = BS()
    names = suite.model_names()
    assert "ResNet18" in names
    assert "VGG16" in names
    assert "LSTMBenchmark" in names
    assert "BERTEncoder" in names


def test_t3_simple_models_in_registry():
    """Simple* 模型应仍可通过 registry 获取（供测试使用）"""
    import deepmt.benchmarks.models.model_registry as m
    registry = m.ModelBenchmarkRegistry()
    assert registry.get("SimpleMLP", with_instance=False) is not None
    assert registry.get("ResNet18", with_instance=False) is not None


# ─── T4: 自适应容差 ────────────────────────────────────────────────────────────


def _import_atol():
    import deepmt.analysis.verification.mr_verifier as m
    return m.calculate_adaptive_tolerance


def test_t4_elementwise_op():
    fn = _import_atol()
    assert fn("relu") == 1e-6
    assert fn("sigmoid") == 1e-6


def test_t4_matmul_scales_with_size():
    fn = _import_atol()
    small = fn("matmul", [(4, 4)])
    large = fn("matmul", [(1024, 1024)])
    assert large > small, "大矩阵容差应大于小矩阵"
    # large ≈ 1e-5 * sqrt(1024*1024) = 1e-5 * 1024 = 0.01024
    assert abs(large - 1e-5 * math.sqrt(1024 * 1024)) < 1e-10


def test_t4_reduction_scales_with_dim():
    fn = _import_atol()
    small = fn("sum", [(10,)])
    large = fn("sum", [(1000,)])
    assert large > small


def test_t4_fp16_multiplier():
    fn = _import_atol()
    fp32 = fn("relu", dtype="float32")
    fp16 = fn("relu", dtype="float16")
    assert abs(fp16 / fp32 - 10.0) < 1e-9


# ─── T5: MutationTester 三层变异类型 ──────────────────────────────────────────


def test_t5_model_mutant_types():
    import deepmt.analysis.reporting.mutation_tester as m
    values = {e.value for e in m.ModelMutantType}
    assert "weight_perturb" in values
    assert "eval_disabled" in values


def test_t5_app_mutant_types():
    import deepmt.analysis.reporting.mutation_tester as m
    values = {e.value for e in m.AppMutantType}
    assert "label_flip" in values
    assert "augment_reverse" in values


def test_t5_original_mutant_types_unchanged():
    import deepmt.analysis.reporting.mutation_tester as m
    values = {e.value for e in m.MutantType}
    assert "negate" in values
    assert "add_const" in values
    assert "scale" in values


# ─── T8: 属性标签体系 ──────────────────────────────────────────────────────────


def _import_pool():
    import importlib
    m = importlib.import_module("deepmt.mr_generator.base.mr_templates")
    return m.MRTemplatePool


def test_t8_pool_loads_property_tags():
    Pool = _import_pool()
    pool = Pool()
    assert len(pool.available_tags()) > 0


def test_t8_known_tags_present():
    Pool = _import_pool()
    pool = Pool()
    tags = set(pool.available_tags())
    for expected in ["commutative", "linear", "monotone", "even_symmetry", "idempotent"]:
        assert expected in tags, f"缺少标签: {expected}"


def test_t8_get_templates_by_tag():
    Pool = _import_pool()
    pool = Pool()
    comm = pool.get_templates_by_tag("commutative")
    assert len(comm) >= 1
    names = {t.name for t in comm}
    assert "commutative" in names


def test_t8_template_property_tags_field():
    Pool = _import_pool()
    pool = Pool()
    tmpl = pool.templates.get("commutative")
    assert tmpl is not None
    assert "commutative" in tmpl.property_tags
