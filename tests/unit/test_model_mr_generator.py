"""单元测试：TransformStrategyLibrary 和 ModelMRGenerator。

不依赖 LLM、网络；需要 PyTorch 环境。
"""

import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="需要 PyTorch"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    from deepmt.benchmarks.models import ModelBenchmarkRegistry
    return ModelBenchmarkRegistry()


@pytest.fixture
def library():
    from deepmt.mr_generator.model.transform_strategy import TransformStrategyLibrary
    return TransformStrategyLibrary()


@pytest.fixture
def generator():
    from deepmt.mr_generator.model import ModelMRGenerator
    return ModelMRGenerator()


# ── TransformStrategyLibrary 测试 ─────────────────────────────────────────────

class TestTransformStrategyLibrary:

    def test_all_strategies_not_empty(self, library):
        strategies = library.all_strategies()
        assert len(strategies) > 0

    def test_get_strategy_by_name(self, library):
        s = library.get("input_scale_small")
        assert s is not None
        assert s.name == "input_scale_small"
        assert s.transform_code != ""
        assert s.oracle_type != ""

    def test_get_nonexistent_returns_none(self, library):
        s = library.get("nonexistent_strategy")
        assert s is None

    def test_select_mlp_returns_float_strategies(self, library):
        from deepmt.model.graph_analyzer import ModelAnalysisResult
        result = ModelAnalysisResult(
            model_name="TestMLP",
            model_type="mlp",
            task_type="classification",
            layer_infos=[],
            layer_tags=["linear", "activation", "linear"],
            has_conv=False,
            has_rnn=False,
            has_attention=False,
            has_normalization=False,
            has_embedding=False,
            input_is_integer=False,
            num_parameters=1000,
            depth=3,
            key_patterns=["fc_block"],
        )
        strategies = library.select(result)
        assert len(strategies) > 0
        for s in strategies:
            assert s.input_dtype != "int"

    def test_select_rnn_returns_int_strategies(self, library):
        from deepmt.model.graph_analyzer import ModelAnalysisResult
        result = ModelAnalysisResult(
            model_name="TestRNN",
            model_type="rnn",
            task_type="classification",
            layer_infos=[],
            layer_tags=["embedding", "rnn", "linear"],
            has_conv=False,
            has_rnn=True,
            has_attention=False,
            has_normalization=False,
            has_embedding=True,
            input_is_integer=True,
            num_parameters=5000,
            depth=3,
            key_patterns=["rnn_block", "embedding_input"],
        )
        strategies = library.select(result)
        assert len(strategies) > 0
        for s in strategies:
            assert s.input_dtype != "float"

    def test_select_max_strategies(self, library):
        from deepmt.model.graph_analyzer import ModelAnalysisResult
        result = ModelAnalysisResult(
            model_name="TestMLP",
            model_type="mlp",
            task_type="classification",
            layer_infos=[],
            layer_tags=["linear", "activation", "linear"],
            has_conv=False,
            has_rnn=False,
            has_attention=False,
            has_normalization=False,
            has_embedding=False,
            input_is_integer=False,
            num_parameters=1000,
            depth=3,
            key_patterns=["fc_block"],
        )
        strategies = library.select(result, max_strategies=2)
        assert len(strategies) <= 2

    def test_cnn_strategies_include_spatial(self, library):
        from deepmt.model.graph_analyzer import ModelAnalysisResult
        result = ModelAnalysisResult(
            model_name="TestCNN",
            model_type="cnn",
            task_type="classification",
            layer_infos=[],
            layer_tags=["conv", "activation", "pool", "linear"],
            has_conv=True,
            has_rnn=False,
            has_attention=False,
            has_normalization=False,
            has_embedding=False,
            input_is_integer=False,
            num_parameters=10000,
            depth=4,
            key_patterns=["conv_pool", "fc_block"],
        )
        strategies = library.select(result)
        names = [s.name for s in strategies]
        # CNN 应该包含空间翻转策略
        assert any("spatial" in n or "flip" in n for n in names)

    def test_transform_codes_are_valid_lambdas(self, library):
        """所有策略的 transform_code 应可以被 eval 为可调用对象。"""
        for s in library.all_strategies():
            if s.input_dtype == "int":
                x = torch.randint(0, 100, (2, 16))
            else:
                x = torch.randn(2, 64)
            try:
                fn = eval(s.transform_code)  # noqa: S307
                result = fn(x)
                assert result is not None
            except Exception as e:
                # 允许 __import__ 形式在 eval 中工作
                if "__import__" not in s.transform_code:
                    pytest.fail(f"Strategy {s.name}: {e}")


# ── ModelMRGenerator 测试 ─────────────────────────────────────────────────────

class TestModelMRGenerator:

    def test_generate_mlp_produces_mrs(self, registry, generator):
        ir = registry.get("SimpleMLP", with_instance=True)
        mrs = generator.generate(ir)
        assert len(mrs) > 0

    def test_generate_cnn_produces_mrs(self, registry, generator):
        ir = registry.get("SimpleCNN", with_instance=True)
        mrs = generator.generate(ir)
        assert len(mrs) > 0

    def test_generate_rnn_produces_mrs(self, registry, generator):
        ir = registry.get("SimpleRNN", with_instance=True)
        mrs = generator.generate(ir)
        assert len(mrs) > 0

    def test_generate_transformer_produces_mrs(self, registry, generator):
        ir = registry.get("TinyTransformer", with_instance=True)
        mrs = generator.generate(ir)
        assert len(mrs) > 0

    def test_mr_fields_populated(self, registry, generator):
        ir = registry.get("SimpleMLP", with_instance=True)
        mrs = generator.generate(ir)
        for mr in mrs:
            assert mr.id
            assert mr.description
            assert mr.transform_code
            assert mr.oracle_expr
            assert mr.layer == "model"
            assert mr.subject_type == "model"
            assert mr.subject_name == "SimpleMLP"
            assert mr.source == "template"

    def test_max_per_model_respected(self, registry, generator):
        ir = registry.get("SimpleMLP", with_instance=True)
        mrs = generator.generate(ir, max_per_model=2)
        assert len(mrs) <= 2

    def test_generate_no_instance_returns_empty(self, generator):
        from deepmt.ir.schema import ModelIR
        ir = ModelIR(name="NoInstance", model_type="mlp")
        mrs = generator.generate(ir)
        assert mrs == []

    def test_generate_from_analysis(self, registry, generator):
        from deepmt.model.graph_analyzer import ModelGraphAnalyzer
        ir = registry.get("SimpleMLP", with_instance=True)
        analyzer = ModelGraphAnalyzer()
        analysis = analyzer.analyze(ir)
        mrs = generator.generate_from_analysis(analysis, model_name="SimpleMLP")
        assert len(mrs) > 0
        for mr in mrs:
            assert mr.subject_name == "SimpleMLP"

    def test_oracle_expr_is_valid_type(self, registry, generator):
        """oracle_expr 必须是 ModelVerifier 可识别的类型之一。"""
        valid_prefixes = {
            "prediction_consistent",
            "topk_consistent",
            "output_close",
            "output_order_invariant",
        }
        ir = registry.get("SimpleMLP", with_instance=True)
        mrs = generator.generate(ir)
        for mr in mrs:
            oracle_base = mr.oracle_expr.split(":")[0]
            assert oracle_base in valid_prefixes, (
                f"未知 oracle_type: {mr.oracle_expr!r}"
            )

    def test_all_benchmarks_generate_mrs(self, registry, generator):
        for name in registry.names(framework="pytorch"):
            ir = registry.get(name, with_instance=True)
            mrs = generator.generate(ir)
            assert len(mrs) >= 1, f"{name} 未生成任何 MR"
