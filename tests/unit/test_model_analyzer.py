"""单元测试：ModelGraphAnalyzer 和 ModelBenchmarkRegistry。

不依赖 LLM、网络；需要 PyTorch 环境。
"""

import pytest

try:
    import torch
    import torch.nn as nn
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
def analyzer():
    from deepmt.model.graph_analyzer import ModelGraphAnalyzer
    return ModelGraphAnalyzer()


# ── ModelBenchmarkRegistry 测试 ────────────────────────────────────────────────

class TestModelBenchmarkRegistry:

    def test_list_models_returns_four(self, registry):
        models = registry.list_models()
        assert len(models) == 4

    def test_list_models_framework_filter(self, registry):
        pt = registry.list_models(framework="pytorch")
        tf = registry.list_models(framework="tensorflow")
        assert len(pt) == 4
        assert len(tf) == 0

    def test_get_returns_model_ir(self, registry):
        ir = registry.get("SimpleMLP", with_instance=False)
        assert ir is not None
        assert ir.name == "SimpleMLP"
        assert ir.model_type == "mlp"
        assert ir.task_type == "classification"
        assert ir.num_classes == 10
        assert ir.input_shape == (64,)

    def test_get_with_instance_attaches_model(self, registry):
        ir = registry.get("SimpleMLP", with_instance=True)
        assert ir is not None
        assert ir.model_instance is not None

    def test_get_without_instance_no_model(self, registry):
        ir = registry.get("SimpleMLP", with_instance=False)
        assert ir.model_instance is None

    def test_get_nonexistent_returns_none(self, registry):
        ir = registry.get("NonExistentModel", with_instance=False)
        assert ir is None

    def test_names_returns_all(self, registry):
        names = registry.names()
        assert "SimpleMLP" in names
        assert "SimpleCNN" in names
        assert "SimpleRNN" in names
        assert "TinyTransformer" in names

    def test_cnn_input_shape(self, registry):
        ir = registry.get("SimpleCNN", with_instance=False)
        assert ir.input_shape == (1, 28, 28)

    def test_rnn_metadata(self, registry):
        ir = registry.get("SimpleRNN", with_instance=False)
        assert ir.metadata.get("input_dtype") == "int64"
        assert ir.metadata.get("vocab_size") == 100

    def test_transformer_metadata(self, registry):
        ir = registry.get("TinyTransformer", with_instance=False)
        assert ir.metadata.get("input_dtype") == "int64"

    def test_model_instances_run_forward(self, registry):
        """基准模型的前向推理应成功运行。"""
        for name in ["SimpleMLP", "SimpleCNN"]:
            ir = registry.get(name, with_instance=True)
            assert ir.model_instance is not None
            # 构造输入并推理
            batch = 2
            if ir.input_shape:
                x = torch.randn(batch, *ir.input_shape)
            else:
                x = torch.randn(batch, 64)
            with torch.no_grad():
                out = ir.model_instance(x)
            assert out.shape == (batch, ir.num_classes)

    def test_rnn_forward(self, registry):
        ir = registry.get("SimpleRNN", with_instance=True)
        seq_len = ir.metadata.get("seq_len", 16)
        vocab_size = ir.metadata.get("vocab_size", 100)
        x = torch.randint(0, vocab_size, (2, seq_len))
        with torch.no_grad():
            out = ir.model_instance(x)
        assert out.shape == (2, ir.num_classes)

    def test_transformer_forward(self, registry):
        ir = registry.get("TinyTransformer", with_instance=True)
        seq_len = ir.metadata.get("seq_len", 16)
        vocab_size = ir.metadata.get("vocab_size", 100)
        x = torch.randint(0, vocab_size, (2, seq_len))
        with torch.no_grad():
            out = ir.model_instance(x)
        assert out.shape == (2, ir.num_classes)


# ── ModelGraphAnalyzer 测试 ────────────────────────────────────────────────────

class TestModelGraphAnalyzer:

    def test_analyze_mlp(self, registry, analyzer):
        ir = registry.get("SimpleMLP", with_instance=True)
        result = analyzer.analyze(ir)
        assert result.model_type == "mlp"
        assert result.has_conv is False
        assert result.has_rnn is False
        assert result.has_attention is False
        assert result.depth >= 2  # Linear + ReLU + Linear at minimum
        assert result.num_parameters > 0

    def test_analyze_cnn(self, registry, analyzer):
        ir = registry.get("SimpleCNN", with_instance=True)
        result = analyzer.analyze(ir)
        assert result.model_type == "cnn"
        assert result.has_conv is True
        assert "conv_pool" in result.key_patterns
        assert "fc_block" in result.key_patterns

    def test_analyze_rnn(self, registry, analyzer):
        ir = registry.get("SimpleRNN", with_instance=True)
        result = analyzer.analyze(ir)
        assert result.model_type == "rnn"
        assert result.has_rnn is True
        assert result.has_embedding is True
        assert result.input_is_integer is True
        assert "rnn_block" in result.key_patterns
        assert "embedding_input" in result.key_patterns

    def test_analyze_transformer(self, registry, analyzer):
        ir = registry.get("TinyTransformer", with_instance=True)
        result = analyzer.analyze(ir)
        assert result.model_type == "transformer"
        assert result.has_attention is True
        assert result.has_embedding is True
        assert "attention_block" in result.key_patterns

    def test_analyze_fills_input_shape(self, registry, analyzer):
        ir = registry.get("SimpleCNN", with_instance=True)
        result = analyzer.analyze(ir)
        assert result.input_shape == (1, 28, 28)

    def test_analyze_to_dict(self, registry, analyzer):
        ir = registry.get("SimpleMLP", with_instance=True)
        result = analyzer.analyze(ir)
        d = result.to_dict()
        assert "model_type" in d
        assert "has_conv" in d
        assert "layer_tags" in d
        assert isinstance(d["layer_tags"], list)

    def test_analyze_and_fill_updates_model_ir(self, registry, analyzer):
        ir = registry.get("SimpleMLP", with_instance=True)
        analyzer.analyze_and_fill(ir)
        assert ir.model_type == "mlp"
        assert "has_conv" in ir.analysis_summary

    def test_analyze_no_instance_raises(self, analyzer):
        from deepmt.ir.schema import ModelIR
        ir = ModelIR(name="NoInstance", model_type="mlp")
        with pytest.raises(ValueError, match="model_instance"):
            analyzer.analyze(ir)

    def test_layer_infos_not_empty(self, registry, analyzer):
        ir = registry.get("SimpleMLP", with_instance=True)
        result = analyzer.analyze(ir)
        assert len(result.layer_infos) > 0
        for li in result.layer_infos:
            assert li.class_name
            assert li.tag
            assert li.params >= 0
