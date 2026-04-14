"""单元测试：ModelTestRunner — 单模型和批量模型测试执行。

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
def runner(tmp_path):
    from deepmt.engine.model_test_runner import ModelTestRunner
    from deepmt.core.results_manager import ResultsManager
    rm = ResultsManager(db_path=str(tmp_path / "test.db"))
    return ModelTestRunner(results_manager=rm)


# ── 基础测试 ──────────────────────────────────────────────────────────────────

class TestModelTestRunner:

    def test_run_model_mlp(self, runner):
        summary = runner.run_model("SimpleMLP", n_samples=5, batch_size=2)
        assert summary.model_name == "SimpleMLP"
        assert summary.model_type == "mlp"
        assert summary.mr_count > 0
        assert summary.total_cases > 0
        assert summary.passed + summary.failed == summary.total_cases

    def test_run_model_cnn(self, runner):
        summary = runner.run_model("SimpleCNN", n_samples=3, batch_size=2)
        assert summary.model_name == "SimpleCNN"
        assert summary.model_type == "cnn"
        assert summary.mr_count > 0

    def test_run_model_rnn(self, runner):
        summary = runner.run_model("SimpleRNN", n_samples=3, batch_size=2)
        assert summary.model_name == "SimpleRNN"
        assert summary.model_type == "rnn"
        assert summary.mr_count > 0

    def test_run_model_transformer(self, runner):
        summary = runner.run_model("TinyTransformer", n_samples=3, batch_size=2)
        assert summary.model_name == "TinyTransformer"
        assert summary.model_type == "transformer"
        assert summary.mr_count > 0

    def test_nonexistent_model_returns_empty(self, runner):
        summary = runner.run_model("NonExistentModel", n_samples=5)
        assert summary.mr_count == 0
        assert summary.total_cases == 0

    def test_max_mrs_respected(self, runner):
        summary = runner.run_model("SimpleMLP", n_samples=3, max_mrs=2, batch_size=2)
        assert summary.mr_count <= 2

    def test_run_all_returns_four_summaries(self, runner):
        summaries = runner.run_all(framework="pytorch", n_samples=3, batch_size=2)
        assert len(summaries) == 4
        names = {s.model_name for s in summaries}
        assert "SimpleMLP" in names
        assert "SimpleCNN" in names
        assert "SimpleRNN" in names
        assert "TinyTransformer" in names

    def test_mr_summaries_populated(self, runner):
        summary = runner.run_model("SimpleMLP", n_samples=5, batch_size=2)
        assert len(summary.mr_summaries) == summary.mr_count
        for mr_sum in summary.mr_summaries:
            assert mr_sum.mr_id
            assert mr_sum.oracle_expr
            assert mr_sum.total == mr_sum.passed + mr_sum.failed

    def test_to_dict_serializable(self, runner):
        import json
        summary = runner.run_model("SimpleMLP", n_samples=3, batch_size=2)
        d = summary.to_dict()
        # 应能序列化为 JSON
        json_str = json.dumps(d)
        assert len(json_str) > 10

    def test_to_dict_keys(self, runner):
        summary = runner.run_model("SimpleMLP", n_samples=3, batch_size=2)
        d = summary.to_dict()
        assert "model_name" in d
        assert "model_type" in d
        assert "pass_rate" in d
        assert "mr_summaries" in d
        assert "failure_cases" in d

    def test_pass_rate_in_range(self, runner):
        summary = runner.run_model("SimpleMLP", n_samples=5, batch_size=2)
        assert 0.0 <= summary.pass_rate <= 1.0

    def test_identity_transform_high_pass_rate(self, runner):
        """恒等变换（output_close）应几乎全部通过。"""
        from deepmt.ir import MetamorphicRelation
        import uuid
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        registry = ModelBenchmarkRegistry()
        ir = registry.get("SimpleMLP", with_instance=True)

        identity_mr = MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="identity test",
            subject_name="SimpleMLP",
            subject_type="model",
            transform_code="lambda x: x + 0.0",
            oracle_expr="output_close:1e-4",
            layer="model",
        )

        summary = runner.run_model_ir(ir, mrs=[identity_mr], n_samples=10, batch_size=2)
        # 恒等变换：输出应完全一致
        assert summary.failed == 0
        assert summary.passed == 10

    def test_prediction_consistent_passing(self, runner):
        """预测一致性 oracle 对确定性模型应通过。"""
        from deepmt.ir import MetamorphicRelation
        import uuid
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        registry = ModelBenchmarkRegistry()
        ir = registry.get("SimpleMLP", with_instance=True)

        # 恒等变换 + 预测一致性：模型是确定性的，应全部通过
        mr = MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="identity prediction test",
            subject_name="SimpleMLP",
            subject_type="model",
            transform_code="lambda x: x.clone()",
            oracle_expr="prediction_consistent",
            layer="model",
        )

        summary = runner.run_model_ir(ir, mrs=[mr], n_samples=10, batch_size=2)
        assert summary.failed == 0
        assert summary.passed == 10
