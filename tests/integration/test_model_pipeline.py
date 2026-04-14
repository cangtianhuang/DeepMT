"""集成测试：模型层完整主链（Phase I 端到端验证）。

测试内容：
  registry → analyze → generate MRs → execute tests → verify results

不依赖 LLM、网络；需要 PyTorch 环境。
运行方式：
    source .venv/bin/activate && PYTHONPATH=$(pwd) python -m pytest tests/integration/test_model_pipeline.py -v
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


class TestModelPipeline:
    """端到端：Registry → Analyzer → Generator → Runner → Report"""

    @pytest.fixture
    def tmp_runner(self, tmp_path):
        from deepmt.engine.model_test_runner import ModelTestRunner
        from deepmt.core.results_manager import ResultsManager
        rm = ResultsManager(db_path=str(tmp_path / "model_test.db"))
        return ModelTestRunner(results_manager=rm)

    def test_mlp_full_pipeline(self, tmp_runner):
        """SimpleMLP 完整主链：加载 → 分析 → 生成 MR → 执行 → 统计。"""
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        from deepmt.mr_generator.model.graph_analyzer import ModelGraphAnalyzer
        from deepmt.mr_generator.model import ModelMRGenerator

        registry = ModelBenchmarkRegistry()
        analyzer = ModelGraphAnalyzer()
        generator = ModelMRGenerator()

        # 1. 加载模型
        ir = registry.get("SimpleMLP", with_instance=True)
        assert ir is not None
        assert ir.model_instance is not None

        # 2. 结构分析
        analysis = analyzer.analyze_and_fill(ir)
        assert analysis.model_type == "mlp"
        assert ir.analysis_summary.get("model_type") == "mlp"

        # 3. 生成 MR
        mrs = generator.generate(ir)
        assert len(mrs) > 0
        for mr in mrs:
            assert mr.layer == "model"
            assert mr.subject_type == "model"

        # 4. 执行测试
        summary = tmp_runner.run_model_ir(ir, mrs=mrs, n_samples=5, batch_size=2)
        assert summary.model_name == "SimpleMLP"
        assert summary.mr_count == len(mrs)
        assert summary.total_cases > 0
        assert summary.passed + summary.failed == summary.total_cases

        # 5. 报告输出
        report = summary.to_dict()
        assert isinstance(report, dict)
        assert report["model_name"] == "SimpleMLP"
        assert 0.0 <= report["pass_rate"] <= 1.0

    def test_all_benchmarks_pipeline(self, tmp_runner):
        """所有 4 个基准模型应都能完成完整主链，无未处理异常。"""
        summaries = tmp_runner.run_all(
            framework="pytorch",
            n_samples=3,
            max_mrs=3,
            batch_size=2,
        )
        assert len(summaries) == 4
        for s in summaries:
            # 所有模型都应完成测试（无崩溃）
            assert s.model_name
            assert s.model_type
            assert s.mr_count > 0
            assert s.total_cases > 0
            # pass_rate 在有效范围内
            assert 0.0 <= s.pass_rate <= 1.0

    def test_model_ir_fields_after_analysis(self):
        """通过 analyze_and_fill 后 ModelIR 的 analysis_summary 应被正确填充。"""
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        from deepmt.mr_generator.model.graph_analyzer import ModelGraphAnalyzer

        registry = ModelBenchmarkRegistry()
        analyzer = ModelGraphAnalyzer()

        for name in ["SimpleMLP", "SimpleCNN", "SimpleRNN", "TinyTransformer"]:
            ir = registry.get(name, with_instance=True)
            analyzer.analyze_and_fill(ir)
            assert ir.model_type != ""
            assert ir.analysis_summary.get("num_parameters", 0) > 0
            assert isinstance(ir.analysis_summary.get("layer_tags"), list)

    def test_mr_generation_consistent_oracle_types(self):
        """生成的 MR oracle_expr 类型必须均可被 ModelVerifier 处理。"""
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        from deepmt.mr_generator.model import ModelMRGenerator
        from deepmt.analysis.verification.model_verifier import ModelVerifier
        import uuid
        from deepmt.ir import MetamorphicRelation

        registry = ModelBenchmarkRegistry()
        generator = ModelMRGenerator()
        verifier = ModelVerifier()

        ir = registry.get("SimpleMLP", with_instance=True)
        mrs = generator.generate(ir)

        dummy_orig = torch.randn(2, 10)
        dummy_trans = torch.randn(2, 10)

        for mr in mrs:
            result = verifier.verify(dummy_orig, dummy_trans, mr)
            # 只要不抛出异常，passed 为 True 或 False 均可
            assert isinstance(result.passed, bool)
            assert result.expr == mr.oracle_expr

    def test_cli_test_model_list(self):
        """CLI deepmt test model --list 应不报错并返回模型列表。"""
        from click.testing import CliRunner
        from deepmt.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["test", "model", "--list"])
        assert result.exit_code == 0, result.output
        assert "SimpleMLP" in result.output
        assert "SimpleCNN" in result.output

    def test_cli_mr_model_generate(self):
        """CLI deepmt mr model-generate SimpleMLP 应输出 MR 描述。"""
        from click.testing import CliRunner
        from deepmt.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["mr", "model-generate", "SimpleMLP"])
        assert result.exit_code == 0, result.output
        assert "SimpleMLP" in result.output
        assert "MRs" in result.output

    def test_cli_test_model_single(self):
        """CLI deepmt test model SimpleMLP 应输出测试结果摘要。"""
        from click.testing import CliRunner
        from deepmt.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "test", "model", "SimpleMLP",
            "--n-samples", "3",
            "--batch-size", "2",
        ])
        assert result.exit_code == 0, result.output
        assert "SimpleMLP" in result.output
        assert "passed" in result.output.lower() or "MR" in result.output
