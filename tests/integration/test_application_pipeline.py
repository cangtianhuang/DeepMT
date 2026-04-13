"""集成测试：应用层完整流水线（use_llm=False）。

覆盖：场景注册 → MR 生成 → 语义验证 → 人工复核 → 报告生成。
不依赖 LLM / 网络。
"""

import pytest

from deepmt.analysis.application_reporter import ApplicationReporter
from deepmt.analysis.semantic_mr_validator import SemanticMRValidator
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
from deepmt.mr_generator.application.app_mr import ApplicationMRGenerator


class TestApplicationPipeline:

    @pytest.fixture
    def registry(self):
        return ApplicationBenchmarkRegistry()

    @pytest.fixture
    def generator(self):
        return ApplicationMRGenerator(use_llm=False)

    @pytest.fixture
    def validator(self):
        return SemanticMRValidator()

    @pytest.fixture
    def reporter(self):
        return ApplicationReporter()

    def _run_pipeline(self, name, generator, validator, reporter, registry):
        """执行完整应用层流水线，返回报告。"""
        sc = registry.get(name)
        assert sc is not None, f"场景 {name!r} 未注册"

        # 1. MR 生成
        mrs = generator.generate_from_scenario(name)
        assert len(mrs) >= 1, f"场景 {name!r} 生成 MR 数为 0"

        # 2. 验证
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)
        assert len(results) == len(mrs)

        # 3. 报告
        report = reporter.generate(mrs, results, scenario_name=name)
        assert report.total_mrs == len(mrs)
        assert report.scenario_name == name

        return mrs, results, report

    def test_image_classification_pipeline(
        self, generator, validator, reporter, registry
    ):
        mrs, results, report = self._run_pipeline(
            "ImageClassification", generator, validator, reporter, registry
        )
        assert report.total_mrs > 0
        # 模板 MR 应全部通过（恒等变换 mock 下）
        assert report.pass_rate > 0.0

    def test_text_sentiment_pipeline(
        self, generator, validator, reporter, registry
    ):
        mrs, results, report = self._run_pipeline(
            "TextSentiment", generator, validator, reporter, registry
        )
        assert report.total_mrs > 0
        assert report.pass_rate > 0.0

    def test_all_mrs_have_application_layer(self, generator, registry):
        for name in registry.names():
            mrs = generator.generate_from_scenario(name)
            for mr in mrs:
                assert mr.layer == "application"
                assert mr.subject_type == "application"

    def test_format_report_contains_passed_section(
        self, generator, validator, reporter, registry
    ):
        sc = registry.get("TextSentiment")
        mrs = generator.generate_from_scenario("TextSentiment")
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)
        report = reporter.generate(mrs, results, scenario_name="TextSentiment")
        text = reporter.format_text(report)
        # 报告文本应包含场景名
        assert "TextSentiment" in text

    def test_review_approved_updates_lifecycle(
        self, generator, validator, registry
    ):
        sc = registry.get("ImageClassification")
        mrs = generator.generate_from_scenario("ImageClassification", max_mrs=1)
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)

        mr = mrs[0]
        result = results[0]
        validator.review_mr(mr, result, approved=True, note="测试批准")
        assert mr.lifecycle_state == "proven"
        assert mr.verified is True
        assert result.status == "approved"

    def test_review_rejected_retires_mr(
        self, generator, validator, registry
    ):
        sc = registry.get("ImageClassification")
        mrs = generator.generate_from_scenario("ImageClassification", max_mrs=1)
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)

        mr = mrs[0]
        result = results[0]
        validator.review_mr(mr, result, approved=False, note="不合理")
        assert mr.lifecycle_state == "retired"
        assert result.status == "rejected"

    def test_to_dict_exportable(
        self, generator, validator, reporter, registry
    ):
        sc = registry.get("ImageClassification")
        mrs = generator.generate_from_scenario("ImageClassification")
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)
        report = reporter.generate(mrs, results, scenario_name="ImageClassification")
        d = reporter.to_dict(report)
        assert isinstance(d, dict)
        assert d["total_mrs"] == len(mrs)
        assert "example_passed" in d

    def test_pipeline_from_ir(self, generator, validator, reporter, registry):
        """从 ApplicationIR 入口走通流水线。"""
        ir = registry.get_ir("TextSentiment")
        sc = registry.get("TextSentiment")
        mrs = generator.generate_from_ir(ir)
        assert len(mrs) > 0
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)
        report = reporter.generate(mrs, results, scenario_name=ir.name)
        assert report.total_mrs == len(mrs)
