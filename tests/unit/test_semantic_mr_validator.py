"""单元测试：SemanticMRValidator 与 ApplicationReporter。

不依赖 LLM / 网络。
"""

import uuid
import pytest

from deepmt.analysis.verification.semantic_mr_validator import (
    SemanticMRValidator,
    SemanticValidationResult,
)
from deepmt.analysis.reporting.application_reporter import ApplicationReporter
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.application.app_mr import ApplicationMRGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    return ApplicationBenchmarkRegistry()


@pytest.fixture
def generator():
    return ApplicationMRGenerator(use_llm=False)


@pytest.fixture
def validator():
    return SemanticMRValidator()


@pytest.fixture
def image_mrs(generator):
    return generator.generate_from_scenario("ImageClassification")


@pytest.fixture
def text_mrs(generator):
    return generator.generate_from_scenario("TextSentiment")


@pytest.fixture
def image_samples(registry):
    sc = registry.get("ImageClassification")
    return sc.sample_inputs, sc.sample_labels


@pytest.fixture
def text_samples(registry):
    sc = registry.get("TextSentiment")
    return sc.sample_inputs, sc.sample_labels


def _make_mr(transform_code: str, oracle_expr: str = "label_consistent") -> MetamorphicRelation:
    fn = eval(transform_code, {"__builtins__": {}}, {})  # noqa: S307
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description="test mr",
        subject_name="TestApp",
        subject_type="application",
        transform_code=transform_code,
        transform=fn,
        oracle_expr=oracle_expr,
        layer="application",
        source="template",
    )


# ── SemanticMRValidator 测试 ──────────────────────────────────────────────────


class TestSemanticMRValidator:

    def test_validate_identity_transform_passes(self, validator):
        """恒等变换：标签应 100% 一致。"""
        mr = _make_mr("lambda s: s")
        samples = [{"text": "good"}, {"text": "bad"}]
        labels = [1, 0]
        result = validator.validate_one(mr, samples, labels)
        assert result.status == "passed"
        assert result.passed_samples == 2
        assert result.failed_samples == 0

    def test_validate_empty_samples_needs_review(self, validator):
        mr = _make_mr("lambda s: s")
        result = validator.validate_one(mr, [], [])
        assert result.status == "needs_review"
        assert "无样例输入" in result.detail

    def test_validate_invalid_transform_code_needs_review(self, validator):
        mr = MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="bad",
            subject_name="X",
            subject_type="application",
            transform_code="not_a_lambda",
            transform=None,
            oracle_expr="label_consistent",
            layer="application",
            source="template",
        )
        result = validator.validate_one(mr, [{"text": "x"}], [1])
        assert result.status == "needs_review"
        assert "transform_code" in result.detail

    def test_validate_label_consistent_all_pass(self, validator):
        """恒等变换在所有样例上应通过 label_consistent。"""
        mr = _make_mr("lambda s: {**s, 'text': s['text'].strip()}")
        samples = [{"text": "  good  "}, {"text": " bad "}]
        labels = [1, 0]
        result = validator.validate_one(mr, samples, labels)
        assert result.status == "passed"

    def test_validate_confidence_acceptable(self, validator):
        mr = _make_mr("lambda s: s", oracle_expr="confidence_acceptable:0.1")
        samples = [{"text": "great"}]
        labels = [1]
        result = validator.validate_one(mr, samples, labels)
        # mock 置信度函数返回 0.9，drop=0 < 0.1 → passed
        assert result.status == "passed"

    def test_validate_label_consistent_soft(self, validator):
        mr = _make_mr("lambda s: s", oracle_expr="label_consistent_soft:3")
        samples = [{"text": "great"}, {"text": "bad"}]
        labels = [1, 0]
        result = validator.validate_one(mr, samples, labels)
        assert result.status == "passed"

    def test_validate_image_mrs_with_mock(self, validator, image_mrs, image_samples):
        sample_inputs, sample_labels = image_samples
        results = validator.validate_batch(image_mrs, sample_inputs, sample_labels)
        assert len(results) == len(image_mrs)
        for res in results:
            assert res.status in ("passed", "failed", "needs_review")

    def test_validate_text_mrs_with_mock(self, validator, text_mrs, text_samples):
        sample_inputs, sample_labels = text_samples
        results = validator.validate_batch(text_mrs, sample_inputs, sample_labels)
        assert len(results) == len(text_mrs)

    def test_mr_lifecycle_updated_on_pass(self, validator):
        mr = _make_mr("lambda s: s")
        validator.validate_one(mr, [{"text": "x"}], [1])
        assert mr.lifecycle_state == "checked"
        assert mr.checked is True

    def test_mr_lifecycle_updated_on_fail(self, validator):
        """制造失败：让 predict_fn 对变换后输入返回不同标签。"""
        mr = _make_mr("lambda s: {**s, 'text': s['text'].upper()}")
        call_count = [0]

        def predict_fn(inp):
            call_count[0] += 1
            # 第二次调用（变换后）返回不同标签
            return 1 if call_count[0] % 2 == 1 else 0

        result = validator.validate_one(
            mr, [{"text": "good"}], [1], predict_fn=predict_fn
        )
        # 原始标签1，变换后标签0 → 失败
        assert result.status in ("failed", "needs_review")

    def test_review_approve(self, validator):
        mr = _make_mr("lambda s: s")
        result = SemanticValidationResult(
            mr_id=mr.id,
            mr_description=mr.description,
            status="needs_review",
            total_samples=2,
            passed_samples=1,
            failed_samples=1,
        )
        validator.review_mr(mr, result, approved=True, note="人工确认通过")
        assert result.status == "approved"
        assert result.review_note == "人工确认通过"
        assert mr.lifecycle_state == "proven"
        assert mr.verified is True

    def test_review_reject(self, validator):
        mr = _make_mr("lambda s: s")
        result = SemanticValidationResult(
            mr_id=mr.id,
            mr_description=mr.description,
            status="needs_review",
            total_samples=2,
            passed_samples=1,
            failed_samples=1,
        )
        validator.review_mr(mr, result, approved=False, note="不合理的变换")
        assert result.status == "rejected"
        assert mr.lifecycle_state == "retired"
        assert mr.verified is False

    def test_validate_result_fields(self, validator):
        mr = _make_mr("lambda s: s")
        result = validator.validate_one(mr, [{"text": "x"}], [1])
        assert result.mr_id == mr.id
        assert result.mr_description == mr.description
        assert result.total_samples == 1
        assert result.passed_samples + result.failed_samples == result.total_samples


# ── ApplicationReporter 测试 ──────────────────────────────────────────────────


class TestApplicationReporter:

    @pytest.fixture
    def reporter(self):
        return ApplicationReporter()

    @pytest.fixture
    def mrs_and_results(self, generator, registry, validator):
        sc = registry.get("TextSentiment")
        mrs = generator.generate_from_scenario("TextSentiment")
        results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)
        return mrs, results

    def test_generate_report_basic(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results, scenario_name="TextSentiment")
        assert report.scenario_name == "TextSentiment"
        assert report.total_mrs == len(mrs)
        assert 0.0 <= report.pass_rate <= 1.0

    def test_by_source_contains_template(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results)
        assert "template" in report.by_source

    def test_by_status_populated(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results)
        assert sum(report.by_status.values()) == len(mrs)

    def test_by_category_populated(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results)
        assert len(report.by_category) > 0

    def test_format_text_returns_string(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results, scenario_name="TextSentiment")
        text = reporter.format_text(report)
        assert isinstance(text, str)
        assert "TextSentiment" in text
        assert "MR 总数" in text

    def test_to_dict_keys(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results)
        d = reporter.to_dict(report)
        for key in ("scenario_name", "total_mrs", "pass_rate", "by_status",
                    "by_source", "by_category", "example_passed", "example_failed"):
            assert key in d

    def test_empty_mrs(self, reporter):
        report = reporter.generate([], [], scenario_name="Empty")
        assert report.total_mrs == 0
        assert report.pass_rate == 0.0

    def test_examples_not_exceed_max(self, reporter, mrs_and_results):
        mrs, results = mrs_and_results
        report = reporter.generate(mrs, results, max_examples=1)
        assert len(report.example_passed) <= 1
        assert len(report.example_failed) <= 1
        assert len(report.example_review) <= 1
