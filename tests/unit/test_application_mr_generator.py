"""单元测试：ApplicationMRGenerator（use_llm=False 模式）与 AppContextBuilder。

不依赖 LLM / 网络。
"""

import pytest
from deepmt.application.scenario import ApplicationScenario
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
from deepmt.ir.schema import ApplicationIR, MetamorphicRelation
from deepmt.mr_generator.application.app_context_builder import AppContextBuilder
from deepmt.mr_generator.application.app_mr import ApplicationMRGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    return ApplicationBenchmarkRegistry()


@pytest.fixture
def generator():
    return ApplicationMRGenerator(use_llm=False)


@pytest.fixture
def image_scenario(registry):
    return registry.get("ImageClassification")


@pytest.fixture
def text_scenario(registry):
    return registry.get("TextSentiment")


# ── AppContextBuilder 测试 ────────────────────────────────────────────────────


class TestAppContextBuilder:

    def test_build_returns_string(self, image_scenario):
        builder = AppContextBuilder()
        ctx = builder.build(image_scenario)
        assert isinstance(ctx, str)
        assert len(ctx) > 50

    def test_build_contains_task_type(self, image_scenario):
        builder = AppContextBuilder()
        ctx = builder.build(image_scenario)
        assert "image_classification" in ctx

    def test_build_contains_domain(self, text_scenario):
        builder = AppContextBuilder()
        ctx = builder.build(text_scenario)
        assert "nlp" in ctx

    def test_build_contains_domain_facts(self, image_scenario):
        builder = AppContextBuilder()
        ctx = builder.build(image_scenario)
        # 领域知识片段应在上下文中
        assert "Domain Knowledge" in ctx

    def test_source_snapshot_keys(self, image_scenario):
        builder = AppContextBuilder()
        snap = builder.build_source_snapshot(image_scenario)
        assert "scenario_name" in snap
        assert "task_type" in snap
        assert "num_domain_facts" in snap
        assert snap["source_type"] == "static_knowledge"

    def test_source_snapshot_values(self, image_scenario):
        builder = AppContextBuilder()
        snap = builder.build_source_snapshot(image_scenario)
        assert snap["scenario_name"] == "ImageClassification"
        assert snap["num_domain_facts"] == len(image_scenario.domain_facts)


# ── ApplicationMRGenerator 测试（use_llm=False）────────────────────────────────


class TestApplicationMRGeneratorTemplate:

    def test_generate_image_classification_produces_mrs(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        assert len(mrs) > 0

    def test_generate_text_sentiment_produces_mrs(self, generator):
        mrs = generator.generate_from_scenario("TextSentiment")
        assert len(mrs) > 0

    def test_all_scenarios_produce_mrs(self, generator, registry):
        for name in registry.names():
            mrs = generator.generate_from_scenario(name)
            assert len(mrs) >= 1, f"场景 {name!r} 未生成任何 MR"

    def test_mr_layer_is_application(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        for mr in mrs:
            assert mr.layer == "application"

    def test_mr_subject_type_is_application(self, generator):
        mrs = generator.generate_from_scenario("TextSentiment")
        for mr in mrs:
            assert mr.subject_type == "application"

    def test_mr_subject_name(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        for mr in mrs:
            assert mr.subject_name == "ImageClassification"

    def test_mr_source_is_template(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        for mr in mrs:
            assert mr.source == "template"

    def test_mr_fields_populated(self, generator):
        mrs = generator.generate_from_scenario("TextSentiment")
        for mr in mrs:
            assert mr.id
            assert mr.description
            assert mr.transform_code
            assert mr.oracle_expr

    def test_mr_oracle_expr_valid(self, generator, registry):
        valid_prefixes = {"label_consistent", "label_consistent_soft", "confidence_acceptable"}
        for name in registry.names():
            mrs = generator.generate_from_scenario(name)
            for mr in mrs:
                prefix = mr.oracle_expr.split(":")[0]
                assert prefix in valid_prefixes, f"未知 oracle_expr: {mr.oracle_expr}"

    def test_max_mrs_respected(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification", max_mrs=2)
        assert len(mrs) <= 2

    def test_no_duplicate_transform_oracle(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        keys = [(mr.transform_code.strip(), mr.oracle_expr.strip()) for mr in mrs]
        assert len(keys) == len(set(keys)), "存在重复的 transform_code + oracle_expr"

    def test_generate_from_ir(self, generator, registry):
        ir = registry.get_ir("TextSentiment")
        mrs = generator.generate_from_ir(ir)
        assert len(mrs) > 0
        for mr in mrs:
            assert mr.layer == "application"

    def test_generate_from_ir_unregistered(self, generator):
        ir = ApplicationIR(
            name="CustomApp",
            task_type="image_classification",
            domain="computer_vision",
            input_description="pixel list",
            output_description="label",
        )
        # 未注册场景，known_transforms 为空，应返回空列表
        mrs = generator.generate_from_ir(ir)
        assert isinstance(mrs, list)

    def test_generate_unified_entry(self, generator, registry):
        sc = registry.get("ImageClassification")
        mrs_sc = generator.generate(sc)
        mrs_name = generator.generate("ImageClassification")
        assert len(mrs_sc) == len(mrs_name)

    def test_transform_callable(self, generator):
        mrs = generator.generate_from_scenario("ImageClassification")
        for mr in mrs:
            assert mr.transform is not None
            assert callable(mr.transform)

    def test_transform_works_on_image_sample(self, generator, registry):
        mrs = generator.generate_from_scenario("ImageClassification")
        sc = registry.get("ImageClassification")
        for mr in mrs:
            for sample in sc.sample_inputs:
                result = mr.transform(sample)
                assert "input" in result

    def test_transform_works_on_text_sample(self, generator, registry):
        mrs = generator.generate_from_scenario("TextSentiment")
        sc = registry.get("TextSentiment")
        for mr in mrs:
            for sample in sc.sample_inputs:
                result = mr.transform(sample)
                assert "text" in result
