"""单元测试：ApplicationScenario 与 ApplicationBenchmarkRegistry。

不依赖 LLM / 网络 / PyTorch。
"""

import pytest
from deepmt.mr_generator.application.scenario import ApplicationScenario
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
from deepmt.ir import ApplicationIR

_SAFE_BUILTINS = {
    "min": min, "max": max, "sum": sum, "len": len, "abs": abs,
    "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "bool": bool, "tuple": tuple,
    "set": set, "range": range,
}


class TestApplicationScenario:

    def test_dataclass_fields(self):
        sc = ApplicationScenario(
            name="TestApp",
            task_type="image_classification",
            domain="computer_vision",
            input_type="image_array",
            output_type="class_label",
            input_schema="flat float list",
            output_schema="int label",
            description="Test scenario",
        )
        assert sc.name == "TestApp"
        assert sc.task_type == "image_classification"
        assert sc.domain_facts == []
        assert sc.sample_inputs == []
        assert sc.known_transforms == []

    def test_with_domain_facts(self):
        sc = ApplicationScenario(
            name="S",
            task_type="text_sentiment",
            domain="nlp",
            input_type="text_string",
            output_type="sentiment_label",
            input_schema="str",
            output_schema="int",
            description="d",
            domain_facts=["fact1", "fact2"],
        )
        assert len(sc.domain_facts) == 2

    def test_known_transforms_structure(self):
        sc = ApplicationScenario(
            name="S",
            task_type="text_sentiment",
            domain="nlp",
            input_type="text_string",
            output_type="sentiment_label",
            input_schema="str",
            output_schema="int",
            description="d",
            known_transforms=[
                {
                    "name": "lowercase",
                    "transform_code": "lambda s: {**s, 'text': s['text'].lower()}",
                    "oracle_expr": "label_consistent",
                }
            ],
        )
        assert len(sc.known_transforms) == 1
        assert sc.known_transforms[0]["name"] == "lowercase"


class TestApplicationBenchmarkRegistry:

    @pytest.fixture
    def registry(self):
        return ApplicationBenchmarkRegistry()

    def test_list_scenarios_returns_two(self, registry):
        scenarios = registry.list_scenarios()
        assert len(scenarios) == 2

    def test_names(self, registry):
        names = registry.names()
        assert "ImageClassification" in names
        assert "TextSentiment" in names

    def test_get_image_classification(self, registry):
        sc = registry.get("ImageClassification")
        assert sc is not None
        assert sc.task_type == "image_classification"
        assert sc.domain == "computer_vision"
        assert len(sc.domain_facts) > 0
        assert len(sc.sample_inputs) > 0
        assert len(sc.known_transforms) > 0

    def test_get_text_sentiment(self, registry):
        sc = registry.get("TextSentiment")
        assert sc is not None
        assert sc.task_type == "text_sentiment"
        assert sc.domain == "nlp"
        assert len(sc.domain_facts) > 0

    def test_get_nonexistent_returns_none(self, registry):
        sc = registry.get("NonExistentScenario")
        assert sc is None

    def test_get_ir_returns_application_ir(self, registry):
        ir = registry.get_ir("ImageClassification")
        assert ir is not None
        assert isinstance(ir, ApplicationIR)
        assert ir.name == "ImageClassification"
        assert ir.task_type == "image_classification"
        assert ir.subject_type == "application"

    def test_get_ir_text_sentiment(self, registry):
        ir = registry.get_ir("TextSentiment")
        assert ir is not None
        assert ir.task_type == "text_sentiment"

    def test_get_ir_nonexistent_returns_none(self, registry):
        ir = registry.get_ir("DoesNotExist")
        assert ir is None

    def test_sample_inputs_are_dicts(self, registry):
        for name in registry.names():
            sc = registry.get(name)
            for inp in sc.sample_inputs:
                assert isinstance(inp, dict)

    def test_sample_labels_count_matches_inputs(self, registry):
        for name in registry.names():
            sc = registry.get(name)
            assert len(sc.sample_inputs) == len(sc.sample_labels)

    def test_known_transforms_have_required_keys(self, registry):
        required = {"transform_code", "oracle_expr"}
        for name in registry.names():
            sc = registry.get(name)
            for t in sc.known_transforms:
                for key in required:
                    assert key in t, f"[{name}] transform 缺少 '{key}': {t}"

    def test_known_transforms_code_is_valid_lambda(self, registry):
        for name in registry.names():
            sc = registry.get(name)
            for t in sc.known_transforms:
                code = t["transform_code"]
                fn = eval(code, {"__builtins__": _SAFE_BUILTINS}, {})  # noqa: S307
                assert callable(fn)

    def test_image_transforms_work_on_sample_input(self, registry):
        sc = registry.get("ImageClassification")
        for t in sc.known_transforms:
            fn = eval(t["transform_code"], {"__builtins__": _SAFE_BUILTINS}, {})  # noqa: S307
            for inp in sc.sample_inputs:
                result = fn(inp)
                assert "input" in result
                assert isinstance(result["input"], list)

    def test_text_transforms_work_on_sample_input(self, registry):
        sc = registry.get("TextSentiment")
        for t in sc.known_transforms:
            fn = eval(t["transform_code"], {"__builtins__": _SAFE_BUILTINS}, {})  # noqa: S307
            for inp in sc.sample_inputs:
                result = fn(inp)
                assert "text" in result
                assert isinstance(result["text"], str)
