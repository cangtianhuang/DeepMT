"""G 阶段单元测试：统一 IR 基类与 MR 生命周期。

测试覆盖：
  - TestSubject 基类字段
  - OperatorIR / ModelIR / ApplicationIR 继承与独立字段
  - MetamorphicRelation 新字段（subject_name / lifecycle_state / sync_lifecycle）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from deepmt.ir import (
    ApplicationIR,
    MetamorphicRelation,
    ModelIR,
    OperatorIR,
    TestSubject,
)


# ── TestSubject 基类 ──────────────────────────────────────────────────────────


class TestTestSubjectBase:
    def test_operator_ir_is_test_subject(self):
        op = OperatorIR(name="torch.add")
        assert isinstance(op, TestSubject)

    def test_model_ir_is_test_subject(self):
        m = ModelIR(name="ResNet50")
        assert isinstance(m, TestSubject)

    def test_application_ir_is_test_subject(self):
        a = ApplicationIR(name="ImageClassifier")
        assert isinstance(a, TestSubject)

    def test_subject_type_defaults(self):
        assert OperatorIR(name="x").subject_type == "operator"
        assert ModelIR(name="x").subject_type == "model"
        assert ApplicationIR(name="x").subject_type == "application"

    def test_framework_default_none(self):
        op = OperatorIR(name="torch.relu")
        assert op.framework is None

    def test_framework_can_be_set(self):
        op = OperatorIR(name="torch.relu", framework="pytorch")
        assert op.framework == "pytorch"

    def test_metadata_default_empty_dict(self):
        op = OperatorIR(name="torch.relu")
        assert op.metadata == {}

    def test_metadata_independent_between_instances(self):
        a = OperatorIR(name="a")
        b = OperatorIR(name="b")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata


# ── OperatorIR 字段 ───────────────────────────────────────────────────────────


class TestOperatorIR:
    def test_name_required(self):
        op = OperatorIR(name="torch.add")
        assert op.name == "torch.add"

    def test_all_optional_fields_default(self):
        op = OperatorIR(name="torch.add")
        assert op.inputs is None
        assert op.outputs is None
        assert op.properties is None
        assert op.api_path == ""
        assert op.api_style == "function"
        assert op.input_specs is None

    def test_input_specs_can_be_list(self):
        specs = [{"name": "input", "shape": [3, 3], "dtype": "float32"}]
        op = OperatorIR(name="torch.relu", input_specs=specs)
        assert op.input_specs == specs

    def test_repr_contains_name(self):
        op = OperatorIR(name="torch.add")
        assert "torch.add" in repr(op)


# ── ModelIR 占位字段 ──────────────────────────────────────────────────────────


class TestModelIR:
    def test_layers_default_empty(self):
        m = ModelIR(name="MyModel")
        assert m.layers == []

    def test_connections_default_empty(self):
        m = ModelIR(name="MyModel")
        assert m.connections == []

    def test_layers_independent_between_instances(self):
        a = ModelIR(name="A")
        b = ModelIR(name="B")
        a.layers.append("layer1")
        assert b.layers == []


# ── ApplicationIR 字段（Phase J 完善）───────────────────────────────────────


class TestApplicationIR:
    def test_task_type_default_empty(self):
        a = ApplicationIR(name="App")
        assert a.task_type == ""

    def test_domain_default_empty(self):
        a = ApplicationIR(name="App")
        assert a.domain == ""

    def test_input_output_description(self):
        a = ApplicationIR(
            name="App",
            input_description="image/224x224",
            output_description="json/labels",
        )
        assert a.input_description == "image/224x224"
        assert a.output_description == "json/labels"

    def test_sample_inputs_default_empty(self):
        a = ApplicationIR(name="App")
        assert a.sample_inputs == []
        assert a.sample_labels == []
        assert a.context_snippets == []

    def test_subject_type_is_application(self):
        a = ApplicationIR(name="App")
        assert a.subject_type == "application"


# ── MetamorphicRelation 新字段 ────────────────────────────────────────────────


class TestMetamorphicRelationNew:
    def _make_mr(self, **kwargs) -> MetamorphicRelation:
        defaults = {"id": "mr-1", "description": "test MR"}
        defaults.update(kwargs)
        return MetamorphicRelation(**defaults)

    def test_subject_name_default_empty(self):
        mr = self._make_mr()
        assert mr.subject_name == ""

    def test_subject_name_can_be_set(self):
        mr = self._make_mr(subject_name="torch.add")
        assert mr.subject_name == "torch.add"

    def test_subject_type_default_operator(self):
        mr = self._make_mr()
        assert mr.subject_type == "operator"

    def test_lifecycle_state_default_pending(self):
        mr = self._make_mr()
        assert mr.lifecycle_state == "pending"

    def test_sync_lifecycle_checked(self):
        mr = self._make_mr(checked=True, proven=False, verified=False)
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "checked"

    def test_sync_lifecycle_proven(self):
        mr = self._make_mr(checked=True, proven=True, verified=False)
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "proven"

    def test_sync_lifecycle_verified_implies_proven(self):
        mr = self._make_mr(verified=True)
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "proven"

    def test_sync_lifecycle_all_false_is_pending(self):
        mr = self._make_mr(checked=False, proven=False, verified=False)
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "pending"

    def test_old_fields_still_present(self):
        mr = self._make_mr(checked=True, proven=True, verified=True)
        assert mr.checked is True
        assert mr.proven is True
        assert mr.verified is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
