"""G 阶段单元测试：统一 IR 基类、MR 生命周期、SubjectRegistry、RunManifest。

测试覆盖：
  - TestSubject 基类字段
  - OperatorIR / ModelIR / ApplicationIR 继承与独立字段
  - MetamorphicRelation 新字段（subject_name / lifecycle_state / sync_lifecycle）
  - SubjectRegistry 注册、查找、枚举
  - RunManifest 序列化与反序列化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from deepmt.core.run_manifest import RunManifest
from deepmt.core.subject_registry import SubjectRegistry
from deepmt.ir.schema import (
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


# ── SubjectRegistry ───────────────────────────────────────────────────────────


class TestSubjectRegistry:
    def test_empty_registry(self):
        reg = SubjectRegistry()
        assert len(reg) == 0
        assert reg.all_subjects() == []

    def test_register_and_lookup(self):
        reg = SubjectRegistry()
        op = OperatorIR(name="torch.add")
        reg.register(op)
        assert reg.lookup("torch.add") is op

    def test_lookup_missing_returns_none(self):
        reg = SubjectRegistry()
        assert reg.lookup("nonexistent") is None

    def test_get_missing_raises_key_error(self):
        reg = SubjectRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_contains_operator(self):
        reg = SubjectRegistry()
        op = OperatorIR(name="torch.relu")
        reg.register(op)
        assert "torch.relu" in reg

    def test_register_duplicate_skips_by_default(self):
        reg = SubjectRegistry()
        op1 = OperatorIR(name="torch.add")
        op2 = OperatorIR(name="torch.add", api_path="new")
        reg.register(op1)
        reg.register(op2)  # should skip
        assert reg.lookup("torch.add") is op1

    def test_register_overwrite(self):
        reg = SubjectRegistry()
        op1 = OperatorIR(name="torch.add", api_path="old")
        op2 = OperatorIR(name="torch.add", api_path="new")
        reg.register(op1)
        reg.register(op2, overwrite=True)
        assert reg.lookup("torch.add").api_path == "new"

    def test_list_by_type_operator(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="torch.add"))
        reg.register(OperatorIR(name="torch.relu"))
        reg.register(ModelIR(name="ResNet"))
        ops = reg.list_by_type("operator")
        assert len(ops) == 2
        assert all(s.subject_type == "operator" for s in ops)

    def test_list_by_type_model(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="torch.add"))
        reg.register(ModelIR(name="ResNet"))
        models = reg.list_by_type("model")
        assert len(models) == 1
        assert models[0].name == "ResNet"

    def test_list_by_type_empty(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="torch.add"))
        assert reg.list_by_type("application") == []

    def test_names_returns_all(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="torch.add"))
        reg.register(OperatorIR(name="torch.relu"))
        names = reg.names()
        assert set(names) == {"torch.add", "torch.relu"}

    def test_unregister(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="torch.add"))
        result = reg.unregister("torch.add")
        assert result is True
        assert "torch.add" not in reg

    def test_unregister_missing_returns_false(self):
        reg = SubjectRegistry()
        assert reg.unregister("nonexistent") is False

    def test_register_many(self):
        reg = SubjectRegistry()
        subjects = [OperatorIR(name="a"), OperatorIR(name="b"), OperatorIR(name="c")]
        count = reg.register_many(subjects)
        assert count == 3
        assert len(reg) == 3

    def test_register_non_subject_raises_type_error(self):
        reg = SubjectRegistry()
        with pytest.raises(TypeError):
            reg.register("not_a_subject")  # type: ignore

    def test_iter(self):
        reg = SubjectRegistry()
        reg.register(OperatorIR(name="a"))
        reg.register(OperatorIR(name="b"))
        names = {s.name for s in reg}
        assert names == {"a", "b"}


# ── RunManifest ───────────────────────────────────────────────────────────────


class TestRunManifest:
    def test_auto_run_id_generated(self):
        m = RunManifest(subject_name="torch.add")
        assert m.run_id != ""
        assert len(m.run_id) == 36  # UUID4 格式

    def test_two_manifests_have_different_run_ids(self):
        m1 = RunManifest(subject_name="torch.add")
        m2 = RunManifest(subject_name="torch.add")
        assert m1.run_id != m2.run_id

    def test_auto_timestamp_generated(self):
        m = RunManifest(subject_name="torch.add")
        assert m.timestamp != ""

    def test_defaults(self):
        m = RunManifest(subject_name="torch.relu")
        assert m.subject_type == "operator"
        assert m.framework == "pytorch"
        assert m.framework_version == ""
        assert m.random_seed is None
        assert m.n_samples == 0
        assert m.mr_ids == []
        assert m.notes == ""

    def test_env_summary_populated(self):
        m = RunManifest(subject_name="torch.add")
        assert "python" in m.env_summary
        assert "os" in m.env_summary

    def test_to_dict_roundtrip(self):
        m = RunManifest(
            subject_name="torch.add",
            framework="pytorch",
            framework_version="2.3.0",
            random_seed=42,
            n_samples=10,
            mr_ids=["mr-1", "mr-2"],
            notes="RQ1 experiment",
        )
        d = m.to_dict()
        m2 = RunManifest.from_dict(d)
        assert m2.run_id == m.run_id
        assert m2.subject_name == "torch.add"
        assert m2.framework_version == "2.3.0"
        assert m2.random_seed == 42
        assert m2.n_samples == 10
        assert m2.mr_ids == ["mr-1", "mr-2"]
        assert m2.notes == "RQ1 experiment"

    def test_from_dict_missing_optional_fields(self):
        m = RunManifest.from_dict({"subject_name": "torch.relu"})
        assert m.subject_name == "torch.relu"
        assert m.framework == "pytorch"
        assert m.random_seed is None

    def test_mr_ids_independent_between_instances(self):
        m1 = RunManifest(subject_name="a")
        m2 = RunManifest(subject_name="b")
        m1.mr_ids.append("mr-1")
        assert m2.mr_ids == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
