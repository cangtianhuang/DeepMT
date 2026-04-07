"""
Unit tests for MR generation pipeline improvements:
- MRTemplate.transform_code preserved from YAML
- _apply_precheck sets mr.verified = True
- Template oracle expressions are correctly evaluated
- _try_import_operator / _make_default_inputs helpers
No LLM or network dependencies.
"""

import torch
import pytest

from deepmt.ir.schema import MetamorphicRelation, OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.core.oracle_evaluator import OracleEvaluator
from deepmt.plugins.framework_adapter import FrameworkAdapter


# ── Template transform_code 保存正确性 ────────────────────────────────────────

class TestTemplateTransformCode:
    def test_transform_code_preserved_from_yaml(self):
        pool = MRTemplatePool()
        t = pool.templates.get("relu_positive_homogeneity")
        assert t is not None
        assert "lambda" in t.transform_code
        assert "2.0" in t.transform_code

    def test_create_mr_uses_yaml_transform_code(self):
        pool = MRTemplatePool()
        t = pool.templates["relu_positive_homogeneity"]
        mr = pool.create_mr_from_template(t)
        # transform_code must be a valid lambda string (not a comment)
        assert mr.transform_code.startswith("lambda")
        assert "2.0" in mr.transform_code

    def test_transform_code_is_bindable(self):
        """transform_code 必须能被 FrameworkAdapter.bind_transform_code 解析"""
        pool = MRTemplatePool()
        adapter = FrameworkAdapter("pytorch")

        for name, t in pool.templates.items():
            mr = pool.create_mr_from_template(t)
            if not mr.transform_code or mr.transform_code.startswith("#"):
                pytest.fail(f"Template '{name}' has invalid transform_code: {mr.transform_code!r}")
            bound = adapter.bind_transform_code(mr.transform_code, None)
            assert bound is not None, f"Template '{name}' failed to bind: {mr.transform_code}"

    def test_new_operator_templates_loaded(self):
        pool = MRTemplatePool()
        expected = [
            "relu_positive_homogeneity", "relu_nonnegative",
            "sigmoid_complement", "sigmoid_monotone_scale",
            "exp_additive", "exp_positive",
        ]
        for name in expected:
            assert name in pool.templates, f"Template '{name}' not found"

    def test_operator_mapping_new_operators(self):
        pool = MRTemplatePool()
        assert "torch.nn.functional.relu" in pool.operator_mr_mapping
        assert "torch.nn.functional.sigmoid" in pool.operator_mr_mapping
        assert "torch.exp" in pool.operator_mr_mapping


# ── 新模板的 oracle 表达式语义正确性 ────────────────────────────────────────────

class TestOracleExpressions:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ev = OracleEvaluator()
        self.x = torch.tensor([1.0, -2.0, 0.5, -0.3])

    def test_relu_positive_homogeneity(self):
        import torch.nn.functional as F
        orig = F.relu(self.x)
        trans = F.relu(2.0 * self.x)
        f = self.ev.compile_expression("trans == 2.0 * orig", "pytorch")
        assert bool(f(orig, trans, self.x))

    def test_relu_nonnegative(self):
        import torch.nn.functional as F
        orig = F.relu(self.x)
        trans = F.relu(-self.x)
        f = self.ev.compile_expression("orig + trans == abs(x)", "pytorch")
        assert bool(f(orig, trans, self.x))

    def test_sigmoid_complement(self):
        import torch.nn.functional as F
        orig = F.sigmoid(self.x)
        trans = F.sigmoid(-self.x)
        f = self.ev.compile_expression("trans == 1.0 - orig", "pytorch")
        assert bool(f(orig, trans, self.x))

    def test_sigmoid_monotone(self):
        import torch.nn.functional as F
        orig = F.sigmoid(self.x)
        trans = F.sigmoid(self.x + 1.0)
        f = self.ev.compile_expression("all(trans >= orig)", "pytorch")
        assert bool(f(orig, trans, self.x))

    def test_exp_additive(self):
        orig = torch.exp(self.x)
        trans = torch.exp(self.x + 1.0)
        f = self.ev.compile_expression("trans == orig * 2.718281828", "pytorch")
        assert bool(f(orig, trans, self.x))

    def test_exp_positive(self):
        orig = torch.exp(self.x)
        trans = torch.exp(self.x + 1.0)
        f = self.ev.compile_expression("all(orig > 0)", "pytorch")
        assert bool(f(orig, trans, self.x))


# ── precheck 后 mr.verified 被正确设置 ──────────────────────────────────────────

class TestPrecheckSetsVerified():
    def test_precheck_marks_verified(self):
        import torch.nn.functional as F
        from deepmt.mr_generator.operator.operator_mr import OperatorMRGenerator

        pool = MRTemplatePool()
        templates = pool.get_applicable_templates("torch.nn.functional.relu")
        assert len(templates) >= 1

        mrs = [pool.create_mr_from_template(t) for t in templates]
        for mr in mrs:
            assert mr.verified is False  # 初始状态

        generator = OperatorMRGenerator()
        operator_ir = OperatorIR(
            name="torch.nn.functional.relu",
            inputs=[torch.randn(4, 4, dtype=torch.float32)],
        )

        verified = generator._apply_precheck(
            operator_func=F.relu,
            mr_candidates=mrs,
            original_inputs=operator_ir.inputs,
            framework="pytorch",
        )

        # 通过 precheck 的 MR 必须标记为 verified
        for mr in verified:
            assert mr.verified is True, f"MR '{mr.description}' should be verified after precheck"


# ── _try_import_operator helper ───────────────────────────────────────────────

class TestTryImportOperator:
    def test_import_valid_function(self):
        from deepmt.commands.mr import _try_import_operator
        func = _try_import_operator("torch.nn.functional.relu", "pytorch")
        assert func is not None
        assert callable(func)

    def test_import_torch_exp(self):
        from deepmt.commands.mr import _try_import_operator
        func = _try_import_operator("torch.exp", "pytorch")
        assert func is not None

    def test_import_nonexistent_returns_none(self):
        from deepmt.commands.mr import _try_import_operator
        func = _try_import_operator("torch.nonexistent_op_xyz", "pytorch")
        assert func is None

    def test_import_bare_name_returns_none(self):
        from deepmt.commands.mr import _try_import_operator
        func = _try_import_operator("relu", "pytorch")
        assert func is None


# ── _make_default_inputs helper ───────────────────────────────────────────────

class TestMakeDefaultInputs:
    def test_returns_single_tensor(self):
        from deepmt.commands.mr import _make_default_inputs
        import torch.nn.functional as F
        inputs = _make_default_inputs(F.relu, "pytorch")
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert inputs[0].dtype == torch.float32
        assert inputs[0].shape == (4, 4)

    def test_unknown_framework_returns_empty(self):
        from deepmt.commands.mr import _make_default_inputs
        import torch.nn.functional as F
        inputs = _make_default_inputs(F.relu, "tensorflow")
        assert inputs == []
