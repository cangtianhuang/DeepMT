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

from deepmt.ir import MetamorphicRelation, OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.plugins.pytorch_plugin import PyTorchPlugin


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
        """transform_code 必须能被 MRPreChecker._bind_transform_code 解析"""
        from deepmt.analysis.verification.mr_prechecker import MRPreChecker
        pool = MRTemplatePool()

        for name, t in pool.templates.items():
            mr = pool.create_mr_from_template(t)
            if not mr.transform_code or mr.transform_code.startswith("#"):
                pytest.fail(f"Template '{name}' has invalid transform_code: {mr.transform_code!r}")
            bound = MRPreChecker._bind_transform_code(mr.transform_code, None)
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
        self.verifier = MRVerifier()
        self.plugin = PyTorchPlugin()
        self.x = torch.tensor([1.0, -2.0, 0.5, -0.3])

    def _mr(self, oracle_expr: str) -> MetamorphicRelation:
        return MetamorphicRelation(
            id="test", description="test",
            transform=lambda k: k, transform_code="",
            oracle_expr=oracle_expr, tolerance=1e-5,
        )

    def _verify(self, orig, trans, expr: str) -> bool:
        result = self.verifier.verify(orig, trans, self._mr(expr), self.plugin, x_input=self.x)
        return result.passed

    def test_relu_positive_homogeneity(self):
        import torch.nn.functional as F
        orig = F.relu(self.x)
        trans = F.relu(2.0 * self.x)
        assert self._verify(orig, trans, "trans == 2.0 * orig")

    def test_relu_nonnegative(self):
        import torch.nn.functional as F
        orig = F.relu(self.x)
        trans = F.relu(-self.x)
        assert self._verify(orig, trans, "orig + trans == abs(x)")

    def test_sigmoid_complement(self):
        import torch.nn.functional as F
        orig = F.sigmoid(self.x)
        trans = F.sigmoid(-self.x)
        assert self._verify(orig, trans, "trans == 1.0 - orig")

    def test_sigmoid_monotone(self):
        import torch.nn.functional as F
        orig = F.sigmoid(self.x)
        trans = F.sigmoid(self.x + 1.0)
        assert self._verify(orig, trans, "all(trans >= orig)")

    def test_exp_additive(self):
        orig = torch.exp(self.x)
        trans = torch.exp(self.x + 1.0)
        assert self._verify(orig, trans, "trans == orig * 2.718281828")

    def test_exp_positive(self):
        orig = torch.exp(self.x)
        trans = torch.exp(self.x + 1.0)
        assert self._verify(orig, trans, "all(orig > 0)")


# ── precheck 后 mr.verified 被正确设置 ──────────────────────────────────────────

class TestPrecheckSetsVerified():
    def test_precheck_marks_verified(self):
        import torch.nn.functional as F
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator

        pool = MRTemplatePool()
        templates = pool.get_applicable_templates("torch.nn.functional.relu")
        assert len(templates) >= 1

        mrs = [pool.create_mr_from_template(t) for t in templates]
        for mr in mrs:
            assert mr.verified is False  # 初始状态

        generator = OperatorMRGenerator()
        operator_ir = OperatorIR(name="torch.nn.functional.relu")

        verified = generator._apply_precheck(
            operator_func=F.relu,
            mr_candidates=mrs,
            operator_ir=operator_ir,
            framework="pytorch",
        )

        # 通过 precheck 的 MR 必须标记为 checked
        assert len(verified) >= 1, "Expected at least one MR to pass precheck"
        for mr in verified:
            assert mr.checked is True, f"MR '{mr.description}' should have checked=True after precheck"


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


# ── RandomGenerator ───────────────────────────────────────────────────────────

class TestRandomGenerator:
    def setup_method(self):
        from deepmt.analysis.verification.random_generator import RandomGenerator
        from deepmt.plugins.pytorch_plugin import PyTorchPlugin
        self.gen = RandomGenerator()
        self.plugin = PyTorchPlugin()

    def test_empty_specs_returns_default(self):
        inputs = self.gen.generate([], self.plugin)
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert inputs[0].dtype == torch.float32

    def test_basic_float_spec(self):
        inputs = self.gen.generate([
            {"name": "input", "dtype": ["float32"], "shape": "any", "value_range": None, "required": True}
        ], self.plugin)
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert inputs[0].dtype == torch.float32

    def test_shape_nd_ge(self):
        inputs = self.gen.generate([
            {"name": "x", "dtype": ["float64"], "shape": "nd>=2", "value_range": None, "required": True}
        ], self.plugin)
        assert len(inputs[0].shape) >= 2
        assert inputs[0].dtype == torch.float64

    def test_value_range(self):
        inputs = self.gen.generate([
            {"name": "x", "dtype": ["float32"], "shape": "any", "value_range": [0.0, 1.0], "required": True}
        ], self.plugin)
        t = inputs[0]
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_optional_param_skipped(self):
        inputs = self.gen.generate([
            {"name": "x", "dtype": ["float32"], "shape": "any", "value_range": None, "required": True},
            {"name": "bias", "dtype": ["float32"], "shape": "any", "value_range": None, "required": False},
        ], self.plugin)
        assert len(inputs) == 1
