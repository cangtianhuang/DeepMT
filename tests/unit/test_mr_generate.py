"""
Unit tests for MR generation pipeline:
- MRTemplate.transform_code preserved from YAML
- discover_all_templates / generate_mr_candidates pure-discovery semantics
- _apply_precheck sets mr.checked = True
- Template oracle expressions are correctly evaluated on canonical operators
- _try_import_operator helper
- RandomGenerator input synthesis
No LLM or network dependencies.
"""

import torch
import pytest

from deepmt.ir import MetamorphicRelation, OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.plugins.pytorch_plugin import PyTorchPlugin


# ── 模板池只存放通用数学律，不包含算子专属条目 ──────────────────────────────

class TestTemplatePoolContents:
    def test_templates_loaded(self):
        pool = MRTemplatePool()
        assert len(pool.templates) > 0

    def test_no_operator_specific_entries(self):
        """确保已移除的 7 条算子专属条目不再存在。"""
        pool = MRTemplatePool()
        forbidden = [
            "relu_positive_homogeneity", "relu_nonnegative",
            "sigmoid_complement", "sigmoid_monotone_scale",
            "exp_additive", "exp_positive", "abs_even_symmetry",
        ]
        for name in forbidden:
            assert name not in pool.templates, \
                f"算子专属模板 '{name}' 不应存在于通用模板池中"

    def test_no_operator_mr_mapping_attribute(self):
        """operator_mr_mapping 已被彻底删除。"""
        pool = MRTemplatePool()
        assert not hasattr(pool, "operator_mr_mapping")

    def test_generic_templates_present(self):
        """保留的通用模板应包含常见数学律。"""
        pool = MRTemplatePool()
        for name in ("commutative", "associative_left", "additive_identity",
                     "identity", "unary_scale_linear", "unary_monotone_increase"):
            assert name in pool.templates

    def test_transform_code_is_bindable(self):
        """所有模板的 transform_code 必须能被 MRPreChecker 解析。"""
        from deepmt.analysis.verification.mr_prechecker import MRPreChecker
        pool = MRTemplatePool()
        for name, t in pool.templates.items():
            mr = pool.create_mr_from_template(t)
            assert mr.transform_code and not mr.transform_code.startswith("#"), \
                f"Template '{name}' has invalid transform_code: {mr.transform_code!r}"
            bound = MRPreChecker._bind_transform_code(mr.transform_code, None)
            assert bound is not None, f"Template '{name}' failed to bind"


# ── discover_all_templates 按 arity 过滤 ─────────────────────────────────────

class TestDiscoverAllTemplates:
    def test_discover_unary(self):
        pool = MRTemplatePool()
        templates = pool.discover_all_templates(num_inputs=1)
        # 至少应找到若干 unary 模板和 identity（max_inputs=None）
        names = {t.name for t in templates}
        assert "identity" in names
        assert "unary_scale_linear" in names
        # 二元模板不应出现在一元结果中
        assert "commutative" not in names

    def test_discover_binary_includes_commutative(self):
        pool = MRTemplatePool()
        templates = pool.discover_all_templates(num_inputs=2)
        names = {t.name for t in templates}
        assert "commutative" in names

    def test_generate_mr_candidates_pure_discovery(self):
        """generate_mr_candidates 无需预映射即可为任意算子产出候选。"""
        pool = MRTemplatePool()
        mrs = pool.generate_mr_candidates("relu", num_inputs=1)
        assert len(mrs) > 0
        assert all(mr.source == "template" for mr in mrs)
        assert all(mr.verified is False for mr in mrs)  # 候选阶段均未验证


# ── oracle 表达式语义正确性（以通用模板 × 已知算子举例） ─────────────────────

class TestGenericOracleOnKnownOperators:
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

    def test_unary_scale_linear_on_relu(self):
        """relu 满足一元正齐次性：unary_scale_linear 模板 + relu → 成立。"""
        import torch.nn.functional as F
        orig = F.relu(self.x)
        trans = F.relu(2.0 * self.x)
        assert self._verify(orig, trans, "trans == 2.0 * orig")

    def test_unary_reflect_even_on_abs(self):
        """abs 满足偶对称：unary_reflect 模板 + abs → 成立。"""
        orig = torch.abs(self.x)
        trans = torch.abs(-self.x)
        assert self._verify(orig, trans, "trans == orig")

    def test_unary_monotone_on_sigmoid(self):
        """sigmoid 单调递增：unary_monotone_increase 模板 + sigmoid → 成立。"""
        import torch.nn.functional as F
        orig = F.sigmoid(self.x)
        trans = F.sigmoid(self.x + 1.0)
        assert self._verify(orig, trans, "all(trans >= orig)")

    def test_unary_nonnegative_on_exp(self):
        """exp 恒正：unary_nonnegative_output 模板 + exp → 成立。"""
        orig = torch.exp(self.x)
        trans = torch.exp(self.x)
        assert self._verify(orig, trans, "all(orig > 0)")


# ── precheck 后 mr.checked 被正确设置 ──────────────────────────────────────

class TestPrecheckSetsChecked():
    def test_precheck_marks_checked(self):
        import torch.nn.functional as F
        from unittest.mock import patch
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator
        from deepmt.analysis.verification.mr_prechecker import MRPreChecker

        pool = MRTemplatePool()
        candidates = pool.generate_mr_candidates("relu", operator_func=F.relu)
        assert len(candidates) >= 1

        for mr in candidates:
            assert mr.verified is False  # 初始状态

        # 直接构建 _apply_precheck 所需的最小对象，绕过 LLM 初始化
        with patch.object(OperatorMRGenerator, "__init__", return_value=None):
            gen = OperatorMRGenerator.__new__(OperatorMRGenerator)
            gen.prechecker = MRPreChecker()

        operator_ir = OperatorIR(name="relu")

        verified = gen._apply_precheck(
            operator_func=F.relu,
            mr_candidates=candidates,
            operator_ir=operator_ir,
            framework="pytorch",
        )

        # precheck 至少应筛出一条成立的 MR（如 unary_scale_linear / identity）
        assert len(verified) >= 1, "Expected at least one MR to pass precheck"
        for mr in verified:
            assert mr.checked is True, \
                f"MR '{mr.description}' should have checked=True after precheck"


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

    def test_import_bare_name_resolves_via_plugin(self):
        from deepmt.commands.mr import _try_import_operator
        func = _try_import_operator("relu", "pytorch")
        assert func is not None and callable(func)


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
