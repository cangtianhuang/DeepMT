"""核心模块单元测试：MRVerifier"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn.functional as F

from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.ir import MetamorphicRelation, OracleResult
from deepmt.plugins.framework_plugin import CompareResult
from deepmt.plugins.pytorch_plugin import PyTorchPlugin


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def backend():
    return PyTorchPlugin()


@pytest.fixture(scope="module")
def verifier():
    return MRVerifier()


def _mr(oracle_expr: str, tolerance: float = 1e-6) -> MetamorphicRelation:
    return MetamorphicRelation(
        id="test",
        description="test",
        transform=lambda k: k,
        transform_code="",
        oracle_expr=oracle_expr,
        tolerance=tolerance,
    )


# ── MRVerifier：基础正确性 ────────────────────────────────────────────────────


class TestMRVerifier:

    def test_equal_expression(self, verifier, backend):
        x = torch.tensor([-5.0, 0.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(x)
        result = verifier.verify(orig, trans, _mr("orig == trans"), backend, x_input=x)
        assert isinstance(result, OracleResult)
        assert result.passed

    def test_proportional_expression(self, verifier, backend):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = 3 * orig
        result = verifier.verify(orig, trans, _mr("trans == 3 * orig"), backend, x_input=x)
        assert result.passed

    def test_monotonicity_expression(self, verifier, backend):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x + 1)
        result = verifier.verify(orig, trans, _mr("trans >= orig"), backend, x_input=x)
        assert result.passed

    def test_relu_composition(self, verifier, backend):
        """relu(x) + relu(-x) == abs(x)"""
        x = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(-x)
        result = verifier.verify(
            orig, trans, _mr("orig + trans == abs(x)"), backend, x_input=x
        )
        assert result.passed

    def test_scalar_rhs_broadcast(self, verifier, backend):
        """rhs 为标量时（如 orig * trans == 1）需 broadcast 处理，不应崩溃"""
        x = torch.tensor([1.0, 1.0, 1.0])
        orig = torch.ones(3)
        trans = torch.ones(3)
        result = verifier.verify(orig, trans, _mr("orig * trans == 1"), backend, x_input=x)
        assert result.passed

    def test_failing_expression(self, verifier, backend):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x) + 100
        result = verifier.verify(orig, trans, _mr("orig == trans"), backend, x_input=x)
        assert not result.passed
        assert result.actual_diff > 0

    def test_empty_oracle_expr_treated_as_equality(self, verifier, backend):
        """oracle_expr 为空时等同于 'orig == trans'，经由 allclose 路径"""
        x = torch.tensor([1.0, 2.0])
        orig = F.relu(x)
        trans = F.relu(x)
        result = verifier.verify(orig, trans, _mr(""), backend)
        assert result.passed
        assert result.expr == "orig == trans"

    def test_all_quantifier_stripped(self, verifier, backend):
        """all(trans >= orig) 剥离 all() 后等同于 trans >= orig"""
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x + 1)
        result = verifier.verify(orig, trans, _mr("all(trans >= orig)"), backend, x_input=x)
        assert result.passed

    def test_shape_mismatch(self, verifier, backend):
        orig = torch.tensor([1.0, 2.0])
        trans = torch.tensor([1.0, 2.0, 3.0])
        result = verifier.verify(orig, trans, _mr("orig == trans"), backend)
        assert not result.passed
        assert "SHAPE_MISMATCH" in result.detail

    def test_result_fields_complete(self, verifier, backend):
        """OracleResult 所有字段均存在且类型正确"""
        x = torch.tensor([1.0])
        orig = F.relu(x)
        result = verifier.verify(orig, orig, _mr("orig == trans"), backend)
        assert isinstance(result.passed, bool)
        assert isinstance(result.expr, str)
        assert isinstance(result.actual_diff, float)
        assert isinstance(result.tolerance, float)
        assert isinstance(result.detail, str)
        assert isinstance(result.max_rel_diff, float)
        assert isinstance(result.mismatched_elements, int)
        assert isinstance(result.total_elements, int)


# ── MRVerifier：诊断字段填充 ──────────────────────────────────────────────────


class TestOracleDiagnostics:

    def test_equality_diagnostics_on_pass(self, verifier, backend):
        """== 通过时：mismatched=0, total>0, max_rel_diff=0"""
        orig = torch.tensor([1.0, 2.0, 3.0])
        result = verifier.verify(orig, orig.clone(), _mr("orig == trans"), backend)
        assert result.passed
        assert result.mismatched_elements == 0
        assert result.total_elements == 3
        assert result.max_rel_diff == pytest.approx(0.0, abs=1e-9)
        assert result.actual_diff == pytest.approx(0.0, abs=1e-9)

    def test_equality_diagnostics_on_fail(self, verifier, backend):
        """== 失败时：mismatched/total/max_rel_diff 均有效填充"""
        orig = torch.tensor([1.0, 2.0, 3.0])
        trans = torch.tensor([1.0, 2.0, 5.0])  # 第三个元素差 2
        result = verifier.verify(orig, trans, _mr("orig == trans"), backend)
        assert not result.passed
        assert result.actual_diff == pytest.approx(2.0)
        assert result.mismatched_elements == 1
        assert result.total_elements == 3
        assert result.max_rel_diff > 0
        assert "max_abs" in result.detail
        assert "max_rel" in result.detail
        assert "mismatched" in result.detail

    def test_proportional_diagnostics_on_fail(self, verifier, backend):
        """trans == 3*orig 失败时，diagnostics 来自 allclose(trans, 3*orig)"""
        orig = torch.tensor([1.0, 2.0])
        trans = torch.tensor([3.0, 7.0])  # 第二个应为 6.0，差 1
        result = verifier.verify(orig, trans, _mr("trans == 3 * orig"), backend)
        assert not result.passed
        assert result.actual_diff == pytest.approx(1.0)
        assert result.mismatched_elements == 1
        assert result.total_elements == 2

    def test_inequality_diagnostics_on_fail(self, verifier, backend):
        """trans >= orig 失败时：mismatched_elements 计数违规元素"""
        orig = torch.tensor([5.0, 2.0, 3.0])
        trans = torch.tensor([1.0, 3.0, 3.0])  # 第一个元素违反 >=
        result = verifier.verify(orig, trans, _mr("trans >= orig"), backend)
        assert not result.passed
        assert result.mismatched_elements == 1
        assert result.total_elements == 3
        assert "INEQUALITY_VIOLATION" in result.detail

    def test_empty_oracle_detail_on_fail(self, verifier, backend):
        """oracle_expr 为空且不通过时，detail 含 max_abs/max_rel/mismatched"""
        orig = torch.tensor([1.0, 2.0])
        trans = torch.tensor([1.0, 5.0])
        result = verifier.verify(orig, trans, _mr(""), backend)
        assert not result.passed
        assert "max_abs" in result.detail
        assert "max_rel" in result.detail
        assert "mismatched" in result.detail

    def test_actual_diff_populated(self, verifier, backend):
        """actual_diff 应始终被填充（非 inf）"""
        x = torch.tensor([1.0, 2.0])
        orig = F.relu(x)
        trans = F.relu(x * 2)
        result = verifier.verify(orig, trans, _mr("orig == trans"), backend, x_input=x)
        assert result.actual_diff >= 0


# ── CompareResult / allclose 测试 ─────────────────────────────────────────────


class TestAllclose:

    def test_equal_tensors_passed(self, backend):
        a = torch.tensor([1.0, 2.0, 3.0])
        r = backend.allclose(a, a.clone(), atol=1e-6)
        assert isinstance(r, CompareResult)
        assert r.passed
        assert r.max_abs_diff == pytest.approx(0.0, abs=1e-9)
        assert r.max_rel_diff == pytest.approx(0.0, abs=1e-9)
        assert r.mismatched_elements == 0
        assert r.total_elements == 3

    def test_different_tensors_fails(self, backend):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 5.0])
        r = backend.allclose(a, b, atol=1e-6)
        assert not r.passed
        assert r.max_abs_diff == pytest.approx(2.0)
        assert r.mismatched_elements == 1
        assert r.total_elements == 3
        assert r.mismatched_ratio == pytest.approx(1 / 3)

    def test_within_atol_passes(self, backend):
        a = torch.tensor([0.0, 1.0])
        b = torch.tensor([0.05, 1.05])
        r = backend.allclose(a, b, atol=0.1)
        assert r.passed

    def test_rtol_respected(self, backend):
        a = torch.tensor([100.0])
        b = torch.tensor([101.0])
        r_pass = backend.allclose(a, b, atol=0.0, rtol=0.02)
        assert r_pass.passed
        r_fail = backend.allclose(a, b, atol=0.0, rtol=0.005)
        assert not r_fail.passed

    def test_dtype_mismatch_handled(self, backend):
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([1.0], dtype=torch.float64)
        r = backend.allclose(a, b, atol=1e-5)
        assert r.passed

    def test_shape_mismatch_returns_inf(self, backend):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        r = backend.allclose(a, b, atol=1e-6)
        assert not r.passed
        assert r.max_abs_diff == float("inf")
        assert r.total_elements == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
