"""核心模块单元测试：MRVerifier"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn.functional as F

from deepmt.analysis.mr_verifier import MRVerifier
from deepmt.ir.schema import MetamorphicRelation, OracleResult
from deepmt.plugins.framework_plugin import CompareResult
from deepmt.plugins.pytorch_plugin import PyTorchPlugin


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def plugin():
    return PyTorchPlugin()


@pytest.fixture(scope="module")
def verifier():
    return MRVerifier()


def _mr(oracle_expr: str, tolerance: float = 1e-6) -> MetamorphicRelation:
    """构造只含 oracle_expr 的最小 MR（用于验证器单测）"""
    return MetamorphicRelation(
        id="test",
        description="test",
        transform=lambda k: k,
        transform_code="",
        oracle_expr=oracle_expr,
        tolerance=tolerance,
    )


# ── MRVerifier 测试 ───────────────────────────────────────────────────────────


class TestMRVerifier:

    def test_equal_expression(self, verifier, plugin):
        x = torch.tensor([-5.0, 0.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(x)
        result = verifier.verify(orig, trans, _mr("orig == trans"), plugin, x_input=x)
        assert isinstance(result, OracleResult)
        assert result.passed

    def test_proportional_expression(self, verifier, plugin):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = 3 * orig
        result = verifier.verify(
            orig, trans, _mr("trans == 3 * orig"), plugin, x_input=x
        )
        assert result.passed

    def test_monotonicity_expression(self, verifier, plugin):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x + 1)
        result = verifier.verify(orig, trans, _mr("trans >= orig"), plugin, x_input=x)
        assert result.passed

    def test_relu_composition(self, verifier, plugin):
        """relu(x) + relu(-x) == abs(x)"""
        x = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(-x)
        result = verifier.verify(
            orig, trans, _mr("orig + trans == abs(x)"), plugin, x_input=x
        )
        assert result.passed

    def test_failing_expression(self, verifier, plugin):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x) + 100  # 故意造假
        result = verifier.verify(orig, trans, _mr("orig == trans"), plugin, x_input=x)
        assert not result.passed
        assert result.actual_diff > 0

    def test_actual_diff_populated(self, verifier, plugin):
        """actual_diff 应始终被填充"""
        x = torch.tensor([1.0, 2.0])
        orig = F.relu(x)
        trans = F.relu(x * 2)
        result = verifier.verify(orig, trans, _mr("orig == trans"), plugin, x_input=x)
        assert result.actual_diff >= 0

    def test_shape_mismatch(self, verifier, plugin):
        orig = torch.tensor([1.0, 2.0])
        trans = torch.tensor([1.0, 2.0, 3.0])
        result = verifier.verify(orig, trans, _mr("orig == trans"), plugin)
        assert not result.passed
        assert "SHAPE_MISMATCH" in result.detail

    def test_empty_oracle_expr_uses_allclose(self, verifier, plugin):
        """oracle_expr 为空时走 plugin.allclose 默认等值路径"""
        x = torch.tensor([1.0, 2.0])
        orig = F.relu(x)
        trans = F.relu(x)
        result = verifier.verify(orig, trans, _mr(""), plugin)
        assert result.passed

    def test_result_fields_complete(self, verifier, plugin):
        """OracleResult 所有字段均存在且类型正确"""
        x = torch.tensor([1.0])
        orig = F.relu(x)
        result = verifier.verify(orig, orig, _mr("orig == trans"), plugin)
        assert isinstance(result.passed, bool)
        assert isinstance(result.expr, str)
        assert isinstance(result.actual_diff, float)
        assert isinstance(result.tolerance, float)
        assert isinstance(result.detail, str)


# ── CompareResult / allclose 测试 ─────────────────────────────────────────────


class TestAllclose:

    def test_equal_tensors_passed(self, plugin):
        a = torch.tensor([1.0, 2.0, 3.0])
        r = plugin.allclose(a, a.clone(), atol=1e-6)
        assert isinstance(r, CompareResult)
        assert r.passed
        assert r.max_abs_diff == pytest.approx(0.0, abs=1e-9)
        assert r.max_rel_diff == pytest.approx(0.0, abs=1e-9)
        assert r.mismatched_elements == 0
        assert r.total_elements == 3

    def test_different_tensors_fails(self, plugin):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 5.0])  # 最后一个元素差 2
        r = plugin.allclose(a, b, atol=1e-6)
        assert not r.passed
        assert r.max_abs_diff == pytest.approx(2.0)
        assert r.mismatched_elements == 1
        assert r.total_elements == 3
        assert r.mismatched_ratio == pytest.approx(1 / 3)

    def test_within_atol_passes(self, plugin):
        a = torch.tensor([0.0, 1.0])
        b = torch.tensor([0.05, 1.05])
        r = plugin.allclose(a, b, atol=0.1)
        assert r.passed

    def test_rtol_respected(self, plugin):
        a = torch.tensor([100.0])
        b = torch.tensor([101.0])  # abs_diff=1, rel_diff=0.01
        # atol=0, rtol=0.02 → threshold=2 → pass
        r_pass = plugin.allclose(a, b, atol=0.0, rtol=0.02)
        assert r_pass.passed
        # atol=0, rtol=0.005 → threshold=0.5 → fail
        r_fail = plugin.allclose(a, b, atol=0.0, rtol=0.005)
        assert not r_fail.passed

    def test_dtype_mismatch_handled(self, plugin):
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([1.0], dtype=torch.float64)
        r = plugin.allclose(a, b, atol=1e-5)
        assert r.passed

    def test_shape_mismatch_returns_inf(self, plugin):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        r = plugin.allclose(a, b, atol=1e-6)
        assert not r.passed
        assert r.max_abs_diff == float("inf")
        assert r.total_elements == 0

    def test_compare_result_detail_in_verify(self, verifier, plugin):
        """oracle_expr 为空且不通过时 detail 含 max_abs/max_rel/mismatched"""
        orig = torch.tensor([1.0, 2.0])
        trans = torch.tensor([1.0, 5.0])
        result = verifier.verify(orig, trans, _mr(""), plugin)
        assert not result.passed
        assert "max_abs" in result.detail
        assert "max_rel" in result.detail
        assert "mismatched" in result.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
