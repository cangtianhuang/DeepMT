"""单元测试：ModelVerifier 的各 oracle 类型验证。

不依赖 LLM、网络；需要 PyTorch 环境。
"""

import numpy as np
import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="需要 PyTorch"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def verifier():
    from deepmt.analysis.verification.model_verifier import ModelVerifier
    return ModelVerifier()


def _make_mr(oracle_expr: str, tolerance: float = 1e-5):
    """构造用于测试的最小 MetamorphicRelation。"""
    from deepmt.ir.schema import MetamorphicRelation
    return MetamorphicRelation(
        id="test-mr-001",
        description="test MR",
        subject_name="TestModel",
        subject_type="model",
        transform_code="lambda x: x",
        oracle_expr=oracle_expr,
        layer="model",
        tolerance=tolerance,
    )


# ── prediction_consistent 测试 ────────────────────────────────────────────────

class TestPredictionConsistent:

    def test_passes_same_argmax(self, verifier):
        orig = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]])
        trans = torch.tensor([[1.8, 0.3, 0.0], [0.2, 2.8, 0.1]])
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True
        assert result.mismatched_elements == 0

    def test_fails_different_argmax(self, verifier):
        orig = torch.tensor([[2.0, 0.5, 0.1]])      # argmax=0
        trans = torch.tensor([[0.1, 3.0, 0.2]])     # argmax=1
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert result.mismatched_elements == 1
        assert "PREDICTION_MISMATCH" in result.detail

    def test_partial_mismatch(self, verifier):
        orig = torch.tensor([[2.0, 0.1], [0.1, 3.0]])
        trans = torch.tensor([[0.1, 2.0], [0.2, 2.8]])   # sample 0 flipped
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert result.mismatched_elements == 1
        assert result.total_elements == 2

    def test_1d_raises_error(self, verifier):
        """1D 输出（非 batch 格式）应返回失败并带 SHAPE_ERROR。"""
        orig = torch.tensor([2.0, 0.5, 0.1])
        trans = torch.tensor([1.8, 0.3, 0.0])
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert "SHAPE_ERROR" in result.detail

    def test_numpy_input(self, verifier):
        """验证器应接受 numpy array 输入。"""
        orig = np.array([[2.0, 0.5, 0.1]])
        trans = np.array([[1.8, 0.3, 0.0]])
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True


# ── topk_consistent 测试 ──────────────────────────────────────────────────────

class TestTopkConsistent:

    def test_passes_same_topk(self, verifier):
        # top-2: [0,1] vs [0,1]（同集合，不同顺序）
        orig = torch.tensor([[3.0, 2.0, 0.1]])
        trans = torch.tensor([[2.5, 2.8, 0.0]])
        mr = _make_mr("topk_consistent:2")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True

    def test_fails_different_topk(self, verifier):
        orig = torch.tensor([[3.0, 2.0, 0.1]])   # top-2: {0,1}
        trans = torch.tensor([[3.0, 0.1, 2.0]])  # top-2: {0,2}
        mr = _make_mr("topk_consistent:2")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert "TOPK_MISMATCH" in result.detail

    def test_topk_default_k3(self, verifier):
        orig = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.1]])
        trans = torch.tensor([[3.9, 3.1, 2.1, 0.9, 0.0]])
        mr = _make_mr("topk_consistent")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True  # top-3 = {0,1,2} for both


# ── output_close 测试 ─────────────────────────────────────────────────────────

class TestOutputClose:

    def test_passes_identical(self, verifier):
        t = torch.tensor([[1.0, 2.0, 3.0]])
        mr = _make_mr("output_close:1e-6")
        result = verifier.verify(t, t.clone(), mr)
        assert result.passed is True
        assert result.actual_diff < 1e-9

    def test_passes_within_tolerance(self, verifier):
        orig = torch.tensor([[1.0, 2.0]])
        trans = torch.tensor([[1.0 + 1e-7, 2.0 + 1e-7]])
        mr = _make_mr("output_close:1e-5")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True

    def test_fails_outside_tolerance(self, verifier):
        orig = torch.tensor([[1.0, 2.0]])
        trans = torch.tensor([[2.0, 3.0]])  # diff = 1.0
        mr = _make_mr("output_close:1e-5")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert "NUMERICAL_DEVIATION" in result.detail

    def test_shape_mismatch_fails(self, verifier):
        orig = torch.tensor([[1.0, 2.0]])
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        mr = _make_mr("output_close:1e-5")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False
        assert "SHAPE_MISMATCH" in result.detail


# ── output_order_invariant 测试 ───────────────────────────────────────────────

class TestOutputOrderInvariant:

    def test_passes_batch_flip(self, verifier):
        """model(flip(x)) 应等于 flip(model(x))。"""
        # 模拟：orig = model(x)，trans = model(x.flip(0))
        # oracle 验证：orig.flip(0) ≈ trans
        orig = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        trans = orig.flip(0).clone()   # = [[3,4],[1,2]]
        mr = _make_mr("output_order_invariant")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True

    def test_fails_non_inverted(self, verifier):
        """若 trans 不等于 orig.flip(0)，应失败。"""
        orig = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        trans = orig.clone()  # 不翻转
        mr = _make_mr("output_order_invariant")
        result = verifier.verify(orig, trans, mr)
        assert result.passed is False


# ── 未知 oracle 退化 ───────────────────────────────────────────────────────────

class TestUnknownOracle:

    def test_unknown_oracle_falls_back_to_close(self, verifier):
        orig = torch.tensor([[1.0, 2.0]])
        trans = orig.clone()
        mr = _make_mr("unknown_oracle_type", tolerance=1e-5)
        result = verifier.verify(orig, trans, mr)
        assert result.passed is True  # 退化为 output_close，相同应通过


# ── OracleResult 字段检查 ──────────────────────────────────────────────────────

class TestOracleResultFields:

    def test_result_fields_populated(self, verifier):
        orig = torch.tensor([[1.0, 2.0, 3.0]])
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        mr = _make_mr("prediction_consistent")
        result = verifier.verify(orig, trans, mr)
        assert hasattr(result, "passed")
        assert hasattr(result, "expr")
        assert hasattr(result, "actual_diff")
        assert hasattr(result, "tolerance")
        assert hasattr(result, "detail")
        assert result.expr == "prediction_consistent"
