"""
NumpyPlugin 单元测试

验证重点：
  1. _to_tensor / make_tensor / to_numpy / get_shape 基础接口
  2. allclose：正确判断通过/失败，精度指标计算
  3. eval_expr：oracle 子表达式正确求值
  4. element_compare：不等式比较分类
  5. _resolve_operator：已注册算子返回可调用，未知算子抛 ValueError
  6. 核心激活函数的数值正确性
"""

import numpy as np
import pytest

from deepmt.plugins.numpy_plugin import NumpyPlugin, OPERATOR_EQUIVALENCE_MAP


@pytest.fixture
def plugin():
    return NumpyPlugin()


# ── 基础张量接口 ──────────────────────────────────────────────────────────────

class TestTensorOps:
    def test_to_tensor_numpy(self, plugin):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        t = plugin._to_tensor(arr)
        assert t.dtype == np.float32
        np.testing.assert_allclose(t, arr.astype(np.float32))

    def test_to_tensor_list(self, plugin):
        t = plugin._to_tensor([1.0, -1.0])
        assert isinstance(t, np.ndarray)
        assert t.dtype == np.float32

    def test_make_tensor_shape(self, plugin):
        t = plugin.make_tensor((3, 4))
        assert t.shape == (3, 4)
        assert t.dtype == np.float32

    def test_make_tensor_value_range(self, plugin):
        t = plugin.make_tensor((100,), value_range=(0.0, 1.0))
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_make_tensor_int(self, plugin):
        t = plugin.make_tensor((20,), dtype="int32", value_range=(-5, 5))
        assert t.dtype == np.int32
        assert int(t.min()) >= -5
        assert int(t.max()) <= 5

    def test_to_numpy_passthrough(self, plugin):
        arr = np.array([1.0, 2.0])
        result = plugin.to_numpy(arr)
        assert result.dtype == float  # upcast to float64

    def test_get_shape(self, plugin):
        t = plugin.make_tensor((2, 3, 4))
        assert plugin.get_shape(t) == (2, 3, 4)


# ── allclose ──────────────────────────────────────────────────────────────────

class TestAllclose:
    def test_identical_arrays_pass(self, plugin):
        a = np.array([1.0, 2.0, 3.0])
        result = plugin.allclose(a, a.copy(), atol=1e-6)
        assert result.passed is True
        assert result.max_abs_diff == pytest.approx(0.0)
        assert result.mismatched_elements == 0

    def test_small_diff_within_tolerance(self, plugin):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0 + 1e-7])
        result = plugin.allclose(a, b, atol=1e-6)
        assert result.passed is True

    def test_large_diff_fails(self, plugin):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        result = plugin.allclose(a, b, atol=1e-6)
        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(1.0)
        assert result.mismatched_elements == 1
        assert result.total_elements == 2

    def test_mismatched_ratio(self, plugin):
        a = np.zeros(4)
        b = np.ones(4) * 0.5
        result = plugin.allclose(a, b, atol=1e-6)
        assert result.mismatched_ratio == pytest.approx(1.0)


# ── eval_expr ─────────────────────────────────────────────────────────────────

class TestEvalExpr:
    def test_orig_plus_trans(self, plugin):
        orig = np.array([1.0, 2.0])
        trans = np.array([3.0, 4.0])
        x = orig.copy()
        result = plugin.eval_expr("orig + trans", orig, trans, x)
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_abs_x(self, plugin):
        orig = np.array([1.0])
        x = np.array([-2.0])
        result = plugin.eval_expr("abs(x)", orig, orig, x)
        np.testing.assert_allclose(result, [2.0])

    def test_scalar_exp(self, plugin):
        orig = np.array([1.0])
        result = plugin.eval_expr("exp(1)", orig, orig, orig)
        np.testing.assert_allclose(float(result), np.e, rtol=1e-5)


# ── element_compare ──────────────────────────────────────────────────────────

class TestElementCompare:
    def test_ge_all_satisfy(self, plugin):
        a = np.array([0.0, 1.0, 2.0])
        b = np.zeros(3)
        result = plugin.element_compare(a, b, ">=")
        assert result.passed is True
        assert result.mismatched_elements == 0

    def test_ge_partial_fail(self, plugin):
        a = np.array([-1.0, 1.0])
        b = np.zeros(2)
        result = plugin.element_compare(a, b, ">=")
        assert result.passed is False
        assert result.mismatched_elements == 1

    def test_lt(self, plugin):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = plugin.element_compare(a, b, "<")
        assert result.passed is False  # 3 >= 2 fails


# ── operator resolution ───────────────────────────────────────────────────────

class TestResolveOperator:
    def test_relu_resolved(self, plugin):
        fn = plugin._resolve_operator("relu")
        assert callable(fn)

    def test_exp_resolved(self, plugin):
        fn = plugin._resolve_operator("exp")
        assert callable(fn)

    def test_unknown_raises(self, plugin):
        with pytest.raises(ValueError, match="NumpyPlugin"):
            plugin._resolve_operator("nonexistent_op_xyz_abc")

    def test_operator_equivalence_map_not_empty(self):
        assert len(OPERATOR_EQUIVALENCE_MAP) > 0


# ── 数值正确性 ────────────────────────────────────────────────────────────────

class TestNumericalCorrectness:
    """验证 NumPy 算子与 PyTorch 算子数值一致（允许 float32 精度范围内的误差）。"""

    def test_relu(self, plugin):
        fn = plugin._resolve_operator("relu")
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = fn(input=x)
        np.testing.assert_allclose(out, [0, 0, 0, 1, 2])

    def test_exp(self, plugin):
        fn = plugin._resolve_operator("exp")
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        out = fn(input=x)
        np.testing.assert_allclose(out, np.exp([0.0, 1.0, 2.0]), rtol=1e-5)

    def test_sigmoid_range(self, plugin):
        fn = plugin._resolve_operator("sigmoid")
        x = np.linspace(-5, 5, 20).astype(np.float32)
        out = fn(input=x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_tanh_antisymmetry(self, plugin):
        fn = plugin._resolve_operator("tanh")
        x = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        out = fn(input=x)
        out_neg = fn(input=-x)
        np.testing.assert_allclose(out, -out_neg, rtol=1e-5)

    def test_abs_nonnegative(self, plugin):
        fn = plugin._resolve_operator("abs")
        x = np.random.randn(10).astype(np.float32)
        out = fn(input=x)
        assert float(out.min()) >= 0.0

    def test_sqrt_positive_input(self, plugin):
        fn = plugin._resolve_operator("sqrt")
        x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        out = fn(input=x)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_softmax_sum_to_one(self, plugin):
        fn = plugin._resolve_operator("softmax")
        x = np.random.randn(4, 5).astype(np.float32)
        out = fn(input=x)
        row_sums = out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), rtol=1e-5)

    def test_log_exp_inverse(self, plugin):
        exp_fn = plugin._resolve_operator("exp")
        log_fn = plugin._resolve_operator("log")
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        result = log_fn(input=exp_fn(input=x))
        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_supported_operators_list(self, plugin):
        ops = NumpyPlugin.supported_operators()
        assert "relu" in ops
        assert "exp" in ops
        assert len(ops) >= 8
