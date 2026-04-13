"""
PaddlePlugin 单元测试

覆盖：
  - 所有 FrameworkPlugin 抽象方法
  - 算子解析（torch.* 名称 + paddle.* 名称）
  - 跨框架基本流程（CrossFrameworkTester with paddlepaddle）
  - DiffType 分类字段
  - is_available / supported_operators

依赖：paddlepaddle 已安装（若未安装则跳过相关测试）
"""

import math
import pytest
import numpy as np

try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PADDLE_AVAILABLE, reason="paddlepaddle not installed")


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def plugin():
    from deepmt.plugins.paddle_plugin import PaddlePlugin
    return PaddlePlugin()


@pytest.fixture
def float_tensor():
    return paddle.to_tensor([1.0, -1.0, 2.0, -2.0], dtype="float32")


@pytest.fixture
def pos_tensor():
    """正值张量（用于 log/sqrt 等需要正数的算子）"""
    return paddle.to_tensor([0.5, 1.0, 1.5, 2.0], dtype="float32")


# ── 基础接口测试 ──────────────────────────────────────────────────────────────


class TestPaddlePluginBasicInterface:
    def test_is_available(self):
        from deepmt.plugins.paddle_plugin import PaddlePlugin
        assert PaddlePlugin.is_available() is True

    def test_supported_operators_not_empty(self):
        from deepmt.plugins.paddle_plugin import PaddlePlugin
        ops = PaddlePlugin.supported_operators()
        assert len(ops) > 0
        assert "torch.nn.functional.relu" in ops
        assert "torch.exp" in ops

    def test_to_tensor_from_numpy(self, plugin):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = plugin._to_tensor(arr)
        assert isinstance(t, paddle.Tensor)
        np.testing.assert_array_almost_equal(t.numpy(), arr)

    def test_to_tensor_from_list(self, plugin):
        t = plugin._to_tensor([1.0, 2.0, 3.0])
        assert isinstance(t, paddle.Tensor)
        assert t.shape[0] == 3

    def test_to_tensor_from_scalar_float(self, plugin):
        t = plugin._to_tensor(3.14)
        assert isinstance(t, paddle.Tensor)

    def test_to_tensor_passthrough(self, plugin, float_tensor):
        t = plugin._to_tensor(float_tensor)
        assert t is float_tensor

    def test_to_numpy(self, plugin, float_tensor):
        arr = plugin.to_numpy(float_tensor)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, [1.0, -1.0, 2.0, -2.0])

    def test_to_numpy_from_plain_array(self, plugin):
        arr = np.array([1.0, 2.0])
        result = plugin.to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_get_shape(self, plugin, float_tensor):
        shape = plugin.get_shape(float_tensor)
        assert shape == (4,)

    def test_get_shape_2d(self, plugin):
        t = paddle.rand([3, 4])
        assert plugin.get_shape(t) == (3, 4)


# ── make_tensor 测试 ──────────────────────────────────────────────────────────


class TestMakeTensor:
    def test_make_float32(self, plugin):
        t = plugin.make_tensor((4, 4), "float32")
        assert isinstance(t, paddle.Tensor)
        assert t.shape == [4, 4]
        assert str(t.dtype) == "paddle.float32"

    def test_make_float64(self, plugin):
        t = plugin.make_tensor((2, 3), "float64")
        assert str(t.dtype) == "paddle.float64"

    def test_make_int64(self, plugin):
        t = plugin.make_tensor((3,), "int64")
        assert str(t.dtype) == "paddle.int64"

    def test_make_with_value_range(self, plugin):
        t = plugin.make_tensor((100,), "float32", value_range=(0.0, 1.0))
        arr = t.numpy()
        assert float(arr.min()) >= 0.0 - 1e-5
        assert float(arr.max()) <= 1.0 + 1e-5

    def test_make_int_with_value_range(self, plugin):
        t = plugin.make_tensor((100,), "int64", value_range=(-5, 5))
        arr = t.numpy()
        assert int(arr.min()) >= -5
        assert int(arr.max()) <= 5

    def test_make_no_range_gives_varying_values(self, plugin):
        t = plugin.make_tensor((50,), "float32")
        # 50 个随机值应有差异
        assert float(t.numpy().std()) > 0.0


# ── allclose 测试 ─────────────────────────────────────────────────────────────


class TestAllclose:
    def test_identical_tensors(self, plugin, float_tensor):
        result = plugin.allclose(float_tensor, float_tensor, atol=1e-5)
        assert result.passed is True
        assert result.max_abs_diff == pytest.approx(0.0, abs=1e-9)
        assert result.mismatched_elements == 0

    def test_within_tolerance(self, plugin):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([1.0001, 2.0001], dtype="float32")
        result = plugin.allclose(a, b, atol=1e-3)
        assert result.passed is True

    def test_outside_tolerance(self, plugin):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([2.0, 3.0], dtype="float32")
        result = plugin.allclose(a, b, atol=1e-5)
        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(1.0, rel=1e-3)
        assert result.mismatched_elements == 2

    def test_shape_mismatch_returns_failed(self, plugin):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        result = plugin.allclose(a, b, atol=1e-5)
        assert result.passed is False

    def test_from_numpy_arrays(self, plugin):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        result = plugin.allclose(a, b, atol=1e-5)
        assert result.passed is True


# ── eval_expr 测试 ────────────────────────────────────────────────────────────


class TestEvalExpr:
    def test_simple_equality(self, plugin, float_tensor):
        orig = float_tensor
        trans = paddle.abs(float_tensor)
        x = float_tensor
        result = plugin.eval_expr("abs(orig) - trans", orig, trans, x)
        assert isinstance(result, paddle.Tensor)

    def test_abs_expression(self, plugin, float_tensor):
        orig = float_tensor
        trans = orig
        x = orig
        result = plugin.eval_expr("abs(orig)", orig, trans, x)
        expected = np.abs(float_tensor.numpy())
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_exp_expression(self, plugin):
        t = paddle.to_tensor([0.0, 1.0], dtype="float32")
        result = plugin.eval_expr("exp(orig)", t, t, t)
        np.testing.assert_array_almost_equal(result.numpy(), [1.0, math.e], decimal=5)

    def test_returns_tensor(self, plugin, float_tensor):
        result = plugin.eval_expr("all(orig == orig)", float_tensor, float_tensor, float_tensor)
        assert isinstance(result, paddle.Tensor)


# ── element_compare 测试 ──────────────────────────────────────────────────────


class TestElementCompare:
    def test_ge_all_satisfy(self, plugin):
        a = paddle.to_tensor([2.0, 3.0, 4.0], dtype="float32")
        b = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        result = plugin.element_compare(a, b, ">=")
        assert result.passed is True
        assert result.mismatched_elements == 0

    def test_ge_some_fail(self, plugin):
        a = paddle.to_tensor([1.0, 0.0, 4.0], dtype="float32")
        b = paddle.to_tensor([2.0, 2.0, 3.0], dtype="float32")
        result = plugin.element_compare(a, b, ">=")
        assert result.passed is False
        assert result.mismatched_elements == 2

    def test_ne_operator(self, plugin):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([1.0, 3.0], dtype="float32")
        result = plugin.element_compare(a, b, "!=")
        assert result.passed is False   # a[0]==b[0] → 未满足 !=
        assert result.mismatched_elements == 1

    def test_invalid_op_raises(self, plugin):
        a = paddle.to_tensor([1.0])
        with pytest.raises(ValueError):
            plugin.element_compare(a, a, "~~")


# ── 算子解析测试 ──────────────────────────────────────────────────────────────


class TestResolveOperator:
    def test_resolve_torch_relu(self, plugin):
        fn = plugin._resolve_operator("torch.nn.functional.relu")
        x = paddle.to_tensor([-1.0, 0.0, 1.0], dtype="float32")
        result = fn(input=x)
        np.testing.assert_array_almost_equal(result.numpy(), [0.0, 0.0, 1.0])

    def test_resolve_torch_exp(self, plugin):
        fn = plugin._resolve_operator("torch.exp")
        x = paddle.to_tensor([0.0, 1.0], dtype="float32")
        result = fn(input=x)
        np.testing.assert_array_almost_equal(result.numpy(), [1.0, math.e], decimal=5)

    def test_resolve_torch_abs(self, plugin):
        fn = plugin._resolve_operator("torch.abs")
        x = paddle.to_tensor([-1.0, -2.0, 3.0], dtype="float32")
        result = fn(input=x)
        np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0])

    def test_resolve_paddle_native(self, plugin):
        fn = plugin._resolve_operator("paddle.exp")
        x = paddle.to_tensor([0.0, 1.0], dtype="float32")
        result = fn(x)
        np.testing.assert_array_almost_equal(result.numpy(), [1.0, math.e], decimal=5)

    def test_resolve_unknown_raises(self, plugin):
        with pytest.raises(ValueError, match="PaddlePlugin"):
            plugin._resolve_operator("torch.nonexistent_op_xyz")

    @pytest.mark.parametrize("op_name", [
        "torch.nn.functional.relu",
        "torch.nn.functional.sigmoid",
        "torch.nn.functional.softmax",
        "torch.nn.functional.silu",
        "torch.exp",
        "torch.log",
        "torch.sqrt",
        "torch.abs",
        "torch.tanh",
        "torch.sin",
        "torch.cos",
        "torch.neg",
        "torch.sign",
        "torch.floor",
        "torch.ceil",
        "torch.round",
    ])
    def test_resolve_all_supported_unary_ops(self, plugin, op_name, pos_tensor):
        fn = plugin._resolve_operator(op_name)
        result = fn(input=pos_tensor)
        assert isinstance(result, paddle.Tensor)

    @pytest.mark.parametrize("op_name,kw", [
        ("torch.add",  {"input": None, "other": None}),
        ("torch.mul",  {"input": None, "other": None}),
        ("torch.div",  {"input": None, "other": None}),
    ])
    def test_resolve_binary_ops(self, plugin, pos_tensor, op_name, kw):
        fn = plugin._resolve_operator(op_name)
        kw["input"] = pos_tensor
        kw["other"] = pos_tensor
        result = fn(**kw)
        assert isinstance(result, paddle.Tensor)


# ── 数值正确性对比（与 PyTorch 结果比对） ─────────────────────────────────────


class TestNumericalAccuracy:
    """验证 PaddlePlugin 的数值计算与 PyTorch 结果一致（浮点容差内）。"""

    @pytest.fixture
    def torch_plugin(self):
        from deepmt.plugins.pytorch_plugin import PyTorchPlugin
        return PyTorchPlugin()

    @pytest.mark.parametrize("op_torch, op_paddle", [
        ("torch.nn.functional.relu",    "torch.nn.functional.relu"),
        ("torch.nn.functional.sigmoid", "torch.nn.functional.sigmoid"),
        ("torch.exp",                   "torch.exp"),
        ("torch.abs",                   "torch.abs"),
        ("torch.tanh",                  "torch.tanh"),
        ("torch.sin",                   "torch.sin"),
        ("torch.cos",                   "torch.cos"),
        ("torch.neg",                   "torch.neg"),
        ("torch.sign",                  "torch.sign"),
        ("torch.floor",                 "torch.floor"),
        ("torch.ceil",                  "torch.ceil"),
    ])
    def test_numerical_parity_with_pytorch(self, torch_plugin, plugin, op_torch, op_paddle):
        import torch
        data = np.array([0.5, 1.0, 1.5, -0.5, -1.0], dtype=np.float32)
        t_torch = torch.tensor(data)
        t_paddle = paddle.to_tensor(data)

        fn_torch = torch_plugin._resolve_operator(op_torch)
        fn_paddle = plugin._resolve_operator(op_paddle)

        out_torch = fn_torch(input=t_torch)
        out_paddle = fn_paddle(input=t_paddle)

        np.testing.assert_array_almost_equal(
            torch_plugin.to_numpy(out_torch),
            plugin.to_numpy(out_paddle),
            decimal=5,
            err_msg=f"Numerical mismatch for {op_torch}",
        )

    def test_relu_zero_boundary(self, plugin):
        t = paddle.to_tensor([-0.001, 0.0, 0.001], dtype="float32")
        fn = plugin._resolve_operator("torch.nn.functional.relu")
        out = fn(input=t)
        np.testing.assert_array_almost_equal(out.numpy(), [0.0, 0.0, 0.001])

    def test_log_positive_values(self, plugin):
        t = paddle.to_tensor([1.0, math.e, math.e ** 2], dtype="float32")
        fn = plugin._resolve_operator("torch.log")
        out = fn(input=t)
        np.testing.assert_array_almost_equal(out.numpy(), [0.0, 1.0, 2.0], decimal=5)

    def test_sqrt_values(self, plugin):
        t = paddle.to_tensor([1.0, 4.0, 9.0], dtype="float32")
        fn = plugin._resolve_operator("torch.sqrt")
        out = fn(input=t)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 2.0, 3.0], decimal=5)


# ── 跨框架测试集成（CrossFrameworkTester with paddle） ──────────────────────


class TestCrossFrameworkWithPaddle:
    """验证 CrossFrameworkTester 可以将 paddlepaddle 作为第二框架运行。"""

    def test_get_paddle_backend(self):
        from deepmt.analysis.cross_framework_tester import CrossFrameworkTester
        tester = CrossFrameworkTester()
        backend = tester._get_backend("paddlepaddle")
        from deepmt.plugins.paddle_plugin import PaddlePlugin
        assert isinstance(backend, PaddlePlugin)

    def test_get_paddle_alias(self):
        from deepmt.analysis.cross_framework_tester import CrossFrameworkTester
        tester = CrossFrameworkTester()
        backend = tester._get_backend("paddle")
        from deepmt.plugins.paddle_plugin import PaddlePlugin
        assert isinstance(backend, PaddlePlugin)

    def test_normalize_framework_paddle(self):
        from deepmt.analysis.cross_framework_tester import CrossFrameworkTester
        tester = CrossFrameworkTester()
        assert tester._normalize_framework("paddle") == "paddlepaddle"
        assert tester._normalize_framework("paddlepaddle") == "paddlepaddle"
        assert tester._normalize_framework("numpy") == "numpy"


# ── DiffType 常量测试 ─────────────────────────────────────────────────────────


class TestDiffType:
    def test_diff_type_constants_defined(self):
        from deepmt.analysis.cross_framework_tester import DiffType
        assert DiffType.NUMERIC_DIFF == "numeric_diff"
        assert DiffType.SHAPE_MISMATCH == "shape_mismatch"
        assert DiffType.DTYPE_MISMATCH == "dtype_mismatch"
        assert DiffType.BEHAVIOR_DIFF == "behavior_diff"
        assert DiffType.EXCEPTION_F1 == "exception_f1"
        assert DiffType.EXCEPTION_F2 == "exception_f2"
        assert DiffType.BOTH_EXCEPTION == "both_exception"


# ── CrossConsistencyResult 差异字段测试 ────────────────────────────────────────


class TestCrossConsistencyDiffFields:
    def test_diff_type_counts_in_result(self):
        from deepmt.analysis.cross_framework_tester import CrossConsistencyResult
        r = CrossConsistencyResult(
            operator="torch.abs",
            framework1="pytorch",
            framework2="paddlepaddle",
            mr_id="test-id",
            mr_description="test",
            oracle_expr="trans >= 0",
            n_samples=10,
            both_pass=8, only_f1_pass=1, only_f2_pass=0, both_fail=1,
            errors=0,
            output_max_diff=1e-6,
            output_mean_diff=5e-7,
            output_close=True,
            diff_type_counts={"behavior_diff": 1, "numeric_diff": 0},
        )
        assert r.diff_type_counts["behavior_diff"] == 1
        assert r.consistency_rate == pytest.approx(0.9, rel=1e-3)

    def test_diff_type_counts_serialized_in_to_dict(self):
        from deepmt.analysis.cross_framework_tester import CrossConsistencyResult
        r = CrossConsistencyResult(
            operator="torch.abs",
            framework1="pytorch",
            framework2="paddlepaddle",
            mr_id="test-id",
            mr_description="test",
            oracle_expr="trans >= 0",
            n_samples=5,
            both_pass=5, only_f1_pass=0, only_f2_pass=0, both_fail=0,
            errors=0,
            output_max_diff=1e-7,
            output_mean_diff=5e-8,
            output_close=True,
            diff_type_counts={"numeric_diff": 2},
        )
        d = r.to_dict()
        assert "diff_type_counts" in d
        assert d["diff_type_counts"]["numeric_diff"] == 2

    def test_cross_session_diff_type_summary(self):
        from deepmt.analysis.cross_framework_tester import CrossConsistencyResult, CrossSessionResult
        mr1 = CrossConsistencyResult(
            operator="op", framework1="pytorch", framework2="paddlepaddle",
            mr_id="id1", mr_description="d1", oracle_expr="e1", n_samples=5,
            both_pass=5, only_f1_pass=0, only_f2_pass=0, both_fail=0, errors=0,
            output_max_diff=0.0, output_mean_diff=0.0, output_close=True,
            diff_type_counts={"numeric_diff": 2},
        )
        mr2 = CrossConsistencyResult(
            operator="op", framework1="pytorch", framework2="paddlepaddle",
            mr_id="id2", mr_description="d2", oracle_expr="e2", n_samples=5,
            both_pass=4, only_f1_pass=1, only_f2_pass=0, both_fail=0, errors=0,
            output_max_diff=0.01, output_mean_diff=0.005, output_close=False,
            diff_type_counts={"behavior_diff": 1, "numeric_diff": 1},
        )
        session = CrossSessionResult(
            session_id="test", timestamp="2026-01-01T00:00:00",
            operator="op", framework1="pytorch", framework2="paddlepaddle",
            n_samples=5, mr_results=[mr1, mr2],
        )
        summary = session.diff_type_summary
        assert summary["numeric_diff"] == 3
        assert summary["behavior_diff"] == 1

    def test_cross_session_to_dict_includes_diff_summary(self):
        from deepmt.analysis.cross_framework_tester import CrossSessionResult
        session = CrossSessionResult(
            session_id="test", timestamp="2026-01-01T00:00:00",
            operator="op", framework1="pytorch", framework2="paddlepaddle",
            n_samples=5, mr_results=[],
        )
        d = session.to_dict()
        assert "diff_type_summary" in d
