"""测试 OracleEvaluator 的功能"""

import unittest
import torch
import torch.nn.functional as F
from core.oracle_evaluator import OracleEvaluator


class TestOracleEvaluator(unittest.TestCase):
    """测试 OracleEvaluator 类"""

    def setUp(self):
        """初始化测试环境"""
        self.oracle_eval = OracleEvaluator()
        self.test_input = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0])

    def test_equal_expression(self):
        """测试相等比较表达式"""
        expr = "orig == trans"
        oracle_func = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )

        # 测试相等情况
        x = self.test_input
        orig = F.relu(x)
        trans = F.relu(x)  # 相同输入，应该相等

        result = oracle_func(orig, trans, x, 1e-6)
        self.assertTrue(result, "相同输入应该产生相等的输出")

    def test_proportional_expression(self):
        """测试比例关系表达式"""
        expr = "trans == 3 * orig"
        oracle_func = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )

        # 测试比例关系
        x = self.test_input
        orig = F.relu(x)
        trans = 3 * orig

        result = oracle_func(orig, trans, x, 1e-6)
        self.assertTrue(result, "trans 应该等于 3 * orig")

    def test_monotonicity_expression(self):
        """测试单调性表达式"""
        expr = "trans >= orig"
        oracle_func = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )

        # 测试单调性：使用正数输入，确保 x+1 > x 且都为正
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(x + 1)  # 输入增加，输出应该不减

        result = oracle_func(orig, trans, x, 1e-6)
        if isinstance(result, torch.Tensor):
            result = result.item() if result.numel() == 1 else bool(result.all())
        self.assertTrue(result, "trans 应该 >= orig")

    def test_all_zero_expression(self):
        """测试所有元素为零的表达式"""
        expr = "all(trans == 0)"
        oracle_func = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )

        # 测试全零情况
        x = torch.tensor([-5.0, -2.0, -1.0])
        orig = F.relu(x)
        trans = torch.zeros_like(x)

        result = oracle_func(orig, trans, x, 1e-6)
        # 修复：result 可能是 tensor，需要转换为 Python bool
        if isinstance(result, torch.Tensor):
            result = result.item() if result.numel() == 1 else bool(result.all())
        self.assertTrue(result, "trans 的所有元素应该为零")

    def test_combined_expression(self):
        """测试组合性质表达式"""
        expr = "orig + trans == abs(x)"
        oracle_func = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )

        # 测试组合性质：relu(x) + relu(-x) == |x|
        x = self.test_input
        orig = F.relu(x)
        trans = F.relu(-x)

        result = oracle_func(orig, trans, x, 1e-6)
        self.assertTrue(result, "orig + trans 应该等于 abs(x)")

    def test_invalid_expression(self):
        """测试无效表达式"""
        expr = "invalid syntax here"

        # 应该能够编译（可能会在运行时失败）
        try:
            oracle_func = self.oracle_eval.compile_expression(
                expr=expr,
                framework="pytorch",
                tolerance=1e-6,
            )
            # 如果编译成功，尝试执行应该失败
            x = self.test_input
            orig = F.relu(x)
            trans = F.relu(x)
            with self.assertRaises(Exception):
                oracle_func(orig, trans, x, 1e-6)
        except Exception:
            # 编译失败也是可以接受的
            pass

    def test_different_frameworks(self):
        """测试不同框架的表达式编译"""
        expr = "orig == trans"

        # 测试 PyTorch
        oracle_func_pytorch = self.oracle_eval.compile_expression(
            expr=expr,
            framework="pytorch",
            tolerance=1e-6,
        )
        self.assertIsNotNone(oracle_func_pytorch)

        # 测试 TensorFlow（可能未实现）
        try:
            oracle_func_tf = self.oracle_eval.compile_expression(
                expr=expr,
                framework="tensorflow",
                tolerance=1e-6,
            )
            self.assertIsNotNone(oracle_func_tf)
        except NotImplementedError:
            # TensorFlow 支持可能未实现
            pass


if __name__ == "__main__":
    unittest.main()
