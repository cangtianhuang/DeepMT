"""核心模块单元测试：OracleEvaluator"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn.functional as F

from deepmt.core.oracle_evaluator import OracleEvaluator


def _bool(result) -> bool:
    """将可能为 Tensor/numpy 的结果转为 Python bool"""
    if isinstance(result, torch.Tensor):
        return result.item() if result.numel() == 1 else bool(result.all())
    try:
        import numpy as np
        if isinstance(result, np.generic):
            return bool(result)
    except ImportError:
        pass
    return bool(result)


@pytest.fixture(scope="module")
def evaluator():
    return OracleEvaluator()


class TestOracleEvaluator:

    def test_equal_expression(self, evaluator):
        x = torch.tensor([-5.0, 0.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(x)
        fn = evaluator.compile_expression("orig == trans", framework="pytorch", tolerance=1e-6)
        assert _bool(fn(orig, trans, x, 1e-6))

    def test_proportional_expression(self, evaluator):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = 3 * orig
        fn = evaluator.compile_expression("trans == 3 * orig", framework="pytorch", tolerance=1e-6)
        assert _bool(fn(orig, trans, x, 1e-6))

    def test_monotonicity_expression(self, evaluator):
        x = torch.tensor([1.0, 2.0, 3.0])
        orig = F.relu(x)
        trans = F.relu(x + 1)
        fn = evaluator.compile_expression("trans >= orig", framework="pytorch", tolerance=1e-6)
        assert _bool(fn(orig, trans, x, 1e-6))

    def test_relu_composition(self, evaluator):
        """relu(x) + relu(-x) == abs(x)"""
        x = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0])
        orig = F.relu(x)
        trans = F.relu(-x)
        fn = evaluator.compile_expression("orig + trans == abs(x)", framework="pytorch", tolerance=1e-6)
        assert _bool(fn(orig, trans, x, 1e-6))

    def test_invalid_expression_raises_or_returns_falsy(self, evaluator):
        """无效表达式应编译后运行失败或返回 False，不应静默成功"""
        try:
            fn = evaluator.compile_expression("invalid_syntax_xyz", framework="pytorch", tolerance=1e-6)
            x = torch.tensor([1.0])
            orig = F.relu(x)
            trans = F.relu(x)
            with pytest.raises(Exception):
                fn(orig, trans, x, 1e-6)
        except Exception:
            pass  # 编译阶段失败也可接受

    def test_compile_returns_callable(self, evaluator):
        fn = evaluator.compile_expression("orig == trans", framework="pytorch", tolerance=1e-6)
        assert callable(fn)
