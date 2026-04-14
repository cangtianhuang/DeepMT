"""
MutationTester 单元测试

验证重点：
  1. 各变异类型对 relu MR 的检出情况（核心逻辑验证）
  2. create_mutant_func 工厂函数输出正确
  3. MutantResult 数据结构与序列化
  4. run_all_mutants 遍历全部变异类型

设计思路：
  - 使用真实 PyTorch 算子（relu）和知识库中已有的 MR，确保端到端有效性
  - mock MRRepository 以隔离对文件系统的依赖
  - 关注"检出率是否合理"，而非精确数值
"""

import uuid
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from deepmt.analysis.reporting.mutation_tester import (
    MutantResult,
    MutantType,
    MutationTester,
    create_mutant_func,
)
from deepmt.core.results_manager import ResultsManager
from deepmt.engine.batch_test_runner import BatchTestRunner
from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_mr(transform_code: str, oracle_expr: str, verified: bool = True) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description="test MR",
        transform_code=transform_code,
        oracle_expr=oracle_expr,
        verified=verified,
        transform=eval(transform_code),
    )


# relu 知识库中已有的两条 MR
RELU_LINEARITY_MR = _make_mr(
    transform_code="lambda k: {**k, 'input': 2.0 * k['input']}",
    oracle_expr="trans == 2.0 * orig",
)
RELU_NONNEGATIVITY_MR = _make_mr(
    transform_code="lambda k: {**k, 'input': -k['input']}",
    oracle_expr="orig + trans == abs(x)",
)


def _make_tester_with_mrs(*mrs) -> MutationTester:
    """构造注入指定 MR 的 MutationTester（mock repo + mock results_manager）。"""
    mock_repo = MagicMock(spec=MRRepository)
    mock_repo.load.return_value = list(mrs)
    mock_rm = MagicMock(spec=ResultsManager)
    runner = BatchTestRunner(repo=mock_repo, catalog=MagicMock(), results_manager=mock_rm)
    return MutationTester(repo=mock_repo, runner=runner)


# ── create_mutant_func 工厂测试 ────────────────────────────────────────────────

class TestCreateMutantFunc:
    """验证变异函数工厂输出的张量形状与值语义正确。"""

    def setup_method(self):
        self.x = torch.tensor([1.0, -2.0, 3.0])
        self.real_func = F.relu

    def test_negate_output(self):
        mutant = create_mutant_func(self.real_func, MutantType.NEGATE_OUTPUT)
        result = mutant(input=self.x)
        expected = -self.real_func(self.x)
        assert torch.allclose(result, expected)

    def test_add_constant(self):
        mutant = create_mutant_func(self.real_func, MutantType.ADD_CONSTANT, const=5.0)
        result = mutant(input=self.x)
        expected = self.real_func(self.x) + 5.0
        assert torch.allclose(result, expected)

    def test_scale_wrong(self):
        mutant = create_mutant_func(self.real_func, MutantType.SCALE_WRONG, scale=3.0)
        result = mutant(input=self.x)
        expected = 3.0 * self.real_func(self.x)
        assert torch.allclose(result, expected)

    def test_identity(self):
        mutant = create_mutant_func(self.real_func, MutantType.IDENTITY)
        result = mutant(input=self.x)
        assert torch.allclose(result, self.x)

    def test_zero_output(self):
        mutant = create_mutant_func(self.real_func, MutantType.ZERO_OUTPUT)
        result = mutant(input=self.x)
        assert torch.allclose(result, torch.zeros_like(self.x))


# ── MutationTester 检出率测试 ─────────────────────────────────────────────────

class TestMutationTesterDetection:
    """
    验证各变异类型对 relu MR 的检出情况。

    预期行为：
      - NEGATE_OUTPUT + 非负性 MR → 检出（-relu(x) + -relu(-x) ≠ |x|）
      - ADD_CONSTANT + 线性 MR → 检出（relu(2x)+1 ≠ 2*(relu(x)+1)）
      - SCALE_WRONG(k=3) + 线性 MR → 检出（3*relu(2x) ≠ 2*(3*relu(x))，但等式两边都乘3，实际仍成立 → 用非负性 MR 检出）
      - IDENTITY + 非负性 MR → 检出（x + (-x) = 0 ≠ |x| 对非零 x）
      - ZERO_OUTPUT + 非负性 MR → 检出（0 + 0 = 0 ≠ |x| 对非零 x）
    """

    def test_negate_detected_by_nonnegativity(self):
        """取反输出 → 非负性 MR 检出"""
        tester = _make_tester_with_mrs(RELU_NONNEGATIVITY_MR)
        result = tester.run("torch.nn.functional.relu", MutantType.NEGATE_OUTPUT, "pytorch", n_samples=10)

        assert result.mr_count == 1
        assert result.detected, f"negate 变异未被检出，检出率={result.detection_rate:.1%}"
        assert result.detection_rate > 0.5, f"检出率过低: {result.detection_rate:.1%}"

    def test_add_constant_detected_by_linearity(self):
        """常数偏置 → 线性 MR 检出（relu(2x)+1 ≠ 2*(relu(x)+1) for x>0）"""
        tester = _make_tester_with_mrs(RELU_LINEARITY_MR)
        result = tester.run(
            "torch.nn.functional.relu", MutantType.ADD_CONSTANT, "pytorch",
            n_samples=10, const=100.0,  # 大偏置确保检出
        )

        assert result.detected, f"add_const 变异未被检出，检出率={result.detection_rate:.1%}"

    def test_identity_detected_by_nonnegativity(self):
        """恒等函数 → 非负性 MR 检出（x + (-x) = 0 ≠ |x|）"""
        tester = _make_tester_with_mrs(RELU_NONNEGATIVITY_MR)
        result = tester.run("torch.nn.functional.relu", MutantType.IDENTITY, "pytorch", n_samples=10)

        assert result.detected, f"identity 变异未被检出，检出率={result.detection_rate:.1%}"

    def test_zero_detected_by_linearity(self):
        """零输出 → 线性 MR 检出（对正值输入：zero(2x)=0 ≠ 2*zero(x)=0...实际0==0→通过）

        注意：zero_output 对线性 MR 0=2*0 是平凡满足。
        使用非负性 MR：zero(x)+zero(-x)=0 ≠ |x|（对非零输入检出）。
        """
        tester = _make_tester_with_mrs(RELU_NONNEGATIVITY_MR)
        result = tester.run("torch.nn.functional.relu", MutantType.ZERO_OUTPUT, "pytorch", n_samples=10)

        assert result.detected, f"zero_output 变异未被检出，检出率={result.detection_rate:.1%}"

    def test_correct_impl_passes(self):
        """正确实现（不注入变异，直接用 BatchTestRunner）应全部通过"""
        mock_repo = MagicMock(spec=MRRepository)
        mock_repo.load.return_value = [RELU_LINEARITY_MR, RELU_NONNEGATIVITY_MR]
        mock_rm = MagicMock(spec=ResultsManager)
        runner = BatchTestRunner(repo=mock_repo, catalog=MagicMock(), results_manager=mock_rm)

        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=10)
        assert summary.failed == 0, f"正确实现出现失败: {summary.failed}/{summary.total_cases}"

    def test_run_all_mutants_returns_all_types(self):
        """run_all_mutants 应返回与 MutantType 数量相同的结果"""
        tester = _make_tester_with_mrs(RELU_NONNEGATIVITY_MR)
        results = tester.run_all_mutants(
            "torch.nn.functional.relu", "pytorch", n_samples=5
        )
        assert len(results) == len(MutantType)

    def test_no_mrs_returns_empty_result(self):
        """知识库无 MR 时应返回空结果而非报错"""
        mock_repo = MagicMock(spec=MRRepository)
        mock_repo.load.return_value = []
        mock_rm = MagicMock(spec=ResultsManager)
        runner = BatchTestRunner(repo=mock_repo, catalog=MagicMock(), results_manager=mock_rm)
        tester = MutationTester(repo=mock_repo, runner=runner)

        result = tester.run("torch.nn.functional.relu", MutantType.NEGATE_OUTPUT, "pytorch")
        assert result.mr_count == 0
        assert result.total_cases == 0
        assert result.detection_rate == 0.0


# ── MutantResult 数据结构测试 ──────────────────────────────────────────────────

class TestMutantResult:
    def test_to_dict_structure(self):
        r = MutantResult(
            operator_name="torch.nn.functional.relu",
            mutant_type=MutantType.NEGATE_OUTPUT,
            framework="pytorch",
            n_samples=10,
            mr_count=2,
            detected_cases=8,
            total_cases=10,
            errors=0,
            mr_details=[{"mr_id": "abc", "detected_cases": 8, "total_cases": 10}],
        )
        d = r.to_dict()
        assert d["operator"] == "torch.nn.functional.relu"
        assert d["mutant_type"] == "negate"
        assert d["detection_rate"] == pytest.approx(0.8)
        assert d["detected"] is True
        assert len(d["mr_details"]) == 1

    def test_detection_rate_zero_cases(self):
        r = MutantResult(
            operator_name="op", mutant_type=MutantType.NEGATE_OUTPUT,
            framework="pytorch", n_samples=10,
            mr_count=0, detected_cases=0, total_cases=0, errors=0,
        )
        assert r.detection_rate == 0.0
        assert r.detected is False
