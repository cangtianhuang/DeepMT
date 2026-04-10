"""
BatchTestRunner 单元测试

测试覆盖：
  - run_operator：对有 MR 的算子正常执行（使用真实 PyTorch 后端）
  - run_operator：算子无 MR 时返回空结果
  - run_operator：verified_only 过滤
  - run_batch：遍历多个算子
  - OperatorTestSummary.to_dict：序列化格式正确
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from deepmt.engine.batch_test_runner import BatchTestRunner, OperatorTestSummary, MRTestSummary
from deepmt.ir.schema import MetamorphicRelation


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_mr(transform_code: str, oracle_expr: str, verified: bool = True) -> MetamorphicRelation:
    """构造一个最简 MR 对象"""
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description="test MR",
        transform_code=transform_code,
        oracle_expr=oracle_expr,
        verified=verified,
        transform=eval(transform_code),
    )


# ── 基本逻辑测试（mock repo / catalog）──────────────────────────────────────

class TestBatchTestRunnerBasic:
    """使用 mock repo 和真实 PyTorch 后端测试 BatchTestRunner 逻辑"""

    def _make_runner_with_mr(self, mr: MetamorphicRelation, operator_name: str) -> BatchTestRunner:
        """构造一个 repo 中包含指定 MR 的 BatchTestRunner。"""
        mock_repo = MagicMock()
        mock_repo.load.return_value = [mr]
        mock_repo.list_operators_by_framework.return_value = [operator_name]

        mock_catalog = MagicMock()
        mock_catalog.get_operator_info.return_value = None  # 无 input_specs，使用默认值
        mock_catalog.get_by_category.return_value = []

        mock_results_manager = MagicMock()

        return BatchTestRunner(
            repo=mock_repo,
            catalog=mock_catalog,
            results_manager=mock_results_manager,
        )

    def test_run_operator_relu_linearity(self):
        """relu 正齐次性 MR：ReLU(2*x) == 2*ReLU(x) 应全部通过"""
        mr = _make_mr(
            transform_code="lambda k: {**k, 'input': 2.0 * k['input']}",
            oracle_expr="trans == 2.0 * orig",
        )
        runner = self._make_runner_with_mr(mr, "torch.nn.functional.relu")
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=5)

        assert summary.mr_count == 1
        assert summary.total_cases == 5
        assert summary.passed == 5
        assert summary.failed == 0
        assert summary.errors == 0
        assert len(summary.mr_summaries) == 1
        assert summary.pass_rate == 1.0

    def test_run_operator_relu_nonnegativity(self):
        """relu 非负性 MR：ReLU(x) + ReLU(-x) == |x| 应全部通过"""
        mr = _make_mr(
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="orig + trans == abs(x)",
        )
        runner = self._make_runner_with_mr(mr, "torch.nn.functional.relu")
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=5)

        assert summary.passed == 5
        assert summary.failed == 0

    def test_run_operator_failing_mr(self):
        """故意错误的 oracle（不等式永远假）应全部失败"""
        mr = _make_mr(
            transform_code="lambda k: {**k, 'input': 2.0 * k['input']}",
            oracle_expr="trans == orig",  # 错误：2*x 的 relu != x 的 relu
        )
        runner = self._make_runner_with_mr(mr, "torch.nn.functional.relu")
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=5)

        # 对于正值输入：relu(2x)=2*relu(x) != relu(x)（不相等），所以应该失败
        assert summary.total_cases == 5
        assert summary.failed > 0

    def test_run_operator_no_mrs(self):
        """知识库中无 MR 时返回零计数摘要"""
        mock_repo = MagicMock()
        mock_repo.load.return_value = []

        runner = BatchTestRunner(
            repo=mock_repo,
            catalog=MagicMock(),
            results_manager=MagicMock(),
        )
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=5)

        assert summary.mr_count == 0
        assert summary.total_cases == 0
        assert summary.passed == 0

    def test_run_operator_verified_only_filters(self):
        """verified_only=True 时应过滤掉未验证的 MR"""
        unverified_mr = _make_mr(
            transform_code="lambda k: {**k, 'input': 2.0 * k['input']}",
            oracle_expr="trans == 2.0 * orig",
            verified=False,
        )
        mock_repo = MagicMock()
        mock_repo.load.return_value = [unverified_mr]

        runner = BatchTestRunner(
            repo=mock_repo,
            catalog=MagicMock(),
            results_manager=MagicMock(),
        )
        summary = runner.run_operator(
            "torch.nn.functional.relu", "pytorch", n_samples=5, verified_only=True
        )

        assert summary.mr_count == 0
        assert summary.total_cases == 0

    def test_run_operator_invalid_transform(self):
        """transform 返回非 dict 时应记录为 error 而非崩溃"""
        mr = _make_mr(
            transform_code="lambda k: list(k.values())",  # 返回 list，非 dict
            oracle_expr="trans == orig",
        )
        runner = self._make_runner_with_mr(mr, "torch.nn.functional.relu")
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=3)

        assert summary.errors == 3
        assert summary.total_cases == 0

    def test_run_batch(self):
        """run_batch 应遍历 repo 中的所有算子"""
        mr = _make_mr(
            transform_code="lambda k: {**k, 'input': 2.0 * k['input']}",
            oracle_expr="trans == 2.0 * orig",
        )
        mock_repo = MagicMock()
        mock_repo.list_operators_by_framework.return_value = [
            "torch.nn.functional.relu",
            "torch.nn.functional.sigmoid",
        ]
        mock_repo.load.return_value = [mr]

        runner = BatchTestRunner(
            repo=mock_repo,
            catalog=MagicMock(),
            results_manager=MagicMock(),
        )
        runner.catalog.get_by_category.return_value = []

        results = runner.run_batch("pytorch", n_samples=3)

        assert len(results) == 2
        for s in results:
            assert s.mr_count == 1
            assert s.n_samples == 3


# ── OperatorTestSummary 序列化测试 ────────────────────────────────────────────

class TestOperatorTestSummary:
    def test_to_dict_structure(self):
        summary = OperatorTestSummary(
            operator="torch.nn.functional.relu",
            framework="pytorch",
            mr_count=2,
            n_samples=10,
            total_cases=20,
            passed=18,
            failed=2,
            errors=0,
            mr_summaries=[
                MRTestSummary(mr_id="abc", description="test", passed=10, failed=0, errors=0),
                MRTestSummary(mr_id="def", description="test2", passed=8, failed=2, errors=0),
            ],
        )
        d = summary.to_dict()
        assert d["operator"] == "torch.nn.functional.relu"
        assert d["passed"] == 18
        assert d["failed"] == 2
        assert d["pass_rate"] == pytest.approx(0.9)
        assert len(d["mr_summaries"]) == 2
        assert d["mr_summaries"][0]["mr_id"] == "abc"

    def test_pass_rate_zero_cases(self):
        """total_cases=0 时 pass_rate 应返回 0.0 而非 ZeroDivisionError"""
        summary = OperatorTestSummary(
            operator="op", framework="pytorch",
            mr_count=0, n_samples=10,
            total_cases=0, passed=0, failed=0, errors=0,
        )
        assert summary.pass_rate == 0.0
