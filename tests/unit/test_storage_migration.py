"""G 阶段单元测试：ResultsManager 存储功能与新列字段写入。

测试覆盖：
  - test_results 表新列（run_id / framework_version / random_seed）的写入与读取
  - 旧接口（store_result 不传新参数）仍可正常工作
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import MetamorphicRelation, OperatorIR, OracleResult


def _make_mr(mr_id: str = "mr-1") -> MetamorphicRelation:
    return MetamorphicRelation(
        id=mr_id,
        description="test MR",
        oracle_expr="orig == trans",
        transform_code="lambda k: k",
    )


def _make_oracle(passed: bool = True) -> OracleResult:
    return OracleResult(
        passed=passed,
        expr="orig == trans",
        actual_diff=0.0 if passed else 1.0,
        tolerance=1e-6,
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def manager(db_path):
    return ResultsManager(db_path=db_path)


# ── 旧接口向后兼容 ────────────────────────────────────────────────────────────


class TestStoreResultBackwardCompat:
    def test_store_without_new_params(self, manager):
        op = OperatorIR(name="torch.relu")
        mr = _make_mr()
        oracle = _make_oracle(passed=True)
        # 不传 run_id / framework_version / random_seed，不应抛出
        manager.store_result(op, [(mr, oracle)], framework="pytorch")

    def test_failed_result_stored(self, manager):
        op = OperatorIR(name="torch.relu")
        mr = _make_mr()
        oracle = _make_oracle(passed=False)
        manager.store_result(op, [(mr, oracle)], framework="pytorch")
        failures = manager.get_failed_tests()
        assert len(failures) == 1
        assert failures[0]["status"] == "FAIL"

    def test_summary_updated(self, manager):
        op = OperatorIR(name="torch.add")
        results = [
            (_make_mr("mr-1"), _make_oracle(True)),
            (_make_mr("mr-2"), _make_oracle(False)),
        ]
        manager.store_result(op, results, framework="pytorch")
        summary = manager.get_summary("torch.add")
        assert len(summary) == 1
        assert summary[0]["total_tests"] == 2
        assert summary[0]["passed_tests"] == 1
        assert summary[0]["failed_tests"] == 1


# ── 新列写入与读取 ────────────────────────────────────────────────────────────


class TestNewColumns:
    def test_store_with_run_id(self, manager, db_path):
        op = OperatorIR(name="torch.relu")
        mr = _make_mr()
        oracle = _make_oracle(True)
        manager.store_result(
            op, [(mr, oracle)],
            framework="pytorch",
            run_id="test-run-001",
        )

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM test_results WHERE mr_id = 'mr-1'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "test-run-001"

    def test_store_with_framework_version(self, manager, db_path):
        op = OperatorIR(name="torch.relu")
        manager.store_result(
            op, [(_make_mr(), _make_oracle())],
            framework="pytorch",
            framework_version="2.3.0",
        )

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT framework_version FROM test_results")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == "2.3.0"

    def test_store_with_random_seed(self, manager, db_path):
        op = OperatorIR(name="torch.relu")
        manager.store_result(
            op, [(_make_mr(), _make_oracle())],
            framework="pytorch",
            random_seed=42,
        )

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT random_seed FROM test_results")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == 42

    def test_new_columns_null_when_not_provided(self, manager, db_path):
        op = OperatorIR(name="torch.relu")
        manager.store_result(
            op, [(_make_mr(), _make_oracle())],
            framework="pytorch",
        )

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT run_id, framework_version, random_seed FROM test_results")
        row = cursor.fetchone()
        conn.close()
        assert row == (None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
