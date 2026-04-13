"""G 阶段单元测试：ResultsManager 存储迁移与 RunManifest 持久化。

测试覆盖：
  - test_results 表新列（run_id / framework_version / random_seed）的写入与读取
  - run_manifests 表的写入与查询
  - 旧接口（store_result 不传新参数）仍可正常工作
  - migrate_g_phase 迁移脚本的 dry_run 路径
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from deepmt.core.results_manager import ResultsManager
from deepmt.core.run_manifest import RunManifest
from deepmt.ir.schema import MetamorphicRelation, OperatorIR, OracleResult
from deepmt.migrations.migrate_g_phase import run as migrate_run


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


# ── RunManifest 持久化 ────────────────────────────────────────────────────────


class TestRunManifestStorage:
    def test_store_and_retrieve(self, manager):
        manifest = RunManifest(
            subject_name="torch.add",
            framework="pytorch",
            framework_version="2.3.0",
            random_seed=42,
            n_samples=10,
            mr_ids=["mr-1", "mr-2"],
            notes="RQ1",
        )
        manager.store_run_manifest(manifest)
        result = manager.get_run_manifest(manifest.run_id)
        assert result is not None
        assert result["subject_name"] == "torch.add"
        assert result["framework_version"] == "2.3.0"
        assert result["random_seed"] == 42
        assert result["n_samples"] == 10
        assert result["mr_ids"] == ["mr-1", "mr-2"]
        assert result["notes"] == "RQ1"

    def test_get_nonexistent_returns_none(self, manager):
        result = manager.get_run_manifest("nonexistent-id")
        assert result is None

    def test_upsert_run_manifest(self, manager):
        manifest = RunManifest(subject_name="torch.relu", notes="v1")
        manager.store_run_manifest(manifest)
        manifest.notes = "v2"
        manager.store_run_manifest(manifest)
        result = manager.get_run_manifest(manifest.run_id)
        assert result["notes"] == "v2"

    def test_env_summary_deserialized_as_dict(self, manager):
        manifest = RunManifest(subject_name="torch.relu")
        manager.store_run_manifest(manifest)
        result = manager.get_run_manifest(manifest.run_id)
        assert isinstance(result["env_summary"], dict)
        assert "python" in result["env_summary"]


# ── 迁移脚本 ──────────────────────────────────────────────────────────────────


class TestMigrationScript:
    def test_dry_run_no_exception(self, tmp_path):
        db = str(tmp_path / "test.db")
        # 先创建数据库
        ResultsManager(db_path=db)
        # 再 dry_run 不应抛出
        migrate_run(db_path=db, dry_run=True)

    def test_migrate_existing_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        import sqlite3
        # 建一个只有旧 test_results 表（无新列）的数据库
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE test_results (
               id INTEGER PRIMARY KEY,
               ir_name TEXT, ir_type TEXT, framework TEXT,
               mr_id TEXT, status TEXT, timestamp TEXT
            )"""
        )
        conn.commit()
        conn.close()

        migrate_run(db_path=db)

        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(test_results)")
        columns = {row[1] for row in cursor.fetchall()}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "run_id" in columns
        assert "framework_version" in columns
        assert "random_seed" in columns
        assert "run_manifests" in tables

    def test_migrate_idempotent(self, tmp_path):
        db = str(tmp_path / "test.db")
        ResultsManager(db_path=db)
        # 执行两次迁移，不应报错
        migrate_run(db_path=db)
        migrate_run(db_path=db)

    def test_migrate_nonexistent_db_skipped(self, tmp_path):
        db = str(tmp_path / "does_not_exist.db")
        # 不应抛出，只是跳过
        migrate_run(db_path=db)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
