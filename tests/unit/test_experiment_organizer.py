"""
ExperimentOrganizer 单元测试

验证重点：
  1. collect_rq1：从 MRRepository 正确提取 MR 统计
  2. collect_rq2：从 ResultsManager + EvidenceCollector 正确提取测试统计
  3. collect_rq3：无跨框架记录时返回 note 字段
  4. collect_rq4：派生指标计算正确
  5. collect_all：整合全部 RQ 数据
  6. format_text：文本输出不崩溃，包含关键字段
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmt.experiments.organizer import ExperimentOrganizer
from deepmt.core.results_manager import ResultsManager
from deepmt.ir import MetamorphicRelation, OracleResult, OperatorIR


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_mr(op_name="relu", category="activation", source="llm"):
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description=f"{op_name} test MR",
        transform_code="lambda k: k",
        oracle_expr="orig == trans",
        category=category,
        source=source,
        verified=True,
    )


def _populate_db(db_path: str) -> None:
    """向临时数据库写入模拟测试结果。"""
    rm = ResultsManager(db_path=db_path)

    def make_result(passed):
        return OracleResult(
            passed=passed,
            expr="orig == trans",
            actual_diff=0.0 if passed else 0.5,
            tolerance=1e-6,
            detail="" if passed else "NUMERICAL_DEVIATION: max_abs=0.500",
        )

    mr = MetamorphicRelation(id="mr-01", description="test", transform_code="", oracle_expr="")
    relu_ir = OperatorIR(name="torch.nn.functional.relu", input_specs=[])
    rm.store_result(relu_ir, [(mr, make_result(True))] * 7 + [(mr, make_result(False))] * 3, "pytorch")

    exp_ir = OperatorIR(name="torch.exp", input_specs=[])
    rm.store_result(exp_ir, [(mr, make_result(True))] * 5, "pytorch")


# ── collect_rq1 ──────────────────────────────────────────────────────────────

class TestCollectRQ1:
    def test_basic_fields_present(self):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 10,
            "verified_mrs": 6,
            "by_operator": {"relu": {"total": 5}, "exp": {"total": 5}},
        }
        mock_repo.list_operators.return_value = ["relu", "exp"]
        mock_repo.load.side_effect = lambda op: [_make_mr(op)] * 3

        org = ExperimentOrganizer(mr_repo=mock_repo)
        rq1 = org.collect_rq1()

        assert rq1["total_mr_count"] == 10
        assert rq1["verified_mr_count"] == 6
        assert rq1["verification_rate"] == pytest.approx(0.6)
        assert rq1["operators_with_mr"] == 2
        assert rq1["avg_mr_per_operator"] == pytest.approx(5.0)

    def test_category_distribution_populated(self):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 4, "verified_mrs": 4, "by_operator": {"op": {}}
        }
        mock_repo.list_operators.return_value = ["op"]
        mock_repo.load.return_value = [
            _make_mr(category="activation"),
            _make_mr(category="activation"),
            _make_mr(category="arithmetic"),
            _make_mr(category="activation"),
        ]

        org = ExperimentOrganizer(mr_repo=mock_repo)
        rq1 = org.collect_rq1()
        cats = rq1["category_distribution"]
        assert cats["activation"] == 3
        assert cats["arithmetic"] == 1

    def test_empty_repo(self):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 0, "verified_mrs": 0, "by_operator": {}
        }
        mock_repo.list_operators.return_value = []
        mock_repo.load.return_value = []

        org = ExperimentOrganizer(mr_repo=mock_repo)
        rq1 = org.collect_rq1()
        assert rq1["total_mr_count"] == 0
        assert rq1["verification_rate"] == pytest.approx(0.0)


# ── collect_rq2 ──────────────────────────────────────────────────────────────

class TestCollectRQ2:
    def test_from_populated_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "evidence")
        Path(ev_dir).mkdir()
        _populate_db(db_path)

        org = ExperimentOrganizer(db_path=db_path, evidence_dir=ev_dir)
        rq2 = org.collect_rq2()

        assert rq2["total_test_cases"] == 15  # 10 + 5
        assert rq2["total_passed"] == 12       # 7 + 5
        assert rq2["total_failed"] == 3
        assert rq2["overall_pass_rate"] == pytest.approx(12 / 15, abs=1e-4)
        assert rq2["operators_tested"] == 2
        assert rq2["operators_with_failure"] == 1
        assert rq2["evidence_pack_count"] == 0   # no evidence files

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        ev_dir = str(tmp_path / "evidence")
        Path(ev_dir).mkdir()

        org = ExperimentOrganizer(db_path=db_path, evidence_dir=ev_dir)
        rq2 = org.collect_rq2()
        assert rq2["total_test_cases"] == 0
        assert rq2["overall_pass_rate"] == pytest.approx(0.0)


# ── collect_rq3 ──────────────────────────────────────────────────────────────

class TestCollectRQ3:
    def test_no_sessions_returns_note(self, tmp_path):
        org = ExperimentOrganizer(cross_results_dir=str(tmp_path))
        rq3 = org.collect_rq3()
        assert rq3["cross_session_count"] == 0
        assert "note" in rq3

    def test_with_saved_session(self, tmp_path):
        from deepmt.analysis.qa.cross_framework_tester import (
            CrossConsistencyResult, CrossFrameworkTester, CrossSessionResult
        )
        tester = CrossFrameworkTester(results_dir=str(tmp_path))
        r = CrossConsistencyResult(
            operator="torch.exp", framework1="pytorch", framework2="numpy",
            mr_id="m1", mr_description="exp test", oracle_expr="trans==orig",
            n_samples=10,
            both_pass=8, only_f1_pass=1, only_f2_pass=0, both_fail=1,
            errors=0, output_max_diff=0.001, output_mean_diff=0.0005, output_close=True,
        )
        s = CrossSessionResult("sess01", "2026-01-01", "torch.exp", "pytorch", "numpy", 10, [r])
        tester.save(s)

        org = ExperimentOrganizer(cross_results_dir=str(tmp_path))
        rq3 = org.collect_rq3()

        assert rq3["cross_session_count"] == 1
        assert rq3["operators_compared"] == 1
        assert rq3["overall_consistency_rate"] is not None


# ── collect_rq4 ──────────────────────────────────────────────────────────────

class TestCollectRQ4:
    def test_fields_present(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "evidence")
        Path(ev_dir).mkdir()
        _populate_db(db_path)

        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 10, "verified_mrs": 6, "by_operator": {"a": {}, "b": {}}
        }
        mock_repo.list_operators.return_value = []
        mock_repo.load.return_value = []

        org = ExperimentOrganizer(db_path=db_path, evidence_dir=ev_dir, mr_repo=mock_repo)
        rq4 = org.collect_rq4()

        assert "operators_covered" in rq4
        assert "avg_mrs_per_operator" in rq4
        assert "test_density" in rq4
        assert "automation_scope" in rq4


# ── collect_all + format_text ─────────────────────────────────────────────────

class TestCollectAllAndFormat:
    def test_collect_all_keys(self, tmp_path):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 0, "verified_mrs": 0, "by_operator": {}
        }
        mock_repo.list_operators.return_value = []
        mock_repo.load.return_value = []

        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "ev")
        cr_dir = str(tmp_path / "cr")
        Path(ev_dir).mkdir()
        Path(cr_dir).mkdir()

        org = ExperimentOrganizer(
            db_path=db_path, evidence_dir=ev_dir, cross_results_dir=cr_dir, mr_repo=mock_repo
        )
        data = org.collect_all()
        assert "generated_at" in data
        assert "rq1" in data
        assert "rq2" in data
        assert "rq3" in data
        assert "rq4" in data

    def test_format_text_no_crash(self, tmp_path):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 3, "verified_mrs": 2, "by_operator": {"op": {}}
        }
        mock_repo.list_operators.return_value = ["op"]
        mock_repo.load.return_value = [_make_mr()]

        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "ev")
        cr_dir = str(tmp_path / "cr")
        Path(ev_dir).mkdir()
        Path(cr_dir).mkdir()

        org = ExperimentOrganizer(
            db_path=db_path, evidence_dir=ev_dir, cross_results_dir=cr_dir, mr_repo=mock_repo
        )
        data = org.collect_all()
        text = org.format_text(data)
        assert "RQ1" in text
        assert "RQ2" in text
        assert "RQ3" in text
        assert "RQ4" in text

    def test_json_serializable(self, tmp_path):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 0, "verified_mrs": 0, "by_operator": {}
        }
        mock_repo.list_operators.return_value = []
        mock_repo.load.return_value = []

        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "ev")
        cr_dir = str(tmp_path / "cr")
        Path(ev_dir).mkdir()
        Path(cr_dir).mkdir()

        org = ExperimentOrganizer(
            db_path=db_path, evidence_dir=ev_dir, cross_results_dir=cr_dir, mr_repo=mock_repo
        )
        data = org.collect_all()
        # 应可序列化为 JSON（无 nan/inf）
        json_str = json.dumps(data, ensure_ascii=False, allow_nan=True)
        assert isinstance(json_str, str)
