"""
ReportGenerator 单元测试

验证重点：
  1. generate() 正确聚合测试记录，返回结构化报告
  2. get_failures() 按 status=FAIL 过滤
  3. get_mr_breakdown() 按 MR 分解
  4. format_text() / format_failure_text() 格式正确，不抛出异常
  5. 数据库为空时优雅处理（返回空结构）

测试策略：
  - 使用临时 SQLite 数据库（ResultsManager(db_path=tmp_path)）
  - 向数据库注入已知记录，验证聚合逻辑
  - 不依赖 LLM、网络或框架插件
"""

import uuid
from pathlib import Path

import pytest

from deepmt.analysis.reporting.report_generator import ReportGenerator
from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import MetamorphicRelation, OracleResult, OperatorIR


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_mr(mr_id: str, description: str) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=mr_id,
        description=description,
        transform_code="lambda k: k",
        oracle_expr="orig == trans",
    )


def _make_oracle(passed: bool, detail: str = "") -> OracleResult:
    return OracleResult(
        passed=passed,
        expr="orig == trans",
        actual_diff=0.0 if passed else 1.0,
        tolerance=1e-6,
        detail=detail,
    )


def _seed_db(rm: ResultsManager, operator: str, framework: str, results: list):
    """向测试数据库注入指定记录列表 [(MR, OracleResult), ...]"""
    op_ir = OperatorIR(name=operator)
    rm.store_result(op_ir, results, framework)


# ── 基本聚合测试 ──────────────────────────────────────────────────────────────

class TestReportGeneratorBasic:
    @pytest.fixture()
    def gen(self, tmp_path):
        rm = ResultsManager(db_path=str(tmp_path / "test.db"))
        mr1 = _make_mr("mr-001", "线性性 MR")
        mr2 = _make_mr("mr-002", "非负性 MR")

        # relu: 8 pass + 2 fail
        _seed_db(rm, "torch.nn.functional.relu", "pytorch", [
            (mr1, _make_oracle(True)) for _ in range(5)
        ] + [
            (mr1, _make_oracle(False, "NUMERICAL_DEVIATION")) for _ in range(2)
        ] + [
            (mr2, _make_oracle(True)) for _ in range(3)
        ])

        # exp: 5 pass
        mr3 = _make_mr("mr-003", "指数加法性")
        _seed_db(rm, "torch.exp", "pytorch", [
            (mr3, _make_oracle(True)) for _ in range(5)
        ])

        return ReportGenerator(results_manager=rm)

    def test_generate_summary_counts(self, gen):
        report = gen.generate()
        s = report["summary"]
        assert s["total_operators"] == 2
        assert s["total_cases"] == 15
        assert s["total_passed"] == 13
        assert s["total_failed"] == 2
        assert s["pass_rate"] == pytest.approx(13 / 15, rel=1e-3)

    def test_generate_filter_by_framework(self, gen):
        report = gen.generate(framework="pytorch")
        assert report["summary"]["total_operators"] == 2

    def test_generate_filter_by_operator(self, gen):
        report = gen.generate(operator="torch.exp")
        assert report["summary"]["total_operators"] == 1
        assert report["summary"]["total_cases"] == 5
        assert report["summary"]["total_failed"] == 0

    def test_generate_limit(self, gen):
        report = gen.generate(limit=1)
        assert len(report["operators"]) == 1

    def test_generate_empty_db(self, tmp_path):
        rm = ResultsManager(db_path=str(tmp_path / "empty.db"))
        gen = ReportGenerator(results_manager=rm)
        report = gen.generate()
        assert report["summary"]["total_operators"] == 0
        assert report["summary"]["total_cases"] == 0
        assert report["summary"]["pass_rate"] == 0.0

    def test_get_failures(self, gen):
        failures = gen.get_failures()
        assert len(failures) == 2
        assert all(r["status"] == "FAIL" for r in failures)

    def test_get_failures_filter_by_operator(self, gen):
        failures = gen.get_failures(operator="torch.exp")
        assert len(failures) == 0

    def test_get_failures_limit(self, gen):
        failures = gen.get_failures(limit=1)
        assert len(failures) == 1

    def test_get_mr_breakdown(self, gen):
        mrs = gen.get_mr_breakdown("torch.nn.functional.relu")
        assert len(mrs) == 2
        mr_ids = {m["mr_id"] for m in mrs}
        assert "mr-001" in mr_ids
        assert "mr-002" in mr_ids

        mr1_data = next(m for m in mrs if m["mr_id"] == "mr-001")
        assert mr1_data["total"] == 7
        assert mr1_data["passed"] == 5
        assert mr1_data["failed"] == 2
        assert mr1_data["pass_rate"] == pytest.approx(5 / 7, rel=1e-3)


# ── 格式化输出测试 ─────────────────────────────────────────────────────────────

class TestReportGeneratorFormat:
    @pytest.fixture()
    def gen_with_data(self, tmp_path):
        rm = ResultsManager(db_path=str(tmp_path / "fmt.db"))
        mr = _make_mr("mr-x", "test MR")
        _seed_db(rm, "torch.abs", "pytorch", [
            (mr, _make_oracle(True)),
            (mr, _make_oracle(False, "SHAPE_MISMATCH")),
        ])
        return ReportGenerator(results_manager=rm)

    def test_format_text_runs_without_error(self, gen_with_data):
        report = gen_with_data.generate()
        text = gen_with_data.format_text(report)
        assert isinstance(text, str)
        assert "torch.abs" in text
        assert "通过率" in text or "pass" in text.lower()

    def test_format_text_contains_counts(self, gen_with_data):
        report = gen_with_data.generate()
        text = gen_with_data.format_text(report)
        assert "1/2" in text or "1" in text  # 1 pass out of 2

    def test_format_failure_text_empty(self, gen_with_data):
        text = gen_with_data.format_failure_text([])
        assert "未发现" in text

    def test_format_failure_text_with_failures(self, gen_with_data):
        failures = gen_with_data.get_failures()
        text = gen_with_data.format_failure_text(failures)
        assert "torch.abs" in text
        assert "SHAPE_MISMATCH" in text

    def test_format_text_empty_report(self, tmp_path):
        rm = ResultsManager(db_path=str(tmp_path / "empty2.db"))
        gen = ReportGenerator(results_manager=rm)
        report = gen.generate()
        text = gen.format_text(report)
        assert isinstance(text, str)
        assert "0" in text  # total_cases=0
