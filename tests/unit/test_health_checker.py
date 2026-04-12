"""健康检查器单元测试

测试覆盖：
  - HealthChecker.run_all_checks() 运行正常，所有核心模块通过
  - 每个分类（核心/引擎/MR生成/分析/插件）均有结果
  - HealthReport 统计字段正确
  - 数据目录检查：目录存在时 HEALTHY，目录不存在时自动创建并返回 HEALTHY
  - MR 知识库与结果数据库运行时检查可执行（不崩溃）
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from deepmt.monitoring.health_checker import HealthChecker, HealthReport, HealthStatus


@pytest.fixture(scope="module")
def checker():
    return HealthChecker()


@pytest.fixture(scope="module")
def report(checker):
    return checker.run_all_checks()


# ── 报告整体结构 ───────────────────────────────────────────────────────────────


class TestHealthReportStructure:

    def test_report_is_health_report(self, report):
        assert isinstance(report, HealthReport)

    def test_timestamp_set(self, report):
        from datetime import datetime
        assert isinstance(report.timestamp, datetime)

    def test_results_non_empty(self, report):
        assert len(report.results) > 0

    def test_overall_status_is_enum(self, report):
        assert isinstance(report.overall_status, HealthStatus)

    def test_counts_consistent(self, report):
        total = report.healthy_count + report.warning_count + report.error_count
        assert total == len(report.results)

    def test_overall_healthy_no_errors(self, report):
        """当前环境应全部通过（PyTorch + 所有 deepmt 模块已安装）"""
        assert report.error_count == 0


# ── 核心模块覆盖 ───────────────────────────────────────────────────────────────


class TestCoreModuleCoverage:
    """验证每类核心模块都有对应的检查结果"""

    EXPECTED_MODULES = [
        "deepmt.core.config_manager",
        "deepmt.core.logger",
        "deepmt.core.plugins_manager",
        "deepmt.core.results_manager",
        "deepmt.ir.schema",
        "deepmt.engine.batch_test_runner",
        "deepmt.mr_generator.base.mr_repository",
        "deepmt.analysis.report_generator",
        "deepmt.analysis.evidence_collector",
        "deepmt.plugins.pytorch_plugin",
    ]

    def test_all_expected_modules_checked(self, report):
        checked = {r.name for r in report.results}
        for mod in self.EXPECTED_MODULES:
            assert mod in checked, f"核心模块 {mod} 未被健康检查覆盖"

    def test_all_expected_modules_healthy(self, report):
        result_map = {r.name: r for r in report.results}
        for mod in self.EXPECTED_MODULES:
            if mod in result_map:
                assert result_map[mod].status == HealthStatus.HEALTHY, (
                    f"{mod} 健康检查失败: {result_map[mod].message}"
                )

    def test_pytorch_detected(self, report):
        pytorch_result = next((r for r in report.results if r.name == "pytorch"), None)
        assert pytorch_result is not None
        assert pytorch_result.status == HealthStatus.HEALTHY
        assert "torch" in pytorch_result.message.lower() or "pytorch" in pytorch_result.message.lower()


# ── 数据目录检查 ───────────────────────────────────────────────────────────────


class TestDataDirectoryChecks:

    def test_existing_dir_healthy(self, checker):
        with tempfile.TemporaryDirectory() as tmp:
            result = checker._check_dir(tmp, "临时目录")
            assert result.status == HealthStatus.HEALTHY

    def test_nonexistent_dir_created_and_healthy(self, checker):
        with tempfile.TemporaryDirectory() as parent:
            new_dir = str(Path(parent) / "new_subdir" / "nested")
            result = checker._check_dir(new_dir, "嵌套目录")
            assert result.status == HealthStatus.HEALTHY
            assert Path(new_dir).exists()


# ── 配置检查 ───────────────────────────────────────────────────────────────────


class TestConfigCheck:

    def test_config_check_runs(self, checker):
        result = checker._check_config()
        assert result.name == "config"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.WARNING)

    def test_config_check_message_informative(self, checker):
        result = checker._check_config()
        assert len(result.message) > 0


# ── 运行时检查 ─────────────────────────────────────────────────────────────────


class TestRuntimeChecks:

    def test_mr_repository_check_does_not_crash(self, checker):
        result = checker._check_mr_repository()
        assert result.name == "mr_repository"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.ERROR)

    def test_results_db_check_does_not_crash(self, checker):
        result = checker._check_results_db()
        assert result.name == "results_db"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.ERROR)


# ── print_report 无崩溃 ────────────────────────────────────────────────────────


def test_print_report_no_crash(capsys, checker, report):
    checker.print_report(report)
    captured = capsys.readouterr()
    assert "DeepMT" in captured.out
    assert "HEALTHY" in captured.out or "WARNING" in captured.out or "ERROR" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
