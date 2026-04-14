"""
Phase L 单元测试：experiments 包与 analysis.stats 模块。

覆盖范围：
  - rq_config：RQConfig 结构正确性
  - version_matrix：VersionMatrix 结构与 installed_versions 调用
  - benchmark_suite：BenchmarkSuite 清单内容
  - run_manifest：RunManifest 序列化/反序列化
  - environment_recorder：EnvironmentSnapshot 生成
  - aggregator：StatsAggregator.collect 无报错（mock ExperimentOrganizer）
  - exporter：StatsExporter JSON/CSV/Markdown 导出文件存在且非空
  - case_study：CaseStudy 序列化与 CaseStudyIndex CRUD
"""

import json
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# rq_config
# ─────────────────────────────────────────────────────────────────────────────

class TestRQConfig:
    def test_all_rqs_defined(self):
        from deepmt.experiments.rq_config import RQ_DEFINITIONS, list_rqs
        assert set(RQ_DEFINITIONS.keys()) == {"rq1", "rq2", "rq3", "rq4"}
        assert len(list_rqs()) == 4

    def test_get_rq_returns_config(self):
        from deepmt.experiments.rq_config import get_rq
        cfg = get_rq("rq1")
        assert cfg is not None
        assert cfg.id == "rq1"
        assert len(cfg.metrics) > 0

    def test_get_rq_case_insensitive(self):
        from deepmt.experiments.rq_config import get_rq
        assert get_rq("RQ2") is get_rq("rq2")

    def test_get_rq_unknown_returns_none(self):
        from deepmt.experiments.rq_config import get_rq
        assert get_rq("rq99") is None

    def test_metric_names_nonempty(self):
        from deepmt.experiments.rq_config import list_rqs
        for cfg in list_rqs():
            names = cfg.metric_names()
            assert len(names) > 0, f"{cfg.id} 无指标"

    def test_metric_by_name_found(self):
        from deepmt.experiments.rq_config import get_rq
        cfg = get_rq("rq1")
        m = cfg.metric_by_name("total_mr_count")
        assert m is not None
        assert m.unit == "条"

    def test_metric_by_name_not_found(self):
        from deepmt.experiments.rq_config import get_rq
        cfg = get_rq("rq1")
        assert cfg.metric_by_name("nonexistent") is None

    def test_rq4_has_notes(self):
        from deepmt.experiments.rq_config import get_rq
        cfg = get_rq("rq4")
        assert cfg.notes  # RQ4 有人工填写说明


# ─────────────────────────────────────────────────────────────────────────────
# version_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionMatrix:
    def test_version_matrix_has_pytorch(self):
        from deepmt.experiments.version_matrix import VERSION_MATRIX
        frameworks = {e.framework for e in VERSION_MATRIX}
        assert "pytorch" in frameworks

    def test_get_entry_found(self):
        from deepmt.experiments.version_matrix import get_entry
        entry = get_entry("pytorch")
        assert entry is not None
        assert entry.role == "primary"

    def test_get_entry_not_found(self):
        from deepmt.experiments.version_matrix import get_entry
        assert get_entry("unknown_framework") is None

    def test_get_installed_versions_returns_dict(self):
        from deepmt.experiments.version_matrix import get_installed_versions
        result = get_installed_versions()
        assert isinstance(result, dict)
        assert "pytorch" in result
        assert "numpy" in result

    def test_check_version_compatibility_returns_list(self):
        from deepmt.experiments.version_matrix import check_version_compatibility
        report = check_version_compatibility()
        assert isinstance(report, list)
        assert len(report) > 0
        for row in report:
            assert "framework" in row
            assert "status" in row
            assert row["status"] in ("ok", "mismatch", "not_installed")


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_suite
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkSuite:
    def test_operator_count(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        ops = suite.operator_benchmark()
        assert len(ops) >= 20, "算子 benchmark 数量应 >= 20"

    def test_model_names_nonempty(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        names = suite.model_names()
        assert len(names) > 0
        assert "SimpleMLP" in names

    def test_application_names_nonempty(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        names = suite.application_names()
        assert len(names) > 0
        assert "ImageClassification" in names

    def test_operator_category_filter(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        activation = suite.operator_benchmark(category="activation")
        assert len(activation) > 0
        assert all(e.category == "activation" for e in activation)

    def test_operator_framework_filter(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        pytorch = suite.operator_benchmark(framework="pytorch")
        assert len(pytorch) > 0
        assert all(e.framework == "pytorch" for e in pytorch)

    def test_summary_keys(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        s = suite.summary()
        assert "operator_count" in s
        assert "model_count" in s
        assert "application_count" in s
        assert s["operator_count"] > 0

    def test_operator_names_unique(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        names = suite.operator_names()
        assert len(names) == len(set(names)), "算子名称不应有重复"

    def test_model_benchmark_without_instance(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        models = suite.model_benchmark(with_instance=False)
        assert len(models) == len(suite.model_names())

    def test_application_benchmark(self):
        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()
        apps = suite.application_benchmark()
        assert len(apps) > 0


# ─────────────────────────────────────────────────────────────────────────────
# run_manifest
# ─────────────────────────────────────────────────────────────────────────────

class TestRunManifest:
    def test_create_and_serialise(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            m = mgr.create(rqs=["rq1", "rq2"], seed=42, capture_env=False)
            assert m.run_id
            assert m.seed == 42
            assert "rq1" in m.rqs
            d = m.to_dict()
            assert d["seed"] == 42

    def test_save_and_load_roundtrip(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            m = mgr.create(seed=99, capture_env=False)
            mgr.save(m)
            loaded = mgr.load(m.run_id)
            assert loaded is not None
            assert loaded.run_id == m.run_id
            assert loaded.seed == 99

    def test_load_nonexistent_returns_none(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            assert mgr.load("nonexistent_id") is None

    def test_list_all_sorted(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            m1 = mgr.create(capture_env=False)
            m2 = mgr.create(capture_env=False)
            mgr.save(m1)
            mgr.save(m2)
            all_m = mgr.list_all()
            assert len(all_m) == 2
            assert all_m[0].created_at <= all_m[1].created_at

    def test_mark_status(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            m = mgr.create(capture_env=False)
            m.mark_running()
            assert m.status == "running"
            m.mark_completed("some/path.json")
            assert m.status == "completed"
            assert m.result_summary_path == "some/path.json"

    def test_benchmark_lists_populated(self):
        from deepmt.experiments.run_manifest import RunManifestManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunManifestManager(runs_dir=tmpdir)
            m = mgr.create(capture_env=False)
            assert len(m.benchmark_operators) > 0
            assert len(m.benchmark_models) > 0
            assert len(m.benchmark_applications) > 0

    def test_from_dict_roundtrip(self):
        from deepmt.experiments.run_manifest import RunManifest
        m = RunManifest(
            run_id="test123",
            created_at="2026-01-01T00:00:00",
            seed=42,
            rqs=["rq1"],
            notes="test",
        )
        d = m.to_dict()
        m2 = RunManifest.from_dict(d)
        assert m2.run_id == m.run_id
        assert m2.notes == "test"


# ─────────────────────────────────────────────────────────────────────────────
# environment_recorder
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentRecorder:
    def test_capture_returns_snapshot(self):
        from deepmt.experiments.environment_recorder import EnvironmentRecorder
        snap = EnvironmentRecorder().capture()
        assert snap.python_version
        assert snap.platform_info
        assert "pytorch" in snap.framework_versions

    def test_to_dict_serializable(self):
        from deepmt.experiments.environment_recorder import EnvironmentRecorder
        snap = EnvironmentRecorder().capture()
        d = snap.to_dict()
        # 可 JSON 序列化
        json.dumps(d)

    def test_format_text_nonempty(self):
        from deepmt.experiments.environment_recorder import EnvironmentRecorder
        snap = EnvironmentRecorder().capture()
        txt = snap.format_text()
        assert "Python" in txt


# ─────────────────────────────────────────────────────────────────────────────
# stats aggregator (使用 mock 的 ExperimentOrganizer)
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsAggregator:
    def _make_mock_stats(self, rqs=None):
        """创建一个使用 mock 数据的 ThesisStats。"""
        from unittest.mock import MagicMock, patch
        from deepmt.experiments.aggregator import StatsAggregator

        mock_rq1 = {
            "total_mr_count": 10, "verified_mr_count": 8,
            "verification_rate": 0.8, "operators_with_mr": 5,
            "avg_mr_per_operator": 2.0,
            "category_distribution": {"monotonicity": 5, "symmetry": 3},
            "source_distribution": {"llm": 7, "template": 3},
        }
        mock_rq2 = {
            "total_test_cases": 100, "total_passed": 90, "total_failed": 10,
            "overall_pass_rate": 0.9, "operators_tested": 5,
            "operators_with_failure": 2, "failure_rate": 0.1,
            "evidence_pack_count": 3, "unique_defect_leads": 2,
        }
        mock_rq3 = {
            "cross_session_count": 0, "operators_compared": 0,
            "overall_consistency_rate": None, "avg_output_max_diff": None,
            "inconsistent_mr_count": 0, "framework_pairs": [],
            "note": "尚无跨框架实验记录",
        }
        mock_rq4 = {
            "operators_covered": 5, "avg_mrs_per_operator": 2.0,
            "test_density": 20.0, "automation_scope": "全链路自动化",
            "note": "基线数值需手动填入",
        }

        # ExperimentOrganizer 在 aggregator 内部通过 import 引入，需 patch 其所在模块
        with patch(
            "deepmt.experiments.organizer.ExperimentOrganizer"
        ) as MockOrg:
            org_instance = MagicMock()
            org_instance.collect_rq1.return_value = mock_rq1
            org_instance.collect_rq2.return_value = mock_rq2
            org_instance.collect_rq3.return_value = mock_rq3
            org_instance.collect_rq4.return_value = mock_rq4
            MockOrg.return_value = org_instance

            agg = StatsAggregator()
            return agg.collect(rqs=rqs)

    def test_collect_all_rqs(self):
        stats = self._make_mock_stats()
        assert "rq1" in stats.rq_data
        assert "rq2" in stats.rq_data
        assert "rq3" in stats.rq_data
        assert "rq4" in stats.rq_data

    def test_collect_subset_rqs(self):
        stats = self._make_mock_stats(rqs=["rq1", "rq2"])
        assert "rq1" in stats.rq_data
        assert "rq2" in stats.rq_data
        assert "rq3" not in stats.rq_data

    def test_get_metric(self):
        stats = self._make_mock_stats()
        val = stats.get_metric("rq1", "total_mr_count")
        assert val == 10

    def test_to_dict_serializable(self):
        stats = self._make_mock_stats()
        d = stats.to_dict()
        json.dumps(d, default=str)

    def test_benchmark_summary_present(self):
        stats = self._make_mock_stats()
        assert "operator_count" in stats.benchmark_summary


# ─────────────────────────────────────────────────────────────────────────────
# stats exporter
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsExporter:
    def _make_stats(self):
        from unittest.mock import MagicMock, patch
        from deepmt.experiments.aggregator import StatsAggregator, ThesisStats
        from datetime import datetime

        stats = ThesisStats(
            generated_at=datetime.now().isoformat(),
            rq_data={
                "rq1": {"total_mr_count": 10, "verified_mr_count": 8,
                        "verification_rate": 0.8, "operators_with_mr": 5,
                        "avg_mr_per_operator": 2.0,
                        "category_distribution": {"monotonicity": 5},
                        "source_distribution": {"llm": 8, "template": 2}},
                "rq2": {"total_test_cases": 100, "total_passed": 90,
                        "total_failed": 10, "overall_pass_rate": 0.9,
                        "operators_tested": 5, "operators_with_failure": 2,
                        "failure_rate": 0.1, "evidence_pack_count": 3,
                        "unique_defect_leads": 2},
                "rq3": {"cross_session_count": 0, "operators_compared": 0,
                        "overall_consistency_rate": None, "avg_output_max_diff": None,
                        "inconsistent_mr_count": 0, "framework_pairs": [],
                        "note": "no data"},
                "rq4": {"operators_covered": 5, "avg_mrs_per_operator": 2.0,
                        "test_density": 20.0, "automation_scope": "全链路",
                        "note": "手动填写基线"},
            },
            benchmark_summary={"operator_count": 31, "model_count": 4, "application_count": 2},
        )
        return stats

    def test_export_json(self):
        from deepmt.experiments.exporter import StatsExporter
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StatsExporter(output_dir=tmpdir)
            stats = self._make_stats()
            path = exporter.export_json(stats)
            assert path.exists()
            assert path.stat().st_size > 0
            with open(path) as f:
                data = json.load(f)
            assert "rq_data" in data

    def test_export_markdown(self):
        from deepmt.experiments.exporter import StatsExporter
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StatsExporter(output_dir=tmpdir)
            stats = self._make_stats()
            path = exporter.export_markdown(stats)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "RQ1" in content
            assert "RQ2" in content

    def test_export_csv_per_rq(self):
        from deepmt.experiments.exporter import StatsExporter
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StatsExporter(output_dir=tmpdir)
            stats = self._make_stats()
            path = exporter.export_csv_per_rq(stats)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "rq" in content.lower()
            assert "metric" in content.lower()

    def test_export_benchmark_csv(self):
        from deepmt.experiments.exporter import StatsExporter
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StatsExporter(output_dir=tmpdir)
            path = exporter.export_benchmark_csv()
            assert path.exists()

    def test_export_all_returns_four_paths(self):
        from deepmt.experiments.exporter import StatsExporter
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StatsExporter(output_dir=tmpdir)
            stats = self._make_stats()
            result = exporter.export_all(stats)
            assert "json" in result
            assert "markdown" in result
            assert "csv" in result
            assert "benchmark_csv" in result
            for p in result.values():
                assert p.exists()


# ─────────────────────────────────────────────────────────────────────────────
# case_study
# ─────────────────────────────────────────────────────────────────────────────

class TestCaseStudy:
    def test_to_dict_and_from_dict(self):
        from deepmt.experiments.case_study import CaseStudy
        c = CaseStudy(
            case_id="test001",
            operator="torch.relu",
            framework="pytorch",
            summary="测试摘要",
            status="draft",
        )
        d = c.to_dict()
        c2 = CaseStudy.from_dict(d)
        assert c2.case_id == "test001"
        assert c2.summary == "测试摘要"

    def test_to_markdown_contains_operator(self):
        from deepmt.experiments.case_study import CaseStudy
        c = CaseStudy(
            case_id="test002",
            operator="torch.sigmoid",
            framework="pytorch",
            summary="Sigmoid 精度问题",
        )
        md = c.to_markdown()
        assert "torch.sigmoid" in md
        assert "test002" in md

    def test_to_dict_json_serializable(self):
        from deepmt.experiments.case_study import CaseStudy
        c = CaseStudy(
            case_id="test003",
            input_example=[[1.0, 2.0], [3.0, 4.0]],
            output_original=0.5,
            output_transformed=0.51,
        )
        json.dumps(c.to_dict(), default=str)


class TestCaseStudyIndex:
    def test_create_empty(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c = idx.create_empty(operator="relu", framework="pytorch")
            assert c.case_id
            assert c.operator == "relu"

    def test_save_and_load(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c = idx.create_empty(operator="sigmoid")
            c.summary = "测试案例"
            idx.save(c)
            loaded = idx.load(c.case_id)
            assert loaded is not None
            assert loaded.summary == "测试案例"

    def test_load_nonexistent_returns_none(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            assert idx.load("nonexistent") is None

    def test_list_all(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c1 = idx.create_empty(operator="relu")
            c2 = idx.create_empty(operator="sigmoid")
            c2.status = "confirmed"
            idx.save(c1)
            idx.save(c2)
            all_cases = idx.list_all()
            assert len(all_cases) == 2

    def test_list_all_status_filter(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c1 = idx.create_empty(operator="relu")
            c1.status = "draft"
            c2 = idx.create_empty(operator="sigmoid")
            c2.status = "confirmed"
            idx.save(c1)
            idx.save(c2)
            confirmed = idx.list_all(status="confirmed")
            assert len(confirmed) == 1
            assert confirmed[0].operator == "sigmoid"

    def test_export_markdown_catalog(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c = idx.create_empty(operator="relu")
            c.summary = "测试"
            idx.save(c)
            out = Path(tmpdir) / "catalog.md"
            idx.export_markdown_catalog(output_path=out)
            assert out.exists()
            content = out.read_text(encoding="utf-8")
            assert "Case Study" in content

    def test_index_file_updated_on_save(self):
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c = idx.create_empty(operator="tanh")
            idx.save(c)
            index = idx.load_index()
            assert c.case_id in index

    def test_from_evidence_nonexistent_returns_case(self):
        """从不存在的 evidence 创建 CaseStudy 不应抛出异常。"""
        from deepmt.experiments.case_study import CaseStudyIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CaseStudyIndex(case_dir=tmpdir)
            c = idx.from_evidence("nonexistent_evidence.json")
            assert c.case_id  # 即使 evidence 不存在也返回空白案例
