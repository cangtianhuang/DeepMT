"""
Phase M — 真实案例流水线集成测试

测试覆盖：
  1. DefectCaseBuilder.build_from_lead()   — 从 DefectLead 构建 CaseStudy
  2. DefectCaseBuilder.build_from_evidence() — 从证据包 ID 构建 CaseStudy
  3. DefectCaseBuilder.build_case_package()  — 生成案例包目录及四个文件
  4. DefectCaseBuilder.build_top_leads()     — 批量构建 top-N 案例包
  5. reproduce.py 语法有效性检查（py_compile）
  6. 案例包 metadata.json 结构完整性

所有测试均使用临时目录，不污染真实案例库。
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepmt.analysis.defect_case_builder import DefectCaseBuilder
from deepmt.analysis.qa.defect_deduplicator import DefectLead
from deepmt.experiments.case_study import CaseStudy, CaseStudyIndex


# ── 工厂函数 ────────────────────────────────────────────────────────────────────

def _make_lead(
    operator: str = "relu",
    framework: str = "pytorch",
    error_bucket: str = "numerical",
    occurrence_count: int = 5,
    evidence_id: str = "test_ev_001",
) -> DefectLead:
    return DefectLead(
        lead_id="testlead001234",
        operator=operator,
        framework=framework,
        mr_id="mr-test-001",
        mr_description="relu正齐次性测试",
        error_bucket=error_bucket,
        occurrence_count=occurrence_count,
        first_seen="2026-04-14T10:00:00",
        last_seen="2026-04-14T16:00:00",
        detail_sample="NUMERICAL_DEVIATION: max_abs=0.5",
        representative_evidence_id=evidence_id,
    )


def _make_mock_evidence_pack(
    operator: str = "relu",
    framework: str = "pytorch",
    evidence_id: str = "test_ev_001",
) -> MagicMock:
    pack = MagicMock()
    pack.evidence_id = evidence_id
    pack.operator = operator
    pack.framework = framework
    pack.framework_version = "2.11.0"
    pack.mr_id = "mr-test-001"
    pack.mr_description = "relu 正齐次性"
    pack.transform_code = "lambda k: {**k, 'input': 2.0 * k['input']}"
    pack.oracle_expr = "trans == 2.0 * orig"
    pack.input_summary = {"shape": [4, 4], "dtype": "float32", "min": -1.0, "max": 1.0}
    pack.actual_diff = 0.5
    pack.tolerance = 1e-6
    pack.detail = "NUMERICAL_DEVIATION: max_abs=0.5"
    pack.reproduce_script = "# placeholder reproduce script\nprint('reproduced')\n"
    return pack


# ── 测试类 ──────────────────────────────────────────────────────────────────────

class TestDefectCaseBuilderFromLead(unittest.TestCase):
    """测试从 DefectLead 构建 CaseStudy"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._case_dir = Path(self._tmpdir) / "cases"
        self._cases_output = Path(self._tmpdir) / "output"
        self._index = CaseStudyIndex(case_dir=self._case_dir)

        # Mock 证据收集器
        self._mock_collector = MagicMock()
        self._mock_evidence = _make_mock_evidence_pack()
        self._mock_collector.load.return_value = self._mock_evidence
        self._mock_collector.evidence_dir = Path(self._tmpdir) / "evidence"
        self._mock_collector.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._builder = DefectCaseBuilder(
            case_index=self._index,
            evidence_collector=self._mock_collector,
            cases_dir=self._cases_output,
        )

    def test_build_from_lead_creates_case_study(self):
        """build_from_lead 应创建并保存 CaseStudy"""
        lead = _make_lead()
        case = self._builder.build_from_lead(lead)

        self.assertIsInstance(case, CaseStudy)
        self.assertEqual(case.operator, "relu")
        self.assertEqual(case.framework, "pytorch")
        self.assertEqual(case.mr_id, "mr-test-001")

    def test_build_from_lead_infers_defect_type(self):
        """build_from_lead 应根据 error_bucket 推断 defect_type"""
        lead = _make_lead(error_bucket="numerical")
        case = self._builder.build_from_lead(lead)
        self.assertEqual(case.defect_type, "numerical_precision")

    def test_build_from_lead_infers_severity_high(self):
        """出现次数 >= 10 的 numerical 错误应被评为 high"""
        lead = _make_lead(error_bucket="numerical", occurrence_count=10)
        case = self._builder.build_from_lead(lead)
        self.assertEqual(case.severity, "high")

    def test_build_from_lead_infers_severity_low(self):
        """出现次数 < 5 的非 exception 错误应评为 low"""
        lead = _make_lead(error_bucket="numerical", occurrence_count=2)
        case = self._builder.build_from_lead(lead)
        self.assertEqual(case.severity, "low")

    def test_build_from_lead_notes_contains_lead_id(self):
        """notes 字段应包含 lead_id"""
        lead = _make_lead()
        case = self._builder.build_from_lead(lead)
        self.assertIn("testlead001234", case.notes)

    def test_build_from_lead_no_evidence_creates_empty(self):
        """representative_evidence_id 为 None 时应创建空 CaseStudy"""
        self._mock_collector.load.return_value = None
        lead = _make_lead(evidence_id=None)
        lead.representative_evidence_id = None
        case = self._builder.build_from_lead(lead)
        self.assertIsInstance(case, CaseStudy)
        self.assertEqual(case.operator, "relu")

    def test_build_from_lead_persists_to_index(self):
        """build_from_lead 应将案例保存到 CaseStudyIndex"""
        lead = _make_lead()
        case = self._builder.build_from_lead(lead)

        loaded = self._index.load(case.case_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.operator, "relu")


class TestDefectCaseBuilderPackage(unittest.TestCase):
    """测试 build_case_package 生成的目录结构"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._case_dir = Path(self._tmpdir) / "cases"
        self._cases_output = Path(self._tmpdir) / "output"
        self._index = CaseStudyIndex(case_dir=self._case_dir)

        self._mock_collector = MagicMock()
        self._mock_evidence = _make_mock_evidence_pack()
        self._mock_collector.load.return_value = self._mock_evidence
        self._mock_collector.evidence_dir = Path(self._tmpdir) / "evidence"
        self._mock_collector.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._builder = DefectCaseBuilder(
            case_index=self._index,
            evidence_collector=self._mock_collector,
            cases_dir=self._cases_output,
        )

    def _build_case(self) -> CaseStudy:
        lead = _make_lead()
        return self._builder.build_from_lead(lead)

    def test_package_directory_created(self):
        """build_case_package 应创建以 case_id 命名的目录"""
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case)
        self.assertTrue(pkg_dir.exists())
        self.assertTrue(pkg_dir.is_dir())
        self.assertEqual(pkg_dir.name, case.case_id)

    def test_package_contains_four_files(self):
        """案例包应包含 reproduce.py / case_summary.md / metadata.json 共3个文件
           （evidence.json 仅在证据文件存在时生成）"""
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case)
        files = {f.name for f in pkg_dir.iterdir()}
        self.assertIn("reproduce.py", files)
        self.assertIn("case_summary.md", files)
        self.assertIn("metadata.json", files)

    def test_reproduce_script_is_valid_python(self):
        """reproduce.py 应是语法正确的 Python 脚本"""
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case)
        script = (pkg_dir / "reproduce.py").read_text(encoding="utf-8")
        try:
            ast.parse(script)
        except SyntaxError as e:
            self.fail(f"reproduce.py 语法错误: {e}")

    def test_case_summary_contains_case_id(self):
        """case_summary.md 应包含案例 case_id"""
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case)
        summary = (pkg_dir / "case_summary.md").read_text(encoding="utf-8")
        self.assertIn(case.case_id, summary)

    def test_metadata_json_structure(self):
        """metadata.json 应包含必要字段"""
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case)
        metadata = json.loads((pkg_dir / "metadata.json").read_text(encoding="utf-8"))
        required_keys = {"case_id", "operator", "framework", "defect_type",
                         "severity", "status", "created_at", "package_generated_at"}
        for key in required_keys:
            self.assertIn(key, metadata, f"metadata.json 缺少字段: {key}")
        self.assertEqual(metadata["case_id"], case.case_id)
        self.assertEqual(metadata["operator"], "relu")

    def test_custom_output_dir(self):
        """build_case_package 应支持自定义输出目录"""
        custom_dir = Path(self._tmpdir) / "custom_output"
        case = self._build_case()
        pkg_dir = self._builder.build_case_package(case, output_dir=custom_dir)
        self.assertTrue((custom_dir / case.case_id).exists())

    def test_build_top_leads_creates_multiple_packages(self):
        """build_top_leads 应为每个 lead 生成独立案例包"""
        leads = [
            _make_lead(operator="relu"),
            _make_lead(operator="sigmoid"),
        ]
        pkg_dirs = self._builder.build_top_leads(leads, top_n=2)
        self.assertEqual(len(pkg_dirs), 2)
        for pkg_dir in pkg_dirs:
            self.assertTrue(pkg_dir.exists())
            self.assertIn("reproduce.py", {f.name for f in pkg_dir.iterdir()})

    def test_build_top_leads_respects_top_n(self):
        """build_top_leads 应只处理前 top_n 个 lead"""
        leads = [_make_lead(operator=op) for op in ["relu", "sigmoid", "tanh"]]
        pkg_dirs = self._builder.build_top_leads(leads, top_n=2)
        self.assertEqual(len(pkg_dirs), 2)


class TestDefectCaseBuilderFromEvidence(unittest.TestCase):
    """测试 build_from_evidence 直接从证据包 ID 构建"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._case_dir = Path(self._tmpdir) / "cases"
        self._cases_output = Path(self._tmpdir) / "output"
        self._index = CaseStudyIndex(case_dir=self._case_dir)

        self._mock_collector = MagicMock()
        self._mock_evidence = _make_mock_evidence_pack()
        self._mock_collector.load.return_value = self._mock_evidence
        self._mock_collector.evidence_dir = Path(self._tmpdir) / "evidence"
        self._mock_collector.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._builder = DefectCaseBuilder(
            case_index=self._index,
            evidence_collector=self._mock_collector,
            cases_dir=self._cases_output,
        )

    def test_build_from_evidence_id(self):
        """build_from_evidence 应从证据包 ID 创建 CaseStudy"""
        case = self._builder.build_from_evidence("test_ev_001")
        self.assertIsInstance(case, CaseStudy)
        self.assertEqual(case.operator, "relu")

    def test_build_from_evidence_sets_defect_type(self):
        """build_from_evidence 应根据证据包 detail 推断 defect_type"""
        case = self._builder.build_from_evidence("test_ev_001")
        self.assertEqual(case.defect_type, "numerical_precision")

    def test_build_from_evidence_missing_pack(self):
        """证据包不存在时，build_from_evidence 应返回空 CaseStudy 而不崩溃"""
        self._mock_collector.load.return_value = None
        case = self._builder.build_from_evidence("nonexistent_id")
        self.assertIsInstance(case, CaseStudy)


if __name__ == "__main__":
    unittest.main()
