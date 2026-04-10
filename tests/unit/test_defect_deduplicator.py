"""
DefectDeduplicator 单元测试。

覆盖：
  - _extract_error_bucket: 各种 detail 字符串 → 正确桶
  - _make_lead_id: 相同签名 → 相同 ID，不同签名 → 不同 ID
  - DefectDeduplicator.deduplicate: 空集、聚类、过滤、排序
  - DefectDeduplicator.format_text: 输出包含关键信息
"""

import pytest
from unittest.mock import MagicMock, patch

from deepmt.analysis.defect_deduplicator import (
    DefectDeduplicator,
    DefectLead,
    _extract_error_bucket,
    _make_lead_id,
)


# ── _extract_error_bucket ─────────────────────────────────────────────────────

class TestExtractErrorBucket:
    def test_numerical_deviation(self):
        assert _extract_error_bucket("NUMERICAL_DEVIATION: max_abs=0.5") == "numerical"

    def test_shape_mismatch(self):
        assert _extract_error_bucket("SHAPE_MISMATCH: (2, 3) vs (3, 2)") == "shape"

    def test_exception(self):
        assert _extract_error_bucket("EXCEPTION: RuntimeError occurred") == "exception"

    def test_transform_error(self):
        assert _extract_error_bucket("TRANSFORM_ERROR: lambda failed") == "transform"

    def test_type_error(self):
        assert _extract_error_bucket("TYPE_ERROR: expected tensor") == "type"

    def test_unknown_returns_other(self):
        assert _extract_error_bucket("UNKNOWN_ERROR: something else") == "other"

    def test_empty_string_returns_other(self):
        assert _extract_error_bucket("") == "other"

    def test_case_insensitive(self):
        assert _extract_error_bucket("numerical_deviation: foo") == "numerical"


# ── _make_lead_id ─────────────────────────────────────────────────────────────

class TestMakeLeadId:
    def test_same_signature_same_id(self):
        id1 = _make_lead_id("op.a", "mr-1", "numerical", "pytorch")
        id2 = _make_lead_id("op.a", "mr-1", "numerical", "pytorch")
        assert id1 == id2

    def test_different_operator_different_id(self):
        id1 = _make_lead_id("op.a", "mr-1", "numerical", "pytorch")
        id2 = _make_lead_id("op.b", "mr-1", "numerical", "pytorch")
        assert id1 != id2

    def test_different_mr_different_id(self):
        id1 = _make_lead_id("op.a", "mr-1", "numerical", "pytorch")
        id2 = _make_lead_id("op.a", "mr-2", "numerical", "pytorch")
        assert id1 != id2

    def test_different_bucket_different_id(self):
        id1 = _make_lead_id("op.a", "mr-1", "numerical", "pytorch")
        id2 = _make_lead_id("op.a", "mr-1", "shape", "pytorch")
        assert id1 != id2

    def test_id_is_12_chars(self):
        lead_id = _make_lead_id("op", "mr", "numerical", "pytorch")
        assert len(lead_id) == 12


# ── DefectDeduplicator ────────────────────────────────────────────────────────

def _make_mock_pack(
    evidence_id: str,
    operator: str,
    framework: str,
    mr_id: str,
    mr_description: str,
    detail: str,
    timestamp: str = "2026-04-10T12:00:00",
):
    """创建 mock EvidencePack。"""
    pack = MagicMock()
    pack.evidence_id = evidence_id
    pack.operator = operator
    pack.framework = framework
    pack.mr_id = mr_id
    pack.mr_description = mr_description
    pack.detail = detail
    pack.timestamp = timestamp
    return pack


class TestDefectDeduplicator:
    @pytest.fixture
    def dedup_with_collector(self, tmp_path):
        """返回 (dedup, mock_collector) 对，collector 的 list_all 可被控制。"""
        dedup = DefectDeduplicator.__new__(DefectDeduplicator)
        mock_collector = MagicMock()
        dedup._collector = mock_collector
        return dedup, mock_collector

    def test_empty_returns_empty_list(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        collector.list_all.return_value = []
        leads = dedup.deduplicate()
        assert leads == []

    def test_single_pack_becomes_single_lead(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        pack = _make_mock_pack(
            "ev-001", "op.relu", "pytorch", "mr-1", "non-negativity",
            "NUMERICAL_DEVIATION: max_abs=0.5"
        )
        collector.list_all.return_value = [pack]
        leads = dedup.deduplicate()
        assert len(leads) == 1
        assert leads[0].operator == "op.relu"
        assert leads[0].mr_id == "mr-1"
        assert leads[0].error_bucket == "numerical"
        assert leads[0].occurrence_count == 1
        assert leads[0].representative_evidence_id == "ev-001"

    def test_same_signature_clusters_into_one_lead(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack(f"ev-{i:03d}", "op.relu", "pytorch", "mr-1", "desc",
                            "NUMERICAL_DEVIATION: max_abs=0.5", f"2026-04-10T12:0{i}:00")
            for i in range(5)
        ]
        # list_all 返回按时间倒序（最新在前）
        collector.list_all.return_value = list(reversed(packs))
        leads = dedup.deduplicate()
        assert len(leads) == 1
        assert leads[0].occurrence_count == 5

    def test_different_operators_create_separate_leads(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack("ev-001", "op.relu", "pytorch", "mr-1", "desc", "NUMERICAL_DEVIATION: x"),
            _make_mock_pack("ev-002", "op.exp", "pytorch", "mr-1", "desc", "NUMERICAL_DEVIATION: x"),
        ]
        collector.list_all.return_value = packs
        leads = dedup.deduplicate()
        assert len(leads) == 2
        ops = {l.operator for l in leads}
        assert ops == {"op.relu", "op.exp"}

    def test_different_error_buckets_create_separate_leads(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack("ev-001", "op.relu", "pytorch", "mr-1", "desc", "NUMERICAL_DEVIATION: x"),
            _make_mock_pack("ev-002", "op.relu", "pytorch", "mr-1", "desc", "SHAPE_MISMATCH: x"),
        ]
        collector.list_all.return_value = packs
        leads = dedup.deduplicate()
        assert len(leads) == 2

    def test_sorted_by_occurrence_count_descending(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            # op.relu: 3 occurrences
            _make_mock_pack("ev-001", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x"),
            _make_mock_pack("ev-002", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x"),
            _make_mock_pack("ev-003", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x"),
            # op.exp: 1 occurrence
            _make_mock_pack("ev-004", "op.exp", "pytorch", "mr-2", "d", "NUMERICAL_DEVIATION: x"),
        ]
        collector.list_all.return_value = packs
        leads = dedup.deduplicate()
        assert leads[0].operator == "op.relu"
        assert leads[0].occurrence_count == 3
        assert leads[1].occurrence_count == 1

    def test_limit_respected(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack(f"ev-{i:03d}", f"op.{i}", "pytorch", f"mr-{i}", "d",
                            "NUMERICAL_DEVIATION: x")
            for i in range(5)
        ]
        collector.list_all.return_value = packs
        leads = dedup.deduplicate(limit=3)
        assert len(leads) == 3

    def test_filter_by_operator_passed_to_collector(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        collector.list_all.return_value = []
        dedup.deduplicate(operator="op.relu", framework="pytorch")
        collector.list_all.assert_called_once_with(operator="op.relu", framework="pytorch", limit=0)

    def test_first_last_seen_timestamps(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack("ev-002", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x",
                            "2026-04-10T14:00:00"),
            _make_mock_pack("ev-001", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x",
                            "2026-04-10T12:00:00"),
        ]
        collector.list_all.return_value = packs  # 最新在前
        leads = dedup.deduplicate()
        assert len(leads) == 1
        lead = leads[0]
        assert lead.last_seen == "2026-04-10T14:00:00"
        assert lead.first_seen == "2026-04-10T12:00:00"

    def test_representative_evidence_is_newest(self, dedup_with_collector):
        dedup, collector = dedup_with_collector
        packs = [
            _make_mock_pack("ev-newest", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x",
                            "2026-04-10T14:00:00"),
            _make_mock_pack("ev-oldest", "op.relu", "pytorch", "mr-1", "d", "NUMERICAL_DEVIATION: x",
                            "2026-04-10T12:00:00"),
        ]
        collector.list_all.return_value = packs
        leads = dedup.deduplicate()
        assert leads[0].representative_evidence_id == "ev-newest"


# ── format_text ───────────────────────────────────────────────────────────────

class TestFormatText:
    def test_empty_leads_returns_hint(self):
        dedup = DefectDeduplicator.__new__(DefectDeduplicator)
        dedup._collector = MagicMock()
        text = dedup.format_text([])
        assert "deepmt test batch" in text

    def test_text_contains_operator(self):
        dedup = DefectDeduplicator.__new__(DefectDeduplicator)
        dedup._collector = MagicMock()
        lead = DefectLead(
            lead_id="abcdef012345",
            operator="torch.nn.functional.relu",
            framework="pytorch",
            mr_id="mr-1",
            mr_description="output non-negative",
            error_bucket="numerical",
            occurrence_count=3,
            first_seen="2026-04-10T10:00:00",
            last_seen="2026-04-10T12:00:00",
            detail_sample="NUMERICAL_DEVIATION: max_abs=0.5",
            representative_evidence_id="abcdef012345",
        )
        text = dedup.format_text([lead])
        assert "torch.nn.functional.relu" in text
        assert "NUMERICAL" in text
        assert "3" in text  # occurrence count
        assert "abcdef012345" in text

    def test_text_contains_count_header(self):
        dedup = DefectDeduplicator.__new__(DefectDeduplicator)
        dedup._collector = MagicMock()
        lead = DefectLead(
            lead_id="abc",
            operator="op", framework="pytorch", mr_id="m", mr_description="d",
            error_bucket="numerical", occurrence_count=1,
            first_seen="2026-01-01T00:00:00", last_seen="2026-01-01T00:00:00",
            detail_sample="x", representative_evidence_id=None,
        )
        text = dedup.format_text([lead])
        assert "1 条独立模式" in text or "1 条" in text
