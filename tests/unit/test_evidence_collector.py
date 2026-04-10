"""
EvidenceCollector 单元测试。

覆盖：
  - _summarize_tensor：小张量含 values，大张量仅摘要
  - _generate_reproduce_script：脚本结构正确
  - EvidencePack：序列化 / 反序列化
  - EvidenceCollector.create：字段填充正确
  - EvidenceCollector.save / load：持久化往返
  - EvidenceCollector.list_all：过滤与排序
  - EvidenceCollector.count：计数
  - BatchTestRunner.run_operator collect_evidence：失败时生成证据包
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmt.analysis.evidence_collector import (
    EvidenceCollector,
    EvidencePack,
    _generate_reproduce_script,
    _summarize_tensor,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_evidence_dir(tmp_path):
    return tmp_path / "evidence"


@pytest.fixture
def collector(tmp_evidence_dir):
    return EvidenceCollector(evidence_dir=str(tmp_evidence_dir))


def _make_pack(collector: EvidenceCollector, operator="torch.nn.functional.relu", framework="pytorch") -> EvidencePack:
    arr = np.array([[1.0, -1.0], [2.0, 0.5]])
    return collector.create(
        operator=operator,
        framework=framework,
        mr_id="test-mr-id",
        mr_description="output should be non-negative",
        transform_code="lambda k: {**k, 'input': -k['input']}",
        oracle_expr="orig + trans == abs(x)",
        input_tensor=arr,
        actual_diff=0.5,
        tolerance=1e-6,
        detail="NUMERICAL_DEVIATION: max_abs=0.5",
    )


# ── _summarize_tensor ─────────────────────────────────────────────────────────

class TestSummarizeTensor:
    def test_small_tensor_has_values(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _summarize_tensor(arr)
        assert result["shape"] == [3]
        assert "values" in result
        assert result["values"] == pytest.approx([1.0, 2.0, 3.0])

    def test_large_tensor_no_values(self):
        arr = np.ones((20, 20))  # 400 elements > 200 threshold
        result = _summarize_tensor(arr)
        assert result["shape"] == [20, 20]
        assert result["n_elements"] == 400
        assert "values" not in result
        assert "min" in result
        assert "max" in result
        assert "mean" in result

    def test_stats_are_correct(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _summarize_tensor(arr)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)
        assert result["mean"] == pytest.approx(2.5)

    def test_non_array_falls_back(self):
        result = _summarize_tensor("not-a-tensor")
        assert "error" in result

    def test_2d_array(self):
        arr = np.eye(3)
        result = _summarize_tensor(arr)
        assert result["shape"] == [3, 3]
        assert result["n_elements"] == 9
        assert "values" in result  # 9 <= 200


# ── _generate_reproduce_script ────────────────────────────────────────────────

class TestGenerateReproduceScript:
    def test_script_contains_operator(self):
        summary = {"shape": [4], "dtype": "float64", "values": [1.0, 2.0, 3.0, 4.0]}
        script = _generate_reproduce_script(
            operator_name="torch.nn.functional.relu",
            framework="pytorch",
            framework_version="2.0.0",
            mr_description="non-negativity",
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="orig + trans == abs(x)",
            input_summary=summary,
            actual_diff=0.5,
            tolerance=1e-6,
        )
        assert "torch.nn.functional.relu" in script
        assert "non-negativity" in script
        assert "import torch" in script
        assert "orig" in script
        assert "trans" in script

    def test_script_uses_captured_values(self):
        summary = {"shape": [2], "dtype": "float64", "values": [1.0, -1.0]}
        script = _generate_reproduce_script(
            operator_name="torch.relu",
            framework="pytorch",
            framework_version="2.0.0",
            mr_description="test",
            transform_code="lambda k: k",
            oracle_expr="orig == trans",
            input_summary=summary,
            actual_diff=0.0,
            tolerance=1e-6,
        )
        assert "torch.tensor([1.0, -1.0]" in script

    def test_script_uses_random_when_no_values(self):
        summary = {"shape": [10, 10], "dtype": "float64"}
        script = _generate_reproduce_script(
            operator_name="torch.relu",
            framework="pytorch",
            framework_version="2.0.0",
            mr_description="test",
            transform_code="lambda k: k",
            oracle_expr="orig == trans",
            input_summary=summary,
            actual_diff=0.0,
            tolerance=1e-6,
        )
        assert "torch.randn" in script
        assert "manual_seed" in script


# ── EvidencePack ──────────────────────────────────────────────────────────────

class TestEvidencePack:
    def test_to_dict_and_from_dict(self, collector):
        pack = _make_pack(collector)
        d = pack.to_dict()
        restored = EvidencePack.from_dict(d)
        assert restored.evidence_id == pack.evidence_id
        assert restored.operator == pack.operator
        assert restored.actual_diff == pack.actual_diff
        assert restored.reproduce_script == pack.reproduce_script

    def test_all_fields_present(self, collector):
        pack = _make_pack(collector)
        d = pack.to_dict()
        required = [
            "evidence_id", "timestamp", "operator", "framework", "framework_version",
            "mr_id", "mr_description", "transform_code", "oracle_expr",
            "input_summary", "actual_diff", "tolerance", "detail", "reproduce_script",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"


# ── EvidenceCollector.create ──────────────────────────────────────────────────

class TestEvidenceCollectorCreate:
    def test_creates_pack_with_correct_fields(self, collector):
        pack = _make_pack(collector)
        assert pack.operator == "torch.nn.functional.relu"
        assert pack.framework == "pytorch"
        assert pack.mr_id == "test-mr-id"
        assert pack.actual_diff == pytest.approx(0.5)
        assert pack.tolerance == pytest.approx(1e-6)
        assert len(pack.evidence_id) == 12
        assert pack.reproduce_script != ""

    def test_input_summary_contains_shape(self, collector):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        pack = collector.create(
            operator="op", framework="pytorch", mr_id="m1", mr_description="d",
            transform_code="t", oracle_expr="e", input_tensor=arr,
            actual_diff=0.0, tolerance=1e-6, detail="x",
        )
        assert pack.input_summary["shape"] == [2, 2]
        assert "values" in pack.input_summary  # 4 elements ≤ 200

    def test_unique_ids(self, collector):
        arr = np.zeros(3)
        ids = set()
        for _ in range(5):
            pack = collector.create(
                operator="op", framework="pytorch", mr_id="m", mr_description="d",
                transform_code="t", oracle_expr="e", input_tensor=arr,
                actual_diff=0.0, tolerance=1e-6, detail="x",
            )
            ids.add(pack.evidence_id)
        assert len(ids) == 5


# ── EvidenceCollector.save / load ─────────────────────────────────────────────

class TestEvidenceCollectorPersistence:
    def test_save_creates_file(self, collector, tmp_evidence_dir):
        pack = _make_pack(collector)
        path = collector.save(pack)
        assert path.exists()
        assert path.suffix == ".json"
        assert pack.evidence_id in path.name

    def test_load_roundtrip(self, collector):
        pack = _make_pack(collector)
        collector.save(pack)
        loaded = collector.load(pack.evidence_id)
        assert loaded is not None
        assert loaded.evidence_id == pack.evidence_id
        assert loaded.operator == pack.operator
        assert loaded.actual_diff == pytest.approx(pack.actual_diff)

    def test_load_nonexistent_returns_none(self, collector):
        result = collector.load("nonexistent-id-xyz")
        assert result is None

    def test_file_is_valid_json(self, collector, tmp_evidence_dir):
        pack = _make_pack(collector)
        path = collector.save(pack)
        with open(path) as f:
            data = json.load(f)
        assert data["evidence_id"] == pack.evidence_id


# ── EvidenceCollector.list_all / count ───────────────────────────────────────

class TestEvidenceCollectorQuery:
    def test_list_all_empty(self, collector):
        assert collector.list_all() == []

    def test_list_all_returns_saved_packs(self, collector):
        pack1 = _make_pack(collector, operator="op.a")
        pack2 = _make_pack(collector, operator="op.b")
        collector.save(pack1)
        collector.save(pack2)
        packs = collector.list_all()
        assert len(packs) == 2

    def test_list_all_filter_by_operator(self, collector):
        pack1 = _make_pack(collector, operator="op.a")
        pack2 = _make_pack(collector, operator="op.b")
        collector.save(pack1)
        collector.save(pack2)
        packs = collector.list_all(operator="op.a")
        assert len(packs) == 1
        assert packs[0].operator == "op.a"

    def test_list_all_filter_by_framework(self, collector):
        pack1 = _make_pack(collector, framework="pytorch")
        pack2 = _make_pack(collector, framework="pytorch")
        collector.save(pack1)
        collector.save(pack2)
        packs = collector.list_all(framework="tensorflow")
        assert len(packs) == 0

    def test_list_all_limit(self, collector):
        for i in range(5):
            pack = _make_pack(collector)
            collector.save(pack)
        packs = collector.list_all(limit=3)
        assert len(packs) == 3

    def test_list_all_sorted_by_time_desc(self, collector):
        import time
        pack1 = _make_pack(collector)
        collector.save(pack1)
        time.sleep(0.01)
        pack2 = _make_pack(collector)
        collector.save(pack2)
        packs = collector.list_all()
        # 最新的在前
        assert packs[0].timestamp >= packs[1].timestamp

    def test_count(self, collector):
        assert collector.count() == 0
        pack = _make_pack(collector)
        collector.save(pack)
        assert collector.count() == 1
        pack2 = _make_pack(collector)
        collector.save(pack2)
        assert collector.count() == 2


# ── BatchTestRunner integration with collect_evidence ─────────────────────────

class TestBatchTestRunnerEvidence:
    """验证 BatchTestRunner.run_operator(collect_evidence=True) 在失败时生成证据包。"""

    def _make_failing_mr(self):
        from deepmt.ir.schema import MetamorphicRelation
        return MetamorphicRelation(
            id="failing-mr-id-123",
            description="always failing test",
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="orig == trans",  # 对 relu 必然失败
            verified=True,
        )

    def test_collect_evidence_on_failure(self, tmp_path):
        from deepmt.engine.batch_test_runner import BatchTestRunner
        from deepmt.analysis.evidence_collector import EvidenceCollector

        ev_dir = tmp_path / "ev"
        collector = EvidenceCollector(evidence_dir=str(ev_dir))

        # 构造一个 mock MR 知识库，返回必然失败的 MR
        mock_repo = MagicMock()
        mr = self._make_failing_mr()
        mock_repo.load.return_value = [mr]

        # 构造 mock ResultsManager
        mock_rm = MagicMock()

        runner = BatchTestRunner(
            repo=mock_repo,
            results_manager=mock_rm,
            evidence_collector=collector,
        )

        import torch

        def real_relu(**kwargs):
            return torch.relu(kwargs["input"])

        summary = runner.run_operator(
            operator_name="torch.nn.functional.relu",
            framework="pytorch",
            n_samples=3,
            operator_func=real_relu,
            collect_evidence=True,
        )

        # 应该有失败，且生成了证据包
        assert summary.failed > 0
        assert len(summary.evidence_ids) > 0
        assert collector.count() > 0

    def test_no_evidence_when_flag_off(self, tmp_path):
        from deepmt.engine.batch_test_runner import BatchTestRunner
        from deepmt.analysis.evidence_collector import EvidenceCollector

        ev_dir = tmp_path / "ev"
        collector = EvidenceCollector(evidence_dir=str(ev_dir))

        mock_repo = MagicMock()
        mr = self._make_failing_mr()
        mock_repo.load.return_value = [mr]
        mock_rm = MagicMock()

        runner = BatchTestRunner(
            repo=mock_repo,
            results_manager=mock_rm,
            evidence_collector=collector,
        )

        import torch

        def real_relu(**kwargs):
            return torch.relu(kwargs["input"])

        summary = runner.run_operator(
            operator_name="torch.nn.functional.relu",
            framework="pytorch",
            n_samples=3,
            operator_func=real_relu,
            collect_evidence=False,
        )

        # 无证据包
        assert collector.count() == 0
        assert summary.evidence_ids == []
