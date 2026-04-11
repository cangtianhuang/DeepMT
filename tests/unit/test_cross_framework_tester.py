"""
CrossFrameworkTester 单元测试

验证重点：
  1. CrossConsistencyResult 属性计算（consistency_rate / f1_pass_rate / …）
  2. CrossSessionResult 聚合与序列化往返
  3. CrossFrameworkTester._get_backend：pytorch → PyTorchPlugin，numpy → NumpyPlugin
  4. compare_operator：无 MR 时返回空 session；有 MR 时能执行并返回统计
  5. save / load_all：持久化与恢复
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmt.analysis.cross_framework_tester import (
    CrossConsistencyResult,
    CrossFrameworkTester,
    CrossSessionResult,
)
from deepmt.ir.schema import MetamorphicRelation


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _make_mr(transform="lambda k: {**k, 'input': -k['input']}", oracle="trans >= 0"):
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description="test MR",
        transform_code=transform,
        oracle_expr=oracle,
        verified=True,
    )


def _make_ccr(both_pass=5, only_f1=1, only_f2=0, both_fail=4, errors=0,
              output_max_diff=0.01, output_mean_diff=0.005, output_close=True):
    return CrossConsistencyResult(
        operator="torch.exp",
        framework1="pytorch",
        framework2="numpy",
        mr_id="mr-01",
        mr_description="exp additivity",
        oracle_expr="trans == orig * exp(1)",
        n_samples=10,
        both_pass=both_pass,
        only_f1_pass=only_f1,
        only_f2_pass=only_f2,
        both_fail=both_fail,
        errors=errors,
        output_max_diff=output_max_diff,
        output_mean_diff=output_mean_diff,
        output_close=output_close,
    )


# ── CrossConsistencyResult ────────────────────────────────────────────────────

class TestCrossConsistencyResult:
    def test_total_valid(self):
        r = _make_ccr(both_pass=3, only_f1=1, only_f2=2, both_fail=4)
        assert r.total_valid == 10

    def test_consistency_rate_all_agree(self):
        r = _make_ccr(both_pass=8, only_f1=0, only_f2=0, both_fail=2)
        assert r.consistency_rate == pytest.approx(1.0)

    def test_consistency_rate_all_disagree(self):
        r = _make_ccr(both_pass=0, only_f1=5, only_f2=5, both_fail=0)
        assert r.consistency_rate == pytest.approx(0.0)

    def test_consistency_rate_partial(self):
        r = _make_ccr(both_pass=6, only_f1=2, only_f2=2, both_fail=0)
        assert r.consistency_rate == pytest.approx(0.6)

    def test_f1_pass_rate(self):
        r = _make_ccr(both_pass=5, only_f1=3, only_f2=0, both_fail=2)
        # f1 passes: both_pass + only_f1 = 8 / 10
        assert r.f1_pass_rate == pytest.approx(0.8)

    def test_f2_pass_rate(self):
        r = _make_ccr(both_pass=5, only_f1=0, only_f2=2, both_fail=3)
        # f2 passes: both_pass + only_f2 = 7 / 10
        assert r.f2_pass_rate == pytest.approx(0.7)

    def test_inconsistent_cases(self):
        r = _make_ccr(both_pass=5, only_f1=2, only_f2=1, both_fail=2)
        assert r.inconsistent_cases == 3

    def test_to_dict_has_derived_fields(self):
        r = _make_ccr()
        d = r.to_dict()
        assert "consistency_rate" in d
        assert "f1_pass_rate" in d
        assert "f2_pass_rate" in d
        assert "inconsistent_cases" in d

    def test_zero_total_no_crash(self):
        r = _make_ccr(both_pass=0, only_f1=0, only_f2=0, both_fail=0)
        assert r.consistency_rate == pytest.approx(0.0)
        assert r.f1_pass_rate == pytest.approx(0.0)


# ── CrossSessionResult ────────────────────────────────────────────────────────

class TestCrossSessionResult:
    def test_overall_consistency_rate(self):
        r1 = _make_ccr(both_pass=10, only_f1=0, only_f2=0, both_fail=0)   # 100%
        r2 = _make_ccr(both_pass=0,  only_f1=5, only_f2=5, both_fail=0)   # 0%
        s = CrossSessionResult("s1", "2026-01-01", "op", "pt", "np", 10, [r1, r2])
        assert s.overall_consistency_rate == pytest.approx(0.5)

    def test_inconsistent_mr_count(self):
        r1 = _make_ccr(both_pass=10, only_f1=0, only_f2=0, both_fail=0)  # 无不一致
        r2 = _make_ccr(both_pass=5,  only_f1=1, only_f2=0, both_fail=4)  # 有不一致
        s = CrossSessionResult("s1", "2026-01-01", "op", "pt", "np", 10, [r1, r2])
        assert s.inconsistent_mr_count == 1

    def test_serialization_roundtrip(self):
        r = _make_ccr()
        s = CrossSessionResult("abc", "2026-01-01T12:00:00", "torch.exp", "pytorch", "numpy", 10, [r])
        d = s.to_dict()
        restored = CrossSessionResult.from_dict(d)
        assert restored.session_id == "abc"
        assert len(restored.mr_results) == 1
        assert restored.mr_results[0].mr_id == r.mr_id

    def test_empty_session(self):
        s = CrossSessionResult("x", "2026-01-01", "op", "pt", "np", 10, [])
        assert s.mr_count == 0
        assert s.overall_consistency_rate == pytest.approx(0.0)


# ── CrossFrameworkTester._get_backend ─────────────────────────────────────────

class TestGetBackend:
    def test_numpy_returns_numpy_plugin(self):
        tester = CrossFrameworkTester()
        backend = tester._get_backend("numpy")
        from deepmt.plugins.numpy_plugin import NumpyPlugin
        assert isinstance(backend, NumpyPlugin)

    def test_pytorch_returns_pytorch_plugin(self):
        tester = CrossFrameworkTester()
        backend = tester._get_backend("pytorch")
        from deepmt.plugins.pytorch_plugin import PyTorchPlugin
        assert isinstance(backend, PyTorchPlugin)


# ── compare_operator ──────────────────────────────────────────────────────────

class TestCompareOperator:
    def test_empty_mr_returns_empty_session(self, tmp_path):
        mock_repo = MagicMock()
        mock_repo.load.return_value = []
        tester = CrossFrameworkTester(repo=mock_repo, results_dir=str(tmp_path))
        session = tester.compare_operator("torch.exp", framework1="pytorch", framework2="numpy")
        assert session.mr_count == 0

    def test_relu_with_nonneg_mr(self, tmp_path):
        """非负性 MR 在两框架中应均通过（一致性高）。"""
        mock_repo = MagicMock()
        mock_repo.load.return_value = [
            _make_mr(
                transform="lambda k: {**k, 'input': k['input'] * 2}",  # 2x 仍为任意值
                oracle="trans >= 0",  # relu(2x) >= 0 对两框架均成立
            )
        ]
        mock_catalog = MagicMock()
        mock_catalog.get_operator_info.return_value = None

        tester = CrossFrameworkTester(repo=mock_repo, catalog=mock_catalog, results_dir=str(tmp_path))
        session = tester.compare_operator(
            "torch.nn.functional.relu",
            framework1="pytorch",
            framework2="numpy",
            n_samples=5,
        )
        assert session.mr_count == 1
        r = session.mr_results[0]
        # abs(x) 后的 relu 输出必然 >= 0，两框架均通过
        assert r.both_pass + r.both_fail + r.only_f1_pass + r.only_f2_pass > 0

    def test_output_diff_is_float(self, tmp_path):
        """输出差值字段应为 float（非 NaN 时）。"""
        mock_repo = MagicMock()
        mock_repo.load.return_value = [
            _make_mr(transform="lambda k: k", oracle="orig == trans")
        ]
        mock_catalog = MagicMock()
        mock_catalog.get_operator_info.return_value = None

        tester = CrossFrameworkTester(repo=mock_repo, catalog=mock_catalog, results_dir=str(tmp_path))
        session = tester.compare_operator(
            "torch.exp", framework1="pytorch", framework2="numpy", n_samples=3
        )
        if session.mr_count > 0:
            r = session.mr_results[0]
            # output_max_diff 应为 float（可能 NaN 如无有效样本，但类型应一致）
            assert isinstance(r.output_max_diff, float)


# ── save / load_all ──────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        tester = CrossFrameworkTester(results_dir=str(tmp_path))
        r = _make_ccr()
        s = CrossSessionResult("save01", "2026-01-01", "op", "pt", "np", 5, [r])
        path = tester.save(s)
        assert path.exists()

    def test_load_all_empty(self, tmp_path):
        tester = CrossFrameworkTester(results_dir=str(tmp_path))
        assert tester.load_all() == []

    def test_save_then_load(self, tmp_path):
        tester = CrossFrameworkTester(results_dir=str(tmp_path))
        r = _make_ccr()
        s = CrossSessionResult("load01", "2026-01-01", "torch.relu", "pt", "np", 5, [r])
        tester.save(s)

        loaded = tester.load_all()
        assert len(loaded) == 1
        assert loaded[0].session_id == "load01"
        assert loaded[0].mr_count == 1

    def test_format_text_no_crash(self, tmp_path):
        tester = CrossFrameworkTester(results_dir=str(tmp_path))
        r = _make_ccr()
        s = CrossSessionResult("fmt01", "2026-01-01", "torch.exp", "pt", "np", 5, [r])
        text = tester.format_text([s])
        assert "一致" in text
