"""
Phase K 治理机制回归测试

覆盖：
    - QualityLevel 枚举与质量等级计算
    - ProvenanceInfo 构建与序列化
    - MRRepository 多层支持（operator/model/application）
    - MRRepository lifecycle_state 序列化/反序列化
    - MRRepository retire / update_lifecycle
    - MRRepository get_statistics 质量分布
    - MRDeduplicator 重复检测
    - MRGovernanceManager 入库检查、批量准入、退役、晋升
    - RepoAuditor 审计报告

无 LLM / 网络依赖。
"""

import pytest

from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.mr_governance.quality import (
    QualityLevel,
    filter_by_quality,
    quality_level_from_mr,
)
from deepmt.mr_governance.provenance import ProvenanceInfo, build_provenance
from deepmt.mr_governance.deduplicator import MRDeduplicator
from deepmt.mr_governance.governance import MRGovernanceManager


# ── 工厂函数 ──────────────────────────────────────────────────────────────────


def _mr(
    mr_id: str,
    oracle_expr: str = "orig == trans",
    lifecycle_state: str = "pending",
    source: str = "template",
    layer: str = "operator",
    verified: bool = False,
    checked: bool = False,
    proven: bool = False,
) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=mr_id,
        description=f"MR {mr_id}",
        transform_code="lambda x: x",
        transform=lambda x: x,
        oracle_expr=oracle_expr,
        lifecycle_state=lifecycle_state,
        source=source,
        layer=layer,
        verified=verified,
        checked=checked,
        proven=proven,
    )


# ── QualityLevel ─────────────────────────────────────────────────────────────


class TestQualityLevel:
    def test_from_lifecycle_pending(self):
        assert QualityLevel.from_lifecycle("pending") == QualityLevel.CANDIDATE

    def test_from_lifecycle_checked(self):
        assert QualityLevel.from_lifecycle("checked") == QualityLevel.CHECKED

    def test_from_lifecycle_proven(self):
        assert QualityLevel.from_lifecycle("proven") == QualityLevel.PROVEN

    def test_from_lifecycle_curated(self):
        assert QualityLevel.from_lifecycle("curated") == QualityLevel.CURATED

    def test_from_lifecycle_retired(self):
        assert QualityLevel.from_lifecycle("retired") == QualityLevel.RETIRED

    def test_ordering(self):
        assert QualityLevel.RETIRED < QualityLevel.CANDIDATE < QualityLevel.CHECKED
        assert QualityLevel.CHECKED < QualityLevel.PROVEN < QualityLevel.CURATED

    def test_label(self):
        assert QualityLevel.PROVEN.label == "proven"
        assert str(QualityLevel.CURATED) == "curated"

    def test_quality_level_from_mr(self):
        mr = _mr("x", lifecycle_state="checked")
        assert quality_level_from_mr(mr) == QualityLevel.CHECKED

    def test_filter_by_quality_exclude_low(self):
        mrs = [
            _mr("a", lifecycle_state="pending"),
            _mr("b", lifecycle_state="checked"),
            _mr("c", lifecycle_state="proven"),
        ]
        result = filter_by_quality(mrs, min_quality=QualityLevel.CHECKED)
        ids = [m.id for m in result]
        assert "a" not in ids
        assert "b" in ids
        assert "c" in ids

    def test_filter_by_quality_exclude_retired(self):
        mrs = [
            _mr("a", lifecycle_state="proven"),
            _mr("b", lifecycle_state="retired"),
        ]
        result = filter_by_quality(mrs, min_quality=QualityLevel.CANDIDATE, exclude_retired=True)
        assert len(result) == 1
        assert result[0].id == "a"


# ── QualityLevel property on MetamorphicRelation ─────────────────────────────


class TestMRQualityProperty:
    def test_quality_level_property_pending(self):
        mr = _mr("x", lifecycle_state="pending")
        assert mr.quality_level == "candidate"

    def test_quality_level_property_proven(self):
        mr = _mr("x", lifecycle_state="proven")
        assert mr.quality_level == "proven"

    def test_quality_level_property_curated(self):
        mr = _mr("x", lifecycle_state="curated")
        assert mr.quality_level == "curated"

    def test_sync_lifecycle_skips_curated(self):
        mr = _mr("x", lifecycle_state="curated")
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "curated"  # 不被降级

    def test_sync_lifecycle_skips_retired(self):
        mr = _mr("x", lifecycle_state="retired")
        mr.sync_lifecycle()
        assert mr.lifecycle_state == "retired"


# ── ProvenanceInfo ────────────────────────────────────────────────────────────


class TestProvenanceInfo:
    def test_to_dict_omits_none(self):
        info = ProvenanceInfo(generator_id="test_gen")
        d = info.to_dict()
        assert "generator_id" in d
        assert "prompt_hash" not in d  # None 省略

    def test_from_dict_roundtrip(self):
        info = ProvenanceInfo(generator_id="gen_v1", source_detail="llm:gpt-4")
        d = info.to_dict()
        restored = ProvenanceInfo.from_dict(d)
        assert restored.generator_id == "gen_v1"
        assert restored.source_detail == "llm:gpt-4"

    def test_build_provenance_with_prompt(self):
        prov = build_provenance(
            generator_id="my_gen",
            prompt_text="test prompt",
        )
        assert prov["generator_id"] == "my_gen"
        assert "prompt_hash" in prov
        assert len(prov["prompt_hash"]) == 8  # sha256 前8位

    def test_build_provenance_no_prompt(self):
        prov = build_provenance(generator_id="gen")
        assert "prompt_hash" not in prov


# ── MRRepository 多层支持 ─────────────────────────────────────────────────────


class TestMRRepositoryMultiLayer:
    def test_invalid_layer_raises(self, tmp_path):
        with pytest.raises(ValueError):
            MRRepository(layer="invalid", repo_dir=str(tmp_path))

    def test_operator_layer(self, tmp_path):
        r = MRRepository(layer="operator", repo_dir=str(tmp_path))
        assert r.layer == "operator"
        assert "operator" in str(r.repo_dir)

    def test_model_layer(self, tmp_path):
        r = MRRepository(layer="model", repo_dir=str(tmp_path))
        assert r.layer == "model"
        assert "model" in str(r.repo_dir)

    def test_application_layer(self, tmp_path):
        r = MRRepository(layer="application", repo_dir=str(tmp_path))
        assert r.layer == "application"

    def test_layers_are_independent(self, tmp_path):
        r_op = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r_mo = MRRepository(layer="model", repo_dir=str(tmp_path))

        mr_op = _mr("op-001", layer="operator")
        mr_mo = _mr("mo-001", layer="model")

        r_op.save("torch.relu", [mr_op])
        r_mo.save("ResNet50", [mr_mo])

        assert r_op.exists("torch.relu")
        assert not r_op.exists("ResNet50")
        assert r_mo.exists("ResNet50")
        assert not r_mo.exists("torch.relu")

    def test_list_subjects_alias(self, tmp_path):
        r = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r.save("op1", [_mr("a")])
        r.save("op2", [_mr("b")])
        subjects = r.list_subjects()
        assert "op1" in subjects and "op2" in subjects

        # 向后兼容别名
        assert r.list_operators() == r.list_subjects()


# ── lifecycle_state 序列化/反序列化 ──────────────────────────────────────────


class TestLifecycleSerialization:
    @pytest.fixture
    def repo(self, tmp_path):
        return MRRepository(layer="operator", repo_dir=str(tmp_path))

    def test_pending_state_persists(self, repo):
        mr = _mr("a", lifecycle_state="pending")
        repo.save("op", [mr])
        loaded = repo.load("op")
        assert loaded[0].lifecycle_state == "pending"

    def test_proven_state_persists(self, repo):
        mr = _mr("b", lifecycle_state="proven", verified=True, proven=True)
        repo.save("op", [mr])
        loaded = repo.load("op")
        assert loaded[0].lifecycle_state == "proven"

    def test_curated_state_persists(self, repo):
        mr = _mr("c", lifecycle_state="curated")
        repo.save("op", [mr])
        loaded = repo.load("op")
        assert loaded[0].lifecycle_state == "curated"

    def test_provenance_persists(self, repo):
        prov = build_provenance("test_gen", source_detail="template_v1")
        mr = _mr("d")
        mr.provenance = prov
        repo.save("op", [mr])
        loaded = repo.load("op")
        assert loaded[0].provenance.get("generator_id") == "test_gen"


# ── retire / update_lifecycle ─────────────────────────────────────────────────


class TestRepoRetireAndUpdate:
    @pytest.fixture
    def repo(self, tmp_path):
        r = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r.save("relu", [_mr("mr-1"), _mr("mr-2", lifecycle_state="checked")])
        return r

    def test_retire_marks_as_retired(self, repo):
        assert repo.retire("relu", "mr-1") is True
        mrs = repo.load("relu")
        retired = [m for m in mrs if m.id == "mr-1"]
        assert retired[0].lifecycle_state == "retired"

    def test_retire_nonexistent_returns_false(self, repo):
        assert repo.retire("relu", "nonexistent") is False

    def test_retire_preserves_other_mrs(self, repo):
        repo.retire("relu", "mr-1")
        mrs = repo.load("relu")
        other = [m for m in mrs if m.id == "mr-2"]
        assert other[0].lifecycle_state == "checked"

    def test_update_lifecycle_to_curated(self, repo):
        assert repo.update_lifecycle("relu", "mr-1", "curated") is True
        mrs = repo.load("relu")
        curated = [m for m in mrs if m.id == "mr-1"]
        assert curated[0].lifecycle_state == "curated"
        assert curated[0].verified is True

    def test_update_lifecycle_invalid_state_raises(self, repo):
        with pytest.raises(ValueError):
            repo.update_lifecycle("relu", "mr-1", "unknown_state")


# ── get_statistics 质量分布 ───────────────────────────────────────────────────


class TestGetStatisticsQuality:
    @pytest.fixture
    def repo(self, tmp_path):
        r = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r.save("op1", [
            _mr("a", lifecycle_state="pending"),
            _mr("b", lifecycle_state="checked"),
            _mr("c", lifecycle_state="proven", verified=True),
        ])
        return r

    def test_quality_dist_in_stats(self, repo):
        stats = repo.get_statistics("op1")
        qd = stats["quality_dist"]
        assert qd.get("candidate", 0) == 1
        assert qd.get("checked", 0) == 1
        assert qd.get("proven", 0) == 1

    def test_retired_count(self, repo):
        repo.retire("op1", "a")
        stats = repo.get_statistics("op1")
        assert stats["retired"] == 1

    def test_global_stats(self, repo):
        stats = repo.get_statistics()
        assert stats["total_mrs"] == 3


# ── min_quality 过滤 ──────────────────────────────────────────────────────────


class TestLoadMinQuality:
    @pytest.fixture
    def repo(self, tmp_path):
        r = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r.save("op", [
            _mr("p", lifecycle_state="pending"),
            _mr("c", lifecycle_state="checked"),
            _mr("v", lifecycle_state="proven"),
        ])
        return r

    def test_min_quality_checked(self, repo):
        mrs = repo.load("op", min_quality="checked")
        ids = [m.id for m in mrs]
        assert "p" not in ids
        assert "c" in ids
        assert "v" in ids

    def test_min_quality_proven(self, repo):
        mrs = repo.load("op", min_quality="proven")
        ids = [m.id for m in mrs]
        assert "p" not in ids
        assert "c" not in ids
        assert "v" in ids


# ── MRDeduplicator ────────────────────────────────────────────────────────────


class TestMRDeduplicator:
    def test_no_duplicates(self):
        mrs = [
            _mr("a", oracle_expr="orig == trans"),
            _mr("b", oracle_expr="trans == 2 * orig"),
        ]
        dedup = MRDeduplicator()
        groups = dedup.find_duplicates(mrs)
        assert len(groups) == 0

    def test_strong_duplicate(self):
        mrs = [
            _mr("a", oracle_expr="orig == trans", lifecycle_state="pending"),
            _mr("b", oracle_expr="orig == trans", lifecycle_state="proven"),
        ]
        dedup = MRDeduplicator()
        groups = dedup.find_duplicates(mrs)
        assert len(groups) == 1
        g = groups[0]
        # 保留质量更高的 b
        assert g.canonical_id == "b"
        assert "a" in g.duplicate_ids

    def test_whitespace_normalization(self):
        mrs = [
            _mr("a", oracle_expr="orig  ==  trans"),
            _mr("b", oracle_expr="orig==trans"),
        ]
        dedup = MRDeduplicator()
        groups = dedup.find_duplicates(mrs)
        assert len(groups) == 1

    def test_filter_unique(self):
        mrs = [
            _mr("a", oracle_expr="orig == trans"),
            _mr("b", oracle_expr="orig == trans"),
            _mr("c", oracle_expr="trans == -orig"),
        ]
        dedup = MRDeduplicator()
        unique, groups = dedup.filter_unique(mrs)
        assert len(unique) == 2  # 保留1个重复中的canonical + c
        assert len(groups) == 1


# ── MRGovernanceManager ───────────────────────────────────────────────────────


class TestMRGovernanceManager:
    def test_admit_check_passes_checked(self):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CHECKED)
        mr = _mr("a", lifecycle_state="checked")
        result = mgr.admit_check(mr)
        assert result.admitted is True

    def test_admit_check_rejects_pending(self):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CHECKED)
        mr = _mr("a", lifecycle_state="pending")
        result = mgr.admit_check(mr)
        assert result.admitted is False
        assert "质量不足" in result.reason

    def test_admit_check_rejects_retired(self):
        mgr = MRGovernanceManager()
        mr = _mr("a", lifecycle_state="retired")
        result = mgr.admit_check(mr)
        assert result.admitted is False
        assert "retired" in result.reason

    def test_admit_check_rejects_empty_oracle(self):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CANDIDATE)
        mr = _mr("a", oracle_expr="", lifecycle_state="proven")
        result = mgr.admit_check(mr)
        assert result.admitted is False
        assert "oracle_expr" in result.reason

    def test_admit_batch_filters_low_quality(self):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CHECKED)
        mrs = [
            _mr("good", lifecycle_state="checked"),
            _mr("bad", lifecycle_state="pending"),
        ]
        report = mgr.admit_batch(mrs)
        assert "good" in report.admitted
        assert "bad" not in report.admitted
        assert report.rejected_count == 1

    def test_admit_batch_deduplicates(self):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CANDIDATE)
        mrs = [
            _mr("a", oracle_expr="orig == trans", lifecycle_state="proven"),
            _mr("b", oracle_expr="orig == trans", lifecycle_state="pending"),
        ]
        report = mgr.admit_batch(mrs)
        # 两条通过基础检查，但 b 被去重
        assert "a" in report.admitted
        assert "b" not in report.admitted
        assert len(report.duplicate_groups) == 1

    def test_admit_batch_saves_to_repo(self, tmp_path):
        mgr = MRGovernanceManager(min_quality=QualityLevel.CANDIDATE)
        repo = MRRepository(layer="operator", repo_dir=str(tmp_path))
        mrs = [_mr("a", lifecycle_state="checked")]
        mgr.admit_batch(mrs, repo=repo, subject_name="my_op")
        assert repo.exists("my_op")
        loaded = repo.load("my_op")
        assert loaded[0].id == "a"

    def test_retire_via_manager(self, tmp_path):
        mgr = MRGovernanceManager()
        repo = MRRepository(layer="operator", repo_dir=str(tmp_path))
        repo.save("op", [_mr("x", lifecycle_state="checked")])
        success = mgr.retire(repo, "op", "x")
        assert success is True
        loaded = repo.load("op")
        assert loaded[0].lifecycle_state == "retired"

    def test_promote_to_curated(self, tmp_path):
        mgr = MRGovernanceManager()
        repo = MRRepository(layer="operator", repo_dir=str(tmp_path))
        repo.save("op", [_mr("x", lifecycle_state="proven")])
        success = mgr.promote(repo, "op", "x", new_state="curated")
        assert success is True
        loaded = repo.load("op")
        assert loaded[0].lifecycle_state == "curated"

    def test_retire_duplicates(self, tmp_path):
        mgr = MRGovernanceManager()
        repo = MRRepository(layer="operator", repo_dir=str(tmp_path))
        repo.save("op", [
            _mr("a", oracle_expr="orig == trans", lifecycle_state="proven"),
            _mr("b", oracle_expr="orig == trans", lifecycle_state="pending"),
        ])
        count = mgr.retire_duplicates(repo, "op")
        assert count == 1
        # b 被退役
        mrs = repo.load("op")
        b_mr = [m for m in mrs if m.id == "b"]
        assert b_mr[0].lifecycle_state == "retired"


# ── RepoAuditor ───────────────────────────────────────────────────────────────


class TestRepoAuditor:
    @pytest.fixture
    def repo_dir(self, tmp_path):
        """预填充多层测试数据。"""
        r_op = MRRepository(layer="operator", repo_dir=str(tmp_path))
        r_op.save("relu", [
            _mr("o1", lifecycle_state="proven", verified=True),
            _mr("o2", lifecycle_state="pending"),
            _mr("o3", lifecycle_state="retired"),
        ])

        r_mo = MRRepository(layer="model", repo_dir=str(tmp_path))
        r_mo.save("ResNet50", [
            _mr("m1", lifecycle_state="checked", layer="model"),
        ])
        return str(tmp_path)

    def test_audit_total_mrs(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit()
        assert report.total_mrs == 4  # 3 operator + 1 model

    def test_audit_retired_count(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit()
        assert report.total_retired == 1

    def test_audit_quality_distribution(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit()
        qd = report.quality_distribution()
        assert qd.get("proven", 0) >= 1
        assert qd.get("candidate", 0) >= 1
        assert qd.get("retired", 0) >= 1

    def test_audit_single_layer(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit(layers=["operator"])
        assert "operator" in report.layers
        assert "model" not in report.layers

    def test_audit_anomalies_retired_ratio(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit(layers=["operator"])
        # operator 层 3 条中 1 条 retired（33.3% > 30%）
        anomalies = report.anomalies()
        assert any("退役比例过高" in a for a in anomalies)

    def test_audit_summary_text(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit()
        text = report.summary_text()
        assert "MR 知识库审计报告" in text
        assert "质量等级分布" in text

    def test_pending_review_list(self, repo_dir):
        from deepmt.analysis.repo_audit import RepoAuditor
        from deepmt.mr_governance.quality import QualityLevel
        auditor = RepoAuditor(repo_dir=repo_dir)
        report = auditor.run_audit()
        items = report.pending_review_list(min_quality=QualityLevel.PROVEN)
        # candidate 和 checked 的 MR 应出现在待复核列表中
        total_count = sum(item["count"] for item in items)
        assert total_count >= 2  # o2(pending) + m1(checked)
