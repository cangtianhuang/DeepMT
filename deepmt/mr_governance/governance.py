"""MR 治理管理器（K4）：统一入库规则与淘汰规则。

MRGovernanceManager 是 Phase K 的核心治理接口，负责：
    1. 入库检查 (admit_check)     — 判断 MR 是否满足入库条件
    2. 批量准入 (admit_batch)     — 过滤出可入库的 MR 列表
    3. 重复检测 (find_duplicates) — 委托 MRDeduplicator
    4. 归档退役 (retire)          — 在 repository 中将 MR 标记为 retired
    5. 质量晋升 (promote)         — 将 MR lifecycle_state 升级

入库门槛默认为 CHECKED（数值 precheck 通过），可按实验需求调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from deepmt.core.logger import logger
from deepmt.mr_governance.deduplicator import DuplicateGroup, MRDeduplicator
from deepmt.mr_governance.quality import QualityLevel

if TYPE_CHECKING:
    from deepmt.ir.schema import MetamorphicRelation
    from deepmt.mr_generator.base.mr_repository import MRRepository


@dataclass
class AdmitResult:
    """单条 MR 入库检查结果。"""

    mr_id: str
    admitted: bool
    reason: str = ""


@dataclass
class GovernanceReport:
    """一次 admit_batch 的治理报告。"""

    admitted: List[str] = field(default_factory=list)      # 通过入库的 MR ID
    rejected: List[AdmitResult] = field(default_factory=list)  # 被拒绝的 MR
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)

    @property
    def admitted_count(self) -> int:
        return len(self.admitted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class MRGovernanceManager:
    """MR 治理管理器。

    用法::

        mgr = MRGovernanceManager()
        result = mgr.admit_batch(mrs, repo, subject_name="relu")
        print(result.admitted_count, "条入库，", result.rejected_count, "条被拒")

        # 退役低质量 MR
        mgr.retire(repo, subject_name="relu", mr_id="abc-123")

        # 晋升 MR 至 curated
        mgr.promote(repo, subject_name="relu", mr_id="abc-123", new_state="curated")
    """

    def __init__(
        self,
        min_quality: QualityLevel = QualityLevel.CHECKED,
        deduplicator: Optional[MRDeduplicator] = None,
        allow_duplicates: bool = False,
    ):
        """
        Args:
            min_quality:      入库最低质量要求（默认 CHECKED）
            deduplicator:     重复检测器（默认使用 MRDeduplicator）
            allow_duplicates: 是否允许重复入库（默认 False）
        """
        self.min_quality = min_quality
        self.deduplicator = deduplicator or MRDeduplicator()
        self.allow_duplicates = allow_duplicates

    # ── 入库检查 ──────────────────────────────────────────────────────────────

    def admit_check(self, mr: "MetamorphicRelation") -> AdmitResult:
        """检查单条 MR 是否满足入库条件。"""
        ql = QualityLevel.from_lifecycle(mr.lifecycle_state)

        # 规则 1：已退役 → 拒绝
        if ql == QualityLevel.RETIRED:
            return AdmitResult(mr.id, False, "已退役 (lifecycle_state=retired)")

        # 规则 2：质量不达标 → 拒绝
        if ql < self.min_quality:
            return AdmitResult(
                mr.id,
                False,
                f"质量不足: {ql.label} < {self.min_quality.label}",
            )

        # 规则 3：oracle_expr 为空 → 拒绝
        if not mr.oracle_expr or not mr.oracle_expr.strip():
            return AdmitResult(mr.id, False, "oracle_expr 为空")

        # 规则 4：transform_code 为空 → 警告但允许（模型层/应用层可能有合理情况）
        # （不拒绝，仅记录）

        return AdmitResult(mr.id, True, "通过")

    def admit_batch(
        self,
        mrs: List["MetamorphicRelation"],
        repo: Optional["MRRepository"] = None,
        subject_name: Optional[str] = None,
    ) -> GovernanceReport:
        """批量入库检查，可选地写入 repository。

        Args:
            mrs:          待入库 MR 列表
            repo:         可选，若提供则将通过的 MR 保存到 repo
            subject_name: 主体名称（repo.save 所需）

        Returns:
            GovernanceReport 包含入库/拒绝/重复信息
        """
        report = GovernanceReport()

        # 1. 基础质量过滤
        passed: List["MetamorphicRelation"] = []
        for mr in mrs:
            result = self.admit_check(mr)
            if result.admitted:
                passed.append(mr)
            else:
                report.rejected.append(result)
                logger.debug(f"🔒 [GOV] Reject MR {mr.id}: {result.reason}")

        # 2. 重复检测
        if not self.allow_duplicates and passed:
            unique, dup_groups = self.deduplicator.filter_unique(passed)
            report.duplicate_groups = dup_groups
            if dup_groups:
                for g in dup_groups:
                    logger.info(
                        f"🔍 [GOV] 重复组: canonical={g.canonical_id}, "
                        f"retire={g.duplicate_ids}, reason={g.reason}"
                    )
            passed = unique

        report.admitted = [m.id for m in passed]

        # 3. 可选持久化
        if repo is not None and subject_name and passed:
            repo.save(subject_name, passed)
            logger.info(
                f"📦 [GOV] 已写入 {len(passed)} 条 MR → [{repo.layer}] '{subject_name}'"
            )

        return report

    # ── 退役 ──────────────────────────────────────────────────────────────────

    def retire(
        self,
        repo: "MRRepository",
        subject_name: str,
        mr_id: str,
    ) -> bool:
        """将 MR 标记为 retired，不删除记录。"""
        success = repo.retire(subject_name, mr_id)
        if success:
            logger.info(
                f"🗂 [GOV] Retired MR {mr_id} in [{repo.layer}] '{subject_name}'"
            )
        else:
            logger.warning(
                f"🗂 [GOV] MR {mr_id} not found in [{repo.layer}] '{subject_name}'"
            )
        return success

    def retire_duplicates(
        self,
        repo: "MRRepository",
        subject_name: str,
    ) -> int:
        """自动检测并退役重复 MR，保留质量最高的。

        Returns:
            退役的 MR 数量
        """
        mrs = repo.load(subject_name)
        _, dup_groups = self.deduplicator.filter_unique(mrs)
        count = 0
        for g in dup_groups:
            for dup_id in g.duplicate_ids:
                if repo.retire(subject_name, dup_id):
                    count += 1
        if count:
            logger.info(
                f"🗂 [GOV] Auto-retired {count} duplicate MR(s) in [{repo.layer}] '{subject_name}'"
            )
        return count

    # ── 质量晋升 ──────────────────────────────────────────────────────────────

    def promote(
        self,
        repo: "MRRepository",
        subject_name: str,
        mr_id: str,
        new_state: str = "curated",
    ) -> bool:
        """将 MR 晋升到更高的 lifecycle_state。"""
        success = repo.update_lifecycle(subject_name, mr_id, new_state)
        if success:
            logger.info(
                f"⬆️ [GOV] Promoted MR {mr_id} → {new_state} in [{repo.layer}] '{subject_name}'"
            )
        return success
