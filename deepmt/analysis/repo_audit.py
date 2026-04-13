"""MR 仓库审计工具（K5）。

RepoAuditor 扫描全部三层 MR 仓库，输出：
    - 跨层统计摘要
    - 质量等级分布
    - 来源分布
    - 异常项清单（空 oracle、retired 占比高、无 provenance 等）
    - 重复关系检测报告

典型用法::

    auditor = RepoAuditor()
    report = auditor.run_audit()
    print(report.summary_text())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from deepmt.core.logger import logger
from deepmt.mr_governance.deduplicator import DuplicateGroup, MRDeduplicator
from deepmt.mr_governance.quality import QualityLevel
from deepmt.mr_generator.base.mr_repository import MRRepository

_LAYERS = ("operator", "model", "application")


@dataclass
class LayerStats:
    """单层统计快照。"""

    layer: str
    total: int = 0
    by_quality: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    retired: int = 0
    no_oracle: int = 0
    no_provenance: int = 0
    subjects: List[str] = field(default_factory=list)
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)


@dataclass
class AuditReport:
    """完整审计报告。"""

    layers: Dict[str, LayerStats] = field(default_factory=dict)
    repo_dir: str = "data/knowledge/mr_repository"

    @property
    def total_mrs(self) -> int:
        return sum(s.total for s in self.layers.values())

    @property
    def total_retired(self) -> int:
        return sum(s.retired for s in self.layers.values())

    @property
    def total_no_oracle(self) -> int:
        return sum(s.no_oracle for s in self.layers.values())

    @property
    def total_duplicate_groups(self) -> int:
        return sum(len(s.duplicate_groups) for s in self.layers.values())

    def quality_distribution(self) -> Dict[str, int]:
        """跨层合并的质量等级分布。"""
        from collections import Counter
        c: Counter = Counter()
        for s in self.layers.values():
            c.update(s.by_quality)
        return dict(c)

    def source_distribution(self) -> Dict[str, int]:
        """跨层合并的来源分布。"""
        from collections import Counter
        c: Counter = Counter()
        for s in self.layers.values():
            c.update(s.by_source)
        return dict(c)

    def anomalies(self) -> List[str]:
        """返回异常项描述列表（用于告警）。"""
        items = []
        for layer, stats in self.layers.items():
            if stats.total == 0:
                continue
            retired_ratio = stats.retired / stats.total
            if retired_ratio > 0.3:
                items.append(
                    f"[{layer}] 退役比例过高: {retired_ratio:.0%} "
                    f"({stats.retired}/{stats.total})"
                )
            if stats.no_oracle > 0:
                items.append(
                    f"[{layer}] {stats.no_oracle} 条 MR 的 oracle_expr 为空"
                )
            if stats.no_provenance > stats.total // 2:
                items.append(
                    f"[{layer}] 超过一半 MR 缺少 provenance 信息 "
                    f"({stats.no_provenance}/{stats.total})"
                )
            if stats.duplicate_groups:
                total_dups = sum(len(g.duplicate_ids) for g in stats.duplicate_groups)
                items.append(
                    f"[{layer}] 检测到 {len(stats.duplicate_groups)} 组重复，"
                    f"共 {total_dups} 条建议退役"
                )
        return items

    def pending_review_list(self, min_quality: QualityLevel = QualityLevel.PROVEN) -> List[Dict]:
        """导出待复核列表（质量低于 min_quality 但非 retired 的 MR）。"""
        result = []
        for layer, stats in self.layers.items():
            for qname, count in stats.by_quality.items():
                ql = QualityLevel.from_lifecycle(
                    {"candidate": "pending", "checked": "checked",
                     "proven": "proven", "curated": "curated",
                     "retired": "retired"}.get(qname, "pending")
                )
                if ql < min_quality and ql != QualityLevel.RETIRED and count > 0:
                    result.append({
                        "layer": layer,
                        "quality": qname,
                        "count": count,
                        "action": f"升级到 {min_quality.label} 或退役",
                    })
        return result

    def summary_text(self) -> str:
        """输出格式化的摘要文本（用于 CLI 展示）。"""
        lines = []
        lines.append("=" * 60)
        lines.append("MR 知识库审计报告")
        lines.append("=" * 60)
        lines.append(f"  仓库根目录: {self.repo_dir}")
        lines.append(f"  总 MR 数:   {self.total_mrs}")
        lines.append(f"  已退役:     {self.total_retired}")
        lines.append(f"  重复组:     {self.total_duplicate_groups}")
        lines.append("")

        # 质量分布
        qd = self.quality_distribution()
        if qd:
            lines.append("  质量等级分布:")
            for level in ["curated", "proven", "checked", "candidate", "retired"]:
                cnt = qd.get(level, 0)
                if cnt:
                    lines.append(f"    {level:<12} {cnt:>5}")
            lines.append("")

        # 来源分布
        sd = self.source_distribution()
        if sd:
            lines.append("  来源分布:")
            for src, cnt in sorted(sd.items(), key=lambda x: -x[1]):
                lines.append(f"    {src:<16} {cnt:>5}")
            lines.append("")

        # 各层摘要
        lines.append("  各层摘要:")
        lines.append(f"  {'层次':<16} {'总数':>6} {'退役':>6} {'无Oracle':>9} {'重复组':>8}")
        lines.append("  " + "─" * 50)
        for layer in _LAYERS:
            stats = self.layers.get(layer)
            if stats is None:
                continue
            lines.append(
                f"  {layer:<16} {stats.total:>6} {stats.retired:>6} "
                f"{stats.no_oracle:>9} {len(stats.duplicate_groups):>8}"
            )
        lines.append("")

        # 异常告警
        anomalies = self.anomalies()
        if anomalies:
            lines.append("  ⚠ 异常项:")
            for a in anomalies:
                lines.append(f"    • {a}")
            lines.append("")
        else:
            lines.append("  ✓ 未发现异常项")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class RepoAuditor:
    """MR 仓库审计器。

    用法::

        auditor = RepoAuditor()
        report = auditor.run_audit()
        print(report.summary_text())
    """

    def __init__(
        self,
        repo_dir: str = "data/knowledge/mr_repository",
        deduplicator: Optional[MRDeduplicator] = None,
    ):
        self.repo_dir = repo_dir
        self.deduplicator = deduplicator or MRDeduplicator()

    def _audit_layer(self, layer: str) -> LayerStats:
        """审计单层仓库，返回该层的统计信息。"""
        from collections import Counter
        repo = MRRepository(layer=layer, repo_dir=self.repo_dir)
        subjects = repo.list_subjects()

        stats = LayerStats(layer=layer, subjects=subjects)
        q_counter: Counter = Counter()
        s_counter: Counter = Counter()
        all_mrs = []

        for subject in subjects:
            mrs = repo.load(subject)
            all_mrs.extend(mrs)
            for mr in mrs:
                stats.total += 1
                q_counter[mr.quality_level] += 1
                s_counter[mr.source or "unknown"] += 1
                if mr.lifecycle_state == "retired":
                    stats.retired += 1
                if not mr.oracle_expr or not mr.oracle_expr.strip():
                    stats.no_oracle += 1
                if not mr.provenance:
                    stats.no_provenance += 1

        stats.by_quality = dict(q_counter)
        stats.by_source = dict(s_counter)

        # 重复检测（跨所有主体）
        if all_mrs:
            _, dup_groups = self.deduplicator.filter_unique(all_mrs)
            stats.duplicate_groups = dup_groups

        logger.info(
            f"🔍 [AUDIT] [{layer}] {stats.total} MRs across {len(subjects)} subjects"
        )
        return stats

    def run_audit(self, layers: Optional[List[str]] = None) -> AuditReport:
        """执行完整审计，返回 AuditReport。

        Args:
            layers: 要审计的层列表，默认全部三层
        """
        layers = layers or list(_LAYERS)
        report = AuditReport(repo_dir=self.repo_dir)

        for layer in layers:
            stats = self._audit_layer(layer)
            report.layers[layer] = stats

        logger.info(
            f"🔍 [AUDIT] Complete — total={report.total_mrs}, "
            f"retired={report.total_retired}, "
            f"dup_groups={report.total_duplicate_groups}"
        )
        return report

    def export_pending_review(
        self,
        min_quality: QualityLevel = QualityLevel.PROVEN,
        output_format: str = "text",
    ) -> str:
        """导出待复核清单（质量低于 min_quality 的 MR）。

        Args:
            min_quality:   最低质量门槛
            output_format: "text" 或 "json"
        """
        report = self.run_audit()
        items = report.pending_review_list(min_quality)

        if output_format == "json":
            import json
            return json.dumps(items, ensure_ascii=False, indent=2)

        if not items:
            return "无待复核项目。"
        lines = ["待复核 MR 列表:"]
        for item in items:
            lines.append(
                f"  [{item['layer']}] {item['quality']} × {item['count']} → {item['action']}"
            )
        return "\n".join(lines)
