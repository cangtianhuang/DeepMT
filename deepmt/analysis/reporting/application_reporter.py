"""应用层 MR 结果展示与样例报告生成器（J7）。

功能：
  - 汇总应用层 MR 数量与质量统计（按状态、来源、类别分组）
  - 输出典型通过/失败案例的结构化样例
  - 为实验统计与论文展示提供数据接口
  - 支持文本格式报告输出（用于 CLI 展示或写入文件）

用法::

    from deepmt.analysis.reporting.application_reporter import ApplicationReporter
    from deepmt.analysis.verification.semantic_mr_validator import SemanticValidationResult

    reporter = ApplicationReporter()
    report = reporter.generate(mrs, validation_results, scenario_name="ImageClassification")
    print(reporter.format_text(report))
    data = reporter.to_dict(report)   # 供仪表盘或 JSON 导出使用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepmt.analysis.verification.semantic_mr_validator import SemanticValidationResult
from deepmt.ir import MetamorphicRelation


# ── 报告数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class ApplicationMRReport:
    """应用层 MR 质量报告。

    Attributes:
        scenario_name:      场景名称
        total_mrs:          MR 总数
        by_status:          按状态分组的数量 {status: count}
        by_source:          按来源分组的数量 {source: count}
        by_category:        按类别分组的数量 {category: count}
        pass_rate:          验证通过率（passed + approved）/ total
        example_passed:     通过验证的典型 MR 样例列表（含验证结果）
        example_failed:     未通过验证的典型 MR 样例列表（含失败原因）
        example_review:     待复核的典型 MR 样例列表
        notes:              额外说明文字
    """

    scenario_name: str
    total_mrs: int
    by_status: Dict[str, int]
    by_source: Dict[str, int]
    by_category: Dict[str, int]
    pass_rate: float
    example_passed: List[Dict[str, Any]] = field(default_factory=list)
    example_failed: List[Dict[str, Any]] = field(default_factory=list)
    example_review: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


# ── 报告生成器 ─────────────────────────────────────────────────────────────────


class ApplicationReporter:
    """应用层 MR 报告生成器。

    将 MR 列表与验证结果汇总为结构化报告，支持文本格式化输出和 dict 导出。
    """

    def generate(
        self,
        mrs: List[MetamorphicRelation],
        validation_results: List[SemanticValidationResult],
        scenario_name: str = "Unknown",
        max_examples: int = 3,
    ) -> ApplicationMRReport:
        """生成应用层 MR 报告。

        Args:
            mrs:                MR 列表
            validation_results: 与 mrs 一一对应的验证结果列表
            scenario_name:      场景名称（用于报告标题）
            max_examples:       每类最多展示的样例数

        Returns:
            ApplicationMRReport 对象
        """
        total = len(mrs)
        results_by_id = {r.mr_id: r for r in validation_results}

        # 统计
        by_status: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        passed_ids = set()
        failed_ids = set()
        review_ids = set()

        for mr in mrs:
            res = results_by_id.get(mr.id)
            status = res.status if res else mr.lifecycle_state
            by_status[status] = by_status.get(status, 0) + 1
            by_source[mr.source] = by_source.get(mr.source, 0) + 1
            by_category[mr.category] = by_category.get(mr.category, 0) + 1

            if status in ("passed", "approved"):
                passed_ids.add(mr.id)
            elif status in ("failed", "rejected"):
                failed_ids.add(mr.id)
            elif status == "needs_review":
                review_ids.add(mr.id)

        pass_rate = len(passed_ids) / total if total > 0 else 0.0

        # 收集样例
        example_passed = []
        example_failed = []
        example_review = []

        for mr in mrs:
            res = results_by_id.get(mr.id)
            entry = _mr_to_example(mr, res)
            if mr.id in passed_ids and len(example_passed) < max_examples:
                example_passed.append(entry)
            elif mr.id in failed_ids and len(example_failed) < max_examples:
                example_failed.append(entry)
            elif mr.id in review_ids and len(example_review) < max_examples:
                example_review.append(entry)

        notes = (
            f"本报告覆盖 {scenario_name} 场景，共 {total} 条应用层 MR。"
            f"通过率 {pass_rate:.0%}，"
            f"待复核 {by_status.get('needs_review', 0)} 条。"
        )

        return ApplicationMRReport(
            scenario_name=scenario_name,
            total_mrs=total,
            by_status=by_status,
            by_source=by_source,
            by_category=by_category,
            pass_rate=pass_rate,
            example_passed=example_passed,
            example_failed=example_failed,
            example_review=example_review,
            notes=notes,
        )

    def format_text(self, report: ApplicationMRReport) -> str:
        """将报告格式化为可读文本（供 CLI 展示或写入文件）。"""
        lines = [
            f"=== 应用层 MR 质量报告：{report.scenario_name} ===",
            "",
            f"MR 总数：{report.total_mrs}",
            f"通过率：{report.pass_rate:.1%}",
            "",
            "── 按验证状态 ──",
        ]
        for status, count in sorted(report.by_status.items()):
            lines.append(f"  {status:20s}: {count}")

        lines += ["", "── 按来源 ──"]
        for src, count in sorted(report.by_source.items()):
            lines.append(f"  {src:20s}: {count}")

        lines += ["", "── 按类别 ──"]
        for cat, count in sorted(report.by_category.items()):
            lines.append(f"  {cat:30s}: {count}")

        if report.example_passed:
            lines += ["", "── 通过验证的典型 MR ──"]
            for ex in report.example_passed:
                lines.append(f"  [PASSED] {ex['description']}")
                lines.append(f"           transform: {ex['transform_code'][:70]}")
                lines.append(f"           oracle:    {ex['oracle_expr']}")
                lines.append(f"           category:  {ex['category']}")
                lines.append("")

        if report.example_failed:
            lines += ["", "── 未通过验证的典型 MR ──"]
            for ex in report.example_failed:
                lines.append(f"  [FAILED] {ex['description']}")
                if ex.get("detail"):
                    lines.append(f"           原因: {ex['detail'][:80]}")
                lines.append("")

        if report.example_review:
            lines += ["", "── 待人工复核的典型 MR ──"]
            for ex in report.example_review:
                lines.append(f"  [REVIEW] {ex['description']}")
                if ex.get("detail"):
                    lines.append(f"           说明: {ex['detail'][:80]}")
                lines.append("")

        lines += ["", f"说明：{report.notes}"]
        return "\n".join(lines)

    def to_dict(self, report: ApplicationMRReport) -> Dict[str, Any]:
        """将报告转换为 dict（供仪表盘或 JSON 导出）。"""
        return {
            "scenario_name": report.scenario_name,
            "total_mrs": report.total_mrs,
            "pass_rate": report.pass_rate,
            "by_status": report.by_status,
            "by_source": report.by_source,
            "by_category": report.by_category,
            "example_passed": report.example_passed,
            "example_failed": report.example_failed,
            "example_review": report.example_review,
            "notes": report.notes,
        }


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _mr_to_example(
    mr: MetamorphicRelation,
    res: Optional[SemanticValidationResult],
) -> Dict[str, Any]:
    """将 MR + 验证结果转换为展示用 dict。"""
    entry: Dict[str, Any] = {
        "id": mr.id,
        "description": mr.description,
        "transform_code": mr.transform_code,
        "oracle_expr": mr.oracle_expr,
        "category": mr.category,
        "source": mr.source,
        "layer": mr.layer,
        "subject_name": mr.subject_name,
    }
    if res is not None:
        entry["status"] = res.status
        entry["passed_samples"] = res.passed_samples
        entry["total_samples"] = res.total_samples
        entry["detail"] = res.detail
        entry["review_note"] = res.review_note
    return entry
