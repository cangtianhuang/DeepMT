"""
Case Study 数据模板。

职责：
  为典型缺陷案例定义统一数据结构，支持：
  - 从 evidence pack（JSON）自动提取基础字段
  - 人工补充差异描述、根因分析等深度字段
  - 导出为 Markdown（适合写入论文或答辩材料）
  - 建立案例目录与索引（CaseStudyIndex）

存储路径：data/experiments/case_studies/<case_id>.json

用法::

    from deepmt.experiments.case_study import CaseStudy, CaseStudyIndex

    # 从 evidence pack 自动生成
    idx = CaseStudyIndex()
    case = idx.from_evidence("evidence_abc123")
    case.summary = "ReLU 在极小负数输入时输出不一致"
    case.root_cause = "框架内部 in-place 操作顺序差异"
    idx.save(case)

    # 导出 Markdown
    print(case.to_markdown())

    # 列出所有案例
    for c in idx.list_all():
        print(c.case_id, c.operator, c.status)
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger

_DEFAULT_CASE_DIR = Path("data/experiments/case_studies")


@dataclass
class CaseStudy:
    """
    单个缺陷案例的完整描述。

    字段分两类：
      - 自动提取（从 evidence pack 填充）：case_id, operator, framework, mr_id,
        mr_description, input_example, output_original, output_transformed,
        evidence_pack_path, created_at
      - 人工补充（深度分析字段）：summary, root_cause, defect_type, severity,
        affected_versions, reproduction_steps, notes, status
    """

    case_id: str
    """案例唯一 ID（UUID 短格式）。"""

    # ── 自动提取字段 ──────────────────────────────────────────────────────────
    operator: str = ""
    """涉及的算子/模型/应用名称。"""

    framework: str = ""
    """涉及的框架（如 pytorch）。"""

    framework_version: str = ""
    """框架版本（如 2.1.0）。"""

    mr_id: str = ""
    """触发该案例的 MR ID。"""

    mr_description: str = ""
    """MR 的人类可读描述。"""

    layer: str = ""
    """MR 所在层次：operator / model / application。"""

    input_example: Any = None
    """触发失败的典型输入（可序列化为 JSON 的任意类型）。"""

    output_original: Any = None
    """原始输出（蜕变前）。"""

    output_transformed: Any = None
    """变换后输出（蜕变后）。"""

    oracle_violation: str = ""
    """oracle 违反描述（如 'output_diff > tolerance'）。"""

    evidence_pack_path: str = ""
    """来源 evidence pack 文件路径（相对于项目根）。"""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """案例创建时间。"""

    # ── 人工补充字段 ──────────────────────────────────────────────────────────
    summary: str = ""
    """案例一句话摘要（适合论文表格行）。"""

    root_cause: str = ""
    """根本原因分析（自由文本）。"""

    defect_type: str = ""
    """缺陷类型（如 numerical_precision / incorrect_semantics / api_inconsistency）。"""

    severity: str = "unknown"
    """严重程度：critical / high / medium / low / unknown。"""

    affected_versions: List[str] = field(default_factory=list)
    """受影响的框架版本列表。"""

    reproduction_steps: str = ""
    """复现步骤（自由文本）。"""

    notes: str = ""
    """其他备注（如已提 issue、已修复版本等）。"""

    status: str = "draft"
    """案例状态：draft / confirmed / closed。"""

    # ── 序列化 ────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operator": self.operator,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "mr_id": self.mr_id,
            "mr_description": self.mr_description,
            "layer": self.layer,
            "input_example": self.input_example,
            "output_original": self.output_original,
            "output_transformed": self.output_transformed,
            "oracle_violation": self.oracle_violation,
            "evidence_pack_path": self.evidence_pack_path,
            "created_at": self.created_at,
            "summary": self.summary,
            "root_cause": self.root_cause,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "affected_versions": self.affected_versions,
            "reproduction_steps": self.reproduction_steps,
            "notes": self.notes,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaseStudy":
        return cls(
            case_id=d["case_id"],
            operator=d.get("operator", ""),
            framework=d.get("framework", ""),
            framework_version=d.get("framework_version", ""),
            mr_id=d.get("mr_id", ""),
            mr_description=d.get("mr_description", ""),
            layer=d.get("layer", ""),
            input_example=d.get("input_example"),
            output_original=d.get("output_original"),
            output_transformed=d.get("output_transformed"),
            oracle_violation=d.get("oracle_violation", ""),
            evidence_pack_path=d.get("evidence_pack_path", ""),
            created_at=d.get("created_at", datetime.now().isoformat()),
            summary=d.get("summary", ""),
            root_cause=d.get("root_cause", ""),
            defect_type=d.get("defect_type", ""),
            severity=d.get("severity", "unknown"),
            affected_versions=d.get("affected_versions", []),
            reproduction_steps=d.get("reproduction_steps", ""),
            notes=d.get("notes", ""),
            status=d.get("status", "draft"),
        )

    # ── Markdown 导出 ────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """生成适合论文或答辩材料的 Markdown 案例描述。"""
        lines = [
            f"## Case Study: {self.case_id}",
            "",
            f"**状态**: {self.status}  ",
            f"**严重程度**: {self.severity}  ",
            f"**创建时间**: {self.created_at[:19]}",
            "",
            "### 摘要",
            "",
            self.summary or "_（未填写）_",
            "",
            "### 基本信息",
            "",
            f"| 字段 | 值 |",
            f"|------|----|",
            f"| 算子/场景 | `{self.operator}` |",
            f"| 框架 | {self.framework} {self.framework_version} |",
            f"| 层次 | {self.layer} |",
            f"| MR ID | `{self.mr_id}` |",
            f"| MR 描述 | {self.mr_description} |",
            f"| 缺陷类型 | {self.defect_type or '_未分类_'} |",
            "",
        ]

        if self.input_example is not None:
            lines += [
                "### 触发输入",
                "",
                "```python",
                str(self.input_example),
                "```",
                "",
            ]

        if self.output_original is not None or self.output_transformed is not None:
            lines += [
                "### 输出对比",
                "",
                f"| | 值 |",
                f"|-|----|",
                f"| 原始输出 | `{self.output_original}` |",
                f"| 变换后输出 | `{self.output_transformed}` |",
                f"| Oracle 违反 | {self.oracle_violation} |",
                "",
            ]

        if self.root_cause:
            lines += [
                "### 根因分析",
                "",
                self.root_cause,
                "",
            ]

        if self.reproduction_steps:
            lines += [
                "### 复现步骤",
                "",
                self.reproduction_steps,
                "",
            ]

        if self.affected_versions:
            lines += [
                "### 受影响版本",
                "",
                ", ".join(self.affected_versions),
                "",
            ]

        if self.evidence_pack_path:
            lines += [
                "### 证据来源",
                "",
                f"`{self.evidence_pack_path}`",
                "",
            ]

        if self.notes:
            lines += [
                "### 备注",
                "",
                self.notes,
                "",
            ]

        return "\n".join(lines)


# ── CaseStudyIndex ─────────────────────────────────────────────────────────────

class CaseStudyIndex:
    """
    Case Study 目录与索引管理器。

    负责案例的创建、保存、加载、列举和 Markdown 导出。
    存储目录：data/experiments/case_studies/
    索引文件：data/experiments/case_studies/index.json
    """

    def __init__(self, case_dir: Optional[Path] = None):
        self._dir = Path(case_dir) if case_dir else _DEFAULT_CASE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_empty(self, operator: str = "", framework: str = "") -> CaseStudy:
        """创建空白案例（填充 case_id + 基础字段）。"""
        return CaseStudy(
            case_id=uuid.uuid4().hex[:10],
            operator=operator,
            framework=framework,
        )

    def from_evidence(self, evidence_id_or_path: str) -> CaseStudy:
        """
        从 evidence pack JSON 文件自动提取基础字段，返回预填充的 CaseStudy。

        Args:
            evidence_id_or_path: evidence pack 的文件名（不含路径）或完整路径
        """
        pack = self._load_evidence(evidence_id_or_path)
        case = CaseStudy(case_id=uuid.uuid4().hex[:10])

        if pack is None:
            logger.warning(f"[CaseStudy] 无法加载 evidence: {evidence_id_or_path}")
            case.evidence_pack_path = str(evidence_id_or_path)
            return case

        case.operator = pack.get("operator", "")
        case.framework = pack.get("framework", "")
        case.framework_version = pack.get("framework_version", "")
        case.mr_id = pack.get("mr_id", "")
        case.mr_description = pack.get("mr_description", "")
        case.layer = pack.get("layer", "operator")
        case.input_example = pack.get("input")
        case.output_original = pack.get("output_original")
        case.output_transformed = pack.get("output_transformed")
        case.oracle_violation = pack.get("oracle_violation", "")
        case.evidence_pack_path = str(evidence_id_or_path)
        logger.debug(f"[CaseStudy] 已从 evidence 提取: {evidence_id_or_path}")
        return case

    def save(self, case: CaseStudy) -> Path:
        """保存案例 JSON 并更新索引，返回文件路径。"""
        path = self._dir / f"{case.case_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        self._update_index(case)
        logger.debug(f"[CaseStudy] 已保存: {path}")
        return path

    def load(self, case_id: str) -> Optional[CaseStudy]:
        """按 case_id 加载案例，未找到返回 None。"""
        path = self._dir / f"{case_id}.json"
        if not path.exists():
            logger.warning(f"[CaseStudy] 未找到: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return CaseStudy.from_dict(json.load(f))

    def list_all(self, status: Optional[str] = None) -> List[CaseStudy]:
        """列举所有案例（按创建时间升序）。"""
        cases = []
        for p in sorted(self._dir.glob("*.json")):
            if p.name == "index.json":
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = CaseStudy.from_dict(json.load(f))
                if status is None or c.status == status:
                    cases.append(c)
            except Exception as e:
                logger.warning(f"[CaseStudy] 读取 {p.name} 失败: {e}")
        cases.sort(key=lambda c: c.created_at)
        return cases

    # ── Markdown 导出 ────────────────────────────────────────────────────────

    def export_markdown_catalog(
        self,
        output_path: Optional[Path] = None,
        status: Optional[str] = None,
    ) -> Path:
        """
        将所有案例导出为单个 Markdown 文件（含目录索引）。

        Args:
            output_path: 输出路径（默认 data/experiments/case_studies/catalog.md）
            status:      过滤状态（None=全部）
        """
        if output_path is None:
            output_path = self._dir / "catalog.md"

        cases = self.list_all(status=status)
        lines = [
            "# Case Study 案例目录",
            "",
            f"> 生成时间：{datetime.now().isoformat()[:19]}  ",
            f"> 案例总数：{len(cases)}",
            "",
            "## 目录",
            "",
        ]

        # 目录索引
        for c in cases:
            status_icon = {"confirmed": "✅", "draft": "📝", "closed": "🔒"}.get(
                c.status, "❓"
            )
            lines.append(
                f"- {status_icon} [{c.case_id}] **{c.operator}** "
                f"({c.framework}) — {c.summary or '_(摘要未填写)_'}"
            )
        lines.append("")

        # 各案例详情
        for c in cases:
            lines.append(c.to_markdown())
            lines.append("---")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[CaseStudy] Markdown 目录已写入: {output_path}")
        return output_path

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _load_evidence(self, evidence_id_or_path: str) -> Optional[Dict]:
        """尝试从多个路径加载 evidence pack。"""
        candidates = [
            Path(evidence_id_or_path),
            Path("data/results/evidence") / evidence_id_or_path,
            Path("data/results/evidence") / f"{evidence_id_or_path}.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"[CaseStudy] 读取 {p} 失败: {e}")
        return None

    def _update_index(self, case: CaseStudy) -> None:
        """更新索引文件（写入简短元信息）。"""
        try:
            index: Dict[str, Any] = {}
            if self._index_path.exists():
                with open(self._index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            index[case.case_id] = {
                "operator": case.operator,
                "framework": case.framework,
                "status": case.status,
                "severity": case.severity,
                "summary": case.summary[:80] if case.summary else "",
                "created_at": case.created_at[:19],
            }
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[CaseStudy] 索引更新失败: {e}")

    def load_index(self) -> Dict[str, Any]:
        """加载索引文件，返回 {case_id: meta_dict}。"""
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
