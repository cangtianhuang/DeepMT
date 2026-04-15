"""
缺陷案例构建器：将去重后的缺陷线索或证据包自动转换为完整、可复现的案例包。

设计目的：
  - 连接"缺陷发现"（DefectLead / EvidencePack）与"案例沉淀"（CaseStudy + 案例目录）
  - 自动提取证据包中的结构化信息，填充 CaseStudy 基础字段
  - 生成独立可运行的案例目录，包含复现脚本、摘要和原始证据
  - 支持批量构建（高优先级线索 top-N 自动打包）

案例包目录结构：
  <output_dir>/<case_id>/
    reproduce.py       — 可直接运行的 Python 复现脚本
    case_summary.md    — Markdown 格式案例摘要
    evidence.json      — 原始证据包（完整 JSON 副本）
    metadata.json      — 案例元数据（case_id、创建时间、状态等）

数据流：
  DefectLead
    → load representative EvidencePack
    → CaseStudyIndex.from_evidence()   （自动填充基础字段）
    → DefectCaseBuilder._enrich()      （补充 defect_type / severity 推断）
    → CaseStudyIndex.save()            （持久化到 index）
    → build_case_package()             （输出案例目录）
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from deepmt.analysis.qa.defect_deduplicator import DefectLead
from deepmt.analysis.reporting.evidence_collector import EvidenceCollector
from deepmt.core.logger import logger
from deepmt.experiments.case_study import CaseStudy, CaseStudyIndex

# 案例包默认输出目录
_DEFAULT_CASES_DIR = Path("data/cases/real_defects")

# ── 错误类型 → defect_type 映射 ────────────────────────────────────────────────

_BUCKET_TO_DEFECT_TYPE = {
    "numerical": "numerical_precision",
    "shape": "shape_mismatch",
    "exception": "runtime_exception",
    "transform": "transform_error",
    "type": "type_error",
    "other": "unknown",
}

# ── 优先级评分规则 ─────────────────────────────────────────────────────────────

def _infer_severity(lead: DefectLead) -> str:
    """
    根据缺陷线索特征推断严重程度。

    规则（由高到低）：
      - 出现次数 >= 10 且 numerical/exception → high
      - 出现次数 >= 5 或 exception → medium
      - 其余 → low
    """
    bucket = lead.error_bucket
    count = lead.occurrence_count
    if bucket == "exception" or (count >= 10 and bucket in ("numerical", "shape")):
        return "high"
    if count >= 5 or bucket == "exception":
        return "medium"
    return "low"


# ── DefectCaseBuilder ──────────────────────────────────────────────────────────


class DefectCaseBuilder:
    """
    缺陷案例构建器。

    用法示例::

        builder = DefectCaseBuilder()

        # 从缺陷线索构建
        from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator
        leads = DefectDeduplicator().deduplicate()
        case = builder.build_from_lead(leads[0])
        pkg_dir = builder.build_case_package(case)
        print(f"案例包已生成: {pkg_dir}")

        # 直接从证据包 ID 构建
        case = builder.build_from_evidence("bee67689-15b")
        pkg_dir = builder.build_case_package(case)

        # 批量构建前 N 个高优先级线索
        dirs = builder.build_top_leads(leads, top_n=3)
    """

    def __init__(
        self,
        case_index: Optional[CaseStudyIndex] = None,
        evidence_collector: Optional[EvidenceCollector] = None,
        cases_dir: Optional[Path] = None,
    ):
        self._index = case_index or CaseStudyIndex()
        self._collector = evidence_collector or EvidenceCollector()
        self._cases_dir = Path(cases_dir) if cases_dir else _DEFAULT_CASES_DIR
        self._cases_dir.mkdir(parents=True, exist_ok=True)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def build_from_lead(self, lead: DefectLead) -> CaseStudy:
        """
        从 DefectLead 构建 CaseStudy。

        流程：
          1. 加载代表性证据包（representative_evidence_id）
          2. 若有证据包，从中自动提取基础字段
          3. 否则，用 lead 元信息创建空白 CaseStudy
          4. 推断 defect_type 和 severity
          5. 保存到 CaseStudyIndex

        Args:
            lead: 去重后的缺陷线索

        Returns:
            已保存的 CaseStudy 对象
        """
        if lead.representative_evidence_id:
            case = self._index.from_evidence(lead.representative_evidence_id)
        else:
            case = self._index.create_empty(
                operator=lead.operator,
                framework=lead.framework,
            )

        # 补充来自 lead 的信息
        case.mr_id = case.mr_id or lead.mr_id
        case.mr_description = case.mr_description or lead.mr_description
        case.framework = case.framework or lead.framework
        case.operator = case.operator or lead.operator

        # 推断字段
        self._enrich(case, lead)

        self._index.save(case)
        logger.info(f"[CASE] Built case {case.case_id} from lead {lead.lead_id}")
        return case

    def build_from_evidence(self, evidence_id: str) -> CaseStudy:
        """
        直接从证据包 ID 构建 CaseStudy（不经过 DefectLead）。

        适用于手工挑选特定证据包进行案例化的场景。

        Args:
            evidence_id: 证据包 ID（不含 .json 扩展名）

        Returns:
            已保存的 CaseStudy 对象
        """
        pack = self._collector.load(evidence_id)
        if pack is not None:
            # 使用证据包数据构建 CaseStudy
            case = self._index.create_empty(
                operator=pack.operator,
                framework=pack.framework,
            )
            case.framework_version = pack.framework_version
            case.mr_id = pack.mr_id
            case.mr_description = pack.mr_description
            case.layer = "operator"
            case.oracle_violation = pack.detail
            case.evidence_pack_path = evidence_id
        else:
            # 证据包不存在，创建空白案例
            case = self._index.create_empty()
            case.evidence_pack_path = evidence_id
            logger.warning(f"[CASE] Evidence pack not found: {evidence_id}")

        self._enrich_from_evidence_id(case, evidence_id)
        self._index.save(case)
        logger.info(f"[CASE] Built case {case.case_id} from evidence {evidence_id}")
        return case

    def build_case_package(
        self,
        case: CaseStudy,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        将 CaseStudy 输出为可独立运行的案例目录。

        目录结构::

            <output_dir>/<case_id>/
              reproduce.py      — 可直接运行的复现脚本
              case_summary.md   — Markdown 摘要
              evidence.json     — 原始证据包副本
              metadata.json     — 案例元数据

        Args:
            case:       已构建的 CaseStudy 对象
            output_dir: 输出根目录（默认 data/cases/real_defects/）

        Returns:
            案例目录的 Path 对象
        """
        root = Path(output_dir) if output_dir else self._cases_dir
        pkg_dir = root / case.case_id
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # 1. reproduce.py
        reproduce_script = self._get_reproduce_script(case)
        (pkg_dir / "reproduce.py").write_text(reproduce_script, encoding="utf-8")

        # 2. case_summary.md
        (pkg_dir / "case_summary.md").write_text(
            case.to_markdown(), encoding="utf-8"
        )

        # 3. evidence.json（复制原始证据包）
        self._copy_evidence(case, pkg_dir)

        # 4. metadata.json
        metadata = {
            "case_id": case.case_id,
            "operator": case.operator,
            "framework": case.framework,
            "framework_version": case.framework_version,
            "mr_id": case.mr_id,
            "defect_type": case.defect_type,
            "severity": case.severity,
            "status": case.status,
            "created_at": case.created_at,
            "package_generated_at": datetime.now().isoformat(),
            "evidence_pack_path": case.evidence_pack_path,
        }
        (pkg_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(f"[CASE] Package written to: {pkg_dir}")
        return pkg_dir

    def build_top_leads(
        self,
        leads: List[DefectLead],
        top_n: int = 5,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        批量构建前 top_n 个高优先级线索的案例包。

        Args:
            leads:      DefectLead 列表（已按优先级排序）
            top_n:      最多处理的线索数
            output_dir: 案例包输出根目录

        Returns:
            已生成的案例包目录列表
        """
        pkg_dirs: List[Path] = []
        for lead in leads[:top_n]:
            try:
                case = self.build_from_lead(lead)
                pkg_dir = self.build_case_package(case, output_dir)
                pkg_dirs.append(pkg_dir)
            except Exception as e:
                logger.error(f"[CASE] Failed to build case for lead {lead.lead_id}: {e}")
        logger.info(f"[CASE] Built {len(pkg_dirs)} case packages from top {top_n} leads")
        return pkg_dirs

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _enrich(self, case: CaseStudy, lead: DefectLead) -> None:
        """根据 DefectLead 信息推断并补充 CaseStudy 字段。"""
        if not case.defect_type:
            case.defect_type = _BUCKET_TO_DEFECT_TYPE.get(
                lead.error_bucket, "unknown"
            )
        if case.severity == "unknown":
            case.severity = _infer_severity(lead)
        if not case.notes:
            case.notes = (
                f"来自缺陷线索 {lead.lead_id}，"
                f"共出现 {lead.occurrence_count} 次，"
                f"时间范围：{lead.first_seen[:10]} ~ {lead.last_seen[:10]}"
            )

    def _enrich_from_evidence_id(self, case: CaseStudy, evidence_id: str) -> None:
        """从证据包原始 JSON 补充字段（不经过 DefectLead）。"""
        pack = self._collector.load(evidence_id)
        if pack is None:
            return
        if not case.defect_type:
            from deepmt.analysis.qa.defect_deduplicator import _extract_error_bucket
            bucket = _extract_error_bucket(pack.detail)
            case.defect_type = _BUCKET_TO_DEFECT_TYPE.get(bucket, "unknown")
        if case.severity == "unknown":
            # 无 occurrence_count 时默认 medium
            case.severity = "medium"
        if not case.oracle_violation:
            case.oracle_violation = pack.detail
        if not case.notes:
            case.notes = f"直接从证据包 {evidence_id} 构建"

    def _get_reproduce_script(self, case: CaseStudy) -> str:
        """
        生成复现脚本。

        若存在证据包，使用证据包中的元数据重新生成脚本（确保算子名称可执行）；
        若无证据包，生成包含已知信息的占位脚本。
        """
        from deepmt.analysis.reporting.evidence_collector import (
            _generate_cross_reproduce_script,
            _generate_reproduce_script,
        )

        if case.evidence_pack_path:
            pack = self._collector.load(Path(case.evidence_pack_path).stem)
            if pack:
                if getattr(pack, "kind", "") == "cross_framework_divergence":
                    return _generate_cross_reproduce_script(
                        operator_name=pack.operator,
                        framework1=pack.framework,
                        framework1_version=pack.framework_version,
                        framework2=pack.framework2 or "",
                        framework2_version=pack.framework2_version or "",
                        mr_description=pack.mr_description,
                        transform_code=pack.transform_code,
                        oracle_expr=pack.oracle_expr,
                        input_summary=pack.input_summary,
                        f1_output_summary={},
                        f2_output_summary=pack.f2_output_summary or {},
                        diff_type=pack.diff_type or "unknown",
                        numeric_diff=pack.actual_diff,
                    )
                return _generate_reproduce_script(
                    operator_name=pack.operator,
                    framework=pack.framework,
                    framework_version=pack.framework_version,
                    mr_description=pack.mr_description,
                    transform_code=pack.transform_code,
                    oracle_expr=pack.oracle_expr,
                    input_summary=pack.input_summary,
                    actual_diff=pack.actual_diff,
                    tolerance=pack.tolerance,
                )

        # 占位脚本（无证据包时）
        return (
            f'"""\n'
            f"DeepMT 案例复现脚本（手工构建）\n"
            f"案例: {case.case_id}\n"
            f"算子: {case.operator}\n"
            f"框架: {case.framework} {case.framework_version}\n"
            f"MR:  {case.mr_description}\n"
            f'"""\n\n'
            f"# 根因: {case.root_cause or '（待填写）'}\n"
            f"# 复现步骤:\n"
            f"# {case.reproduction_steps or '（待填写）'}\n\n"
            f"# TODO: 在此填写复现代码\n"
        )

    def _copy_evidence(self, case: CaseStudy, pkg_dir: Path) -> None:
        """将证据包 JSON 复制到案例目录。"""
        if not case.evidence_pack_path:
            return
        evidence_id = Path(case.evidence_pack_path).stem
        src = self._collector.evidence_dir / f"{evidence_id}.json"
        if src.exists():
            shutil.copy2(src, pkg_dir / "evidence.json")
        else:
            logger.warning(f"[CASE] Evidence file not found: {src}")
