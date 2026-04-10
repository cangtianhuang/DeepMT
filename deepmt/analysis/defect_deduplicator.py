"""
缺陷线索去重器：将大量重复的测试失败压缩为可人工复核的"缺陷线索集"。

设计目的：
  - 避免同一问题在多次批量测试中产生大量重复失败记录淹没分析者
  - 为论文 D2 任务提供"可管理的缺陷线索数量"实证
  - 每条缺陷线索代表一个独立的失败模式（算子 × MR × 错误类型）

去重维度：
  (operator, mr_id, error_bucket) 三元组构成"缺陷签名"
  - error_bucket: 从 detail 字段提取的错误类别（NUMERICAL_DEVIATION / SHAPE_MISMATCH /
    EXCEPTION / TRANSFORM_ERROR / OTHER）

数据来源：
  EvidenceCollector — 读取已保存的证据包（.json）

数据流：
  EvidenceCollector.list_all()
    → 按签名聚类 → DefectLead（每个签名保留最具代表性的1条）
    → deepmt test dedup（CLI 展示）
"""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger


# ── 错误类型桶 ────────────────────────────────────────────────────────────────

_ERROR_BUCKET_KEYWORDS = {
    "NUMERICAL_DEVIATION": "numerical",
    "SHAPE_MISMATCH": "shape",
    "EXCEPTION": "exception",
    "TRANSFORM_ERROR": "transform",
    "TYPE_ERROR": "type",
}


def _extract_error_bucket(detail: str) -> str:
    """
    从 detail 字符串提取标准化错误类别。

    detail 通常为 "NUMERICAL_DEVIATION: max_abs=0.5" 或 "EXCEPTION: ..." 格式。
    匹配失败时返回 "other"。
    """
    if not detail:
        return "other"
    detail_upper = detail.upper()
    for keyword, bucket in _ERROR_BUCKET_KEYWORDS.items():
        if detail_upper.startswith(keyword):
            return bucket
    # 尝试从冒号前取首词
    first_word = detail.split(":")[0].strip().upper()
    for keyword, bucket in _ERROR_BUCKET_KEYWORDS.items():
        if keyword in first_word:
            return bucket
    return "other"


def _make_lead_id(operator: str, mr_id: str, error_bucket: str, framework: str) -> str:
    """生成缺陷线索的唯一 ID（12 位十六进制）。"""
    sig = f"{operator}|{mr_id}|{error_bucket}|{framework}"
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


# ── DefectLead ─────────────────────────────────────────────────────────────────


@dataclass
class DefectLead:
    """
    单条去重后的缺陷线索。

    代表一类失败模式（算子 × MR × 错误类型 × 框架），
    聚合了多次相同模式的失败记录。
    """

    lead_id: str               # 缺陷签名哈希（12 位）
    operator: str              # 失败算子名称
    framework: str             # 目标框架
    mr_id: str                 # 触发失败的 MR ID
    mr_description: str        # MR 描述
    error_bucket: str          # 错误类别（numerical/shape/exception/...）
    occurrence_count: int      # 相同模式的出现次数
    first_seen: str            # 最早时间戳
    last_seen: str             # 最晚时间戳
    detail_sample: str         # 最近一次失败的 detail 字段（供快速定位）
    representative_evidence_id: Optional[str]  # 最近失败对应的证据包 ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── DefectDeduplicator ────────────────────────────────────────────────────────


class DefectDeduplicator:
    """
    缺陷线索去重器。

    从 EvidenceCollector 读取证据包，按缺陷签名聚类，
    输出每类失败模式的代表性缺陷线索。

    用法示例：
        dedup = DefectDeduplicator()
        leads = dedup.deduplicate()
        print(f"发现 {len(leads)} 条独立缺陷线索")
        for lead in leads:
            print(f"  {lead.operator} / {lead.mr_description}: {lead.occurrence_count} 次")
    """

    def __init__(self, evidence_dir: Optional[str] = None):
        from deepmt.analysis.evidence_collector import EvidenceCollector
        self._collector = EvidenceCollector(evidence_dir=evidence_dir)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def deduplicate(
        self,
        operator: Optional[str] = None,
        framework: Optional[str] = None,
        limit: int = 0,
    ) -> List[DefectLead]:
        """
        读取证据包并返回去重后的缺陷线索列表，按出现次数降序排列。

        Args:
            operator:  按算子名称过滤；None 表示全部
            framework: 按框架过滤；None 表示全部
            limit:     最多返回条数（0 = 不限）

        Returns:
            DefectLead 列表，按 occurrence_count 降序
        """
        packs = self._collector.list_all(operator=operator, framework=framework, limit=0)

        if not packs:
            return []

        # 按签名聚类
        clusters: Dict[str, List] = {}
        for pack in packs:
            bucket = _extract_error_bucket(pack.detail)
            lead_id = _make_lead_id(pack.operator, pack.mr_id, bucket, pack.framework)
            if lead_id not in clusters:
                clusters[lead_id] = []
            clusters[lead_id].append(pack)

        # 每个聚类生成一条 DefectLead
        leads = []
        for lead_id, cluster_packs in clusters.items():
            # 证据包已按时间倒序排列（list_all 保证），第一个最新
            newest = cluster_packs[0]
            oldest = cluster_packs[-1]
            bucket = _extract_error_bucket(newest.detail)

            lead = DefectLead(
                lead_id=lead_id,
                operator=newest.operator,
                framework=newest.framework,
                mr_id=newest.mr_id,
                mr_description=newest.mr_description,
                error_bucket=bucket,
                occurrence_count=len(cluster_packs),
                first_seen=oldest.timestamp,
                last_seen=newest.timestamp,
                detail_sample=newest.detail[:120],
                representative_evidence_id=newest.evidence_id,
            )
            leads.append(lead)

        # 按出现次数降序，相同次数则按最后出现时间降序
        leads.sort(key=lambda l: (-l.occurrence_count, l.last_seen), reverse=False)
        leads.sort(key=lambda l: l.occurrence_count, reverse=True)

        if limit > 0:
            leads = leads[:limit]

        logger.info(
            f"[DEDUP] {len(packs)} evidence packs → {len(leads)} unique defect leads"
            + (f" (showing top {limit})" if limit > 0 else "")
        )
        return leads

    def format_text(self, leads: List[DefectLead]) -> str:
        """格式化为可读文本摘要。"""
        if not leads:
            return "未发现缺陷线索（证据包为空或过滤条件无匹配）。\n提示：运行 deepmt test batch --collect-evidence 收集证据包。"

        lines = [
            f"\n缺陷线索摘要（共 {len(leads)} 条独立模式）",
            "─" * 72,
        ]

        for i, lead in enumerate(leads, 1):
            lines.append(
                f"  [{i:02d}] {lead.lead_id}  {lead.error_bucket.upper()}"
                f"  ×{lead.occurrence_count}"
            )
            lines.append(f"       算子: {lead.operator}  [{lead.framework}]")
            lines.append(f"       MR:   {lead.mr_description[:65]}")
            lines.append(f"       详情: {lead.detail_sample[:65]}")
            if lead.representative_evidence_id:
                lines.append(f"       证据: {lead.representative_evidence_id}")
            lines.append(
                f"       时间: {lead.first_seen[:16]} → {lead.last_seen[:16]}"
            )
            lines.append("")

        lines.append("─" * 72)
        lines.append(
            f"总计: {len(leads)} 条缺陷线索  "
            f"| 运行 'deepmt test evidence show <id>' 查看完整证据包"
        )
        return "\n".join(lines)
