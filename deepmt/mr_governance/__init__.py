"""MR 治理模块：统一质量保障与知识库治理（Phase K）。

模块组成：
    quality     — QualityLevel 枚举与质量评分工具
    provenance  — ProvenanceInfo 来源信息构建
    deduplicator — MRDeduplicator 重复检测
    governance  — MRGovernanceManager 统一入库/淘汰规则
"""

from deepmt.mr_governance.quality import QualityLevel, quality_level_from_mr
from deepmt.mr_governance.provenance import ProvenanceInfo, build_provenance
from deepmt.mr_governance.governance import MRGovernanceManager

__all__ = [
    "QualityLevel",
    "quality_level_from_mr",
    "ProvenanceInfo",
    "build_provenance",
    "MRGovernanceManager",
]
