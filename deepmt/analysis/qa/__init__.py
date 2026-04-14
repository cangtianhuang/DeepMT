"""deepmt.analysis.qa — 跨框架一致性、缺陷去重与知识库审计"""

from deepmt.analysis.qa.cross_framework_tester import CrossFrameworkTester, CrossSessionResult, DiffType, CrossConsistencyResult
from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator
from deepmt.analysis.qa.repo_audit import RepoAuditor

__all__ = [
    "CrossFrameworkTester",
    "CrossSessionResult",
    "DiffType",
    "CrossConsistencyResult",
    "DefectDeduplicator",
    "RepoAuditor",
]
