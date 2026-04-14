"""deepmt.analysis — 验证、报告与证据分析

子包：
  verification/  — MR 验证核心（MRVerifier, MRPreChecker, ModelVerifier, RandomGenerator…）
  reporting/     — 测试报告、证据收集与变异测试
  qa/            — 跨框架一致性、缺陷去重与知识库审计
"""

from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.analysis.verification.mr_prechecker import MRPreChecker
from deepmt.analysis.verification.model_verifier import ModelVerifier
from deepmt.analysis.verification.random_generator import RandomGenerator
from deepmt.analysis.reporting.report_generator import ReportGenerator
from deepmt.analysis.reporting.evidence_collector import EvidenceCollector

__all__ = [
    "MRVerifier",
    "MRPreChecker",
    "ModelVerifier",
    "RandomGenerator",
    "ReportGenerator",
    "EvidenceCollector",
]
