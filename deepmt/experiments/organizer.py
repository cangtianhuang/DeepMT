"""
实验数据组织器：将系统产出映射到论文 RQ1-RQ4。

设计目的：
  保证实验结果天然能对应到论文研究问题，避免后期手工拼凑。

RQ 映射：
  RQ1 — MR 自动生成质量：数量、验证率、分类分布（来源：MRRepository）
  RQ2 — 缺陷检测能力：通过率、失败分布、证据包数量（来源：ResultsManager + EvidenceCollector）
  RQ3 — 跨框架一致性：一致率、典型差异（来源：CrossFrameworkTester 持久化结果）
  RQ4 — 与基线对比：覆盖度、自动化程度、用例密度（来源：综合统计）

数据来源汇总：
  MRRepository → data/knowledge/mr_repository/operator/*.yaml（含 applicable_frameworks、verified 字段）
  ResultsManager → data/results/defects.db（test_results 表）
  EvidenceCollector → data/results/evidence/*.json
  CrossFrameworkTester → data/results/cross_framework/*.json

输出格式：
  dict（可 JSON 序列化）或终端可读文本
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger


class ExperimentOrganizer:
    """
    论文实验数据组织器。

    从现有数据源自动收集 RQ1-RQ4 所需的核心指标，
    生成结构化报告供论文写作和答辩直接使用。

    用法示例：
        org = ExperimentOrganizer()
        data = org.collect_all()
        print(org.format_text(data))

        # 仅收集 RQ2
        rq2 = org.collect_rq2()

    CLI 等价：
        deepmt test experiment
        deepmt test experiment --rq 2
        deepmt test experiment --json > experiment_data.json
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        evidence_dir: Optional[str] = None,
        cross_results_dir: Optional[str] = None,
        mr_repo: Optional[Any] = None,
    ):
        self._db_path = db_path
        self._evidence_dir = evidence_dir
        self._cross_results_dir = cross_results_dir
        self._mr_repo = mr_repo

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def collect_all(self) -> Dict[str, Any]:
        """
        收集 RQ1-RQ4 全部数据，返回结构化字典。

        Returns:
            {
              "generated_at": str,
              "rq1": {...},
              "rq2": {...},
              "rq3": {...},
              "rq4": {...},
            }
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "rq1": self.collect_rq1(),
            "rq2": self.collect_rq2(),
            "rq3": self.collect_rq3(),
            "rq4": self.collect_rq4(),
        }

    def collect_rq1(self) -> Dict[str, Any]:
        """
        RQ1：MR 自动生成质量。

        指标：
          total_mr_count         — 知识库中 MR 总数
          verified_mr_count      — 已验证 MR 数量
          verification_rate      — 验证率
          operators_with_mr      — 有至少一条 MR 的算子数
          avg_mr_per_operator    — 平均每算子 MR 数
          category_distribution  — 各 category 的 MR 数分布
          source_distribution    — MR 来源分布（llm / template / manual）
        """
        try:
            from deepmt.mr_generator.base.mr_repository import MRRepository
            repo = self._mr_repo or MRRepository()

            # get_statistics() 提供 total/verified 聚合（高效，一次遍历）
            stats = repo.get_statistics()
            total = stats.get("total_mrs", 0)
            verified = stats.get("verified_mrs", 0)
            n_ops = len(stats.get("by_operator", {}))

            # 收集 category / source 分布（需逐算子加载 MR）
            categories: Dict[str, int] = {}
            sources: Dict[str, int] = {}
            for op in repo.list_operators():
                for mr in repo.load(op):
                    cat = mr.category or "uncategorized"
                    categories[cat] = categories.get(cat, 0) + 1
                    src = mr.source or "unknown"
                    sources[src] = sources.get(src, 0) + 1

            return {
                "total_mr_count": total,
                "verified_mr_count": verified,
                "verification_rate": round(verified / total, 4) if total > 0 else 0.0,
                "operators_with_mr": n_ops,
                "avg_mr_per_operator": round(total / n_ops, 2) if n_ops > 0 else 0.0,
                "category_distribution": dict(sorted(categories.items(), key=lambda x: -x[1])),
                "source_distribution": dict(sorted(sources.items(), key=lambda x: -x[1])),
            }

        except Exception as e:
            logger.warning(f"[RQ1] 数据收集失败: {e}")
            return {"error": str(e)}

    def collect_rq2(self) -> Dict[str, Any]:
        """
        RQ2：缺陷检测能力。

        指标：
          total_test_cases       — 总测试用例数
          total_passed           — 通过数
          total_failed           — 失败数
          overall_pass_rate      — 整体通过率
          operators_tested       — 被测算子数
          operators_with_failure — 有失败的算子数
          failure_rate           — 失败率
          evidence_pack_count    — 证据包数量（可复现缺陷数）
          unique_defect_leads    — 去重后独立缺陷线索数
        """
        try:
            from deepmt.core.results_manager import ResultsManager
            rm = ResultsManager(db_path=self._db_path) if self._db_path else ResultsManager()
            summary = rm.get_summary()

            total_cases = sum(r.get("total_tests", 0) for r in summary)
            total_passed = sum(r.get("passed_tests", 0) for r in summary)
            total_failed = total_cases - total_passed
            ops_tested = len(summary)
            ops_with_failure = sum(1 for r in summary if r.get("failed_tests", 0) > 0)

        except Exception as e:
            logger.warning(f"[RQ2] ResultsManager 读取失败: {e}")
            total_cases = total_passed = total_failed = ops_tested = ops_with_failure = 0

        try:
            from deepmt.analysis.reporting.evidence_collector import EvidenceCollector
            ec = EvidenceCollector(evidence_dir=self._evidence_dir) if self._evidence_dir else EvidenceCollector()
            evidence_count = ec.count()
        except Exception as e:
            logger.warning(f"[RQ2] EvidenceCollector 读取失败: {e}")
            evidence_count = 0

        try:
            from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator
            dedup = DefectDeduplicator(evidence_dir=self._evidence_dir)
            unique_leads = len(dedup.deduplicate())
        except Exception as e:
            logger.warning(f"[RQ2] DefectDeduplicator 读取失败: {e}")
            unique_leads = 0

        return {
            "total_test_cases": total_cases,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_pass_rate": round(total_passed / total_cases, 4) if total_cases > 0 else 0.0,
            "operators_tested": ops_tested,
            "operators_with_failure": ops_with_failure,
            "failure_rate": round(total_failed / total_cases, 4) if total_cases > 0 else 0.0,
            "evidence_pack_count": evidence_count,
            "unique_defect_leads": unique_leads,
        }

    def collect_rq3(self) -> Dict[str, Any]:
        """
        RQ3：跨框架一致性。

        指标：
          cross_session_count       — 已完成的跨框架实验次数
          operators_compared        — 参与对比的算子数
          overall_consistency_rate  — 所有实验的平均一致率
          avg_output_max_diff       — 平均输出最大差值（体现实现差异的量级）
          inconsistent_mr_count     — 至少存在一个不一致样本的 MR 数量
          framework_pairs           — 参与对比的框架对
        """
        try:
            from deepmt.analysis.qa.cross_framework_tester import CrossFrameworkTester
            tester = CrossFrameworkTester(
                results_dir=self._cross_results_dir
            )
            sessions = tester.load_all()
        except Exception as e:
            logger.warning(f"[RQ3] 跨框架结果读取失败: {e}")
            sessions = []

        if not sessions:
            return {
                "cross_session_count": 0,
                "operators_compared": 0,
                "overall_consistency_rate": None,
                "avg_output_max_diff": None,
                "inconsistent_mr_count": 0,
                "framework_pairs": [],
                "note": "尚无跨框架实验记录。运行 deepmt test cross <operator> 生成。",
            }

        ops_compared = len({s.operator for s in sessions})
        fw_pairs = list({(s.framework1, s.framework2) for s in sessions})
        all_consistency = [s.overall_consistency_rate for s in sessions]
        all_diffs = [s.output_max_diff for s in sessions if s.output_max_diff == s.output_max_diff]
        total_incon_mr = sum(s.inconsistent_mr_count for s in sessions)

        return {
            "cross_session_count": len(sessions),
            "operators_compared": ops_compared,
            "overall_consistency_rate": round(sum(all_consistency) / len(all_consistency), 4),
            "avg_output_max_diff": round(sum(all_diffs) / len(all_diffs), 6) if all_diffs else None,
            "inconsistent_mr_count": total_incon_mr,
            "framework_pairs": [f"{f1} vs {f2}" for f1, f2 in fw_pairs],
            "sessions": [
                {
                    "operator": s.operator,
                    "framework_pair": f"{s.framework1} vs {s.framework2}",
                    "consistency_rate": round(s.overall_consistency_rate, 4),
                    "output_max_diff": s.output_max_diff,
                    "inconsistent_mr": s.inconsistent_mr_count,
                }
                for s in sessions
            ],
        }

    def collect_rq4(self) -> Dict[str, Any]:
        """
        RQ4：与基线方法对比（覆盖度、自动化程度、用例密度）。

        此维度部分为论文写作支撑数据，部分指标需人工填写基线值。
        自动计算的指标：
          operators_covered      — 覆盖算子数（有 MR 或有测试结果的）
          avg_mrs_per_operator   — 平均每算子 MR 数（from RQ1）
          test_density           — 平均每算子测试用例数（from RQ2）
          automation_scope       — 自动化程度说明（静态文本）
        """
        rq1 = self.collect_rq1()
        rq2 = self.collect_rq2()

        ops_covered = max(
            rq1.get("operators_with_mr", 0),
            rq2.get("operators_tested", 0),
        )
        avg_mrs = rq1.get("avg_mr_per_operator", 0.0)
        ops_tested = rq2.get("operators_tested", 0)
        total_cases = rq2.get("total_test_cases", 0)
        test_density = round(total_cases / ops_tested, 1) if ops_tested > 0 else 0.0

        return {
            "operators_covered": ops_covered,
            "avg_mrs_per_operator": avg_mrs,
            "test_density": test_density,
            "automation_scope": (
                "MR 生成（LLM + 模板 + SymPy 验证）、输入生成（RandomGenerator）、"
                "Oracle 评估（MRVerifier）、证据捕获（EvidenceCollector）全链路自动化"
            ),
            "note": (
                "与基线对比的绝对数值（如 MTR、语句覆盖率）需手动填入基线方法的测量结果，"
                "DeepMT 侧数据已在 RQ1/RQ2/RQ3 中提供。"
            ),
        }

    # ── 论文指标计算接口（T6/T7）────────────────────────────────────────────────

    def compute_retention_rate(
        self,
        layer: str = "operator",
        method: str = "all",
    ) -> float:
        """
        有效留存率 = 有效（已验证）MR 数 / 初始候选 MR 数。

        Args:
            layer:  目标层次（当前仅 "operator" 层有知识库支持）
            method: 来源过滤（"llm" / "template" / "all"）

        Returns:
            留存率 [0, 1]；无数据时返回 0.0
        """
        try:
            from deepmt.mr_generator.base.mr_repository import MRRepository
            repo = self._mr_repo or MRRepository()
            total = 0
            verified = 0
            for op in repo.list_operators():
                for mr in repo.load(op):
                    if method != "all" and mr.source != method:
                        continue
                    if layer != "all" and mr.layer != layer:
                        continue
                    total += 1
                    if getattr(mr, "verified", False) or mr.lifecycle_state in ("verified", "active"):
                        verified += 1
            return round(verified / total, 4) if total > 0 else 0.0
        except Exception as e:
            logger.warning(f"[compute_retention_rate] 计算失败: {e}")
            return 0.0

    def compute_mutation_score(
        self,
        layer: str = "operator",
        framework: str = "pytorch",
    ) -> float:
        """
        变异检出率 = 被至少一个测试用例检出的变异数 / 注入变异体总数。

        从 MutationTester 的历史执行结果（内存或数据库）中读取，
        若无持久化结果，返回 None 并记录日志提示先运行变异测试。

        Args:
            layer:     目标层次（"operator" / "model" / "application"）
            framework: 目标框架

        Returns:
            变异检出率 [0, 1]；无数据时返回 0.0
        """
        try:
            from deepmt.core.results_manager import ResultsManager
            rm = ResultsManager(db_path=self._db_path) if self._db_path else ResultsManager()
            mutation_data = rm.get_mutation_results(layer=layer, framework=framework)
            if not mutation_data:
                logger.info(
                    "[compute_mutation_score] 无变异测试历史记录，"
                    "请先运行 deepmt test mutate 生成数据"
                )
                return 0.0
            detected = sum(1 for r in mutation_data if r.get("detected", False))
            return round(detected / len(mutation_data), 4)
        except Exception as e:
            logger.warning(f"[compute_mutation_score] 计算失败: {e}")
            return 0.0

    def compute_stat_confidence(
        self,
        layer: str = "operator",
        threshold: float = 0.95,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        N≥100 统计置信度验证（T7）。

        对已入库的 MR 执行 n_samples 次采样测试，
        通过率低于 threshold 的 MR 标记为 stat_unverified。

        Args:
            layer:      目标层次
            threshold:  通过率阈值（默认 0.95）
            n_samples:  采样次数（默认 100）
            seed:       随机种子，确保可复现

        Returns:
            {
              "total_mrs":          int,     # 参与评估的 MR 数
              "passed_mrs":         int,     # 通过率 >= threshold 的 MR 数
              "stat_confidence":    float,   # 通过 MR 占比
              "flagged_mrs":        list,    # 通过率 < threshold 的 MR id 列表
            }
        """
        try:
            from deepmt.mr_generator.base.mr_repository import MRRepository
            from deepmt.engine.batch_test_runner import BatchTestRunner
            from deepmt.core.results_manager import ResultsManager
            from unittest.mock import MagicMock

            repo = self._mr_repo or MRRepository()
            mock_rm = MagicMock(spec=ResultsManager)
            runner = BatchTestRunner(repo=repo, results_manager=mock_rm)

            total_mrs = 0
            passed_mrs = 0
            flagged: List[str] = []

            for op in repo.list_operators():
                for mr in repo.load(op):
                    if mr.layer != layer:
                        continue
                    total_mrs += 1
                    try:
                        summary = runner.run_operator(
                            operator_name=op,
                            framework="pytorch",
                            n_samples=n_samples,
                            seed=seed,
                            mr_id=mr.id,
                        )
                        rate = summary.passed / summary.total_cases if summary.total_cases > 0 else 0.0
                        if rate >= threshold:
                            passed_mrs += 1
                        else:
                            flagged.append(mr.id)
                            # 标记为 stat_unverified（更新知识库中的状态）
                            try:
                                repo.update_lifecycle_state(mr.id, "stat_unverified")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"[stat_confidence] {op}/{mr.id} 测试失败: {e}")
                        flagged.append(mr.id)

            return {
                "total_mrs": total_mrs,
                "passed_mrs": passed_mrs,
                "stat_confidence": round(passed_mrs / total_mrs, 4) if total_mrs > 0 else 0.0,
                "flagged_mrs": flagged,
                "threshold": threshold,
                "n_samples": n_samples,
            }

        except Exception as e:
            logger.warning(f"[compute_stat_confidence] 计算失败: {e}")
            return {"error": str(e)}

    def format_text(self, data: Dict[str, Any]) -> str:
        """将 collect_all() 的输出格式化为终端可读文本。"""
        now = data.get("generated_at", "")[:19]
        lines = [
            f"\nDeepMT 实验数据报告  [{now}]",
            "═" * 72,
        ]

        # ── RQ1 ────────────────────────────────────────────────────────────────
        rq1 = data.get("rq1", {})
        lines += [
            "\n【RQ1】MR 自动生成质量",
            "─" * 60,
            f"  MR 总数:          {rq1.get('total_mr_count', '?')}",
            f"  已验证 MR:        {rq1.get('verified_mr_count', '?')}",
            f"  验证率:           {_pct(rq1.get('verification_rate'))}",
            f"  覆盖算子数:       {rq1.get('operators_with_mr', '?')}",
            f"  平均每算子 MR 数: {rq1.get('avg_mr_per_operator', '?')}",
        ]
        cats = rq1.get("category_distribution", {})
        if cats:
            lines.append("  分类分布:")
            for cat, cnt in cats.items():
                lines.append(f"    {cat:<20} {cnt} 条")

        # ── RQ2 ────────────────────────────────────────────────────────────────
        rq2 = data.get("rq2", {})
        lines += [
            "\n【RQ2】缺陷检测能力",
            "─" * 60,
            f"  总测试用例:       {rq2.get('total_test_cases', '?')}",
            f"  通过:             {rq2.get('total_passed', '?')}",
            f"  失败:             {rq2.get('total_failed', '?')}",
            f"  通过率:           {_pct(rq2.get('overall_pass_rate'))}",
            f"  被测算子数:       {rq2.get('operators_tested', '?')}",
            f"  有失败的算子数:   {rq2.get('operators_with_failure', '?')}",
            f"  可复现证据包:     {rq2.get('evidence_pack_count', '?')} 个",
            f"  独立缺陷线索:     {rq2.get('unique_defect_leads', '?')} 条",
        ]

        # ── RQ3 ────────────────────────────────────────────────────────────────
        rq3 = data.get("rq3", {})
        lines += [
            "\n【RQ3】跨框架一致性",
            "─" * 60,
            f"  跨框架实验次数:   {rq3.get('cross_session_count', '?')}",
            f"  对比算子数:       {rq3.get('operators_compared', '?')}",
            f"  平均一致率:       {_pct(rq3.get('overall_consistency_rate'))}",
            f"  平均输出最大差:   {_fmt(rq3.get('avg_output_max_diff'))}",
            f"  不一致 MR 数:     {rq3.get('inconsistent_mr_count', '?')}",
        ]
        if rq3.get("note"):
            lines.append(f"  注: {rq3['note']}")
        for s in rq3.get("sessions", []):
            lines.append(
                f"    {s['operator']:<45}"
                f"  {s['framework_pair']}"
                f"  一致率={_pct(s['consistency_rate'])}"
            )

        # ── RQ4 ────────────────────────────────────────────────────────────────
        rq4 = data.get("rq4", {})
        lines += [
            "\n【RQ4】与基线对比（自动化覆盖）",
            "─" * 60,
            f"  覆盖算子总数:     {rq4.get('operators_covered', '?')}",
            f"  平均每算子 MR 数: {rq4.get('avg_mrs_per_operator', '?')}",
            f"  平均每算子用例:   {rq4.get('test_density', '?')}",
            f"  自动化范围:       {rq4.get('automation_scope', '')}",
        ]

        lines.append("\n" + "═" * 72)
        return "\n".join(lines)


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _pct(v) -> str:
    if v is None:
        return "N/A（尚无数据）"
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v)


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)
