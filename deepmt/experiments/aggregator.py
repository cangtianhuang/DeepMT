"""
实验统计聚合器。

职责：
  以 RQConfig 为口径规格，从 ExperimentOrganizer 的原始输出中提取、
  重命名并聚合指标，生成结构化聚合结果。

聚合结果格式（ThesisStats）：
  - per-RQ 的指标 dict（与 MetricSpec.name 对齐）
  - benchmark 规模快照
  - 生成时间与运行关联 run_id

用法::

    from deepmt.experiments.aggregator import StatsAggregator

    agg = StatsAggregator()
    stats = agg.collect()          # 全量收集 RQ1-RQ4
    stats = agg.collect(rqs=["rq1","rq2"])  # 按需收集

    print(stats.to_dict())
    print(stats.get_rq("rq1"))
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger
from deepmt.experiments.rq_config import RQ_DEFINITIONS, RQConfig


@dataclass
class ThesisStats:
    """聚合后的论文实验统计结果对象。"""

    generated_at: str
    """生成时间（ISO 格式）。"""

    run_id: Optional[str] = None
    """关联的 RunManifest ID（如有）。"""

    rq_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """各 RQ 的指标 dict {rq_id: {metric_name: value}}。"""

    benchmark_summary: Dict[str, Any] = field(default_factory=dict)
    """benchmark 规模快照。"""

    def get_rq(self, rq_id: str) -> Dict[str, Any]:
        """返回指定 RQ 的指标 dict，未找到返回 {}。"""
        return self.rq_data.get(rq_id.lower(), {})

    def get_metric(self, rq_id: str, metric_name: str) -> Any:
        """返回指定 RQ 的单个指标值，未找到返回 None。"""
        return self.get_rq(rq_id).get(metric_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "rq_data": dict(self.rq_data),
            "benchmark_summary": dict(self.benchmark_summary),
        }


class StatsAggregator:
    """
    从 ExperimentOrganizer 聚合论文所需统计数据。

    Args:
        db_path:          ResultsManager 的数据库路径（可选）
        evidence_dir:     证据包目录（可选）
        cross_results_dir: 跨框架结果目录（可选）
        mr_repo:          MRRepository 实例（可选）
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

    def collect(
        self,
        rqs: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> ThesisStats:
        """
        收集并聚合实验统计数据。

        Args:
            rqs:    要收集的 RQ 列表（None=全部 rq1-rq4）
            run_id: 关联的 RunManifest ID

        Returns:
            ThesisStats 对象
        """
        if rqs is None:
            rqs = ["rq1", "rq2", "rq3", "rq4"]

        from deepmt.experiments.organizer import ExperimentOrganizer

        org = ExperimentOrganizer(
            db_path=self._db_path,
            evidence_dir=self._evidence_dir,
            cross_results_dir=self._cross_results_dir,
            mr_repo=self._mr_repo,
        )

        stats = ThesisStats(
            generated_at=datetime.now().isoformat(),
            run_id=run_id,
        )

        _collectors = {
            "rq1": org.collect_rq1,
            "rq2": org.collect_rq2,
            "rq3": org.collect_rq3,
            "rq4": org.collect_rq4,
        }

        for rq_id in rqs:
            key = rq_id.lower()
            if key not in _collectors:
                logger.warning(f"[StatsAggregator] 未知 RQ: {rq_id}，已跳过")
                continue
            try:
                raw = _collectors[key]()
                stats.rq_data[key] = raw
            except Exception as e:
                logger.warning(f"[StatsAggregator] {rq_id} 聚合失败: {e}")
                stats.rq_data[key] = {"error": str(e)}

        # benchmark 规模快照
        try:
            from deepmt.benchmarks.suite import BenchmarkSuite
            stats.benchmark_summary = BenchmarkSuite().summary()
        except Exception as e:
            logger.warning(f"[StatsAggregator] benchmark 汇总失败: {e}")

        return stats

    def collect_rq_flat_table(
        self,
        rq_id: str,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        将单个 RQ 的指标聚合为"扁平表格"（每个指标占一行）。

        Returns:
            [{"rq": str, "metric": str, "value": Any, "unit": str, "description": str}, ...]
        """
        rq_cfg = RQ_DEFINITIONS.get(rq_id.lower())
        stats = self.collect(rqs=[rq_id], run_id=run_id)
        raw = stats.get_rq(rq_id)

        rows = []
        for m in (rq_cfg.metrics if rq_cfg else []):
            val = raw.get(m.name)
            # dict/list 类型展开为字符串摘要
            if isinstance(val, dict):
                val_str = "; ".join(f"{k}={v}" for k, v in val.items())
            elif isinstance(val, list):
                val_str = str(val)
            else:
                val_str = val
            rows.append({
                "rq": rq_id.upper(),
                "metric": m.name,
                "thesis_label": m.thesis_label,
                "value": val_str,
                "unit": m.unit,
                "description": m.description,
            })
        return rows

    def collect_summary_table(self) -> List[Dict[str, Any]]:
        """
        生成覆盖 RQ1-RQ4 核心指标的汇总表（每个 RQ 一行，含主要数值指标）。

        Returns:
            [{"rq": str, "question_short": str, ...scalar metrics...}, ...]
        """
        stats = self.collect()
        rows = []
        for rq_id, cfg in RQ_DEFINITIONS.items():
            raw = stats.get_rq(rq_id)
            row: Dict[str, Any] = {
                "rq": rq_id.upper(),
                "question_short": cfg.question[:60] + "...",
            }
            # 只取标量指标（排除 dict/list）
            for m in cfg.metrics:
                val = raw.get(m.name)
                if isinstance(val, (int, float, str, type(None))):
                    row[m.thesis_label or m.name] = val
            rows.append(row)
        return rows
