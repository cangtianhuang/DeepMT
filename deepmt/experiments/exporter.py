"""
实验统计导出器。

职责：
  将 ThesisStats / 扁平表格数据导出为多种格式：
  - JSON：完整原始数据（机器可读）
  - CSV：扁平表格（可直接导入 Excel/pandas）
  - Markdown：论文可读表格

用法::

    from deepmt.experiments.aggregator import StatsAggregator
    from deepmt.experiments.exporter import StatsExporter

    agg = StatsAggregator()
    stats = agg.collect()

    exp = StatsExporter(output_dir="data/experiments/exports")
    exp.export_json(stats)
    exp.export_markdown(stats)
    exp.export_csv_per_rq(stats)
    exp.export_all(stats)  # 一次导出全部格式
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger


class StatsExporter:
    """
    将 ThesisStats 导出为多种格式。

    Args:
        output_dir: 导出目录（默认 data/experiments/exports）
    """

    def __init__(self, output_dir: Optional[str] = None):
        self._dir = Path(output_dir) if output_dir else Path("data/experiments/exports")
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────

    def export_json(
        self,
        stats: Any,
        filename: Optional[str] = None,
    ) -> Path:
        """
        将 ThesisStats 导出为 JSON 文件。

        Returns: 写入的文件路径
        """
        ts = self._ts()
        fname = filename or f"thesis_stats_{ts}.json"
        path = self._dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[StatsExporter] JSON 已写入: {path}")
        return path

    # ── Markdown ──────────────────────────────────────────────────────────────

    def export_markdown(
        self,
        stats: Any,
        filename: Optional[str] = None,
    ) -> Path:
        """将所有 RQ 的核心指标导出为 Markdown 表格文件。"""
        ts = self._ts()
        fname = filename or f"thesis_stats_{ts}.md"
        path = self._dir / fname

        lines = [
            f"# DeepMT 论文实验统计报告",
            f"",
            f"> 生成时间：{stats.generated_at[:19]}  ",
            f"> 关联 Run ID：{stats.run_id or 'N/A'}",
            f"",
        ]

        # benchmark 规模
        bsummary = stats.benchmark_summary
        if bsummary:
            lines += [
                "## Benchmark 规模",
                "",
                f"| 层次 | 数量 |",
                f"|------|------|",
                f"| 算子层 | {bsummary.get('operator_count', '?')} |",
                f"| 模型层 | {bsummary.get('model_count', '?')} |",
                f"| 应用层 | {bsummary.get('application_count', '?')} |",
                "",
            ]

        # 各 RQ
        from deepmt.experiments.rq_config import RQ_DEFINITIONS
        for rq_id, cfg in RQ_DEFINITIONS.items():
            raw = stats.get_rq(rq_id)
            lines += [
                f"## {rq_id.upper()}：{cfg.question[:60]}...",
                "",
                f"| 指标 | 值 | 单位 |",
                f"|------|----|------|",
            ]
            for m in cfg.metrics:
                val = raw.get(m.name, "N/A")
                if isinstance(val, dict):
                    val_str = _dict_to_inline(val, max_items=5)
                elif isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val[:5])
                elif isinstance(val, float):
                    val_str = f"{val:.4g}"
                else:
                    val_str = str(val) if val is not None else "N/A"
                label = m.thesis_label or m.name
                lines.append(f"| {label} | {val_str} | {m.unit} |")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"[StatsExporter] Markdown 已写入: {path}")
        return path

    # ── CSV ───────────────────────────────────────────────────────────────────

    def export_csv_per_rq(
        self,
        stats: Any,
        filename: Optional[str] = None,
    ) -> Path:
        """
        将所有 RQ 的指标以扁平格式导出为单个 CSV 文件。

        列：rq, metric, thesis_label, value, unit, description
        """
        ts = self._ts()
        fname = filename or f"thesis_stats_{ts}.csv"
        path = self._dir / fname

        from deepmt.experiments.aggregator import StatsAggregator
        from deepmt.experiments.rq_config import RQ_DEFINITIONS

        rows: List[Dict[str, str]] = []
        for rq_id, cfg in RQ_DEFINITIONS.items():
            raw = stats.get_rq(rq_id)
            for m in cfg.metrics:
                val = raw.get(m.name, "")
                if isinstance(val, dict):
                    val_str = _dict_to_inline(val, max_items=10)
                elif isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val)
                elif isinstance(val, float):
                    val_str = f"{val:.6g}"
                else:
                    val_str = str(val) if val is not None else ""
                rows.append({
                    "rq": rq_id.upper(),
                    "metric": m.name,
                    "thesis_label": m.thesis_label,
                    "value": val_str,
                    "unit": m.unit,
                    "description": m.description,
                })

        with open(path, "w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"[StatsExporter] CSV 已写入: {path}")
        return path

    def export_benchmark_csv(
        self,
        filename: Optional[str] = None,
    ) -> Path:
        """将 benchmark 清单（算子/模型/应用）导出为 CSV。"""
        ts = self._ts()
        fname = filename or f"benchmark_suite_{ts}.csv"
        path = self._dir / fname

        from deepmt.benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite()

        rows = []
        for e in suite.operator_benchmark():
            rows.append({
                "layer": "operator",
                "name": e.name,
                "framework": e.framework,
                "category": e.category,
                "notes": e.notes,
            })
        for name in suite.model_names():
            rows.append({
                "layer": "model",
                "name": name,
                "framework": "pytorch",
                "category": "neural_network",
                "notes": "",
            })
        for name in suite.application_names():
            rows.append({
                "layer": "application",
                "name": name,
                "framework": "pytorch",
                "category": "end_to_end",
                "notes": "",
            })

        with open(path, "w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"[StatsExporter] Benchmark CSV 已写入: {path}")
        return path

    # ── 一键导出 ──────────────────────────────────────────────────────────────

    def export_all(
        self,
        stats: Any,
        prefix: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        一次性导出 JSON + Markdown + CSV 三种格式。

        Returns:
            {"json": path, "markdown": path, "csv": path, "benchmark_csv": path}
        """
        ts = self._ts()
        p = prefix or f"thesis_{ts}"
        return {
            "json":          self.export_json(stats, filename=f"{p}.json"),
            "markdown":      self.export_markdown(stats, filename=f"{p}.md"),
            "csv":           self.export_csv_per_rq(stats, filename=f"{p}.csv"),
            "benchmark_csv": self.export_benchmark_csv(filename=f"benchmark_{ts}.csv"),
        }

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def _dict_to_inline(d: Dict, max_items: int = 5) -> str:
    """将 dict 转为紧凑内联字符串，超出 max_items 时截断。"""
    items = list(d.items())
    shown = items[:max_items]
    tail = f"... (+{len(items) - max_items})" if len(items) > max_items else ""
    return "; ".join(f"{k}={v}" for k, v in shown) + tail
