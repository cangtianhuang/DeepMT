"""
论文表格导出脚本。

功能：
  从当前系统状态收集 RQ1-RQ4 统计数据，导出为：
  - Markdown 表格文件（可直接嵌入论文草稿）
  - CSV 文件（可导入 Excel/LaTeX）
  - JSON 文件（完整原始数据存档）

运行方式::

    # 方式 1：直接运行
    python -m deepmt.scripts.export_thesis_tables

    # 方式 2：通过 CLI（推荐）
    deepmt experiment export --format all

    # 方式 3：指定输出目录
    python -m deepmt.scripts.export_thesis_tables --output data/experiments/exports

选项：
  --output DIR   导出目录（默认 data/experiments/exports）
  --rq RQ_ID     只导出指定 RQ（如 --rq rq1），默认全部
  --format FMT   导出格式：json / csv / markdown / all（默认 all）
  --run-id ID    关联的 RunManifest ID（可选）
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_thesis_tables",
        description="导出 DeepMT 论文实验统计表格",
    )
    parser.add_argument(
        "--output",
        default="data/experiments/exports",
        help="导出目录（默认 data/experiments/exports）",
    )
    parser.add_argument(
        "--rq",
        nargs="*",
        default=None,
        metavar="RQ",
        help="要导出的 RQ，如 --rq rq1 rq2（默认全部）",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="导出格式（默认 all）",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="关联的 RunManifest ID（可选）",
    )

    args = parser.parse_args(argv)

    # ── 收集数据 ────────────────────────────────────────────────────────────
    from deepmt.experiments.aggregator import StatsAggregator
    from deepmt.experiments.exporter import StatsExporter

    rqs = args.rq
    agg = StatsAggregator()
    print(f"[export] 收集实验数据 rqs={rqs or 'all'} ...")
    stats = agg.collect(rqs=rqs, run_id=args.run_id)

    # ── 导出 ────────────────────────────────────────────────────────────────
    exporter = StatsExporter(output_dir=args.output)
    fmt = args.format

    exported = {}
    if fmt in ("json", "all"):
        exported["json"] = exporter.export_json(stats)
    if fmt in ("csv", "all"):
        exported["csv"] = exporter.export_csv_per_rq(stats)
        exported["benchmark_csv"] = exporter.export_benchmark_csv()
    if fmt in ("markdown", "all"):
        exported["markdown"] = exporter.export_markdown(stats)

    print(f"\n[export] 导出完成，共 {len(exported)} 个文件：")
    for key, path in exported.items():
        print(f"  {key:<15} → {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
