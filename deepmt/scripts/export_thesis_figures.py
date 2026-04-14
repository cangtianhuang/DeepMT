"""
论文图表导出脚本。

功能：
  从当前系统状态收集 RQ1-RQ4 统计数据，生成：
  - ASCII 图表（始终可用，不依赖 matplotlib）
  - PNG 图表（需要 matplotlib；若未安装则自动跳过）

支持图表类型：
  rq1_category_bar    — RQ1：MR 分类分布柱状图
  rq1_source_pie      — RQ1：MR 来源饼图
  rq2_pass_fail_bar   — RQ2：通过/失败用例数柱状图（按算子）
  rq3_consistency_bar — RQ3：跨框架一致率柱状图
  benchmark_coverage  — benchmark 三层覆盖比例图

运行方式::

    python -m deepmt.scripts.export_thesis_figures
    python -m deepmt.scripts.export_thesis_figures --output data/experiments/figures
    python -m deepmt.scripts.export_thesis_figures --ascii-only
    python -m deepmt.scripts.export_thesis_figures --chart rq1_category_bar
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── ASCII 图表工具 ────────────────────────────────────────────────────────────

def _ascii_bar_chart(
    data: Dict[str, Any],
    title: str = "",
    width: int = 40,
    unit: str = "",
) -> str:
    """生成横向 ASCII 柱状图字符串。"""
    if not data:
        return f"  [{title}]  (无数据)\n"

    max_val = max(float(v) for v in data.values() if v is not None) or 1
    label_w = max(len(k) for k in data.keys())

    lines = []
    if title:
        lines.append(f"\n  {title}")
        lines.append("  " + "─" * (label_w + width + 12))
    for label, val in data.items():
        if val is None:
            val = 0
        bar_len = int(float(val) / max_val * width)
        bar = "█" * bar_len
        lines.append(f"  {label:<{label_w}}  {bar:<{width}}  {val}{unit}")
    return "\n".join(lines)


def _ascii_summary_text(stats: Any) -> str:
    """生成 RQ1-RQ4 核心指标的 ASCII 摘要文本（可直接打印或写文件）。"""
    from deepmt.experiments.organizer import ExperimentOrganizer

    org = ExperimentOrganizer()
    data = {
        "generated_at": stats.generated_at,
        "rq1": stats.get_rq("rq1"),
        "rq2": stats.get_rq("rq2"),
        "rq3": stats.get_rq("rq3"),
        "rq4": stats.get_rq("rq4"),
    }
    return org.format_text(data)


# ── 图表生成函数 ──────────────────────────────────────────────────────────────

def _chart_rq1_category_bar(stats: Any, output_dir: Path, ascii_only: bool) -> List[Path]:
    """RQ1：MR 分类分布柱状图。"""
    raw = stats.get_rq("rq1")
    cats = raw.get("category_distribution", {})
    written = []

    # ASCII 版本
    txt = _ascii_bar_chart(cats, title="RQ1：MR 分类分布", unit=" 条")
    p = output_dir / "rq1_category_bar.txt"
    p.write_text(txt, encoding="utf-8")
    written.append(p)
    print(txt)

    # PNG 版本（可选）
    if not ascii_only and cats:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(list(cats.keys()), list(cats.values()), color="#4C72B0")
            ax.set_xlabel("MR 数量")
            ax.set_title("RQ1：MR 分类分布")
            ax.invert_yaxis()
            plt.tight_layout()
            png = output_dir / "rq1_category_bar.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            written.append(png)
        except ImportError:
            pass

    return written


def _chart_rq2_pass_fail(stats: Any, output_dir: Path, ascii_only: bool) -> List[Path]:
    """RQ2：通过/失败汇总柱状图。"""
    raw = stats.get_rq("rq2")
    data = {
        "总用例": raw.get("total_test_cases", 0),
        "通过": raw.get("total_passed", 0),
        "失败": raw.get("total_failed", 0),
    }
    written = []

    txt = _ascii_bar_chart(data, title="RQ2：测试用例通过/失败")
    p = output_dir / "rq2_pass_fail_bar.txt"
    p.write_text(txt, encoding="utf-8")
    written.append(p)
    print(txt)

    if not ascii_only:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = list(data.keys())
            values = list(data.values())
            colors = ["#4472C4", "#70AD47", "#FF0000"]
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values, color=colors[:len(labels)])
            ax.set_ylabel("用例数")
            ax.set_title("RQ2：测试结果概况")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(int(bar.get_height())),
                    ha="center",
                )
            plt.tight_layout()
            png = output_dir / "rq2_pass_fail_bar.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            written.append(png)
        except ImportError:
            pass

    return written


def _chart_rq3_consistency(stats: Any, output_dir: Path, ascii_only: bool) -> List[Path]:
    """RQ3：跨框架一致率。"""
    raw = stats.get_rq("rq3")
    sessions = raw.get("sessions", [])
    written = []

    if not sessions:
        txt = "  [RQ3 一致率]  (尚无跨框架实验数据)\n"
    else:
        data = {
            f"{s['operator'][:20]} ({s['framework_pair']})": s["consistency_rate"]
            for s in sessions
        }
        txt = _ascii_bar_chart(data, title="RQ3：跨框架一致率", unit="")

    p = output_dir / "rq3_consistency_bar.txt"
    p.write_text(txt, encoding="utf-8")
    written.append(p)
    print(txt)

    if not ascii_only and sessions:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [s["operator"][:15] for s in sessions]
            values = [s["consistency_rate"] for s in sessions]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, values, color="#70AD47")
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("一致率")
            ax.set_title("RQ3：跨框架一致率")
            ax.invert_yaxis()
            plt.tight_layout()
            png = output_dir / "rq3_consistency_bar.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            written.append(png)
        except ImportError:
            pass

    return written


def _chart_benchmark_coverage(stats: Any, output_dir: Path, ascii_only: bool) -> List[Path]:
    """Benchmark 三层覆盖规模。"""
    bsummary = stats.benchmark_summary
    data = {
        "算子层": bsummary.get("operator_count", 0),
        "模型层": bsummary.get("model_count", 0),
        "应用层": bsummary.get("application_count", 0),
    }
    written = []

    txt = _ascii_bar_chart(data, title="Benchmark 三层覆盖规模", unit=" 个")
    p = output_dir / "benchmark_coverage.txt"
    p.write_text(txt, encoding="utf-8")
    written.append(p)
    print(txt)

    if not ascii_only:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = list(data.keys())
            values = list(data.values())
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(labels, values, color=["#4472C4", "#ED7D31", "#A9D18E"])
            ax.set_ylabel("数量")
            ax.set_title("Benchmark 三层覆盖规模")
            plt.tight_layout()
            png = output_dir / "benchmark_coverage.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            written.append(png)
        except ImportError:
            pass

    return written


# ── 图表注册表 ────────────────────────────────────────────────────────────────

_CHART_REGISTRY = {
    "rq1_category_bar":    _chart_rq1_category_bar,
    "rq2_pass_fail_bar":   _chart_rq2_pass_fail,
    "rq3_consistency_bar": _chart_rq3_consistency,
    "benchmark_coverage":  _chart_benchmark_coverage,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_thesis_figures",
        description="导出 DeepMT 论文实验图表",
    )
    parser.add_argument(
        "--output",
        default="data/experiments/figures",
        help="导出目录（默认 data/experiments/figures）",
    )
    parser.add_argument(
        "--chart",
        nargs="*",
        choices=list(_CHART_REGISTRY.keys()),
        default=None,
        metavar="CHART",
        help=f"指定图表（默认全部）：{', '.join(_CHART_REGISTRY)}",
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="只生成 ASCII 文本图表，跳过 PNG",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集数据
    from deepmt.experiments.stats.aggregator import StatsAggregator
    print("[figures] 收集实验数据 ...")
    stats = StatsAggregator().collect()

    # 生成 ASCII 摘要
    try:
        summary_txt = _ascii_summary_text(stats)
        p = output_dir / "experiment_summary.txt"
        p.write_text(summary_txt, encoding="utf-8")
        print(summary_txt)
        print(f"\n[figures] 摘要已写入: {p}")
    except Exception as e:
        print(f"[figures] 摘要生成失败: {e}")

    # 生成图表
    charts = args.chart or list(_CHART_REGISTRY.keys())
    all_written = []
    for chart_name in charts:
        fn = _CHART_REGISTRY.get(chart_name)
        if fn is None:
            print(f"[figures] 未知图表: {chart_name}，已跳过")
            continue
        try:
            written = fn(stats, output_dir, args.ascii_only)
            all_written.extend(written)
        except Exception as e:
            print(f"[figures] {chart_name} 生成失败: {e}")

    print(f"\n[figures] 共生成 {len(all_written)} 个文件：")
    for p in all_written:
        print(f"  → {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
