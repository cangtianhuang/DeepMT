"""
deepmt experiment — 论文实验基准与数据生产子命令组（Phase L）

命令:
    run      创建并记录实验运行清单（不执行测试，仅记录配置与环境）
    collect  收集 RQ1-RQ4 当前统计数据并打印
    export   将统计数据导出为文件（JSON / CSV / Markdown / 全部）
    list     列出历史实验运行清单
    show     查看单条运行清单详情
    benchmark 查看固定 benchmark 清单
    case     case study 管理（list / show / export）
    env      查看当前运行环境与版本矩阵
"""

import json
import sys

import click

from deepmt._utils import not_implemented_error


@click.group()
def experiment():
    """论文实验基准与自动化数据生产（Phase L）。"""


# ── run ───────────────────────────────────────────────────────────────────────

@experiment.command("run")
@click.option("--rq", multiple=True, default=(), metavar="RQ",
              help="目标 RQ，如 --rq rq1 --rq rq2（默认全部）")
@click.option("--seed", default=42, show_default=True, help="随机种子")
@click.option("--notes", default="", help="附加备注")
@click.option("--no-env", is_flag=True, default=False, help="跳过环境快照（加速）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 输出")
def experiment_run(rq, seed, notes, no_env, as_json):
    """创建实验运行清单（记录 seed / benchmark / 环境快照）。

    \b
    示例:
      deepmt experiment run
      deepmt experiment run --rq rq1 --rq rq2 --seed 42
      deepmt experiment run --notes "第一次完整实验" --json
    """
    from deepmt.experiments.runs.run_manifest import RunManifestManager

    rqs = list(rq) if rq else None
    mgr = RunManifestManager()
    manifest = mgr.create(
        rqs=rqs,
        seed=seed,
        capture_env=not no_env,
        notes=notes,
    )
    mgr.save(manifest)

    if as_json:
        click.echo(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    else:
        click.echo(manifest.format_text())
        click.echo(f"\n[experiment run] 已保存: data/experiments/runs/{manifest.run_id}.json")


# ── collect ───────────────────────────────────────────────────────────────────

@experiment.command("collect")
@click.option("--rq", multiple=True, default=(), metavar="RQ",
              help="目标 RQ（默认全部）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 输出")
@click.option("--run-id", default=None, help="关联的 RunManifest ID")
def experiment_collect(rq, as_json, run_id):
    """收集 RQ1-RQ4 当前统计数据并打印。

    \b
    示例:
      deepmt experiment collect
      deepmt experiment collect --rq rq1 --rq rq2
      deepmt experiment collect --json
    """
    from deepmt.analysis.stats.aggregator import StatsAggregator
    from deepmt.analysis.experiment_organizer import ExperimentOrganizer

    rqs = list(rq) if rq else None
    agg = StatsAggregator()
    stats = agg.collect(rqs=rqs, run_id=run_id)

    if as_json:
        click.echo(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        org = ExperimentOrganizer()
        data = {
            "generated_at": stats.generated_at,
            "rq1": stats.get_rq("rq1"),
            "rq2": stats.get_rq("rq2"),
            "rq3": stats.get_rq("rq3"),
            "rq4": stats.get_rq("rq4"),
        }
        click.echo(org.format_text(data))


# ── export ────────────────────────────────────────────────────────────────────

@experiment.command("export")
@click.option("--format", "fmt",
              type=click.Choice(["json", "csv", "markdown", "all"], case_sensitive=False),
              default="all", show_default=True, help="导出格式")
@click.option("--output", default="data/experiments/exports", show_default=True,
              help="导出目录")
@click.option("--rq", multiple=True, default=(), metavar="RQ",
              help="目标 RQ（默认全部）")
@click.option("--run-id", default=None, help="关联的 RunManifest ID")
@click.option("--figures", is_flag=True, default=False, help="同时导出图表")
@click.option("--ascii-only", is_flag=True, default=False,
              help="图表只生成 ASCII（不依赖 matplotlib）")
def experiment_export(fmt, output, rq, run_id, figures, ascii_only):
    """将统计数据导出为论文可用文件。

    \b
    示例:
      deepmt experiment export
      deepmt experiment export --format markdown
      deepmt experiment export --format all --output data/my_export
      deepmt experiment export --figures --ascii-only
    """
    from deepmt.analysis.stats.aggregator import StatsAggregator
    from deepmt.analysis.stats.exporter import StatsExporter

    rqs = list(rq) if rq else None
    click.echo(f"[experiment export] 收集数据 ...")
    stats = StatsAggregator().collect(rqs=rqs, run_id=run_id)

    exporter = StatsExporter(output_dir=output)
    exported = {}
    if fmt in ("json", "all"):
        exported["json"] = exporter.export_json(stats)
    if fmt in ("csv", "all"):
        exported["csv"] = exporter.export_csv_per_rq(stats)
        exported["benchmark_csv"] = exporter.export_benchmark_csv()
    if fmt in ("markdown", "all"):
        exported["markdown"] = exporter.export_markdown(stats)

    click.echo(f"\n[experiment export] 共导出 {len(exported)} 个文件：")
    for key, path in exported.items():
        click.echo(f"  {key:<15} → {path}")

    if figures:
        click.echo("\n[experiment export] 导出图表 ...")
        from deepmt.scripts.export_thesis_figures import main as figures_main
        argv = ["--output", f"{output}/../figures"]
        if ascii_only:
            argv.append("--ascii-only")
        figures_main(argv)


# ── list ──────────────────────────────────────────────────────────────────────

@experiment.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def experiment_list(as_json):
    """列出历史实验运行清单。

    \b
    示例:
      deepmt experiment list
      deepmt experiment list --json
    """
    from deepmt.experiments.runs.run_manifest import RunManifestManager

    mgr = RunManifestManager()
    manifests = mgr.list_all()

    if not manifests:
        click.echo("（尚无实验运行记录。运行 deepmt experiment run 创建。）")
        return

    if as_json:
        click.echo(json.dumps([m.to_dict() for m in manifests],
                               ensure_ascii=False, indent=2))
        return

    click.echo(f"{'Run ID':<14}  {'Created':<19}  {'Status':<10}  {'Seed':>6}  RQs")
    click.echo("─" * 72)
    for m in manifests:
        click.echo(
            f"{m.run_id:<14}  {m.created_at[:19]}  {m.status:<10}  "
            f"{m.seed:>6}  {', '.join(m.rqs)}"
        )


# ── show ──────────────────────────────────────────────────────────────────────

@experiment.command("show")
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def experiment_show(run_id, as_json):
    """查看单条运行清单详情。

    \b
    示例:
      deepmt experiment show abc123def456
    """
    from deepmt.experiments.runs.run_manifest import RunManifestManager

    mgr = RunManifestManager()
    manifest = mgr.load(run_id)
    if manifest is None:
        click.echo(f"[experiment show] 未找到 run_id={run_id}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    else:
        click.echo(manifest.format_text())
        if manifest.environment:
            env = manifest.environment
            click.echo("\n环境快照:")
            click.echo(f"  Python:   {env.get('python_version', '?')}")
            click.echo(f"  Platform: {env.get('platform_info', '?')}")
            for fw, ver in env.get("framework_versions", {}).items():
                click.echo(f"  {fw:<16} {ver}")


# ── benchmark ────────────────────────────────────────────────────────────────

@experiment.command("benchmark")
@click.option("--layer", type=click.Choice(["operator", "model", "application", "all"]),
              default="all", show_default=True, help="只显示指定层次")
@click.option("--json", "as_json", is_flag=True, default=False)
def experiment_benchmark(layer, as_json):
    """查看固化的论文实验 benchmark 清单。

    \b
    示例:
      deepmt experiment benchmark
      deepmt experiment benchmark --layer operator
      deepmt experiment benchmark --json
    """
    from deepmt.experiments.benchmarks.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite()
    summary = suite.summary()

    if as_json:
        data = {}
        if layer in ("operator", "all"):
            data["operators"] = [
                {"name": e.name, "framework": e.framework, "category": e.category}
                for e in suite.operator_benchmark()
            ]
        if layer in ("model", "all"):
            data["models"] = suite.model_names()
        if layer in ("application", "all"):
            data["applications"] = suite.application_names()
        data["summary"] = summary
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    click.echo("\nDeepMT 论文实验 Benchmark Suite")
    click.echo("═" * 60)
    click.echo(
        f"  算子层: {summary['operator_count']} 个  "
        f"| 模型层: {summary['model_count']} 个  "
        f"| 应用层: {summary['application_count']} 个"
    )
    click.echo("")

    if layer in ("operator", "all"):
        click.echo("【算子层 Benchmark】")
        click.echo(f"  {'名称':<45}  {'分类':<16}  框架")
        click.echo("  " + "─" * 72)
        for e in suite.operator_benchmark():
            click.echo(f"  {e.name:<45}  {e.category:<16}  {e.framework}")
        click.echo("")

    if layer in ("model", "all"):
        click.echo("【模型层 Benchmark】")
        for name in suite.model_names():
            click.echo(f"  {name}")
        click.echo("")

    if layer in ("application", "all"):
        click.echo("【应用层 Benchmark】")
        for name in suite.application_names():
            click.echo(f"  {name}")
        click.echo("")


# ── case ──────────────────────────────────────────────────────────────────────

@experiment.group("case")
def experiment_case():
    """Case Study 管理。"""


@experiment_case.command("list")
@click.option("--status", default=None,
              type=click.Choice(["draft", "confirmed", "closed"]),
              help="过滤状态")
@click.option("--json", "as_json", is_flag=True, default=False)
def case_list(status, as_json):
    """列出所有 case study。"""
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    cases = idx.list_all(status=status)
    if not cases:
        click.echo("（尚无 case study 记录。）")
        return

    if as_json:
        click.echo(json.dumps([c.to_dict() for c in cases],
                               ensure_ascii=False, indent=2, default=str))
        return

    click.echo(f"{'Case ID':<12}  {'算子':<30}  {'框架':<10}  {'状态':<10}  摘要")
    click.echo("─" * 80)
    for c in cases:
        summary = (c.summary[:30] + "...") if len(c.summary) > 30 else c.summary
        click.echo(
            f"{c.case_id:<12}  {c.operator:<30}  {c.framework:<10}  "
            f"{c.status:<10}  {summary or '_(未填写)_'}"
        )


@experiment_case.command("show")
@click.argument("case_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def case_show(case_id, as_json):
    """查看单个 case study 详情。"""
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    case = idx.load(case_id)
    if case is None:
        click.echo(f"[case show] 未找到 case_id={case_id}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(case.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(case.to_markdown())


@experiment_case.command("export")
@click.option("--output", default=None, help="输出文件路径（默认 case_studies/catalog.md）")
@click.option("--status", default=None,
              type=click.Choice(["draft", "confirmed", "closed"]),
              help="过滤状态")
def case_export(output, status):
    """将所有 case study 导出为 Markdown 目录文件。"""
    from pathlib import Path
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    out_path = Path(output) if output else None
    path = idx.export_markdown_catalog(output_path=out_path, status=status)
    click.echo(f"[case export] Markdown 目录已写入: {path}")


# ── env ───────────────────────────────────────────────────────────────────────

@experiment.command("env")
@click.option("--json", "as_json", is_flag=True, default=False)
def experiment_env(as_json):
    """查看当前运行环境与框架版本矩阵。

    \b
    示例:
      deepmt experiment env
      deepmt experiment env --json
    """
    from deepmt.experiments.runs.environment_recorder import EnvironmentRecorder
    from deepmt.experiments.version_matrix import check_version_compatibility

    snap = EnvironmentRecorder().capture()
    compat = check_version_compatibility()

    if as_json:
        click.echo(json.dumps({
            "snapshot": snap.to_dict(),
            "version_compatibility": compat,
        }, ensure_ascii=False, indent=2))
        return

    click.echo(snap.format_text())
    click.echo("\n版本矩阵对比（pinned vs installed）：")
    click.echo(f"  {'框架':<16}  {'固定版本':<12}  {'已安装版本':<16}  状态")
    click.echo("  " + "─" * 60)
    for row in compat:
        status_icon = {"ok": "✅", "mismatch": "⚠️ ", "not_installed": "❌"}.get(
            row["status"], "?"
        )
        click.echo(
            f"  {row['framework']:<16}  {row['pinned']:<12}  "
            f"{row['installed']:<16}  {status_icon} {row['status']}"
        )
