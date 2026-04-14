"""
deepmt case — 真实缺陷案例管理命令组（Phase M）

命令:
    list     列出所有案例及状态
    show     查看单个案例详情
    confirm  人工确认案例（更新状态、根因、严重程度等）
    build    从缺陷线索或证据包构建案例包
    export   导出案例目录（Markdown）
"""

import json
import sys

import click


@click.group()
def case():
    """真实缺陷案例管理（Phase M）。"""


# ── list ──────────────────────────────────────────────────────────────────────

@case.command("list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(["draft", "confirmed", "closed"]),
    help="按状态过滤",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def case_list(status, as_json):
    """列出所有案例。

    \b
    示例:
      deepmt case list
      deepmt case list --status confirmed
      deepmt case list --json
    """
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    cases = idx.list_all(status=status)

    if as_json:
        click.echo(json.dumps([c.to_dict() for c in cases], ensure_ascii=False, indent=2))
        return

    if not cases:
        click.echo("暂无案例。使用 'deepmt case build' 构建案例包。")
        return

    status_icons = {"confirmed": "✅", "draft": "📝", "closed": "🔒"}
    click.echo(f"\n案例库（共 {len(cases)} 个）")
    click.echo("─" * 70)
    for c in cases:
        icon = status_icons.get(c.status, "❓")
        click.echo(
            f"  {icon} [{c.case_id}]  {c.operator:<14} {c.framework:<10}"
            f"  {(c.defect_type or 'unknown'):<22}  severity={c.severity}"
        )
        if c.summary:
            click.echo(f"       {c.summary[:65]}")
    click.echo("─" * 70)
    click.echo(f"运行 'deepmt case show <case_id>' 查看详情\n")


# ── show ──────────────────────────────────────────────────────────────────────

@case.command("show")
@click.argument("case_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def case_show(case_id, as_json):
    """查看单个案例详情。

    \b
    示例:
      deepmt case show 009eb89bcb
      deepmt case show 009eb89bcb --json
    """
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    c = idx.load(case_id)
    if c is None:
        click.echo(f"错误：未找到案例 '{case_id}'", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(c.to_dict(), ensure_ascii=False, indent=2))
        return

    click.echo(c.to_markdown())


# ── confirm ───────────────────────────────────────────────────────────────────

@case.command("confirm")
@click.argument("case_id")
@click.option(
    "--status",
    default=None,
    type=click.Choice(["draft", "confirmed", "closed"]),
    help="更新案例状态",
)
@click.option(
    "--severity",
    default=None,
    type=click.Choice(["critical", "high", "medium", "low", "unknown"]),
    help="更新严重程度",
)
@click.option("--root-cause", default=None, metavar="TEXT", help="填写根因分析")
@click.option("--summary", default=None, metavar="TEXT", help="更新案例摘要")
@click.option("--defect-type", default=None, metavar="TYPE", help="更新缺陷类型")
@click.option("--notes", default=None, metavar="TEXT", help="追加备注（附加到现有备注后）")
def case_confirm(case_id, status, severity, root_cause, summary, defect_type, notes):
    """人工确认和更新案例字段。

    \b
    示例:
      deepmt case confirm 009eb89bcb --status confirmed --severity low
      deepmt case confirm e861263744 --root-cause "IEEE 754 浮点溢出边界行为"
      deepmt case confirm 009eb89bcb --summary "gelu MR 下界定义偏紧" --notes "已修复MR"
    """
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()
    c = idx.load(case_id)
    if c is None:
        click.echo(f"错误：未找到案例 '{case_id}'", err=True)
        sys.exit(1)

    updated_fields = []
    if status is not None:
        c.status = status
        updated_fields.append(f"status={status}")
    if severity is not None:
        c.severity = severity
        updated_fields.append(f"severity={severity}")
    if root_cause is not None:
        c.root_cause = root_cause
        updated_fields.append("root_cause")
    if summary is not None:
        c.summary = summary
        updated_fields.append("summary")
    if defect_type is not None:
        c.defect_type = defect_type
        updated_fields.append(f"defect_type={defect_type}")
    if notes is not None:
        c.notes = (c.notes + "\n" + notes) if c.notes else notes
        updated_fields.append("notes")

    if not updated_fields:
        click.echo("未指定任何更新选项，案例未修改。")
        return

    idx.save(c)
    click.echo(f"✅ 案例 {case_id} 已更新：{', '.join(updated_fields)}")


# ── build ─────────────────────────────────────────────────────────────────────

@case.command("build")
@click.option("--from-evidence", "evidence_id", default=None, metavar="ID",
              help="从指定证据包 ID 构建单个案例")
@click.option("--top", "top_n", default=0, type=int, show_default=True,
              metavar="N", help="从去重缺陷线索中构建前 N 个高优先级案例（0=全部）")
@click.option("--output", "output_dir", default=None, metavar="DIR",
              help="案例包输出目录（默认 data/cases/real_defects/）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 输出结果")
def case_build(evidence_id, top_n, output_dir, as_json):
    """从缺陷线索或证据包构建案例包。

    \b
    示例:
      deepmt case build --from-evidence bee67689-15b
      deepmt case build --top 3
      deepmt case build --top 5 --output /tmp/my_cases
    """
    from pathlib import Path
    from deepmt.analysis.defect_case_builder import DefectCaseBuilder
    from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator

    builder = DefectCaseBuilder(
        cases_dir=Path(output_dir) if output_dir else None
    )

    if evidence_id:
        # 从指定证据包 ID 构建
        case = builder.build_from_evidence(evidence_id)
        pkg_dir = builder.build_case_package(case)
        if as_json:
            click.echo(json.dumps({"case_id": case.case_id, "pkg_dir": str(pkg_dir)},
                                  ensure_ascii=False))
        else:
            click.echo(f"✅ 案例已构建")
            click.echo(f"   case_id: {case.case_id}")
            click.echo(f"   类型:    {case.defect_type}")
            click.echo(f"   案例包:  {pkg_dir}")
        return

    # 从去重缺陷线索批量构建
    dedup = DefectDeduplicator()
    leads = dedup.deduplicate()

    if not leads:
        click.echo("未发现缺陷线索。请先运行 'deepmt test batch --collect-evidence' 收集证据包。")
        return

    n = top_n if top_n > 0 else len(leads)
    click.echo(f"发现 {len(leads)} 条缺陷线索，构建前 {n} 个...")
    pkg_dirs = builder.build_top_leads(leads, top_n=n)

    if as_json:
        click.echo(json.dumps([str(p) for p in pkg_dirs], ensure_ascii=False))
        return

    click.echo(f"\n✅ 已构建 {len(pkg_dirs)} 个案例包：")
    for p in pkg_dirs:
        click.echo(f"   {p}")


# ── export ────────────────────────────────────────────────────────────────────

@case.command("export")
@click.option(
    "--format", "fmt",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    show_default=True,
    help="导出格式",
)
@click.option("--status", default=None,
              type=click.Choice(["draft", "confirmed", "closed"]),
              help="只导出指定状态的案例")
@click.option("--output", "output_path", default=None, metavar="PATH",
              help="输出文件路径（默认 data/experiments/case_studies/catalog.md）")
def case_export(fmt, status, output_path):
    """导出案例目录。

    \b
    示例:
      deepmt case export
      deepmt case export --status confirmed
      deepmt case export --format json --output /tmp/cases.json
      deepmt case export --output docs/case_catalog.md
    """
    from pathlib import Path
    from deepmt.experiments.case_study import CaseStudyIndex

    idx = CaseStudyIndex()

    if fmt == "markdown":
        out = Path(output_path) if output_path else None
        path = idx.export_markdown_catalog(output_path=out, status=status)
        click.echo(f"✅ Markdown 目录已导出：{path}")

    elif fmt == "json":
        cases = idx.list_all(status=status)
        data = [c.to_dict() for c in cases]
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            click.echo(f"✅ JSON 导出：{out}  （{len(cases)} 个案例）")
        else:
            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
