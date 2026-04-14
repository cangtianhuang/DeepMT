"""
deepmt.commands.test.evidence — 证据包管理子命令组

命令组: evidence (list, show, script)
"""

import json
import sys

import click

from deepmt.commands.test._group import test


# ── evidence group ────────────────────────────────────────────────────────────

@test.group("evidence")
def test_evidence():
    """证据包管理：查看、展示可复现失败案例。"""


@test_evidence.command("list")
@click.option("--operator", default=None, help="按算子名称过滤")
@click.option("--framework", default=None, help="按框架过滤")
@click.option("--limit", default=20, show_default=True, type=int, help="最多显示条数（0=不限）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def evidence_list(operator, framework, limit, as_json):
    """列出已保存的证据包。

    \b
    示例:
      deepmt test evidence list
      deepmt test evidence list --operator torch.nn.functional.relu
      deepmt test evidence list --limit 5 --json
    """
    try:
        from deepmt.analysis.reporting.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        packs = collector.list_all(operator=operator, framework=framework, limit=limit)

        if as_json:
            click.echo(json.dumps([p.to_dict() for p in packs], ensure_ascii=False, indent=2))
            return

        if not packs:
            click.echo("暂无证据包记录。运行 deepmt test batch 时添加 --collect-evidence 可生成证据包。")
            return

        click.echo(f"\n证据包列表（共 {len(packs)} 条）")
        click.echo("─" * 70)
        for p in packs:
            click.echo(
                f"  {p.evidence_id}  {p.timestamp[:16]}"
                f"  {p.operator}  [{p.framework} {p.framework_version}]"
            )
            click.echo(f"    MR: {p.mr_description[:60]}")
            click.echo(f"    diff={p.actual_diff:.4g}  tol={p.tolerance:.4g}  {p.detail[:50]}")
        click.echo("─" * 70)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


@test_evidence.command("show")
@click.argument("evidence_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出完整数据")
def evidence_show(evidence_id, as_json):
    """显示单个证据包的详细信息。

    \b
    示例:
      deepmt test evidence show abc123def456
      deepmt test evidence show abc123def456 --json
    """
    try:
        from deepmt.analysis.reporting.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        pack = collector.load(evidence_id)

        if pack is None:
            click.echo(click.style(f"未找到证据包 {evidence_id!r}", fg="red"), err=True)
            sys.exit(1)

        if as_json:
            click.echo(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2))
            return

        click.echo(f"\n证据包详情 — {pack.evidence_id}")
        click.echo("─" * 70)
        click.echo(f"  时间:     {pack.timestamp}")
        click.echo(f"  算子:     {pack.operator}")
        click.echo(f"  框架:     {pack.framework} {pack.framework_version}")
        click.echo(f"  MR ID:    {pack.mr_id}")
        click.echo(f"  MR 描述:  {pack.mr_description}")
        click.echo(f"  变换代码: {pack.transform_code}")
        click.echo(f"  Oracle:   {pack.oracle_expr}")
        click.echo(f"  实测差值: {pack.actual_diff:.6g}  (容忍阈值: {pack.tolerance:.6g})")
        click.echo(f"  失败原因: {pack.detail}")
        shape = pack.input_summary.get("shape", "?")
        dtype = pack.input_summary.get("dtype", "?")
        click.echo(f"  输入形状: {shape}  dtype={dtype}")
        click.echo("─" * 70)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


@test_evidence.command("script")
@click.argument("evidence_id")
def evidence_script(evidence_id):
    """打印指定证据包的可复现 Python 脚本。

    \b
    示例:
      deepmt test evidence script abc123def456
      deepmt test evidence script abc123def456 > repro.py
    """
    try:
        from deepmt.analysis.reporting.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        pack = collector.load(evidence_id)

        if pack is None:
            click.echo(click.style(f"未找到证据包 {evidence_id!r}", fg="red"), err=True)
            sys.exit(1)

        click.echo(pack.reproduce_script)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
