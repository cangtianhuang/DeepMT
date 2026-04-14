"""
deepmt.commands.test.history — 测试历史查询子命令

命令: history, failures
"""

import json
import sys

import click

from deepmt._utils import get_results_manager
from deepmt.commands.test._group import test


# ── history ───────────────────────────────────────────────────────────────────

@test.command("history")
@click.argument("name", required=False, default=None)
@click.option("--limit", default=20, show_default=True, type=int, help="最多显示条数")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_history(name, limit, as_json):
    """查看测试历史。

    \b
    示例:
      deepmt test history
      deepmt test history relu
      deepmt test history --limit 50 --json
    """
    try:
        rm = get_results_manager()
        records = rm.get_summary(name)

        if limit:
            records = records[:limit]

        if as_json:
            click.echo(json.dumps(records, ensure_ascii=False, indent=2))
            return

        if not records:
            click.echo("暂无测试历史记录。")
            return

        click.echo(f"测试历史（共 {len(records)} 条）:")
        click.echo("─" * 70)
        for r in records:
            total = r.get("total_tests", 0)
            passed = r.get("passed_tests", 0)
            failed = r.get("failed_tests", 0)
            ir_name = r.get("ir_name", "?")
            status_str = click.style("PASS", fg="green") if failed == 0 else click.style("FAIL", fg="red")
            click.echo(f"  [{status_str}] {ir_name}  passed={passed}/{total}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── failures ──────────────────────────────────────────────────────────────────

@test.command("failures")
@click.option("--limit", default=50, show_default=True, type=int, help="最多显示条数")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_failures(limit, as_json):
    """查看失败的测试用例。

    \b
    示例:
      deepmt test failures
      deepmt test failures --limit 100
      deepmt test failures --json
    """
    try:
        rm = get_results_manager()
        records = rm.get_failed_tests(limit)

        if as_json:
            click.echo(json.dumps(records, ensure_ascii=False, indent=2))
            return

        if not records:
            click.echo(click.style("未发现失败测试用例。", fg="green"))
            return

        click.echo(f"失败测试用例（共 {len(records)} 条）:")
        click.echo("─" * 70)
        for r in records:
            click.echo(f"  {r.get('ir_name', '?')} / MR: {r.get('mr_description', '?')}")
            if r.get("defect_type"):
                click.echo(f"    缺陷类型: {r['defect_type']}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
