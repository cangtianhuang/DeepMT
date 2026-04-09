"""
deepmt repo — MR 知识库管理子命令组

命令:
    list    列出知识库中所有算子
    stats   显示统计信息
    info    显示算子详情（MR 数量等）
"""

import json
import sys

import click

from deepmt._utils import get_repo


@click.group()
def repo():
    """MR 知识库管理（查看算子、统计信息）。"""


# ── list ──────────────────────────────────────────────────────────────────────

@repo.command("list")
@click.option("--framework", default=None, help="按框架过滤（如 pytorch）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_list(framework, as_json):
    """列出知识库中所有算子。

    \b
    示例:
      deepmt repo list
      deepmt repo list --framework pytorch
      deepmt repo list --json
    """
    r = get_repo()
    if framework:
        ops = r.list_operators_by_framework(framework)
    else:
        ops = r.list_operators()

    if as_json:
        result = []
        for op in ops:
            stats = r.get_statistics(op)
            result.append({"operator": op, **stats})
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not ops:
        click.echo("知识库为空，尚未保存任何 MR。")
        click.echo("提示: 运行 'deepmt mr generate <operator> --save' 来生成并保存 MR。")
        return

    header = f"知识库中共有 {len(ops)} 个算子"
    if framework:
        header += f"（框架: {framework}）"
    click.echo(header + ":\n")
    click.echo(f"  {'算子':<40} {'总数':>6} {'已验证':>8} {'未验证':>8}")
    click.echo("  " + "─" * 64)
    for op in ops:
        s = r.get_statistics(op)
        click.echo(
            f"  {op:<40} {s['total_mrs']:>6} {s['verified_mrs']:>8} {s['unverified_mrs']:>8}"
        )


# ── stats ─────────────────────────────────────────────────────────────────────

@repo.command("stats")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_stats(as_json):
    """显示知识库整体统计信息。

    \b
    示例:
      deepmt repo stats
      deepmt repo stats --json
    """
    r = get_repo()
    stats = r.get_statistics()

    if as_json:
        click.echo(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    click.echo("\nMR 知识库统计")
    click.echo("─" * 40)
    click.echo(f"  总 MR 数:   {stats['total_mrs']}")
    click.echo(f"  已验证:     {stats['verified_mrs']}")
    click.echo(f"  未验证:     {stats['unverified_mrs']}")
    click.echo(f"  数值检查:   {stats['checked']}")
    click.echo(f"  符号证明:   {stats['proven']}")

    if stats["by_operator"]:
        click.echo("\n  按算子分布:")
        click.echo(f"  {'算子':<40} {'总数':>6} {'已验证':>8} {'未验证':>8}")
        click.echo("  " + "─" * 64)
        for op, s in stats["by_operator"].items():
            click.echo(f"  {op:<40} {s['total']:>6} {s['verified']:>8} {s['unverified']:>8}")


# ── info ──────────────────────────────────────────────────────────────────────

@repo.command("info")
@click.argument("operator")
@click.option("--framework", default=None, help="按框架过滤 MR（如 pytorch）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_info(operator, framework, as_json):
    """显示算子的详细信息（MR 摘要）。

    \b
    示例:
      deepmt repo info relu
      deepmt repo info relu --framework pytorch
      deepmt repo info relu --json
    """
    r = get_repo()

    if not r.exists(operator):
        click.echo(click.style(f"知识库中未找到算子 '{operator}'", fg="yellow"))
        sys.exit(1)

    stats = r.get_statistics(operator)

    if as_json:
        mrs = r.get_mr_with_validation_status(operator, framework=framework)
        result = {
            "operator": operator,
            "stats": stats,
            "mrs": [
                {
                    "id": m.id,
                    "description": m.description,
                    "category": m.category,
                    "oracle_expr": m.oracle_expr,
                    "transform_code": m.transform_code,
                    "verified": m.verified,
                    "applicable_frameworks": m.applicable_frameworks,
                }
                for m in mrs
            ],
        }
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    click.echo(f"\n算子: {click.style(operator, bold=True)}")
    click.echo(f"  总 MR 数:  {stats['total_mrs']}")
    click.echo(f"  已验证:    {stats['verified_mrs']}")

    mrs = r.get_mr_with_validation_status(operator, framework=framework)
    fw_note = f"  (框架: {framework})" if framework else ""
    click.echo(f"\n  {fw_note}共 {len(mrs)} 个 MR:")
    for m in mrs:
        mark = click.style("✓", fg="green") if m.verified else click.style("✗", fg="red")
        fw_tag = f" [{'/'.join(m.applicable_frameworks)}]" if m.applicable_frameworks else ""
        click.echo(f"    [{mark}] [{m.category}]{fw_tag} {m.description}")
        click.echo(f"         oracle: {m.oracle_expr}")


# ── delete ────────────────────────────────────────────────────────────────────

@repo.command("delete")
@click.argument("operator")
@click.option("--id", "mr_id", default=None, help="只删除该 MR ID")
@click.option("--all", "delete_all", is_flag=True, default=False, help="删除算子的全部 MR")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def repo_delete(operator, mr_id, delete_all, yes):
    """删除知识库中的 MR 记录。

    \b
    示例:
      deepmt repo delete relu --id <MR_ID>   # 删除单条 MR
      deepmt repo delete relu --all           # 删除算子所有 MR
      deepmt repo delete relu --all --yes     # 跳过确认
    """
    r = get_repo()

    if not r.exists(operator):
        click.echo(click.style(f"知识库中未找到算子 '{operator}'", fg="yellow"))
        sys.exit(1)

    if not mr_id and not delete_all:
        click.echo(click.style("请指定 --id <MR_ID> 或 --all", fg="red"))
        sys.exit(1)

    if mr_id:
        desc = f"MR ID={mr_id}（算子 '{operator}'）"
    else:
        desc = f"算子 '{operator}' 的全部 MR"

    if not yes:
        confirmed = click.confirm(f"确认删除 {desc}？", default=False)
        if not confirmed:
            click.echo("已取消。")
            return

    deleted = r.delete(operator, mr_id=mr_id)
    click.echo(click.style(f"已删除 {deleted} 条 MR。", fg="green"))
