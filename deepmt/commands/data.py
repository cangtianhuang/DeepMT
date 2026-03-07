"""
deepmt data — 数据目录管理子命令组

命令:
    clean-logs    清理 data/logs 下的日志文件
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

_DEFAULT_LOG_DIR = Path("data/logs")


def _parse_date(date_str: str) -> datetime:
    """解析 YYYY-MM-DD 格式日期字符串。"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter(f"日期格式错误 '{date_str}'，期望 YYYY-MM-DD")


@click.group()
def data():
    """数据目录管理（日志清理等）。"""


@data.command("clean-logs")
@click.option(
    "--log-dir",
    default=str(_DEFAULT_LOG_DIR),
    show_default=True,
    type=click.Path(file_okay=False),
    help="日志目录路径",
)
@click.option(
    "--before",
    default=None,
    metavar="YYYY-MM-DD",
    help="仅删除此日期（不含）之前的日志文件",
)
@click.option(
    "--keep-days",
    default=None,
    type=int,
    metavar="N",
    help="保留最近 N 天的日志，更早的删除（与 --before 互斥）",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="仅预览，不实际删除",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def clean_logs(log_dir, before, keep_days, dry_run, yes):
    """清理 data/logs 目录下的日志文件。

    不指定任何过滤条件时，清理所有日志文件。
    --before 与 --keep-days 互斥，不可同时使用。

    \b
    示例:
      deepmt data clean-logs                          # 清理全部日志
      deepmt data clean-logs --dry-run                # 预览要删除的文件
      deepmt data clean-logs --keep-days 7            # 仅保留最近 7 天
      deepmt data clean-logs --before 2026-01-01      # 删除 2026-01-01 之前的日志
      deepmt data clean-logs --keep-days 3 -y         # 保留 3 天，跳过确认
    """
    if before and keep_days is not None:
        raise click.UsageError("--before 与 --keep-days 不可同时使用")

    log_path = Path(log_dir)
    if not log_path.exists():
        click.echo(f"日志目录不存在: {log_path}")
        return

    # 计算截止时间
    cutoff: datetime | None = None
    if before:
        cutoff = _parse_date(before)
    elif keep_days is not None:
        cutoff = datetime.now() - timedelta(days=keep_days)

    # 收集候选文件（*.log）
    all_logs = sorted(log_path.glob("*.log"))
    if not all_logs:
        click.echo("日志目录为空，无需清理。")
        return

    if cutoff is not None:
        targets = [f for f in all_logs if datetime.fromtimestamp(f.stat().st_mtime) < cutoff]
    else:
        targets = all_logs

    if not targets:
        if cutoff:
            click.echo(f"没有满足条件的日志文件（截止: {cutoff.date()}）。")
        else:
            click.echo("没有日志文件可删除。")
        return

    # 打印预览
    total_size = sum(f.stat().st_size for f in targets)
    click.echo(f"\n将删除 {len(targets)} 个日志文件（共 {total_size / 1024:.1f} KB）:")
    for f in targets:
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size = f.stat().st_size / 1024
        click.echo(f"  {f.name:<35} {mtime}  {size:>7.1f} KB")

    if dry_run:
        click.echo(click.style("\n[dry-run] 未实际删除任何文件。", fg="yellow"))
        return

    if not yes:
        click.confirm(f"\n确认删除以上 {len(targets)} 个文件？", abort=True)

    deleted, failed = 0, 0
    for f in targets:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            click.echo(click.style(f"  删除失败 {f.name}: {e}", fg="red"), err=True)
            failed += 1

    click.echo(click.style(f"\n已删除 {deleted} 个文件", fg="green") +
               (click.style(f"，失败 {failed} 个", fg="red") if failed else ""))
